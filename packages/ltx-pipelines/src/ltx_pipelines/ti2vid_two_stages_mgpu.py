"""Multi-GPU two-stage text/image-to-video runner.
Runs :class:`TI2VidTwoStagesPipeline` across multiple GPUs with:
- **Stage 1** -- sequence parallelism (SP)
- **Stage 2** -- tiled data parallelism (TDP) on height + width with overlap
- **Gemma** -- Accelerate-based parallelization
- **VAE** -- distributed decoding
Requires ``ltx-kernels`` to be installed (transitive via SP builder).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from multiprocessing import SimpleQueue

import torch
import torch.distributed as dist

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.loader.registry import StateDictRegistry
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae import get_video_chunks_number
from ltx_core.model.video_vae.tiling import TilingConfig
from ltx_core.multigpu.transformer.attention import AttentionManager
from ltx_core.quantization import QuantizationPolicy
from ltx_core.quantization.fp8_cast import build_policy as _build_fp8_cast_policy
from ltx_core.tiling import DimensionTilingConfig, TileCountConfig, balanced_tile_split
from ltx_pipelines.multigpu.controller import MGPUController
from ltx_pipelines.multigpu.gemma_builders import AccelerateGemmaBuilder
from ltx_pipelines.multigpu.runner import MGPURunner
from ltx_pipelines.multigpu.sp_builder import SequenceParallelBuilder
from ltx_pipelines.multigpu.tdp_builder import TiledDataParallelBuilder
from ltx_pipelines.multigpu.vae_builders import DistributedDecoderBuilder
from ltx_pipelines.multigpu.weight_tracker import TransformerWeightTracker
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_pipelines.utils.constants import TDP_DISTILLED_SIGMAS
from ltx_pipelines.utils.media_io import encode_video

logger = logging.getLogger(__name__)

# Stage 1 at 512x768, 121 frames = 6144 video tokens + audio tokens.
_DEFAULT_SP_MAX_TOKENS = 32768
# Rank that collects distributed-VAE tiles and encodes the assembled video.
_DRIVER_RANK = 0


class TI2VidTwoStagesRunner(MGPURunner):
    """Distributed :class:`TI2VidTwoStagesPipeline`: SP stage 1 + TDP stage 2 + Gemma + distributed VAE."""

    @torch.inference_mode()
    def setup(
        self,
        *,
        checkpoint_path: str,
        gemma_root: str,
        spatial_upsampler_path: str,
        vae_queue: SimpleQueue,
        distilled_lora_path: str,
        compilation_config: CompilationConfig | None = None,
        sp_max_tokens: int = _DEFAULT_SP_MAX_TOKENS,
        quantization: Callable[[], QuantizationPolicy] | None = None,
    ) -> None:
        # quantization is a picklable zero-arg builder (built per worker, post-spawn); default fp8-cast.
        quantization_policy = quantization() if quantization is not None else _build_fp8_cast_policy(checkpoint_path)
        distilled_lora = [LoraPathStrengthAndSDOps(distilled_lora_path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP)]
        registry = StateDictRegistry()
        pipeline = TI2VidTwoStagesPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora=distilled_lora,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root=gemma_root,
            loras=[],
            registry=registry,
            quantization=quantization_policy,
            compilation_config=compilation_config,
            alloc_trim_strategy=AllocatorTrimStrategy.DEFER,
        )
        tracker = TransformerWeightTracker(group=self.groups.transformer_group)

        # Stage 1: sequence parallelism.
        model_cfg = pipeline.stage_1._transformer_builder.model_config().get("transformer", {})
        attn_mgr = AttentionManager(
            max_tokens=sp_max_tokens,
            num_heads=model_cfg["num_attention_heads"],
            head_dim=model_cfg["attention_head_dim"],
            tensor_dtype=pipeline.dtype,
            group=self.groups.transformer_group,
        )
        pipeline.stage_1._transformer_builder = SequenceParallelBuilder(
            inner=pipeline.stage_1._transformer_builder,
            attn_mgr=attn_mgr,
            registry=registry,
            tracker=tracker,
        )

        # Stage 2: tiled data parallelism -- balanced 2D spatial grid over the group (one tile/rank).
        # height takes the smaller factor of world_size, width the larger; size-aware split is a follow-up.
        tdp_height_tiles, tdp_width_tiles = balanced_tile_split(dist.get_world_size(self.groups.transformer_group))
        tdp_tiling = TileCountConfig(
            height=DimensionTilingConfig(num_tiles=tdp_height_tiles, overlap=5),
            width=DimensionTilingConfig(num_tiles=tdp_width_tiles, overlap=5),
        )
        pipeline.stage_2._transformer_builder = TiledDataParallelBuilder(
            inner=pipeline.stage_2._transformer_builder,
            group=self.groups.transformer_group,
            tiling=tdp_tiling,
            registry=registry,
            tracker=tracker,
        )

        # Accelerate Gemma parallelization.
        pipeline.prompt_encoder._text_encoder_builder = AccelerateGemmaBuilder(
            gemma_root_path=gemma_root,
            gemma_group=self.groups.gemma_group,
            broadcast_group=self.groups.transformer_group,
            registry=registry,
            src_rank=_DRIVER_RANK,
            dtype=pipeline.dtype,
        )

        # Distributed VAE decoding: balanced 2D spatial grid over the group (one tile/rank).
        # height takes the smaller factor of world_size, width the larger; size-aware split is a follow-up.
        vae_height_tiles, vae_width_tiles = balanced_tile_split(dist.get_world_size(self.groups.vae_group))
        vae_tiling = TileCountConfig(
            height=DimensionTilingConfig(num_tiles=vae_height_tiles, overlap=4),
            width=DimensionTilingConfig(num_tiles=vae_width_tiles, overlap=4),
        )
        pipeline.video_decoder._decoder_builder = DistributedDecoderBuilder(
            inner=pipeline.video_decoder._decoder_builder,
            queue=vae_queue,
            vae_group=self.groups.vae_group,
            vae_tiling=vae_tiling,
            driver_rank=_DRIVER_RANK,
            registry=registry,
        )

        self._pipeline = pipeline

    @torch.inference_mode()
    def __call__(  # noqa: PLR0913
        self,
        *,
        output_path: str,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams,
        audio_guider_params: MultiModalGuiderParams,
        images: list | None = None,
    ) -> Iterator[str | None]:
        # The pipeline raises ValueError on invalid input (symmetric across ranks); the controller
        # catches that and turns it into a recoverable RunnerError. Anything else is fatal.
        video, audio = self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
            images=images or [],
            tiling_config=None,
            stage_2_sigmas=TDP_DISTILLED_SIGMAS,
        )
        if dist.get_rank() != _DRIVER_RANK:
            yield None  # workers: nothing to encode
            return
        encode_video(
            video=video,
            fps=frame_rate,
            audio=audio,
            output_path=output_path,
            video_chunks_number=get_video_chunks_number(num_frames, TilingConfig.default()),
        )
        yield output_path


if __name__ == "__main__":
    from ltx_pipelines.utils.args import (
        default_2_stage_arg_parser,
        resolve_cli_params,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    params = resolve_cli_params()
    args = default_2_stage_arg_parser(params=params).parse_args()

    vae_queue = torch.multiprocessing.get_context("spawn").SimpleQueue()
    controller = MGPUController(TI2VidTwoStagesRunner)
    controller.start(
        checkpoint_path=args.checkpoint_path,
        gemma_root=args.gemma_root,
        spatial_upsampler_path=args.spatial_upsampler_path,
        vae_queue=vae_queue,
        distilled_lora_path=args.distilled_lora[0].path,
        compilation_config=args.compile,
    )
    try:
        for _ in controller.stream(
            output_path=args.output_path,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            num_inference_steps=args.num_inference_steps,
            video_guider_params=MultiModalGuiderParams(
                cfg_scale=args.video_cfg_guidance_scale,
                stg_scale=args.video_stg_guidance_scale,
                rescale_scale=args.video_rescale_scale,
                modality_scale=args.a2v_guidance_scale,
                skip_step=args.video_skip_step,
                stg_blocks=args.video_stg_blocks,
            ),
            audio_guider_params=MultiModalGuiderParams(
                cfg_scale=args.audio_cfg_guidance_scale,
                stg_scale=args.audio_stg_guidance_scale,
                rescale_scale=args.audio_rescale_scale,
                modality_scale=args.v2a_guidance_scale,
                skip_step=args.audio_skip_step,
                stg_blocks=args.audio_stg_blocks,
            ),
            images=args.images,
        ):
            pass  # drive the job to completion; the runner writes the file as a side effect
    finally:
        controller.shutdown()
