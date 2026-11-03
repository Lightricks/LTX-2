"""Multi-GPU distilled video runner.
Runs :class:`DistilledPipeline` across multiple GPUs with:
- **Shared stage** -- sequence parallelism (SP); the same DiffusionStage is
  reused for both stage 1 (half-res) and stage 2 (full-res), so a single
  SP wrapping covers both invocations.
- **Gemma** -- Accelerate-based parallelization
- **VAE** -- distributed decoding
Requires ``ltx-kernels`` to be installed (transitive via SP builder).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from multiprocessing import SimpleQueue
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.loader.distributed_builders import AccelerateGemmaBuilder, SequenceParallelBuilder
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae.transformer import DiffVAEMode
from ltx_core.multigpu.transformer.attention import AttentionManager
from ltx_core.quantization import QuantizationPolicy
from ltx_core.quantization.fp8_cast import build_policy as _build_fp8_cast_policy
from ltx_core.tiling import DimensionTilingConfig, TileCountConfig, balanced_tile_split
from ltx_pipelines.chunks import ChunkConfig
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.multigpu.controller import MGPUController
from ltx_pipelines.multigpu.runner import MGPURunner, rank_cuda_registry
from ltx_pipelines.multigpu.vae_builders import DistributedDecoderBuilder
from ltx_pipelines.utils.args import markers_from_args
from ltx_pipelines.utils.helpers import get_device
from ltx_pipelines.utils.media_io import (
    EXRColorSpace,
    encode_media_segments,
    resolve_hdr_color_space,
    vae_dtype_for_hdr,
)
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.types import DEFAULT_AUTO_DURATION, AutoDuration

logger = logging.getLogger(__name__)

# Stage 2 (full-res) dominates: 1024x1536, 121 frames ~= 24576 video tokens + audio tokens.
_DEFAULT_SP_MAX_TOKENS = 32768
# Rank that collects distributed-VAE tiles and encodes the assembled video.
_DRIVER_RANK = 0


class DistilledRunner(MGPURunner):
    """Distributed :class:`DistilledPipeline`: SP shared stage + Accelerate Gemma + distributed VAE."""

    @torch.inference_mode()
    def setup(
        self,
        *,
        model_paths: ModelPaths,
        prompt_enhancer_gemma_root: str | None = None,
        spatial_upsampler_path: str,
        vae_queue: SimpleQueue,
        compilation_config: CompilationConfig | None = None,
        sp_max_tokens: int = _DEFAULT_SP_MAX_TOKENS,
        quantization: Callable[[], QuantizationPolicy] | None = None,
        diffvae_optimization: DiffVAEMode = DiffVAEMode.CHUNKED_EAGER,
    ) -> None:
        # quantization is a picklable zero-arg builder (built per worker, post-spawn); default fp8-cast.
        quantization_policy = (
            quantization() if quantization is not None else _build_fp8_cast_policy(model_paths.transformer())
        )
        unit_registry = rank_cuda_registry()
        pipeline = DistilledPipeline(
            model_paths=model_paths,
            prompt_enhancer_gemma_root=prompt_enhancer_gemma_root,
            spatial_upsampler_path=spatial_upsampler_path,
            loras=[],
            registry=unit_registry,
            quantization=quantization_policy,
            compilation_config=compilation_config,
            alloc_trim_strategy=AllocatorTrimStrategy.DEFER,
            diffvae_optimization=diffvae_optimization,
        )
        # Every distributed transformer build of this runner rewrites one shared set of active weights.
        active_weights: dict[str, torch.Tensor] = {}

        # Shared stage: sequence parallelism (covers both stage 1 and stage 2 invocations).
        model_cfg = pipeline.stage._transformer_builder.model_config().get("transformer", {})
        attn_mgr = AttentionManager(
            max_tokens=sp_max_tokens,
            num_heads=model_cfg["num_attention_heads"],
            head_dim=model_cfg["attention_head_dim"],
            tensor_dtype=pipeline.dtype,
            group=self.groups.transformer_group,
        )
        pipeline.stage._transformer_builder = SequenceParallelBuilder(
            pipeline.stage._transformer_builder.with_registry(unit_registry), attn_mgr, dst=active_weights
        )

        # Accelerate Gemma parallelization. Capture shared-vs-separate before replacing
        # the encode builder so a shared alias is re-bound to the new instance.
        pe = pipeline.prompt_encoder
        separate_enhancer = pe._enhancer_text_encoder_builder is not pe._text_encoder_builder
        pe._text_encoder_builder = AccelerateGemmaBuilder.from_root(
            model_paths.text_encoder(), unit_registry, self.groups.gemma_group, _DRIVER_RANK
        )
        if separate_enhancer:
            assert prompt_enhancer_gemma_root is not None
            pe._enhancer_text_encoder_builder = AccelerateGemmaBuilder.from_root(
                prompt_enhancer_gemma_root, unit_registry, self.groups.gemma_group, _DRIVER_RANK
            )
        else:
            pe._enhancer_text_encoder_builder = pe._text_encoder_builder

        # Distributed VAE decoding: balanced 2D spatial grid over the group (one tile/rank).
        # height takes the smaller factor of world_size, width the larger; size-aware split is a follow-up.
        vae_height_tiles, vae_width_tiles = balanced_tile_split(dist.get_world_size(self.groups.vae_group))
        vae_tiling = TileCountConfig(
            height=DimensionTilingConfig(num_tiles=vae_height_tiles, overlap=4),
            width=DimensionTilingConfig(num_tiles=vae_width_tiles, overlap=4),
        )
        pipeline.video_decoder._decoder_builder = DistributedDecoderBuilder(  # type: ignore[assignment]
            inner=pipeline.video_decoder._decoder_builder,
            queue=vae_queue,
            vae_group=self.groups.vae_group,
            vae_tiling=vae_tiling,
            driver_rank=_DRIVER_RANK,
            registry=unit_registry,
            # The driver concatenates the decoded clip on one device anyway, so assembling it
            # in more temporal batches only adds a copy of the canvas.
            num_temporal_batches=1,
        )

        self._pipeline = pipeline

    @torch.inference_mode()
    def __call__(  # noqa: PLR0913
        self,
        *,
        output_path: str,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        frame_rate: float,
        num_frames: int | AutoDuration = DEFAULT_AUTO_DURATION,
        images: list[Any] | None = None,
        enhance_prompt: bool = False,
        enhance_static_cache: bool = False,
        hdr: EXRColorSpace | None = None,
        generated_keyframes: int | Sequence[int] = 0,
        decode_with_keyframes: bool = False,
        chunk_config: ChunkConfig | None = None,
        marker_root_path: str | None = None,
    ) -> Iterator[str | None]:
        # The pipeline raises ValueError on invalid input (symmetric across ranks); the controller
        # catches that and turns it into a recoverable RunnerError. Anything else is fatal.
        hdr = resolve_hdr_color_space(images=images or [], hdr=hdr)
        video_marker, audio_marker = markers_from_args(
            SimpleNamespace(marker_root_path=marker_root_path),
            device=get_device(),
            color_space=hdr,
        )
        vae_dtype = vae_dtype_for_hdr(hdr, torch.bfloat16)
        chunks, _, _ = self._pipeline.stream_chunks(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            frame_rate=frame_rate,
            images=images or [],
            num_frames=num_frames,
            vae_dtype=vae_dtype,
            tiling_config=None,
            enhance_prompt=enhance_prompt,
            enhance_static_cache=enhance_static_cache,
            color_space=hdr,
            generated_keyframes=generated_keyframes,
            decode_with_keyframes=decode_with_keyframes,
            chunk_config=chunk_config,
        )
        is_driver = dist.get_rank() == _DRIVER_RANK
        if not is_driver:
            for _ in chunks:
                pass
        else:
            try:
                encode_media_segments(
                    chunks,
                    fps=frame_rate,
                    output_path=output_path,
                    video_chunks_number=None,
                    color_space=hdr,
                    video_marker=video_marker,
                    audio_marker=audio_marker,
                )
            except ValueError as error:
                # Encoding runs only on the driver, so its failures cannot use the
                # recoverable ValueError path that requires every rank to exit together.
                raise RuntimeError("media encoding failed on the driver rank") from error
        yield output_path if is_driver else None


if __name__ == "__main__":
    from ltx_pipelines.utils.args import (
        add_chunk_layout_args,
        add_generated_keyframes_arg,
        add_keyframe_decode_arg,
        chunk_config_from_args,
        default_2_stage_distilled_arg_parser,
        resolve_cli_params,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    params = resolve_cli_params(distilled=True)
    parser = add_chunk_layout_args(
        add_keyframe_decode_arg(
            add_generated_keyframes_arg(
                default_2_stage_distilled_arg_parser(params=params, supports_auto_duration=True)
            )
        )
    )
    args = parser.parse_args()

    vae_queue = torch.multiprocessing.get_context("spawn").SimpleQueue()
    controller = MGPUController(DistilledRunner)
    controller.start(
        model_paths=args.model_paths,
        prompt_enhancer_gemma_root=args.prompt_enhancer_gemma_root,
        spatial_upsampler_path=args.spatial_upsampler_path,
        vae_queue=vae_queue,
        compilation_config=args.compile,
        diffvae_optimization=args.diffvae_optimization,
    )
    try:
        for _ in controller.stream(
            output_path=args.output_path,
            prompt=args.prompt,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            images=args.images,
            enhance_prompt=args.enhance_prompt,
            enhance_static_cache=args.enhance_static_cache,
            hdr=args.hdr,
            generated_keyframes=args.num_generated_keyframes,
            decode_with_keyframes=args.decode_with_keyframes,
            chunk_config=chunk_config_from_args(args),
            marker_root_path=args.marker_root_path,
        ):
            pass  # drive the job to completion; the runner writes the file as a side effect
    finally:
        controller.shutdown()
