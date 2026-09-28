"""DFR (Diffusion Fidelity Rendering): keyframe-slot base, spatial detailing, tiled temporal rounds."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import torch

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae import (
    AUTO_TILING,
    AutoTiling,
    TilingConfig,
    get_video_chunks_number,
)
from ltx_core.model.video_vae.transformer import DiffVAEMode
from ltx_core.quantization import QuantizationPolicy, fp8_cast_fuse_rule
from ltx_core.quantization.fp8_cast import supports_fp8_gemm, with_rowwise_fp8_gemm
from ltx_core.types import VIDEO_SCALE_FACTORS, VideoPixelShape
from ltx_pipelines.dfr_helpers.layout import resolve_canvas
from ltx_pipelines.dfr_helpers.ops import (
    DETAILING_LORA_STRENGTH,
    trim_audio_to_video_duration,
    trim_video_to_requested_frames,
)
from ltx_pipelines.dfr_stages import (
    KeyframeCarry,
    denoise_stage1,
    denoise_stage2,
    run_one_temporal_round,
    run_spatial_epilogue,
)
from ltx_pipelines.distilled import (
    ANCESTRAL_NOISE_SEED_OFFSET,
    ANCESTRAL_STAGE_2_NOISE_SEED_OFFSET,
    ancestral_sampler_kwargs,
    should_use_ancestral_sampler,
)
from ltx_pipelines.iclora_utils import read_lora_reference_downscale_factor
from ltx_pipelines.utils.args import (
    LoraAction,
    default_2_stage_distilled_arg_parser,
    markers_from_args,
    resolve_cli_params,
    resolve_existing_path,
)
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    DurationPredictor,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
    require_num_frames_source,
    resolve_num_frames,
)
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    assert_stage_supports_generated_keyframes,
    decode_keyframes_from_slots,
    ensure_tiling_config,
    get_device,
    tiling_scale_factors_for_vae,
)
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.types import (
    DEFAULT_AUTO_DURATION,
    AutoDuration,
    ImageConditioningInput,
    OffloadMode,
    PipelineOutput,
)

_EPILOGUE_ANCESTRAL_NOISE_SEED_OFFSET = 30000


class DFRPipeline:
    """
    DFR pipeline on a keyframe-slot-capable distilled checkpoint.
    Stage 1 (half-res) generates video and keyframe slots on an x8-border segment grid; the half-res
    video is reserved as the IC-LoRA reference; stage 2 spatially upsamples video and slots before denoising.
    Stage 2 re-denoises at full resolution with an x2 detailing IC-LoRA. Both stages and the spatial
    epilogue sample according to ``self.use_ancestral_sampler`` (see
    :func:`~ltx_pipelines.distilled.should_use_ancestral_sampler`). Shipped audio comes from stage 1:
    stage 2 still runs an audio pass because video needs
    the cross-modal attention, but it re-noises audio under the detailing LoRA. Temporal tiles pass
    frozen stage-1 audio sliced to the tile's playback window (retiled onto the snapped-fps token
    count) for cross-attention only; they do not refine or ship audio.
    Optional ``temporal_upscalings`` (0-2): each round temporally x2-upsamples, splits the canvas
    into ``2**round`` keyframe-seam tiles, invents mid-segment slots per tile, densifies with ancestral
    Euler, and stitches. A non-first tile starts on the last keyframe plane before its seam with a
    pinned prefix. Optional ``spatial_upscalings`` (1 or 2): ``1`` is stage 1 at ``h/2`` and
    stage 2 at ``h``; ``2`` is stage 1 at ``h/4``, stage 2 at ``h/2``, then a full-res spatial
    detailing epilogue at ``h``. Carry keyframes are decoded one plane at a time, Lanczos-stretched
    x2 in RGB, and encoded again; only the video latent is spatially upsampled. The epilogue runs
    one 3-step detailing pass per temporal window with those keyframes as strength-1 conditions and
    the previous-stage video as the IC-LoRA reference. Spatial tiles inside a time window blend and
    retile after every Euler step; time windows are sequential and not blended into each other. The
    caller always gets ``(num_frames - 1) * 2**rounds + 1`` frames even when the canvas padded its tail.
    Diffusion work lives in ``dfr_stages`` so other adapters can import the same functions without
    constructing this class or a prompt encoder.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_paths: ModelPaths,
        spatial_upsampler_path: str,
        loras: list[LoraPathStrengthAndSDOps],
        detailing_lora: list[LoraPathStrengthAndSDOps],
        temporal_upsampler_path: str | None = None,
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        compilation_config: CompilationConfig | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
        prompt_enhancer_gemma_root: str | None = None,
        diffvae_optimization: DiffVAEMode = DiffVAEMode.CHUNKED_EAGER,
    ):
        self.device = device or get_device()
        self.dtype = torch.bfloat16
        self._user_loras = tuple(loras)
        if not detailing_lora:
            raise ValueError("detailing_lora is required")
        self._detailing_lora = tuple(
            LoraPathStrengthAndSDOps(item.path, DETAILING_LORA_STRENGTH, item.sd_ops) for item in detailing_lora
        )
        self._detailing_downscale = read_lora_reference_downscale_factor(self._detailing_lora[0].path)

        self.prompt_encoder = PromptEncoder(
            model_paths,
            self.dtype,
            self.device,
            registry=registry,
            offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim_strategy,
            prompt_enhancer_gemma_root=prompt_enhancer_gemma_root,
        )
        self.image_conditioner = ImageConditioner(
            model_paths.video_vae(),
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        stage_loras = self._user_loras
        self.stage = DiffusionStage.from_checkpoint(
            model_paths.transformer(),
            self.dtype,
            self.device,
            loras=stage_loras,
            quantization=quantization,
            registry=registry,
            compilation_config=compilation_config,
            offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.stage_detailing = self.stage.with_loras((*stage_loras, *self._detailing_lora))
        self.upsampler = VideoUpsampler(
            model_paths.video_vae(),
            spatial_upsampler_path,
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.temporal_upsampler = (
            VideoUpsampler(
                model_paths.video_vae(),
                temporal_upsampler_path,
                self.dtype,
                self.device,
                registry=registry,
                alloc_trim_strategy=alloc_trim_strategy,
            )
            if temporal_upsampler_path
            else None
        )
        self.video_decoder = VideoDecoder(
            model_paths.video_vae(),
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
            diffvae_optimization=diffvae_optimization,
        )
        self.audio_decoder = AudioDecoder(
            model_paths.audio_vae(),
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.duration_predictor = DurationPredictor.from_checkpoint(
            model_paths.duration_head_path,
            self.dtype,
            self.device,
        )
        self.use_ancestral_sampler = should_use_ancestral_sampler(model_paths.transformer())

    def _sampler_kwargs(self, seed: int, noise_seed_offset: int) -> dict[str, Any]:
        """Ancestral overrides, or an empty dict to leave ``DiffusionStage`` on its Euler defaults."""
        if not self.use_ancestral_sampler:
            return {}
        return ancestral_sampler_kwargs(seed, self.dtype, noise_seed_offset)

    def _emit_stage(
        self,
        name: str,
        *,
        latent: torch.Tensor,
        keyframes: KeyframeCarry,
        height: int,
        width: int,
        num_frames: int,
        fps: float,
    ) -> None:
        """Optional dump hook. No-op unless the caller set ``_stage_latent_sink``."""
        sink = getattr(self, "_stage_latent_sink", None)
        if sink is None:
            return
        positions = list(keyframes)
        stacked_keyframes = torch.cat(list(keyframes.values()), dim=2) if keyframes else None
        sink(
            name,
            latent=latent,
            keyframes=stacked_keyframes,
            positions=positions,
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps,
        )

    def _epilogue_stage(self) -> DiffusionStage:
        """The detailing stage the spatial epilogue denoises with.
        On an fp8-cast policy and a device with fp8 GEMMs, its long video projections multiply
        e4m3 activations (:func:`~ltx_core.quantization.fp8_cast.with_rowwise_fp8_gemm`): the
        epilogue is the last pass, where the precision costs least. Derived per call so it
        follows whatever builder the stage carries (e.g. a multi-GPU runner's).
        """
        stage = self.stage_detailing
        policy = stage.quantization
        if policy is None or policy.fuse_rule is not fp8_cast_fuse_rule or not supports_fp8_gemm(self.device):
            return stage
        return stage.with_quantization(with_rowwise_fp8_gemm(policy))

    @torch.inference_mode()
    def __call__(  # noqa: PLR0913, PLR0915
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        num_frames: int | AutoDuration = DEFAULT_AUTO_DURATION,
        temporal_upscalings: int = 0,
        spatial_upscalings: int = 1,
        tiling_config: TilingConfig | AutoTiling | None = AUTO_TILING,
        decode_device_fn: Callable[[int], str | torch.device] | None = None,
        enhance_prompt: bool = False,
        enhance_static_cache: bool = False,
        stage_1_sigmas: torch.Tensor = DISTILLED_SIGMAS,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
    ) -> PipelineOutput:
        if temporal_upscalings not in (0, 1, 2):
            raise ValueError(f"temporal_upscalings must be 0, 1, or 2, got {temporal_upscalings}")
        if temporal_upscalings > 0 and self.temporal_upsampler is None:
            raise ValueError("temporal_upscalings > 0 requires temporal_upsampler_path")
        if spatial_upscalings not in (1, 2):
            raise ValueError(f"spatial_upscalings must be 1 or 2, got {spatial_upscalings}")

        require_num_frames_source(num_frames, self.duration_predictor)
        enhance_prompt_image = images[0].path if images else None
        images = self.image_conditioner.load_conditionings(images)
        assert_resolution(
            height=height,
            width=width,
            is_two_stage=True,
            divisor=64 if spatial_upscalings == 1 else 128,
        )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16
        temporal_scale = VIDEO_SCALE_FACTORS.time

        (ctx_p,) = self.prompt_encoder(
            [prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_static_cache=enhance_static_cache,
            enhance_prompt_image=enhance_prompt_image,
        )
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding

        num_frames = resolve_num_frames(
            num_frames,
            self.duration_predictor,
            video_encoding=video_context,
            audio_encoding=audio_context,
            frame_rate=frame_rate,
        )
        requested_frames = num_frames
        num_frames, _, positions = resolve_canvas(num_frames)
        stage_1_frames = num_frames
        source_duration = stage_1_frames / frame_rate
        assert_stage_supports_generated_keyframes(self.stage)

        spatial_div = 2**spatial_upscalings
        stage_1_w, stage_1_h = width // spatial_div, height // spatial_div
        stage_2_w, stage_2_h = width // (spatial_div // 2), height // (spatial_div // 2)

        # --- Stage 1: half-res (or quarter-res) base + keyframe slots -------------------
        stage_1 = denoise_stage1(
            stage=self.stage,
            image_conditioner=self.image_conditioner,
            images=images,
            video_shape=VideoPixelShape(batch=1, frames=num_frames, height=stage_1_h, width=stage_1_w, fps=frame_rate),
            positions=positions,
            video_context=video_context,
            audio_context=audio_context,
            sigmas=stage_1_sigmas,
            noiser=noiser,
            device=self.device,
            dtype=dtype,
            **self._sampler_kwargs(seed, ANCESTRAL_NOISE_SEED_OFFSET),
        )
        audio_latent = stage_1.audio_state.latent.detach().clone()

        # --- Stage 2: spatial detailing ------------------------------------------------
        video_shape = VideoPixelShape(batch=1, frames=num_frames, height=stage_2_h, width=stage_2_w, fps=frame_rate)
        stage_2 = denoise_stage2(
            stage=self.stage_detailing,
            image_conditioner=self.image_conditioner,
            upsampler=self.upsampler,
            images=images,
            video_shape=video_shape,
            keyframes=stage_1.keyframes,
            half_res_reference=stage_1.video_state.latent[:1].detach().clone(),
            downscale_factor=self._detailing_downscale,
            audio_latent=audio_latent,
            video_context=video_context,
            audio_context=audio_context,
            sigmas=stage_2_sigmas,
            noiser=noiser,
            device=self.device,
            dtype=dtype,
            **self._sampler_kwargs(seed, ANCESTRAL_STAGE_2_NOISE_SEED_OFFSET),
        )
        video_state = stage_2.video_state
        keyframes = stage_2.keyframes
        self._emit_stage(
            "s2",
            latent=video_state.latent,
            keyframes=keyframes,
            height=video_shape.height,
            width=video_shape.width,
            num_frames=video_shape.frames,
            fps=video_shape.fps,
        )

        last_window_seams: list[int] = []
        # One transformer for every temporal tile of every round instead of a build per tile.
        with self.stage.resident():
            for round_idx in range(1, temporal_upscalings + 1):
                assert self.temporal_upsampler is not None
                if not keyframes:
                    raise RuntimeError(f"Temporal round {round_idx}: missing carry-forward keyframes")
                video_shape = video_shape._replace(frames=2 * (video_shape.frames - 1) + 1, fps=2 * video_shape.fps)
                video_state, keyframes, last_window_seams = run_one_temporal_round(
                    round_idx=round_idx,
                    stage=self.stage,
                    temporal_upsampler=self.temporal_upsampler,
                    image_conditioner=self.image_conditioner,
                    video_state=video_state,
                    keyframes=keyframes,
                    video_shape=video_shape,
                    images=images,
                    video_context=video_context,
                    audio_context=audio_context,
                    audio_latent=audio_latent,
                    source_duration=source_duration,
                    seed=seed,
                    noiser=noiser,
                    sigmas=DISTILLED_SIGMAS[4:].to(dtype=torch.float32),
                    device=self.device,
                    dtype=dtype,
                    temporal_scale=temporal_scale,
                )
                self._emit_stage(
                    f"t{round_idx}",
                    latent=video_state.latent,
                    keyframes=keyframes,
                    height=video_shape.height,
                    width=video_shape.width,
                    num_frames=video_shape.frames,
                    fps=video_shape.fps,
                )
        num_frames = video_shape.frames
        current_fps = video_shape.fps

        # --- Spatial epilogue (spatial_upscalings == 2): s2 is h/2; this pass is final H x W ---
        if spatial_upscalings == 2:
            epi = run_spatial_epilogue(
                stage=self._epilogue_stage(),
                upsampler=self.upsampler,
                image_conditioner=self.image_conditioner,
                video_decoder=self.video_decoder,
                video_state=video_state,
                keyframes=keyframes,
                images=images,
                video_shape=VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=current_fps),
                temporal_upscalings=temporal_upscalings,
                last_window_seams=last_window_seams,
                video_context=video_context,
                audio_context=audio_context,
                audio_latent=audio_latent,
                source_duration=source_duration,
                sigmas=stage_2_sigmas,
                downscale_factor=self._detailing_downscale,
                seed=seed,
                device=self.device,
                dtype=dtype,
                temporal_scale=temporal_scale,
                **self._sampler_kwargs(seed, _EPILOGUE_ANCESTRAL_NOISE_SEED_OFFSET),
            )
            video_state = epi.video_state
            keyframes = epi.keyframes

        trimmed_latent, num_frames = trim_video_to_requested_frames(
            video_state.latent,
            canvas_frames=num_frames,
            requested_frames=requested_frames,
            temporal_upscalings=temporal_upscalings,
            temporal_scale=temporal_scale,
        )
        video_state = replace(video_state, latent=trimmed_latent)
        keyframe_positions = list(keyframes)
        stacked_keyframes = torch.cat(list(keyframes.values()), dim=2)
        final_keyframes = decode_keyframes_from_slots(stacked_keyframes, keyframe_positions, num_frames)

        playback_fps = frame_rate * 2**temporal_upscalings
        tiling_config = ensure_tiling_config(
            tiling_config,
            scale_factors=tiling_scale_factors_for_vae(self.video_decoder.checkpoint_path),
            vae_checkpoint_path=self.video_decoder.checkpoint_path,
            video_shape=VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=playback_fps),
            diffvae_optimization=self.video_decoder.diffvae_optimization,
            device=self.device,
            keyframes=final_keyframes is not None,
        )
        decoded_video = self.video_decoder(
            video_state.latent,
            tiling_config,
            generator,
            keyframes=final_keyframes,
            device_fn=decode_device_fn,
        )
        decoded_audio = trim_audio_to_video_duration(
            self.audio_decoder(audio_latent),
            num_frames=num_frames,
            playback_fps=playback_fps,
        )
        return PipelineOutput(decoded_video, decoded_audio, num_frames, tiling_config)


def add_dfr_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """DFR-specific flags shared by the single-GPU and MGPU CLIs."""
    parser.add_argument(
        "--detailing-lora",
        dest="detailing_lora",
        action=LoraAction,
        nargs="+",
        metavar=("PATH", "STRENGTH"),
        required=True,
        help="Stage-2 x2 spatial detailing IC-LoRA (path; strength is hardcoded to 0.5).",
    )
    parser.add_argument(
        "--temporal-upsampler-path",
        type=resolve_existing_path,
        default=None,
        help="Path to the temporal x2 latent upsampler (required when --temporal-upscalings > 0).",
    )
    parser.add_argument(
        "--temporal-upscalings",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="Number of temporal x2 refine rounds (0->base fps, 1->2x with 2 tiles, 2->4x with 4 tiles).",
    )
    parser.add_argument(
        "--spatial-upscalings",
        type=int,
        choices=(1, 2),
        default=1,
        help="Spatial upsample count: 1 is stage 1 at h/2 and stage 2 at h; 2 adds a full-res spatial "
        "detailing epilogue after the temporal rounds (stage 1 at h/4, stage 2 at h/2, epilogue at h).",
    )
    return parser


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    params = resolve_cli_params(distilled=True)
    parser = add_dfr_cli_args(default_2_stage_distilled_arg_parser(params=params, supports_auto_duration=True))
    args = parser.parse_args()
    video_marker, audio_marker = markers_from_args(args, device=get_device())

    pipeline = DFRPipeline(
        model_paths=args.model_paths,
        spatial_upsampler_path=args.spatial_upsampler_path,
        loras=tuple(args.lora) if args.lora else (),
        detailing_lora=args.detailing_lora,
        temporal_upsampler_path=args.temporal_upsampler_path,
        quantization=args.quantization,
        compilation_config=args.compile,
        offload_mode=args.offload_mode,
        prompt_enhancer_gemma_root=args.prompt_enhancer_gemma_root,
        diffvae_optimization=args.diffvae_optimization,
    )
    result = pipeline(
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        images=args.images,
        tiling_config=AUTO_TILING,
        enhance_prompt=args.enhance_prompt,
        enhance_static_cache=args.enhance_static_cache,
        temporal_upscalings=args.temporal_upscalings,
        spatial_upscalings=args.spatial_upscalings,
    )

    encode_video(
        video=result.video,
        fps=int(args.frame_rate * (2**args.temporal_upscalings)),
        audio=result.audio,
        output_path=args.output_path,
        video_chunks_number=get_video_chunks_number(result.num_frames, result.tiling_config),
        video_marker=video_marker,
        audio_marker=audio_marker,
    )


if __name__ == "__main__":
    main()
