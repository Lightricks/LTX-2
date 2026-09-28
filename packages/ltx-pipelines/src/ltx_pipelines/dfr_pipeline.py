"""DFR (Diffusion Fidelity Rendering): keyframe-slot base, spatial detailing, tiled temporal rounds."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.components.diffusion_steps import EulerAncestralDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.conditioning import (
    ConditioningItem,
    VideoConditionByKeyframeIndex,
    VideoConditionByReferenceLatent,
    VideoGeneratedKeyframeSlots,
)
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.modality_tiling import VideoModalityTilingHelper
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.video_vae import (
    AUTO_TILING,
    AutoTiling,
    TilingConfig,
    VideoEncoder,
    get_video_chunks_number,
)
from ltx_core.model.video_vae.transformer import DiffVAEMode
from ltx_core.quantization import QuantizationPolicy
from ltx_core.tiling import DimensionTilingConfig, Tile, TileCountConfig
from ltx_core.tools import LatentTools, VideoLatentTools
from ltx_core.types import VIDEO_SCALE_FACTORS, AudioLatentShape, LatentState, VideoPixelShape
from ltx_pipelines.dfr_layout import (
    TemporalTilePlan,
    pixel_to_latent_index,
    resolve_canvas,
)
from ltx_pipelines.iclora_utils import read_lora_reference_downscale_factor
from ltx_pipelines.utils.args import (
    ImageConditioningInput,
    LoraAction,
    default_2_stage_distilled_arg_parser,
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
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    combined_image_conditionings,
    decode_keyframes_from_slots,
    ensure_tiling_config,
    get_device,
    tiling_scale_factors_for_vae,
)
from ltx_pipelines.utils.media_io import encode_video, to_vae_range
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.samplers import euler_ancestral_denoising_loop
from ltx_pipelines.utils.types import DEFAULT_AUTO_DURATION, AutoDuration, ModalitySpec, OffloadMode, PipelineOutput

if TYPE_CHECKING:
    from ltx_core.guidance.perturbations import BatchedPerturbationConfig

logger = logging.getLogger(__name__)

# Anchor keyframes carried between temporal rounds are ours, pinned just short of fully clean so a
# tile can still settle its seam frame.
_ANCHOR_KEYFRAME_STRENGTH = 0.95
_TEMPORAL_ANCESTRAL_ETA = 0.5
# Transformer fps is independent of playback fps. RoPE time is ``pixel_frame / fps``, so rates the
# model never saw (48, 50, 120, ...) cannot lay out the 8 pixel frames inside one latent token --
# they decode as a motion spike at each latent border followed by a stall. Anything above 30 snaps
# to 60; playback fps is used for decoding only.
_MAX_CONDITIONING_FPS = 60.0
_SNAP_CONDITIONING_FPS_ABOVE = 30.0


def _conditioning_fps(playback_fps: float) -> float:
    """RoPE/token fps for the transformer. Values above 30 snap to 60; playback fps is unchanged."""
    return _MAX_CONDITIONING_FPS if playback_fps > _SNAP_CONDITIONING_FPS_ABOVE else playback_fps


# HDR stage-2 spatial overlap. Temporal epilogue tiles cut on the last t-round's window
# seams (same lead-in dropout; the earlier tile keeps the seam cell). Spatial tiles inside
# a time window blend and retile after every Euler step; time windows are not blended.
_EPILOGUE_SPATIAL_OVERLAP = 12
_EPILOGUE_KEYFRAME_STRENGTH = 1.0
_DETAILING_LORA_STRENGTH = 0.5


def _lanczos_x2_fhwc(frames: torch.Tensor) -> torch.Tensor:
    """Stretch each RGB frame 2x with Lanczos. ``frames`` is ``(F, H, W, C)`` in ``[0, 1]``."""
    if frames.ndim != 4:
        raise ValueError(f"Expected (F, H, W, C) RGB, got shape {tuple(frames.shape)}")
    if frames.shape[0] < 1:
        raise ValueError("Need at least one frame to Lanczos-upsample")
    out: list[torch.Tensor] = []
    for frame in frames:
        height, width, channels = frame.shape
        array = (frame.detach().float().clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
        if channels == 1:
            image = Image.fromarray(array[..., 0], mode="L")
        elif channels == 3:
            image = Image.fromarray(array, mode="RGB")
        else:
            raise ValueError(f"Lanczos x2 expects 1 or 3 channels, got {channels}")
        image = image.resize((width * 2, height * 2), resample=Image.Resampling.LANCZOS)
        resized = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
        if resized.ndim == 2:
            resized = resized.unsqueeze(-1)
        out.append(resized)
    return torch.stack(out, dim=0)


def _clamp_dim_tiling(cfg: DimensionTilingConfig, dim_size: int, axis: str) -> DimensionTilingConfig:
    """Clamp a single dim's tile count and overlap to the latent's extent.
    ``split_by_count`` requires ``overlap < tile_size``; with
    ``tile_size = (dim_size + overlap*(n-1)) // n`` this reduces to
    ``overlap <= dim_size - n``. When the configured overlap exceeds this
    bound it is clamped; if the latent is too small to hold ``n`` tiles
    at all, tiling falls back to a single tile on this axis.
    """
    n = cfg.num_tiles
    if n <= 1:
        return cfg
    if dim_size < n:
        logger.warning(
            "%s tiling: dim_size=%d < num_tiles=%d; falling back to 1 tile on this axis.",
            axis,
            dim_size,
            n,
        )
        return DimensionTilingConfig(1, 0)
    max_overlap = dim_size - n
    if cfg.overlap <= max_overlap:
        return cfg
    logger.warning(
        "%s tiling: overlap=%d exceeds latent bound (%d); clamping to %d.",
        axis,
        cfg.overlap,
        max_overlap,
        max_overlap,
    )
    return DimensionTilingConfig(n, max_overlap)


def _clamp_tile_to_latent(tiling: TileCountConfig, latent_shape: tuple[int, int, int]) -> TileCountConfig:
    """Clamp frame, height, and width tilings to the latent's extents.
    ``latent_shape`` is ``(F, H, W)`` in latent units.
    """
    frames, height, width = latent_shape
    return replace(
        tiling,
        frames=_clamp_dim_tiling(tiling.frames, frames, "Frame"),
        height=_clamp_dim_tiling(tiling.height, height, "Height"),
        width=_clamp_dim_tiling(tiling.width, width, "Width"),
    )


class _TiledModelWrapper(torch.nn.Module):
    """Runs the transformer once per latent tile and blends the tiles into one prediction.
    The single-GPU counterpart of
    :class:`~ltx_core.multigpu.transformer.tiled_data_parallel.TiledDataParallelModelWrapper`:
    same :class:`VideoModalityTilingHelper`, same trapezoidal blend masks, same
    conditioning-token filtering. Two differences: every tile runs here rather than round-robin
    across ranks, so the ranks' ``all_reduce`` is just the accumulator; and this wraps the
    ``X0Model`` instead of the bare velocity model, so it blends x0 predictions rather than
    velocities. ``to_denoised`` is affine in the velocity and every tile sees the same
    per-token timesteps, so with blend weights summing to 1 the two are the same value.
    Tiling inside the model call is what makes the overlap agree at *every* denoising step: the
    sampler above still sees one full-canvas latent and takes one Euler step on it.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        video_tools: VideoLatentTools,
        tiling: TileCountConfig,
        seams: Sequence[int] = (),
        normalize_positions: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self._normalize_positions = normalize_positions
        self._helper = VideoModalityTilingHelper(tiling, video_tools, seams=seams)

    @property
    def num_blocks(self) -> int:
        return self.model.num_blocks

    @property
    def tiles(self) -> list[Tile]:
        """The tiles every forward pass walks, in blend order."""
        return self._helper.tiles

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if video is None:
            return self.model(video, audio, perturbations)

        denoised_video: torch.Tensor | None = None
        denoised_audio: torch.Tensor | None = None
        for tile in self._helper.tiles:
            tiled_video, ctx = self._helper.tile_modality(video, tile, normalize_positions=self._normalize_positions)
            tile_out, audio_out = self.model(tiled_video, audio, perturbations)
            denoised_video = self._helper.blend(tile_out, tile, ctx, denoised_video)
            if audio_out is not None:
                denoised_audio = audio_out if denoised_audio is None else denoised_audio + audio_out

        assert denoised_video is not None
        if denoised_audio is not None:
            # Every tile saw the whole audio but a different video context; average as TDP does.
            denoised_audio = denoised_audio / len(self._helper.tiles)
        return denoised_video, denoised_audio


def _resample_audio_time(
    audio_latent: torch.Tensor,
    src_start: float,
    src_end: float,
    out_frames: int,
) -> torch.Tensor:
    """Linearly sample ``audio_latent`` along T over ``[src_start, src_end)`` latent cells."""
    if out_frames < 1:
        raise ValueError(f"out_frames must be >= 1, got {out_frames}")
    full_t = audio_latent.shape[2]
    if full_t < 1:
        raise ValueError("Cannot resample an empty audio latent")
    span = src_end - src_start
    if span <= 0:
        raise ValueError(f"Audio window is empty: [{src_start}, {src_end})")
    step = span / out_frames
    positions = src_start + step * torch.arange(out_frames, device=audio_latent.device, dtype=torch.float32)
    positions = positions.clamp(0, full_t - 1)
    lo = positions.floor().long()
    hi = (lo + 1).clamp(max=full_t - 1)
    weight = (positions - lo.to(torch.float32)).to(dtype=audio_latent.dtype).view(1, 1, -1, 1)
    return audio_latent[:, :, lo] * (1 - weight) + audio_latent[:, :, hi] * weight


def _audio_latent_for_tile(
    audio_latent: torch.Tensor,
    *,
    pixel_start: int,
    local_frames: int,
    playback_fps: float,
    source_duration: float,
    cond_fps: float,
) -> torch.Tensor:
    """Stage-1 audio for one temporal tile: playback window, resampled to the stage's token count.
    Source bounds are wall-clock: ``pixel_start / playback_fps`` through
    ``(pixel_start + local_frames) / playback_fps``, as a fraction of ``source_duration`` (stage 1's
    ``N / fps``). Temporal upsample uses ``N -> 2(N-1)+1`` while fps doubles, so the new canvas is
    slightly shorter than ``2x``; a fraction of *canvas frames* would therefore pull audio from past
    the tile's playback. ``cond_fps`` sizes only the *output* token count DiffusionStage allocates.
    """
    if local_frames <= 0:
        raise ValueError(f"local_frames must be >= 1, got {local_frames}")
    if playback_fps <= 0:
        raise ValueError(f"playback_fps must be > 0, got {playback_fps}")
    if source_duration <= 0:
        raise ValueError(f"source_duration must be > 0, got {source_duration}")
    full_t = audio_latent.shape[2]
    if full_t <= 0:
        raise ValueError("Cannot slice audio for a tile with an empty audio latent")
    src_start = pixel_start / playback_fps / source_duration * full_t
    src_end = (pixel_start + local_frames) / playback_fps / source_duration * full_t
    target_frames = AudioLatentShape.from_video_pixel_shape(
        VideoPixelShape(batch=1, frames=local_frames, height=1, width=1, fps=cond_fps)
    ).frames
    return _resample_audio_time(audio_latent, src_start, src_end, target_frames)


def _keyframe_conditionings_from_latents(
    keyframes: torch.Tensor,
    positions: Sequence[int],
    strength: float,
) -> list[ConditioningItem]:
    """Build ``VideoConditionByKeyframeIndex`` guides from already-encoded keyframe latents."""
    if keyframes.ndim != 5:
        raise ValueError(f"Expected keyframes (B, C, K, H, W), got {tuple(keyframes.shape)}")
    if keyframes.shape[2] != len(positions):
        raise ValueError(f"Expected {len(positions)} keyframe latents, got K={keyframes.shape[2]}")
    return [
        VideoConditionByKeyframeIndex(
            keyframes=keyframes[:, :, index : index + 1],
            frame_idx=int(frame_idx),
            strength=strength,
        )
        for index, frame_idx in enumerate(positions)
    ]


def _slot_initials_from_video(
    video_latent: torch.Tensor,
    positions: Sequence[int],
    temporal_scale: int,
) -> torch.Tensor:
    """Stack the nearest video latent frames as ``(B, C, K, H, W)`` slot seeds."""
    frames = []
    for position in positions:
        index = min(max(round(int(position) / temporal_scale), 0), video_latent.shape[2] - 1)
        frames.append(video_latent[:, :, index : index + 1])
    return torch.cat(frames, dim=2)


def _merge_carry_forward_keyframes(
    anchor_positions: Sequence[int],
    anchor_latents: torch.Tensor | None,
    slot_positions: Sequence[int],
    slot_latents: torch.Tensor | None,
) -> tuple[list[int], torch.Tensor]:
    """Build the next round's anchor bag: carried keyframe stills plus this round's denoised slots.
    Positions must already be on the current round's pixel grid; callers remap (x2) for the next round.
    """
    by_position: dict[int, torch.Tensor] = {}
    for positions, latents, label in (
        (anchor_positions, anchor_latents, "anchor"),
        (slot_positions, slot_latents, "slot"),
    ):
        if not positions:
            continue
        if latents is None:
            raise RuntimeError(f"Missing {label} keyframe latents for carry-forward merge")
        if latents.shape[2] != len(positions):
            raise ValueError(f"{label} latents K={latents.shape[2]} != {len(positions)} positions")
        for index, position in enumerate(positions):
            by_position[int(position)] = latents[:, :, index : index + 1]
    if not by_position:
        raise RuntimeError("Carry-forward keyframe bag is empty")
    ordered = sorted(by_position)
    return ordered, torch.cat([by_position[position] for position in ordered], dim=2)


def _rebase_image_conditionings(
    images: Sequence[ImageConditioningInput],
    *,
    pixel_scale: int,
    pixel_start: int = 0,
    pixel_end: int | None = None,
) -> list[ImageConditioningInput]:
    """Map user ``frame_idx`` values from the requested canvas onto a temporally-upsampled grid.
    After ``r`` temporal rounds the same moment sits at ``frame_idx * 2**r``. When ``pixel_end`` is
    set, only images inside ``[pixel_start, pixel_end]`` are kept and indices become tile-local.
    """
    rebased: list[ImageConditioningInput] = []
    for image in images:
        scaled = image.frame_idx * pixel_scale
        if pixel_end is not None and not (pixel_start <= scaled <= pixel_end):
            continue
        rebased.append(ImageConditioningInput(image.path, scaled - pixel_start, image.strength, image.crf))
    return rebased


class DFRPipeline:
    """
    DFR pipeline on a keyframe-slot-capable distilled checkpoint.
    Stage 1 (half-res) generates video and keyframe slots on an x8-border segment grid; the half-res
    video is reserved as the IC-LoRA reference while video and slots are spatially latent-upsampled.
    Stage 2 re-denoises at full resolution with an x2 detailing
    IC-LoRA. Shipped audio comes from stage 1: stage 2 still runs an audio pass because video needs
    the cross-modal attention, but it re-noises audio under the detailing LoRA. Temporal tiles pass
    frozen stage-1 audio sliced to the tile's playback window (retiled onto the snapped-fps token
    count) for cross-attention only; they do not refine or ship audio.
    Optional ``temporal_upscalings`` (0-2): each round temporally x2-upsamples, splits the canvas
    into ``2**round`` keyframe-seam tiles, invents mid-segment slots per tile, densifies with ancestral
    Euler, and stitches. Optional ``spatial_upscalings`` (1 or 2): ``1`` is stage 1 at ``h/2`` and
    stage 2 at ``h``; ``2`` is stage 1 at ``h/4``, stage 2 at ``h/2``, then a full-res spatial
    detailing epilogue at ``h``. Carry keyframes are decoded one plane at a time, Lanczos-stretched
    x2 in RGB, and encoded again; only the video latent is spatially upsampled. The epilogue runs
    the 3-step detailing schedule with those keyframes as strength-1 conditions and the previous-stage
    video as the IC-LoRA reference. Temporal tiles are seam-cut (lead-in dropped, earlier tile
    keeps the seam). Spatial tiles inside a time window blend and retile after every Euler
    step; time windows are not blended into each other. The caller always
    gets ``(num_frames - 1) * 2**rounds + 1`` frames even when the canvas padded its tail.
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
            LoraPathStrengthAndSDOps(item.path, _DETAILING_LORA_STRENGTH, item.sd_ops) for item in detailing_lora
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

    def _emit_stage(
        self,
        name: str,
        *,
        latent: torch.Tensor,
        keyframes: torch.Tensor | None,
        positions: Sequence[int],
        height: int,
        width: int,
        num_frames: int,
        fps: float,
    ) -> None:
        """Optional dump hook. No-op unless the caller set ``_stage_latent_sink``."""
        sink = getattr(self, "_stage_latent_sink", None)
        if sink is None:
            return
        sink(
            name,
            latent=latent,
            keyframes=keyframes,
            positions=positions,
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps,
        )

    @torch.inference_mode()
    def __call__(  # noqa: PLR0912, PLR0913, PLR0915
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
        images = self.image_conditioner.resolve_crf(images)
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
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
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
        stage_1_duration = stage_1_frames / frame_rate
        self.stage.assert_generated_keyframes_supported()

        spatial_div = 2**spatial_upscalings
        stage_1_w, stage_1_h = width // spatial_div, height // spatial_div
        stage_2_w, stage_2_h = width // (spatial_div // 2), height // (spatial_div // 2)

        # --- Stage 1: half-res (or quarter-res) base + keyframe slots -------------------
        stage_1_sigmas = stage_1_sigmas.to(dtype=torch.float32, device=self.device)
        stage_1_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=stage_1_h,
                width=stage_1_w,
                video_encoder=enc,
                dtype=dtype,
                device=self.device,
            )
        )
        stage_1_conditionings.append(VideoGeneratedKeyframeSlots(pixel_frame_indices=positions))

        video_state, audio_state = self.stage(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_1_sigmas,
            noiser=noiser,
            width=stage_1_w,
            height=stage_1_h,
            frames=num_frames,
            fps=_conditioning_fps(frame_rate),
            # Shipped audio is stage 1's, so it must be as long as the clip plays. The snapped
            # transformer fps would otherwise size it from ``num_frames / 60``.
            audio_fps=frame_rate,
            video=ModalitySpec(context=video_context, conditionings=stage_1_conditionings),
            audio=ModalitySpec(context=audio_context),
        )

        reserved_half_res_video = video_state.latent[:1].detach().clone()
        stage_1_audio_latent = audio_state.latent.detach().clone() if audio_state is not None else None
        if video_state.generated_keyframes is None:
            raise RuntimeError("Stage 1 did not return generated_keyframes despite requesting slots")
        upsampled_slot_keyframes = self.upsampler(video_state.generated_keyframes)
        upscaled_video_latent = self.upsampler(reserved_half_res_video)

        # --- Stage 2: spatial detailing ------------------------------------------------
        stage_2_sigmas = stage_2_sigmas.to(dtype=torch.float32, device=self.device)
        stage_2_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=stage_2_h,
                width=stage_2_w,
                video_encoder=enc,
                dtype=dtype,
                device=self.device,
            )
        )
        stage_2_conditionings.append(
            VideoGeneratedKeyframeSlots(pixel_frame_indices=positions, initial_keyframes=upsampled_slot_keyframes)
        )
        stage_2_conditionings.append(
            VideoConditionByReferenceLatent(
                latent=reserved_half_res_video,
                downscale_factor=self._detailing_downscale,
                strength=1.0,
            )
        )

        video_state, audio_state = self.stage_detailing(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_2_sigmas,
            noiser=noiser,
            width=stage_2_w,
            height=stage_2_h,
            frames=num_frames,
            fps=_conditioning_fps(frame_rate),
            # Same basis as stage 1: this pass re-noises stage 1's audio latent.
            audio_fps=frame_rate,
            video=ModalitySpec(
                context=video_context,
                conditionings=stage_2_conditionings,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=upscaled_video_latent,
            ),
            audio=ModalitySpec(
                context=audio_context,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=audio_state.latent,
            ),
        )

        # Stage 2's slots become the next round's anchors.
        carry_positions = list(positions)
        carry_keyframes = video_state.generated_keyframes
        current_fps = frame_rate
        temporal_sigmas = DISTILLED_SIGMAS[4:].to(dtype=torch.float32, device=self.device)
        last_window_seams: list[int] = []
        self._emit_stage(
            "s2",
            latent=video_state.latent,
            keyframes=carry_keyframes,
            positions=carry_positions,
            height=stage_2_h,
            width=stage_2_w,
            num_frames=num_frames,
            fps=current_fps,
        )

        for round_idx in range(1, temporal_upscalings + 1):
            assert self.temporal_upsampler is not None
            if carry_keyframes is None or not carry_positions:
                raise RuntimeError(f"Temporal round {round_idx}: missing carry-forward keyframes")

            video_latent = self.temporal_upsampler(video_state.latent[:1])
            num_frames = 2 * (num_frames - 1) + 1
            current_fps = 2 * current_fps
            # Carried keyframes are single-frame latents, so only their positions scale with the round.
            seam_positions = [2 * position for position in carry_positions]
            anchor_keyframes = carry_keyframes
            seam_to_index = {seam: index for index, seam in enumerate(seam_positions)}
            cond_fps = _conditioning_fps(current_fps)
            windows = TemporalTilePlan(seam_positions, num_frames, 2**round_idx, temporal_scale)
            last_window_seams = list(seam_positions)

            tile_latents: list[torch.Tensor] = []
            slot_positions: list[int] = []
            slot_latent_slices: list[torch.Tensor] = []
            if stage_1_audio_latent is None:
                raise RuntimeError("DFR temporal tiles need the frozen stage-1 audio latent")

            for tile_index, (interval, pixel_start, pixel_end, anchor_global, slot_global) in enumerate(windows):
                local_frames = (interval.end - interval.start - 1) * temporal_scale + 1
                tile_video = video_latent[:, :, interval.start : interval.end]

                # Image conditioning is tile-local: ``frame_idx`` is on the requested canvas, so
                # scale onto this round's grid (``* 2**round``) before the window test and re-base.
                # ``frame_idx=0`` still means this tile's first frame after re-basing, so only
                # images that actually fall inside the window are re-attached.
                tile_images = _rebase_image_conditionings(
                    images,
                    pixel_scale=2**round_idx,
                    pixel_start=pixel_start,
                    pixel_end=pixel_end,
                )
                round_conditionings = (
                    self.image_conditioner(
                        lambda enc, _images=tile_images: combined_image_conditionings(
                            images=_images,
                            height=stage_2_h,
                            width=stage_2_w,
                            video_encoder=enc,
                            dtype=dtype,
                            device=self.device,
                        )
                    )
                    if tile_images
                    else []
                )

                # Every seam in the window is a hard keyframe, including the one at local frame 0.
                if anchor_global:
                    missing = [position for position in anchor_global if position not in seam_to_index]
                    if missing:
                        raise RuntimeError(f"Anchor seams {missing} missing from the carry-forward bag")
                    anchor_latents = torch.cat(
                        [anchor_keyframes[:, :, seam_to_index[p] : seam_to_index[p] + 1] for p in anchor_global], dim=2
                    )
                    round_conditionings.extend(
                        _keyframe_conditionings_from_latents(
                            anchor_latents,
                            [int(position) - pixel_start for position in anchor_global],
                            strength=_ANCHOR_KEYFRAME_STRENGTH,
                        )
                    )

                if slot_global:
                    slot_local = [int(position) - pixel_start for position in slot_global]
                    round_conditionings.append(
                        VideoGeneratedKeyframeSlots(
                            pixel_frame_indices=slot_local,
                            initial_keyframes=_slot_initials_from_video(tile_video, slot_local, temporal_scale),
                        )
                    )

                tile_state, _ = self.stage(
                    denoiser=SimpleDenoiser(video_context, audio_context),
                    sigmas=temporal_sigmas,
                    noiser=noiser,
                    width=stage_2_w,
                    height=stage_2_h,
                    frames=local_frames,
                    fps=cond_fps,
                    video=ModalitySpec(
                        context=video_context,
                        conditionings=round_conditionings,
                        noise_scale=temporal_sigmas[0].item(),
                        initial_latent=tile_video,
                    ),
                    audio=ModalitySpec(
                        context=audio_context,
                        frozen=True,
                        noise_scale=0.0,
                        initial_latent=_audio_latent_for_tile(
                            stage_1_audio_latent,
                            pixel_start=pixel_start,
                            local_frames=local_frames,
                            playback_fps=current_fps,
                            source_duration=stage_1_duration,
                            cond_fps=cond_fps,
                        ),
                    ),
                    stepper=EulerAncestralDiffusionStep(eta=_TEMPORAL_ANCESTRAL_ETA),
                    # Tiles are positionally identical, so a shared ancestral seed would inject
                    # byte-identical noise into every one of them.
                    loop=partial(euler_ancestral_denoising_loop, noise_seed=seed + 1000 * round_idx + tile_index),
                )
                tile_latents.append(tile_state.latent[:1, :, interval.left_ramp :])

                if slot_global:
                    if tile_state.generated_keyframes is None:
                        raise RuntimeError(f"Temporal round {round_idx}: tile produced no keyframe slots")
                    slot_positions.extend(slot_global)
                    slot_latent_slices.append(tile_state.generated_keyframes)

            stitched = torch.cat(tile_latents, dim=2)
            expected_t = (num_frames - 1) // temporal_scale + 1
            if stitched.shape[2] != expected_t:
                raise RuntimeError(f"Stitched latent T={stitched.shape[2]} != expected {expected_t}")
            if not isinstance(video_state, LatentState):
                raise TypeError(f"Expected LatentState, got {type(video_state)}")
            video_state = replace(video_state, latent=stitched, generated_keyframes=None)

            slot_latents = torch.cat(slot_latent_slices, dim=2) if slot_latent_slices else None
            if slot_positions and slot_latents is not None:
                # Lead-in segments repeat the previous tile's slots; the earlier tile's version wins.
                first_index: dict[int, int] = {}
                for index, position in enumerate(slot_positions):
                    first_index.setdefault(position, index)
                slot_positions = sorted(first_index)
                slot_latents = torch.cat(
                    [slot_latents[:, :, first_index[p] : first_index[p] + 1] for p in slot_positions], dim=2
                )

            carry_positions, carry_keyframes = _merge_carry_forward_keyframes(
                seam_positions, anchor_keyframes, slot_positions, slot_latents
            )
            self._emit_stage(
                f"t{round_idx}",
                latent=video_state.latent,
                keyframes=carry_keyframes,
                positions=carry_positions,
                height=stage_2_h,
                width=stage_2_w,
                num_frames=num_frames,
                fps=current_fps,
            )

        # --- Spatial epilogue (spatial_upscalings == 2): s2 is h/2; this pass is final H x W ---
        if spatial_upscalings == 2:
            if carry_keyframes is None or not carry_positions:
                raise RuntimeError("Spatial epilogue: missing carry-forward keyframes")
            guide = video_state.latent[:1]
            pixel_keyframes = self._decode_lanczos_carry_keyframes(carry_keyframes, seed)
            upscaled_video = self.upsampler(guide)

            def _encode_epilogue_keyframes(enc: VideoEncoder) -> tuple[list[ConditioningItem], torch.Tensor]:
                image_conds = combined_image_conditionings(
                    images=_rebase_image_conditionings(images, pixel_scale=2**temporal_upscalings),
                    height=height,
                    width=width,
                    video_encoder=enc,
                    dtype=dtype,
                    device=self.device,
                )
                planes = []
                for rgb in pixel_keyframes:
                    sample = to_vae_range(rgb.to(device=self.device, dtype=torch.float32))
                    sample = sample.permute(3, 0, 1, 2).unsqueeze(0).contiguous().to(dtype=dtype)
                    planes.append(enc(sample))
                return image_conds, torch.cat(planes, dim=2)

            epi_conditionings, encoded_kfs = self.image_conditioner(_encode_epilogue_keyframes)
            epi_conditionings.extend(
                _keyframe_conditionings_from_latents(encoded_kfs, carry_positions, strength=_EPILOGUE_KEYFRAME_STRENGTH)
            )
            epi_conditionings.append(
                VideoConditionByReferenceLatent(
                    latent=guide,
                    downscale_factor=self._detailing_downscale,
                    strength=1.0,
                )
            )
            n_time_tiles = max(1, 2**temporal_upscalings)
            epi_seams = [pixel_to_latent_index(position, temporal_scale) for position in last_window_seams]
            temporal_overlap = (epi_seams[0] + 1) if n_time_tiles > 1 and epi_seams else 0
            epi_tiling = TileCountConfig(
                frames=DimensionTilingConfig(n_time_tiles, overlap=temporal_overlap),
                height=DimensionTilingConfig(2, _EPILOGUE_SPATIAL_OVERLAP),
                width=DimensionTilingConfig(2, _EPILOGUE_SPATIAL_OVERLAP),
            )
            if stage_1_audio_latent is None:
                raise RuntimeError("Spatial epilogue: missing stage-1 audio latent")
            cond_fps = _conditioning_fps(current_fps)
            epi_pixel_frames = (upscaled_video.shape[2] - 1) * temporal_scale + 1
            epi_latent = self._run_spatial_epilogue(
                latent=upscaled_video,
                conditionings=epi_conditionings,
                tiling=epi_tiling,
                sigmas=stage_2_sigmas,
                v_ctx=video_context,
                a_ctx=audio_context,
                audio_latent=_audio_latent_for_tile(
                    stage_1_audio_latent,
                    pixel_start=0,
                    local_frames=epi_pixel_frames,
                    playback_fps=current_fps,
                    source_duration=stage_1_duration,
                    cond_fps=cond_fps,
                ),
                fps=cond_fps,
                seed=seed,
                seams=epi_seams,
            )
            video_state = replace(video_state, latent=epi_latent, generated_keyframes=None)
            carry_keyframes = encoded_kfs

        # The canvas may have padded its tail, and each round maps N -> 2(N-1)+1, so the caller's
        # contract is ``(requested - 1) * 2**rounds + 1``. ``requested - 1`` is a multiple of the VAE
        # temporal scale, so the trim always lands on a latent boundary.
        target_frames = (requested_frames - 1) * 2**temporal_upscalings + 1
        if target_frames > num_frames:
            raise RuntimeError(f"Target {target_frames} frames exceeds the generated canvas {num_frames}")
        if target_frames != num_frames:
            keep_latents = (target_frames - 1) // temporal_scale + 1
            video_state = replace(video_state, latent=video_state.latent[:, :, :keep_latents])
            num_frames = target_frames
        # After the trim, so keyframes past the caller's last frame are dropped rather than
        # shipped pointing off-canvas.
        final_keyframes = decode_keyframes_from_slots(carry_keyframes, carry_positions, num_frames)

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
        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator, keyframes=final_keyframes)
        if stage_1_audio_latent is None:
            raise RuntimeError("Stage 1 produced no audio latent to ship")
        decoded_audio = self.audio_decoder(stage_1_audio_latent)
        # Audio was generated for the padded canvas, so cut it to the video's duration or the muxed
        # container outlasts the picture.
        video_seconds = num_frames / playback_fps
        audio_samples = min(decoded_audio.waveform.shape[-1], round(video_seconds * decoded_audio.sampling_rate))
        if audio_samples != decoded_audio.waveform.shape[-1]:
            decoded_audio = replace(decoded_audio, waveform=decoded_audio.waveform[..., :audio_samples])
        return PipelineOutput(
            decoded_video, decoded_audio, num_frames, tiling_config, final_keyframes, video_state.latent
        )

    def _decode_lanczos_carry_keyframes(self, keyframe_latents: torch.Tensor, seed: int) -> list[torch.Tensor]:
        """Decode each carry keyframe as its own 1-frame clip, then Lanczos-x2 the RGB.
        The VAE is causal, so a stacked encode/decode would bleed neighbouring planes. Each
        plane is a standalone one-pixel-frame latent, matching ``DecodeKeyframes``. Dist
        ``decode_single_frames`` is a local SGPU passthrough, so every rank gets pixels
        instead of the empty worker yield from ``decode_video``.
        """
        if keyframe_latents.ndim != 5:
            raise ValueError(f"Expected carry keyframes (B, C, K, H, W), got {tuple(keyframe_latents.shape)}")
        n_planes = keyframe_latents.shape[2]
        logger.info("Epilogue: decoding %d carry keyframes separately, then Lanczos x2", n_planes)
        planes = [keyframe_latents[:, :, index : index + 1] for index in range(n_planes)]
        generators = [torch.Generator(device=self.device).manual_seed(seed + 4000 + index) for index in range(n_planes)]
        rgb_planes: list[torch.Tensor] = []
        for index, rgb in enumerate(self.video_decoder.decode_single_frames(planes, generators)):
            if rgb.shape[0] < 1:
                raise RuntimeError(f"Decoder returned no pixels for carry keyframe {index}")
            rgb_planes.append(_lanczos_x2_fhwc(rgb.detach().cpu()))
        if len(rgb_planes) != n_planes:
            raise RuntimeError(f"Expected {n_planes} decoded carry keyframes, got {len(rgb_planes)}")
        return rgb_planes

    def _run_spatial_epilogue(
        self,
        *,
        latent: torch.Tensor,
        conditionings: list[ConditioningItem],
        tiling: TileCountConfig,
        sigmas: torch.Tensor,
        v_ctx: torch.Tensor,
        a_ctx: torch.Tensor,
        audio_latent: torch.Tensor,
        fps: float,
        seed: int,
        seams: Sequence[int] = (),
    ) -> torch.Tensor:
        """Full-res spatial detailing: one denoising loop over a canvas tiled inside the model.
        :class:`_TiledModelWrapper` tiles each transformer call and blends the per-tile
        predictions, so the sampler steps a single full-resolution canvas and the spatial overlap
        is reconciled at every Euler step rather than once at the end. Temporal tiles are cut on
        ``seams``, which makes their blend masks rectangular: the lead-in is denoised for context
        and then discarded, and the earlier tile keeps the seam cell. Conditionings are filtered
        per tile at the token level, so they are passed here on the full canvas.
        Audio is the frozen stage-1 latent, resampled onto this canvas, so video-audio cross
        attention still sees it. The returned audio state is discarded; the pipeline ships the
        original stage-1 audio.
        """
        tiling = _clamp_tile_to_latent(tiling, tuple(latent.shape[2:5]))
        _, _, n_frames, n_height, n_width = latent.shape
        scale_factors = self.stage_detailing.video_scale_factors
        pixel_frames = (n_frames - 1) * scale_factors.time + 1

        def _tile(model: torch.nn.Module, tools: LatentTools | None) -> torch.nn.Module:
            if not isinstance(tools, VideoLatentTools):
                raise TypeError(f"Spatial epilogue needs video tools to tile, got {type(tools).__name__}")
            return _TiledModelWrapper(model, video_tools=tools, tiling=tiling, seams=seams)

        stage = self.stage_detailing.with_model_wrapper(_tile)
        logger.info(
            "Spatial epilogue: %d step(s) over %s tiles (frames=%s height=%s width=%s, seams=%s)",
            sigmas.numel() - 1,
            tiling.frames.num_tiles * tiling.height.num_tiles * tiling.width.num_tiles,
            tiling.frames,
            tiling.height,
            tiling.width,
            list(seams),
        )
        video_state, _ = stage(
            denoiser=SimpleDenoiser(v_ctx, a_ctx),
            sigmas=sigmas.to(dtype=torch.float32, device=self.device),
            noiser=GaussianNoiser(generator=torch.Generator(device=self.device).manual_seed(seed + 2000)),
            width=n_width * scale_factors.width,
            height=n_height * scale_factors.height,
            frames=pixel_frames,
            fps=fps,
            video=ModalitySpec(
                context=v_ctx,
                conditionings=conditionings,
                noise_scale=float(sigmas[0].item()),
                initial_latent=latent.to(device=self.device, dtype=self.dtype),
            ),
            audio=ModalitySpec(
                context=a_ctx,
                frozen=True,
                noise_scale=0.0,
                initial_latent=audio_latent,
            ),
        )
        if video_state is None:
            raise RuntimeError("Spatial epilogue produced no video state")
        return video_state.latent


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
    )


if __name__ == "__main__":
    main()
