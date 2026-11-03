"""DFR diffusion stages: caller-built ``DiffusionStage`` / VAE blocks, no Gemma.
``DFRPipeline`` and other adapters construct their own stages -- including
sequence-parallel wrappers and resident weights -- and pass them in. These functions must not
instantiate ``ModelPaths`` or a prompt encoder.
Public stage functions: ``denoise_stage1``, ``denoise_stage2``, ``run_one_temporal_round``,
``run_spatial_epilogue``.
Each public stage function prepares modality specs internally (upsample, conditionings,
``ModalitySpec`` assembly) and calls ``DiffusionStage``. Adapters call only the public
entry points.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from functools import partial
from typing import NamedTuple

import torch

from ltx_core.components.noisers import GaussianNoiser, Noiser
from ltx_core.conditioning import ConditioningItem, VideoConditionByReferenceLatent, VideoGeneratedKeyframeSlots
from ltx_core.devices import upload_from_host
from ltx_core.model.video_vae import VideoEncoder
from ltx_core.tiling import DimensionTilingConfig, TileCountConfig
from ltx_core.tools import LatentTools, VideoLatentTools
from ltx_core.types import VIDEO_SCALE_FACTORS, LatentState, SpatioTemporalScaleFactors, VideoPixelShape
from ltx_pipelines.dfr_helpers.layout import TemporalTile, TemporalTilePlan
from ltx_pipelines.dfr_helpers.ops import (
    ANCHOR_KEYFRAME_STRENGTH,
    EPILOGUE_KEYFRAME_STRENGTH,
    EPILOGUE_SPATIAL_OVERLAP,
    TEMPORAL_ANCESTRAL_ETA,
    TilePrefix,
    audio_latent_for_tile,
    clamp_tile_to_latent,
    conditioning_fps,
    epilogue_tile_count,
    keyframe_conditionings_from_latents,
    lanczos_x2_fhwc,
    lead_in_carryover,
    rebase_image_conditionings,
    slot_initials_from_video,
    tile_prefix,
)
from ltx_pipelines.utils.blocks import DiffusionStage, ImageConditioner, VideoDecoder, VideoUpsampler
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import (
    create_initial_audio_latent,
    create_initial_video_latent,
    image_conditionings,
)
from ltx_pipelines.utils.media_io import to_vae_range
from ltx_pipelines.utils.samplers import euler_ancestral_denoising_loop
from ltx_pipelines.utils.tiled_diffusion import TiledDiffusionModel
from ltx_pipelines.utils.types import LoadedImageConditioning, ModalitySpec, VideoAudio

logger = logging.getLogger(__name__)

KeyframeCarry = dict[int, torch.Tensor]


class DenoiseStage1Result(NamedTuple):
    """Half-res stage-1 outputs consumed by stage 2 and later passes.
    ``video_state`` / ``audio_state`` are the denoise outputs as produced. An orchestrator
    that reuses ``.latent`` (IC-LoRA, upsample, shipped audio) should take a detached clone
    of the positive batch. ``keyframes`` maps pixel indices on the stage-1 canvas to generated
    slot planes. Stage 2 spatially upsamples video and slots before denoising.
    """

    video_state: LatentState
    audio_state: LatentState
    keyframes: KeyframeCarry


class DenoiseStage2Result(NamedTuple):
    """Full-res (or half-res) detailing pass. Shipped audio is still stage 1's clone."""

    video_state: LatentState
    audio_state: LatentState | None
    keyframes: KeyframeCarry


class SpatialEpilogueResult(NamedTuple):
    """Full-res detailing canvas. Carry bag is Lanczos-rebuilt encoded keyframe planes."""

    video_state: LatentState
    keyframes: KeyframeCarry


class TemporalTileResult(NamedTuple):
    """One ancestral temporal tile: owned latent slice plus bookkeeping for the next tile."""

    owned_latent: torch.Tensor
    previous_tile: tuple[int, torch.Tensor]
    new_planes: dict[int, torch.Tensor]


class TemporalTileVideo(NamedTuple):
    """Latent slice and pixel window one temporal tile denoises."""

    tile_video: torch.Tensor
    pixel_start: int
    pixel_end: int
    resume_pixel: int
    pinned_cells: int

    def local_frames(self, temporal_scale: int) -> int:
        return (self.tile_video.shape[2] - 1) * temporal_scale + 1


class _PreparedDenoise(NamedTuple):
    """Video/audio modality specs and canvas shape for one ``DiffusionStage`` call."""

    video: ModalitySpec
    audio: ModalitySpec
    canvas: VideoPixelShape


class _PreparedTemporalTile(NamedTuple):
    """Prepared specs plus tile geometry for one ancestral temporal tile."""

    denoise: _PreparedDenoise
    tile_video: TemporalTileVideo


class _PreparedEpilogueWindow(NamedTuple):
    """Prepared specs plus bookkeeping for one spatial-epilogue temporal window."""

    window_latent: torch.Tensor
    denoise: _PreparedDenoise
    pinned_cells: int
    prefix: TilePrefix | None
    interval_start: int


def _keyframes_from_stacked(positions: Sequence[int], planes: torch.Tensor) -> KeyframeCarry:
    """Associate a stacked keyframe tensor with pixel indices on the current canvas."""
    if planes.shape[2] != len(positions):
        raise ValueError(f"Keyframe planes K={planes.shape[2]} != {len(positions)} positions")
    return {int(position): planes[:, :, index : index + 1] for index, position in enumerate(positions)}


def _stack_keyframes(keyframes: KeyframeCarry) -> tuple[list[int], torch.Tensor]:
    """Adapt keyframe carry to APIs that require parallel positions and a stacked tensor."""
    if not keyframes:
        raise RuntimeError("Keyframe carry is empty")
    positions = list(keyframes)
    return positions, torch.cat(list(keyframes.values()), dim=2)


def _image_conditionings(
    image_conditioner: ImageConditioner,
    images: Sequence[LoadedImageConditioning],
    *,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
) -> list[ConditioningItem]:
    return image_conditioner(
        lambda enc: image_conditionings(
            images=list(images),
            height=height,
            width=width,
            video_encoder=enc,
            dtype=dtype,
            device=device,
        )
    )


def _prepare_stage1_specs(
    *,
    image_conditioner: ImageConditioner,
    images: list[LoadedImageConditioning],
    video_shape: VideoPixelShape,
    positions: Sequence[int],
    video_context: torch.Tensor,
    audio_context: torch.Tensor,
    audio_spec: ModalitySpec | None,
    device: torch.device,
    dtype: torch.dtype,
    scale_factors: SpatioTemporalScaleFactors,
) -> _PreparedDenoise:
    video_fps = conditioning_fps(video_shape.fps)
    video_latent = create_initial_video_latent(
        pixel_shape=video_shape._replace(fps=video_fps),
        device=device,
        dtype=dtype,
        scale_factors=scale_factors,
    )
    if audio_spec is None:
        audio_spec = ModalitySpec(
            latent=create_initial_audio_latent(
                video_latent,
                fps=video_shape.fps,
                video_scale_factors=scale_factors,
            ),
            conditioning_fps=video_shape.fps,
            context=audio_context,
        )
    conditionings = _image_conditionings(
        image_conditioner,
        images,
        height=video_shape.height,
        width=video_shape.width,
        dtype=dtype,
        device=device,
    )
    conditionings.append(VideoGeneratedKeyframeSlots(pixel_frame_indices=list(positions)))
    return _PreparedDenoise(
        video=ModalitySpec(
            latent=video_latent,
            conditioning_fps=video_fps,
            context=video_context,
            conditionings=conditionings,
        ),
        audio=audio_spec,
        canvas=video_shape,
    )


def _prepare_stage2_specs(  # noqa: PLR0913
    *,
    upsampler: VideoUpsampler,
    image_conditioner: ImageConditioner,
    images: list[LoadedImageConditioning],
    video_shape: VideoPixelShape,
    keyframes: KeyframeCarry,
    half_res_reference: torch.Tensor,
    downscale_factor: int,
    audio_latent: torch.Tensor,
    video_context: torch.Tensor,
    audio_context: torch.Tensor,
    sigmas: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> _PreparedDenoise:
    upscaled_video_latent = upsampler(half_res_reference)
    positions, stacked_keyframes = _stack_keyframes(keyframes)
    upscaled_keyframes = upsampler(stacked_keyframes)
    conditionings = _image_conditionings(
        image_conditioner,
        images,
        height=video_shape.height,
        width=video_shape.width,
        dtype=dtype,
        device=device,
    )
    conditionings.append(
        VideoGeneratedKeyframeSlots(
            pixel_frame_indices=positions,
            initial_keyframes=upscaled_keyframes,
        )
    )
    conditionings.append(
        VideoConditionByReferenceLatent(
            latent=half_res_reference,
            downscale_factor=downscale_factor,
            strength=1.0,
        )
    )
    return _PreparedDenoise(
        video=ModalitySpec(
            latent=upscaled_video_latent,
            conditioning_fps=conditioning_fps(video_shape.fps),
            context=video_context,
            conditionings=conditionings,
            noise_scale=sigmas[0].item(),
        ),
        audio=ModalitySpec(
            latent=audio_latent,
            conditioning_fps=video_shape.fps,
            context=audio_context,
            noise_scale=sigmas[0].item(),
        ),
        canvas=video_shape,
    )


def denoise_stage1(  # noqa: PLR0913
    *,
    stage: DiffusionStage,
    image_conditioner: ImageConditioner,
    images: list[LoadedImageConditioning],
    video_shape: VideoPixelShape,
    positions: Sequence[int],
    video_context: torch.Tensor,
    audio_context: torch.Tensor,
    sigmas: torch.Tensor,
    noiser: Noiser,
    device: torch.device,
    dtype: torch.dtype,
    audio_spec: ModalitySpec | None = None,
    loop: Callable[..., tuple[LatentState | None, LatentState | None]] | None = None,
) -> DenoiseStage1Result:
    """Stage 1: generate half-res video + keyframe slots; reserve video latent for IC-LoRA.
    ``audio_spec`` defaults to a generated soundtrack. For A2V pass
    ``ModalitySpec(latent=..., context=audio_context, frozen=True, noise_scale=0.0)`` so the
    supplied latent stays frozen without re-noising (``ModalitySpec`` defaults ``noise_scale`` to
    ``1.0``, which would noise the latent before freezing it).
    Transformer fps snaps above 30; audio duration stays at playback ``video_shape.fps``.
    """
    # Kept on the host: the loop uploads the schedule, and host reads of it stay free.
    sigmas = sigmas.to(dtype=torch.float32)
    prepared = _prepare_stage1_specs(
        image_conditioner=image_conditioner,
        images=images,
        video_shape=video_shape,
        positions=positions,
        video_context=video_context,
        audio_context=audio_context,
        audio_spec=audio_spec,
        device=device,
        dtype=dtype,
        scale_factors=stage.video_scale_factors,
    )
    video_state, audio_state = stage(
        denoiser=SimpleDenoiser(),
        sigmas=sigmas,
        noiser=noiser,
        modalities=VideoAudio(video=prepared.video, audio=prepared.audio),
        loop=loop,
    )
    if video_state is None:
        raise RuntimeError("Stage 1 produced no video state")
    if audio_state is None:
        raise RuntimeError("Stage 1 produced no audio state")
    if video_state.generated_keyframes is None:
        raise RuntimeError("Stage 1 did not return generated_keyframes despite requesting slots")
    return DenoiseStage1Result(
        video_state=video_state,
        audio_state=audio_state,
        keyframes=_keyframes_from_stacked(positions, video_state.generated_keyframes),
    )


def denoise_stage2(  # noqa: PLR0913
    *,
    stage: DiffusionStage,
    image_conditioner: ImageConditioner,
    upsampler: VideoUpsampler,
    images: list[LoadedImageConditioning],
    video_shape: VideoPixelShape,
    keyframes: KeyframeCarry,
    half_res_reference: torch.Tensor,
    downscale_factor: int,
    audio_latent: torch.Tensor,
    video_context: torch.Tensor,
    audio_context: torch.Tensor,
    sigmas: torch.Tensor,
    noiser: Noiser,
    device: torch.device,
    dtype: torch.dtype,
    loop: Callable[..., tuple[LatentState | None, LatentState | None]] | None = None,
) -> DenoiseStage2Result:
    """Stage 2: spatial upsample, then detailing IC-LoRA + seeded slots.
    Upsamples ``half_res_reference`` and ``keyframes`` as its first step, matching temporal
    rounds and the epilogue. Audio is re-noised for cross-modal attention only.
    """
    # Kept on the host: the loop uploads the schedule, and host reads of it stay free.
    sigmas = sigmas.to(dtype=torch.float32)
    prepared = _prepare_stage2_specs(
        upsampler=upsampler,
        image_conditioner=image_conditioner,
        images=images,
        video_shape=video_shape,
        keyframes=keyframes,
        half_res_reference=half_res_reference,
        downscale_factor=downscale_factor,
        audio_latent=audio_latent,
        video_context=video_context,
        audio_context=audio_context,
        sigmas=sigmas,
        device=device,
        dtype=dtype,
    )
    video_state, audio_state = stage(
        denoiser=SimpleDenoiser(),
        sigmas=sigmas,
        noiser=noiser,
        modalities=VideoAudio(video=prepared.video, audio=prepared.audio),
        loop=loop,
    )
    if video_state is None:
        raise RuntimeError("Stage 2 produced no video state")
    if video_state.generated_keyframes is None:
        raise RuntimeError("Stage 2 did not return generated_keyframes despite requesting slots")
    return DenoiseStage2Result(
        video_state=video_state,
        audio_state=audio_state,
        keyframes=_keyframes_from_stacked(list(keyframes), video_state.generated_keyframes),
    )


def _temporal_tile_video_input(
    *,
    video_latent: torch.Tensor,
    tile: TemporalTile,
    prefix: TilePrefix | None,
    plane_at: dict[int, torch.Tensor],
    temporal_scale: int,
) -> TemporalTileVideo:
    """Build the latent slice and pixel window one temporal tile denoises."""
    interval = tile.interval
    if prefix is None:
        tile_video = video_latent[:, :, interval.start : interval.end]
        return TemporalTileVideo(tile_video, 0, (tile_video.shape[2] - 1) * temporal_scale, 0, 0)

    tile_video = torch.cat(
        [
            plane_at[prefix.keyframe_position].to(dtype=video_latent.dtype),
            video_latent[:, :, prefix.video_start_cell : interval.end],
        ],
        dim=2,
    )
    local_frames = (tile_video.shape[2] - 1) * temporal_scale + 1
    pixel_start = prefix.keyframe_position
    return TemporalTileVideo(tile_video, pixel_start, pixel_start + local_frames - 1, prefix.resume_pixel, prefix.cells)


def _temporal_tile_conditionings(  # noqa: PLR0913
    *,
    image_conditioner: ImageConditioner,
    images: list[LoadedImageConditioning],
    canvas: VideoPixelShape,
    tile_video: TemporalTileVideo,
    tile: TemporalTile,
    prefix: TilePrefix | None,
    previous_tile: tuple[int, torch.Tensor] | None,
    keyframes: KeyframeCarry,
    pixel_scale: int,
    temporal_scale: int,
    round_idx: int,
    tile_index: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[ConditioningItem]:
    """Assemble image, anchor, slot, and lead-in conditionings for one temporal tile."""
    tile_images = rebase_image_conditionings(
        images,
        pixel_scale=pixel_scale,
        pixel_start=tile_video.pixel_start,
        pixel_end=tile_video.pixel_end,
    )
    conditionings = (
        _image_conditionings(
            image_conditioner,
            tile_images,
            height=canvas.height,
            width=canvas.width,
            dtype=dtype,
            device=device,
        )
        if tile_images
        else []
    )

    kept_anchors = tuple(position for position in tile.anchors if position >= tile_video.resume_pixel)
    if kept_anchors:
        missing = [position for position in kept_anchors if position not in keyframes]
        if missing:
            raise RuntimeError(f"Anchor seams {missing} missing from the carry-forward bag")
        anchor_latents = torch.cat([keyframes[position] for position in kept_anchors], dim=2)
        conditionings.extend(
            keyframe_conditionings_from_latents(
                anchor_latents,
                [int(position) - tile_video.pixel_start for position in kept_anchors],
                strength=ANCHOR_KEYFRAME_STRENGTH,
            )
        )

    if tile.slots:
        slot_local = [int(position) - tile_video.pixel_start for position in tile.slots]
        conditionings.append(
            VideoGeneratedKeyframeSlots(
                pixel_frame_indices=slot_local,
                initial_keyframes=slot_initials_from_video(tile_video.tile_video, slot_local, temporal_scale),
            )
        )

    if prefix is not None and previous_tile is not None:
        conditionings.append(lead_in_carryover(previous_tile, prefix, tile_video.tile_video[:, :, :1]))
        logger.info(
            "Temporal round %d tile %d: starts on the plane at %d, %d cell(s) pinned, denoises from %d",
            round_idx,
            tile_index,
            prefix.keyframe_position,
            prefix.cells,
            tile_video.resume_pixel,
        )

    return conditionings


def _prepare_temporal_tile_specs(  # noqa: PLR0913
    *,
    image_conditioner: ImageConditioner,
    images: list[LoadedImageConditioning],
    canvas: VideoPixelShape,
    video_latent: torch.Tensor,
    tile_index: int,
    tile: TemporalTile,
    prefix: TilePrefix | None,
    previous_tile: tuple[int, torch.Tensor] | None,
    plane_at: dict[int, torch.Tensor],
    keyframes: KeyframeCarry,
    round_idx: int,
    pixel_scale: int,
    temporal_scale: int,
    audio_latent: torch.Tensor,
    source_duration: float,
    video_context: torch.Tensor,
    audio_context: torch.Tensor,
    sigmas: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> _PreparedTemporalTile:
    """Upsample is done at round scope; build tile slice, conditionings, and modality specs."""
    tile_video = _temporal_tile_video_input(
        video_latent=video_latent,
        tile=tile,
        prefix=prefix,
        plane_at=plane_at,
        temporal_scale=temporal_scale,
    )
    local_frames = tile_video.local_frames(temporal_scale)
    cond_fps = conditioning_fps(canvas.fps)
    conditionings = _temporal_tile_conditionings(
        image_conditioner=image_conditioner,
        images=images,
        canvas=canvas,
        tile_video=tile_video,
        tile=tile,
        prefix=prefix,
        previous_tile=previous_tile,
        keyframes=keyframes,
        pixel_scale=pixel_scale,
        temporal_scale=temporal_scale,
        round_idx=round_idx,
        tile_index=tile_index,
        device=device,
        dtype=dtype,
    )
    tile_canvas = canvas._replace(frames=local_frames)
    return _PreparedTemporalTile(
        denoise=_PreparedDenoise(
            video=ModalitySpec(
                latent=tile_video.tile_video,
                conditioning_fps=cond_fps,
                context=video_context,
                conditionings=conditionings,
                noise_scale=sigmas[0].item(),
            ),
            audio=ModalitySpec(
                latent=audio_latent_for_tile(
                    audio_latent,
                    pixel_start=tile_video.pixel_start,
                    local_frames=local_frames,
                    playback_fps=canvas.fps,
                    source_duration=source_duration,
                    cond_fps=cond_fps,
                ),
                conditioning_fps=cond_fps,
                context=audio_context,
                frozen=True,
                noise_scale=0.0,
            ),
            canvas=tile_canvas,
        ),
        tile_video=tile_video,
    )


def _collect_temporal_tile_result(
    *,
    tile_state: LatentState,
    tile: TemporalTile,
    tile_video: TemporalTileVideo,
    prefix: TilePrefix | None,
    round_idx: int,
) -> TemporalTileResult:
    interval = tile.interval
    next_previous = (
        prefix.video_start_cell - 1 if prefix else interval.start,
        tile_state.latent[:1].detach(),
    )
    owned_latent = tile_state.latent[:1, :, tile_video.pinned_cells :]

    if not tile.slots:
        return TemporalTileResult(owned_latent, next_previous, {})

    if tile_state.generated_keyframes is None:
        raise RuntimeError(f"Temporal round {round_idx}: tile produced no keyframe slots")
    new_planes = {
        int(position): tile_state.generated_keyframes[:, :, slot_index : slot_index + 1]
        for slot_index, position in enumerate(tile.slots)
    }
    return TemporalTileResult(owned_latent, next_previous, new_planes)


def _denoise_temporal_tile(  # noqa: PLR0913
    *,
    stage: DiffusionStage,
    image_conditioner: ImageConditioner,
    images: list[LoadedImageConditioning],
    canvas: VideoPixelShape,
    video_latent: torch.Tensor,
    tile_index: int,
    tile: TemporalTile,
    prefix: TilePrefix | None,
    previous_tile: tuple[int, torch.Tensor] | None,
    plane_at: dict[int, torch.Tensor],
    keyframes: KeyframeCarry,
    round_idx: int,
    pixel_scale: int,
    temporal_scale: int,
    audio_latent: torch.Tensor,
    source_duration: float,
    video_context: torch.Tensor,
    audio_context: torch.Tensor,
    sigmas: torch.Tensor,
    noiser: Noiser,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> TemporalTileResult:
    """Ancestral-densify one temporal tile. Returns owned latent and any new slot planes."""
    prepared = _prepare_temporal_tile_specs(
        image_conditioner=image_conditioner,
        images=images,
        canvas=canvas,
        video_latent=video_latent,
        tile_index=tile_index,
        tile=tile,
        prefix=prefix,
        previous_tile=previous_tile,
        plane_at=plane_at,
        keyframes=keyframes,
        round_idx=round_idx,
        pixel_scale=pixel_scale,
        temporal_scale=temporal_scale,
        audio_latent=audio_latent,
        source_duration=source_duration,
        video_context=video_context,
        audio_context=audio_context,
        sigmas=sigmas,
        device=device,
        dtype=dtype,
    )
    denoise = prepared.denoise
    tile_state, _ = stage(
        denoiser=SimpleDenoiser(),
        sigmas=sigmas,
        noiser=noiser,
        modalities=VideoAudio(video=denoise.video, audio=denoise.audio),
        loop=partial(
            euler_ancestral_denoising_loop,
            noise_seed=seed + 1000 * round_idx + tile_index,
            eta=TEMPORAL_ANCESTRAL_ETA,
        ),
    )
    if tile_state is None:
        raise RuntimeError(f"Temporal round {round_idx}: tile produced no video state")
    return _collect_temporal_tile_result(
        tile_state=tile_state,
        tile=tile,
        tile_video=prepared.tile_video,
        prefix=prefix,
        round_idx=round_idx,
    )


def run_one_temporal_round(  # noqa: PLR0913
    *,
    round_idx: int,
    stage: DiffusionStage,
    temporal_upsampler: VideoUpsampler,
    image_conditioner: ImageConditioner,
    video_state: LatentState,
    keyframes: KeyframeCarry,
    video_shape: VideoPixelShape,
    images: list[LoadedImageConditioning],
    video_context: torch.Tensor,
    audio_context: torch.Tensor,
    audio_latent: torch.Tensor,
    source_duration: float,
    seed: int,
    noiser: Noiser,
    sigmas: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    temporal_scale: int,
) -> tuple[LatentState, KeyframeCarry, list[int]]:
    """Upsample once, ancestral-densify all tiles, stitch, and rebuild the carry-forward bag.
    ``video_shape`` is the desired canvas *after* this round (x2 frames and fps vs the previous
    stage), matching how stage 1 / stage 2 take their target ``VideoPixelShape``.
    """
    video_latent = temporal_upsampler(video_state.latent[:1])
    scaled_keyframes = {2 * position: plane for position, plane in keyframes.items()}
    seam_positions = list(scaled_keyframes)
    windows = TemporalTilePlan(seam_positions, video_shape.frames, 2**round_idx, temporal_scale)
    tile_latents: list[torch.Tensor] = []
    previous_tile: tuple[int, torch.Tensor] | None = None
    plane_at = dict(scaled_keyframes)

    for tile_index, tile in enumerate(windows):
        prefix = (
            None
            if tile_index == 0
            else tile_prefix((tile.interval.start - 1) * temporal_scale, plane_at, temporal_scale)
        )
        tile_result = _denoise_temporal_tile(
            stage=stage,
            image_conditioner=image_conditioner,
            images=images,
            canvas=video_shape,
            video_latent=video_latent,
            tile_index=tile_index,
            tile=tile,
            prefix=prefix,
            previous_tile=previous_tile,
            plane_at=plane_at,
            keyframes=scaled_keyframes,
            round_idx=round_idx,
            pixel_scale=2**round_idx,
            temporal_scale=temporal_scale,
            audio_latent=audio_latent,
            source_duration=source_duration,
            video_context=video_context,
            audio_context=audio_context,
            sigmas=sigmas,
            noiser=noiser,
            seed=seed,
            device=device,
            dtype=dtype,
        )
        previous_tile = tile_result.previous_tile
        tile_latents.append(tile_result.owned_latent)
        for position, plane in tile_result.new_planes.items():
            plane_at.setdefault(position, plane)

    stitched = torch.cat(tile_latents, dim=2)
    expected_t = (video_shape.frames - 1) // temporal_scale + 1
    if stitched.shape[2] != expected_t:
        raise RuntimeError(f"Stitched latent T={stitched.shape[2]} != expected {expected_t}")
    next_video_state = replace(video_state, latent=stitched, generated_keyframes=None)
    return next_video_state, dict(sorted(plane_at.items())), seam_positions


def _rebuild_carry_keyframes(
    video_decoder: VideoDecoder,
    image_conditioner: ImageConditioner,
    keyframes: KeyframeCarry,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> KeyframeCarry:
    """Decode, Lanczos-x2 and re-encode each carry plane on the rank that owns it.
    Each plane is a standalone one-pixel-frame clip with its own generator, so its pixels do
    not depend on the world size. A Dist decoder splits the planes round-robin over the group;
    each rank resamples and encodes its own planes and only the small encoded latents cross the
    group. The Lanczos runs on the decode device right after each plane; every encode is queued
    before any read-back. Returns every plane on every rank.
    """
    for position, plane in keyframes.items():
        if plane.ndim != 5 or plane.shape[2] != 1:
            raise ValueError(f"Expected one keyframe plane at {position}, got {tuple(plane.shape)}")
    positions = list(keyframes)
    logger.info("Epilogue: decoding %d carry keyframes separately", len(positions))
    generators = [torch.Generator(device=device).manual_seed(seed + 4000 + index) for index in range(len(positions))]
    owned = video_decoder.decode_owned_single_frames(list(keyframes.values()), generators)

    resampled: dict[int, torch.Tensor] = {}
    for index, rgb in zip(owned.indices, owned.planes, strict=True):
        if rgb.shape[0] < 1:
            raise RuntimeError(f"Decoder returned no pixels for carry keyframe {index}")
        resampled[index] = lanczos_x2_fhwc(rgb.detach())

    def encode(enc: VideoEncoder) -> dict[int, torch.Tensor]:
        on_device = {}
        for index, rgb in resampled.items():
            sample = to_vae_range(rgb.to(device=device, dtype=torch.float32))
            on_device[index] = enc(sample.permute(3, 0, 1, 2).unsqueeze(0).contiguous().to(dtype=dtype))
        return {index: plane.cpu() for index, plane in on_device.items()}

    encoded = owned.gather(image_conditioner(encode))
    if len(encoded) != len(positions):
        raise RuntimeError(f"Expected {len(positions)} encoded carry keyframes, got {len(encoded)}")
    return {position: upload_from_host(encoded[index], device) for index, position in enumerate(positions)}


def _rebuild_epilogue_keyframes(
    *,
    video_decoder: VideoDecoder,
    image_conditioner: ImageConditioner,
    keyframes: KeyframeCarry,
    video_state: LatentState,
    images: Sequence[LoadedImageConditioning],
    temporal_upscalings: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[KeyframeCarry, torch.Tensor | None]:
    """Rebuild the carry planes at the epilogue resolution, plus the opening plane if no image conditions frame 0."""
    opening_position = -1
    to_decode = dict(keyframes)
    epi_images = rebase_image_conditionings(images, pixel_scale=2**temporal_upscalings)
    if not any(image.frame_idx == 0 for image in epi_images):
        to_decode[opening_position] = video_state.latent[:1, :, :1]
    encoded = _rebuild_carry_keyframes(
        video_decoder, image_conditioner, to_decode, seed=seed, device=device, dtype=dtype
    )
    opening = encoded.pop(opening_position, None)
    return encoded, opening


def _prepare_tiled_spatial_denoise_specs(
    *,
    latent: torch.Tensor,
    conditionings: list[ConditioningItem],
    video_context: torch.Tensor,
    audio_context: torch.Tensor,
    audio_latent: torch.Tensor,
    fps: float,
    scale_factors: SpatioTemporalScaleFactors,
    device: torch.device,
    dtype: torch.dtype,
    noise_scale: float,
) -> _PreparedDenoise:
    _, _, n_frames, n_height, n_width = latent.shape
    pixel_frames = (n_frames - 1) * scale_factors.time + 1
    return _PreparedDenoise(
        video=ModalitySpec(
            latent=latent.to(device=device, dtype=dtype),
            conditioning_fps=fps,
            context=video_context,
            conditionings=conditionings,
            noise_scale=noise_scale,
        ),
        audio=ModalitySpec(
            latent=audio_latent,
            conditioning_fps=fps,
            context=audio_context,
            frozen=True,
            noise_scale=0.0,
        ),
        canvas=VideoPixelShape(
            batch=1,
            frames=pixel_frames,
            height=n_height * scale_factors.height,
            width=n_width * scale_factors.width,
            fps=fps,
        ),
    )


def _prepare_epilogue_window_specs(  # noqa: PLR0913
    *,
    index: int,
    num_windows: int,
    window: TemporalTile,
    prefix: TilePrefix | None,
    latent: torch.Tensor,
    guide: torch.Tensor,
    plane_at: dict[int, torch.Tensor],
    window_images: list[ConditioningItem],
    keyframes: KeyframeCarry,
    opening_plane: torch.Tensor | None,
    previous: tuple[int, torch.Tensor] | None,
    downscale_factor: int,
    canvas: VideoPixelShape,
    audio_latent: torch.Tensor,
    source_duration: float,
    video_context: torch.Tensor,
    audio_context: torch.Tensor,
    noise_scale: float,
    temporal_scale: int,
    scale_factors: SpatioTemporalScaleFactors,
    device: torch.device,
    dtype: torch.dtype,
) -> _PreparedEpilogueWindow:
    interval = window.interval
    if prefix is None:
        window_latent_in = latent[:, :, interval.start : interval.end]
        reference = guide[:, :, interval.start : interval.end]
        origin, resume_pixel, pinned_cells = window.pixel_start, 0, 0
    else:
        window_latent_in = torch.cat(
            [
                plane_at[prefix.keyframe_position].to(dtype=latent.dtype),
                latent[:, :, prefix.video_start_cell : interval.end],
            ],
            dim=2,
        )
        reference = guide[:, :, prefix.video_start_cell - 1 : interval.end]
        origin, resume_pixel, pinned_cells = (
            prefix.keyframe_position,
            prefix.resume_pixel,
            prefix.cells,
        )
    local_frames = (window_latent_in.shape[2] - 1) * temporal_scale + 1
    conditionings: list[ConditioningItem] = list(window_images)

    for position, plane in keyframes.items():
        if origin <= position <= window.pixel_end and position >= resume_pixel:
            conditionings.extend(
                keyframe_conditionings_from_latents(
                    plane,
                    [int(position) - origin],
                    strength=EPILOGUE_KEYFRAME_STRENGTH,
                )
            )
    if opening_plane is not None and prefix is None:
        conditionings.extend(
            keyframe_conditionings_from_latents(opening_plane, [0], strength=EPILOGUE_KEYFRAME_STRENGTH)
        )
    conditionings.append(
        VideoConditionByReferenceLatent(
            latent=reference,
            downscale_factor=downscale_factor,
            strength=1.0,
        )
    )
    if prefix is not None and previous is not None:
        conditionings.append(lead_in_carryover(previous, prefix, window_latent_in[:, :, :1]))
    logger.info(
        "Epilogue window %d/%d: starts at px %d, %d cell(s) pinned, denoises from %d",
        index + 1,
        num_windows,
        origin,
        pinned_cells,
        resume_pixel,
    )

    cond_fps = conditioning_fps(canvas.fps)
    return _PreparedEpilogueWindow(
        window_latent=window_latent_in,
        denoise=_prepare_tiled_spatial_denoise_specs(
            latent=window_latent_in,
            conditionings=conditionings,
            video_context=video_context,
            audio_context=audio_context,
            audio_latent=audio_latent_for_tile(
                audio_latent,
                pixel_start=origin,
                local_frames=local_frames,
                playback_fps=canvas.fps,
                source_duration=source_duration,
                cond_fps=cond_fps,
            ),
            fps=cond_fps,
            scale_factors=scale_factors,
            device=device,
            dtype=dtype,
            noise_scale=noise_scale,
        ),
        pinned_cells=pinned_cells,
        prefix=prefix,
        interval_start=interval.start,
    )


def _execute_tiled_spatial_denoise(
    stage: DiffusionStage,
    *,
    prepared: _PreparedDenoise,
    latent: torch.Tensor,
    tiling: TileCountConfig,
    sigmas: torch.Tensor,
    seed: int,
    device: torch.device,
    loop: Callable[..., tuple[LatentState | None, LatentState | None]] | None = None,
) -> torch.Tensor:
    """Run one spatially tiled denoising pass from prepared modality specs."""
    tiling = clamp_tile_to_latent(tiling, tuple(latent.shape[2:5]))

    def _tile(model: torch.nn.Module, tools: LatentTools | None) -> torch.nn.Module:
        if not isinstance(tools, VideoLatentTools):
            raise TypeError(f"Spatial epilogue needs video tools to tile, got {type(tools).__name__}")
        return TiledDiffusionModel(model, video_tools=tools, tiling=tiling)

    tiled_stage = stage.with_model_wrapper(_tile)
    logger.info(
        "Spatial epilogue: %d step(s) over %s tiles (frames=%s height=%s width=%s)",
        sigmas.numel() - 1,
        tiling.frames.num_tiles * tiling.height.num_tiles * tiling.width.num_tiles,
        tiling.frames,
        tiling.height,
        tiling.width,
    )
    video_state, _ = tiled_stage(
        denoiser=SimpleDenoiser(),
        sigmas=sigmas.to(dtype=torch.float32),
        noiser=GaussianNoiser(generator=torch.Generator(device=device).manual_seed(seed + 2000)),
        modalities=VideoAudio(video=prepared.video, audio=prepared.audio),
        loop=loop,
    )
    if video_state is None:
        raise RuntimeError("Spatial epilogue produced no video state")
    return video_state.latent


def _epilogue_sigma_phases(sigmas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Split a detailing schedule into one coarse step and any remaining fine steps."""
    if sigmas.ndim != 1 or sigmas.numel() < 2:
        raise ValueError(f"Epilogue schedule needs at least one step (2 sigma values), got shape {tuple(sigmas.shape)}")
    if sigmas.numel() == 2:
        return sigmas, None
    return sigmas[:2], sigmas[1:]


def _run_epilogue_windows(  # noqa: PLR0913
    *,
    stage: DiffusionStage,
    latent: torch.Tensor,
    guide: torch.Tensor,
    windows: TemporalTilePlan,
    prefixes: Sequence[TilePrefix | None],
    window_images: Sequence[list[ConditioningItem]],
    keyframes: KeyframeCarry,
    opening_plane: torch.Tensor | None,
    sigmas: torch.Tensor,
    video_context: torch.Tensor,
    audio_context: torch.Tensor,
    audio_latent: torch.Tensor,
    source_duration: float,
    canvas: VideoPixelShape,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    downscale_factor: int,
    temporal_scale: int,
    loop: Callable[..., tuple[LatentState | None, LatentState | None]] | None = None,
) -> torch.Tensor:
    """Detail each temporal window with one step, then any remaining steps, on one spatial tile grid."""
    stitched: list[torch.Tensor] = []
    previous: tuple[int, torch.Tensor] | None = None
    coarse_sigmas, fine_sigmas = _epilogue_sigma_phases(sigmas)
    tiling = TileCountConfig(
        frames=DimensionTilingConfig(1, overlap=0),
        height=DimensionTilingConfig(epilogue_tile_count(int(latent.shape[3])), EPILOGUE_SPATIAL_OVERLAP),
        width=DimensionTilingConfig(epilogue_tile_count(int(latent.shape[4])), EPILOGUE_SPATIAL_OVERLAP),
    )
    scale_factors = stage.video_scale_factors
    for index, window in enumerate(windows):
        prepared_window = _prepare_epilogue_window_specs(
            index=index,
            num_windows=len(windows),
            window=window,
            prefix=prefixes[index],
            latent=latent,
            guide=guide,
            plane_at=keyframes,
            window_images=window_images[index],
            keyframes=keyframes,
            opening_plane=opening_plane,
            previous=previous,
            downscale_factor=downscale_factor,
            canvas=canvas,
            audio_latent=audio_latent,
            source_duration=source_duration,
            video_context=video_context,
            audio_context=audio_context,
            noise_scale=float(coarse_sigmas[0].item()),
            temporal_scale=temporal_scale,
            scale_factors=scale_factors,
            device=device,
            dtype=dtype,
        )
        window_latent = _execute_tiled_spatial_denoise(
            stage,
            prepared=prepared_window.denoise,
            latent=prepared_window.window_latent,
            tiling=tiling,
            sigmas=coarse_sigmas,
            seed=seed + 100 * index,
            device=device,
            loop=loop,
        )
        if fine_sigmas is not None:
            audio_latent_in = prepared_window.denoise.audio.latent
            if audio_latent_in is None:
                raise RuntimeError("Epilogue window prepared no audio latent")
            fine_prepared = _prepare_tiled_spatial_denoise_specs(
                latent=window_latent,
                conditionings=list(prepared_window.denoise.video.conditionings),
                video_context=video_context,
                audio_context=audio_context,
                audio_latent=audio_latent_in,
                fps=prepared_window.denoise.canvas.fps,
                scale_factors=scale_factors,
                device=device,
                dtype=dtype,
                noise_scale=0.0,
            )
            window_latent = _execute_tiled_spatial_denoise(
                stage,
                prepared=fine_prepared,
                latent=window_latent,
                tiling=tiling,
                sigmas=fine_sigmas,
                seed=seed + 100 * index,
                device=device,
                loop=loop,
            )
        prefix = prepared_window.prefix
        previous = (
            prefix.video_start_cell - 1 if prefix else prepared_window.interval_start,
            window_latent.detach(),
        )
        stitched.append(window_latent[:, :, prepared_window.pinned_cells :])
    return torch.cat(stitched, dim=2)


def _plan_epilogue_windows(
    *,
    n_latent_frames: int,
    last_window_seams: Sequence[int],
    keyframe_positions: Iterable[int],
    temporal_upscalings: int,
    temporal_scale: int,
) -> tuple[TemporalTilePlan, tuple[TilePrefix | None, ...]]:
    """Cut the upscaled canvas into temporal tiles and pin each non-first tile on a carry keyframe."""
    pixel_frames = (n_latent_frames - 1) * temporal_scale + 1
    windows = TemporalTilePlan(
        last_window_seams,
        pixel_frames,
        max(1, 2**temporal_upscalings),
        temporal_scale,
    )
    prefixes = tuple(
        None
        if index == 0
        else tile_prefix((window.interval.start - 1) * temporal_scale, keyframe_positions, temporal_scale)
        for index, window in enumerate(windows)
    )
    return windows, prefixes


def _encode_epilogue_window_images(
    enc: VideoEncoder,
    *,
    windows: TemporalTilePlan,
    prefixes: Sequence[TilePrefix | None],
    images: Sequence[LoadedImageConditioning],
    temporal_upscalings: int,
    video_shape: VideoPixelShape,
    dtype: torch.dtype,
    device: torch.device,
) -> list[list[ConditioningItem]]:
    """Encode user image conditionings that fall inside each epilogue window."""
    per_window: list[list[ConditioningItem]] = []
    for index, window in enumerate(windows):
        prefix = prefixes[index]
        origin = window.pixel_start if prefix is None else prefix.keyframe_position
        resume = 0 if prefix is None else prefix.resume_pixel
        inside = [
            image
            for image in rebase_image_conditionings(
                images,
                pixel_scale=2**temporal_upscalings,
                pixel_start=origin,
                pixel_end=window.pixel_end,
            )
            if image.frame_idx + origin >= resume
        ]
        per_window.append(
            image_conditionings(
                images=inside,
                height=video_shape.height,
                width=video_shape.width,
                video_encoder=enc,
                dtype=dtype,
                device=device,
            )
            if inside
            else []
        )
    return per_window


def run_spatial_epilogue(  # noqa: PLR0913
    *,
    stage: DiffusionStage,
    upsampler: VideoUpsampler,
    image_conditioner: ImageConditioner,
    video_decoder: VideoDecoder,
    video_state: LatentState,
    keyframes: KeyframeCarry,
    images: Sequence[LoadedImageConditioning],
    video_shape: VideoPixelShape,
    temporal_upscalings: int,
    last_window_seams: Sequence[int],
    video_context: torch.Tensor,
    audio_context: torch.Tensor,
    audio_latent: torch.Tensor | None,
    source_duration: float,
    sigmas: torch.Tensor,
    downscale_factor: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    temporal_scale: int = VIDEO_SCALE_FACTORS.time,
    loop: Callable[..., tuple[LatentState | None, LatentState | None]] | None = None,
) -> SpatialEpilogueResult:
    """Full-res spatial detailing: Lanczos KF rebuild, x2 video upsample, per-window tiled passes.
    ``video_shape`` is the epilogue canvas (final output height, width, frame count, fps).
    Temporal tiles cut on
    ``last_window_seams`` from the last t-round (empty when ``temporal_upscalings == 0``).
    Conditioning fps snaps to 60 above 30. Carry keyframes are decoded one plane at a time,
    Lanczos-stretched x2, and encoded as strength-1 conditions; only the video latent is
    spatially upsampled. Frozen stage-1 audio is resampled onto each window for
    cross-attention; the caller still ships the original stage-1 audio. Image ``frame_idx`` is
    scaled by ``2**temporal_upscalings`` onto the final canvas. Trim stays with the caller.
    """
    if not keyframes:
        raise RuntimeError("Spatial epilogue: missing carry-forward keyframes")
    if audio_latent is None:
        raise RuntimeError("Spatial epilogue: missing stage-1 audio latent")
    encoded_keyframes, encoded_opening = _rebuild_epilogue_keyframes(
        video_decoder=video_decoder,
        image_conditioner=image_conditioner,
        keyframes=keyframes,
        video_state=video_state,
        images=images,
        temporal_upscalings=temporal_upscalings,
        seed=seed,
        device=device,
        dtype=dtype,
    )
    guide = video_state.latent[:1]
    upscaled = upsampler(guide)
    windows, prefixes = _plan_epilogue_windows(
        n_latent_frames=upscaled.shape[2],
        last_window_seams=last_window_seams,
        keyframe_positions=keyframes,
        temporal_upscalings=temporal_upscalings,
        temporal_scale=temporal_scale,
    )
    window_images = image_conditioner(
        lambda enc: _encode_epilogue_window_images(
            enc,
            windows=windows,
            prefixes=prefixes,
            images=images,
            temporal_upscalings=temporal_upscalings,
            video_shape=video_shape,
            dtype=dtype,
            device=device,
        )
    )
    if encoded_opening is not None:
        logger.info("Spatial epilogue: anchoring frame 0 on the generated opening frame (not shipped)")
    epi_latent = _run_epilogue_windows(
        stage=stage,
        latent=upscaled,
        guide=guide,
        windows=windows,
        prefixes=prefixes,
        window_images=window_images,
        keyframes=encoded_keyframes,
        opening_plane=encoded_opening,
        sigmas=sigmas,
        video_context=video_context,
        audio_context=audio_context,
        audio_latent=audio_latent,
        source_duration=source_duration,
        canvas=video_shape,
        seed=seed,
        device=device,
        dtype=dtype,
        downscale_factor=downscale_factor,
        temporal_scale=temporal_scale,
        loop=loop,
    )
    if epi_latent.shape[2] != upscaled.shape[2]:
        raise RuntimeError(f"Epilogue stitched T={epi_latent.shape[2]} != expected {upscaled.shape[2]}")
    return SpatialEpilogueResult(
        video_state=replace(video_state, latent=epi_latent, generated_keyframes=None),
        keyframes=encoded_keyframes,
    )
