"""Plan chunk keyframes and emit zero-latent Chunks with injected contexts and factories."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence

import torch

from ltx_core.conditioning import ConditioningItem
from ltx_core.types import SpatioTemporalScaleFactors, VideoPixelShape
from ltx_pipelines.chunks.conditionings import generated_keyframe_slots_for_chunk
from ltx_pipelines.chunks.layout import ChunkConfig, ChunkLayout, audio_latent_for_layout, uniform_chunk_layouts
from ltx_pipelines.chunks.state import (
    Chunk,
    MakeAudioConditionings,
    MakeVideoConditionings,
    VideoConditioningsWithGeneratedKeyframes,
    _empty_conditionings,
)
from ltx_pipelines.utils.helpers import (
    create_initial_av_latents,
    evenly_spaced_keyframe_positions,
    has_generated_keyframes,
    resolve_generated_keyframes,
)
from ltx_pipelines.utils.types import VideoAudio

logger = logging.getLogger(__name__)


def plan_chunk_keyframes(
    generated_keyframes: int | Sequence[int],
    *,
    num_frames: int,
    layouts: Sequence[ChunkLayout],
) -> tuple[int, ...]:
    """Resolve ``generated_keyframes`` to global pixel indices for overlapping chunks.
    An integer is a per-chunk budget. Non-final chunks reserve one slot at their seam
    and spread the rest through their new pixel range; final chunks spread every slot.
    A sequence is an exact list of global positions, assigned to chunks by layout ownership.
    """
    if not has_generated_keyframes(generated_keyframes):
        return ()
    if len(layouts) <= 1 or not isinstance(generated_keyframes, int):
        return tuple(resolve_generated_keyframes(generated_keyframes, num_frames))

    last = len(layouts) - 1
    positions = {
        position
        for index, layout in enumerate(layouts)
        for position in _budget_positions_for_layout(generated_keyframes, layout, is_final=index == last)
    }
    return tuple(sorted(positions))


def _budget_positions_for_layout(budget: int, layout: ChunkLayout, *, is_final: bool) -> tuple[int, ...]:
    start, end = layout.owned_pixel_range
    if end <= start:
        return ()
    interior_count = budget if is_final else budget - 1
    try:
        positions = tuple(
            start + position for position in evenly_spaced_keyframe_positions(interior_count, end - start)
        )
    except ValueError as error:
        raise ValueError(
            f"generated_keyframes={budget} does not fit in chunk owned pixel range [{start}, {end})"
        ) from error
    if is_final:
        return positions
    return (*positions, end - 1)


def generate_uniform_chunks(
    *,
    target: VideoPixelShape,
    context: VideoAudio[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    config: ChunkConfig,
    video_scale_factors: SpatioTemporalScaleFactors,
    make_video_conditionings: MakeVideoConditionings = _empty_conditionings,
    make_audio_conditionings: MakeAudioConditionings = _empty_conditionings,
    audio_latent: torch.Tensor | None = None,
    generated_keyframes: int | Sequence[int] = 0,
) -> Iterator[Chunk]:
    """Yield zero-latent ``Chunk``s from :func:`uniform_chunk_layouts`, with contexts and factories.
    ``config`` sets the window size and carry. When ``chunk_pixel_frames`` covers the snapped
    clip, the plan is a single chunk.
    When ``audio_latent`` is set, each chunk's audio is the matching slice rather than zeros.
    ``generated_keyframes`` is planned against these layouts and appended after the
    pipeline-owned video conditionings. The same resulting callables are stamped on
    every chunk; each factory sees that chunk's layout and latents when denoise materializes.
    """
    layouts = uniform_chunk_layouts(target_pixel_shape=target, config=config)
    keyframe_indices = plan_chunk_keyframes(
        generated_keyframes,
        num_frames=target.frames,
        layouts=layouts,
    )
    if keyframe_indices:

        def make_generated_keyframes(chunk: Chunk) -> list[ConditioningItem]:
            return generated_keyframe_slots_for_chunk(chunk, keyframe_indices=keyframe_indices)

        make_video_conditionings = VideoConditioningsWithGeneratedKeyframes(
            inner=make_video_conditionings,
            generated_keyframes=make_generated_keyframes,
        )
    logger.info("Plan: %d chunks", len(layouts))

    def _chunks() -> Iterator[Chunk]:
        for layout in layouts:
            pixel_shape = target._replace(frames=layout.pixel_frames)
            video, initial_audio = create_initial_av_latents(
                pixel_shape=pixel_shape,
                device=device,
                dtype=dtype,
                video_scale_factors=video_scale_factors,
            )
            audio = (
                audio_latent_for_layout(audio_latent, layout, target.fps).to(device=device, dtype=dtype)
                if audio_latent is not None
                else initial_audio
            )
            yield Chunk(
                layout=layout,
                video=video,
                audio=audio,
                video_context=context.video,
                audio_context=context.audio,
                make_video_conditionings=make_video_conditionings,
                make_audio_conditionings=make_audio_conditionings,
            )

    return _chunks()
