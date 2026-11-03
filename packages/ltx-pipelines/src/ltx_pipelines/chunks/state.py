"""The chunk wire: layout, latents, contexts, and conditioning factories."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from ltx_core.conditioning import ConditioningItem
from ltx_pipelines.chunks.layout import ChunkLayout


def _empty_conditionings(_chunk: Chunk) -> list[ConditioningItem]:
    return []


MakeVideoConditionings = Callable[["Chunk"], list[ConditioningItem]]
MakeAudioConditionings = Callable[["Chunk"], list[ConditioningItem]]


@dataclass(frozen=True)
class VideoConditioningsWithGeneratedKeyframes:
    """Append planned generated-keyframe slots after pipeline-owned conditionings."""

    inner: MakeVideoConditionings
    generated_keyframes: MakeVideoConditionings

    def __call__(self, chunk: Chunk) -> list[ConditioningItem]:
        return [*self.inner(chunk), *self.generated_keyframes(chunk)]

    def replace_inner(self, inner: MakeVideoConditionings) -> VideoConditioningsWithGeneratedKeyframes:
        return VideoConditioningsWithGeneratedKeyframes(inner=inner, generated_keyframes=self.generated_keyframes)


@dataclass(frozen=True)
class GeneratedKeyframes:
    """Extracted slot planes after denoise, with their chunk-local pixel indices.
    Attributes:
        latent: Unpatchified ``(B, C, K, H, W)`` slot latent (same sidecar as
            ``LatentState.generated_keyframes``).
        pixel_frame_indices: Chunk-local pixel-frame index of each plane, parallel to K.
    """

    latent: torch.Tensor
    pixel_frame_indices: tuple[int, ...]


@dataclass(frozen=True)
class Chunk:
    """One chunk in transit between chunk operators.
    Attributes:
        layout: Temporal window this chunk covers on the stitched pixel timeline.
        video: Raw (never patchified) video latent, or ``None`` if this stream is unused.
        audio: Raw audio latent, or ``None`` if this stream is unused.
        video_context: Prompt embeddings for the video denoiser.
        audio_context: Prompt embeddings for the audio denoiser.
        make_video_conditionings: Factory that materializes video
            :class:`~ltx_core.conditioning.ConditioningItem` lists at the current latent
            shape when the chunk reaches denoise. The same callable survives a spatial
            upsample.
        make_audio_conditionings: Same as ``make_video_conditionings`` for audio.
        generated_keyframes: Slot planes extracted after denoise, or ``None`` if this
            chunk did not produce them.
        incoming_keyframes: Planes received from the previous chunk's outgoing overlap,
            rebased onto this chunk's local timeline for fixed conditioning and decode.
    """

    layout: ChunkLayout
    video: torch.Tensor | None
    audio: torch.Tensor | None
    video_context: torch.Tensor | None = None
    audio_context: torch.Tensor | None = None
    make_video_conditionings: MakeVideoConditionings = _empty_conditionings
    make_audio_conditionings: MakeAudioConditionings = _empty_conditionings
    generated_keyframes: GeneratedKeyframes | None = None
    incoming_keyframes: GeneratedKeyframes | None = None
