"""Payload types shared by media encoding stages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ltx_core.types import Audio


@dataclass(frozen=True)
class PlaybackSegment:
    """Contiguous decoded output segment ready for progressive encoding.
    Attributes:
        video: Pixel frames for this segment, shaped ``(F, H, W, 3)``.
        audio: Audio matching this video segment, or ``None`` for video-only output.
    """

    video: torch.Tensor
    audio: Audio | None = None


@dataclass(frozen=True, slots=True)
class H264Frames:
    """Packed H.264 input frames and their PyAV pixel format."""

    frames: np.ndarray
    av_format: str


@dataclass(frozen=True, slots=True)
class HlgPlanes:
    """Batched planar uint16 YUV420P10 frames."""

    y: np.ndarray
    u: np.ndarray
    v: np.ndarray


@dataclass(frozen=True, slots=True)
class ProResPlanes:
    """Planar uint16 YUVA444P10 frames, shaped ``(F, 4, H, W)``."""

    planes: np.ndarray


VideoPayload = H264Frames | HlgPlanes | ProResPlanes
