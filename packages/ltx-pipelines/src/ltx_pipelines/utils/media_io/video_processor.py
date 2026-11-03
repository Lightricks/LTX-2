"""Producer-side preparation of decoded playback-segment video."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

import numpy as np
import torch

from ltx_core.color.hlg import HlgGpuConverter
from ltx_core.color.primaries import Primaries
from ltx_core.color.yuv import ColorRange, ColorSpace, FrameConverter, rgba_to_yuva444p10
from ltx_pipelines.marking.types import VideoMarker
from ltx_pipelines.utils.media_io.color_config import decode_hdr_video
from ltx_pipelines.utils.media_io.types import H264Frames, HlgPlanes, ProResPlanes, VideoPayload

_HOST_SLICE_FRAMES = 8

# One conversion stream per device for the life of the process. cuBLAS keeps a workspace per
# (handle, stream) until exit, so a fresh stream per encode pins another workspace on the allocator.
_CONVERT_STREAMS: dict[torch.device, torch.cuda.Stream] = {}


class VideoSegmentProcessor(Protocol):
    """Prepare decoded playback-segment video for one stream encoding format."""

    progress_description: str

    def process(self, video: torch.Tensor) -> Iterator[VideoPayload]:
        """Yield prepared payloads for one decoded playback segment."""


class SdrChunkProcessor:
    """Prepare decoded SDR frames for an H.264 stream."""

    progress_description = "H.264 encode"

    def __init__(
        self,
        *,
        frame_converter: FrameConverter,
        video_marker: VideoMarker | None = None,
        chunks_complete: bool = False,
    ) -> None:
        self._frame_converter = frame_converter
        self._video_marker = video_marker
        self._chunks_complete = chunks_complete

    def process(self, video: torch.Tensor) -> Iterator[VideoPayload]:
        for piece in video.split(_HOST_SLICE_FRAMES):
            yield self._prepare_piece(piece)

    def _prepare_piece(self, piece: torch.Tensor) -> H264Frames:
        if self._chunks_complete and piece.is_cuda:
            with torch.cuda.stream(_convert_stream(piece.device)):
                return self._convert_and_copy(piece)
        return self._convert_and_copy(piece)

    def _convert_and_copy(self, piece: torch.Tensor) -> H264Frames:
        if self._video_marker is not None:
            piece = self._video_marker.embed(piece)
        moved = piece.movedim(-1, -3)
        converted = self._frame_converter(moved)
        return H264Frames(_to_host(converted), self._frame_converter.pixel_format.av_format)


class ProResChunkProcessor:
    """Prepare straight-alpha RGBA frames for a ProRes 4444 stream."""

    progress_description = "ProRes 4444 encode"

    def __init__(
        self,
        *,
        video_marker: VideoMarker | None = None,
        chunks_complete: bool = False,
    ) -> None:
        self._video_marker = video_marker
        self._chunks_complete = chunks_complete

    def process(self, video: torch.Tensor) -> Iterator[VideoPayload]:
        for piece in video.split(_HOST_SLICE_FRAMES):
            yield self._prepare_piece(piece)

    def _prepare_piece(self, piece: torch.Tensor) -> ProResPlanes:
        if self._chunks_complete and piece.is_cuda:
            with torch.cuda.stream(_convert_stream(piece.device)):
                return self._convert_and_copy(piece)
        return self._convert_and_copy(piece)

    def _convert_and_copy(self, piece: torch.Tensor) -> ProResPlanes:
        if piece.shape[-1] != 4:
            raise ValueError(f"ProRes 4444 expects (F, H, W, 4) RGBA chunks, got shape {tuple(piece.shape)}")
        if self._video_marker is not None:
            rgb = self._video_marker.embed(piece[..., :3].contiguous())
            piece = torch.cat([rgb, piece[..., 3:]], dim=-1)
        planes = rgba_to_yuva444p10(piece.movedim(-1, -3), ColorSpace.BT_709, ColorRange.MPEG)
        return ProResPlanes(_to_host(planes))


class HlgChunkProcessor:
    """Convert decoded ACEScct HDR frames into HLG stream planes."""

    progress_description = "HLG encode"

    def __init__(self, *, device: torch.device | None) -> None:
        self._converter = HlgGpuConverter.create(Primaries.REC709, device=device)

    def process(self, video: torch.Tensor) -> Iterator[VideoPayload]:
        rec709_linear = decode_hdr_video(video)
        y, u, v = self._converter(rec709_linear)
        yield HlgPlanes(y.cpu().numpy(), u.cpu().numpy(), v.cpu().numpy())


def _to_host(frames: torch.Tensor) -> np.ndarray:
    """Copy converted frames to the host through a pinned buffer."""
    if frames.device.type != "cuda":
        return frames.to("cpu").numpy()
    host = torch.empty(frames.shape, dtype=frames.dtype, pin_memory=True)
    host.copy_(frames)
    return host.numpy()


def _convert_stream(device: torch.device) -> torch.cuda.Stream:
    stream = _CONVERT_STREAMS.get(device)
    if stream is None:
        stream = _CONVERT_STREAMS[device] = torch.cuda.Stream(device=device)
    return stream
