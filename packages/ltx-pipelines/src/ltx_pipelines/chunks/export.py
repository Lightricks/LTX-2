"""Chunk stream I/O: overlap-aware reference decode, lazy stitch, and ``PipelineOutput`` packing."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Generator, Iterator

import torch
from torch._prims_common import DeviceLikeType

from ltx_core.model.video_vae import TilingConfig
from ltx_core.types import Audio
from ltx_pipelines.chunks.audio import stitch_audio
from ltx_pipelines.utils.media_io import decode_video_by_frame
from ltx_pipelines.utils.media_io.audio_stream import validate_audio_waveform
from ltx_pipelines.utils.media_io.types import PlaybackSegment
from ltx_pipelines.utils.types import PipelineOutput

logger = logging.getLogger(__name__)


class SequentialVideoFrameSource:
    """Decode a video once, in order, reusing overlap frames already yielded.
    ``take(start, count)`` yields frames ``[start, start+count)``. Later windows may
    overlap; kept frames stay until a later ``take`` starts past them. Asking for a
    frame that has already been dropped raises.
    """

    def __init__(self, path: str, device: DeviceLikeType) -> None:
        self._path = path
        self._device = device
        self._gen: Generator[torch.Tensor, None, None] | None = None
        self._next = 0
        self._kept_start = 0
        self._kept: deque[torch.Tensor] = deque()

    def take(self, start: int, count: int) -> Iterator[torch.Tensor]:
        if start < 0 or count < 0:
            raise ValueError(f"start and count must be >= 0, got start={start} count={count}")
        end = start + count
        if start < self._kept_start:
            raise ValueError(f"cannot take frames [{start}:{end}); lookback starts at {self._kept_start}")
        if start > self._next:
            raise ValueError(f"cannot skip from frame {self._next} to {start}; take windows in order")
        while self._kept_start < start:
            self._kept.popleft()
            self._kept_start += 1
        if end > self._next:
            if self._gen is None:
                self._gen = decode_video_by_frame(self._path, self._device, starting_frame=self._next)
            while self._next < end:
                try:
                    self._kept.append(next(self._gen))
                except StopIteration:
                    break
                self._next += 1
        return iter(list(self._kept)[:count])

    def close(self) -> None:
        if self._gen is not None:
            self._gen.close()
            self._gen = None


def pipeline_output_from_chunks(
    chunks: Iterator[PlaybackSegment],
    *,
    num_frames: int,
    tiling_config: TilingConfig | None,
) -> PipelineOutput:
    """Pack decoded chunks into :class:`PipelineOutput` for ``__call__``.
    One chunk keeps that chunk's video/audio identity. Several chunks yield
    each video tensor in order and one concatenated audio track.
    """
    decoded = list(chunks)
    if not decoded:
        raise ValueError("expected a decoded chunk")
    audios: list[Audio] = []
    for chunk in decoded:
        if chunk.audio is None:
            raise ValueError("expected decoded audio")
        validate_audio_waveform(chunk.audio)
        audios.append(chunk.audio)
    if len(decoded) == 1:
        only = decoded[0]
        return PipelineOutput(iter((only.video,)), audios[0], num_frames, tiling_config)
    stitched = stitch_audio(audios)
    assert stitched is not None
    return PipelineOutput((chunk.video for chunk in decoded), stitched, num_frames, tiling_config)
