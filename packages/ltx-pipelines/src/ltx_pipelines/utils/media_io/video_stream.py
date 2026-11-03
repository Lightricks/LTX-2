"""PyAV video stream specs, encode settings, and frame packaging."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol

import av
import numpy as np

from ltx_core.color.yuv import ColorRange, ColorSpace
from ltx_pipelines.utils.media_io.types import H264Frames, HlgPlanes, ProResPlanes, VideoPayload

_AV_COLOR_PRIMARIES_BT2020 = 9
_AV_COLOR_PRIMARIES_BT709 = 1
_AV_COLOR_TRC_BT709 = 1
_AV_COLOR_TRC_ARIB_STD_B67 = 18
_PRORES_4444_PIX_FMT = "yuva444p10le"


def _frame_rate(fps: float) -> Fraction:
    return Fraction(fps).limit_denominator(1000)


class VideoStreamSpec(Protocol):
    """Video stream description with the timing and container flags the writer needs."""

    def open_stream(self, container: av.container.OutputContainer) -> av.VideoStream:
        """Add and configure a video stream in ``container``."""
        ...

    def frames(self, payload: VideoPayload) -> Iterator[av.VideoFrame]:
        """Convert one payload into uncompressed video frames."""
        ...

    @property
    def frame_rate(self) -> Fraction:
        """Exact stream frame rate."""
        ...

    @property
    def container_options(self) -> dict[str, str]:
        """Options passed to ``av.open`` for containers holding this stream."""
        ...


class VideoEncodeConfig(Protocol):
    """Video encode settings that can build a :class:`VideoStreamSpec`."""

    def to_spec(self, *, width: int, height: int) -> VideoStreamSpec:
        """Return the stream spec for ``width`` x ``height`` output."""
        ...


@dataclass(frozen=True, slots=True)
class H264EncodeConfig:
    """H.264 stream settings."""

    fps: float
    crf: int = 19
    preset: str = "veryfast"
    thread_count: int = 0
    color_space: ColorSpace | None = ColorSpace.BT_709
    color_range: ColorRange | None = ColorRange.MPEG

    def to_spec(self, *, width: int, height: int) -> VideoStreamSpec:
        return H264StreamSpec(width=width, height=height, config=self)


@dataclass(frozen=True, slots=True)
class ProRes4444EncodeConfig:
    """ProRes 4444 stream settings. Colour is BT.709 limited-range 10-bit; alpha is full range."""

    fps: float

    def to_spec(self, *, width: int, height: int) -> VideoStreamSpec:
        return ProRes4444StreamSpec(width=width, height=height, config=self)


@dataclass(frozen=True, slots=True)
class HlgEncodeConfig:
    """BT.2020/HLG Main10 stream settings."""

    fps: float
    crf: int = 12
    preset: str = "ultrafast"
    thread_count: int = 0
    zero_latency: bool = False

    def to_spec(self, *, width: int, height: int) -> VideoStreamSpec:
        return HlgStreamSpec(width=width, height=height, config=self)


@dataclass(frozen=True, slots=True)
class H264StreamSpec:
    """libx264 stream fed with packed ``H264Frames``."""

    width: int
    height: int
    config: H264EncodeConfig

    @property
    def frame_rate(self) -> Fraction:
        return _frame_rate(self.config.fps)

    @property
    def container_options(self) -> dict[str, str]:
        return {}

    def open_stream(self, container: av.container.OutputContainer) -> av.VideoStream:
        stream = container.add_stream(
            "libx264",
            rate=self.frame_rate,
            options={"crf": str(self.config.crf), "preset": self.config.preset},
        )
        stream.width = self.width
        stream.height = self.height
        stream.pix_fmt = "yuv420p"
        stream.codec_context.thread_count = self.config.thread_count
        stream.codec_context.thread_type = "FRAME"
        if self.config.color_space is not None:
            stream.codec_context.colorspace = self.config.color_space.av_colorspace
        if self.config.color_range is not None:
            stream.codec_context.color_range = self.config.color_range.av_color_range
        return stream

    def frames(self, payload: VideoPayload) -> Iterator[av.VideoFrame]:
        if not isinstance(payload, H264Frames):
            raise TypeError(f"H264StreamSpec expected H264Frames, got {type(payload).__name__}")
        for frame_array in payload.frames:
            yield av.VideoFrame.from_ndarray(frame_array, format=payload.av_format)


@dataclass(frozen=True, slots=True)
class ProRes4444StreamSpec:
    """prores_ks 4444 stream fed with planar ``ProResPlanes``."""

    width: int
    height: int
    config: ProRes4444EncodeConfig

    @property
    def frame_rate(self) -> Fraction:
        return _frame_rate(self.config.fps)

    @property
    def container_options(self) -> dict[str, str]:
        return {}

    def open_stream(self, container: av.container.OutputContainer) -> av.VideoStream:
        stream = container.add_stream("prores_ks", rate=self.frame_rate, options={"profile": "4444", "vendor": "apl0"})
        stream.width = self.width
        stream.height = self.height
        stream.pix_fmt = _PRORES_4444_PIX_FMT
        codec = stream.codec_context
        codec.color_primaries = _AV_COLOR_PRIMARIES_BT709
        codec.color_trc = _AV_COLOR_TRC_BT709
        codec.colorspace = ColorSpace.BT_709.av_colorspace
        codec.color_range = ColorRange.MPEG.av_color_range
        return stream

    def frames(self, payload: VideoPayload) -> Iterator[av.VideoFrame]:
        if not isinstance(payload, ProResPlanes):
            raise TypeError(f"ProRes4444StreamSpec expected ProResPlanes, got {type(payload).__name__}")
        for planes in payload.planes:
            frame = av.VideoFrame(self.width, self.height, _PRORES_4444_PIX_FMT)
            for plane, source in zip(frame.planes, planes, strict=True):
                destination = np.frombuffer(plane, dtype=np.uint16).reshape(plane.height, plane.line_size // 2)
                destination[:, : source.shape[1]] = source
            frame.colorspace = ColorSpace.BT_709.av_colorspace
            frame.color_range = ColorRange.MPEG.av_color_range
            yield frame


@dataclass(frozen=True, slots=True)
class HlgStreamSpec:
    """libx265 BT.2020/HLG Main10 stream fed with planar ``HlgPlanes``."""

    width: int
    height: int
    config: HlgEncodeConfig

    @property
    def frame_rate(self) -> Fraction:
        return _frame_rate(self.config.fps)

    @property
    def container_options(self) -> dict[str, str]:
        return {"movflags": "+faststart"}

    def open_stream(self, container: av.container.OutputContainer) -> av.VideoStream:
        threads = _resolve_x265_thread_count(self.config.thread_count)
        stream = container.add_stream("libx265", rate=self.frame_rate)
        stream.width = self.width
        stream.height = self.height
        stream.pix_fmt = "yuv420p10le"
        stream.codec_tag = "hvc1"
        stream.options = {
            "crf": str(self.config.crf),
            "preset": self.config.preset,
            "x265-params": _x265_encode_params(threads, self.width, self.height, zero_latency=self.config.zero_latency),
        }
        codec = stream.codec_context
        codec.thread_count = threads
        codec.thread_type = "FRAME"
        codec.color_primaries = _AV_COLOR_PRIMARIES_BT2020
        codec.color_trc = _AV_COLOR_TRC_ARIB_STD_B67
        codec.colorspace = ColorSpace.BT_2020_NCL.av_colorspace
        codec.color_range = ColorRange.MPEG.av_color_range
        return stream

    def frames(self, payload: VideoPayload) -> Iterator[av.VideoFrame]:
        if not isinstance(payload, HlgPlanes):
            raise TypeError(f"HlgStreamSpec expected HlgPlanes, got {type(payload).__name__}")
        for frame_index in range(payload.y.shape[0]):
            frame = av.VideoFrame(self.width, self.height, "yuv420p10le")
            sources = (payload.y[frame_index], payload.u[frame_index], payload.v[frame_index])
            for plane, source in zip(frame.planes, sources, strict=True):
                destination = np.frombuffer(plane, dtype=np.uint16).reshape(plane.height, plane.line_size // 2)
                destination[:, : source.shape[1]] = source
            frame.colorspace = ColorSpace.BT_2020_NCL.av_colorspace
            frame.color_range = ColorRange.MPEG.av_color_range
            yield frame


def _resolve_x265_thread_count(thread_count: int) -> int:
    """Clamp libx265 threads because auto can hang on high-CPU hosts."""
    if thread_count > 0:
        return thread_count
    return max(1, min(os.cpu_count() or 8, 16))


def _x265_encode_params(threads: int, width: int, height: int, *, zero_latency: bool = False) -> str:
    """Return stable HLG x265 parameters, including the single-CTU workaround."""
    base = (
        "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:range=limited:"
        f"repeat-headers=1:info=0:pools={threads}"
    )
    if (width <= 32 and height <= 32) or zero_latency:
        return f"{base}:frame-threads=1:bframes=0:lookahead=0"
    return f"{base}:frame-threads=4"
