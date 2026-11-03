"""Single-owner PyAV container writer."""

from __future__ import annotations

import contextlib
from collections import deque
from fractions import Fraction
from pathlib import Path

import av

from ltx_core.types import Audio
from ltx_pipelines.utils.media_io.audio_stream import AudioStreamSpec
from ltx_pipelines.utils.media_io.types import VideoPayload
from ltx_pipelines.utils.media_io.video_stream import VideoStreamSpec


class ContainerWriter:
    """Open one output container and encode, interleave, and mux its streams synchronously.
    Not thread-safe: every method must run on the thread that constructed the writer.
    """

    def __init__(
        self,
        output_path: Path,
        *,
        video_stream_spec: VideoStreamSpec | None,
        audio_stream_spec: AudioStreamSpec | None,
    ) -> None:
        if video_stream_spec is None and audio_stream_spec is None:
            raise ValueError("a container needs at least one stream")
        options = {} if video_stream_spec is None else video_stream_spec.container_options
        self._container = av.open(str(output_path), mode="w", options=options)
        try:
            self._video_stream_spec = video_stream_spec
            self._audio_stream_spec = audio_stream_spec
            self._video_stream = None if video_stream_spec is None else video_stream_spec.open_stream(self._container)
            self._audio_stream = None if audio_stream_spec is None else audio_stream_spec.open_stream(self._container)
            self._resampler = (
                None
                if audio_stream_spec is None or self._audio_stream is None
                else _audio_resampler(self._audio_stream, audio_stream_spec)
            )
        except BaseException:
            self.abort()
            raise
        self._interleaver = (
            _PacketInterleaver() if video_stream_spec is not None and audio_stream_spec is not None else None
        )
        self._video_time_base = (
            None
            if video_stream_spec is None
            else Fraction(video_stream_spec.frame_rate.denominator, video_stream_spec.frame_rate.numerator)
        )
        self._audio_time_base = None if audio_stream_spec is None else Fraction(1, audio_stream_spec.sample_rate)
        self._video_pts = 0
        self._audio_input_pts = 0
        self._audio_output_pts = 0
        self._finished = False

    def write_video(self, payload: VideoPayload) -> None:
        """Encode one video payload and mux every packet the codec emits."""
        self._check_open()
        if self._video_stream_spec is None or self._video_stream is None:
            raise ValueError("received video without a configured video stream")
        for frame in self._video_stream_spec.frames(payload):
            frame.pts = self._video_pts
            frame.time_base = self._video_time_base
            self._video_pts += 1
            for packet in self._video_stream.encode(frame):
                self._mux_video(packet)

    def write_audio(self, audio: Audio) -> None:
        """Resample and encode one contiguous audio segment without draining."""
        self._check_open()
        if self._audio_stream_spec is None or self._resampler is None:
            raise ValueError("received audio without a configured audio stream")
        for frame in self._audio_stream_spec.frames(audio):
            frame.time_base = self._audio_time_base
            frame.pts = self._audio_input_pts
            self._audio_input_pts += frame.samples
            self._encode_resampled(self._resampler.resample(frame))

    def finish(self) -> None:
        """Drain video, release held audio, drain resampler and audio codec, write trailer, close."""
        self._check_open()
        self._finished = True
        if self._video_stream is not None:
            for packet in self._video_stream.encode():
                self._mux_video(packet)
        if self._interleaver is not None:
            self._mux_all(self._interleaver.finish_video())
        if self._audio_stream is not None and self._resampler is not None:
            self._encode_resampled(self._resampler.resample(None))
            for packet in self._audio_stream.encode():
                self._mux_audio(packet)
        self._container.close()

    def abort(self) -> None:
        """Close the container without draining codecs. Never raises."""
        self._finished = True
        with contextlib.suppress(Exception):
            self._container.close()

    def _check_open(self) -> None:
        if self._finished:
            raise RuntimeError("container writer is already finished")

    def _encode_resampled(self, frames: list[av.AudioFrame]) -> None:
        for frame in frames:
            frame.sample_rate = self._audio_stream_spec.sample_rate
            frame.time_base = self._audio_time_base
            frame.pts = self._audio_output_pts
            self._audio_output_pts += frame.samples
            for packet in self._audio_stream.encode(frame):
                self._mux_audio(packet)

    def _mux_video(self, packet: av.Packet) -> None:
        if self._interleaver is None:
            self._container.mux(packet)
        else:
            self._mux_all(self._interleaver.video(packet))

    def _mux_audio(self, packet: av.Packet) -> None:
        if self._interleaver is None:
            self._container.mux(packet)
        else:
            self._mux_all(self._interleaver.audio(packet))

    def _mux_all(self, packets: list[av.Packet]) -> None:
        for packet in packets:
            self._container.mux(packet)


def _audio_resampler(stream: av.AudioStream, spec: AudioStreamSpec) -> av.audio.resampler.AudioResampler:
    codec = stream.codec_context
    return av.audio.resampler.AudioResampler(
        format=codec.format or ("s16" if spec.codec_name == "pcm_s16le" else "fltp"),
        layout=codec.layout or "stereo",
        rate=spec.sample_rate,
    )


def _packet_dts_seconds(packet: av.Packet) -> Fraction | None:
    if packet.dts is None or packet.time_base is None:
        return None
    return Fraction(packet.dts) * Fraction(packet.time_base)


class _PacketInterleaver:
    """Hold audio packets until video DTS proves they cannot overtake video.
    Pure ordering filter: each call returns the packets the caller must mux now, in order.
    """

    def __init__(self) -> None:
        self._pending_audio: deque[av.Packet] = deque()
        self._latest_video_dts: Fraction | None = None
        self._video_finished = False

    def video(self, packet: av.Packet) -> list[av.Packet]:
        """Return released audio packets followed by ``packet``."""
        released: list[av.Packet] = []
        packet_dts = _packet_dts_seconds(packet)
        if packet_dts is not None:
            self._latest_video_dts = packet_dts
            while self._pending_audio and self._audio_packet_is_safe(self._pending_audio[0]):
                released.append(self._pending_audio.popleft())
        released.append(packet)
        return released

    def audio(self, packet: av.Packet) -> list[av.Packet]:
        """Return ``[packet]`` if it is safe to mux now, otherwise hold it and return ``[]``."""
        if self._video_finished or self._audio_packet_is_safe(packet):
            return [packet]
        self._pending_audio.append(packet)
        return []

    def finish_video(self) -> list[av.Packet]:
        """Release every held audio packet once video has emitted its delayed packets."""
        self._video_finished = True
        released = list(self._pending_audio)
        self._pending_audio.clear()
        return released

    def _audio_packet_is_safe(self, packet: av.Packet) -> bool:
        packet_dts = _packet_dts_seconds(packet)
        return packet_dts is not None and self._latest_video_dts is not None and packet_dts <= self._latest_video_dts
