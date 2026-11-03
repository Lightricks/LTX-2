"""PyAV audio stream spec and stereo waveform validation."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction

import av
import torch

from ltx_core.types import Audio


@dataclass(frozen=True, slots=True)
class AudioStreamSpec:
    """Stereo audio stream at one fixed sample rate (``aac`` for MP4, ``pcm_s16le`` for WAV)."""

    sample_rate: int
    codec_name: str = "aac"

    def open_stream(self, container: av.container.OutputContainer) -> av.AudioStream:
        stream = container.add_stream(self.codec_name, rate=self.sample_rate)
        stream.codec_context.sample_rate = self.sample_rate
        stream.codec_context.layout = "stereo"
        stream.codec_context.time_base = Fraction(1, self.sample_rate)
        return stream

    def frames(self, payload: Audio) -> Iterator[av.AudioFrame]:
        validate_audio_waveform(payload)
        if payload.sampling_rate != self.sample_rate:
            raise ValueError(
                f"audio sampling rate {payload.sampling_rate} does not match stream rate {self.sample_rate}"
            )
        samples = normalize_audio_waveform(payload.waveform)
        if samples.dtype != torch.int16:
            samples = torch.clip(samples, -1.0, 1.0)
            samples = (samples * 32767.0).to(torch.int16)
        frame = av.AudioFrame.from_ndarray(
            samples.contiguous().reshape(1, -1).cpu().numpy(),
            format="s16",
            layout="stereo",
        )
        frame.sample_rate = self.sample_rate
        yield frame


def validate_audio_waveform(audio: Audio) -> None:
    """Validate a non-empty stereo waveform shaped ``(2, N)`` or ``(N, 2)``."""
    samples = audio.waveform
    if samples.numel() == 0:
        raise ValueError("audio.waveform is empty; pass audio=None for no audio.")
    if samples.ndim != 2 or 2 not in samples.shape:
        raise ValueError(f"audio.waveform must be stereo (2, N) or (N, 2); got shape {tuple(samples.shape)}.")


def normalize_audio_waveform(samples: torch.Tensor) -> torch.Tensor:
    """Return validated stereo samples in channel-last ``(N, 2)`` order."""
    return samples.T if samples.shape[1] != 2 else samples
