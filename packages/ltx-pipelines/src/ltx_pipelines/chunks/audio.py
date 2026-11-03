"""Concatenate decoded chunk audio for in-memory ``PipelineOutput`` values."""

from __future__ import annotations

import torch

from ltx_core.types import Audio
from ltx_pipelines.utils.media_io.audio_stream import normalize_audio_waveform


def stitch_audio(parts: list[Audio]) -> Audio | None:
    """Concatenate chunk parts into one ``Audio``, or ``None`` when empty.
    Parts must already be valid stereo. Call only after every part is available
    (e.g. after the video iterator has been consumed).
    """
    if not parts:
        return None
    sample_rate = parts[0].sampling_rate
    tensors: list[torch.Tensor] = []
    for audio in parts:
        if audio.sampling_rate != sample_rate:
            raise ValueError(f"chunk audio sampling rate {audio.sampling_rate} does not match expected {sample_rate}")
        tensors.append(normalize_audio_waveform(audio.waveform).cpu())
    return Audio(waveform=torch.cat(tensors, dim=0), sampling_rate=sample_rate)
