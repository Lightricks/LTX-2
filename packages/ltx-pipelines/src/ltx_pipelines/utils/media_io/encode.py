"""Incremental SDR H.264, ProRes 4444, and HDR EXR+HLG media encoding."""

from __future__ import annotations

import enum
import logging
from collections.abc import Iterator
from itertools import chain
from pathlib import Path

import torch
from tqdm import tqdm

from ltx_core.color.yuv import FrameConverter, yuv420p_bt709_converter_
from ltx_core.types import Audio
from ltx_pipelines.marking.types import AudioMarker, VideoMarker
from ltx_pipelines.utils.media_io.audio_stream import AudioStreamSpec, validate_audio_waveform
from ltx_pipelines.utils.media_io.color_config import EXRColorSpace
from ltx_pipelines.utils.media_io.exr import exr_dir_label
from ltx_pipelines.utils.media_io.exr_writer import ExrSequenceWriter
from ltx_pipelines.utils.media_io.threaded_writer import ThreadedContainerWriter
from ltx_pipelines.utils.media_io.types import PlaybackSegment
from ltx_pipelines.utils.media_io.video_processor import (
    HlgChunkProcessor,
    ProResChunkProcessor,
    SdrChunkProcessor,
    VideoSegmentProcessor,
)
from ltx_pipelines.utils.media_io.video_stream import (
    H264EncodeConfig,
    HlgEncodeConfig,
    ProRes4444EncodeConfig,
    VideoEncodeConfig,
)
from ltx_pipelines.utils.media_io.writer import ContainerWriter

logger = logging.getLogger(__name__)

_SDR_CRF = 19
_SDR_PRESET = "veryfast"
_HLG_CRF = 12
_HLG_PRESET = "ultrafast"


class VideoCodec(enum.Enum):
    """SDR video stream written by :func:`encode_video`."""

    H264 = "h264"
    """H.264 yuv420p from ``(F, H, W, 3)`` RGB."""
    PRORES_4444 = "prores_4444"
    """ProRes 4444 10-bit yuva444p, BT.709, from ``(F, H, W, 4)`` straight RGBA. Use a ``.mov`` path."""


def encode_media_segments(  # noqa: PLR0912, PLR0913
    segments: Iterator[PlaybackSegment],
    *,
    fps: float,
    output_path: str | Path,
    video_chunks_number: int | None,
    frame_converter: FrameConverter = yuv420p_bt709_converter_,
    crf: int | None = None,
    preset: str | None = None,
    thread_count: int = 0,
    color_space: EXRColorSpace | None = None,
    device: torch.device | None = None,
    video_marker: VideoMarker | None = None,
    audio_marker: AudioMarker | None = None,
    chunks_complete: bool = False,
    codec: VideoCodec = VideoCodec.H264,
) -> Path | None:
    """Encode one ordered stream of decoded video/audio playback segments.
    Each segment is validated before conversion and queued in video-then-audio
    order. The PyAV container is owned by one writer thread.
    Args:
        segments: Ordered playback segments. The first segment determines
            dimensions and whether the output has an audio stream.
        fps: Output video frame rate.
        output_path: Destination MP4 path.
        video_chunks_number: Expected segment count for progress reporting, or
            ``None`` when a streaming source does not know it upfront.
        frame_converter: Producer-side SDR frame conversion.
        crf: Codec quality. Defaults to 19 for H.264 and 12 for HLG.
        preset: Codec preset. Defaults to ``veryfast`` for H.264 and
            ``ultrafast`` for HLG.
        thread_count: Codec thread count, or 0 for codec-specific automatic
            selection.
        color_space: When set, write HLG video and an EXR sidecar directory.
        device: HLG conversion device. Ignored for SDR.
        video_marker: Optional pixel marker. HDR output does not support one.
            ProRes stamps RGB only; alpha passes through.
        audio_marker: Optional marker applied to each audio segment.
        chunks_complete: When true, each CUDA segment is already finished on
            every stream when yielded (e.g. distributed decode batches). SDR
            conversion and host copies then run on a dedicated encoder stream.
        codec: SDR video stream. :attr:`VideoCodec.PRORES_4444` reads
            ``(F, H, W, 4)`` straight RGBA and cannot be combined with ``color_space``.
    Returns:
        The published EXR directory for HDR, otherwise ``None``.
    Raises:
        ValueError: If segments are empty or violate media stream invariants.
    """
    if video_marker is not None and color_space is not None:
        raise ValueError("video_marker is not supported on HDR encode")
    if codec is not VideoCodec.H264 and color_space is not None:
        raise ValueError(f"HDR encode writes its own streams; codec {codec.value} is SDR only")

    first = next(segments, None)
    if first is None:
        raise ValueError("media stream is empty; expected at least one playback segment.")
    channels = 4 if codec is VideoCodec.PRORES_4444 else 3
    height, width = _validate_video_frames(first.video, channels=channels)
    audio_sample_rate = _validate_first_audio(first.audio)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    exr_destination = (
        None if color_space is None else destination.parent / f"{destination.stem}_{exr_dir_label(color_space)}_exr"
    )

    video_config: VideoEncodeConfig
    exr_writer: ExrSequenceWriter | None = None
    if color_space is None and codec is VideoCodec.PRORES_4444:
        processor: VideoSegmentProcessor = ProResChunkProcessor(
            video_marker=video_marker,
            chunks_complete=chunks_complete,
        )
        video_config = ProRes4444EncodeConfig(fps=fps)
    elif color_space is None:
        processor = SdrChunkProcessor(
            frame_converter=frame_converter,
            video_marker=video_marker,
            chunks_complete=chunks_complete,
        )
        video_config = H264EncodeConfig(
            fps=fps,
            crf=crf if crf is not None else _SDR_CRF,
            preset=preset if preset is not None else _SDR_PRESET,
            thread_count=thread_count,
            color_space=frame_converter.color_space,
            color_range=frame_converter.color_range,
        )
    else:
        processor = HlgChunkProcessor(device=device)
        exr_writer = ExrSequenceWriter(exr_destination)
        video_config = HlgEncodeConfig(
            fps=fps,
            crf=crf if crf is not None else _HLG_CRF,
            preset=preset if preset is not None else _HLG_PRESET,
            thread_count=thread_count,
            zero_latency=audio_sample_rate is not None,
        )

    writer = ThreadedContainerWriter.open(
        destination,
        video_stream_spec=video_config.to_spec(width=width, height=height),
        audio_stream_spec=AudioStreamSpec(sample_rate=audio_sample_rate) if audio_sample_rate is not None else None,
    )
    ordered_segments = chain((first,), segments)
    try:
        for segment in tqdm(
            ordered_segments,
            total=video_chunks_number,
            desc=processor.progress_description,
        ):
            marked_segment = PlaybackSegment(
                video=segment.video,
                audio=audio_marker.embed(segment.audio)
                if audio_marker is not None and segment.audio is not None
                else segment.audio,
            )
            _validate_playback_segment(
                marked_segment,
                height=height,
                width=width,
                audio_sample_rate=audio_sample_rate,
                channels=channels,
            )
            for payload in processor.process(marked_segment.video):
                writer.write_video(payload)
            if marked_segment.audio is not None:
                writer.write_audio(marked_segment.audio)
            if exr_writer is not None and color_space is not None:
                exr_writer.write(marked_segment.video, color_space)

        writer.finish()
        if exr_writer is not None:
            exr_writer.finish()
    except BaseException:
        writer.abort()
        if exr_writer is not None:
            exr_writer.abort()
        destination.unlink(missing_ok=True)
        raise

    logger.info("Video saved to %s", destination)
    if exr_destination is not None:
        logger.info("EXR frames saved to %s", exr_destination)
    return exr_destination


def encode_video(  # noqa: PLR0913
    video: torch.Tensor | Iterator[torch.Tensor],
    fps: float,
    audio: Audio | None,
    output_path: str | Path,
    video_chunks_number: int | None,
    frame_converter: FrameConverter = yuv420p_bt709_converter_,
    crf: int | None = None,
    preset: str | None = None,
    thread_count: int = 0,
    *,
    color_space: EXRColorSpace | None = None,
    device: torch.device | None = None,
    video_marker: VideoMarker | None = None,
    audio_marker: AudioMarker | None = None,
    chunks_complete: bool = False,
    codec: VideoCodec = VideoCodec.H264,
) -> Path | None:
    """Encode decoded video batches with an optional complete audio track.
    Video tensors contain float RGB values in ``[0, 1]`` with shape
    ``(frames, height, width, 3)``. Every batch must be non-empty and have the
    same positive, even dimensions. ``audio`` is one complete track beginning
    at time zero; use :func:`encode_media_segments` for segment-aligned audio.
    Without ``color_space``, the output is an SDR H.264 MP4 prepared by
    ``frame_converter``. With ``color_space``, the output is a BT.2020/HLG MP4
    plus a transactionally published ``<output_stem>_<color_space>_exr``
    sidecar directory.
    Args:
        video: One video tensor or an ordered iterator of video tensors.
        fps: Output frame rate.
        audio: Complete stereo audio track shaped ``(2, samples)`` or
            ``(samples, 2)``, or ``None`` for video-only output.
        output_path: Destination MP4 path.
        video_chunks_number: Expected number of video batches for progress
            reporting, or ``None`` when unknown.
        frame_converter: SDR pixel conversion and stream color metadata.
            Ignored for HDR output.
        crf: Codec quality. Defaults to 19 for SDR and 12 for HLG.
        preset: Codec preset. Defaults to ``veryfast`` for SDR and
            ``ultrafast`` for HLG.
        thread_count: Codec thread count, or 0 for codec-specific automatic
            selection.
        color_space: EXR sidecar color space. When set, enables HDR output.
        device: Device used for HDR conversion. Ignored for SDR output.
        video_marker: Optional pixel marker. HDR output does not support one.
        audio_marker: Optional marker applied to the audio track.
        chunks_complete: Each CUDA batch ``video`` yields is already finished on
            every stream (e.g. the distributed decoder's batches). SDR
            conversion and host copies then run on a dedicated encoder stream.
        codec: SDR stream. :attr:`VideoCodec.PRORES_4444` expects straight RGBA
            ``(F, H, W, 4)`` and a ``.mov`` path. It cannot be combined with
            ``color_space``.
    Returns:
        The published EXR sidecar directory for HDR output, otherwise ``None``.
    Raises:
        ValueError: If video or audio violates the documented shape, size, or
            sampling-rate invariants.
    Encoding failures are propagated after incomplete output is cleaned up.
    """
    video_batches = iter((video,)) if isinstance(video, torch.Tensor) else video
    segments = (
        PlaybackSegment(video=video_batch, audio=audio if index == 0 else None)
        for index, video_batch in enumerate(video_batches)
    )

    return encode_media_segments(
        segments,
        fps=fps,
        output_path=output_path,
        video_chunks_number=video_chunks_number,
        frame_converter=frame_converter,
        crf=crf,
        preset=preset,
        thread_count=thread_count,
        color_space=color_space,
        device=device,
        video_marker=video_marker,
        audio_marker=audio_marker,
        chunks_complete=chunks_complete,
        codec=codec,
    )


def encode_audio(
    audio: Audio,
    output_path: str | Path,
    *,
    audio_marker: AudioMarker | None = None,
) -> None:
    """Save stereo audio as a 16-bit PCM WAV file."""
    validate_audio_waveform(audio)
    if audio_marker is not None:
        audio = audio_marker.embed(audio)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer: ContainerWriter | None = None
    try:
        writer = ContainerWriter(
            destination,
            video_stream_spec=None,
            audio_stream_spec=AudioStreamSpec(sample_rate=audio.sampling_rate, codec_name="pcm_s16le"),
        )
        writer.write_audio(audio)
        writer.finish()
    except BaseException:
        if writer is not None:
            writer.abort()
        destination.unlink(missing_ok=True)
        raise
    logger.info("Audio saved to %s", destination)


def _validate_first_audio(audio: Audio | None) -> int | None:
    if audio is None:
        return None
    validate_audio_waveform(audio)
    return audio.sampling_rate


def _validate_playback_segment(
    segment: PlaybackSegment,
    *,
    height: int,
    width: int,
    audio_sample_rate: int | None,
    channels: int,
) -> None:
    segment_height, segment_width = _validate_video_frames(segment.video, channels=channels)
    if (segment_height, segment_width) != (height, width):
        raise ValueError(
            f"video segment size {segment_width}x{segment_height} does not match expected {width}x{height}"
        )
    if segment.audio is None:
        return
    if audio_sample_rate is None:
        raise ValueError("the first playback segment had no audio, but a later segment does")
    validate_audio_waveform(segment.audio)
    if segment.audio.sampling_rate != audio_sample_rate:
        raise ValueError(
            f"audio sampling rate {segment.audio.sampling_rate} does not match expected {audio_sample_rate}"
        )


def _validate_video_frames(frames: torch.Tensor, *, channels: int) -> tuple[int, int]:
    if frames.ndim != 4 or frames.shape[-1] != channels:
        if channels == 4:
            raise ValueError(f"ProRes 4444 expects (F, H, W, 4) RGBA chunks, got shape {tuple(frames.shape)}")
        raise ValueError(f"video frames must have shape (F, H, W, 3); got {tuple(frames.shape)}")
    if frames.shape[0] == 0:
        raise ValueError("video segment has no frames")
    height, width = int(frames.shape[1]), int(frames.shape[2])
    if height <= 0 or width <= 0 or (channels == 3 and (height % 2 != 0 or width % 2 != 0)):
        raise ValueError(f"video frame height and width must be positive and even; got {width}x{height}")
    return height, width
