"""Image/video/audio decode and SDR conditioning preprocess."""

from __future__ import annotations

import logging
from collections.abc import Generator, Iterator
from concurrent.futures import Future
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import av
import numpy as np
import OpenImageIO
import torch
from PIL import ExifTags, Image, ImageCms, UnidentifiedImageError
from torch._prims_common import DeviceLikeType

from ltx_core.types import Audio, VideoPixelShape
from ltx_pipelines.utils.media_io.color_config import EXRColorSpace, srgb_to_acescct, to_working_space
from ltx_pipelines.utils.media_io.exr import (
    _exr_paths_for_conditioning,
    is_exr_dir,
    read_exr,
)
from ltx_pipelines.utils.media_io.range_map import normalize_images, to_vae_range
from ltx_pipelines.utils.media_io.resize import (
    ResizeMode,
    resize_and_center_crop,
    resize_and_reflect_pad,
)

if TYPE_CHECKING:
    from ltx_pipelines.utils.types import DecodedStill

logger = logging.getLogger(__name__)

_ORIENTATION_EXIF_KEY = next(key for key, value in ExifTags.TAGS.items() if value == "Orientation")

_ORIENTATION_TO_ROTATION = {3: 180, 6: 270, 8: 90}

_SRGB_PROFILE = ImageCms.createProfile("sRGB")

_INT_FORMAT_MAX: dict[str, float] = {
    "u8": 128.0,
    "u8p": 128.0,
    "s16": 32768.0,
    "s16p": 32768.0,
    "s32": 2147483648.0,
    "s32p": 2147483648.0,
}


_EXR_NEEDS_COLOR_SPACE = (
    "EXR input requires --hdr {SRGB_LINEAR,ACESCG,ACESCCT,ACES2065_1} to declare the source colour space."
)


def load_image_and_preprocess(
    image_path: str,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    crf: int,
    color_space: EXRColorSpace | None = None,
) -> torch.Tensor:
    """
    Loads an image from a path and preprocesses it for conditioning.
    Note: The image is resized to the nearest multiple of 2 for compatibility with video codecs.
    An ``.exr`` still requires ``color_space`` (from ``--hdr``) and is loaded via the
    HDR conditioning path instead of the SDR decode/normalize path.
    """
    return load_conditioning_still(image_path, crf, color_space).tensor(height, width, dtype, device)


def load_conditioning_still(image_path: str, crf: int, color_space: EXRColorSpace | None = None) -> DecodedStill:
    """Decode an image conditioning's still once, for every resolution a run encodes it at.
    An SDR image is decoded and re-compressed at ``crf``; an ``.exr`` still is read as linear HDR and
    needs ``color_space`` (from ``--hdr``). A missing or undecodable file of either kind raises
    ``ValueError("Cannot decode image file '<path>'.")``.
    """
    if str(image_path).lower().endswith(".exr"):
        if color_space is None:
            raise ValueError(_EXR_NEEDS_COLOR_SPACE)
        try:
            pixels = read_exr(_exr_paths_for_conditioning(image_path)[0])
        except (OSError, RuntimeError, ValueError) as err:
            raise ValueError(f"Cannot decode image file '{image_path}'.") from err
        return ExrStill(pixels, color_space)
    image = preprocess(image=decode_image(image_path=image_path), crf=crf)
    image.setflags(write=False)
    return SdrStill(image)


def check_conditioning_still(image_path: str, color_space: EXRColorSpace | None = None) -> None:
    """Raise what :func:`load_conditioning_still` would for a missing file or an EXR without ``color_space``.
    Cheap enough for a pipeline's prologue when the decode itself runs elsewhere.
    """
    if str(image_path).lower().endswith(".exr") and color_space is None:
        raise ValueError(_EXR_NEEDS_COLOR_SPACE)
    if not Path(image_path).exists():
        raise ValueError(f"Cannot decode image file '{image_path}'.")


@dataclass(frozen=True, eq=False)
class BackgroundStill:
    """A still decoding on a worker thread; :meth:`tensor` waits for it and re-raises its decode error."""

    future: Future[DecodedStill]

    def tensor(self, height: int, width: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return self.future.result().tensor(height, width, dtype, device)


@dataclass(frozen=True, eq=False)
class SdrStill:
    """A decoded SDR still, re-compressed at its conditioning CRF: read-only ``(H, W, 3)`` uint8."""

    pixels: np.ndarray

    def tensor(self, height: int, width: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        image = torch.tensor(self.pixels, dtype=torch.float32, device=device)
        image = resize_and_center_crop(image, height, width)
        return normalize_images(image, device, dtype)


@dataclass(frozen=True, eq=False)
class ExrStill:
    """A linear HDR still as read from EXR, ``(H, W, 3)`` float, mapped to the VAE working space per resolution."""

    pixels: torch.Tensor
    color_space: EXRColorSpace

    def tensor(self, height: int, width: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        frame = resize_and_center_crop(self.pixels.to(device=device, dtype=torch.float32), height, width)
        return to_vae_range(to_working_space(frame, self.color_space)).to(device=device, dtype=dtype)


def video_preprocess(
    frames: Generator[torch.Tensor],
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Preprocesses a video frame generator for conditioning.
    Args:
        frames: Generator of video frames as tensors of shape (1, H, W, C), dtype uint8.
        height: Target height in pixels.
        width: Target width in pixels.
        dtype: Target dtype for the output tensor.
        device: Target device for the output tensor.
    Returns:
        Tensor of shape (1, C, F, height, width) with values in [-1, 1].
    """
    # Collect frames and concatenate once: per-frame cat reserves O(F^2) CUDA
    # cache blocks when expandable segments are unavailable.
    processed: list[torch.Tensor] = []
    for f in frames:
        frame = resize_and_center_crop(f.to(torch.float32), height, width)
        processed.append(normalize_images(frame, device, dtype))
    if not processed:
        raise ValueError("video_preprocess received an empty frame generator; no frames were decoded from the source.")
    return torch.cat(processed, dim=2)


def load_video_as_hdr_conditioning(
    video_path: str | Path,
    height: int,
    width: int,
    frame_cap: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    gamma_encoded: bool = False,
    frame_start: int = 0,
    resize_mode: ResizeMode = ResizeMode.CENTER_CROP,
) -> Iterator[torch.Tensor]:
    """Load an MP4/MOV and yield ACEScct VAE-range frames ``(1, C, 1, height, width)``."""
    resize_fn = resize_and_reflect_pad if resize_mode is ResizeMode.REFLECT_PAD else resize_and_center_crop
    for f in decode_video_by_frame(
        path=str(video_path), frame_cap=frame_cap, device=device, starting_frame=frame_start
    ):
        ldr = (f[0].float() / 255.0).clamp(0.0, 1.0)
        native = ldr.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        working = srgb_to_acescct(native, gamma_encoded=gamma_encoded)
        working_hwc = working[0, :, 0].permute(1, 2, 0).contiguous()
        frame = resize_fn(working_hwc, height, width)
        yield to_vae_range(frame).to(device=device, dtype=dtype)


def decode_image(image_path: str) -> np.ndarray:
    """Load an image as an oriented, sRGB, three-channel uint8 RGB array."""
    try:
        with Image.open(image_path) as source_image:
            image = source_image
            orientation = image.getexif().get(_ORIENTATION_EXIF_KEY)
            if orientation in _ORIENTATION_TO_ROTATION:
                image = image.rotate(_ORIENTATION_TO_ROTATION[orientation], expand=True)

            icc_profile = image.info.get("icc_profile")

            if image.mode == "RGBA":
                image = image.convert("RGB")
            elif image.mode == "LA":
                image = image.convert("L")

            if icc_profile:
                try:
                    source_profile = ImageCms.ImageCmsProfile(BytesIO(icc_profile))
                    destination_profile = ImageCms.ImageCmsProfile(_SRGB_PROFILE)
                    image = ImageCms.profileToProfile(
                        image,
                        source_profile,
                        destination_profile,
                        outputMode="RGB",
                    )
                except (ImageCms.PyCMSError, OSError, ValueError) as error:
                    logger.warning("Failed to convert image to sRGB: %s", error)
                    image = image.convert("RGB")
            else:
                image = image.convert("RGB")

            return np.array(image, dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as err:
        raise ValueError(f"Cannot decode image file '{image_path}'.") from err


def _audio_frame_to_float(frame: av.AudioFrame) -> np.ndarray:
    """Convert an audio frame to a float32 ndarray with values in [-1, 1] and shape (channels, samples)."""
    fmt = frame.format.name
    arr = frame.to_ndarray().astype(np.float32)
    if fmt in _INT_FORMAT_MAX:
        arr = arr / _INT_FORMAT_MAX[fmt]
    if not frame.format.is_planar:
        # Interleaved formats have shape (1, samples * channels) — reshape to (channels, samples).
        channels = len(frame.layout.channels)
        arr = arr.reshape(-1, channels).T
    return arr


def get_videostream_fps(path: str) -> float:
    """Read video stream FPS.
    EXR-frame folders have no embedded frame rate — callers must supply fps via
    :func:`get_videostream_metadata` instead of calling this helper.
    """
    if is_exr_dir(path):
        raise ValueError(
            f"Path '{path}' is an EXR-frame folder and has no embedded FPS; pass fps explicitly (e.g. --frame-rate)."
        )
    container = av.open(path)
    try:
        video_stream = next(s for s in container.streams if s.type == "video")
        return float(video_stream.average_rate)
    finally:
        container.close()


def get_videostream_metadata(path: str, fps: float | None = None) -> VideoPixelShape:
    """Read video stream (or EXR-folder) metadata as a VideoPixelShape with batch=1.
    If frame count is missing in the container, decodes the stream to count frames.
    Args:
        path: Path to a video file, or a directory containing ``*.exr`` frames.
        fps: Frame rate. Required for EXR-frame folders (no container metadata).
            Ignored for regular video files (taken from the stream).
    Returns:
        VideoPixelShape with batch=1, frames, height, width, and fps populated.
    """
    if is_exr_dir(path):
        if fps is None:
            raise ValueError(f"Path '{path}' is an EXR-frame folder; fps must be provided (e.g. via --frame-rate).")
        exr_files = sorted(Path(path).glob("*.exr"))
        spec = OpenImageIO.ImageBuf(str(exr_files[0])).spec()
        return VideoPixelShape(
            batch=1,
            frames=len(exr_files),
            height=int(spec.height),
            width=int(spec.width),
            fps=float(fps),
        )

    container = av.open(path)
    try:
        video_stream = next(s for s in container.streams if s.type == "video")
        stream_fps = float(video_stream.average_rate)
        num_frames = video_stream.frames or 0
        if num_frames == 0:
            num_frames = sum(1 for _ in container.decode(video_stream))
        width = video_stream.codec_context.width
        height = video_stream.codec_context.height
        return VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=stream_fps)
    finally:
        container.close()


def decode_audio_from_file(
    path: str, device: torch.device, start_time: float = 0.0, max_duration: float | None = None
) -> Audio | None:
    """Decodes audio from a file, optionally seeking to a start time and limiting duration.
    Args:
        path: Path to the audio/video file containing an audio stream.
        device: Device to place the resulting tensor on.
        start_time: Start time in seconds to begin reading audio from.
        max_duration: Maximum audio duration in seconds. If None, reads to end of stream.
    Returns:
        An Audio object with waveform of shape (1, channels, samples), or None if no audio stream.
    """
    container = av.open(path)
    try:
        audio_stream = next(s for s in container.streams if s.type == "audio")
    except StopIteration:
        container.close()
        return None

    sample_rate = audio_stream.rate
    start_pts = int(start_time / audio_stream.time_base)
    end_time = start_time + max_duration if max_duration else audio_stream.duration * audio_stream.time_base
    container.seek(start_pts, stream=audio_stream)

    samples = []
    first_frame_time = None
    for frame in container.decode(audio=0):
        if frame.pts is None:
            continue
        frame_time = float(frame.pts * audio_stream.time_base)
        frame_end = frame_time + frame.samples / frame.sample_rate
        if frame_end < start_time:
            continue
        if frame_time > end_time:
            break
        if first_frame_time is None:
            first_frame_time = frame_time
        samples.append(_audio_frame_to_float(frame))

    container.close()

    if not samples:
        return None

    audio = np.concatenate(samples, axis=-1)

    # Trim samples that fall outside the requested [start_time, start_time + max_duration] window.
    # Audio codecs decode in fixed-size frames whose boundaries may not align with the requested
    # time range, so the first frame can start before start_time and the last frame can end after
    # start_time + max_duration.
    skip_samples = round((start_time - first_frame_time) * sample_rate)
    if skip_samples > 0:
        audio = audio[..., skip_samples:]

    if max_duration is not None:
        max_samples = round(max_duration * sample_rate)
        audio = audio[..., :max_samples]

    waveform = torch.from_numpy(audio).to(device).unsqueeze(0)

    return Audio(waveform=waveform, sampling_rate=sample_rate)


def decode_video_by_frame(
    path: str,
    device: DeviceLikeType,
    starting_frame: int = 0,
    frame_cap: int | None = None,
) -> Generator[torch.Tensor]:
    """Decodes video from a file by sequential frame index, without relying on pts.
    Args:
        path: Path to the video file.
        device: Device to place the resulting tensors on.
        starting_frame: Number of leading frames to skip (default 0).
        frame_cap: Maximum number of frames to yield. If None, no frame limit (default None).
    Yields:
        Frames as tensors of shape (1, H, W, C), dtype uint8.
    """
    container = av.open(path)
    try:
        video_stream = next(s for s in container.streams if s.type == "video")
        for index, frame in enumerate(container.decode(video_stream)):
            if index < starting_frame:
                continue
            tensor = torch.tensor(frame.to_rgb().to_ndarray(), dtype=torch.uint8, device=device).unsqueeze(0)
            yield tensor
            if frame_cap is not None:
                frame_cap -= 1
                if frame_cap == 0:
                    break
    finally:
        container.close()


def decode_video_from_file(
    path: str,
    device: DeviceLikeType,
    start_time: float = 0.0,
    max_duration: float | None = None,
) -> Generator[torch.Tensor]:
    """Decodes video from a file using presentation timestamps for time-based trimming.
    If a frame with no pts is encountered, falls back to :func:`decode_video_by_frame`
    using FPS-derived frame indices.
    Args:
        path: Path to the video file.
        device: Device to place the resulting tensors on.
        start_time: Start time in seconds (default 0.0).
        max_duration: Maximum duration in seconds to decode. If None, reads to end of
            stream (default None).
    Yields:
        Frames as tensors of shape (1, H, W, C), dtype uint8.
    """
    container = av.open(path)
    try:
        video_stream = next(s for s in container.streams if s.type == "video")
        time_base = float(video_stream.time_base)

        if start_time > 0:
            container.seek(int(start_time / time_base), stream=video_stream)

        end_time = start_time + max_duration if max_duration is not None else None

        for frame in container.decode(video_stream):
            # PyAV may leave pts unset when the demuxer does not expose per-frame
            # timestamps (e.g. some raw/elementary streams, stripped or missing
            # metadata, or certain remux paths). Without pts we cannot map frames to
            # wall-clock time, so we fall back to sequential frame indices using the
            # stream's average frame rate.
            if frame.pts is None:
                fps = float(video_stream.average_rate)
                starting_frame = round(start_time * fps)
                frame_cap = round(max_duration * fps) if max_duration is not None else None
                yield from decode_video_by_frame(
                    path=path, device=device, starting_frame=starting_frame, frame_cap=frame_cap
                )
                return
            frame_time = frame.pts * time_base
            if frame_time < start_time:
                continue
            if end_time is not None and frame_time >= end_time:
                break
            yield torch.tensor(frame.to_rgb().to_ndarray(), dtype=torch.uint8, device=device).unsqueeze(0)
    finally:
        container.close()


def encode_single_frame(output_file: str, image_array: np.ndarray, crf: float) -> None:
    container = av.open(output_file, "w", format="mp4")
    try:
        stream = container.add_stream("libx264", rate=1, options={"crf": str(crf), "preset": "veryfast"})
        # Round to nearest multiple of 2 for compatibility with video codecs
        height = image_array.shape[0] // 2 * 2
        width = image_array.shape[1] // 2 * 2
        image_array = image_array[:height, :width]
        stream.height = height
        stream.width = width
        av_frame = av.VideoFrame.from_ndarray(image_array, format="rgb24").reformat(format="yuv420p")
        container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
    finally:
        container.close()


def decode_single_frame(video_file: str) -> np.array:
    container = av.open(video_file)
    try:
        stream = next(s for s in container.streams if s.type == "video")
        frame = next(container.decode(stream))
    finally:
        container.close()
    return frame.to_ndarray(format="rgb24")


def preprocess(image: np.array, crf: int) -> np.array:
    """Re-compress an image at ``crf`` to match the compression the model was trained on.
    ``crf`` is required: the correct value is a property of the model generation, so a
    code-level default here would silently condition on the wrong compression. Reaching this
    with ``crf=None`` means a conditioning skipped resolution -- see
    ``ImageConditioner.load_conditionings``, which every pipeline runs over its image inputs.
    """
    if crf is None:
        raise ValueError(
            "Image conditioning CRF is unresolved (crf=None). Resolve it against the checkpoint "
            "first -- ImageConditioner.load_conditionings(images), or detect_params(checkpoint_path).default_image_crf."
        )
    if crf == 0:
        return image
    if min(image.shape[0], image.shape[1]) < 2:
        return image

    with BytesIO() as output_file:
        encode_single_frame(output_file, image, crf)
        video_bytes = output_file.getvalue()
    with BytesIO(video_bytes) as video_file:
        image_array = decode_single_frame(video_file)
    return image_array
