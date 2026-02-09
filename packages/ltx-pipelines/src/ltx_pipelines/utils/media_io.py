import logging
import math
from collections.abc import Generator, Iterator
from fractions import Fraction
from io import BytesIO

import av
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torch._prims_common import DeviceLikeType
from tqdm import tqdm

from ltx_pipelines.utils.constants import DEFAULT_IMAGE_CRF

logger = logging.getLogger(__name__)


def resize_aspect_ratio_preserving(image: torch.Tensor, long_side: int) -> torch.Tensor:
    """
    Resize image preserving aspect ratio (filling target long side).
    Preserves the input dimensions order.
    Args:
        image: Input image tensor with shape (F (optional), H, W, C)
        long_side: Target long side size.
    Returns:
        Tensor with shape (F (optional), H, W, C) F = 1 if input is 3D, otherwise input shape[0]
    """
    height, width = image.shape[-3:2]
    max_side = max(height, width)
    scale = long_side / float(max_side)
    target_height = int(height * scale)
    target_width = int(width * scale)
    resized = resize_and_center_crop(image, target_height, target_width)
    # rearrange and remove batch dimension
    result = rearrange(resized, "b c f h w -> b f h w c")[0]
    # preserve input dimensions
    return result[0] if result.shape[0] == 1 else result


def resize_and_center_crop(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Resize tensor preserving aspect ratio (filling target), then center crop to exact dimensions.
    Args:
        latent: Input tensor with shape (H, W, C) or (F, H, W, C)
        height: Target height
        width: Target width
    Returns:
        Tensor with shape (1, C, 1, height, width) for 3D input or (1, C, F, height, width) for 4D input
    """
    if tensor.ndim == 3:
        tensor = rearrange(tensor, "h w c -> 1 c h w")
    elif tensor.ndim == 4:
        tensor = rearrange(tensor, "f h w c -> f c h w")
    else:
        raise ValueError(f"Expected input with 3 or 4 dimensions; got shape {tensor.shape}.")

    _, _, src_h, src_w = tensor.shape

    scale = max(height / src_h, width / src_w)
    # Use ceil to avoid floating-point rounding causing new_h/new_w to be
    # slightly smaller than target, which would result in negative crop offsets.
    new_h = math.ceil(src_h * scale)
    new_w = math.ceil(src_w * scale)

    tensor = torch.nn.functional.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)

    crop_top = (new_h - height) // 2
    crop_left = (new_w - width) // 2
    tensor = tensor[:, :, crop_top : crop_top + height, crop_left : crop_left + width]

    tensor = rearrange(tensor, "f c h w -> 1 c f h w")
    return tensor


def normalize_latent(latent: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return (latent / 127.5 - 1.0).to(device=device, dtype=dtype)


def load_image_conditioning(
    image_path: str, height: int, width: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """
    Loads an image from a path and preprocesses it for conditioning.
    Note: The image is resized to the nearest multiple of 2 for compatibility with video codecs.
    """
    image = decode_image(image_path=image_path)
    image = preprocess(image=image)
    image = torch.tensor(image, dtype=torch.float32, device=device)
    image = resize_and_center_crop(image, height, width)
    image = normalize_latent(image, device, dtype)
    return image


def load_video_conditioning(
    video_path: str, height: int, width: int, frame_cap: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """
    Loads a video from a path and preprocesses it for conditioning.
    Note: The video is resized to the nearest multiple of 2 for compatibility with video codecs.
    """
    frames = decode_video_from_file(path=video_path, frame_cap=frame_cap, device=device)
    result = None
    for f in frames:
        frame = resize_and_center_crop(f.to(torch.float32), height, width)
        frame = normalize_latent(frame, device, dtype)
        result = frame if result is None else torch.cat([result, frame], dim=2)
    return result


def decode_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path)
    np_array = np.array(image)[..., :3]
    return np_array


def _write_audio(
    container: av.container.Container, audio_stream: av.audio.AudioStream, samples: torch.Tensor, audio_sample_rate: int
) -> None:
    if samples.ndim == 1:
        samples = samples[:, None]

    if samples.shape[1] != 2 and samples.shape[0] == 2:
        samples = samples.T

    if samples.shape[1] != 2:
        raise ValueError(f"Expected samples with 2 channels; got shape {samples.shape}.")

    # Convert to int16 packed for ingestion; resampler converts to encoder fmt.
    if samples.dtype != torch.int16:
        samples = torch.clip(samples, -1.0, 1.0)
        samples = (samples * 32767.0).to(torch.int16)

    frame_in = av.AudioFrame.from_ndarray(
        samples.contiguous().reshape(1, -1).cpu().numpy(),
        format="s16",
        layout="stereo",
    )
    frame_in.sample_rate = audio_sample_rate

    _resample_audio(container, audio_stream, frame_in)


def _prepare_audio_stream(container: av.container.Container, audio_sample_rate: int) -> av.audio.AudioStream:
    """
    Prepare the audio stream for writing.
    """
    audio_stream = container.add_stream("aac", rate=audio_sample_rate)
    audio_stream.codec_context.sample_rate = audio_sample_rate
    audio_stream.codec_context.layout = "stereo"
    audio_stream.codec_context.time_base = Fraction(1, audio_sample_rate)
    return audio_stream


def _resample_audio(
    container: av.container.Container, audio_stream: av.audio.AudioStream, frame_in: av.AudioFrame
) -> None:
    cc = audio_stream.codec_context

    # Use the encoder's format/layout/rate as the *target*
    target_format = cc.format or "fltp"  # AAC → usually fltp
    target_layout = cc.layout or "stereo"
    target_rate = cc.sample_rate or frame_in.sample_rate

    audio_resampler = av.audio.resampler.AudioResampler(
        format=target_format,
        layout=target_layout,
        rate=target_rate,
    )

    audio_next_pts = 0
    for rframe in audio_resampler.resample(frame_in):
        if rframe.pts is None:
            rframe.pts = audio_next_pts
        audio_next_pts += rframe.samples
        rframe.sample_rate = frame_in.sample_rate
        container.mux(audio_stream.encode(rframe))

    # flush audio encoder
    for packet in audio_stream.encode():
        container.mux(packet)


def encode_video(
    video: torch.Tensor | Iterator[torch.Tensor],
    fps: int,
    audio: torch.Tensor | None,
    audio_sample_rate: int | None,
    output_path: str,
    video_chunks_number: int,
) -> None:
    if isinstance(video, torch.Tensor):
        video = iter([video])

    first_chunk = next(video)

    _, height, width, _ = first_chunk.shape

    container = av.open(output_path, mode="w")
    stream = container.add_stream("libx264", rate=int(fps))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    if audio is not None:
        if audio_sample_rate is None:
            raise ValueError("audio_sample_rate is required when audio is provided")

        audio_stream = _prepare_audio_stream(container, audio_sample_rate)

    def all_tiles(
        first_chunk: torch.Tensor, tiles_generator: Generator[tuple[torch.Tensor, int], None, None]
    ) -> Generator[tuple[torch.Tensor, int], None, None]:
        yield first_chunk
        yield from tiles_generator

    for video_chunk in tqdm(all_tiles(first_chunk, video), total=video_chunks_number):
        video_chunk_cpu = video_chunk.to("cpu").numpy()
        for frame_array in video_chunk_cpu:
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

    # Flush encoder
    for packet in stream.encode():
        container.mux(packet)

    if audio is not None:
        _write_audio(container, audio_stream, audio, audio_sample_rate)

    container.close()
    logger.info(f"Video saved to {output_path}")


def decode_audio_from_file(path: str, device: torch.device) -> torch.Tensor | None:
    container = av.open(path)
    try:
        audio = []
        audio_stream = next(s for s in container.streams if s.type == "audio")
        for frame in container.decode(audio_stream):
            audio.append(torch.tensor(frame.to_ndarray(), dtype=torch.float32, device=device).unsqueeze(0))
        container.close()
        audio = torch.cat(audio)
    except StopIteration:
        audio = None
    finally:
        container.close()

    return audio


def decode_video_from_file(path: str, frame_cap: int, device: DeviceLikeType) -> Generator[torch.Tensor]:
    container = av.open(path)
    try:
        video_stream = next(s for s in container.streams if s.type == "video")
        for frame in container.decode(video_stream):
            tensor = torch.tensor(frame.to_rgb().to_ndarray(), dtype=torch.uint8, device=device).unsqueeze(0)
            yield tensor
            frame_cap = frame_cap - 1
            if frame_cap == 0:
                break
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


def preprocess(image: np.array, crf: float = DEFAULT_IMAGE_CRF) -> np.array:
    if crf == 0:
        return image

    with BytesIO() as output_file:
        encode_single_frame(output_file, image, crf)
        video_bytes = output_file.getvalue()
    with BytesIO(video_bytes) as video_file:
        image_array = decode_single_frame(video_file)
    return image_array
