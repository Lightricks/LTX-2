from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ltx_core.components.noisers import Noiser
from ltx_core.conditioning import (
    ConditioningItem,
    VideoGeneratedKeyframeSlots,
)
from ltx_core.devices import activation_budget_bytes, cleanup_accelerator_memory, get_preferred_device
from ltx_core.loader.checkpoint import CheckpointReader
from ltx_core.model.audio_vae import encode_audio
from ltx_core.model.transformer import Modality, TokenTimesteps
from ltx_core.model.video_vae import AutoTiling, TileSizeConfig, TilingConfig, VideoEncoder
from ltx_core.model.video_vae.diffusion_tiling import recommended_decode_tiling_config
from ltx_core.model.video_vae.keyframes import DecodeKeyframes
from ltx_core.model.video_vae.model_configurator import (
    diffvae_tiling_geometry_from_vae_config,
    estimate_diffusion_decoder_weight_bytes,
    is_diffusion_video_vae,
)
from ltx_core.model.video_vae.transformer.config import DiffVAEMode
from ltx_core.model.video_vae.transformer.fallback_na import joint_sdpa_materializes_scores
from ltx_core.text_encoders.gemma import LTXGemmaTextEncoder
from ltx_core.tiling import DimensionSizeConfig
from ltx_core.tools import LatentTools
from ltx_core.types import (
    VIDEO_SCALE_FACTORS,
    Audio,
    AudioLatentShape,
    LatentState,
    SpatioTemporalScaleFactors,
    VideoLatentShape,
    VideoPixelShape,
)
from ltx_pipelines.utils.media_io import (
    ResizeMode,
    decode_audio_from_file,
    decode_image,
    decode_video_from_file,
    get_videostream_fps,
    is_exr_dir,
    load_exr_as_hdr_conditioning,
    resize_aspect_ratio_preserving,
    video_preprocess,
)
from ltx_pipelines.utils.media_io.color_config import EXRColorSpace
from ltx_pipelines.utils.types import LoadedImageConditioning

if TYPE_CHECKING:
    from ltx_pipelines.utils.blocks import DiffusionStage

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    return get_preferred_device()


def cleanup_memory() -> None:
    cleanup_accelerator_memory()


def create_initial_video_latent(
    *,
    pixel_shape: VideoPixelShape,
    device: torch.device,
    dtype: torch.dtype,
    scale_factors: SpatioTemporalScaleFactors,
) -> torch.Tensor:
    """Allocate a zero video latent for ``pixel_shape``."""
    shape = VideoLatentShape.from_pixel_shape(pixel_shape, scale_factors=scale_factors)
    return torch.zeros(*shape.to_torch_shape(), device=device, dtype=dtype)


def create_initial_audio_latent(
    video_latent: torch.Tensor,
    *,
    fps: float,
    video_scale_factors: SpatioTemporalScaleFactors,
) -> torch.Tensor:
    """Allocate a zero audio latent whose duration matches ``video_latent``.
    ``video_scale_factors`` are the video VAE's spatiotemporal factors (audio has none of
    its own). Pass the same value used to size ``video_latent``.
    """
    video_shape = VideoLatentShape.from_torch_shape(video_latent.shape)
    audio_shape = AudioLatentShape.from_duration(
        batch=video_shape.batch,
        duration=float(video_shape.upscale(video_scale_factors).frames) / float(fps),
    )
    return torch.zeros(
        *audio_shape.to_torch_shape(),
        device=video_latent.device,
        dtype=video_latent.dtype,
    )


def create_initial_av_latents(
    *,
    pixel_shape: VideoPixelShape,
    device: torch.device,
    dtype: torch.dtype,
    video_scale_factors: SpatioTemporalScaleFactors,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate matching zero video and audio latents.
    ``video_scale_factors`` are the video VAE's spatiotemporal factors. Audio duration
    follows the video canvas those factors produce.
    """
    video_latent = create_initial_video_latent(
        pixel_shape=pixel_shape,
        device=device,
        dtype=dtype,
        scale_factors=video_scale_factors,
    )
    return video_latent, create_initial_audio_latent(
        video_latent, fps=pixel_shape.fps, video_scale_factors=video_scale_factors
    )


# Historical Conv VAE ``TilingConfig.default()`` spatial long-side + temporal chunk
# (main ``latent_tile_splitters`` aspect-coupled a single 768px tile to H/W).
_CONV_AUTO_LONG_SIDE = DimensionSizeConfig(tile_size=768, overlap=64)
_CONV_AUTO_FRAMES = DimensionSizeConfig(tile_size=80, overlap=24)


def tiling_scale_factors_for_vae(vae_checkpoint_path: str) -> SpatioTemporalScaleFactors:
    """Scale factors decode will pass to ``to_splitters`` for this checkpoint."""
    if not is_diffusion_video_vae(vae_checkpoint_path):
        return VIDEO_SCALE_FACTORS
    with CheckpointReader.open((Path(vae_checkpoint_path),)) as reader:
        metadata = reader.read_checkpoint_metadata()
    vae_config = metadata.get("config", {}).get("vae", {})
    return diffvae_tiling_geometry_from_vae_config(vae_config)["pixel_scale"]


def tiling_config_for_vae(
    vae_checkpoint_path: str,
    *,
    height: int,
    width: int,
    num_frames: int,
    diffvae_optimization: DiffVAEMode = DiffVAEMode.CHUNKED_EAGER,
    device: torch.device | None = None,
    free_bytes: int | None = None,
    keyframes: bool = False,
    vae_dtype: torch.dtype = torch.bfloat16,
) -> TilingConfig:
    """Decode tiling for pipelines.
    Conv VAE: aspect-coupled long-side 768/64 + temporal 80/24.
    DiffVAE: halo/memory-aware :func:`recommended_decode_tiling_config`.
    Pass ``keyframes=True`` for a keyframe-aware ``decode_video(keyframes=)`` decode: joint attention runs
    eager blocks, which changes both the stage-5 coefficient and the withheld reserve.
    """
    if not is_diffusion_video_vae(vae_checkpoint_path):
        _ = num_frames  # API parity with DiffVAE; Conv auto layout is aspect-only.
        return TileSizeConfig.from_long_side(
            long_side=_CONV_AUTO_LONG_SIDE,
            height=height,
            width=width,
            scale_factors=VIDEO_SCALE_FACTORS,
            frames=_CONV_AUTO_FRAMES,
        )

    with CheckpointReader.open((Path(vae_checkpoint_path),)) as reader:
        metadata = reader.read_checkpoint_metadata()
    vae_config = metadata.get("config", {}).get("vae", {})
    geometry = diffvae_tiling_geometry_from_vae_config(vae_config)
    model_bytes = estimate_diffusion_decoder_weight_bytes(vae_checkpoint_path, dtype=vae_dtype)

    resolved_device = device if device is not None else get_device()
    if free_bytes is None:
        free_bytes = activation_budget_bytes(resolved_device)

    return recommended_decode_tiling_config(
        **geometry,
        height=height,
        width=width,
        num_frames=num_frames,
        mode=diffvae_optimization,
        free_bytes=free_bytes,
        model_bytes=model_bytes,
        dtype=vae_dtype,
        keyframes=keyframes,
        joint_sdpa_materializes=keyframes and joint_sdpa_materializes_scores(resolved_device),
    )


def ensure_tiling_config(
    tiling_config: TilingConfig | AutoTiling | None,
    *,
    video_shape: VideoPixelShape,
    vae_checkpoint_path: str,
    diffvae_optimization: DiffVAEMode = DiffVAEMode.CHUNKED_EAGER,
    device: torch.device | None = None,
    free_bytes: int | None = None,
    keyframes: bool = False,
    vae_dtype: torch.dtype = torch.bfloat16,
) -> TilingConfig | None:
    """Resolve pipeline tiling: ``AUTO_TILING`` → recommend; ``None`` → untiled; else validate.
    Scale factors for explicit-config validation come from :func:`tiling_scale_factors_for_vae`
    on ``vae_checkpoint_path``. Call only after ``video_shape.frames`` is known.
    Pass ``keyframes=True`` when the decode will use the keyframe-aware VAE path.
    """
    if tiling_config is None:
        return None
    if isinstance(tiling_config, AutoTiling):
        return tiling_config_for_vae(
            vae_checkpoint_path,
            height=video_shape.height,
            width=video_shape.width,
            num_frames=video_shape.frames,
            diffvae_optimization=diffvae_optimization,
            device=device,
            free_bytes=free_bytes,
            keyframes=keyframes,
            vae_dtype=vae_dtype,
        )
    tiling_config.validate(tiling_scale_factors_for_vae(vae_checkpoint_path), video_shape)
    return tiling_config


def _conform_latent_length(latent: torch.Tensor, expected_frames_count: int) -> torch.Tensor:
    actual_frames = latent.shape[2]
    if actual_frames > expected_frames_count:
        latent = latent[:, :, :expected_frames_count]
    elif actual_frames < expected_frames_count:
        shape_as_list = list(latent.shape)
        shape_as_list[2] = expected_frames_count - actual_frames
        pad = torch.zeros(
            shape_as_list,
            device=latent.device,
            dtype=latent.dtype,
        )
        latent = torch.cat([latent, pad], dim=2)
    return latent


def video_latent_from_file(
    video_encoder: VideoEncoder,
    file_path: str,
    output_shape: VideoPixelShape,
    device: torch.device,
    dtype: torch.dtype,
    start_time: float = 0.0,
    max_duration: float | None = None,
    tiling_config: TilingConfig | None = None,
    color_space: EXRColorSpace | None = None,
) -> torch.Tensor | None:
    """Load video from a file or EXR-frame folder, and encode to latents.
    Args:
        video_encoder: Model used to encode pixel frames to latent space.
        file_path: Path to a video file, or a directory of ``*.exr`` frames
            (HDR; requires ``color_space`` from ``--hdr``).
        output_shape: Target pixel shape (height, width, frames, fps) for the conditioning.
        device: Device to run the encoder and hold tensors on.
        dtype: Dtype for the output latents.
        start_time: Start time in seconds to begin reading the video (default 0.0).
        max_duration: Maximum duration in seconds. If None, uses output_shape.frames at
            output_shape.fps (default None).
        tiling_config: Tiling configuration for the encoder. Defaults to
            ``TileSizeConfig.default()`` (independent H=W=768).
        color_space: HDR colour space when *file_path* is an EXR folder.
            Ignored for regular video files.
    Returns:
        Encoded video latents of shape (1, C, T, H, W) with T = required_latent_frames, or
        None (currently this function always returns a tensor).
    """
    fps = output_shape.fps
    max_duration = max_duration or output_shape.frames / fps
    if is_exr_dir(file_path):
        if color_space is None:
            raise ValueError(
                "EXR input requires --hdr {SRGB_LINEAR,ACESCG,ACESCCT} to declare the source colour space."
            )
        # Center-crop matches the SDR video_preprocess path used for mp4 inputs.
        frame_start = round(start_time * fps)
        frame_cap = max(1, round(max_duration * fps))
        frames = torch.cat(
            list(
                load_exr_as_hdr_conditioning(
                    file_path,
                    output_shape.height,
                    output_shape.width,
                    dtype,
                    device,
                    frame_cap=frame_cap,
                    color_space=color_space,
                    resize_mode=ResizeMode.CENTER_CROP,
                    frame_start=frame_start,
                )
            ),
            dim=2,
        )
    else:
        src_fps = get_videostream_fps(file_path)
        if src_fps != output_shape.fps:
            raise ValueError(f"Input video FPS {src_fps} does not match output FPS {output_shape.fps}, not supported")
        frame_gen = decode_video_from_file(
            path=file_path, device=device, start_time=start_time, max_duration=max_duration
        )
        frames = video_preprocess(frame_gen, output_shape.height, output_shape.width, dtype, device)
    latents = video_encoder.tiled_encode(frames, tiling_config or TileSizeConfig.default())
    required_latent_frames = VideoLatentShape.from_pixel_shape(
        output_shape, scale_factors=video_encoder.video_scale_factors
    ).frames
    return _conform_latent_length(latents, required_latent_frames)


def audio_latent_from_file(
    audio_encoder: torch.nn.Module,
    file_path: str,
    output_shape: VideoPixelShape,
    device: torch.device,
    dtype: torch.dtype,
    start_time: float = 0.0,
    max_duration: float | None = None,
) -> torch.Tensor | None:
    """Load audio from a file, and construct the audio latent conforming to video output shape.
    Args:
        audio_encoder: Model used to encode audio to latent space.
        file_path: Path to the audio or video file containing an audio stream.
            EXR-frame folders have no audio and return ``None``.
        output_shape: Target video pixel shape; used to derive required latent frames
            and, when max_duration is None, the audio duration (output_shape.frames / fps).
        device: Device to run the encoder and hold tensors on.
        dtype: Dtype for the output latents.
        start_time: Start time in seconds to begin reading the audio (default 0.0).
        max_duration: Maximum duration in seconds. If None, uses the full span implied
            by output_shape (default None).
    Returns:
        Encoded audio latents of shape (1, C, T, ...) with T = required_latent_frames, or
        None if the file has no audio stream.
    """
    if is_exr_dir(file_path):
        return None
    max_duration = max_duration or output_shape.frames / output_shape.fps
    audio_in = decode_audio_from_file(file_path, device, start_time, max_duration)
    if audio_in is None:
        return None
    latents = encode_audio(audio_in, audio_encoder, None).to(device, dtype)
    required_latent_frames = AudioLatentShape.from_video_pixel_shape(output_shape).frames
    return _conform_latent_length(latents, required_latent_frames)


def image_conditionings(
    images: Sequence[LoadedImageConditioning],
    height: int,
    width: int,
    video_encoder: VideoEncoder,
    dtype: torch.dtype,
    device: torch.device,
) -> list[ConditioningItem]:
    """Encode loaded stills at ``height`` x ``width`` into conditioning items using each ``binding``."""
    return [img.to_item(video_encoder(img.still.tensor(height, width, dtype, device))) for img in images]


def evenly_spaced_keyframe_positions(num_keyframes: int, num_frames: int) -> list[int]:
    """Interior pixel-frame positions for *num_keyframes* generated keyframes. Endpoints excluded."""
    if num_keyframes < 0:
        raise ValueError(f"num_keyframes must be non-negative, got {num_keyframes}")
    if num_keyframes == 0:
        return []
    if num_frames < num_keyframes + 2:
        raise ValueError(
            f"Generated keyframes need at least num_keyframes + 2 target frames, got "
            f"num_keyframes={num_keyframes}, num_frames={num_frames}"
        )
    return torch.linspace(0, num_frames - 1, num_keyframes + 2).round().to(torch.int64).tolist()[1:-1]


def assert_stage_supports_generated_keyframes(stage: DiffusionStage) -> None:
    """Raise unless ``stage``'s checkpoint declares the keyframe absolute-position embedding."""
    if stage.supports_generated_keyframes:
        return
    raise ValueError(
        f"Generated keyframe slots were requested, but the checkpoint at "
        f"{stage._transformer_builder.checkpoint} does not set 'use_keyframes_abs_pos_embedding' "
        f"in its transformer config, so it has no keyframe absolute-position embedding. Use a "
        f"generated-keyframe checkpoint or drop the keyframe request."
    )


def assert_generated_keyframes_request(
    decode_with_keyframes: bool,
    generated_keyframes: int | Sequence[int],
    stage: DiffusionStage,
) -> None:
    """Validate generated-keyframe and keyframe-decode requests before building models."""
    if has_generated_keyframes(generated_keyframes):
        assert_stage_supports_generated_keyframes(stage)
        return
    if decode_with_keyframes:
        raise ValueError(
            "decode_with_keyframes requires generated keyframe slots; pass "
            "generated_keyframes > 0 or an explicit position list."
        )


def has_generated_keyframes(generated_keyframes: int | Sequence[int]) -> bool:
    """Whether a ``generated_keyframes`` request asks for any slots.
    Callers must not test the argument's truthiness directly: a ``Sequence`` may be a tensor or an
    array, whose truth value is ambiguous and raises.
    """
    if isinstance(generated_keyframes, int):
        return generated_keyframes > 0
    return len(generated_keyframes) > 0


def resolve_generated_keyframes(
    generated_keyframes: int | Sequence[int],
    num_frames: int,
) -> list[int]:
    """Normalize the pipeline-level ``generated_keyframes`` argument to pixel-frame indices.
    An ``int`` requests that many evenly spaced interior keyframes; a sequence gives the target
    pixel-frame indices explicitly. ``0`` / empty means the feature is off.
    """
    if isinstance(generated_keyframes, int):
        return evenly_spaced_keyframe_positions(generated_keyframes, num_frames)

    positions = sorted({int(position) for position in generated_keyframes})
    if positions and (positions[0] < 0 or positions[-1] >= num_frames):
        raise ValueError(
            f"Generated keyframe positions must lie in [0, {num_frames}), got "
            f"{sorted(int(p) for p in generated_keyframes)}"
        )
    return positions


def generated_keyframe_conditionings(
    generated_keyframes: int | Sequence[int],
    num_frames: int,
) -> list[ConditioningItem]:
    """Build the generated-keyframe conditioning, or an empty list when the feature is off.
    All slots go into one :class:`VideoGeneratedKeyframeSlots` so their tokens form a single
    contiguous, exactly-locatable range.
    """
    positions = resolve_generated_keyframes(generated_keyframes, num_frames)
    if not positions:
        return []
    return [VideoGeneratedKeyframeSlots(pixel_frame_indices=positions)]


def decode_keyframes_from_slots(
    latents: torch.Tensor | None,
    positions: Sequence[int],
    num_frames: int,
) -> DecodeKeyframes | None:
    """Build ``DecodeKeyframes`` for slots that still land inside a ``num_frames`` canvas.
    The canvas may pad its tail and be trimmed back to the caller's frame count, so a
    carried keyframe can sit past the end. Those must be dropped rather than handed on: a
    decode indexes keyframes by pixel frame and would otherwise anchor off-canvas.
    """
    if latents is None or not positions:
        return None
    kept = [(index, int(position)) for index, position in enumerate(positions) if 0 <= int(position) < num_frames]
    if not kept:
        return None
    if len(kept) != len(positions):
        logger.info("Dropping %d keyframe(s) past the trimmed %d-frame canvas.", len(positions) - len(kept), num_frames)
    return DecodeKeyframes(
        latents=torch.cat([latents[:, :, index : index + 1] for index, _ in kept], dim=2),
        pixel_frame_indices=torch.tensor([position for _, position in kept], dtype=torch.long),
    )


def create_noised_state(
    tools: LatentTools,
    conditionings: list[ConditioningItem],
    noiser: Noiser,
    dtype: torch.dtype,
    device: torch.device,
    noise_scale: float = 1.0,
    initial_latent: torch.Tensor | None = None,
) -> LatentState:
    """Create a noised latent state from empty state, conditionings, and noiser.
    Creates an empty latent state, applies conditionings, and then adds noise
    using the provided noiser. Returns the final noised state ready for diffusion.
    """
    state = tools.create_initial_state(device, dtype, initial_latent)
    state = state_with_conditionings(state, conditionings, tools)
    state = replace(state, denoise_mask=state.denoise_mask.compact(device))
    return noiser(state, noise_scale)


def state_with_conditionings(
    latent_state: LatentState, conditioning_items: list[ConditioningItem], latent_tools: LatentTools
) -> LatentState:
    """Apply a list of conditionings to a latent state.
    Iterates through the conditioning items and applies each one to the latent
    state in sequence. Returns the modified state with all conditionings applied.
    """
    for conditioning in conditioning_items:
        latent_state = conditioning.apply_to(latent_state=latent_state, latent_tools=latent_tools)

    return latent_state


def post_process_latent(denoised: torch.Tensor, denoise_mask: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Blend denoised output with clean state based on mask."""
    return (denoised * denoise_mask + clean.float() * (1 - denoise_mask)).to(denoised.dtype)


def modality_from_latent_state(
    state: LatentState,
    context: torch.Tensor,
    sigma: torch.Tensor,
    enabled: bool = True,
) -> Modality:
    """Create a Modality from a latent state.
    Constructs a Modality object with the latent state's data, timesteps derived
    from the denoise mask and sigma, positions, and the provided context.
    When ``state.frozen`` is True, ``Modality.sigma`` is forced to 0 so prompt AdaLN
    and cross-modality gates match frozen conditioning (per-token timesteps are
    already zeroed via ``denoise_mask``).
    """
    if state.frozen:
        sigma = torch.zeros_like(sigma)
    values = state.denoise_mask.values
    timesteps = timesteps_from_mask(values.reshape(1, -1), sigma)
    # Clamped because a mask value above the table's last entry would otherwise index one
    # past it, and that read happens inside a captured CUDA graph.
    timestep_indices = torch.searchsorted(
        values, state.denoise_mask.tensor.reshape(state.latent.shape[0], -1), out_int32=True
    ).clamp_(max=values.numel() - 1)
    return Modality(
        enabled=enabled,
        latent=state.latent,
        sigma=sigma,
        timesteps=TokenTimesteps(timesteps, timestep_indices, state.denoise_mask.zero_index),
        positions=state.positions,
        context=context,
        context_mask=None,
        attention_mask=state.attention_mask,
        keyframes_mask=state.keyframes_mask,
    )


def timesteps_from_mask(denoise_mask: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
    """Compute timesteps from a denoise mask and sigma value.
    Multiplies the denoise mask by sigma to produce timesteps for each position
    in the latent state. Areas where the mask is 0 will have zero timesteps.
    When sigma is ``(B,)`` it is reshaped to ``(B, 1, ...)`` so the batch
    dimension aligns correctly with ``denoise_mask``.
    """
    if isinstance(sigma, torch.Tensor) and sigma.dim() == 1:
        sigma = sigma.view(-1, *([1] * (denoise_mask.dim() - 1)))
    return denoise_mask * sigma


_UNICODE_REPLACEMENTS = str.maketrans("\u2018\u2019\u201c\u201d\u2014\u2013\u00a0\u2032\u2212", "''\"\"-- '-")


def clean_response(text: str) -> str:
    """Clean a response from curly quotes and leading non-letter characters which Gemma tends to insert."""
    text = text.translate(_UNICODE_REPLACEMENTS)

    # Remove leading non-letter characters
    for i, char in enumerate(text):
        if char.isalpha():
            return text[i:]
    return text


def generate_enhanced_prompt(
    text_encoder: LTXGemmaTextEncoder,
    prompt: str,
    image_path: str | None = None,
    image_long_side: int = 896,
    seed: int = 42,
    static_cache: bool = False,
) -> str:
    """Generate an enhanced prompt from a text encoder and a prompt."""
    if image_path:
        image = decode_image(image_path=image_path)
        image = torch.tensor(image)
        image = resize_aspect_ratio_preserving(image, image_long_side).to(torch.uint8)
        prompt = text_encoder.enhance_i2v(prompt, image, seed=seed, static_cache=static_cache)
    else:
        prompt = text_encoder.enhance_t2v(prompt, seed=seed, static_cache=static_cache)
    logging.info("Enhanced prompt: %s", prompt)
    return clean_response(prompt)


def assert_resolution(height: int, width: int, is_two_stage: bool, *, divisor: int | None = None) -> None:
    """Assert that the resolution is divisible by the required divisor.
    For two-stage pipelines, the resolution must be divisible by 64.
    For one-stage pipelines, the resolution must be divisible by 32.
    Pass ``divisor`` to override (e.g. 128 for a two-spatial-upsample DFR run).
    """
    if divisor is None:
        divisor = 64 if is_two_stage else 32
    if height % divisor != 0 or width % divisor != 0:
        raise ValueError(
            f"Resolution ({height}x{width}) is not divisible by {divisor}. "
            f"For {'two-stage' if is_two_stage else 'one-stage'} pipelines, "
            f"height and width must be multiples of {divisor}."
        )


def snap_frames_to_grid(frames: int, scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS) -> int:
    """Round ``frames`` down to the nearest ``k * scale_factors.time + 1``.
    The model's frame count must satisfy ``(frames - 1) % scale_factors.time == 0``
    (causal VAE temporal grid).
    """
    if frames < 1:
        raise ValueError(f"frames must be >= 1, got {frames}")
    time_scale = scale_factors.time
    return ((frames - 1) // time_scale) * time_scale + 1


def round_up_frames_to_grid(frames: int, scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS) -> int:
    """Round ``frames`` up to the nearest ``k * scale_factors.time + 1``.
    Already-legal counts are returned unchanged, so callers pad by ``0..scale_factors.time - 1`` frames.
    """
    if frames < 1:
        raise ValueError(f"frames must be >= 1, got {frames}")
    time_scale = scale_factors.time
    return math.ceil((frames - 1) / time_scale) * time_scale + 1


def seconds_to_clamped_num_frames(
    seconds: float,
    *,
    frame_rate: float,
    min_frames: int = 1,
    max_frames: int = 1024,
    scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS,
) -> int:
    """Convert a duration in seconds to a frame count snapped to the VAE's temporal grid.
    Outlier durations are clamped to ``[min_frames, max_frames]`` (before snapping) so a
    misbehaving prediction can't request an OOM-sized generation. Snapping floors to the
    grid, which can undershoot ``min_frames``; when that happens the result is snapped up
    to the next grid point instead, so the ``[min_frames, max_frames]`` contract always holds.
    """
    raw_frames = round(seconds * frame_rate)
    raw_frames = max(min_frames, min(raw_frames, max_frames))
    frames = snap_frames_to_grid(raw_frames, scale_factors)
    if frames < min_frames:
        time_scale = scale_factors.time
        frames = min(-(-(min_frames - 1) // time_scale) * time_scale + 1, max_frames)
    return frames


def audio_duration_seconds(audio: Audio) -> float:
    """Wall-clock duration of ``audio`` from its last waveform axis and sample rate."""
    return audio.waveform.shape[-1] / float(audio.sampling_rate)


def num_frames_from_audio_duration(audio: Audio, *, frame_rate: float) -> int:
    """Convert decoded audio length to a VAE-grid frame count at ``frame_rate``.
    ``decode_audio_from_file`` already applies ``audio_max_duration`` and stops at
    EOF, so this uses the *effective* clip duration: ``min(requested max, remaining
    audio after start time)``.
    The raw frame count is floored (not rounded) before snapping so a leftover
    half-frame cannot jump onto the next ``8k+1`` grid point and request more
    video than the clip can cover.
    """
    raw_frames = int(audio_duration_seconds(audio) * frame_rate)
    raw_frames = max(1, min(raw_frames, 1024))
    return snap_frames_to_grid(raw_frames)
