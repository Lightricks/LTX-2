"""Shared IC-LoRA helpers: LoRA metadata, mask downsampling, reference-video conditioning.
Used by ``ic_lora`` and ``dubit``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import torch
from einops import rearrange
from safetensors import safe_open

from ltx_core.components.patchifiers import AudioPatchifier
from ltx_core.conditioning import (
    ConditioningItem,
    ConditioningItemAttentionStrengthWrapper,
    VideoConditionByReferenceLatent,
)
from ltx_core.model.video_vae import TilingConfig, VideoEncoder
from ltx_core.types import AudioLatentShape, SpatioTemporalScaleFactors, VideoLatentShape
from ltx_pipelines.utils.media_io import (
    ResizeMode,
    decode_video_by_frame,
    is_exr_dir,
    load_exr_as_hdr_conditioning,
    video_preprocess,
)
from ltx_pipelines.utils.media_io.color_config import EXRColorSpace


def read_lora_reference_downscale_factor(lora_path: str) -> int:
    """Read ``reference_downscale_factor`` from LoRA safetensors metadata (default 1)."""
    try:
        with safe_open(lora_path, framework="pt") as f:
            metadata = f.metadata() or {}
            return int(metadata.get("reference_downscale_factor", 1))
    except Exception as e:
        logging.warning("Failed to read metadata from LoRA file '%s': %s", lora_path, e)
        return 1


def read_lora_reference_temporal_scale_factor(lora_path: str) -> int:
    """Read ``reference_temporal_scale_factor`` from LoRA safetensors metadata (default 1)."""
    try:
        with safe_open(lora_path, framework="pt") as f:
            metadata = f.metadata() or {}
            return int(metadata.get("reference_temporal_scale_factor", 1))
    except Exception as e:
        logging.warning("Failed to read metadata from LoRA file '%s': %s", lora_path, e)
        return 1


def downsample_mask_video_to_latent(
    mask: torch.Tensor,
    target_latent_shape: VideoLatentShape,
) -> torch.Tensor:
    """Downsample a pixel-space mask video to flattened latent token weights."""
    b = mask.shape[0]
    f_lat = target_latent_shape.frames
    h_lat = target_latent_shape.height
    w_lat = target_latent_shape.width

    f_pix = mask.shape[2]
    spatial_down = torch.nn.functional.interpolate(
        rearrange(mask, "b 1 f h w -> (b f) 1 h w"),
        size=(h_lat, w_lat),
        mode="area",
    )
    spatial_down = rearrange(spatial_down, "(b f) 1 h w -> b 1 f h w", b=b)

    first_frame = spatial_down[:, :, :1, :, :]

    if f_pix > 1 and f_lat > 1:
        t = (f_pix - 1) // (f_lat - 1)
        assert (f_pix - 1) % (f_lat - 1) == 0, (
            f"Pixel frames ({f_pix}) not compatible with latent frames ({f_lat}): "
            f"(f_pix - 1) must be divisible by (f_lat - 1)"
        )
        rest = rearrange(spatial_down[:, :, 1:, :, :], "b 1 (f t) h w -> b 1 f t h w", t=t)
        rest = rest.mean(dim=3)
        latent_mask = torch.cat([first_frame, rest], dim=2)
    else:
        latent_mask = first_frame

    return rearrange(latent_mask, "b 1 f h w -> b (f h w)")


def temporal_subsample(video: torch.Tensor, temporal_scale_factor: int) -> torch.Tensor:
    """VAE-aligned temporal subsampling: keep frame 0, then every Nth frame."""
    indices = [0, *list(range(1, video.shape[2], temporal_scale_factor))]
    return video[:, :, indices]


def _reference_prefix_latent_frames(
    prefix_pixel_frames: int, temporal_scale_factor: int, video_scale_factors: SpatioTemporalScaleFactors
) -> int:
    """Number of leading reference latents whose target-time span touches ``prefix_pixel_frames``."""
    if prefix_pixel_frames < 0:
        raise ValueError(f"prefix_pixel_frames must be >= 0, got {prefix_pixel_frames}")
    if temporal_scale_factor <= 0:
        raise ValueError(f"temporal_scale_factor must be > 0, got {temporal_scale_factor}")
    if prefix_pixel_frames == 0:
        return 0
    target_temporal_stride = video_scale_factors.time * temporal_scale_factor
    return 1 + (prefix_pixel_frames + target_temporal_stride - 2) // target_temporal_stride


def append_ic_lora_reference_video_conditionings(  # noqa: PLR0912, PLR0913
    conditionings: list[ConditioningItem],
    video_conditioning: list[tuple[str, float]],
    *,
    height: int,
    width: int,
    num_frames: int,
    starting_frame: int = 0,
    video_encoder: VideoEncoder,
    dtype: torch.dtype,
    device: torch.device,
    reference_downscale_factor: int,
    reference_temporal_scale_factor: int = 1,
    conditioning_attention_strength: float,
    conditioning_attention_mask: torch.Tensor | None,
    tiling_config: TilingConfig | None = None,
    color_space: EXRColorSpace | None = None,
    frames: Iterator[torch.Tensor] | None = None,
    reference_prefix_frames: int = 0,
) -> None:
    """Append reference items, excluding tokens that touch an already-carried prefix.
    The full pixel window is encoded so the causal VAE receives the same temporal
    context and grid as the target chunk. Prefix removal happens only after encoding;
    retained tokens keep their original temporal-grid indices through ``first_latent_frame``.
    """
    scale = reference_downscale_factor
    if scale != 1 and (height % scale != 0 or width % scale != 0):
        raise ValueError(
            f"Output dimensions ({height}x{width}) must be divisible by reference_downscale_factor ({scale})"
        )
    ref_height = height // scale
    ref_width = width // scale

    for video_path, strength in video_conditioning:
        if is_exr_dir(video_path):
            if color_space is None:
                raise ValueError(
                    "EXR input requires --hdr {SRGB_LINEAR,ACESCG,ACESCCT,ACES2065_1} "
                    "to declare the source colour space."
                )
            video = torch.cat(
                list(
                    load_exr_as_hdr_conditioning(
                        video_path,
                        ref_height,
                        ref_width,
                        dtype,
                        device,
                        frame_cap=num_frames,
                        frame_start=starting_frame,
                        color_space=color_space,
                        resize_mode=ResizeMode.REFLECT_PAD,
                    )
                ),
                dim=2,
            )
        else:
            frame_gen = (
                frames
                if frames is not None
                else decode_video_by_frame(
                    path=video_path, starting_frame=starting_frame, frame_cap=num_frames, device=device
                )
            )
            video = video_preprocess(frame_gen, ref_height, ref_width, dtype, device)
        if reference_temporal_scale_factor > 1:
            video = temporal_subsample(video, reference_temporal_scale_factor)
        if tiling_config is not None:
            encoded_video = video_encoder.tiled_encode(video, tiling_config)
        else:
            encoded_video = video_encoder(video)
        full_reference_shape = VideoLatentShape.from_torch_shape(encoded_video.shape)

        if conditioning_attention_mask is not None:
            latent_mask = downsample_mask_video_to_latent(
                mask=conditioning_attention_mask,
                target_latent_shape=full_reference_shape,
            )
        else:
            latent_mask = None

        first_latent_frame = _reference_prefix_latent_frames(
            reference_prefix_frames,
            reference_temporal_scale_factor,
            video_encoder.video_scale_factors,
        )
        if first_latent_frame >= full_reference_shape.frames:
            raise ValueError(
                f"reference_prefix_frames ({reference_prefix_frames}) removes the entire reference "
                f"window ({full_reference_shape.frames} latent frames)"
            )
        if first_latent_frame:
            encoded_video = encoded_video[:, :, first_latent_frame:]
            if latent_mask is not None:
                latent_mask = rearrange(
                    latent_mask,
                    "b (f h w) -> b f h w",
                    f=full_reference_shape.frames,
                    h=full_reference_shape.height,
                    w=full_reference_shape.width,
                )[:, first_latent_frame:]
                latent_mask = rearrange(latent_mask, "b f h w -> b (f h w)")

        if latent_mask is not None:
            attn_mask = latent_mask * conditioning_attention_strength
        elif conditioning_attention_strength < 1.0:
            attn_mask = conditioning_attention_strength
        else:
            attn_mask = None

        cond = VideoConditionByReferenceLatent(
            latent=encoded_video,
            downscale_factor=scale,
            temporal_scale_factor=reference_temporal_scale_factor,
            strength=strength,
            first_latent_frame=first_latent_frame,
        )
        if attn_mask is not None:
            cond = ConditioningItemAttentionStrengthWrapper(cond, attention_mask=attn_mask)
        conditionings.append(cond)


def patchify_audio_reference_latent(
    vae_latents: torch.Tensor,
    *,
    negative_positions: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Patchify audio VAE latents and build RoPE positions (optional negative shift for reference)."""
    patchifier = AudioPatchifier(patch_size=1)
    patchified = patchifier.patchify(vae_latents)
    b, c, _t, mel_bins = vae_latents.shape
    seq_len = patchified.shape[1]
    latent_coords = patchifier.get_patch_grid_bounds(
        output_shape=AudioLatentShape(batch=b, channels=c, frames=seq_len, mel_bins=mel_bins),
        device=device,
    )
    positions = latent_coords.to(dtype=torch.float32)
    if negative_positions:
        aud_dur = positions[:, :, -1, 1].max().item()
        positions = positions - aud_dur - 0.04
    return patchified, positions
