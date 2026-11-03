"""Mask preprocessing for video and audio latents."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from accelerate import PartialState
from torchvision.transforms.functional import resize as tv_resize
from torchvision.transforms.functional import to_tensor

from ltx_core.types import VIDEO_SCALE_FACTORS, SpatioTemporalScaleFactors
from ltx_trainer import logger
from ltx_trainer.media_utils import load_audio_from_file, open_image_as_srgb, read_video
from ltx_trainer.model_loader import read_video_scale_factors
from ltx_trainer.preprocessing.common import (
    atomic_save,
    load_source_and_naming_paths,
    output_relative,
)
from ltx_trainer.preprocessing.media import IMAGE_FILE_EXTENSIONS
from ltx_trainer.preprocessing.types import PreprocessContext


def compute_video_masks(
    dataset_file: str | Path,
    mask_column: str,
    latents_dir: str,
    output_dir: str,
    main_media_column: str | None = None,
    overwrite: bool = False,
    scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS,
) -> None:
    """Preprocess video mask files to latent-space binary masks.
    For each sample, loads the mask video/image, applies the same spatial resize/crop as the
    target video (read from saved latent metadata), downsamples to latent dimensions, binarizes,
    and saves as a .pt tensor aligned with the target video latents.
    Args:
        dataset_file: Path to metadata file (CSV/JSON/JSONL).
        mask_column: Column name containing mask video/image paths.
        latents_dir: Directory containing the target video latents (for reading
            spatial/temporal metadata to ensure mask alignment).
        output_dir: Directory to save mask .pt files.
        main_media_column: Column for output file naming (defaults to mask_column).
        scale_factors: Video VAE spatiotemporal compression factors (default 32x32x8), used to map
            latent mask dimensions back to pixel space and downsample.
    """
    success = 0
    for mask_file, target_meta, out_file in _iter_aligned_targets(
        dataset_file, mask_column, latents_dir, output_dir, main_media_column, overwrite
    ):
        out_file.parent.mkdir(parents=True, exist_ok=True)
        atomic_save({"mask": _video_mask_to_latent(mask_file, target_meta, scale_factors)}, out_file)
        success += 1

    logger.info(f"Mask preprocessing complete: {success} masks saved to {output_dir}")


def compute_audio_masks(
    dataset_file: str | Path,
    mask_column: str,
    audio_latents_dir: str,
    output_dir: str,
    main_media_column: str | None = None,
    overwrite: bool = False,
) -> None:
    """Preprocess audio mask files to latent-space binary masks.
    For each sample, loads the mask (a .pt tensor or an audio amplitude envelope), resamples it
    to the target audio latent temporal length, binarizes, and saves aligned with the target
    audio latents.
    Args:
        dataset_file: Path to metadata file (CSV/JSON/JSONL).
        mask_column: Column name containing mask file paths (.wav or .pt).
        audio_latents_dir: Directory containing the target audio latents (for reading
            temporal metadata to ensure mask alignment).
        output_dir: Directory to save mask .pt files.
        main_media_column: Column for output file naming (defaults to mask_column).
    """
    success = 0
    for mask_file, target_meta, out_file in _iter_aligned_targets(
        dataset_file, mask_column, audio_latents_dir, output_dir, main_media_column, overwrite
    ):
        mask_binary = _audio_mask_to_latent(mask_file, target_meta["num_time_steps"])
        if mask_binary is None:
            continue
        out_file.parent.mkdir(parents=True, exist_ok=True)
        atomic_save({"mask": mask_binary}, out_file)
        success += 1

    logger.info(f"Audio mask preprocessing complete: {success} masks saved to {output_dir}")


@dataclass(kw_only=True)
class VideoMaskStep:
    """Downsample video masks to match encoded video latents."""

    mask_column: str
    main_media_column: str
    video_latents_subdir: str
    output_subdir: str
    model_path: str

    def run(self, ctx: PreprocessContext) -> None:
        compute_video_masks(
            dataset_file=ctx.dataset_file,
            mask_column=self.mask_column,
            latents_dir=str(ctx.output_root / self.video_latents_subdir),
            output_dir=str(ctx.output_root / self.output_subdir),
            main_media_column=self.main_media_column,
            overwrite=ctx.overwrite,
            scale_factors=read_video_scale_factors(self.model_path),
        )


@dataclass(kw_only=True)
class AudioMaskStep:
    """Downsample audio masks to match encoded audio latents."""

    mask_column: str
    audio_latents_subdir: str
    output_subdir: str
    main_media_column: str | None = None

    def run(self, ctx: PreprocessContext) -> None:
        audio_latents_source = ctx.output_root / self.audio_latents_subdir
        if not audio_latents_source.exists():
            logger.warning(f"audio_mask column found but no {self.audio_latents_subdir}/ — run with audio first")
            return

        compute_audio_masks(
            dataset_file=ctx.dataset_file,
            mask_column=self.mask_column,
            audio_latents_dir=str(audio_latents_source),
            output_dir=str(ctx.output_root / self.output_subdir),
            main_media_column=self.main_media_column,
            overwrite=ctx.overwrite,
        )


def _iter_aligned_targets(
    dataset_file: str | Path,
    source_column: str,
    latents_dir: str,
    output_dir: str,
    naming_column: str | None,
    overwrite: bool,
) -> Iterator[tuple[Path, dict[str, Any], Path]]:
    """Yield ``(source_file, target_meta, out_file)`` for samples whose target latent exists.
    Centralizes the mask scaffolding shared by video and audio: output naming (via
    ``naming_column``), skipping already-computed outputs, and warning when the aligned
    target latent is missing.
    """
    dataset_path = Path(dataset_file)
    data_root = dataset_path.parent
    latents_path = Path(latents_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    source_paths, naming_paths = load_source_and_naming_paths(dataset_path, source_column, naming_column)

    state = PartialState()
    for index in range(state.process_index, len(source_paths), state.num_processes):
        source_file = source_paths[index]
        naming_file = naming_paths[index]
        rel_path = output_relative(naming_file, data_root)
        latent_file = latents_path / rel_path.with_suffix(".pt")
        out_file = output_path / rel_path.with_suffix(".pt")
        if not latent_file.exists():
            logger.warning(f"No target latent found at {latent_file}, skipping mask {source_file}")
            continue
        if not overwrite and out_file.is_file():
            continue
        yield source_file, torch.load(latent_file, map_location="cpu", weights_only=True), out_file


def _video_mask_to_latent(
    mask_file: Path,
    target_meta: dict[str, Any],
    scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS,
) -> torch.Tensor:
    """Turn a mask video/image into a binary latent-space mask aligned to the target video latent."""
    latent_f = target_meta["num_frames"]
    latent_h = target_meta["height"]
    latent_w = target_meta["width"]
    pixel_h = latent_h * scale_factors.height
    pixel_w = latent_w * scale_factors.width
    pixel_f = (latent_f - 1) * scale_factors.time + 1

    return load_video_mask(
        mask_file,
        target_num_frames=pixel_f,
        target_height=pixel_h,
        target_width=pixel_w,
        scale_factors=scale_factors,
    )


def load_video_mask(
    mask_file: str | Path,
    *,
    target_num_frames: int,
    target_height: int,
    target_width: int,
    scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS,
) -> torch.Tensor:
    """Load a binary mask with causal VAE temporal grouping and return [F', H', W']."""
    mask_file = Path(mask_file)
    if (target_num_frames - 1) % scale_factors.time:
        raise ValueError("Video mask target frames must align with the VAE temporal scale")
    if target_height % scale_factors.height or target_width % scale_factors.width:
        raise ValueError("Video mask target dimensions must align with the VAE spatial scale")

    if mask_file.suffix.lower() in IMAGE_FILE_EXTENSIONS:
        frames = to_tensor(open_image_as_srgb(mask_file)).mean(dim=0, keepdim=True)
        # Share the image resize between offline preprocessing and validation without
        # materializing one full-resolution copy per video frame for static masks.
        mask_pixels = tv_resize(frames.unsqueeze(0), [target_height, target_width])
        mask_latent = torch.nn.functional.avg_pool2d(
            mask_pixels, kernel_size=(scale_factors.height, scale_factors.width)
        ).squeeze(0)
        latent_frames = 1 + (target_num_frames - 1) // scale_factors.time
        return (mask_latent > 0.5).float().expand(latent_frames, -1, -1)

    video_frames, _ = read_video(str(mask_file), max_frames=target_num_frames)
    frames = video_frames[:target_num_frames].mean(dim=1)
    # Missing mask frames leave the corresponding video tokens generated. Padding
    # to the full target span keeps training and validation masks shape-aligned.
    if frames.shape[0] < target_num_frames:
        frames = torch.nn.functional.pad(frames, (0, 0, 0, 0, 0, target_num_frames - frames.shape[0]))

    mask_pixels = torch.nn.functional.interpolate(
        frames.unsqueeze(1), size=(target_height, target_width), mode="nearest"
    )
    mask_latent = torch.nn.functional.avg_pool2d(
        mask_pixels, kernel_size=(scale_factors.height, scale_factors.width)
    ).squeeze(1)
    first_frame = mask_latent[:1]
    if mask_latent.shape[0] > 1:
        latent_height, latent_width = mask_latent.shape[1:]
        remaining_frames = mask_latent[1:]
        pad_frames = (-remaining_frames.shape[0]) % scale_factors.time
        if pad_frames:
            remaining_frames = torch.nn.functional.pad(remaining_frames, (0, 0, 0, 0, 0, pad_frames))
        remaining_frames = remaining_frames.reshape(-1, scale_factors.time, latent_height, latent_width).amax(dim=1)
        mask_latent = torch.cat((first_frame, remaining_frames), dim=0)

    return (mask_latent > 0.5).float()


def _audio_mask_to_latent(mask_file: Path, latent_t: int) -> torch.Tensor | None:
    """Turn a mask (.pt tensor or audio amplitude envelope) into a binary latent-length audio mask.
    Returns ``None`` when the mask audio cannot be loaded (the sample is then skipped).
    """
    if mask_file.suffix == ".pt":
        raw_mask = torch.load(mask_file, map_location="cpu", weights_only=True)
        if isinstance(raw_mask, dict):
            raw_mask = raw_mask.get("mask", next(iter(raw_mask.values())))
        raw_mask = raw_mask.float().flatten()
    else:
        audio = load_audio_from_file(mask_file)
        if audio is None:
            logger.warning(f"Could not load audio mask from {mask_file}")
            return None
        raw_mask = audio.waveform.abs().mean(dim=0)  # mono amplitude envelope

    # Resample to target audio latent length
    mask_resampled = torch.nn.functional.interpolate(
        raw_mask.unsqueeze(0).unsqueeze(0), size=latent_t, mode="nearest"
    ).squeeze()  # [latent_t]
    return (mask_resampled > 0.5).float()
