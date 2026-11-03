"""Flexible-strategy preprocessing pipeline composition and dataset role resolution."""

from __future__ import annotations

from pathlib import Path

from ltx_trainer import logger
from ltx_trainer.dataset_metadata import column_names
from ltx_trainer.model_loader import resolve_audio_vae_path, resolve_video_vae_path
from ltx_trainer.preprocessing.captions import CaptionEncodeStep
from ltx_trainer.preprocessing.decode import DecodeLatentsStep
from ltx_trainer.preprocessing.encode import (
    AudioEncodeStep,
    ReferenceAudioEncodeStep,
    ReferenceVideoEncodeStep,
    VideoEncodeStep,
)
from ltx_trainer.preprocessing.masks import AudioMaskStep, VideoMaskStep
from ltx_trainer.preprocessing.types import PreprocessingPipeline, PreprocessStep
from ltx_trainer.training_strategies.flexible.config import (
    FlexibleDataConfig,
    FlexibleStrategyConfig,
    MaskConditionConfig,
    ReferenceConditionConfig,
)

_KNOWN_ROLES = frozenset(
    {"video", "audio", "reference_video", "reference_audio", "video_mask", "audio_mask", "caption"}
)
_LEGACY_ALIASES = {"media_path": "video", "ref_media_path": "reference_video"}


def validate_dataset_file(dataset_path: str) -> None:
    """Validate that the dataset file exists and has a supported format."""
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_file}")
    if not dataset_file.is_file():
        raise ValueError(f"Dataset path must be a file, not a directory: {dataset_file}")
    if dataset_file.suffix.lower() not in {".csv", ".json", ".jsonl"}:
        raise ValueError(f"Dataset file must be CSV, JSON, or JSONL format: {dataset_file}")


def resolve_columns(
    dataset_columns: set[str],
    *,
    video_column_override: str | None = None,
    caption_column_override: str | None = None,
) -> dict[str, str]:
    """Map canonical role names to actual dataset column names."""
    roles: dict[str, str] = {}
    for col in dataset_columns:
        role = _LEGACY_ALIASES.get(col, col)
        if role in _KNOWN_ROLES:
            roles[role] = col

    if video_column_override and video_column_override in dataset_columns:
        roles["video"] = video_column_override
    if caption_column_override and caption_column_override in dataset_columns:
        roles["caption"] = caption_column_override

    return roles


def detect_roles(
    dataset_file: str,
    *,
    video_column_override: str | None = None,
    caption_column_override: str | None = None,
) -> dict[str, str]:
    """Detect dataset columns and resolve them to canonical preprocessing roles."""
    validate_dataset_file(dataset_file)
    dataset_columns = column_names(dataset_file)
    return resolve_columns(
        dataset_columns,
        video_column_override=video_column_override,
        caption_column_override=caption_column_override,
    )


def validate_roles(
    roles: dict[str, str],
    *,
    resolution_buckets: list[tuple[int, int, int]] | None,
) -> None:
    if "caption" not in roles:
        raise ValueError("No caption column found. Expected a 'caption' column in the dataset.")

    has_video = "video" in roles
    has_audio = "audio" in roles
    if not has_video and not has_audio:
        raise ValueError("No media column found. Expected 'video', 'audio', or legacy 'media_path'.")

    if has_video and not resolution_buckets:
        raise ValueError("data.resolution_buckets is required when the dataset has a video column.")


def audio_encode_params(
    roles: dict[str, str],
    data: FlexibleDataConfig,
    resolution_buckets: list[tuple[int, int, int]] | None,
) -> tuple[float | None, list[float] | None]:
    max_audio_duration = None
    audio_duration_buckets = None
    if "video" in roles and resolution_buckets:
        max_audio_duration = max(frames for frames, _height, _width in resolution_buckets) / 25.0
    elif data.audio_durations:
        audio_duration_buckets = data.audio_durations
    return max_audio_duration, audio_duration_buckets


def _reference_latents_dir(cfg: FlexibleStrategyConfig, *, modality: str) -> str | None:
    cond = _first_condition(cfg, modality=modality, cond_type=ReferenceConditionConfig)
    return cond.latents_dir if cond is not None else None


def _mask_dir(cfg: FlexibleStrategyConfig, *, modality: str) -> str | None:
    cond = _first_condition(cfg, modality=modality, cond_type=MaskConditionConfig)
    return cond.mask_dir if cond is not None else None


def _first_condition(cfg: FlexibleStrategyConfig, *, modality: str, cond_type: type) -> object | None:
    """Return the first condition of ``cond_type`` on the given modality, or ``None``."""
    modality_config = cfg.video if modality == "video" else cfg.audio
    if modality_config is None:
        return None
    for cond in modality_config.conditions:
        if isinstance(cond, cond_type):
            return cond
    return None


def build_flexible_preprocessing_pipeline(cfg: FlexibleStrategyConfig) -> PreprocessingPipeline:  # noqa: PLR0912, PLR0915
    """Build the preprocessing pipeline for the flexible strategy from config and dataset roles."""
    data = cfg.data
    model = cfg.model
    dataset_file = data.dataset_file
    if not dataset_file:
        raise ValueError("data.dataset_file is required for preprocessing")

    roles = detect_roles(dataset_file)

    for role, col in sorted(roles.items()):
        alias_note = f" (alias for '{role}')" if col != role else ""
        logger.info(f"Detected column '{col}'{alias_note} → {role}")

    validate_roles(roles, resolution_buckets=data.resolution_buckets)

    video_latents_dir = cfg.video.latents_dir if cfg.video is not None else None
    audio_latents_dir = cfg.audio.latents_dir if cfg.audio is not None else None
    reference_video_dir = _reference_latents_dir(cfg, modality="video")
    reference_audio_dir = _reference_latents_dir(cfg, modality="audio")
    video_mask_dir = _mask_dir(cfg, modality="video")
    audio_mask_dir = _mask_dir(cfg, modality="audio")

    if data.lora_trigger:
        logger.info(f'LoRA trigger word "{data.lora_trigger}" will be prepended to all captions')

    if data.resolution_buckets and len(data.resolution_buckets) > 1:
        logger.warning("Using multiple resolution buckets. Training batch size must be 1.")

    model_path = str(model.model_path)
    text_encoder_path = str(model.text_encoder_path) if model.text_encoder_path else ""
    # validate_roles guarantees at least one of video/audio is present.
    media_column = roles.get("video") or roles["audio"]
    max_audio_duration, audio_duration_buckets = audio_encode_params(roles, data, data.resolution_buckets)

    steps: list[PreprocessStep] = [
        CaptionEncodeStep(
            caption_column=roles["caption"],
            media_column=media_column,
            model_path=model_path,
            text_encoder_path=text_encoder_path,
            lora_trigger=data.lora_trigger,
            batch_size=data.encode_batch_size,
            load_in_8bit=model.load_text_encoder_in_8bit,
        )
    ]

    has_video = "video" in roles
    auto_audio = has_video and audio_latents_dir is not None and not data.skip_audio and "audio" not in roles
    needs_video_vae = has_video and (
        video_latents_dir is not None or reference_video_dir is not None or video_mask_dir is not None or auto_audio
    )
    video_vae_path = resolve_video_vae_path(model_path, model.video_vae_path) if needs_video_vae else model_path
    needs_audio_vae = (
        auto_audio
        or ("audio" in roles and audio_latents_dir is not None)
        or ("reference_audio" in roles and reference_audio_dir is not None)
    )
    audio_vae_path = resolve_audio_vae_path(model_path, model.audio_vae_path) if needs_audio_vae else model_path
    if has_video and (video_latents_dir is not None or auto_audio):
        if not data.resolution_buckets:
            raise ValueError("data.resolution_buckets is required for video encoding.")
        steps.append(
            VideoEncodeStep(
                video_column=roles["video"],
                output_subdir=video_latents_dir,
                resolution_buckets=data.resolution_buckets,
                model_path=video_vae_path,
                audio_model_path=audio_vae_path if auto_audio else None,
                reshape_mode=data.reshape_mode,
                batch_size=data.encode_batch_size,
                vae_tiling=data.vae_tiling,
                num_workers=data.encode_num_workers,
                audio_output_subdir=audio_latents_dir if auto_audio else None,
                save_video_latents=video_latents_dir is not None,
            )
        )

    if "reference_video" in roles and has_video and reference_video_dir is not None:
        if not data.resolution_buckets:
            raise ValueError("data.resolution_buckets is required for reference video encoding.")
        steps.append(
            ReferenceVideoEncodeStep(
                reference_column=roles["reference_video"],
                main_media_column=roles["video"],
                output_subdir=reference_video_dir,
                resolution_buckets=data.resolution_buckets,
                reference_spatial_scale_factor=data.reference_spatial_scale_factor,
                reference_temporal_scale_factor=data.reference_temporal_scale_factor,
                model_path=video_vae_path,
                reshape_mode=data.reshape_mode,
                batch_size=data.encode_batch_size,
                vae_tiling=data.vae_tiling,
                num_workers=data.encode_num_workers,
            )
        )

    if "video_mask" in roles and has_video and video_mask_dir is not None and video_latents_dir is not None:
        steps.append(
            VideoMaskStep(
                mask_column=roles["video_mask"],
                main_media_column=roles["video"],
                video_latents_subdir=video_latents_dir,
                output_subdir=video_mask_dir,
                model_path=video_vae_path,
            )
        )

    if "audio" in roles and audio_latents_dir is not None:
        steps.append(
            AudioEncodeStep(
                audio_column=roles["audio"],
                output_subdir=audio_latents_dir,
                model_path=audio_vae_path,
                main_media_column=roles.get("video"),
                max_audio_duration=max_audio_duration,
                audio_duration_buckets=audio_duration_buckets,
            )
        )

    if "reference_audio" in roles and reference_audio_dir is not None:
        steps.append(
            ReferenceAudioEncodeStep(
                audio_column=roles["reference_audio"],
                output_subdir=reference_audio_dir,
                model_path=audio_vae_path,
                main_media_column=roles.get("video") or roles.get("audio"),
                max_audio_duration=max_audio_duration,
                audio_duration_buckets=audio_duration_buckets,
            )
        )

    if "audio_mask" in roles and audio_mask_dir is not None and audio_latents_dir is not None:
        steps.append(
            AudioMaskStep(
                mask_column=roles["audio_mask"],
                audio_latents_subdir=audio_latents_dir,
                output_subdir=audio_mask_dir,
                main_media_column=roles.get("video") or roles.get("audio"),
            )
        )

    if data.decode:
        has_audio_for_decode = (
            auto_audio
            or ("audio" in roles and audio_latents_dir is not None)
            or ("reference_audio" in roles and reference_audio_dir is not None)
        )
        steps.append(
            DecodeLatentsStep(
                model_path=model_path,
                video_vae_path=str(model.video_vae_path) if model.video_vae_path else None,
                audio_vae_path=str(model.audio_vae_path) if model.audio_vae_path else None,
                vae_tiling=data.vae_tiling,
                with_audio=has_audio_for_decode,
                video_latents_subdir=video_latents_dir if has_video else None,
                reference_video_latents_subdir=reference_video_dir if "reference_video" in roles else None,
                audio_latents_subdir=audio_latents_dir,
                reference_audio_latents_subdir=reference_audio_dir if "reference_audio" in roles else None,
            )
        )

    if not steps:
        raise ValueError("No preprocessing steps are active for this dataset and strategy configuration.")

    return PreprocessingPipeline(
        steps=steps,
        dataset_file=dataset_file,
        output_root=data.resolved_output_root(),
    )
