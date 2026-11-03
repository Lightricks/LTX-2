"""Validation content configuration (samples, conditions, sampling settings).
This is the reusable validation *content* vocabulary: which samples to generate, their
conditions, and the sampling settings (dims/guidance/STG/CFG/modalities). It deliberately
lives in a leaf module with no dependency on ``ltx_trainer.config`` or any strategy package,
so a strategy config can embed it without an import cycle and the shared validation runner can
consume it without importing a strategy.
Validation *cadence* (when to validate) is trainer-owned and lives on the global config
(``ValidationCadenceConfig``); it is intentionally not here.
"""

from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import Field, model_validator

from ltx_trainer.config_base import ConfigBase, ConfigSection

# =============================================================================
# Validation Condition Types
# =============================================================================


class FirstFrameConditionConfig(ConfigBase):
    """First-frame conditioning (intrinsic, latent_idx=0). Always targets video.
    If image_or_video points to a video file, the first frame is automatically extracted.
    """

    type: Literal["first_frame"] = "first_frame"
    image_or_video: str | Path


class PrefixConditionConfig(ConfigBase):
    """Prefix conditioning for temporal extension (intrinsic). Exactly one of video/audio must be set."""

    type: Literal["prefix"] = "prefix"
    video: str | None = None
    audio: str | None = None
    num_frames: int | None = Field(
        default=None,
        ge=1,
        description="Number of pixel frames for video prefix. Must be VAE-aligned: "
        "num_frames %% T == 1, where T is the VAE temporal compression factor (8 by default).",
    )
    duration: float | None = Field(default=None, gt=0, description="Duration in seconds for audio prefix")

    @model_validator(mode="after")
    def validate_exactly_one_modality(self) -> "PrefixConditionConfig":
        if (self.video is None) == (self.audio is None):
            raise ValueError("Exactly one of 'video' or 'audio' must be set for prefix condition")
        return self


class SuffixConditionConfig(ConfigBase):
    """Suffix conditioning for temporal extension (intrinsic). Exactly one of video/audio must be set."""

    type: Literal["suffix"] = "suffix"
    video: str | None = None
    audio: str | None = None
    num_frames: int | None = Field(
        default=None,
        ge=1,
        description="Number of pixel frames for video suffix. Must be VAE-aligned: "
        "num_frames %% T == 0, where T is the VAE temporal compression factor (8 by default).",
    )
    duration: float | None = Field(default=None, gt=0, description="Duration in seconds for audio suffix")

    @model_validator(mode="after")
    def validate_exactly_one_modality(self) -> "SuffixConditionConfig":
        if (self.video is None) == (self.audio is None):
            raise ValueError("Exactly one of 'video' or 'audio' must be set for suffix condition")
        return self


class SpatialCropConditionConfig(ConfigBase):
    """Spatial crop conditioning for outpainting (intrinsic, video only)."""

    type: Literal["spatial_crop"] = "spatial_crop"
    video: str
    spatial_region: tuple[int, int, int, int] = Field(
        ..., description="Spatial crop region as (y1, x1, y2, x2) in pixel coordinates"
    )


class MaskConditionConfig(ConfigBase):
    """Mask-based conditioning for inpainting (intrinsic). Exactly one of video/audio must be set."""

    type: Literal["mask"] = "mask"
    video: str | None = None
    audio: str | None = None
    mask: str

    @model_validator(mode="after")
    def validate_exactly_one_modality(self) -> "MaskConditionConfig":
        if (self.video is None) == (self.audio is None):
            raise ValueError("Exactly one of 'video' or 'audio' must be set for mask condition")
        return self


class ReferenceConditionConfig(ConfigBase):
    """Reference conditioning (IC-LoRA style concatenation). Exactly one of video/audio must be set."""

    type: Literal["reference"] = "reference"
    video: str | None = None
    audio: str | None = None
    include_in_output: bool = False

    @model_validator(mode="after")
    def validate_exactly_one_modality(self) -> "ReferenceConditionConfig":
        if (self.video is None) == (self.audio is None):
            raise ValueError("Exactly one of 'video' or 'audio' must be set for reference condition")
        return self


class VideoToAudioConditionConfig(ConfigBase):
    """Video-to-audio — video is provided as frozen cross-modal conditioning.
    The video is kept clean (sigma=0) and influences audio generation via cross-modal attention.
    """

    type: Literal["video_to_audio"] = "video_to_audio"
    video: str


class AudioToVideoConditionConfig(ConfigBase):
    """Audio-to-video — audio is provided as frozen cross-modal conditioning.
    The audio is kept clean (sigma=0) and influences video generation via cross-modal attention.
    """

    type: Literal["audio_to_video"] = "audio_to_video"
    audio: str


ValidationCondition = Annotated[
    Union[
        FirstFrameConditionConfig,
        PrefixConditionConfig,
        SuffixConditionConfig,
        SpatialCropConditionConfig,
        MaskConditionConfig,
        ReferenceConditionConfig,
        VideoToAudioConditionConfig,
        AudioToVideoConditionConfig,
    ],
    Field(discriminator="type"),
]


class ValidationSample(ConfigBase):
    """Configuration for a single validation sample — fully self-describing."""

    prompt: str
    conditions: list[ValidationCondition] = Field(default_factory=list)

    video_dims: tuple[int, int, int] | None = Field(
        default=None,
        description="Per-sample override for (width, height, frames). None = inherit from ValidationConfig.",
    )
    seed: int | None = Field(
        default=None,
        description="Per-sample override for random seed. None = inherit from ValidationConfig.",
    )

    @model_validator(mode="after")
    def validate_frozen_modality_conflicts(self) -> "ValidationSample":
        frozen_types = {c.type for c in self.conditions if c.type in ("video_to_audio", "audio_to_video")}

        if "video_to_audio" in frozen_types and "audio_to_video" in frozen_types:
            raise ValueError(
                "Cannot have both video_to_audio and audio_to_video conditions — nothing would be generated"
            )

        if "video_to_audio" in frozen_types:
            for c in self.conditions:
                if c.type != "video_to_audio" and _condition_targets_video(c):
                    raise ValueError(
                        f"Cannot use video-targeting '{c.type}' condition when video is frozen (video_to_audio)"
                    )

        if "audio_to_video" in frozen_types:
            for c in self.conditions:
                if c.type != "audio_to_video" and _condition_targets_audio(c):
                    raise ValueError(
                        f"Cannot use audio-targeting '{c.type}' condition when audio is frozen (audio_to_video)"
                    )

        return self


class ValidationConfig(ConfigBase):
    """Validation *content* configuration (owned by the training strategy).
    Holds what to generate and how (samples, conditions, dims, guidance, STG/CFG, modalities).
    Validation *cadence* (interval / skip-initial) is trainer-owned and lives on the global
    ``ValidationCadenceConfig``, not here.
    """

    # Per-sample configuration
    samples: list[ValidationSample] = Field(
        default_factory=list,
        description="List of validation samples. Each sample is fully self-describing with its own "
        "prompt, conditions, and optional overrides.",
    )

    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        description="Negative prompt to use for validation examples",
    )

    video_dims: tuple[int, int, int] = Field(
        default=(960, 544, 97),
        description="Dimensions of validation videos (width, height, frames). "
        "Width and height must be divisible by the VAE spatial factor (32 by default), and frames must "
        "satisfy frames % T == 1, where T is the VAE temporal compression factor (8 by default).",
    )

    frame_rate: float = Field(
        default=24.0,
        description="Frame rate for validation videos",
        gt=0,
    )

    seed: int = Field(
        default=42,
        description="Random seed used when sampling validation videos",
    )

    inference_steps: int = Field(
        default=30,
        description="Number of inference steps for validation",
        gt=0,
    )

    video_cfg_scale: float = Field(
        default=3.0,
        description="Video CFG guidance scale to use during validation",
        ge=1.0,
    )

    audio_cfg_scale: float = Field(
        default=7.0,
        description="Audio CFG guidance scale to use during validation",
        ge=1.0,
    )

    video_stg_scale: float = Field(
        default=1.0,
        description="Video STG scale. 0.0 disables video STG.",
        ge=0.0,
    )

    audio_stg_scale: float = Field(default=1.0, description="Audio STG scale. 0.0 disables audio STG.", ge=0.0)

    stg_blocks: list[int] | None = Field(
        default_factory=lambda: [28],
        description="Which transformer blocks to perturb for STG. "
        "None means all blocks are perturbed. Recommended for LTX-2.3: [28].",
    )

    guidance_rescale: float = Field(
        default=0.7,
        description="Guidance variance rescale factor. 0.0 disables rescaling.",
        ge=0.0,
        le=1.0,
    )

    video_modality_guidance_scale: float = Field(default=3.0, ge=1.0)
    audio_modality_guidance_scale: float = Field(default=3.0, ge=1.0)

    generate_audio: bool = Field(
        default=True,
        description="Whether to generate audio in validation samples. "
        "Independent of training strategy setting - you can generate audio "
        "in validation even when not training the audio branch.",
    )

    generate_video: bool = Field(
        default=True,
        description="Whether to generate video in validation samples. "
        "Set to False for audio-only or v2a validation to save VRAM by skipping video VAE decoder loading. "
        "When False, validation will only generate audio (requires generate_audio=True).",
    )

    # Reference (IC-LoRA) encode scales. These are NOT user-settable here: they live on
    # training_strategy.data (single source of truth) and the strategy injects them via
    # model_copy(update=...) before building the ValidationRunner. _reject_user_reference_scales
    # blocks them from validation YAML; the model_copy update bypasses that validator.
    reference_spatial_scale_factor: int = Field(default=1, ge=1, exclude=True)
    reference_temporal_scale_factor: int = Field(default=1, ge=1, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _reject_user_reference_scales(cls, data: object) -> object:
        """Reject reference-scale factors supplied under ``validation:`` — they belong on ``data``."""
        if isinstance(data, dict):
            offending = {"reference_spatial_scale_factor", "reference_temporal_scale_factor"} & data.keys()
            if offending:
                raise ValueError(
                    f"{', '.join(sorted(offending))} cannot be set under validation; set them on "
                    f"training_strategy.data (they must match the preprocessed dataset)."
                )
        return data

    @model_validator(mode="before")
    @classmethod
    def migrate_deprecated_guidance_fields(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        values = dict(data)
        if "guidance_scale" in values:
            if {"video_cfg_scale", "audio_cfg_scale"} & values.keys():
                raise ValueError(
                    "guidance_scale is deprecated; use video_cfg_scale and audio_cfg_scale instead, "
                    "and do not combine old and new CFG fields."
                )
            guidance_scale = values.pop("guidance_scale")
            values["video_cfg_scale"] = guidance_scale
            values["audio_cfg_scale"] = guidance_scale

        if {"stg_scale", "stg_mode"} & values.keys():
            if {"video_stg_scale", "audio_stg_scale"} & values.keys():
                raise ValueError(
                    "stg_scale and stg_mode are deprecated; use video_stg_scale and audio_stg_scale instead, "
                    "and do not combine old and new STG fields."
                )
            stg_scale = values.pop("stg_scale", 1.0)
            stg_mode = values.pop("stg_mode", "stg_av")
            if stg_mode not in {"stg_av", "stg_v"}:
                raise ValueError("stg_mode must be either 'stg_av' or 'stg_v'")
            values["video_stg_scale"] = stg_scale
            values["audio_stg_scale"] = stg_scale if stg_mode == "stg_av" else 0.0
        return values

    @model_validator(mode="after")
    def validate_output_modality_requirements(self) -> "ValidationConfig":
        """Validate output modality settings when validation is configured."""
        if self.samples and not self.generate_video and not self.generate_audio:
            raise ValueError(
                "At least one of generate_video or generate_audio must be True when validation is configured."
            )
        return self

    def describe(self) -> list[ConfigSection]:
        if not self.samples:
            return []
        return [
            ConfigSection(
                "🎬 Validation Content",
                [
                    ("Samples", f"{len(self.samples)} sample(s)"),
                    (
                        "Video Dims",
                        f"{self.video_dims[0]}x{self.video_dims[1]}, {self.video_dims[2]} frames",
                    ),
                    ("Frame Rate", f"{self.frame_rate} fps"),
                    ("Inference Steps", self.inference_steps),
                    ("CFG Scales", f"video={self.video_cfg_scale}; audio={self.audio_cfg_scale}"),
                    (
                        "STG",
                        f"video={self.video_stg_scale}; audio={self.audio_stg_scale}; blocks={self.stg_blocks}"
                        if self.video_stg_scale > 0 or self.audio_stg_scale > 0
                        else "Disabled",
                    ),
                    ("Seed", self.seed),
                ],
            )
        ]


def _condition_targets_video(cond: ValidationCondition) -> bool:
    """Check if a validation condition targets the video modality."""
    if cond.type in ("first_frame", "spatial_crop", "video_to_audio"):
        return True
    if cond.type in ("prefix", "suffix", "mask", "reference"):
        return getattr(cond, "video", None) is not None
    return False


def _condition_targets_audio(cond: ValidationCondition) -> bool:
    """Check if a validation condition targets the audio modality."""
    if cond.type == "audio_to_video":
        return True
    if cond.type in ("prefix", "suffix", "mask", "reference"):
        return getattr(cond, "audio", None) is not None
    return False
