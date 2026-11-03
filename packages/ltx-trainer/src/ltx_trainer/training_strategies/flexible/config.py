"""Pydantic configuration for the flexible training strategy.
This module owns every strategy-specific config field: optimizer/scheduler,
timestep sampling, the conditioning (intrinsic/reference) configs, and the
per-modality config that ties them together.
"""

from typing import Annotated, Literal, Union

from pydantic import AliasChoices, Field, field_serializer, field_validator, model_validator

from ltx_core.types import VIDEO_SCALE_FACTORS, SpatioTemporalScaleFactors
from ltx_trainer.config_base import ConfigBase, ConfigSection
from ltx_trainer.quantization import QuantizationOptions
from ltx_trainer.training_strategies.base_strategy import TrainingStrategyConfigBase
from ltx_trainer.training_strategies.shared_config import (
    DataConfig,
    LoraConfig,
    ModelSourceConfig,
    OptimizerConfig,
    TimestepSamplingConfig,
)
from ltx_trainer.validation.config import ValidationConfig


class FlexibleDataConfig(DataConfig):
    """Flexible-strategy data config including offline encoding parameters."""

    resolution_buckets: list[tuple[int, int, int]] | None = Field(
        default=None,
        description="Video resolution buckets as [width, height, frames] (same order as validation.video_dims). "
        "Stored internally as (frames, height, width). Required when the dataset has video.",
    )

    reference_spatial_scale_factor: int = Field(
        default=1,
        ge=1,
        description="Spatial scale factor for video reference conditioning (target/reference resolution ratio). "
        "Must match how reference latents were preprocessed — including when training from an existing "
        ".precomputed/ tree. Validation inherits this value; training also infers scale from latent shapes.",
        validation_alias=AliasChoices("reference_spatial_scale_factor", "reference_downscale_factor"),
    )

    reference_temporal_scale_factor: int = Field(
        default=1,
        ge=1,
        description="Temporal scale factor for video reference conditioning (target/reference frame-rate ratio). "
        "Must match how reference latents were preprocessed — including when training from an existing "
        ".precomputed/ tree. Validation inherits this value; training also infers scale from latent shapes.",
    )

    encode_batch_size: int = Field(
        default=1,
        ge=1,
        description="Batch size for offline preprocessing encoders",
    )

    encode_num_workers: int = Field(
        default=4,
        ge=0,
        description="DataLoader worker count for video preprocessing",
    )

    reshape_mode: Literal["center", "random"] = Field(
        default="center",
        description="How to crop videos during preprocessing",
    )

    vae_tiling: bool = Field(
        default=False,
        description="Enable VAE tiling for larger video resolutions during preprocessing",
    )

    skip_audio: bool = Field(
        default=False,
        description="Do not auto-extract audio from video files during preprocessing",
    )

    audio_durations: list[float] | None = Field(
        default=None,
        description="Audio duration buckets in seconds for audio-only datasets",
    )

    lora_trigger: str | None = Field(
        default=None,
        description="Optional trigger word to prepend to each caption during preprocessing",
    )

    decode: bool = Field(
        default=False,
        description="Decode latents after encoding for verification",
    )

    @field_validator("resolution_buckets", mode="before")
    @classmethod
    def normalize_resolution_buckets(cls, v: object) -> list[tuple[int, int, int]] | None:
        """Accept YAML ``[width, height, frames]`` lists or CLI ``WxHxF;...`` strings.
        Always stores internal ``(frames, height, width)`` tuples used by the encoders.
        """
        if v is None:
            return None
        if isinstance(v, str):
            return parse_resolution_buckets(v)
        if isinstance(v, list):
            buckets: list[tuple[int, int, int]] = []
            for bucket in v:
                if not isinstance(bucket, (list, tuple)) or len(bucket) != 3:
                    raise TypeError(
                        "Each resolution bucket must be a [width, height, frames] list "
                        "(same order as validation.video_dims)"
                    )
                width, height, frames = (int(x) for x in bucket)
                if width <= 0 or height <= 0 or frames <= 0:
                    raise ValueError(f"Width, height, and frames must be positive, got {width}x{height}x{frames}")
                buckets.append((frames, height, width))
            return buckets
        raise TypeError("resolution_buckets must be a list of [width, height, frames] or a 'WxHxF;...' string")

    @field_serializer("resolution_buckets")
    def serialize_resolution_buckets(self, value: list[tuple[int, int, int]] | None) -> list[list[int]] | None:
        if value is None:
            return None
        return [[width, height, frames] for frames, height, width in value]

    @field_validator("audio_durations", mode="before")
    @classmethod
    def normalize_audio_durations(cls, v: object) -> list[float] | None:
        """Accept a list or a semicolon-separated string (legacy CLI / YAML shorthand)."""
        if v is None:
            return None
        if isinstance(v, str):
            return [float(d) for d in v.split(";")]
        return v

    @model_validator(mode="after")
    def validate_audio_durations_positive(self) -> "FlexibleDataConfig":
        if self.audio_durations and any(d <= 0 for d in self.audio_durations):
            raise ValueError("All audio durations must be positive")
        return self


class IntrinsicConditionBase(ConfigBase):
    """Base for intrinsic conditioning — tokens get clean latents, timestep=0, no loss."""

    probability: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Probability of applying this condition",
    )


class FirstFrameConditionConfig(IntrinsicConditionBase):
    """First frame conditioning — frame 0 is clean, excluded from loss."""

    type: Literal["first_frame"] = "first_frame"


class PrefixConditionConfig(IntrinsicConditionBase):
    """Prefix conditioning — first N temporal units are clean, excluded from loss."""

    type: Literal["prefix"] = "prefix"
    temporal_boundary: int = Field(
        ...,
        ge=1,
        description="Number of temporal units for prefix region. "
        "For video: number of latent frames. For audio: number of audio latent timesteps.",
    )


class SuffixConditionConfig(IntrinsicConditionBase):
    """Suffix conditioning — last N temporal units are clean, excluded from loss."""

    type: Literal["suffix"] = "suffix"
    temporal_boundary: int = Field(
        ...,
        ge=1,
        description="Number of temporal units for suffix region. "
        "For video: number of latent frames. For audio: number of audio latent timesteps.",
    )


class SpatialCropConditionConfig(IntrinsicConditionBase):
    """Spatial crop conditioning — rectangular pixel region is clean, excluded from loss."""

    type: Literal["spatial_crop"] = "spatial_crop"
    spatial_region: tuple[int, int, int, int] = Field(
        ...,
        description="Spatial crop region as (y1, x1, y2, x2) in pixel coordinates",
    )


class MaskConditionConfig(IntrinsicConditionBase):
    """Mask conditioning — per-sample binary mask determines conditioning tokens."""

    type: Literal["mask"] = "mask"
    mask_dir: str = Field(
        ...,
        description="Directory containing per-sample masks",
    )


class ReferenceConditionConfig(ConfigBase):
    """Reference conditioning (IC-LoRA style concatenation).
    External reference latents are concatenated to the target sequence.
    Reference tokens are clean (timestep=0), excluded from loss, and
    participate in bidirectional self-attention.
    """

    type: Literal["reference"] = "reference"
    latents_dir: str = Field(..., description="Directory for reference latents")
    probability: float = Field(default=1.0, ge=0.0, le=1.0, description="Probability of applying this condition")


# Discriminated union for condition configs
ConditionConfig = Annotated[
    Union[
        FirstFrameConditionConfig,
        PrefixConditionConfig,
        SuffixConditionConfig,
        SpatialCropConditionConfig,
        MaskConditionConfig,
        ReferenceConditionConfig,
    ],
    Field(discriminator="type"),
]


class ModalityConfig(ConfigBase):
    """Configuration for a single modality (video or audio)."""

    is_generated: bool = Field(
        ...,
        description="True = generated modality (denoised, contributes to loss), False = conditioning-only modality",
    )

    latents_dir: str = Field(
        ...,
        description="Directory for latents",
    )

    conditions: list[ConditionConfig] = Field(
        default_factory=list,
        description="List of conditions (e.g. first_frame, prefix, reference). Text conditioning is always applied.",
    )


class FlexibleStrategyConfig(TrainingStrategyConfigBase):
    """Configuration for the flexible training strategy.
    This strategy supports all conditioning scenarios through configuration:
    - Text-to-video/audio with simple fine-tuning
    - Intrinsic conditioning like first-frame, extension, outpainting
    - Reference conditioning like IC-LoRA (concatenation-based reference)
    """

    name: Literal["flexible"] = "flexible"

    model: ModelSourceConfig = Field(
        description="Base model + text-encoder source.",
    )

    training_mode: Literal["lora", "full"] = Field(
        default="lora",
        description="Train LoRA adapters or the full model.",
    )

    lora: LoraConfig | None = Field(
        default=None,
        description="LoRA configuration (required when training_mode is 'lora').",
    )

    quantization: QuantizationOptions | None = Field(
        default=None,
        description="Optional transformer quantization precision (LoRA mode only).",
    )

    data: FlexibleDataConfig = Field(
        description="Preprocessed training data source.",
    )

    video: ModalityConfig | None = Field(
        default=None,
        description="Video modality configuration",
    )

    audio: ModalityConfig | None = Field(
        default=None,
        description="Audio modality configuration",
    )

    share_sigmas: bool = Field(
        default=True,
        description="Reuse the generated video's noise level for generated audio. "
        "Enabled by default for joint audio-video training; set False for independent sampling. "
        "Conditioning-only modalities always remain clean (sigma=0).",
    )

    optimizer: OptimizerConfig = Field(
        default_factory=OptimizerConfig,
        description="Optimizer and scheduler configuration owned by this strategy",
    )

    timestep_sampling: TimestepSamplingConfig = Field(
        default_factory=TimestepSamplingConfig,
        description="Flow-matching timestep sampling configuration owned by this strategy",
    )

    validation: ValidationConfig = Field(
        default_factory=ValidationConfig,
        description="Validation content (samples, conditions, dims, guidance, STG/CFG, modalities) "
        "owned by this strategy. Validation cadence stays trainer-global.",
    )

    @model_validator(mode="after")
    def validate_lora_present_in_lora_mode(self) -> "FlexibleStrategyConfig":
        """LoRA config is required when training in LoRA mode."""
        if self.training_mode == "lora" and self.lora is None:
            raise ValueError("LoRA configuration must be provided when training_mode is 'lora'")
        return self

    @model_validator(mode="after")
    def validate_at_least_one_generated(self) -> "FlexibleStrategyConfig":
        """Ensure at least one modality has is_generated=true."""
        has_video_target = self.video is not None and self.video.is_generated
        has_audio_target = self.audio is not None and self.audio.is_generated
        if not has_video_target and not has_audio_target:
            raise ValueError("At least one modality must have is_generated=true")
        return self

    @model_validator(mode="after")
    def validate_audio_intrinsic_regions(self) -> "FlexibleStrategyConfig":
        """Reject video-only intrinsic regions on the audio modality."""
        if self.audio is None:
            return self
        for cond in self.audio.conditions:
            if isinstance(cond, (FirstFrameConditionConfig, SpatialCropConditionConfig)):
                raise ValueError(
                    f"Intrinsic condition '{cond.type}' is not supported for audio. "
                    f"Audio supports: prefix, suffix, mask."
                )
        return self

    @model_validator(mode="after")
    def validate_reference_video_dims(self) -> "FlexibleStrategyConfig":
        """Validate scaled dimensions for every validation sample with a video reference."""
        spatial = self.data.reference_spatial_scale_factor
        if spatial <= 1 or not self.validation.samples:
            return self

        for sample in self.validation.samples:
            if not any(
                getattr(condition, "type", None) == "reference" and condition.video is not None
                for condition in sample.conditions
            ):
                continue

            width, height, _frames = sample.video_dims or self.validation.video_dims
            if width % spatial != 0:
                raise ValueError(
                    f"Width {width} is not evenly divisible by data.reference_spatial_scale_factor {spatial}."
                )
            if height % spatial != 0:
                raise ValueError(
                    f"Height {height} is not evenly divisible by data.reference_spatial_scale_factor {spatial}."
                )

        return self

    def get_data_sources(self) -> dict[str, str]:
        """Map directory name (under ``preprocessed_data_root``) to dataset batch key.
        Video/audio latents use fixed batch keys (``video_latents`` / ``audio_latents``);
        reference and mask directories are keyed by their directory name.
        """
        sources: dict[str, str] = {"conditions": "conditions"}
        if self.video is not None:
            sources[self.video.latents_dir] = "video_latents"
        if self.audio is not None:
            sources[self.audio.latents_dir] = "audio_latents"

        for modality_config in (self.video, self.audio):
            if modality_config is None:
                continue
            for cond in modality_config.conditions:
                if isinstance(cond, ReferenceConditionConfig):
                    sources[cond.latents_dir] = cond.latents_dir
                elif isinstance(cond, MaskConditionConfig):
                    sources[cond.mask_dir] = cond.mask_dir

        return sources

    def describe(self) -> list[ConfigSection]:
        sections: list[ConfigSection] = [*self.model.describe()]
        sections.append(
            ConfigSection("🎛 Training", [("Mode", self.training_mode.upper()), ("Quantization", self.quantization)])
        )
        if self.lora is not None:
            sections += self.lora.describe()
        sections += self.data.describe()
        sections += self.optimizer.describe()
        if self.validation.samples:
            sections += self.validation.describe()
        return sections


def _validate_scale_alignment(
    width: int,
    height: int,
    frames: int,
    scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS,
) -> None:
    """Raise if a ``[width, height, frames]`` bucket is not aligned to the VAE compression.
    ``scale_factors`` default to 32x32x8; all current LTX checkpoints use the default. Pass explicit
    factors for a VAE with a different compression.
    """
    if width % scale_factors.width != 0 or height % scale_factors.height != 0:
        raise ValueError(f"Width and height must be multiples of {scale_factors.width}, got {width}x{height}")
    if (frames - 1) % scale_factors.time != 0:
        raise ValueError(f"Number of frames must be a multiple of {scale_factors.time} plus 1, got {frames}")


def parse_resolution_buckets(
    resolution_buckets_str: str,
    scale_factors: SpatioTemporalScaleFactors | None = None,
) -> list[tuple[int, int, int]]:
    """Parse ``WxHxF;...`` strings, optionally validating against known VAE scale factors."""
    buckets = []
    for bucket_str in resolution_buckets_str.split(";"):
        try:
            width, height, frames = map(int, bucket_str.split("x"))
        except ValueError:
            raise ValueError(f"Invalid resolution bucket format: {bucket_str!r}") from None
        if width <= 0 or height <= 0 or frames <= 0:
            raise ValueError(f"Width, height, and frames must be positive, got {width}x{height}x{frames}")
        if scale_factors is not None:
            _validate_scale_alignment(width, height, frames, scale_factors)
        buckets.append((frames, height, width))
    return buckets
