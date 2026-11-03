"""Reusable, strategy-agnostic config building blocks.
These pydantic blocks describe *how a strategy trains a model* — the base-model source, LoRA
adapters, and the preprocessed dataset. They are strategy-owned (composed onto a strategy's
config, not the trainer's global config) and shared so multiple strategies can reuse them
instead of redeclaring the same fields.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationInfo, field_validator

from ltx_trainer.config_base import ConfigBase, ConfigSection


class DataConfig(ConfigBase):
    """Preprocessed training data source (shared across strategies)."""

    preprocessed_data_root: str | None = Field(
        default=None,
        description="Path to folder containing preprocessed training data (required for training; "
        "optional for preprocessing — defaults to dataset_file parent /.precomputed)",
    )

    dataset_file: str | None = Field(
        default=None,
        description="Path to raw dataset metadata (CSV/JSON/JSONL). Required for preprocessing.",
    )

    num_dataloader_workers: int = Field(
        default=2,
        description="Number of background processes for training data loading (0 means synchronous loading)",
        ge=0,
    )

    @field_validator("preprocessed_data_root")
    @classmethod
    def validate_preprocessed_data_root(cls, v: str | None, info: ValidationInfo) -> str | None:
        """Validate output root: required for training, optional for preprocessing."""
        mode = (info.context or {}).get("mode", "train")
        if v is None:
            if mode == "train":
                raise ValueError("data.preprocessed_data_root is required for training")
            return None

        path = Path(v).expanduser().resolve()
        if mode == "train":
            if not path.exists():
                raise ValueError(f"Dataset path does not exist: {v}")
            if not path.is_dir():
                raise ValueError(f"Dataset path is not a directory: {v}")
        elif path.exists() and not path.is_dir():
            raise ValueError(f"Dataset path is not a directory: {v}")
        return str(path)

    @field_validator("dataset_file")
    @classmethod
    def validate_dataset_file(cls, v: str | None, info: ValidationInfo) -> str | None:
        mode = (info.context or {}).get("mode", "train")
        if v is None:
            if mode == "preprocess":
                raise ValueError("data.dataset_file is required for preprocessing")
            return None
        path = Path(v).expanduser()
        return str(path.expanduser().resolve()) if path.is_absolute() else str(path)

    def resolved_output_root(self) -> Path:
        """Return the preprocessed data output directory."""
        if self.preprocessed_data_root:
            return Path(self.preprocessed_data_root)
        if self.dataset_file:
            return Path(self.dataset_file).parent / ".precomputed"
        raise ValueError("Cannot resolve output root without preprocessed_data_root or dataset_file")

    def describe(self) -> list[ConfigSection]:
        rows: list[tuple[str, object]] = [
            ("Dataset", self.preprocessed_data_root or "(default at preprocess time)"),
            ("Dataloader Workers", self.num_dataloader_workers),
        ]
        if self.dataset_file:
            rows.append(("Dataset File", self.dataset_file))
        return [ConfigSection("📂 Data", rows)]


class OptimizerConfig(ConfigBase):
    """Strategy-owned optimizer, scheduler, and gradient-checkpointing configuration."""

    learning_rate: float = Field(
        default=5e-4,
        description="Learning rate for optimization",
    )

    gradient_checkpointing: bool = Field(
        default=False,
        description="Enable gradient checkpointing to save memory at the cost of slower training",
    )

    optimizer_type: Literal["adamw", "adamw8bit"] = Field(
        default="adamw",
        description="Type of optimizer to use for training",
    )

    scheduler_type: Literal[
        "constant",
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "step",
    ] = Field(
        default="linear",
        description="Type of learning-rate scheduler to use for training",
    )

    scheduler_params: dict = Field(
        default_factory=dict,
        description="Additional parameters for the scheduler",
    )

    def describe(self) -> list[ConfigSection]:
        return [
            ConfigSection(
                "🛠 Optimizer",
                [
                    ("Learning Rate", f"{self.learning_rate:.2e}"),
                    ("Optimizer", self.optimizer_type),
                    ("Scheduler", self.scheduler_type),
                    ("Grad Checkpointing", self.gradient_checkpointing),
                ],
            )
        ]


class TimestepSamplingConfig(ConfigBase):
    """Strategy-owned flow-matching timestep sampling configuration."""

    timestep_sampling_mode: Literal["uniform", "shifted_logit_normal"] = Field(
        default="shifted_logit_normal",
        description="Mode to use for timestep sampling",
    )

    timestep_sampling_params: dict = Field(
        default_factory=dict,
        description="Parameters for timestep sampling",
    )


class ModelSourceConfig(ConfigBase):
    """Base model and text-encoder source for a training strategy."""

    model_path: str | Path = Field(
        ...,
        description="Path to a unified checkpoint or the transformer safetensors in a split checkpoint pack.",
    )

    text_encoder_path: str | Path | None = Field(
        default=None,
        description="Path to the matching Gemma directory or packed text-encoder safetensors.",
    )

    video_vae_path: str | Path | None = Field(
        default=None,
        description="Video VAE safetensors. Required for video processing with a split pack; "
        "defaults to model_path for a monolith.",
    )

    audio_vae_path: str | Path | None = Field(
        default=None,
        description="Audio VAE/vocoder safetensors. Required for audio processing with a split pack; "
        "defaults to model_path for a monolith.",
    )

    load_text_encoder_in_8bit: bool = Field(
        default=False,
        description=(
            "Load the text encoder in 8-bit to save GPU memory during preprocessing and validation prompt encoding."
        ),
    )

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, v: str | Path) -> str | Path:
        """Validate that model_path is an existing local path (not a URL)."""
        if str(v).startswith(("http://", "https://")):
            raise ValueError(f"Model path cannot be a URL: {v}")
        if not Path(v).exists():
            raise ValueError(f"Model path does not exist: {v}")
        return v

    @field_validator("video_vae_path", "audio_vae_path")
    @classmethod
    def validate_component_path(cls, v: str | Path | None) -> str | Path | None:
        if v is not None and not Path(v).is_file():
            raise ValueError(f"Model component path does not exist or is not a file: {v}")
        return v

    def describe(self) -> list[ConfigSection]:
        return [
            ConfigSection(
                "🎬 Model",
                [
                    ("Base", self.model_path),
                    ("Text Encoder", self.text_encoder_path or "Built-in"),
                    ("Video VAE", self.video_vae_path or "Base checkpoint"),
                    ("Audio VAE", self.audio_vae_path or "Base checkpoint"),
                    ("Text Encoder 8bit", self.load_text_encoder_in_8bit),
                ],
            )
        ]


class LoraConfig(ConfigBase):
    """LoRA adaptation settings (used by strategies that train LoRA adapters)."""

    rank: int = Field(
        default=64,
        description="Rank of LoRA adaptation",
        ge=2,
    )

    alpha: int = Field(
        default=64,
        description="Alpha scaling factor for LoRA",
        ge=1,
    )

    dropout: float = Field(
        default=0.0,
        description="Dropout probability for LoRA layers",
        ge=0.0,
        le=1.0,
    )

    target_modules: list[str] = Field(
        default=["to_k", "to_q", "to_v", "to_out.0"],
        description="List of modules to target with LoRA",
    )

    def describe(self) -> list[ConfigSection]:
        return [
            ConfigSection(
                "🔗 LoRA",
                [
                    ("Rank / Alpha", f"{self.rank} / {self.alpha}"),
                    ("Dropout", self.dropout),
                    ("Target Modules", self.target_modules),
                ],
            )
        ]
