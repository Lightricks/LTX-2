from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from rich import box
from rich.console import Console
from rich.table import Table

from ltx_trainer.config_base import ConfigBase, ConfigSection
from ltx_trainer.config_migration import migrate_legacy_config
from ltx_trainer.training_strategies import TrainingStrategyConfig


class OptimizationConfig(ConfigBase):
    """Configuration for global, trainer-owned optimization parameters.
    Optimizer/scheduler choice, learning rate, and gradient checkpointing are
    strategy-owned and live on the strategy config (e.g. ``FlexibleStrategyConfig``).
    """

    steps: int = Field(
        default=3000,
        description="Number of training steps",
    )

    batch_size: int = Field(
        default=2,
        description="Batch size for training",
    )

    gradient_accumulation_steps: int = Field(
        default=1,
        description="Number of steps to accumulate gradients",
    )

    max_grad_norm: float = Field(
        default=1.0,
        description="Maximum gradient norm for clipping",
    )

    mixed_precision_mode: Literal["no", "fp16", "bf16"] | None = Field(
        default="bf16",
        description="Mixed precision training mode",
    )

    offload_optimizer_during_validation: bool = Field(
        default=False,
        description="Offload optimizer state to CPU before validation video sampling and reload "
        "it afterwards, to free VRAM for inference. Useful when optimizer state is large "
        "(e.g. AdamW for full fine-tuning or high-rank LoRA) and validation OOMs because the "
        "VAE decoder + transformer + optimizer state cannot coexist on the GPU. Has no effect "
        "for FSDP (sharded state). Disabled by default.",
    )

    def describe(self) -> list[ConfigSection]:
        return [
            ConfigSection(
                "⚡ Optimization",
                [
                    ("Steps", self.steps),
                    ("Batch Size", self.batch_size),
                    ("Grad Accumulation", self.gradient_accumulation_steps),
                    ("Max Grad Norm", self.max_grad_norm),
                    ("Mixed Precision", self.mixed_precision_mode),
                    ("Optimizer CPU Offload", self.offload_optimizer_during_validation),
                ],
            )
        ]


class ValidationCadenceConfig(ConfigBase):
    """Trainer-owned validation *cadence* (when to validate).
    Validation *content* (samples/conditions/dims/guidance/STG/CFG/modalities) is strategy-owned
    and lives on the training-strategy config (``ltx_trainer.validation.config.ValidationConfig``).
    """

    interval: int | None = Field(
        default=100,
        description="Number of steps between validation runs. If None, validation is disabled.",
        gt=0,
    )

    skip_initial_validation: bool = Field(
        default=False,
        description="Skip validation video sampling at step 0 (beginning of training)",
    )

    def describe(self) -> list[ConfigSection]:
        return [
            ConfigSection(
                "🎥 Validation",
                [
                    ("Interval", f"Every {self.interval} steps" if self.interval else "Disabled"),
                    ("Skip Initial", self.skip_initial_validation),
                ],
            )
        ]


class CheckpointsConfig(ConfigBase):
    """Configuration for model checkpointing during training"""

    load_checkpoint: str | Path | None = Field(
        default=None,
        description="Path to a checkpoint file or directory to resume from. "
        "If a directory is provided, the latest checkpoint will be used.",
    )

    no_resume: bool = Field(
        default=False,
        description="When True, ignore any saved training state and start from step 0. "
        "Model weights from load_checkpoint are still loaded, but optimizer/scheduler "
        "state and step counter are reset.",
    )

    interval: int | None = Field(
        default=None,
        description="Number of steps between checkpoint saves. If None, intermediate checkpoints are disabled.",
        gt=0,
    )

    keep_last_n: int = Field(
        default=1,
        description="Number of most recent checkpoints to keep. Set to -1 to keep all checkpoints; 0 is invalid.",
        ge=-1,
    )

    @field_validator("keep_last_n")
    @classmethod
    def validate_keep_last_n(cls, value: int) -> int:
        """Require either a positive retention count or -1 (keep all)."""
        if value == 0:
            raise ValueError("keep_last_n must be a positive integer or -1 to keep all checkpoints")
        return value

    precision: Literal["bfloat16", "float32"] = Field(
        default="bfloat16",
        description="Precision to use when saving checkpoint weights. Options: 'bfloat16' or 'float32'.",
    )

    save_training_state: Literal["full", "minimal", "off"] = Field(
        default="minimal",
        description="Save training state alongside checkpoints for resume. "
        "'full': optimizer + scheduler + RNG + step (~800MB for LoRA, much larger for full fine-tuning). "
        "'minimal': scheduler + RNG + step only (~few KB, optimizer state starts fresh). "
        "'off': nothing saved, resume not possible.",
    )

    def describe(self) -> list[ConfigSection]:
        return [
            ConfigSection(
                "💾 Checkpoints",
                [
                    ("Load Checkpoint", self.load_checkpoint),
                    (
                        "Save",
                        f"Every {self.interval} steps (keep {self.keep_last_n})" if self.interval else "Disabled",
                    ),
                    ("Precision", self.precision),
                ],
            )
        ]


class WandbConfig(ConfigBase):
    """Configuration for Weights & Biases logging"""

    enabled: bool = Field(
        default=False,
        description="Whether to enable W&B logging",
    )

    project: str = Field(
        default="ltxv-trainer",
        description="W&B project name",
    )

    entity: str | None = Field(
        default=None,
        description="W&B username or team",
    )

    tags: list[str] = Field(
        default_factory=list,
        description="Tags to add to the W&B run",
    )

    log_validation_videos: bool = Field(
        default=True,
        description="Whether to log validation videos to W&B",
    )

    def describe(self) -> list[ConfigSection]:
        return [ConfigSection("🔌 W&B", [("Status", self.project if self.enabled else "Disabled")])]


class LtxTrainerConfig(ConfigBase):
    """Unified configuration for LTXV training"""

    # Sub-configurations
    training_strategy: TrainingStrategyConfig = Field(
        description="Training strategy configuration. Owns the model, LoRA, data, and algorithm settings."
    )
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    validation: ValidationCadenceConfig = Field(default_factory=ValidationCadenceConfig)
    checkpoints: CheckpointsConfig = Field(default_factory=CheckpointsConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    # General configuration
    allow_stale_preprocessed: bool = Field(
        default=False,
        description="Allow training when the preprocessing manifest differs from the current strategy fingerprint.",
    )

    seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )

    output_dir: str = Field(
        default="outputs",
        description="Directory to save model outputs",
    )

    @field_validator("output_dir")
    @classmethod
    def expand_output_path(cls, v: str) -> str:
        """Expand user home directory in output path."""
        return str(Path(v).expanduser().resolve())

    def describe(self) -> list[ConfigSection]:
        return [
            ConfigSection("🎯 Strategy", [("Name", self.training_strategy.name)]),
            *self.training_strategy.describe(),
            *self.optimization.describe(),
            *self.validation.describe(),
            *self.checkpoints.describe(),
            *self.wandb.describe(),
            ConfigSection("📁 Output", [("Output Dir", self.output_dir), ("Seed", self.seed)]),
        ]

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, mode: str = "train") -> "LtxTrainerConfig":
        """Parse a raw config dict, auto-migrating legacy layout when needed."""
        return cls.model_validate(migrate_legacy_config(raw), context={"mode": mode})

    def print_config(self) -> None:
        """Pretty-print this configuration to the console."""
        sections = self.describe()

        def fmt(v: object, max_len: int = 55) -> str:
            """Format any value for display."""
            if v is None:
                return "[dim]—[/]"
            if isinstance(v, bool):
                return "[green]✓[/]" if v else "[dim]✗[/]"
            if isinstance(v, (list, tuple)):
                if not v:
                    return "[dim]—[/]"
                return ", ".join(str(x) for x in v)
            s = str(v)
            return s[: max_len - 3] + "..." if len(s) > max_len else s

        table = Table(
            title="[bold]⚙️  Training Configuration[/]",
            show_header=False,
            box=box.ROUNDED,
            border_style="bright_blue",
            padding=(0, 1),
            title_style="bold bright_blue",
        )
        table.add_column("Key", style="white", width=20)
        table.add_column("Value", style="cyan")

        for i, section in enumerate(sections):
            if i > 0:
                table.add_row("", "")
            table.add_row(f"[bold yellow]{section.title}[/]", "")
            for key, value in section.rows:
                table.add_row(f"  {key}", fmt(value))

        console = Console()
        console.print()
        console.print(table)
        console.print()
