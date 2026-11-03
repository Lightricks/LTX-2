"""Base class for training strategies.
This module defines the abstract base class that all training strategies must implement,
along with the base configuration class.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

from accelerate import Accelerator
from pydantic import Field
from torch import Tensor

from ltx_trainer.config_base import ConfigBase
from ltx_trainer.preprocessing.types import PreprocessingFingerprint, PreprocessingPipeline
from ltx_trainer.training_strategies.types import (
    CheckpointPayload,
    MetricValue,
    ModelBundle,
    OptimizationBundle,
    ResumeFingerprint,
    StepContext,
    StepResult,
    ValidationContext,
    ValidationResult,
)


class TrainingStrategyConfigBase(ConfigBase):
    """Base configuration class for training strategies.
    All strategy-specific configuration classes should inherit from this.
    """

    name: str = Field(description="Unique name identifying the training strategy type")


class TrainingStrategy(ABC):
    """Default implementation for strategies that use the standard train-step path."""

    config: TrainingStrategyConfigBase
    models: ModelBundle
    optimization: OptimizationBundle
    accelerator: Accelerator

    def __init__(self, config: TrainingStrategyConfigBase) -> None:
        self.config = config

    # Required public contract hooks.

    @abstractmethod
    def build_models(self) -> ModelBundle:
        """Build unprepared named models with a finalized trainable parameter graph.
        Returned modules are not yet ``accelerator.prepare()``-wrapped, but the trainable
        parameter set must be complete: adapters attached, freeze/unfreeze applied, and any
        pre-prepare model surgery (e.g. LoRA injection) done.
        """

    @abstractmethod
    def build_optimizers(self, models: ModelBundle, total_steps: int) -> OptimizationBundle:
        """Build unprepared named optimizers and schedulers (``total_steps`` sets scheduler horizons)."""

    @abstractmethod
    def build_dataloader(self, batch_size: int) -> Iterable:
        """Build an unprepared dataloader or iterable for the given per-process batch size."""

    @abstractmethod
    def collect_checkpoint(self, include_optimizer: bool) -> CheckpointPayload:
        """Return the data to persist for a checkpoint (the trainer writes the files).
        Collective: must run on every rank because gathering a sharded model state dict
        is a collective op. ``include_optimizer`` is False for minimal/off saves and for
        FSDP (where full optimizer state save is unsupported).
        """

    @abstractmethod
    def load_model_state(self, weights: dict[str, Tensor]) -> None:
        """Apply a loaded model state dict. Called before accelerator.prepare."""

    @abstractmethod
    def load_training_state(self, training_state: dict[str, Any]) -> None:
        """Apply loaded optimizer/scheduler (etc.) state. Called after accelerator.prepare."""

    # Default public lifecycle hooks.

    def attach_prepared_components(
        self,
        models: ModelBundle,
        optimization: OptimizationBundle,
        accelerator: Accelerator,
    ) -> None:
        """Attach accelerator-prepared components back to the strategy by name."""
        self.models = models
        self.optimization = optimization
        self.accelerator = accelerator

    def train_step(self, batch: object, ctx: StepContext) -> StepResult:
        """Run one standard optimization step and report logical-step completion."""
        if not hasattr(self, "models") or not hasattr(self, "optimization"):
            raise RuntimeError("Prepared components must be attached before train_step")

        model = next(iter(self.models.models.values()))
        optimizer = next(iter(self.optimization.optimizers.values()))
        scheduler = next(iter(self.optimization.schedulers.values()), None)

        with ctx.accelerator.accumulate(model):
            loss, metrics = self._compute_loss(batch, self.models)
            ctx.accelerator.backward(loss.mean())

            if ctx.accelerator.sync_gradients and ctx.max_grad_norm > 0:
                params = [parameter for model in self.models.models.values() for parameter in model.parameters()]
                ctx.accelerator.clip_grad_norm_(params, ctx.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        return StepResult(
            completed_step=bool(ctx.accelerator.sync_gradients),
            metrics=metrics,
        )

    def validate(self, _ctx: ValidationContext) -> ValidationResult:
        """Default validation hook. Strategies with validation work return media + scalars."""
        return ValidationResult()

    def resume_fingerprint(self) -> ResumeFingerprint:
        """Return structured resume-critical values."""
        return ResumeFingerprint(fields={})

    def preprocessing_fingerprint(self) -> PreprocessingFingerprint | None:
        """Describe preprocessed artifacts for data staleness checks.
        ``None`` opts out for strategies that do not use the trainer's preprocessing
        manifest protocol.
        """
        return None

    def on_before_prepare(self, _accelerator: Accelerator) -> None:
        """Hook called right before ``accelerator.prepare(models)``.
        Strategies can do distributed-aware model surgery here (e.g. an FSDP dtype cast)
        using the accelerator's ``distributed_type``. Default no-op.
        """
        return None

    def on_train_start(self) -> None:
        """Hook called before the first training step."""
        return None

    def on_train_end(self) -> None:
        """Hook called after the training loop finishes."""
        return None

    def build_preprocessing_pipeline(self) -> PreprocessingPipeline:
        """Return the preprocessing pipeline for this strategy.
        Strategies that do not support offline preprocessing should override this to raise
        ``NotImplementedError``.
        """
        raise NotImplementedError(f"Strategy {self.config.name!r} does not support preprocessing")

    # Protected template hook for the default train_step implementation.

    def _compute_loss(
        self,
        batch: object,
        models: ModelBundle,
    ) -> tuple[Tensor, dict[str, float | MetricValue]]:
        """Compute the algorithm loss and strategy-owned metrics."""
        raise NotImplementedError("Strategies using the default train_step must implement _compute_loss.")
