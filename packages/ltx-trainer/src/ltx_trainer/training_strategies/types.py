"""Public trainer-facing contract types for pluggable training strategies.
The trainer imports these types to prepare strategy-owned models,
optimizers, dataloaders, training steps, metrics, media artifacts, and checkpoint
metadata without importing any concrete strategy package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from accelerate import Accelerator
from torch import nn
from torch.optim.lr_scheduler import LRScheduler

if TYPE_CHECKING:
    from ltx_trainer.progress import TrainingProgress


@dataclass
class ModelBundle:
    """Strategy-owned models that the trainer prepares and reattaches by name."""

    #: Unprepared modules keyed by stable strategy-owned names, such as
    #: "transformer", "student", "teacher", or "critic". Names are used to
    #: reattach prepared modules without relying on positional ordering.
    models: dict[str, nn.Module]


@dataclass(frozen=True)
class OptimizationBundle:
    """Strategy-owned optimizers and schedulers prepared by the trainer."""

    #: Unprepared optimizers keyed by strategy-owned role names. A strategy may
    #: expose one optimizer or several optimizers for multi-model algorithms.
    optimizers: dict[str, torch.optim.Optimizer]

    #: Optional unprepared schedulers keyed by the matching optimizer name.
    schedulers: dict[str, LRScheduler] = field(default_factory=dict)


@dataclass(frozen=True)
class StepContext:
    """Read-only run-shell state passed from the trainer into one train step."""

    #: Current global logical step before applying the returned StepResult.
    global_step: int

    #: Accelerator instance owned by the trainer. Strategies use it for backward,
    #: accumulation, clipping, and distributed helpers, but do not construct it.
    accelerator: Accelerator

    #: Device selected by the trainer for this process.
    device: torch.device

    #: Root output directory for strategy-produced artifacts.
    output_dir: Path

    #: Gradient clipping threshold from the trainer optimization config. The
    #: base train_step uses it; custom train_step implementations may ignore it.
    max_grad_norm: float = 0.0


@dataclass(frozen=True)
class ValidationContext:
    """Read-only run-shell state passed from the trainer into one ``strategy.validate`` call.
    Strategy-agnostic and reusable across strategies; the trainer owns these values.
    """

    #: Accelerator owned by the trainer (for distributed helpers / collectives).
    accelerator: Accelerator

    #: Device selected by the trainer for this process.
    device: torch.device

    #: Current global step at which validation is running.
    step: int

    #: Root output directory for validation artifacts.
    output_dir: Path

    #: Progress reporter for nested sampling progress bars.
    progress: TrainingProgress


@dataclass
class MetricValue:
    """Strategy-emitted scalar with explicit cross-rank reduction semantics."""

    #: Scalar value emitted by the strategy on this rank.
    value: float

    #: Weight used for weighted reductions, especially weighted mean.
    weight: float = 1.0

    #: Reduction policy the trainer/shared logger applies across ranks.
    reduction: Literal["mean", "sum", "min", "max", "main_rank", "already_reduced"] = "mean"


@dataclass
class MediaArtifact:
    """Strategy-emitted artifact that the trainer logs through shared sinks."""

    #: Local path to the artifact produced by the strategy.
    path: Path

    #: Logging key the trainer groups artifacts under, for example "validation_samples".
    key: str

    #: Media category. Set correctly at creation; the W&B sink branches on it (image → wandb.Image,
    #: video → wandb.Video, audio → wandb.Audio).
    kind: Literal["image", "video", "audio"]

    #: Optional human-readable caption for display in logging sinks.
    caption: str | None = None

    #: Small JSON-serializable metadata attached to the artifact.
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


@dataclass
class StepResult:
    """Result of one strategy train_step call, consumed by the trainer.
    The trainer uses ``completed_step`` as the strategy-defined logical clock. For
    simple flow-matching training this usually matches ``accelerator.sync_gradients``:
    a completed optimizer/accumulation boundary is one completed training step.
    More complex strategies can define a logical step differently, such as a full
    GAN discriminator/generator cycle, an RL rollout plus update, or any multi-part
    algorithm where intermediate optimizer updates should not advance global step,
    validation cadence, checkpoint cadence, or termination.
    """

    #: True only when this call finished one strategy-defined logical training
    #: iteration on this rank. The trainer all-reduces this value across ranks
    #: before advancing global_step or running validation/checkpoint cadence.
    completed_step: bool

    #: Strategy-owned scalar metrics to log after rank consensus.
    metrics: dict[str, float | MetricValue] = field(default_factory=dict)

    #: Optional media artifacts to log through trainer-owned sinks.
    media: list[MediaArtifact] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of one ``strategy.validate`` call, consumed by the trainer.
    The strategy owns the validation content (media + scalars); the trainer owns logging and
    cross-rank reduction. Scalars are reduced across ranks by the trainer using each value's
    reduction policy (same semantics as :class:`StepResult` metrics).
    """

    #: Media artifacts (samples) produced during validation, logged through trainer-owned sinks.
    media: list[MediaArtifact] = field(default_factory=list)

    #: Strategy-owned validation scalars, reduced across ranks by the trainer before logging.
    scalars: dict[str, MetricValue] = field(default_factory=dict)


@dataclass(frozen=True)
class ResumeFingerprint:
    """Structured resume-critical fields used for checkpoint compatibility."""

    #: Diffable values that must match for a safe training-state resume. Include
    #: architecture/checkpoint-shape decisions, but exclude safe-to-change knobs
    #: such as learning rate.
    fields: dict[str, str | int | float | bool | None]


@dataclass
class CheckpointPayload:
    """Strategy-produced checkpoint contents handed to the trainer for persistence.
    The strategy never touches the filesystem. It returns the data to save; the trainer
    owns serialization, file layout, atomicity, retention, and the resume metadata. The
    method that produces this (``collect_checkpoint``) is **collective**: it must run on
    every rank because gathering a sharded (FSDP) model state dict is a collective op.
    """

    #: Flat model state dict written by the trainer as ``weights.safetensors`` (cast to the
    #: trainer's configured precision). Meaningful on the main rank; may be empty elsewhere.
    weights: dict[str, torch.Tensor]

    #: Optional ``str``-only safetensors metadata stored alongside the weights.
    weights_metadata: dict[str, str] = field(default_factory=dict)

    #: Optional picklable resume state (e.g. ``{"optimizer": ..., "scheduler": ...}``) the
    #: trainer writes into ``state.pt`` under the ``"strategy_state"`` key. Empty means no resume
    #: state for this checkpoint.
    training_state: dict[str, Any] = field(default_factory=dict)
