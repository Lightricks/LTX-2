"""Preprocessing protocol and shared context types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias

from accelerate import PartialState

from ltx_trainer import logger
from ltx_trainer.gpu_utils import free_gpu_memory_context

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class PreprocessingFingerprint:
    """Strategy-declared identity of its preprocessed training artifacts."""

    #: Root containing the artifacts and their manifest.
    artifact_root: Path

    #: Diffable values that affect the generated preprocessing artifacts.
    fields: dict[str, JsonValue]


@dataclass(frozen=True)
class PreprocessContext:
    """Runtime inputs shared across preprocessing steps."""

    #: Path to the dataset manifest (CSV or JSONL) listing source files and metadata.
    dataset_file: str

    #: Root directory where preprocessed artifacts are written (typically ``.precomputed/``).
    output_root: Path

    #: Torch device string passed to GPU-backed encoding steps (for example ``"cuda"`` or ``"cpu"``).
    device: str

    #: When ``True``, re-encode every item even if its output file already exists.
    overwrite: bool


class PreprocessStep(Protocol):
    """Single preprocessing unit executed against a shared :class:`PreprocessContext`."""

    def run(self, ctx: PreprocessContext) -> None:
        """Run this step using the shared preprocessing context.
        Args:
            ctx: Runtime inputs shared across all steps in the pipeline.
        """
        ...


@dataclass(frozen=True)
class PreprocessingPipeline:
    """Ordered, self-running sequence of :class:`PreprocessStep` instances."""

    #: Steps executed in list order; each receives the same :class:`PreprocessContext`.
    steps: list[PreprocessStep]

    #: Path to the dataset manifest (CSV or JSONL) listing source files and metadata.
    dataset_file: str

    #: Root directory where preprocessed artifacts are written (typically ``.precomputed/``).
    output_root: Path

    def run(self, *, device: str, overwrite: bool) -> None:
        """Execute every step in order, creating ``output_root`` if needed.
        Args:
            device: Torch device string passed to GPU-backed encoding steps.
            overwrite: When ``True``, re-encode every item even if its output already exists.
        """
        ctx = PreprocessContext(self.dataset_file, self.output_root, device, overwrite)
        self.output_root.mkdir(parents=True, exist_ok=True)
        for step in self.steps:
            logger.info(f"Running preprocessing step: {type(step).__name__}")
            PartialState().wait_for_everyone()
            # Free GPU memory after each step so a step's models/tensors don't linger into the next.
            with free_gpu_memory_context():
                step.run(ctx)
