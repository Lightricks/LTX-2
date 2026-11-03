"""Checkpoint envelope persistence for the trainer.
Owns the on-disk checkpoint format: each checkpoint is a directory ``checkpoints/step_NNNNN/``
containing ``weights.safetensors`` (the model artifact), ``state.pt`` (optimizer/scheduler/RNG
resume blob), and ``metadata.json`` (step, W&B run id, resume fingerprint, format version).
The manager is **stateful**: constructed once with the output directory and the retention /
precision policy, it tracks the checkpoints it has written so callers don't thread that state
around. It is a pure filesystem + format layer -- it performs no collectives and is unaware of
models, optimizers, or the distributed runtime. The trainer keeps those concerns (running
``collect_checkpoint`` on every rank, RNG capture, ``wait_for_everyone``) and calls the manager
on the main process to read and write bytes.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from ltx_trainer import logger

if TYPE_CHECKING:
    from ltx_trainer.training_strategies.types import CheckpointPayload

# Checkpoint envelope layout. Each checkpoint is a directory `checkpoints/step_NNNNN/`.
CHECKPOINT_FORMAT_VERSION = 1
WEIGHTS_FILE = "weights.safetensors"  # strategy-provided model weights (the artifact)
STATE_FILE = "state.pt"  # optimizer + scheduler + RNG (resume blob); absent for save_training_state="off"
METADATA_FILE = "metadata.json"  # step, wandb run id, resume fingerprint, format version
_STEP_DIR_RE = re.compile(r"step_(\d+)")  # matched with fullmatch so temp dirs like .tmp_step_* are ignored

Fingerprint = dict[str, str | int | float | bool | None]


@dataclass(frozen=True)
class ResumePlan:
    """Resolved intent for restoring from ``load_checkpoint``, computed before model preparation."""

    initial_step: int = 0
    #: Checkpoint dir whose optimizer/scheduler/RNG should be restored after prepare (None = no resume).
    resume_dir: Path | None = None
    #: File or checkpoint dir to load model weights from before prepare (None = fresh init, no weights).
    weights_source: Path | None = None
    wandb_run_id: str | None = None


class CheckpointManager:
    """Stateful reader/writer for directory-based checkpoint envelopes."""

    def __init__(self, output_dir: str | Path, *, precision: str, keep_last_n: int) -> None:
        self._output_dir = Path(output_dir)
        self._save_dtype = torch.bfloat16 if precision == "bfloat16" else torch.float32
        self._keep_last_n = keep_last_n
        checkpoint_root = self._output_dir / "checkpoints"
        existing_dirs: list[tuple[int, Path]] = []
        if checkpoint_root.is_dir():
            for child in checkpoint_root.iterdir():
                match = _STEP_DIR_RE.fullmatch(child.name)
                if child.is_dir() and match:
                    existing_dirs.append((int(match.group(1)), child))
        # Track existing and newly written checkpoints, oldest first. Only the main process writes.
        self._saved_dirs: list[Path] = [path for _, path in sorted(existing_dirs, key=lambda item: item[0])]

    # --- Paths and discovery ---

    def step_dir(self, step: int) -> Path:
        """Path of the checkpoint envelope directory for a given global step."""
        return self._output_dir / "checkpoints" / f"step_{step:05d}"

    @staticmethod
    def latest_step_dir(path: Path) -> Path | None:
        """Return the highest-numbered ``step_NNNNN`` checkpoint dir.
        Accepts a ``step_NNNNN`` dir directly, a ``checkpoints/`` dir, or a run ``output_dir``
        (whose ``checkpoints/`` subdir holds the step dirs).
        """
        if _STEP_DIR_RE.fullmatch(path.name):
            return path
        search_roots = [path]
        checkpoints_subdir = path / "checkpoints"
        if checkpoints_subdir.is_dir():
            search_roots.append(checkpoints_subdir)
        candidates: list[tuple[int, Path]] = []
        for root in search_roots:
            for child in root.iterdir():
                match = _STEP_DIR_RE.fullmatch(child.name)
                if child.is_dir() and match:
                    candidates.append((int(match.group(1)), child))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    @staticmethod
    def guard_legacy_checkpoint(directory: Path) -> None:
        """Reject pre-2.x flat checkpoint files instead of silently starting from step 0."""
        legacy = sorted(directory.glob("training_state_step_*.pt")) + sorted(
            directory.glob("*_weights_step_*.safetensors")
        )
        if legacy:
            raise RuntimeError(CheckpointManager._legacy_checkpoint_message(legacy[0]))

    # --- Resume resolution ---

    def resolve(
        self, load_checkpoint: str | Path | None, *, no_resume: bool, expected_fingerprint: Fingerprint
    ) -> ResumePlan:
        """Resolve ``load_checkpoint`` into a concrete resume / initial-weights plan.
        Handles three intents: (a) no checkpoint, (b) initial weights from an external
        ``.safetensors`` file (step 0, fresh optimizer), and (c) a checkpoint directory to
        resume from. Pre-2.x flat checkpoint files are rejected loudly rather than silently
        falling back to step 0.
        """
        if not load_checkpoint:
            return ResumePlan()

        load = Path(load_checkpoint)
        if load.is_file():
            if load.suffix != ".safetensors":
                raise ValueError(f"Checkpoint file must have a .safetensors extension: {load}")
            # A flat weights file is loaded as initial weights. Only a sibling pre-2.x
            # training-state file (an attempted resume of the old format) is an error.
            legacy_state = sorted(load.parent.glob("training_state_step_*.pt"))
            if legacy_state:
                raise RuntimeError(self._legacy_checkpoint_message(legacy_state[0]))
            logger.info(f"📥 Loading initial weights from {load}")
            return ResumePlan(weights_source=load)

        if not load.is_dir():
            logger.warning(f"⚠️ Could not find checkpoint at {load}")
            return ResumePlan()

        return self._resolve_dir(load, no_resume=no_resume, expected_fingerprint=expected_fingerprint)

    # --- Loading ---

    def load_weights(self, source: Path) -> dict[str, Tensor]:
        """Load a model state dict from a ``weights.safetensors`` file or a checkpoint dir holding one."""
        weights_path = source / WEIGHTS_FILE if source.is_dir() else source
        return load_file(weights_path)

    @staticmethod
    def read_resume_state(resume_dir: Path) -> dict[str, Any]:
        """Read the ``state.pt`` resume blob (``{"rng": ..., "strategy_state": ...}``) from a checkpoint dir."""
        return torch.load(resume_dir / STATE_FILE, map_location="cpu", weights_only=False)

    # --- Writing ---

    def write(
        self,
        *,
        step: int,
        payload: CheckpointPayload,
        rng: dict[str, Tensor | None],
        wandb_run_id: str | None,
        fingerprint: Fingerprint,
        write_resume_state: bool,
    ) -> Path:
        """Write the envelope to a temp dir, atomically rename it into place, and prune old ones.
        Main process only. Returns the checkpoint directory. The trainer owns the surrounding
        collective (``collect_checkpoint`` on every rank, RNG capture, ``wait_for_everyone``).
        """
        step_dir = self.step_dir(step)
        tmp_dir = step_dir.parent / f".tmp_{step_dir.name}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        weights = {
            k: v.to(self._save_dtype).contiguous() if isinstance(v, Tensor) else v for k, v in payload.weights.items()
        }
        save_file(weights, tmp_dir / WEIGHTS_FILE, metadata=payload.weights_metadata or None)

        if write_resume_state:
            torch.save({"rng": rng, "strategy_state": payload.training_state}, tmp_dir / STATE_FILE)
            metadata = {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "global_step": step,
                "wandb_run_id": wandb_run_id,
                "resume_fingerprint": fingerprint,
            }
            (tmp_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2))

        if step_dir.exists():
            shutil.rmtree(step_dir)
        tmp_dir.rename(step_dir)

        self._saved_dirs = [saved_dir for saved_dir in self._saved_dirs if saved_dir != step_dir]
        self._saved_dirs.append(step_dir)
        self._prune()
        logger.info(f"💾 Checkpoint for step {step} saved in {step_dir.relative_to(self._output_dir)}")
        return step_dir

    @staticmethod
    def _legacy_checkpoint_message(example: Path) -> str:
        return (
            f"Found a pre-2.x flat checkpoint file in {example.parent} (e.g. '{example.name}'). "
            "These are incompatible with the directory-based checkpoint format "
            f"('checkpoints/step_NNNNN/{WEIGHTS_FILE}'). Point load_checkpoint at a new-format "
            "checkpoint directory, or start a fresh run."
        )

    def _resolve_dir(self, load: Path, *, no_resume: bool, expected_fingerprint: Fingerprint) -> ResumePlan:
        """Resolve resume intent for a checkpoint dir, a `checkpoints/` dir, or a run `output_dir`."""
        self.guard_legacy_checkpoint(load)
        checkpoints_subdir = load / "checkpoints"
        if checkpoints_subdir.is_dir():
            self.guard_legacy_checkpoint(checkpoints_subdir)

        step_dir = self.latest_step_dir(load)
        if step_dir is None or not (step_dir / WEIGHTS_FILE).exists():
            logger.warning(f"⚠️ No checkpoint directories found under {load}")
            return ResumePlan()

        state_path = step_dir / STATE_FILE
        metadata_path = step_dir / METADATA_FILE
        if no_resume or not state_path.exists() or not metadata_path.exists():
            logger.info(f"📥 Loading weights from {step_dir} (no resume)")
            return ResumePlan(weights_source=step_dir)

        try:
            metadata = json.loads(metadata_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                f"⚠️ Failed to read checkpoint metadata {metadata_path}: {e}. Loading weights and starting from step 0."
            )
            return ResumePlan(weights_source=step_dir)
        saved_fingerprint = metadata.get("resume_fingerprint", {})
        if saved_fingerprint != expected_fingerprint:
            logger.warning(
                f"⚠️ Resume fingerprint mismatch (saved {saved_fingerprint} → current {expected_fingerprint}). "
                "Loading weights and starting from step 0. Set checkpoints.no_resume=true to silence this warning."
            )
            return ResumePlan(weights_source=step_dir)

        step = metadata.get("global_step")
        if not isinstance(step, int) or step < 0:
            logger.warning(f"⚠️ Checkpoint metadata has invalid global_step={step!r}. Starting from step 0.")
            return ResumePlan(weights_source=step_dir)

        logger.info(f"📌 Resuming from step {step} ({step_dir})")
        return ResumePlan(step, step_dir, step_dir, metadata.get("wandb_run_id"))

    def _prune(self) -> None:
        """Remove checkpoint directories beyond ``keep_last_n`` (oldest first)."""
        keep_n = self._keep_last_n
        if 0 < keep_n < len(self._saved_dirs):
            for old_dir in self._saved_dirs[:-keep_n]:
                if old_dir.exists():
                    shutil.rmtree(old_dir)
                    logger.info(f"Removed old checkpoint: {old_dir}")
            self._saved_dirs = self._saved_dirs[-keep_n:]
