"""Weights & Biases logging sink for the trainer.
Wraps a single optional ``wandb`` run behind a no-op-aware interface: when W&B is disabled or
the process is not the main process, the run is never created and every method is a cheap
no-op. This lets the trainer call ``log(...)`` / ``log_media(...)`` unconditionally instead of
guarding each call site with ``if main_process and run is not None``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import wandb

from ltx_trainer.config import WandbConfig
from ltx_trainer.training_strategies.types import MediaArtifact


class WandbRun:
    """Optional W&B run that no-ops when disabled or off the main process."""

    def __init__(
        self, config: WandbConfig, *, run_name: str, config_dump: dict[str, Any], is_main_process: bool
    ) -> None:
        self._config = config
        self._run_name = run_name
        self._config_dump = config_dump
        self._enabled = config.enabled and is_main_process
        self._run = None

    def start(self, resume_run_id: str | None = None) -> None:
        """Initialize the run (no-op when disabled), resuming an existing run when an id is given."""
        if not self._enabled:
            return
        init_kwargs: dict[str, Any] = {
            "project": self._config.project,
            "entity": self._config.entity,
            "name": self._run_name,
            "tags": self._config.tags,
            "config": self._config_dump,
        }
        if resume_run_id is not None:
            init_kwargs["id"] = resume_run_id
            init_kwargs["resume"] = "must"
        self._run = wandb.init(**init_kwargs)

    @property
    def run_id(self) -> str | None:
        """The active run's id (for checkpoint metadata), or None when inactive."""
        return self._run.id if self._run is not None else None

    def log(self, metrics: dict[str, float], *, step: int | None = None) -> None:
        """Log scalar metrics (no-op when inactive)."""
        if self._run is None:
            return
        if step is None:
            self._run.log(metrics)
        else:
            self._run.log(metrics, step=step)

    def log_media(self, media: list[MediaArtifact], *, step: int) -> None:
        """Log media artifacts, grouping by key so same-key artifacts form one gallery.
        No-op when inactive or when there is nothing to log. Each artifact's ``kind`` selects the
        W&B media type; callers set ``kind`` correctly at creation.
        """
        if self._run is None or not media:
            return
        grouped = defaultdict(list)
        for artifact in media:
            grouped[artifact.key].append(self._to_wandb_media(artifact))
        self._run.log(grouped, step=step)

    @staticmethod
    def _to_wandb_media(artifact: MediaArtifact) -> object:
        """Convert a MediaArtifact to its W&B media object via a simple image/video/audio branch."""
        path = str(artifact.path)
        if artifact.kind == "image":
            return wandb.Image(path, caption=artifact.caption)
        if artifact.kind == "audio":
            return wandb.Audio(path, caption=artifact.caption)
        return wandb.Video(path, format=artifact.path.suffix.lstrip("."), caption=artifact.caption)

    def finish(self) -> None:
        """Finish the run if active (idempotent)."""
        if self._run is not None:
            self._run.finish()
            self._run = None
