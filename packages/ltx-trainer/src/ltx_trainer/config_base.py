"""Dependency-free base types for self-describing config classes.
Lives at the trainer level and imports nothing from ``ltx_trainer`` so every config module
(trainer config, shared strategy configs, validation config, strategy config bases) can import
these without creating an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict


@dataclass(frozen=True)
class ConfigSection:
    """A titled group of display rows. Config classes return these; the renderer formats values."""

    title: str
    rows: list[tuple[str, object]]  # (label, raw value); the renderer applies formatting


class ConfigBase(BaseModel):
    """Base for trainer config models: forbids unknown fields and can describe itself for display."""

    model_config = ConfigDict(extra="forbid")

    def describe(self) -> list[ConfigSection]:
        """Return display sections for this config. Default: none."""
        return []
