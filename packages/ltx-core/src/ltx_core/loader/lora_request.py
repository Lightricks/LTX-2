"""Public LoRA request accepted by pipeline entry points."""

from typing import NamedTuple

from ltx_core.loader.sd_ops import SDOps


class LoraPathStrengthAndSDOps(NamedTuple):
    """One adapter checkpoint path, fusion strength, and key conversion."""

    path: str
    strength: float
    sd_ops: SDOps
