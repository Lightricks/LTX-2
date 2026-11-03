"""Flexible training strategy package.
Importing this package registers :class:`FlexibleStrategy` with the strategy registry
(via the ``@register_strategy`` decorator in ``strategy.py``). The public surface is just the
strategy and its top-level config; the conditioning/modality detail configs are imported
directly from ``flexible.config`` by the callers that need them.
"""

from ltx_trainer.training_strategies.flexible.config import FlexibleStrategyConfig
from ltx_trainer.training_strategies.flexible.strategy import FlexibleStrategy

__all__ = [
    "FlexibleStrategy",
    "FlexibleStrategyConfig",
]
