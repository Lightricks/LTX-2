"""Training strategies for different conditioning modes.
This package implements the Strategy Pattern to handle different training modes.
Each strategy encapsulates the specific logic for preparing model inputs and computing loss.
"""

from ltx_trainer import logger
from ltx_trainer.training_strategies.base_strategy import TrainingStrategy, TrainingStrategyConfigBase
from ltx_trainer.training_strategies.registry import (
    build_strategy_config_union,
    discover_strategy_modules,
    get_strategy_class,
)

discover_strategy_modules(__name__, __path__)

TrainingStrategyConfig = build_strategy_config_union()

__all__ = [
    "TrainingStrategy",
    "TrainingStrategyConfig",
    "TrainingStrategyConfigBase",
    "get_training_strategy",
]


def get_training_strategy(config: TrainingStrategyConfig) -> TrainingStrategy:
    """Factory function to create the appropriate training strategy.
    The strategy is determined by the `name` field in the configuration.
    Args:
        config: Strategy-specific configuration with a `name` field.
    Returns:
        The appropriate training strategy instance.
    Raises:
        ValueError: If strategy name is not supported.
    """

    strategy_cls = get_strategy_class(config.name)
    strategy = strategy_cls(config)

    logger.debug(f"Using {strategy.__class__.__name__} training strategy")
    return strategy
