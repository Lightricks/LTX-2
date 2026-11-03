"""Registry and typed config union support for training strategies."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Sequence
from functools import reduce
from operator import or_
from typing import Annotated, Any, Literal, TypeAlias, get_args, get_origin

from pydantic import Discriminator, Tag

from ltx_trainer.training_strategies.base_strategy import TrainingStrategy, TrainingStrategyConfigBase

StrategyEntry: TypeAlias = tuple[type[TrainingStrategyConfigBase], type[TrainingStrategy]]
STRATEGY_REGISTRY: dict[str, StrategyEntry] = {}


def register_strategy(
    name: str,
    config_cls: type[TrainingStrategyConfigBase],
) -> Callable[[type[TrainingStrategy]], type[TrainingStrategy]]:
    """Register a strategy class under a typed config discriminator name."""

    def decorator(strategy_cls: type[TrainingStrategy]) -> type[TrainingStrategy]:
        existing = STRATEGY_REGISTRY.get(name)
        if existing is not None:
            existing_config, existing_strategy = existing
            raise ValueError(
                f"Training strategy '{name}' is already registered by "
                f"{existing_strategy.__module__}.{existing_strategy.__qualname__} "
                f"with config {existing_config.__module__}.{existing_config.__qualname__}; "
                f"cannot register {strategy_cls.__module__}.{strategy_cls.__qualname__}."
            )

        allowed_names = _allowed_config_names(config_cls)
        if not allowed_names:
            raise ValueError(
                f"Training strategy '{name}' config {config_cls.__module__}.{config_cls.__qualname__} "
                "must declare its 'name' field as a Literal containing the registered name."
            )
        if name not in allowed_names:
            raise ValueError(
                f"Training strategy '{name}' cannot use config "
                f"{config_cls.__module__}.{config_cls.__qualname__}; "
                f"config name literal allows {allowed_names}."
            )

        folder_name = _strategy_folder_name(strategy_cls)
        if folder_name is not None and folder_name != name:
            raise ValueError(
                f"Training strategy '{name}' must be registered from "
                f"training_strategies/{name}, got training_strategies/{folder_name}."
            )

        STRATEGY_REGISTRY[name] = (config_cls, strategy_cls)
        return strategy_cls

    return decorator


def discover_strategy_modules(package_name: str, package_path: Sequence[str]) -> None:
    """Import strategy modules/packages so their decorators self-register."""
    excluded = {"base", "base_strategy", "registry"}
    for module_info in pkgutil.iter_modules(package_path, prefix=f"{package_name}."):
        short_name = module_info.name.rsplit(".", maxsplit=1)[-1]
        if short_name.startswith("_") or short_name in excluded:
            continue
        try:
            importlib.import_module(module_info.name)
        except Exception as exc:
            raise ImportError(f"Failed to import training strategy module {module_info.name}") from exc


def build_strategy_config_union() -> object:
    """Build a Pydantic discriminated union from registered strategy config classes."""
    if not STRATEGY_REGISTRY:
        raise RuntimeError("No training strategies registered")
    if len(STRATEGY_REGISTRY) == 1:
        return next(iter(STRATEGY_REGISTRY.values()))[0]

    tagged_configs = tuple(
        Annotated[config_cls, Tag(name)] for name, (config_cls, _strategy_cls) in STRATEGY_REGISTRY.items()
    )
    union = reduce(or_, tagged_configs)
    return Annotated[union, Discriminator(_get_strategy_discriminator)]


def get_strategy_class(name: str) -> type[TrainingStrategy]:
    """Return the registered strategy class for a strategy name."""
    try:
        return STRATEGY_REGISTRY[name][1]
    except KeyError as exc:
        available = ", ".join(sorted(STRATEGY_REGISTRY)) or "<none>"
        raise ValueError(f"Unknown training strategy '{name}'. Available strategies: {available}") from exc


def get_strategy_config_class(name: str) -> type[TrainingStrategyConfigBase]:
    """Return the registered strategy config class for a strategy name."""
    try:
        return STRATEGY_REGISTRY[name][0]
    except KeyError as exc:
        available = ", ".join(sorted(STRATEGY_REGISTRY)) or "<none>"
        raise ValueError(f"Unknown training strategy '{name}'. Available strategies: {available}") from exc


def _get_strategy_discriminator(value: dict[str, Any] | TrainingStrategyConfigBase) -> str:
    """Extract the Pydantic discriminator tag from raw or parsed strategy config."""
    if isinstance(value, dict):
        return str(value.get("name", ""))
    return value.name


def _allowed_config_names(config_cls: type[TrainingStrategyConfigBase]) -> tuple[str, ...]:
    name_field = config_cls.model_fields.get("name")
    if name_field is None:
        return ()
    annotation = name_field.annotation
    if get_origin(annotation) is Literal:
        return tuple(str(arg) for arg in get_args(annotation))
    return ()


def _strategy_folder_name(strategy_cls: type[TrainingStrategy]) -> str | None:
    module_parts = strategy_cls.__module__.split(".")
    try:
        strategy_root_index = module_parts.index("training_strategies")
    except ValueError:
        return None
    if len(module_parts) <= strategy_root_index + 1:
        return None
    return module_parts[strategy_root_index + 1]
