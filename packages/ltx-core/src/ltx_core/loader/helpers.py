"""Shared model-construction helpers used by both SingleGPUModelBuilder and StreamingModelBuilder."""

from __future__ import annotations

from typing import TypeVar

import torch
from torch import nn

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import StateDict, StateDictLoader
from ltx_core.loader.registry import Registry
from ltx_core.loader.sd_ops import SDOps
from ltx_core.model.model_protocol import ModelConfigurator

_M = TypeVar("_M", bound=nn.Module)


def load_state_dict(
    paths: str | tuple[str, ...] | list[str],
    loader: StateDictLoader,
    registry: Registry,
    device: torch.device | None,
    sd_ops: SDOps | None = None,
) -> StateDict:
    """Load a state dict from disk, using registry caching."""
    if isinstance(paths, str):
        path_list = [paths]
    elif isinstance(paths, tuple):
        path_list = list(paths)
    else:
        path_list = paths
    cached = registry.get(path_list, sd_ops)
    if cached is not None:
        return cached
    result = loader.load(path_list, sd_ops=sd_ops, device=device)
    registry.add(path_list, sd_ops=sd_ops, state_dict=result)
    return result


def read_model_config(
    model_path: str | tuple[str, ...],
    loader: StateDictLoader,
) -> dict:
    """Read metadata from the first shard of a checkpoint."""
    first = model_path[0] if isinstance(model_path, tuple) else model_path
    return loader.metadata(first)


def create_meta_model(
    configurator: type[ModelConfigurator[_M]],
    config: dict,
    module_ops: tuple[ModuleOps, ...] = (),
) -> _M:
    """Create a model on the meta device and apply module operations."""
    with torch.device("meta"):
        model = configurator.from_config(config)
    for op in module_ops:
        if op.matcher(model):
            model = op.mutator(model)
    return model
