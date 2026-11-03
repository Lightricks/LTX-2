"""Shared optimizer and LR-scheduler construction for training strategies.
Plain factory functions over primitive arguments (no config objects), so any strategy can build a
standard optimizer/scheduler pair without depending on a particular config schema.
"""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    LRScheduler,
    PolynomialLR,
    StepLR,
)


def create_optimizer(
    trainable_params: list[torch.nn.Parameter],
    *,
    optimizer_type: str,
    learning_rate: float,
) -> torch.optim.Optimizer:
    """Build an optimizer over ``trainable_params``. Supported: ``adamw``, ``adamw8bit``."""
    if optimizer_type == "adamw":
        return AdamW(trainable_params, lr=learning_rate)

    if optimizer_type == "adamw8bit":
        from bitsandbytes.optim import AdamW8bit  # noqa: PLC0415

        return AdamW8bit(trainable_params, lr=learning_rate)

    raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_type: str | None,
    total_steps: int,
    scheduler_params: dict | None = None,
) -> LRScheduler | None:
    """Build an LR scheduler for the named type, or ``None`` for ``constant``/no schedule.
    Supported: ``linear``, ``cosine``, ``cosine_with_restarts``, ``polynomial``, ``step``.
    ``scheduler_params`` supplies optional overrides for that scheduler's keyword arguments.
    """
    params = dict(scheduler_params or {})
    if scheduler_type is None or scheduler_type == "constant":
        return None

    if scheduler_type == "linear":
        return LinearLR(
            optimizer,
            start_factor=params.pop("start_factor", 1.0),
            end_factor=params.pop("end_factor", 0.1),
            total_iters=total_steps,
            **params,
        )

    if scheduler_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=params.pop("eta_min", 0), **params)

    if scheduler_type == "cosine_with_restarts":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=params.pop("T_0", total_steps // 4),
            T_mult=params.pop("T_mult", 1),
            eta_min=params.pop("eta_min", 5e-5),
            **params,
        )

    if scheduler_type == "polynomial":
        return PolynomialLR(optimizer, total_iters=total_steps, power=params.pop("power", 1.0), **params)

    if scheduler_type == "step":
        return StepLR(
            optimizer,
            step_size=params.pop("step_size", total_steps // 2),
            gamma=params.pop("gamma", 0.1),
            **params,
        )

    raise ValueError(f"Unknown scheduler type: {scheduler_type}")
