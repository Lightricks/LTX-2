"""GPU memory management utilities for training and inference."""

import functools
import gc
import subprocess
from typing import Callable, TypeVar

import torch

from ltx_core.devices import (
    cleanup_accelerator_memory,
    device_memory_allocated,
    device_memory_allocated_gb,
    device_memory_reserved,
    get_preferred_device,
)
from ltx_trainer import logger

F = TypeVar("F", bound=Callable)


def free_gpu_memory(log: bool = False) -> None:
    """Free accelerator memory by running garbage collection and emptying CUDA/MPS caches.
    Args:
        log: If True, log memory stats after clearing
    """
    gc.collect()
    cleanup_accelerator_memory()
    if log:
        device = get_preferred_device()
        allocated = device_memory_allocated(device) / 1024**3
        reserved = device_memory_reserved(device) / 1024**3
        logger.debug(f"Accelerator memory freed. Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


class free_gpu_memory_context:  # noqa: N801
    """Context manager and decorator to free GPU memory before and/or after execution.
    Can be used as a decorator:
        @free_gpu_memory_context(after=True)
        def my_function():
            ...
    Or as a context manager:
        with free_gpu_memory_context():
            heavy_operation()
    Args:
        before: Free memory before execution (default: False)
        after: Free memory after execution (default: True)
        log: Log memory stats when freeing (default: False)
    """

    def __init__(self, *, before: bool = False, after: bool = True, log: bool = False) -> None:
        self.before = before
        self.after = after
        self.log = log

    def __enter__(self) -> "free_gpu_memory_context":
        if self.before:
            free_gpu_memory(log=self.log)
        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: object) -> None:
        if self.after:
            free_gpu_memory(log=self.log)

    def __call__(self, func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> object:
            with self:
                return func(*args, **kwargs)

        return wrapper  # type: ignore


def get_gpu_memory_gb(device: torch.device) -> float:
    """Get current accelerator memory usage in GB.
    Args:
        device: torch.device to get memory usage for
    Returns:
        Current GPU memory usage in GB
    """
    if device.type != "cuda":
        return device_memory_allocated_gb(device)

    try:
        device_id = device.index if device.index is not None else 0
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,nounits,noheader",
                "-i",
                str(device_id),
            ],
            encoding="utf-8",
        )
        return float(result.strip()) / 1024  # Convert MB to GB
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        logger.debug(f"Failed to get GPU memory from nvidia-smi: {e}")
        # Fallback to torch
        return device_memory_allocated(device) / 1024**3
