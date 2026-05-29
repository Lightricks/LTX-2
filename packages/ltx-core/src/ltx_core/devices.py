from __future__ import annotations

import gc
import logging
from collections.abc import Iterator
from contextlib import contextmanager

import torch

DeviceSpec = str | int | torch.device | None

ACCELERATOR_DEVICE_TYPES = frozenset({"cuda", "mps"})


def is_mps_available() -> bool:
    """Return whether PyTorch can use the Apple Metal/MPS backend."""
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend is not None and mps_backend.is_available())


def get_preferred_device(local_rank: int | None = None) -> torch.device:
    """Prefer CUDA, then MPS, then CPU.

    ``local_rank`` is only meaningful for CUDA multi-process launches. MPS exposes
    a single logical device in PyTorch, so rank-based indexing is not used there.
    """
    if torch.cuda.is_available():
        index = torch.cuda.current_device() if local_rank is None else local_rank
        return torch.device("cuda", index)
    if is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(device: DeviceSpec = None, *, local_rank: int | None = None) -> torch.device:
    """Resolve ``None``/``auto`` to the best available accelerator."""
    if device is None:
        return get_preferred_device(local_rank=local_rank)
    if isinstance(device, int):
        return torch.device("cuda", device)
    if isinstance(device, str):
        if device.lower() in {"auto", "accelerator", "gpu"}:
            return get_preferred_device(local_rank=local_rank)
        return torch.device(device)
    return device


def is_accelerator_device(device: DeviceSpec) -> bool:
    return resolve_device(device).type in ACCELERATOR_DEVICE_TYPES


def synchronize_device(device: DeviceSpec = None) -> None:
    """Synchronize CUDA or MPS work if the selected backend supports it."""
    if device is None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if is_mps_available() and hasattr(torch, "mps"):
            torch.mps.synchronize()
        return

    resolved = resolve_device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(resolved)
    elif resolved.type == "mps" and is_mps_available() and hasattr(torch, "mps"):
        torch.mps.synchronize()


def empty_device_cache(device: DeviceSpec = None) -> None:
    """Release cached allocator memory for CUDA or MPS."""
    if device is None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if is_mps_available() and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        return

    resolved = resolve_device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif resolved.type == "mps" and is_mps_available() and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def cleanup_accelerator_memory(device: DeviceSpec = None) -> None:
    """Run Python GC and release CUDA/MPS allocator caches."""
    gc.collect()
    empty_device_cache(device)
    synchronize_device(device)
    try:
        if hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()
    except Exception:
        logging.warning("Host empty cache cleanup failed; ignoring.", exc_info=True)


def device_memory_allocated(device: DeviceSpec) -> int:
    resolved = resolve_device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.memory_allocated(resolved)
    if resolved.type == "mps" and is_mps_available() and hasattr(torch, "mps"):
        return torch.mps.current_allocated_memory()
    return 0


def device_memory_reserved(device: DeviceSpec) -> int:
    resolved = resolve_device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.memory_reserved(resolved)
    if resolved.type == "mps" and is_mps_available() and hasattr(torch, "mps"):
        if hasattr(torch.mps, "driver_allocated_memory"):
            return torch.mps.driver_allocated_memory()
        return torch.mps.current_allocated_memory()
    return 0


def device_memory_allocated_gb(device: DeviceSpec) -> float:
    return device_memory_allocated(device) / 1024**3


def get_accelerator_rng_state(device: DeviceSpec = None) -> torch.Tensor | None:
    resolved = resolve_device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_rng_state(resolved)
    if resolved.type == "mps" and is_mps_available() and hasattr(torch, "mps"):
        return torch.mps.get_rng_state()
    return None


def set_accelerator_rng_state(state: torch.Tensor | None, device: DeviceSpec = None) -> None:
    if state is None:
        return
    resolved = resolve_device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_rng_state(state, resolved)
    elif resolved.type == "mps" and is_mps_available() and hasattr(torch, "mps"):
        torch.mps.set_rng_state(state)


@contextmanager
def fork_device_rng(device: DeviceSpec = None) -> Iterator[None]:
    """Temporarily fork CPU plus selected CUDA/MPS RNG state."""
    resolved = resolve_device(device)
    cpu_state = torch.random.get_rng_state()
    accelerator_state = get_accelerator_rng_state(resolved)
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        set_accelerator_rng_state(accelerator_state, resolved)
