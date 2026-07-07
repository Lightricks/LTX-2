from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypeVar

import torch

from ltx_core.devices import synchronize_device
from ltx_pipelines.utils.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_pipelines.utils.helpers import cleanup_memory

_M = TypeVar("_M", bound=torch.nn.Module)


@contextmanager
def gpu_model(model: _M, alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM) -> Iterator[_M]:
    """Context manager that yields a model and releases its memory on exit.
    On ``TRIM`` (default): synchronize, move parameters/buffers to the ``meta``
    device (releasing GPU+CPU storage), then ``cleanup_memory()`` to return
    cached blocks to the OS. ``DEFER`` skips this -- the model's storage is
    reclaimed by normal GC and the CUDA caching allocator stays warm for the
    next build (cheaper for back-to-back runs).
    Usage::
        with gpu_model(build_encoder()) as encoder:
            ...  # use encoder -- typed as the concrete class
        # GPU + CPU memory freed automatically
    """
    try:
        yield model
    finally:
        if alloc_trim_strategy == AllocatorTrimStrategy.TRIM:
            synchronize_device()
            # .to("meta") releases storage for all parameters/buffers regardless
            # of their original device (CUDA or CPU).
            model.to("meta")
            cleanup_memory()
