from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypeVar

import torch

from ltx_pipelines.utils.helpers import cleanup_memory

_M = TypeVar("_M", bound=torch.nn.Module)


@contextmanager
def gpu_model(model: _M) -> Iterator[_M]:
    """Context manager that yields a model and releases its memory on exit.
    Moves all parameters and buffers to ``meta`` device on exit, which
    immediately releases the underlying storage on **both** GPU and CPU,
    then runs ``cleanup_memory()`` to reclaim fragmented CUDA memory.
    Usage::
        with gpu_model(build_encoder()) as encoder:
            ...  # use encoder — typed as the concrete class
        # GPU + CPU memory freed automatically
    """
    try:
        yield model
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()
        # .to("meta") releases storage for all parameters/buffers regardless
        # of their original device (CUDA or CPU).
        model.to("meta")
        cleanup_memory()
