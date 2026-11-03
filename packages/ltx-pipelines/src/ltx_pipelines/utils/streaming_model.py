"""Context manager for unit-streaming model builds."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.loader.streaming_builder import StreamingBuilder
from ltx_core.loader.streaming_model import StreamingModel
from ltx_pipelines.utils.helpers import cleanup_memory


@contextmanager
def streaming_model(
    builder: StreamingBuilder,
    target_device: torch.device,
    dtype: torch.dtype,
    alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
) -> Iterator[StreamingModel]:
    """Build a streamed model, yield it, then close its hooks and worker threads.
    ``close`` also replaces bound storage with meta tensors. ``TRIM`` runs
    ``cleanup_memory()``; ``DEFER`` leaves the allocator cache warm.
    """
    wrapped = builder.build(device=target_device, dtype=dtype)
    try:
        yield wrapped
    finally:
        wrapped.close()
        if alloc_trim_strategy == AllocatorTrimStrategy.TRIM:
            cleanup_memory()
