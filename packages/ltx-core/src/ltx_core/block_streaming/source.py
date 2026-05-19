"""Weight sources for block streaming: protocol and implementations."""

from __future__ import annotations

from collections import OrderedDict
from typing import Protocol

import torch

from ltx_core.block_streaming.disk import DiskBlockReader
from ltx_core.block_streaming.pool import WeightPool
from ltx_core.loader.primitives import TensorLayout


class WeightSource(Protocol):
    """Provides pinned CPU weights for a given block index.
    Assumes all buffers share an identical layout across all block indices.
    """

    @property
    def block_layout(self) -> TensorLayout:
        """Shared per-block buffer layout (shape + dtype for each param)."""
        ...

    def get(self, idx: int) -> dict[str, torch.Tensor]:
        """Return CPU weights for block *idx*."""
        ...

    def release(self, idx: int, event: torch.cuda.Event) -> None:
        """Signal that an async operation using these weights is guarded by *event*."""
        ...

    def cleanup(self) -> None:
        """Release all resources (buffers, readers, events)."""
        ...


class DiskWeightSource(WeightSource):
    """Reads block weights from disk into pinned CPU buffers on demand."""

    def __init__(self, pool: WeightPool, reader: DiskBlockReader) -> None:
        self._pool = pool
        self._cache: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()
        self._events: dict[int, torch.cuda.Event] = {}
        self._reader = reader

    @property
    def block_layout(self) -> TensorLayout:
        return self._pool.buffer_layout

    def get(self, idx: int) -> dict[str, torch.Tensor]:
        """Return CPU weights for block *idx*. Reads from disk on miss."""
        if idx in self._cache:
            return self._cache[idx]

        if len(self._cache) >= self._pool.capacity:
            evicted_idx, evicted_weights = self._cache.popitem(last=False)
            self._pool.release(evicted_weights, event=self._events.pop(evicted_idx, None))

        weights = self._pool.acquire()
        self._reader.read_into(weights, idx)
        self._cache[idx] = weights
        return weights

    def release(self, idx: int, event: torch.cuda.Event) -> None:
        """Attach an H2D event -- waited before this buffer is recycled."""
        self._events[idx] = event

    def cleanup(self) -> None:
        """Clear cache and close the disk reader."""
        self._cache.clear()
        self._events.clear()
        self._reader.cleanup()

    def __len__(self) -> int:
        return len(self._cache)


class PinnedWeightSource(WeightSource):
    """Pre-loaded pinned CPU weights."""

    def __init__(self, weights: dict[int, dict[str, torch.Tensor]]) -> None:
        if not weights:
            raise ValueError("PinnedWeightSource requires at least one block")
        self._weights = weights

    @property
    def block_layout(self) -> TensorLayout:
        first_block = self._weights[min(self._weights)]
        return {name: (t.shape, t.dtype) for name, t in first_block.items()}

    def get(self, idx: int) -> dict[str, torch.Tensor]:
        return self._weights[idx]

    def release(self, idx: int, event: torch.cuda.Event) -> None:
        pass

    def cleanup(self) -> None:
        self._weights.clear()

    def __len__(self) -> int:
        return len(self._weights)
