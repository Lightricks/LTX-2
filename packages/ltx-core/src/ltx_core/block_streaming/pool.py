"""Weight buffer pool for block streaming."""

from __future__ import annotations

from collections import deque
from typing import Callable

import torch

from ltx_core.block_streaming.utils import allocate_layout_views
from ltx_core.loader.primitives import TensorLayout


class WeightPool:
    """Fixed pool of pre-allocated weight buffers with event-based reuse.
    All slots share a single buffer (CPU or GPU); each slot is a
    contiguous slice carved out of it via :func:`allocate_layout_views`.
    Args:
        buffer_layout: ``{name: (shape, dtype)}`` for each buffer.
        capacity: Number of buffers to pre-allocate.
        device: Device for allocation.
        reuse_barrier: Called with the pending event before a buffer is reused.
        pin_memory: Pin buffers (for async H2D copies from CPU).
    """

    def __init__(
        self,
        buffer_layout: TensorLayout,
        capacity: int,
        device: torch.device,
        reuse_barrier: Callable[[torch.cuda.Event], None],
        pin_memory: bool = False,
    ) -> None:
        self._buffer_layout = buffer_layout
        self._capacity = capacity
        self._free: deque[dict[str, torch.Tensor]] = deque()
        self._events: dict[int, torch.cuda.Event] = {}
        self._reuse_barrier = reuse_barrier
        memory_layout = {
            _make_key(slot, name): (shape, dtype)
            for slot in range(capacity)
            for name, (shape, dtype) in buffer_layout.items()
        }
        all_views = allocate_layout_views(memory_layout, device=device, pin_memory=pin_memory)
        for slot in range(capacity):
            self._free.append({name: all_views[_make_key(slot, name)] for name in buffer_layout})

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def buffer_layout(self) -> TensorLayout:
        return self._buffer_layout

    def acquire(self) -> dict[str, torch.Tensor]:
        """Take a free buffer, waiting any pending event before returning."""
        weights = self._free.popleft()
        event = self._events.pop(id(weights), None)
        if event is not None:
            self._reuse_barrier(event)
        return weights

    def release(self, weights: dict[str, torch.Tensor], event: torch.cuda.Event | None = None) -> None:
        """Return a buffer to the free list.
        If *event* is given it is waited on the next :meth:`acquire`
        of this buffer, ensuring the prior operation has completed.
        """
        if event is not None:
            self._events[id(weights)] = event
        self._free.append(weights)


def _make_key(slot: int, name: str) -> str:
    return f"{slot}/{name}"
