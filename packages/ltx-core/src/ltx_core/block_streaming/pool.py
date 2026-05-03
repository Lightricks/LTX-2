"""Weight buffer pool for block streaming."""

from __future__ import annotations

from collections import deque
from typing import Callable

import torch

from ltx_core.block_streaming.utils import allocate_buffer

# Type alias for the buffer layout used by slot allocation.
BlockLayout = dict[str, tuple[torch.Size, torch.dtype]]


class WeightPool:
    """Fixed pool of pre-allocated weight buffers with event-based reuse safety.
    Buffers are allocated once at construction.  :meth:`acquire` pops a
    free buffer (waiting any pending event first).  :meth:`release`
    returns it, optionally attaching an event that must complete before
    the buffer can be reused.
    Args:
        layout: ``{name: (shape, dtype)}`` for each buffer.
        capacity: Number of buffers to pre-allocate.
        device: Device for allocation.
        reuse_barrier: Called with the pending event before a buffer is reused.
        pin_memory: Pin buffers (for async H2D copies from CPU).
    """

    def __init__(
        self,
        layout: BlockLayout,
        capacity: int,
        device: torch.device,
        reuse_barrier: Callable[[torch.cuda.Event], None],
        pin_memory: bool = False,
    ) -> None:
        self._capacity = capacity
        self._free: deque[dict[str, torch.Tensor]] = deque()
        self._events: dict[int, torch.cuda.Event] = {}
        self._reuse_barrier = reuse_barrier
        for _ in range(capacity):
            self._free.append(allocate_buffer(layout, device, pin_memory))

    @property
    def capacity(self) -> int:
        return self._capacity

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
