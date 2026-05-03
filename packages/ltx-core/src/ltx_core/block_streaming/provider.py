"""GPU weights provider for block streaming."""

from __future__ import annotations

from collections import OrderedDict

import torch

from ltx_core.block_streaming.disk import LoraSource
from ltx_core.block_streaming.pool import WeightPool
from ltx_core.block_streaming.source import WeightSource


class WeightsProvider:
    """Provides GPU-ready block weights via H2D copy from a pinned CPU weight source.
    Args:
        pool: Pre-allocated GPU weight buffer pool.
        copy_stream: Dedicated CUDA stream for async H2D copies.
        target_device: GPU device for compute.
        source: Pinned CPU weight source.
        lora_sources: LoRA adapters fused on H2D copy.
        blocks_prefix: State-dict prefix for LoRA key matching.
    """

    def __init__(
        self,
        pool: WeightPool,
        copy_stream: torch.cuda.Stream,
        target_device: torch.device,
        source: WeightSource,
        lora_sources: list[LoraSource] | None = None,
        blocks_prefix: str = "",
    ) -> None:
        self._copy_stream = copy_stream
        self._pool = pool
        self._cache: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()
        self._events: dict[int, torch.cuda.Event] = {}
        self._target_device = target_device
        self._source = source
        self._lora_sources = lora_sources or []
        self._blocks_prefix = blocks_prefix

    def get(self, idx: int) -> dict[str, torch.Tensor]:
        """Return GPU weights for block *idx*. Does H2D copy on miss."""
        if idx in self._cache:
            return self._cache[idx]

        # Evict oldest GPU buffer if at capacity.
        if len(self._cache) >= self._pool.capacity:
            evicted_idx, evicted_weights = self._cache.popitem(last=False)
            self._pool.release(evicted_weights, event=self._events.pop(evicted_idx, None))

        gpu_weights = self._pool.acquire()
        cpu_weights = self._source.get(idx)

        h2d_event = self._copy_to_gpu(idx, gpu_weights, cpu_weights)
        self._source.release(idx, event=h2d_event)

        self._cache[idx] = gpu_weights
        return gpu_weights

    def _copy_to_gpu(
        self,
        idx: int,
        gpu_weights: dict[str, torch.Tensor],
        cpu_weights: dict[str, torch.Tensor],
    ) -> torch.cuda.Event:
        """Enqueue H2D copy + LoRA fusion on the copy stream and wait on compute.
        The wait is intentionally inside this method so callers -- and
        instrumentation regions wrapping it -- observe the full transfer time.
        """
        with torch.cuda.stream(self._copy_stream):
            for name, gpu_tensor in gpu_weights.items():
                gpu_tensor.copy_(cpu_weights[name], non_blocking=True)
            if self._lora_sources:
                self._fuse_block_loras(idx, gpu_weights)
            h2d_event = torch.cuda.Event()
            h2d_event.record(self._copy_stream)

        torch.cuda.current_stream(self._target_device).wait_event(h2d_event)
        return h2d_event

    def release(self, idx: int, event: torch.cuda.Event) -> None:
        """Attach a compute-done event -- waited before this buffer is recycled."""
        self._events[idx] = event

    def cleanup(self) -> None:
        """Synchronize streams and release all resources."""
        self._copy_stream.synchronize()
        torch.cuda.current_stream(self._target_device).synchronize()
        self._cache.clear()
        self._events.clear()
        self._source.cleanup()
        for lora in self._lora_sources:
            lora.cleanup()

    def __len__(self) -> int:
        return len(self._cache)

    def _fuse_block_loras(self, idx: int, weights: dict[str, torch.Tensor]) -> None:
        """Fuse LoRA deltas directly into GPU block weights."""
        for name, tensor in weights.items():
            if not name.endswith(".weight"):
                continue
            full_key = f"{self._blocks_prefix}.{idx}.{name}"
            prefix = full_key[: -len(".weight")]
            for source in self._lora_sources:
                delta = source.get_delta(prefix, device=self._target_device)
                if delta is not None:
                    tensor.add_(delta.to(dtype=tensor.dtype))
