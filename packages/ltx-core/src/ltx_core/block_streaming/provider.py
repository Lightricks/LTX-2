"""GPU weights provider for block streaming."""

from __future__ import annotations

from collections import OrderedDict

import torch

from ltx_core.block_streaming.disk import LoraSource
from ltx_core.block_streaming.pool import WeightPool
from ltx_core.block_streaming.source import WeightSource
from ltx_core.devices import synchronize_device
from ltx_core.loader.fuse_loras import FuseRule, aggregate_lora_products, bf16_fuse_rule
from ltx_core.loader.primitives import StateDict

_EMPTY_STATE_DICT = StateDict(sd={}, device=torch.device("cpu"), size=0, dtype=set())


def _contiguous_byte_view(weights: dict[str, torch.Tensor]) -> torch.Tensor | None:
    """Return a ``uint8`` view spanning every tensor in *weights*, or ``None`` if
    they don't share one contiguous storage region."""
    tensors = list(weights.values())
    if not tensors:
        return None
    storage = tensors[0].untyped_storage()
    storage_ptr = storage.data_ptr()
    start = end = tensors[0].storage_offset() * tensors[0].element_size()
    for t in tensors:
        if t.untyped_storage().data_ptr() != storage_ptr or not t.is_contiguous():
            return None
        offset = t.storage_offset() * t.element_size()
        nbytes = t.numel() * t.element_size()
        start = min(start, offset)
        end = max(end, offset + nbytes)
    view = torch.empty(0, dtype=torch.uint8, device=tensors[0].device)
    view.set_(storage, start, (end - start,), (1,))
    return view


class WeightsProvider:
    """Provides accelerator-ready block weights via copies from a CPU weight source.
    Args:
        pool: Pre-allocated accelerator weight buffer pool.
        copy_stream: Dedicated CUDA stream for async H2D copies. ``None`` uses
            synchronous copies, which is used for MPS/CPU.
        target_device: Accelerator device for compute.
        source: CPU weight source.
        lora_sources: LoRA adapters fused after copying.
        blocks_prefix: State-dict prefix for LoRA key matching.
        fuse_rule: Per-policy LoRA merge rule (must be streaming-compatible:
            no companion-key emission). Defaults to ``bf16_fuse_rule``.
    """

    def __init__(
        self,
        pool: WeightPool,
        copy_stream: torch.cuda.Stream | None,
        target_device: torch.device,
        source: WeightSource,
        lora_sources: list[LoraSource] | None = None,
        blocks_prefix: str = "",
        fuse_rule: FuseRule = bf16_fuse_rule,
    ) -> None:
        self._copy_stream = copy_stream
        self._pool = pool
        self._cache: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()
        self._events: dict[int, object] = {}
        self._target_device = target_device
        self._source = source
        self._lora_sources = lora_sources or []
        self._blocks_prefix = blocks_prefix
        self._fuse_rule = fuse_rule

    def get(self, idx: int) -> dict[str, torch.Tensor]:
        """Return accelerator weights for block *idx*. Copies from CPU on miss."""
        if idx in self._cache:
            return self._cache[idx]

        # Evict oldest GPU buffer if at capacity.
        if len(self._cache) >= self._pool.capacity:
            evicted_idx, evicted_weights = self._cache.popitem(last=False)
            self._pool.release(evicted_weights, event=self._events.pop(evicted_idx, None))

        gpu_weights = self._pool.acquire()
        cpu_weights = self._source.get(idx)

        h2d_event = self._copy_to_device(idx, gpu_weights, cpu_weights)
        self._source.release(idx, event=h2d_event)

        self._cache[idx] = gpu_weights
        return gpu_weights

    def _copy_to_device(
        self,
        idx: int,
        gpu_weights: dict[str, torch.Tensor],
        cpu_weights: dict[str, torch.Tensor],
    ) -> object | None:
        """Copy weights to the target device and fuse LoRAs.
        The wait is intentionally inside this method so callers -- and
        instrumentation regions wrapping it -- observe the full transfer time.
        """
        if self._copy_stream is None:
            self._copy_weights(gpu_weights, cpu_weights, non_blocking=False)
            if self._lora_sources:
                self._fuse_block_loras(idx, gpu_weights)
            return None

        with torch.cuda.stream(self._copy_stream):
            self._copy_weights(gpu_weights, cpu_weights, non_blocking=True)
            if self._lora_sources:
                self._fuse_block_loras(idx, gpu_weights)
            h2d_event = torch.cuda.Event()
            h2d_event.record(self._copy_stream)

        torch.cuda.current_stream(self._target_device).wait_event(h2d_event)
        return h2d_event

    @staticmethod
    def _copy_weights(
        gpu_weights: dict[str, torch.Tensor],
        cpu_weights: dict[str, torch.Tensor],
        *,
        non_blocking: bool,
    ) -> None:
        gpu_view = _contiguous_byte_view(gpu_weights)
        cpu_view = _contiguous_byte_view(cpu_weights)
        if gpu_view is not None and cpu_view is not None and gpu_view.numel() == cpu_view.numel():
            gpu_view.copy_(cpu_view, non_blocking=non_blocking)
        else:
            for name, gpu_tensor in gpu_weights.items():
                gpu_tensor.copy_(cpu_weights[name], non_blocking=non_blocking)

    def release(self, idx: int, event: object | None = None) -> None:
        """Attach a compute-done event -- waited before this buffer is recycled."""
        if event is not None:
            self._events[idx] = event

    def cleanup(self) -> None:
        """Synchronize streams and release all resources."""
        if self._copy_stream is not None:
            self._copy_stream.synchronize()
            torch.cuda.current_stream(self._target_device).synchronize()
        else:
            synchronize_device(self._target_device)
        self._cache.clear()
        self._events.clear()
        self._source.cleanup()
        for lora in self._lora_sources:
            lora.cleanup()

    def __len__(self) -> int:
        return len(self._cache)

    def _fuse_block_loras(self, idx: int, weights: dict[str, torch.Tensor]) -> None:
        """Fuse LoRA deltas directly into GPU block weights via ``fuse_rule``."""
        agg_dtype = self._fuse_rule.aggregation_dtype
        for name, tensor in weights.items():
            if not name.endswith(".weight"):
                continue
            prefix = f"{self._blocks_prefix}.{idx}.{name}".removesuffix(".weight")
            products = (
                ab
                for ab in (s.get_ab(prefix, device=self._target_device, dtype=agg_dtype) for s in self._lora_sources)
                if ab is not None
            )
            deltas = aggregate_lora_products(products, agg_dtype)
            if deltas is None:
                continue
            fused = self._fuse_rule(name, tensor, deltas, _EMPTY_STATE_DICT)
            tensor.copy_(fused[name])
