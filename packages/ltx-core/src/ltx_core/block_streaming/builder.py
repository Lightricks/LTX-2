"""Builder that constructs a BlockStreamingWrapper from safetensors checkpoints."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Generic

import safetensors
import torch
from torch import nn

from ltx_core.block_streaming.disk import DiskBlockReader, DiskTensorReader, LoraSource
from ltx_core.block_streaming.pool import WeightPool
from ltx_core.block_streaming.provider import WeightsProvider
from ltx_core.block_streaming.source import DiskWeightSource, PinnedWeightSource, WeightSource
from ltx_core.block_streaming.utils import allocate_layout_views, derive_layout, make_block_key, resolve_attr
from ltx_core.block_streaming.wrapper import BlockStreamingWrapper
from ltx_core.loader.fuse_loras import aggregate_lora_products, fuse_lora_weights
from ltx_core.loader.helpers import create_meta_model, load_state_dict, read_model_config
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import (
    LoraPathStrengthAndSDOps,
    LoraStateDictWithStrength,
    ModelBuilderProtocol,
    StateDictLoader,
)
from ltx_core.loader.registry import DummyRegistry, Registry
from ltx_core.loader.sd_ops import SDOps
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
from ltx_core.model.model_protocol import ModelConfigurator, ModelType

logger = logging.getLogger(__name__)

DISK_CPU_SLOTS = 2
_DEFAULT_GPU_SLOTS = 2


@dataclass(frozen=True)
class StreamingModelBuilder(Generic[ModelType], ModelBuilderProtocol[ModelType]):
    """Immutable builder for :class:`BlockStreamingWrapper`.
    Reads block weights from safetensors on demand.  ``cpu_slots`` and
    ``gpu_slots`` control the memory/speed trade-off (see :meth:`build`).
    Args:
        model_class_configurator: Creates the model from a config dict.
        model_path: One or more ``.safetensors`` checkpoint paths.
        model_sd_ops: Key remapping applied to safetensors keys.
        module_ops: Module-level mutations for the meta model.
        loras: LoRA adapters fused into weights at load time.
        model_loader: Strategy for reading checkpoint metadata.
        registry: Shared cache for loaded state dicts.
        blocks_attr: Dotted path to the ``nn.ModuleList`` (e.g.
            ``"velocity_model.transformer_blocks"``).
        blocks_prefix: State-dict key prefix for block weights
            (e.g. ``"transformer_blocks"``).
        state_dict_prefix: Wrapper offset prepended to keys when loading into
            the meta model (e.g. ``"velocity_model."`` when wrapped by ``X0Model``).
        model_wrapper: Optional callable wrapping the model
            (e.g. ``X0Model``).
    """

    model_class_configurator: type[ModelConfigurator[ModelType]]
    model_path: str | tuple[str, ...]
    model_sd_ops: SDOps | None = None
    module_ops: tuple[ModuleOps, ...] = field(default_factory=tuple)
    loras: tuple[LoraPathStrengthAndSDOps, ...] = field(default_factory=tuple)
    model_loader: StateDictLoader = field(default_factory=SafetensorsModelStateDictLoader)
    registry: Registry = field(default_factory=DummyRegistry)

    # Streaming-specific
    blocks_attr: str = ""
    blocks_prefix: str = ""
    state_dict_prefix: str = ""
    model_wrapper: Callable[[ModelType], nn.Module] | None = None

    def with_sd_ops(self, sd_ops: SDOps | None) -> StreamingModelBuilder:
        return replace(self, model_sd_ops=sd_ops)

    def with_module_ops(self, module_ops: tuple[ModuleOps, ...]) -> StreamingModelBuilder:
        return replace(self, module_ops=module_ops)

    def with_loras(self, loras: tuple[LoraPathStrengthAndSDOps, ...]) -> StreamingModelBuilder:
        return replace(self, loras=loras)

    def model_config(self) -> dict:
        """Read model configuration from the checkpoint metadata."""
        return read_model_config(self.model_path, self.model_loader)

    def meta_model(self, config: dict, module_ops: tuple[ModuleOps, ...]) -> ModelType:
        """Create a model on the meta device and apply module operations."""
        return create_meta_model(self.model_class_configurator, config, module_ops)

    def build(
        self,
        target_device: torch.device,
        dtype: torch.dtype,
        cpu_slots_count: int | None = None,
        gpu_slots_count: int | None = None,
        **_kwargs: object,
    ) -> BlockStreamingWrapper:
        """Build and return a ready-to-use :class:`BlockStreamingWrapper`.
        Args:
            target_device: GPU device for compute.
            dtype: Weight dtype (e.g. ``torch.bfloat16``).
            cpu_slots_count: Number of pinned CPU buffer slots.
                ``None`` = RAM streaming (all blocks pre-loaded with LoRA fusion).
            gpu_slots_count: Number of GPU buffer slots.
                ``None`` = ``_DEFAULT_GPU_SLOTS`` (2).
        """
        if not self.blocks_prefix:
            raise ValueError("blocks_prefix must be non-empty for streaming")

        config = read_model_config(self.model_path, self.model_loader)
        meta_model: nn.Module = create_meta_model(self.model_class_configurator, config, self.module_ops)
        if self.model_wrapper is not None:
            meta_model = self.model_wrapper(meta_model)
        meta_model.eval()

        blocks = resolve_attr(meta_model, self.blocks_attr)

        checkpoint_paths = list(self.model_path) if isinstance(self.model_path, tuple) else [self.model_path]
        block_key_map, non_block_keys = _scan_checkpoint_keys(checkpoint_paths, self.model_sd_ops, self.blocks_prefix)

        cpu_slots_count = cpu_slots_count if cpu_slots_count is not None else len(blocks)
        gpu_slots_count = gpu_slots_count if gpu_slots_count is not None else _DEFAULT_GPU_SLOTS

        if cpu_slots_count >= len(blocks):
            source, lora_sources = self._build_pinned_source(
                meta_model, target_device, dtype, cpu_slots_count, block_key_map, non_block_keys
            )
        else:
            reader = DiskTensorReader(checkpoint_paths)
            source, lora_sources = self._build_disk_source(
                meta_model, target_device, dtype, cpu_slots_count, reader, block_key_map, non_block_keys
            )

        copy_stream = torch.cuda.Stream(device=target_device)
        gpu_pool = WeightPool(
            source.block_layout,
            gpu_slots_count,
            target_device,
            reuse_barrier=lambda event: copy_stream.wait_event(event),
        )
        provider = WeightsProvider(gpu_pool, copy_stream, target_device, source, lora_sources, self.blocks_prefix)
        return BlockStreamingWrapper(
            model=meta_model,
            blocks=blocks,
            provider=provider,
            target_device=target_device,
        )

    def _build_pinned_source(
        self,
        meta_model: nn.Module,
        target_device: torch.device,
        dtype: torch.dtype,
        cpu_slots_count: int,
        block_key_map: dict[int, list[tuple[str, str]]],
        non_block_keys: list[tuple[str, str]],
    ) -> tuple[WeightSource, list[LoraSource]]:
        """Pre-load all blocks into pinned CPU buffers with LoRA fusion."""
        model_sd = load_state_dict(
            self.model_path, self.model_loader, self.registry, torch.device("cpu"), self.model_sd_ops
        )

        lora_sd_and_strengths = [
            LoraStateDictWithStrength(
                load_state_dict([lora.path], self.model_loader, self.registry, torch.device("cpu"), lora.sd_ops),
                lora.strength,
            )
            for lora in self.loras
        ]

        for block_idx in block_key_map:
            if block_idx >= cpu_slots_count:
                raise ValueError(
                    f"Pinned source requires one CPU slot per block; "
                    f"got block index {block_idx} with only {cpu_slots_count} slots."
                )

        blocks = resolve_attr(meta_model, self.blocks_attr)
        block_tensors: dict[str, torch.Tensor] = {}
        for block_idx, entries in block_key_map.items():
            block_params = dict(blocks[block_idx].named_parameters())
            for _sft_key, param_name in entries:
                key = make_block_key(self.blocks_prefix, block_idx, param_name)
                block_tensors[key] = block_params[param_name]
        blocks_layout = derive_layout(block_tensors, dtype)
        pinned_blocks = allocate_layout_views(blocks_layout, pin_memory=True)

        should_sync = False
        for key, fused in fuse_lora_weights(model_sd, lora_sd_and_strengths, dtype=None, preserve_input_device=False):
            if key in pinned_blocks:
                pinned_blocks[key].copy_(fused, non_blocking=True)
                model_sd.sd[key] = None
                should_sync = True
            else:
                model_sd.sd[key] = fused
        if should_sync:
            torch.cuda.synchronize()

        # Fill remaining pinned keys from the source state dict.
        for key in blocks_layout:
            if model_sd.sd[key] is None:
                continue
            pinned_blocks[key].copy_(model_sd.sd[key])
            model_sd.sd[key] = None

        pinned: dict[int, dict[str, torch.Tensor]] = {
            block_idx: {
                param_name: pinned_blocks[make_block_key(self.blocks_prefix, block_idx, param_name)]
                for _sft_key, param_name in entries
            }
            for block_idx, entries in block_key_map.items()
        }

        non_block_sd: dict[str, torch.Tensor] = {
            self.state_dict_prefix + model_key: model_sd.sd[model_key].to(device=target_device, dtype=dtype)
            for _sft_key, model_key in non_block_keys
        }

        meta_model.load_state_dict(non_block_sd, strict=False, assign=True)

        return PinnedWeightSource(pinned), []

    def _build_disk_source(
        self,
        meta_model: nn.Module,
        target_device: torch.device,
        dtype: torch.dtype,
        cpu_slots_count: int,
        reader: DiskTensorReader,
        block_key_map: dict[int, list[tuple[str, str]]],
        non_block_keys: list[tuple[str, str]],
    ) -> tuple[WeightSource, list[LoraSource]]:
        """Create a DiskWeightSource backed by a DiskBlockReader for lazy loading.
        Derives the shared pool layout from the meta model's block 0 — this
        relies on module_ops (e.g. fp8_cast) leaving the meta param dtype in
        sync with the post-sd_ops checkpoint dtype.
        """
        lora_sources = [LoraSource(lora.path, lora.sd_ops, lora.strength) for lora in self.loras]

        self._load_non_block_weights(
            reader,
            non_block_keys,
            meta_model,
            target_device,
            dtype,
            sd_ops=self.model_sd_ops,
            key_prefix=self.state_dict_prefix,
            lora_sources=lora_sources,
        )

        blocks = resolve_attr(meta_model, self.blocks_attr)
        layout = derive_layout(dict(blocks[0].named_parameters()), dtype)

        cpu_pool = WeightPool(
            layout,
            cpu_slots_count,
            torch.device("cpu"),
            reuse_barrier=lambda event: event.synchronize(),
            pin_memory=True,
        )
        block_reader = DiskBlockReader(
            reader=reader,
            block_key_map=block_key_map,
            sd_ops=self.model_sd_ops,
            blocks_prefix=self.blocks_prefix,
        )
        source = DiskWeightSource(cpu_pool, block_reader)
        return source, lora_sources

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fuse_lora_delta(
        model_key: str,
        tensor: torch.Tensor,
        lora_sources: list[LoraSource],
    ) -> torch.Tensor:
        """Add all matching LoRA deltas to *tensor* in-place via ``addmm_``."""
        if not lora_sources or not model_key.endswith(".weight"):
            return tensor
        prefix = model_key[: -len(".weight")]
        products = (
            ab
            for ab in (s.get_ab(prefix, device=tensor.device, dtype=tensor.dtype) for s in lora_sources)
            if ab is not None
        )
        aggregate_lora_products(products, out=tensor)
        return tensor

    @staticmethod
    @torch.inference_mode()
    def _load_non_block_weights(
        reader: DiskTensorReader,
        non_block_keys: list[tuple[str, str]],
        model: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
        sd_ops: SDOps | None = None,
        key_prefix: str = "",
        lora_sources: list[LoraSource] | None = None,
    ) -> None:
        """Load non-block weights into *model* on *device*."""
        state_dict: dict[str, torch.Tensor] = {}
        sources = lora_sources or []
        for sft_key, model_key in non_block_keys:
            tensor = reader.get_tensor(sft_key).to(device=device, dtype=dtype)
            tensor = StreamingModelBuilder._fuse_lora_delta(model_key, tensor, sources)
            if sd_ops is not None:
                for kv in sd_ops.apply_to_key_value(model_key, tensor):
                    state_dict[key_prefix + kv.new_key] = kv.new_value
                continue
            state_dict[key_prefix + model_key] = tensor
        model.load_state_dict(state_dict, strict=False, assign=True)


def _scan_checkpoint_keys(
    checkpoint_paths: list[str],
    sd_ops: SDOps | None,
    blocks_prefix: str,
) -> tuple[dict[int, list[tuple[str, str]]], list[tuple[str, str]]]:
    """Partition checkpoint keys into per-block and non-block lists.
    Opens the safetensors files for header-only key enumeration; no tensor data
    is read.
    """
    block_key_map: dict[int, list[tuple[str, str]]] = {}
    non_block_keys: list[tuple[str, str]] = []
    prefix_dot = blocks_prefix + "."
    for path in checkpoint_paths:
        with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
            for sft_key in handle.keys():  # noqa: SIM118
                model_key = sd_ops.apply_to_key(sft_key) if sd_ops else sft_key
                if model_key is None:
                    continue
                if model_key.startswith(prefix_dot):
                    rest = model_key[len(prefix_dot) :]
                    idx_str, _, param_name = rest.partition(".")
                    try:
                        block_idx = int(idx_str)
                    except ValueError:
                        non_block_keys.append((sft_key, model_key))
                        continue
                    block_key_map.setdefault(block_idx, []).append((sft_key, param_name))
                else:
                    non_block_keys.append((sft_key, model_key))
    return block_key_map, non_block_keys
