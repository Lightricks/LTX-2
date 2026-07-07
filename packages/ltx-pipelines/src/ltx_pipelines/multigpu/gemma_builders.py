"""Multi-GPU Gemma text encoder builder.
Replaces the text encoder builder on the ``PromptEncoder`` block with an
:class:`AccelerateGemmaBuilder` that uses ``device_map="auto"`` on the
source rank and a broadcast stub elsewhere.
On the source rank the first ``build()`` loads via HuggingFace
``from_pretrained`` and caches the full state dict (including non-persistent
buffers) in the provided :class:`Registry`.  Subsequent calls recreate the
model from cache and reinstall accelerate dispatch hooks — no disk I/O.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from accelerate import dispatch_model
from transformers import Gemma3ForConditionalGeneration

from ltx_core.loader.primitives import BuilderProtocol, StateDict
from ltx_core.loader.registry import Registry
from ltx_core.multigpu.gemma.accelerate_wrapper import AccelerateGemmaWrapper
from ltx_core.multigpu.gemma.loader import load_gemma_with_device_map
from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoder

if TYPE_CHECKING:
    from typing_extensions import Self

logger = logging.getLogger(__name__)


class AccelerateGemmaBuilder(BuilderProtocol):
    """Builder that loads Gemma with ``device_map="auto"`` on the source rank.
    Conforms to the builder interface expected by ``PromptEncoder``:
    ``build(device, dtype) -> model``.  Non-source ranks get a lightweight
    :class:`AccelerateGemmaWrapper` that receives embeddings via broadcast.
    """

    def __init__(
        self,
        gemma_root_path: str,
        gemma_group: dist.ProcessGroup | None,
        broadcast_group: dist.ProcessGroup | None,
        registry: Registry,
        *,
        src_rank: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self._gemma_root_path = gemma_root_path
        self._gemma_group = gemma_group
        self._broadcast_group = broadcast_group
        self._registry = registry
        self._src_rank = src_rank
        self._is_src = dist.get_rank() == src_rank
        self._dtype = dtype
        # Cached on the src rank after first build (non-tensor objects).
        self._config: object | None = None
        self._hf_device_map: dict[str, int | str] | None = None
        self._tokenizer: object | None = None
        self._processor: object | None = None

    @property
    def registry(self) -> Registry:
        return self._registry

    def with_registry(self, registry: Registry) -> Self:
        clone = copy.copy(self)
        clone._registry = registry
        return clone

    def model_config(self) -> dict:
        return {}

    def build(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **_kwargs: Any,  # noqa: ANN401
    ) -> AccelerateGemmaWrapper:
        dtype = dtype or self._dtype

        encoder = self._build_encoder(dtype) if self._is_src else None

        return AccelerateGemmaWrapper(
            encoder=encoder,
            broadcast_group=self._broadcast_group,
            src_rank=dist.get_group_rank(self._broadcast_group, self._src_rank),
            dtype=dtype,
            device=device,
        )

    # -- src-rank helpers ---------------------------------------------------

    def _build_encoder(self, dtype: torch.dtype) -> GemmaTextEncoder:
        cached = self._registry.get([self._gemma_root_path], None)
        if cached is not None:
            logger.info("Rebuilding Gemma from cached state dict (no disk I/O).")
            return self._rebuild_from_cache(cached, dtype)

        encoder = load_gemma_with_device_map(self._gemma_root_path, dtype)

        # Cache non-tensor objects on the builder instance.
        self._config = encoder.model.config
        self._hf_device_map = encoder.model.hf_device_map
        self._tokenizer = encoder.tokenizer
        self._processor = encoder.processor

        # Cache full state dict including non-persistent buffers.
        sd = encoder.model.state_dict()
        for name, buf in encoder.model.named_buffers():
            if name not in sd:
                sd[name] = buf
        total_size = sum(t.nelement() * t.element_size() for t in sd.values())
        dtypes = {t.dtype for t in sd.values()}
        self._registry.add(
            [self._gemma_root_path],
            None,
            StateDict(sd=sd, device=torch.device("meta"), size=total_size, dtype=dtypes),
        )
        logger.info("Cached Gemma state dict in registry (%d entries).", len(sd))

        return encoder

    def _rebuild_from_cache(self, cached: StateDict, dtype: torch.dtype) -> GemmaTextEncoder:
        with torch.device("meta"):
            model = Gemma3ForConditionalGeneration(self._config)

        # Split into persistent (load_state_dict) and non-persistent (manual assign).
        expected_keys = set(model.state_dict().keys())
        persistent_sd = {k: v for k, v in cached.sd.items() if k in expected_keys}
        non_persistent_sd = {k: v for k, v in cached.sd.items() if k not in expected_keys}

        model.load_state_dict(persistent_sd, strict=True, assign=True)
        for name, tensor in non_persistent_sd.items():
            parent_path, attr = name.rsplit(".", 1)
            module = model
            for part in parent_path.split("."):
                module = getattr(module, part)
            setattr(module, attr, tensor)

        dispatch_model(model, self._hf_device_map)

        return GemmaTextEncoder(
            model=model,
            tokenizer=self._tokenizer,
            processor=self._processor,
            dtype=dtype,
        )
