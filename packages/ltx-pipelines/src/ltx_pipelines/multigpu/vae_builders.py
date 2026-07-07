"""Multi-GPU VAE decoder builder.
Wrapping builder that produces a :class:`DistributedVideoDecoder`.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch.multiprocessing import Queue

from ltx_core.loader.primitives import BuilderProtocol
from ltx_core.loader.registry import Registry
from ltx_core.multigpu.vae.distributed_decoder import DistributedVideoDecoder
from ltx_core.tiling import TileCountConfig

if TYPE_CHECKING:
    from typing_extensions import Self


class DistributedDecoderBuilder(BuilderProtocol):
    """Builder that wraps a base decoder builder with distributed logic."""

    def __init__(
        self,
        inner: BuilderProtocol,
        queue: Queue,  # type: ignore[type-arg]
        vae_group: dist.ProcessGroup,
        vae_tiling: TileCountConfig,
        driver_rank: int,
        registry: Registry,
    ) -> None:
        self._inner = inner.with_registry(registry)
        self._queue = queue
        self._vae_group = vae_group
        self._vae_tiling = vae_tiling
        self._driver_rank = driver_rank

    @property
    def registry(self) -> Registry:
        return self._inner.registry

    def with_registry(self, registry: Registry) -> Self:
        clone = copy.copy(self)
        clone._inner = self._inner.with_registry(registry)
        return clone

    def build(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> DistributedVideoDecoder:
        base_decoder = self._inner.build(device=device, dtype=dtype, **kwargs)
        return DistributedVideoDecoder(
            base_decoder,
            queue=self._queue,
            vae_group=self._vae_group,
            vae_tiling=self._vae_tiling,
            driver_rank=dist.get_group_rank(self._vae_group, self._driver_rank),
        )
