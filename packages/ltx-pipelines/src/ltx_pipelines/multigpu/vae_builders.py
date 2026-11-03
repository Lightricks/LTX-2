"""Multi-GPU VAE decoder builder.
Wrapping builder that produces a :class:`DistributedVideoDecoder`.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch.multiprocessing import Queue

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.registry import ModelRegistry, RegistryKeys
from ltx_core.loader.simple_builder import SimpleBuilder
from ltx_core.multigpu.vae.distributed_decoder import DistributedVideoDecoder
from ltx_core.tiling import TileCountConfig

if TYPE_CHECKING:
    from typing_extensions import Self


class DistributedDecoderBuilder:
    """Builder that wraps a base decoder builder with distributed logic.
    Forwards ``module_ops`` / ``with_module_ops`` to the inner builder so
    pipeline flags like ``diffvae_optimization`` (``build_diffvae_mode_op``)
    still apply to the base ``DiffusionVideoDecoder`` before wrapping.
    """

    def __init__(
        self,
        inner: SimpleBuilder,
        queue: Queue,  # type: ignore[type-arg]
        vae_group: dist.ProcessGroup,
        vae_tiling: TileCountConfig,
        driver_rank: int,
        registry: ModelRegistry,
        num_temporal_batches: int | None = None,
    ) -> None:
        self._inner = inner.with_registry(registry)
        self._queue = queue
        self._vae_group = vae_group
        self._vae_tiling = vae_tiling
        self._driver_rank = driver_rank
        self._num_temporal_batches = num_temporal_batches

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
            num_temporal_batches=self._num_temporal_batches,
        )

    def registry_keys(self, dtype: torch.dtype | None = None) -> RegistryKeys:
        """Return the registry keys the inner decoder build at ``dtype`` queries and retains."""
        return self._inner.registry_keys(dtype)

    def model_config(self) -> dict[str, object]:
        """Read checkpoint configuration through the inner builder."""
        return self._inner.model_config()

    def with_module_ops(self, module_ops: tuple[ModuleOps, ...]) -> Self:
        clone = copy.copy(self)
        clone._inner = self._inner.with_module_ops(module_ops)
        return clone

    def with_tiling(self, vae_tiling: TileCountConfig) -> Self:
        """Return a new builder that splits decode with ``vae_tiling``.
        Functional: never mutates ``self``; shares the inner builder, queue, and
        process group. Mirrors stage ``with_*`` helpers so pipelines can swap
        count-based Dist tiling per request without holding one decoder per config.
        """
        clone = copy.copy(self)
        clone._vae_tiling = vae_tiling
        return clone

    @property
    def module_ops(self) -> tuple[ModuleOps, ...]:
        return self._inner.module_ops

    @property
    def keeps_gpu_resident_weights(self) -> bool:
        """Each build loads the decoder through the inner builder."""
        return False

    @property
    def builds_per_shape(self) -> bool:
        """A built decoder serves any latent shape."""
        return False
