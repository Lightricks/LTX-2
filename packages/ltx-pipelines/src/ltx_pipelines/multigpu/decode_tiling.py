"""Distributed-decode tile planning shared by the multi-GPU runners.
A Dist decode splits the latent over the ranks of a VAE group, and each rank decodes its share
with a :class:`TilingConfig` of its own. :func:`dist_auto_tiling` resolves ``AUTO_TILING`` to one
plan for the whole group: every rank plans the group's largest Dist tile from the group's
smallest budgets, so the plans, and whether the fallback budget is needed, agree on every rank,
cold or warm, without a further collective.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Protocol

import torch
import torch.distributed as dist

from ltx_core.devices import cuda_allocatable_bytes, resolve_device
from ltx_core.model.video_vae import AutoTiling, TilingConfig
from ltx_core.model.video_vae.transformer import DiffVAEMode
from ltx_core.multigpu.vae.distributed_decoder import dist_rank_tile_pixel_shape, dist_rank_tiles
from ltx_core.tiling import TileCountConfig
from ltx_core.types import VIDEO_SCALE_FACTORS, SpatioTemporalScaleFactors, VideoLatentShape, VideoPixelShape

# Share of device memory the decode plan may fill. The rest serves what runs beside the decode:
# streamed chunks waiting for their temporal batch, the batch being blended, and the encoder.
DECODE_LIVE_SHARE = 0.858
# How ensure_tiling_config's ValueError starts when no tile fits the budget.
_NO_TILE_FITS = "Cannot fit a DiffVAE decode tile"


class EnsureTilingConfig(Protocol):
    """The signature of :func:`ltx_pipelines.utils.helpers.ensure_tiling_config`."""

    def __call__(
        self,
        tiling_config: TilingConfig | AutoTiling | None,
        *,
        video_shape: VideoPixelShape,
        vae_checkpoint_path: str,
        diffvae_optimization: DiffVAEMode = DiffVAEMode.CHUNKED_EAGER,
        device: torch.device | None = None,
        free_bytes: int | None = None,
        keyframes: bool = False,
        vae_dtype: torch.dtype = torch.bfloat16,
    ) -> TilingConfig | None: ...


@dataclass(frozen=True)
class GroupDecodeBudget:
    """The smallest decode budgets over a VAE group, in bytes.
    Attributes:
        planned: ``DECODE_LIVE_SHARE`` of device memory minus allocated bytes. Allocated bytes do
            not depend on what the caching allocator holds, so this is the same cold or warm.
        fallback: Everything a new allocation can still get (:func:`cuda_allocatable_bytes`), for
            a clip the planned budget fits no tile for.
    """

    planned: int
    fallback: int


def group_decode_budget(device: torch.device | None, group: dist.ProcessGroup | None) -> GroupDecodeBudget:
    """Each rank's complete budgets, MIN-reduced over ``group``; one collective and one host sync."""
    resolved = resolve_device(device)
    total = torch.cuda.get_device_properties(resolved).total_memory
    planned = max(0, int(DECODE_LIVE_SHARE * total) - int(torch.cuda.memory_allocated(resolved)))
    budgets = torch.tensor([planned, cuda_allocatable_bytes(resolved)], dtype=torch.int64, device=resolved)
    dist.all_reduce(budgets, op=dist.ReduceOp.MIN, group=group)
    planned, fallback = budgets.tolist()
    return GroupDecodeBudget(planned=planned, fallback=fallback)


def dist_max_tile_pixel_shape(
    vae_tiling: TileCountConfig,
    video_shape: VideoPixelShape,
    world_size: int,
    scale: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS,
) -> VideoPixelShape:
    """:func:`dist_rank_tile_pixel_shape` of the rank with the largest Dist tile, the same on every rank."""
    latent_shape = VideoLatentShape.from_pixel_shape(video_shape, scale_factors=scale)
    shapes = [
        dist_rank_tile_pixel_shape(
            vae_tiling, latent_shape, scale, rank, world_size, fps=video_shape.fps, batch=video_shape.batch
        )
        for rank in range(world_size)
        if dist_rank_tiles(vae_tiling, latent_shape, scale, rank, world_size)
    ]
    return max(shapes, key=lambda shape: shape.frames * shape.height * shape.width)


def dist_auto_tiling(
    ensure: EnsureTilingConfig,
    vae_tiling: TileCountConfig,
    *,
    world_size: int,
    group: dist.ProcessGroup | None,
    scale: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS,
) -> EnsureTilingConfig:
    """Wrap ``ensure`` so ``AUTO_TILING`` without ``free_bytes`` plans one Dist tile for the group.
    The wrapper plans :func:`dist_max_tile_pixel_shape` from :func:`group_decode_budget`'s
    planned budget, and from its fallback budget if no tile fits that. Every rank joins the one
    collective and plans the same shape from the same numbers, so every rank resolves the same
    plan or raises the same error. Explicit configs, ``None`` and calls with ``free_bytes`` pass
    through unchanged.
    """

    @wraps(ensure)
    def ensure_dist_tile(
        tiling_config: TilingConfig | AutoTiling | None,
        *,
        video_shape: VideoPixelShape,
        vae_checkpoint_path: str,
        diffvae_optimization: DiffVAEMode = DiffVAEMode.CHUNKED_EAGER,
        device: torch.device | None = None,
        free_bytes: int | None = None,
        keyframes: bool = False,
        vae_dtype: torch.dtype = torch.bfloat16,
    ) -> TilingConfig | None:
        def plan(shape: VideoPixelShape, budget: int | None) -> TilingConfig | None:
            return ensure(
                tiling_config,
                video_shape=shape,
                vae_checkpoint_path=vae_checkpoint_path,
                diffvae_optimization=diffvae_optimization,
                device=device,
                free_bytes=budget,
                keyframes=keyframes,
                vae_dtype=vae_dtype,
            )

        if not isinstance(tiling_config, AutoTiling) or free_bytes is not None:
            return plan(video_shape, free_bytes)
        tile_shape = dist_max_tile_pixel_shape(vae_tiling, video_shape, world_size, scale)
        budget = group_decode_budget(device, group)
        try:
            return plan(tile_shape, budget.planned)
        except ValueError as exc:
            if not str(exc).startswith(_NO_TILE_FITS):
                raise
        return plan(tile_shape, budget.fallback)

    return ensure_dist_tile
