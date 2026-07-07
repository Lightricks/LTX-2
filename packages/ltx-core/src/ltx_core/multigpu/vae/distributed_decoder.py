"""Distributed video decoder that partitions the latent across ranks.
Tiles are assigned to ranks via round-robin, so the number of tiles
may exceed the number of GPUs (e.g. 16 tiles on 4 GPUs = 4 tiles per
rank).  Each rank decodes its assigned tiles sequentially.  Workers
put their list of decoded tiles into a ``mp.Queue`` (CUDA IPC —
zero-copy handle sharing).  The driver collects all tiles, blends
overlap zones, and returns temporal batches distributed across devices.
The tiling configuration comes from ``MGPUConfig.vae_tiling`` (set at
construction time), NOT from the pipeline's SGPU tiling kwarg.  MGPU
tiling controls parallelism; SGPU tiling controls single-GPU VRAM
management — they are independent concerns.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from einops import rearrange
from torch.multiprocessing import Queue

from ltx_core.model.video_vae.tiling import TilingConfig
from ltx_core.model.video_vae.video_vae import (
    VideoDecoder,
    map_spatial_slice,
    map_temporal_slice,
    to_mapping_operation,
)
from ltx_core.tiling import (
    Tile,
    create_tiles,
    split_by_count,
    split_by_count_temporal_causal,
)
from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape

if TYPE_CHECKING:
    from ltx_core.tiling import TileCountConfig

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------


@dataclass(frozen=True)
class DecodedTile:
    """A VAE-decoded tile with pixel-space placement.
    Attributes:
        pixels: ``[F_tile, H_tile, W_tile, C]`` in the decoder's native dtype.
        pixel_tile: Carries ``out_coords`` (f, h, w slices) and ``blend_mask``.
    """

    pixels: torch.Tensor
    pixel_tile: Tile


# ------------------------------------------------------------------
# Tile construction helpers
# ------------------------------------------------------------------


def _to_decoded_tile(
    raw: torch.Tensor,
    tile: Tile,
) -> DecodedTile:
    """Convert raw decoder output ``[B, C, F, H, W]`` to a :class:`DecodedTile`.
    Rearranges to ``[F, H, W, C]`` and normalises ``[-1, 1] → [0, 1]``.
    """
    pixels = rearrange(raw[0], "c f h w -> f h w c")
    pixels = ((pixels + 1.0) / 2.0).clamp(0.0, 1.0)
    return DecodedTile(pixels=pixels, pixel_tile=tile)


# ------------------------------------------------------------------
# Tile assembly
# ------------------------------------------------------------------


def compute_summed_weights(
    tiles: list[DecodedTile],
    total_frames: int,
    output_height: int,
    output_width: int,
) -> torch.Tensor:
    """Build the ``[F, H, W]`` denominator for weighted blending."""
    weights = torch.zeros(total_frames, output_height, output_width)
    for tile in tiles:
        f_slice, h_slice, w_slice = tile.pixel_tile.out_coords
        weights[f_slice, h_slice, w_slice] += tile.pixel_tile.blend_mask
    return weights.clamp(min=1e-8)


def gather_frames(
    tiles: list[DecodedTile],
    total_frames: int,
    output_height: int,
    output_width: int,
    num_temporal_batches: int,
    world_size: int,
    weights: torch.Tensor,
    device_fn: Callable[[int], str | torch.device] | None = None,
) -> Iterator[torch.Tensor]:
    """Assemble decoded tiles into temporal batches distributed across GPUs.
    Each temporal batch is allocated on the device returned by *device_fn(batch_index)*.
    By default batches are placed round-robin on ``cuda:0`` … ``cuda:<world_size-1>``.
    """
    if device_fn is None:
        device_fn = lambda b: f"cuda:{b % world_size}"  # noqa: E731

    batch_size = (total_frames + num_temporal_batches - 1) // num_temporal_batches

    for b in range(num_temporal_batches):
        batch_range = slice(b * batch_size, min((b + 1) * batch_size, total_frames))
        batch_len = batch_range.stop - batch_range.start
        if batch_len <= 0:
            break

        device = device_fn(b)
        dtype = tiles[0].pixels.dtype
        output = torch.zeros(batch_len, output_height, output_width, 3, device=device, dtype=dtype)

        for tile in tiles:
            f_slice, h_slice, w_slice = tile.pixel_tile.out_coords

            overlap = slice(max(batch_range.start, f_slice.start), min(batch_range.stop, f_slice.stop))
            if overlap.start >= overlap.stop:
                continue

            tile_frames = slice(overlap.start - f_slice.start, overlap.stop - f_slice.start)
            out_frames = slice(overlap.start - batch_range.start, overlap.stop - batch_range.start)

            blend = tile.pixel_tile.blend_mask[tile_frames].to(device=device)
            output[out_frames, h_slice, w_slice, :] += tile.pixels[tile_frames].to(device=device) * blend[:, :, :, None]

        batch_weights = weights[batch_range.start : batch_range.stop].to(device=device)
        output.div_(batch_weights[:, :, :, None])
        yield output


# ------------------------------------------------------------------
# Main class
# ------------------------------------------------------------------


class DistributedVideoDecoder(torch.nn.Module):
    """Distributed VAE decoder with queue-based tile collection.
    All ranks decode their latent tile in parallel.  Workers send
    their :class:`DecodedTile` to the driver rank via the shared
    ``mp.Queue`` (CUDA IPC — zero-copy).  The driver collects all
    tiles, blends overlapping regions, and returns temporal batches
    as an iterator.
    Parameters
    ----------
    decoder:
        The real (local) ``VideoDecoder`` instance.
    queue:
        ``mp.Queue`` shared across all ranks for CUDA IPC tile transfer.
    vae_group:
        NCCL process group for the VAE ranks. Used to derive
        ``rank`` and ``world_size`` within the group.
    vae_tiling:
        MGPU tiling config that determines how the latent is split.
    driver_rank:
        Group-local rank of the driver process (the rank that collects
        and assembles tiles).
    """

    def __init__(
        self,
        decoder: VideoDecoder,
        queue: Queue,  # type: ignore[type-arg]
        vae_group: dist.ProcessGroup,
        vae_tiling: TileCountConfig,
        driver_rank: int = 0,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.queue = queue
        self.vae_group = vae_group
        self.rank = dist.get_rank(vae_group)
        self.world_size = dist.get_world_size(vae_group)
        self.vae_tiling = vae_tiling
        self.driver_rank = driver_rank

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Non-tiled path: fall back to local decode."""
        return self.decoder(sample, timestep, generator)

    def decode_video(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
        generator: torch.Generator | None = None,
        device_fn: Callable[[int], str | torch.device] | None = None,
    ) -> Iterator[torch.Tensor]:
        """Distributed decode — all ranks decode, driver assembles.
        Not a generator so that worker side-effects (decode + queue.put)
        execute eagerly regardless of whether the caller iterates.
        1. Each rank decodes its latent tile (with optional intra-GPU tiling).
        2. Workers send their :class:`DecodedTile` to the driver via the queue.
        3. The driver collects all tiles, blends overlaps, and returns
           temporal batches distributed across GPUs.
        """
        if (
            self.vae_tiling.frames.num_tiles > 1
            and tiling_config is not None
            and tiling_config.temporal_config is not None
        ):
            raise ValueError(
                "Cannot combine multi-GPU temporal tiling (vae_tiling.frames.num_tiles > 1) "
                "with single-GPU temporal tiling (tiling_config.temporal_config). "
                "Use only one to avoid causal decoding conflicts."
            )

        latent_shape = VideoLatentShape.from_torch_shape(latent.shape)
        scale = self.decoder.video_downscale_factors
        full_shape = latent_shape.upscale(scale)

        # Phase 1: each rank decodes its assigned tiles.
        my_tiles = self._decode_tiles(latent, latent_shape, scale, generator, tiling_config)

        # Phase 2: workers send tiles to driver.
        if self.rank != self.driver_rank:
            self.queue.put((self.rank, my_tiles))
            return iter([])

        # Phase 3: driver collects and assembles.
        all_tiles = self._collect_tiles(my_tiles)
        weights = compute_summed_weights(all_tiles, full_shape.frames, full_shape.height, full_shape.width)
        batches = gather_frames(
            all_tiles,
            full_shape.frames,
            full_shape.height,
            full_shape.width,
            self.world_size,
            self.world_size,
            weights,
            device_fn=device_fn,
        )
        return batches

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _decode_tiles(
        self,
        latent: torch.Tensor,
        latent_shape: VideoLatentShape,
        scale: SpatioTemporalScaleFactors,
        generator: torch.Generator | None,
        tiling_config: TilingConfig | None = None,
    ) -> list[DecodedTile]:
        """Decode this rank's assigned latent tiles and convert to :class:`DecodedTile` list."""
        all_tiles = create_tiles(
            torch.Size([latent_shape.frames, latent_shape.height, latent_shape.width]),
            splitters=[
                split_by_count_temporal_causal(self.vae_tiling.frames.num_tiles, self.vae_tiling.frames.overlap),
                split_by_count(self.vae_tiling.height.num_tiles, self.vae_tiling.height.overlap),
                split_by_count(self.vae_tiling.width.num_tiles, self.vae_tiling.width.overlap),
            ],
            mappers=[
                to_mapping_operation(map_temporal_slice, scale.time),
                to_mapping_operation(map_spatial_slice, scale.height),
                to_mapping_operation(map_spatial_slice, scale.width),
            ],
        )
        my_tiles = [t for i, t in enumerate(all_tiles) if i % self.world_size == self.rank]
        decoded = []
        for tile in my_tiles:
            latent_slice = latent[:, :, tile.in_coords[0], tile.in_coords[1], tile.in_coords[2]]
            if tiling_config is not None:
                chunks = list(self.decoder.tiled_decode(latent_slice, tiling_config, generator=generator))
                raw = torch.cat(chunks, dim=2)
            else:
                raw = self.decoder.forward(latent_slice, generator=generator)
            decoded.append(_to_decoded_tile(raw, tile))
        return decoded

    def _collect_tiles(self, driver_tiles: list[DecodedTile]) -> list[DecodedTile]:
        """Collect tiles from all workers via the queue. Returns flat list of all tiles.
        Sorted by rank so the downstream reduction in ``gather_frames`` /
        ``compute_summed_weights`` (in-place ``+=`` over overlapping pixel
        regions) processes tiles in a fixed order. Queue-arrival order would
        otherwise vary run-to-run and yield 1-ulp bf16 drift from
        non-associative floating-point summation.
        """
        per_rank: dict[int, list[DecodedTile]] = {self.driver_rank: driver_tiles}
        for _ in range(self.world_size - 1):
            worker_rank, worker_tiles = self.queue.get()
            per_rank[worker_rank] = worker_tiles
        result: list[DecodedTile] = []
        for rank in sorted(per_rank):
            result.extend(per_rank[rank])
        return result
