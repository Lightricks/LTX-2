# Tiled Data Parallelism (TDP)

**Source**: [`ltx_core/multigpu/transformer/tiled_data_parallel.py`](../../../ltx-core/src/ltx_core/multigpu/transformer/tiled_data_parallel.py), [`multigpu/tdp_builder.py`](../../src/ltx_pipelines/multigpu/tdp_builder.py)

## What it is

TDP splits the patchified `(frames, height, width)` latent into **tiles** and gives
each tile to a GPU. Every rank runs the full transformer on its own tile(s),
overlapping regions are blended with trapezoidal masks, and a single `all_reduce`
sums the blended tiles into the final result (masks sum to 1 globally). Tiles are
assigned **round-robin**, so the tile count may exceed the GPU count (16 tiles on
4 GPUs = 4 tiles/rank). Audio is processed untiled on every tile forward and averaged
across tiles.

Unlike [sequence parallelism](sequence-parallel.md), TDP is **not** bit-faithful to
single-GPU: each tile is denoised with only local context and blended, so it is an
approximation.

> ## ⚠️ Do not use the TDP stage's audio output
>
> Audio is **not** tiled. It is denoised on **every** tile's forward pass — each with
> a different, partial video context — and those results are **averaged across all
> tiles**. That average is not a meaningful audio latent. Take the final audio from
> the first (SP) stage and keep it frozen through the TDP upscale; treat the TDP
> stage's audio only as the video-conditioning context it needs internally, never as
> output.

## When to use it

TDP is an **upscaler**. It produces video at resolutions the model never saw during
training by running each tile at a resolution the model handles well and blending the
results. This is why the shipped two-stage runner uses TDP for **stage 2** (the
high-resolution upscale).

TDP can also be **faster** than running the whole frame on one GPU: self-attention is
quadratic in the token count, so splitting `N` tokens into `T` tiles drops per-tile
attention cost from `O(N^2)` to `O((N/T)^2)`.

> **Do not run TDP as the first stage.** Starting from pure noise (a high first
> sigma), each tile denoises independently and produces **unrelated content** — the
> tiles never converge on a single coherent video. Generate the first stage with
> [SP](sequence-parallel.md) (faithful, full-frame), then **upscale** that result
> with TDP.
>
> Even as the upscale stage, tiles can **drift** apart, and the drift grows with the
> **first sigma** of the TDP stage (more noise re-injected means more freedom per
> tile). For consistency, either condition on the stage-1 result with
> **negative-index image conditioning** (for i2v), or use a **smaller first sigma**.

## Position normalization

A tile's tokens must carry positions in the range the model was trained on — not the
global positions of a tile in the corner of a large frame, which the model was never
trained to handle. With `normalize_positions=True` (default), each tile's positions
are shifted so the tile's **generated** tokens start at zero in every dimension:

```python
offset = gen_pos[..., 0].amin(dim=2, keepdim=True)...  # min start per (batch, dim)
positions = positions - offset                          # shift generated + conditioning tokens
```

Interval widths are preserved (only the origin moves), so RoPE sees a valid,
in-distribution position grid per tile.

## Shared negative (reference) positions

Conditioning tokens are appended after the generated tokens. A tile keeps a
conditioning token when its `[start, end)` interval overlaps the tile in all three
dimensions — **or** when it has a **negative time coordinate**. Negative-time tokens
are **reference tokens** (e.g. reference frames / audio references): they are kept by
**every** tile so all tiles share the same reference context. To avoid
double-counting a token kept by several tiles, its blend weight is `1 / (number of
tiles that kept it)`.

## API

### Tiling config (`ltx_core.tiling`)

```python
from ltx_core.tiling import TileCountConfig, DimensionTilingConfig

TileCountConfig(
    frames: DimensionTilingConfig = DimensionTilingConfig(num_tiles=1, overlap=0),
    height: DimensionTilingConfig = DimensionTilingConfig(num_tiles=1, overlap=0),
    width:  DimensionTilingConfig = DimensionTilingConfig(num_tiles=1, overlap=0),
)
DimensionTilingConfig(num_tiles: int, overlap: int = 0)   # counts, not sizes; overlap in latent grid units
```

`TileCountConfig` specifies tile **counts** per dimension (contrast the single-GPU
VAE `TilingConfig`, which specifies tile **sizes**).

### `TiledDataParallelBuilder`

```python
from ltx_pipelines.multigpu.tdp_builder import TiledDataParallelBuilder

TiledDataParallelBuilder(
    inner: ModelBuilderProtocol,   # the stage's single-GPU transformer builder
    group: dist.ProcessGroup,      # self.groups.transformer_group
    tiling: TileCountConfig,
    registry: Registry,
    tracker: TransformerWeightTracker,
    normalize_positions: bool = True,
)
```

Wraps a `SingleGPUModelBuilder`. Its `build()` requires a `video_tools` kwarg (the
`VideoLatentTools` for the target shape) so the wrapper can compute tiles — the
pipeline passes this through automatically.

## Usage

```python
# inside runner.setup(), stage 2 -- balanced 2D spatial (height x width) grid over the group:
from ltx_core.tiling import TileCountConfig, DimensionTilingConfig, balanced_tile_split

h_tiles, w_tiles = balanced_tile_split(dist.get_world_size(self.groups.transformer_group))
tdp_tiling = TileCountConfig(
    height=DimensionTilingConfig(num_tiles=h_tiles, overlap=5),
    width=DimensionTilingConfig(num_tiles=w_tiles, overlap=5),
)
pipeline.stage_2._transformer_builder = TiledDataParallelBuilder(
    inner=pipeline.stage_2._transformer_builder,
    group=self.groups.transformer_group,
    tiling=tdp_tiling,
    registry=registry,
    tracker=tracker,
)
```
