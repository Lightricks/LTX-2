# Distributed VAE Decoder

**Source**: [`ltx_core/multigpu/vae/distributed_decoder.py`](../../../ltx-core/src/ltx_core/multigpu/vae/distributed_decoder.py), [`multigpu/vae_builders.py`](../../src/ltx_pipelines/multigpu/vae_builders.py)

## What it is

VAE decode is expensive and embarrassingly parallel over space/time tiles. The
distributed decoder splits the latent into tiles, assigns them **round-robin** to
ranks (the tile count may exceed the GPU count), and every rank decodes its tiles in
parallel.
Workers ship their decoded tiles to the **driver rank** over an `mp.Queue` (CUDA
IPC — zero-copy handle sharing); the driver blends overlaps and yields the assembled
frames as temporal batches spread across the GPUs.

`DistributedVideoDecoder` wraps any `VideoDecoder` — both the conv
`ConvVideoDecoder` and the diffusion `DiffusionVideoDecoder` — by calling the
public `decode_video` API per MGPU tile (already `[F, H, W, C]` in `[0, 1]`).
It does **not** apply a second `[-1, 1] → [0, 1]` map. Full-video assembly is
`gather_frames` on the driver; any `torch.cat` of decode yields is only for
stitching that tile’s optional SGPU temporal chunks.

## Inter-GPU tiling vs intra-GPU tiling

Two independent tilings, commonly conflated:

| | Controls | Config | Set by |
| --- | --- | --- | --- |
| **Inter-GPU** (MGPU) | Which rank decodes which tile — **parallelism** | `vae_tiling: TileCountConfig` (at build time) | `DistributedDecoderBuilder` |
| **Intra-GPU** (SGPU) | Chunking *within* a rank's tile to bound **VRAM** | `tiling_config: TilingConfig` (per call) | the pipeline's usual tiling kwarg |

They compose — a rank can further chunk its assigned tile for VRAM — with one
guard: **multi-GPU temporal tiling and single-GPU temporal tiling cannot both
actually split the temporal axis.** If `vae_tiling.frames.num_tiles > 1` and
SGPU `tiling_config.video_chunks_number(tile_frames) > 1` for a rank's tile
(i.e. SGPU would produce more than one temporal chunk on that extent),
`decode_video` raises, because two causal temporal splits would conflict. A
size-based SGPU config such as `frames=80/24` on a short tile is fine when it
still yields a single chunk.

## API

### `DistributedDecoderBuilder`

```python
from ltx_pipelines.multigpu.vae_builders import DistributedDecoderBuilder
from ltx_core.tiling import TileCountConfig, DimensionTilingConfig

DistributedDecoderBuilder(
    inner: SimpleBuilder,          # the block's decoder builder
    queue: Queue,                  # spawn-context mp.Queue, shared across ranks (CUDA IPC)
    vae_group: dist.ProcessGroup,  # self.groups.vae_group
    vae_tiling: TileCountConfig,   # inter-GPU split
    driver_rank: int,              # rank that collects + assembles (usually 0)
    registry: ModelRegistry,
    num_temporal_batches: int | None = None,
)
```

`build()` returns a `DistributedVideoDecoder`; its `decode_video(latent,
tiling_config=None, ...)` returns an iterator of temporal batches on the driver, and
an empty iterator on workers (they only `put` their tiles).

## Usage

```python
# The queue is created once, in the CLI __main__, and passed to controller.start(...)
# as a setup kwarg so every worker shares it:
vae_queue = torch.multiprocessing.get_context("spawn").SimpleQueue()

# inside runner.setup():
vae_tiling = TileCountConfig(height=DimensionTilingConfig(num_tiles=8, overlap=4))
pipeline.video_decoder._decoder_builder = DistributedDecoderBuilder(
    inner=pipeline.video_decoder._decoder_builder,
    queue=vae_queue,
    vae_group=self.groups.vae_group,
    vae_tiling=vae_tiling,
    driver_rank=0,
    registry=registry,
)
```

The shipped runners tile the VAE across **height** (8 tiles, overlap 4). Only the
driver ends up with the assembled video, which is why the runner's `__call__`
encodes the file on `driver_rank` and yields `None` on the others.
