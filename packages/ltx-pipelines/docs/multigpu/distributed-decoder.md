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

## Inter-GPU tiling vs intra-GPU tiling

Two independent tilings, commonly conflated:

| | Controls | Config | Set by |
| --- | --- | --- | --- |
| **Inter-GPU** (MGPU) | Which rank decodes which tile — **parallelism** | `vae_tiling: TileCountConfig` (at build time) | `DistributedDecoderBuilder` |
| **Intra-GPU** (SGPU) | Chunking *within* a rank's tile to bound **VRAM** | `tiling_config: TilingConfig` (per call) | the pipeline's usual tiling kwarg |

They compose — a rank can further chunk its assigned tile for VRAM — with one
guard: **multi-GPU temporal tiling and single-GPU temporal tiling cannot both be
on.** If `vae_tiling.frames.num_tiles > 1` and `tiling_config.temporal_config` is
set, `decode_video` raises, because two causal temporal splits would conflict.

## API

### `DistributedDecoderBuilder`

```python
from ltx_pipelines.multigpu.vae_builders import DistributedDecoderBuilder
from ltx_core.tiling import TileCountConfig, DimensionTilingConfig

DistributedDecoderBuilder(
    inner: BuilderProtocol,        # the block's single-GPU decoder builder
    queue: Queue,                  # spawn-context mp.Queue, shared across ranks (CUDA IPC)
    vae_group: dist.ProcessGroup,  # self.groups.vae_group
    vae_tiling: TileCountConfig,   # inter-GPU split
    driver_rank: int,              # rank that collects + assembles (usually 0)
    registry: Registry,
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
