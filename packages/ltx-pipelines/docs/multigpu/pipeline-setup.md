# Setting Up an MGPU Pipeline

**Source**: [`ti2vid_two_stages_mgpu.py`](../../src/ltx_pipelines/ti2vid_two_stages_mgpu.py), [`ltx_core/loader/distributed.py`](../../../ltx-core/src/ltx_core/loader/distributed.py), [`ltx_core/loader/distributed_builders.py`](../../../ltx-core/src/ltx_core/loader/distributed_builders.py)

An MGPU pipeline **is** a single-GPU pipeline with its per-block builders swapped
for MGPU builders. Build the standard pipeline, then replace each block's
`_transformer_builder` / `_text_encoder_builder` / `_decoder_builder`.

## The pattern

This is performed inside a runner's `setup()` (which runs on every rank — see
[Controller](controller.md)).

```python
from ltx_core.loader.distributed_builders import (
    AccelerateGemmaBuilder,
    SequenceParallelBuilder,
    TiledDataParallelBuilder,
)
from ltx_core.multigpu.transformer.attention import AttentionManager
from ltx_pipelines.multigpu.runner import rank_cuda_registry
from ltx_pipelines.multigpu.vae_builders import DistributedDecoderBuilder
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

# 1. ONE shared registry for every builder in this process, keeping weights on this rank's GPU.
registry = rank_cuda_registry()

# 2. Build the normal pipeline, handing it the registry.
pipeline = TI2VidTwoStagesPipeline(
    model_paths=..., distilled_lora=..., spatial_upsampler_path=...,
    loras=[], registry=registry, quantization=...,
)

# 3. One set of active weights shared by the transformer stages.
active_weights: dict[str, torch.Tensor] = {}

# 4. Swap each block's builder.
pipeline.stage_1._transformer_builder = SequenceParallelBuilder(
    pipeline.stage_1._transformer_builder, attn_mgr, dst=active_weights
)
pipeline.stage_2._transformer_builder = TiledDataParallelBuilder(
    pipeline.stage_2._transformer_builder, self.groups.transformer_group, tdp_tiling, dst=active_weights
)
pipeline.prompt_encoder._text_encoder_builder = AccelerateGemmaBuilder.from_root(...)
pipeline.video_decoder._decoder_builder = DistributedDecoderBuilder(...)
```

Each MGPU builder **wraps** the block's existing `SimpleBuilder` (its `base`), so it
inherits the checkpoint paths, LoRA requests and registry; the stage still applies
quantization and compilation to that base. Only the parallelism is added. See the
per-technique pages for each builder's constructor.

> **`with_builder` vs direct assignment.** `DiffusionStage.with_builder(builder)`
> returns a *new* stage with the builder swapped (functional, never mutates). The
> runners assign `stage._transformer_builder = ...` directly because they mutate the
> pipeline once, in place, during `setup()`. Both reach the same builder slot.

## The shared model registry

`ModelRegistry` (`ltx_core.loader.registry`) is an in-process cache:

- **Model shells** (`cache_models`, default `True`) — meta shells keyed by
  configurator, checkpoint metadata and module ops. A hit reuses the shell; every
  build assigns weights onto it. Non-persistent buffers installed by module ops
  stay on the shell across `dispose`.
- **Checkpoint metadata** (`cache_metadata`, default `True`) — parsed safetensors
  `__metadata__` per resolved shard, and checkpoint projections per resolved paths
  and SDOps names, so a rebuild neither re-reads headers nor re-projects keys.
- **Unit states** — retained only where a placement policy places them. The
  registry's `placement_policy` is the default of every builder using it; a builder
  may override it with `with_placement_policy`.

The runners create their registry with `rank_cuda_registry()`
(`ltx_pipelines.multigpu.runner`): clean weights and LoRA factors stay on the rank's
GPU. Passing **one** registry to every builder means a checkpoint is read from disk
**once per process**, even though stage 1 (SP) and stage 2 (TDP) are separate
builders on the same file. Gemma and the VAE keep their weights the same way, except
the Accelerate Gemma builder, which keeps its own dispatched encoder and weights.

Retained states are read-only: LoRA fusion writes into separate active storage.

The registry is **per process** — it is not shared across ranks. Each worker loads
its own copy, so the full checkpoint is resident on every GPU (see the
[memory disclaimer](README.md)).

## Shared active weights — one replica, rank-owned clean units

The transformer stage builders share one `dst` dict. The first distributed build
fills it with a **full replica on every GPU**; every later build, from any stage and
with any LoRA set, rewrites those same tensors, so their addresses never change. The
registry records the selection each cached shell holds, so it knows what `dst`
holds. On a build:

- each rank loads only the units it owns, fuses the selected LoRAs into them, and
  writes them into `dst`;
- bucketed all-gather rounds, through a `bucket_bytes` staging buffer the registry
  keeps per device, copy every owner's units into the other ranks' replicas;
- a request `dst` already holds binds the stage's shell to it without loading,
  after one agreement between ranks;
- a LoRA switch rewrites only the weights the outgoing or incoming set patches
  (restoring outgoing ones from clean units, fusing incoming ones) and exchanges
  only those keys. Every stage model bound to `dst` then holds the new weights, so
  stage models must not run concurrently.

Clean units come from the registry. With `rank_cuda_registry()` each rank retains
the clean states of the units it owns, about 1/world_size of the checkpoint, so a
switch reads nothing from disk. Per-GPU transformer memory is therefore one full
replica **plus** a ~1/world_size clean share (and the LoRA factors).

This allows a two-stage pipeline to apply the distilled LoRA to stage 2 and remove
it for stage 1 without reloading the checkpoint. When the LoRA set is fixed for the
whole run (none, or one set), the clean share is unnecessary: give the transformer
builder a placement policy without clean retention, e.g.
`builder.with_placement_policy(RetentionPolicy(lora=rank_device))`; a switch still
works, but reads the rank's units from the checkpoint again.

## Full example

The shipped runners are the reference: read
[`ti2vid_two_stages_mgpu.py`](../../src/ltx_pipelines/ti2vid_two_stages_mgpu.py)
(`setup()` lines ~54–132),
[`distilled_mgpu.py`](../../src/ltx_pipelines/distilled_mgpu.py), and
[`dfr_mgpu.py`](../../src/ltx_pipelines/dfr_mgpu.py). Each ends with a
`__main__` block wiring the runner into an `MGPUController` behind the standard
two-stage CLI parser.
