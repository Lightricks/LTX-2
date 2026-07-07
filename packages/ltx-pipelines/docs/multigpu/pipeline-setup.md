# Setting Up an MGPU Pipeline

**Source**: [`ti2vid_two_stages_mgpu.py`](../../src/ltx_pipelines/ti2vid_two_stages_mgpu.py), [`multigpu/weight_tracker.py`](../../src/ltx_pipelines/multigpu/weight_tracker.py)

An MGPU pipeline **is** a single-GPU pipeline with its per-block builders swapped
for MGPU builders. Build the standard pipeline, then replace each block's
`_transformer_builder` / `_text_encoder_builder` / `_decoder_builder`.

## The pattern

This is performed inside a runner's `setup()` (which runs on every rank — see
[Controller](controller.md)).

```python
from ltx_core.loader.registry import StateDictRegistry
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.multigpu.sp_builder import SequenceParallelBuilder
from ltx_pipelines.multigpu.tdp_builder import TiledDataParallelBuilder
from ltx_pipelines.multigpu.gemma_builders import AccelerateGemmaBuilder
from ltx_pipelines.multigpu.vae_builders import DistributedDecoderBuilder
from ltx_pipelines.multigpu.weight_tracker import TransformerWeightTracker
from ltx_core.multigpu.transformer.attention import AttentionManager

# 1. ONE shared registry for every builder in this process.
registry = StateDictRegistry()

# 2. Build the normal pipeline, handing it the registry.
pipeline = TI2VidTwoStagesPipeline(
    checkpoint_path=..., distilled_lora=..., spatial_upsampler_path=...,
    gemma_root=..., loras=[], registry=registry, quantization=...,
)

# 3. One weight tracker per transformer process group (shared by the stages).
tracker = TransformerWeightTracker(group=self.groups.transformer_group)

# 4. Swap each block's builder.
pipeline.stage_1._transformer_builder = SequenceParallelBuilder(
    inner=pipeline.stage_1._transformer_builder, attn_mgr=attn_mgr,
    registry=registry, tracker=tracker,
)
pipeline.stage_2._transformer_builder = TiledDataParallelBuilder(
    inner=pipeline.stage_2._transformer_builder, group=self.groups.transformer_group,
    tiling=tdp_tiling, registry=registry, tracker=tracker,
)
pipeline.prompt_encoder._text_encoder_builder = AccelerateGemmaBuilder(...)
pipeline.video_decoder._decoder_builder = DistributedDecoderBuilder(...)
```

Each MGPU builder **wraps** the block's existing single-GPU builder (`inner=...`),
so it inherits the checkpoint path, quantization, compilation, and LoRA config —
only the parallelism is added. See the per-technique pages for each builder's
constructor.

> **`with_builder` vs direct assignment.** `DiffusionStage.with_builder(builder)`
> returns a *new* stage with the builder swapped (functional, never mutates). The
> runners assign `stage._transformer_builder = ...` directly because they mutate the
> pipeline once, in place, during `setup()`. Both reach the same builder slot.

## The shared weights registry

`StateDictRegistry` is an in-process cache of loaded state dicts, keyed by
`(resolved paths, sd_ops name)`. Passing **one** registry to every builder means:

- The transformer checkpoint is read from disk **once per process**, even though
  stage 1 (SP) and stage 2 (TDP) are separate builders on the same file.
- Gemma and the VAE cache their weights the same way (rebuild the module tree from
  the cached tensors, skip disk I/O).

The registry is **per process** — it is not shared across ranks. Each worker loads
its own copy, so the full checkpoint is resident on every GPU (see the
[memory disclaimer](README.md)).

## `TransformerWeightTracker` — working copy + sharded clean weights

```python
TransformerWeightTracker(group: dist.ProcessGroup, bucket_mb=256, no_lora_swap=False)
```

The tracker is shared by the transformer stage builders that operate on the same
checkpoint. It does **not** own weights — it references the tensors in the registry
and receives a builder at `build()` time. Two copies of the weights exist per rank,
and they are **not** the same shape of memory:

- **Working copy** — the model the builder returns, backed by the registry's
  tensors. This is a **full replica on every GPU**. LoRAs are fused into it
  **in place**; `broadcast_sd` (a zero-copy `ShardedSD` view over these tensors)
  broadcasts each owner rank's freshly fused shards — bucketed, `bucket_mb` at a
  time — so all ranks converge on identical working weights.
- **Clean weights** (`stored_sd`) — an immutable, cloned backup of the original
  (pre-LoRA) weights, held so the working copy can be reset before a different LoRA
  set is applied. This copy is **sharded** across ranks (deterministic
  `md5(key) % world_size` ownership): each rank stores only its ~1/world_size slice,
  not a full clone.

So per-GPU transformer memory is one full working model **plus** a ~1/world_size
clean-weights shard — the clean backup is distributed, the working copy is not.

This allows a two-stage pipeline to apply the distilled LoRA to stage 2 and reset it
for stage 1 without reloading the checkpoint. Pass `no_lora_swap=True` when the
LoRA set is fixed (none, or one set for the whole run): the clean-weights clone is
skipped (`stored_sd` becomes a zero-copy view) and any swap/reset raises — saves the
~1/N shard, and guards against accidental swaps.

## Full example

The shipped runners are the reference: read
[`ti2vid_two_stages_mgpu.py`](../../src/ltx_pipelines/ti2vid_two_stages_mgpu.py)
(`setup()` lines ~54–132) and
[`distilled_mgpu.py`](../../src/ltx_pipelines/distilled_mgpu.py). Each ends with a
`__main__` block wiring the runner into an `MGPUController` behind the standard
two-stage CLI parser.
