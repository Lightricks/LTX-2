# Gemma Text Encoder (Multi-GPU)

**Source**: [`ltx_core/loader/distributed_builders.py`](../../../ltx-core/src/ltx_core/loader/distributed_builders.py)

Two ways to run the Gemma text encoder across the fleet. Both swap in for
`pipeline.prompt_encoder._text_encoder_builder` and broadcast the resulting
embeddings to every rank (so the transformer ranks all have them).

## `AccelerateGemmaBuilder` (Accelerate `device_map` — the default)

Loads Gemma **once, on the source rank**, with Accelerate `device_map="auto"`, which
**shards Gemma's layers across the available GPUs**. Non-source ranks receive a
lightweight `AccelerateGemmaWrapper` stub that receives the encoded embeddings over
NCCL. The source rank fuses all prompts into one Gemma call, then broadcasts each
output.

Gemma's shards share GPUs with the other ranks, so the non-source ranks must not
post their NCCL collectives while Gemma is still running: a receive kernel spins on
its GPU, and the two processes then time-slice it. Each builder therefore owns a
`BroadcastCoordinator` (`ltx_core.multigpu.broadcast`) on the default store and
hands it to every wrapper it builds. The source rank synchronizes the devices
holding Gemma's outputs and signals; the other ranks poll the store every 1 ms on
the host and only then join the collective. This gates the build and encode
agreements, each output broadcast and the `enhance_*` results. A waiting rank
raises `TimeoutError` after 10 minutes (NCCL's default collective timeout), e.g.
when the source rank failed before signaling. Construct the builders on every rank
in `setup()`, in the same order and before any job runs: each builder's store key
comes from its construction order, and copies of a builder keep its coordinator.

The first `build()` loads Gemma on the CPU through its `SimpleBuilder`, computes a
balanced device map and dispatches it. The builder keeps that encoder and its
device-mapped weights: after the wrapper's `dispose`, a later build reassigns the
weights to the same dispatched encoder — no disk I/O and no new dispatch.

```python
from ltx_core.loader.distributed_builders import AccelerateGemmaBuilder

AccelerateGemmaBuilder.from_root(
    root: str | Path,              # HF gemma dir OR split TE .safetensors
    registry: ModelRegistry,
    group: dist.ProcessGroup,      # self.groups.gemma_group
    src_rank: int,                 # rank that loads + encodes (usually 0)
)
```

Usage (in `runner.setup()`):

Shipped runners take a resolved `ModelPaths` and pass
`model_paths.text_encoder()` as `gemma_root_path` — that is either the monolith
`--gemma-root` directory or the split `--text-encoder-path` single-file TE. A
separate enhancer still comes from `--prompt-enhancer-gemma-root` when set (may
be `None` when enhance shares the encode checkpoint).

```python
from ltx_core.loader.distributed_builders import AccelerateGemmaBuilder

pe = pipeline.prompt_encoder
# Shared enhance aliases the encode builder; capture that before replacing encode.
separate_enhancer = pe._enhancer_text_encoder_builder is not pe._text_encoder_builder
pe._text_encoder_builder = AccelerateGemmaBuilder.from_root(
    model_paths.text_encoder(), registry, self.groups.gemma_group, 0
)
if separate_enhancer:
    pe._enhancer_text_encoder_builder = AccelerateGemmaBuilder.from_root(
        prompt_enhancer_gemma_root, registry, self.groups.gemma_group, 0
    )
else:
    pe._enhancer_text_encoder_builder = pe._text_encoder_builder
```

When a separate enhancer is configured (`--prompt-enhancer-gemma-root` different
from the encode root), patch a second `AccelerateGemmaBuilder` onto
`_enhancer_text_encoder_builder`. Otherwise re-alias it to the encode builder so
shared enhance stays one residency. Enhance goes through `AccelerateGemmaWrapper`
(src generate + string broadcast), same as encode.

## `BatchParallelGemmaBuilder` (replicated — data-parallel over prompts)

Every rank materialises a **full** `GemmaTextEncoder` on its own GPU via its
`SimpleBuilder` (no Accelerate, no `device_map`, no per-layer dispatch hooks). The wrapper (`BatchParallelGemmaWrapper`) then **partitions the prompt list
across ranks** in `encode` and broadcasts each prompt's output, so the forwards run
concurrently on different GPUs. Non-deterministic prompt enhancement
(`enhance_t2v` / `enhance_i2v`) is routed through a single `src_rank`.

```python
from ltx_core.loader.distributed_builders import BatchParallelGemmaBuilder

BatchParallelGemmaBuilder(
    base: SimpleBuilder,           # e.g. gemma3_builder(root) / gemma4_builder(root)
    group: dist.ProcessGroup,
    src_rank: int,
)
```

### Not for the distilled pipeline

Batch-parallel is beneficial only when there is **more than one prompt to encode** —
the typical CFG case, positive + negative (B=2 on 2 ranks = one prompt per rank, both
forwards concurrent). The **distilled** pipeline runs **without CFG**: its `__call__`
accepts a single `prompt` and no `negative_prompt`, so there is only one prompt to
encode and no work to partition; batch-parallel provides no speedup in that case. Use
`AccelerateGemmaBuilder` for the distilled pipeline (as the shipped `distilled` runner
does).
