# Gemma Text Encoder (Multi-GPU)

**Source**: [`multigpu/gemma_builders.py`](../../src/ltx_pipelines/multigpu/gemma_builders.py), [`multigpu/bp_gemma_builder.py`](../../src/ltx_pipelines/multigpu/bp_gemma_builder.py)

Two ways to run the Gemma text encoder across the fleet. Both swap in for
`pipeline.prompt_encoder._text_encoder_builder` and broadcast the resulting
embeddings to every rank (so the transformer ranks all have them).

## `AccelerateGemmaBuilder` (Accelerate `device_map` — the default)

Loads Gemma **once, on the source rank**, with Accelerate `device_map="auto"`, which
**shards Gemma's layers across the available GPUs**. Non-source ranks receive a
lightweight `AccelerateGemmaWrapper` stub that receives the encoded embeddings over
NCCL. The source rank fuses all prompts into one Gemma call, then broadcasts each
output.

The first `build()` loads via HuggingFace `from_pretrained` and caches the full
state dict (including non-persistent buffers) in the registry; later builds recreate
the model from cache and reinstall the dispatch hooks — no disk I/O.

```python
from ltx_pipelines.multigpu.gemma_builders import AccelerateGemmaBuilder

AccelerateGemmaBuilder(
    gemma_root_path: str,
    gemma_group: dist.ProcessGroup | None,      # self.groups.gemma_group
    broadcast_group: dist.ProcessGroup | None,  # self.groups.transformer_group
    registry: Registry,
    *,
    src_rank: int,                              # rank that loads + encodes (usually 0)
    dtype: torch.dtype = torch.bfloat16,
)
```

Usage (in `runner.setup()`):

```python
pipeline.prompt_encoder._text_encoder_builder = AccelerateGemmaBuilder(
    gemma_root_path=gemma_root,
    gemma_group=self.groups.gemma_group,
    broadcast_group=self.groups.transformer_group,
    registry=registry,
    src_rank=0,
    dtype=pipeline.dtype,
)
```

The shipped runners (`ti2vid_two_stages_mgpu`, `ti2vid_two_stages_hq_mgpu`, `distilled_mgpu`) use this builder.

## `BatchParallelGemmaBuilder` (replicated — data-parallel over prompts)

Every rank materialises a **full** `GemmaTextEncoder` on its own GPU via the standard
`SingleGPUModelBuilder` path (no Accelerate, no `device_map`, no per-layer dispatch
hooks). The wrapper (`BatchParallelGemmaWrapper`) then **partitions the prompt list
across ranks** in `encode` and broadcasts each prompt's output, so the forwards run
concurrently on different GPUs. Non-deterministic prompt enhancement
(`enhance_t2v` / `enhance_i2v`) is routed through a single `src_rank`.

```python
from ltx_pipelines.multigpu.bp_gemma_builder import BatchParallelGemmaBuilder

BatchParallelGemmaBuilder(
    gemma_root_path: str,
    broadcast_group: dist.ProcessGroup | None,
    registry: Registry,
    *,
    src_rank: int,
    dtype: torch.dtype = torch.bfloat16,
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
