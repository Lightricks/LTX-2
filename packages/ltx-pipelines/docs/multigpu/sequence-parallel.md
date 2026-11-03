# Sequence Parallelism (SP)

**Source**: [`ltx_core/multigpu/transformer/sequence_parallel.py`](../../../ltx-core/src/ltx_core/multigpu/transformer/sequence_parallel.py), [`ltx_core/loader/distributed_builders.py`](../../../ltx-core/src/ltx_core/loader/distributed_builders.py)

## What it is

SP splits the **token (sequence) dimension** of the video across GPUs. Each rank
holds a slice of the tokens, runs the transformer forward on its slice, and the
outputs are gathered back to all ranks. Self-attention still needs every token to
see every other token, so the Q/K/V heads are exchanged across ranks with a custom
**all2all** kernel: each rank ends up with all tokens for a subset of heads, does
local attention, then the results are shuffled back.

**SP is faithful — numerically equivalent to single-GPU inference.** Attention stays
global (all2all preserves the full token interaction); only the floating-point
reduction order changes. The all2all kernels move bytes only — the round-trip
`gather(send(x)) == x` is byte-exact. SP is the appropriate choice whenever the
single-GPU result is required at lower latency.

This is the default for **stage 1** (`ti2vid_two_stages_mgpu`), the **shared stage**
(`distilled_mgpu`, where one SP wrapping covers both the half-res and full-res calls),
and **both DFR stages** (`dfr_mgpu`: `stage` and `stage_detailing` share one
`AttentionManager`, sized for 4K ~300-frame generation).

## How the forward pass works

Per denoising step, [`SequenceParallelModelWrapper`](../../../ltx-core/src/ltx_core/multigpu/transformer/sequence_parallel.py):

1. Pads the video seq dim up to a multiple of `world_size` (padded keys are masked
   out; padded rows sliced off after the gather) so every rank gets an equal shard.
2. Tiles latent/timesteps/positions to this rank's slice.
3. Runs the model — video self-attention (`attn1`) and video→audio cross-attention
   are patched to route Q/K/V through the all2all kernel.
4. `all_gather`s the output tokens back to full length on every rank and unpads.

## The all2all kernels (`ltx-kernels`)

The custom op is `ltx_kernels.All2All` (from the `ltx-kernels` package); the CUDA
kernels use CUDA-IPC peer buffers to exchange tokens directly between ranks' GPUs.
**`ltx-kernels` must be installed** — the SP builder imports it.

## API

### `AttentionManager`

```python
from ltx_core.multigpu.transformer.attention import AttentionManager

attn_mgr = AttentionManager(
    max_tokens: int,           # upper bound on total video tokens (raises above it)
    num_heads: int,            # transformer.num_attention_heads
    head_dim: int,             # transformer.attention_head_dim
    tensor_dtype: torch.dtype,
    group: dist.ProcessGroup,  # self.groups.transformer_group
    copy_out_: bool = False,
)
```

Owns the all2all buffers, sized `ceil(max_tokens / world_size)` tokens per rank. All2All operations
derive their uniform per-rank token counts from the input tensors; callers pad before sharding.
`num_heads` must be divisible by `world_size`.

### `SequenceParallelBuilder`

```python
from ltx_core.loader.distributed_builders import SequenceParallelBuilder

SequenceParallelBuilder(
    base: SimpleBuilder,                 # the stage's transformer builder
    attention_manager: AttentionManager,
    dst: dict[str, torch.Tensor] | None = None,  # active weights shared with the other stages
)
```

Adds the all2all attention module op to `base` when constructed, so a stage's
compilation op runs after it, and `build()` returns a `SequenceParallelModelWrapper`
around the distributed model (see [shared active weights](pipeline-setup.md#shared-active-weights--one-replica-rank-owned-clean-units)). The all2all barrier timeout is
`attention_manager.all2all_timeout_seconds`.

## Usage

```python
# inside runner.setup(), per stage:
model_cfg = pipeline.stage_1._transformer_builder.model_config().get("transformer", {})
attn_mgr = AttentionManager(
    max_tokens=32768,
    num_heads=model_cfg["num_attention_heads"],
    head_dim=model_cfg["attention_head_dim"],
    tensor_dtype=pipeline.dtype,
    group=self.groups.transformer_group,
)
pipeline.stage_1._transformer_builder = SequenceParallelBuilder(
    pipeline.stage_1._transformer_builder, attn_mgr, dst=active_weights
)
```

`max_tokens` must cover the largest step. Reference: stage 1 at 512x768x121 is
~6k video tokens; the distilled shared stage's full-res call (1024x1536x121) is
~24k — both ship with `sp_max_tokens=32768`. DFR stage 2 at 4K (3840x2176, ~300
frames after canvas padding, plus keyframe slots and the x2 IC-LoRA reference) is
~514k and ships with `sp_max_tokens=524288`. Exceeding it raises with a clear
"use a smaller resolution or fewer frames" message.
