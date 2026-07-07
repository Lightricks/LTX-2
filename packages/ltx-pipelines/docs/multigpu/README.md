# Multi-GPU Inference

Run LTX-2 pipelines across several GPUs on a single machine.

> ## ⚠️ Important
>
> **Multi-GPU (MGPU) is a latency tool, not a memory tool.** It is designed to reduce
> the latency of a single generation on multi-GPU servers (H100, B200) by splitting
> each denoising step and the VAE decode across GPUs.
>
> **MGPU is not a way to fit a bigger model.** The mutable **working copy** of the
> transformer is a **full replica on every GPU** (each rank builds the whole model;
> LoRAs are fused into it in place). MGPU therefore cannot make a checkpoint that
> doesn't fit on one GPU suddenly fit — for that use FP8 quantization and weight
> offloading (see [Optimization Tips](../optimization.md)).
>
> Each rank *also* holds a second, immutable copy of the **clean (pre-LoRA)
> weights** — kept for LoRA hot-swap (reset + broadcast) — but that copy is
> **sharded** across GPUs (`ShardedSD`, ~1/world_size per rank), not replicated.
> Sequence parallelism additionally splits **activation** memory across ranks. See
> the [weight tracker](pipeline-setup.md#transformerweighttracker--working-copy--sharded-clean-weights)
> for the exact layout. The headline purpose is **latency**, not memory.
>
> **Single machine only.** One process per GPU, `MASTER_ADDR=localhost`, one rank
> per GPU. No multi-node.

## Requirements

- **Linux** -- NCCL and CUDA-IPC peer buffers are Linux-only (no macOS/Windows).
- **>=2 CUDA GPUs on a single node** with P2P access (NVLink/PCIe). No multi-node.
- **PyTorch with CUDA.**
- **`ltx-kernels` built** -- the SP all2all kernel is mandatory. Build with
  `uv sync --group kernels` (needs a CUDA toolkit / nvcc and a C++ compiler, gcc or
  clang). See the root README.

## Capabilities

| Technique | Purpose |
| --------- | ------- |
| [Sequence parallelism (SP)](sequence-parallel.md) | Split the token sequence across GPUs; faithful — numerically equivalent to single-GPU |
| [Tiled data parallelism (TDP)](tiled-data-parallel.md) | One spatial (height x width) tile per GPU; for resolutions outside the training distribution. **Upscale only** |
| [Distributed decoder](distributed-decoder.md) | Decode latent tiles in parallel, assemble on the driver |
| [Distributed Gemma](gemma.md) | Shard Gemma across GPUs via Accelerate `device_map`, or replicate + split prompts |
| [MGPU controller](controller.md) | Persistent worker fleet; dispatch a job, stream results |
| [Pipeline setup](pipeline-setup.md) | Swap single-GPU builders for MGPU builders; share one weights registry |

## Architecture overview

The [`MGPUController`](controller.md) spawns one worker process per GPU and runs a
user-defined **runner** (a subclass of `MGPURunner`) in SPMD lockstep. A runner's
`setup()` builds a standard pipeline, then **swaps** each block's builder for an MGPU
builder (SP / TDP / distributed decoder / distributed Gemma). All builders share one
`StateDictRegistry` so the checkpoint loads from disk once per process.

Two runners are provided, each with a CLI:

- [`ltx_pipelines.ti2vid_two_stages_mgpu`](../../src/ltx_pipelines/ti2vid_two_stages_mgpu.py) — SP stage 1 + TDP stage 2 + Accelerate Gemma + distributed VAE.
- [`ltx_pipelines.distilled_mgpu`](../../src/ltx_pipelines/distilled_mgpu.py) — SP (shared stage) + Accelerate Gemma + distributed VAE.

```bash
# Two-stage on all visible GPUs
python -m ltx_pipelines.ti2vid_two_stages_mgpu \
    --checkpoint-path path/to/checkpoint.safetensors \
    --distilled-lora path/to/distilled_lora.safetensors 1.0 \
    --spatial-upsampler-path path/to/upsampler.safetensors \
    --gemma-root path/to/gemma \
    --prompt "A beautiful sunset over the ocean" \
    --output-path output.mp4
```

## Pages

- **[Controller](controller.md)** — `MGPUController` / `MGPURunner` / `Stream`, lifecycle, one-job-at-a-time contract, threading, error handling.
- **[Pipeline setup](pipeline-setup.md)** — swapping builders, the shared weights registry, the LoRA-hot-swap weight tracker.
- **[Sequence parallelism](sequence-parallel.md)** — faithful token-dim split, the all2all kernels, `AttentionManager`, `SequenceParallelBuilder`.
- **[Tiled data parallelism](tiled-data-parallel.md)** — out-of-distribution resolutions, position normalization, shared negative (reference) positions, `TiledDataParallelBuilder`.
- **[Distributed decoder](distributed-decoder.md)** — inter-GPU vs intra-GPU tiling, `DistributedDecoderBuilder`.
- **[Gemma](gemma.md)** — `AccelerateGemmaBuilder` (Accelerate `device_map` sharding) and `BatchParallelGemmaBuilder` (replicated; not for the distilled pipeline).
