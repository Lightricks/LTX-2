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
> Each rank *also* keeps the **clean (pre-LoRA) weights** of the units it owns, for
> LoRA hot-swap: about 1/world_size of the checkpoint per rank, not a replica.
> Sequence parallelism additionally splits **activation** memory across ranks. See
> the [shared active weights](pipeline-setup.md#shared-active-weights--one-replica-rank-owned-clean-units)
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
`ModelRegistry` so the checkpoint loads from disk once per process.

Four runners are provided, each with a CLI:

- [`ltx_pipelines.ti2vid_two_stages_mgpu`](../../src/ltx_pipelines/ti2vid_two_stages_mgpu.py) — SP stage 1 + TDP stage 2 + Accelerate Gemma + distributed VAE.
- [`ltx_pipelines.ti2vid_two_stages_hq_mgpu`](../../src/ltx_pipelines/ti2vid_two_stages_hq_mgpu.py) — SP stage 1 + TDP stage 2 (HQ sampler) + Accelerate Gemma + distributed VAE.
- [`ltx_pipelines.distilled_mgpu`](../../src/ltx_pipelines/distilled_mgpu.py) — SP (shared stage) + Accelerate Gemma + distributed VAE.
- [`ltx_pipelines.dfr_mgpu`](../../src/ltx_pipelines/dfr_mgpu.py) — SP on both DFR stages (base + detailing) + Accelerate Gemma + distributed VAE.

Checkpoint paths follow the same monolith **XOR** split contract as single-GPU CLIs:
the parser builds a `ModelPaths` object and runners read only that (see
[installation](../installation.md#checkpoint-layouts-monolith-vs-split)). Pass either a fat checkpoint +
Gemma directory, or the split component flags — not both.

```bash
# Two-stage on all visible GPUs (monolith layout)
python -m ltx_pipelines.ti2vid_two_stages_mgpu \
    --checkpoint-path path/to/checkpoint.safetensors \
    --distilled-lora path/to/distilled_lora.safetensors 1.0 \
    --spatial-upsampler-path path/to/upsampler.safetensors \
    --gemma-root path/to/gemma \
    --prompt "A beautiful sunset over the ocean" \
    --output-path output.mp4

# DFR on all visible GPUs (distilled checkpoint)
python -m ltx_pipelines.dfr_mgpu \
    --distilled-checkpoint-path path/to/distilled_checkpoint.safetensors \
    --detailing-lora path/to/detailing_lora.safetensors \
    --spatial-upsampler-path path/to/upsampler.safetensors \
    --gemma-root path/to/gemma \
    --prompt "A beautiful sunset over the ocean" \
    --output-path output.mp4

# Same entrypoint with a Comfy-aligned split pack (subset of flags as needed; TI2Vid shown)
python -m ltx_pipelines.ti2vid_two_stages_mgpu \
    --transformer-path path/to/diffusion_models/ltx-….safetensors \
    --text-encoder-path path/to/text_encoders/gemma-….safetensors \
    --video-vae-path path/to/vae/ltx-…-video-vae.safetensors \
    --audio-vae-path path/to/vae/ltx-…-audio-vae.safetensors \
    --duration-head-path path/to/duration_head/ltx-….safetensors \
    --distilled-lora path/to/distilled_lora.safetensors 1.0 \
    --spatial-upsampler-path path/to/upsampler.safetensors \
    --prompt "A beautiful sunset over the ocean" \
    --output-path output.mp4
```

## Pages

- **[Controller](controller.md)** — `MGPUController` / `MGPURunner` / `Stream`, lifecycle, one-job-at-a-time contract, threading, error handling.
- **[Pipeline setup](pipeline-setup.md)** — swapping builders, the shared weights registry, LoRA hot-swap on shared active weights.
- **[Sequence parallelism](sequence-parallel.md)** — faithful token-dim split, the all2all kernels, `AttentionManager`, `SequenceParallelBuilder`.
- **[Tiled data parallelism](tiled-data-parallel.md)** — out-of-distribution resolutions, position normalization, shared negative (reference) positions, `TiledDataParallelBuilder`.
- **[Distributed decoder](distributed-decoder.md)** — inter-GPU vs intra-GPU tiling, `DistributedDecoderBuilder`.
- **[Gemma](gemma.md)** — `AccelerateGemmaBuilder` (Accelerate `device_map` sharding) and `BatchParallelGemmaBuilder` (replicated; not for the distilled pipeline).
