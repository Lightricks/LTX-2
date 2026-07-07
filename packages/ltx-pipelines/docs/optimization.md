# ⚡ Optimization Tips

## Memory Optimization

### FP8 Quantization (Lower Memory Footprint)

For smaller GPU memory footprint, use the `--quantization` flag and set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Two quantization policies are available:

| Policy | CLI Flag | Description |
| ------ | -------- | ----------- |
| **FP8 Cast** | `--quantization fp8-cast` | Downcasts transformer linear weights to FP8 during loading; upcasts on the fly during inference. No extra dependencies. |
| **FP8 Scaled MM** | `--quantization fp8-scaled-mm` | Uses FP8 scaled matrix multiplication via PyTorch's `torch._scaled_mm`. Best performance on Hopper+ GPUs with native FP8 support. |

**CLI:**

```bash
# FP8 Cast (works on any GPU with FP8 support)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m ltx_pipelines.ti2vid_two_stages \
    --quantization fp8-cast --checkpoint-path=...

# FP8 Scaled MM (no extra deps, best on Hopper+ GPUs)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m ltx_pipelines.ti2vid_two_stages \
    --quantization fp8-scaled-mm --checkpoint-path=...
```

**Programmatically:**

When authoring custom scripts, pass a `QuantizationPolicy` to pipeline classes:

```python
from ltx_core.quantization.fp8_cast import build_policy as build_fp8_cast_policy
# Alternative:
# from ltx_core.quantization.fp8_scaled_mm import build_policy as build_fp8_scaled_mm_policy

pipeline = TI2VidTwoStagesPipeline(
    checkpoint_path=ltx_model_path,
    distilled_lora=distilled_lora,
    spatial_upsampler_path=upsampler_path,
    gemma_root=gemma_root_path,
    loras=[],
    quantization=build_fp8_cast_policy(ltx_model_path),
)
pipeline(...)
```

You still need to use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` when launching:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python my_denoising_pipeline.py
```

### Memory Cleanup Between Stages

By default, pipelines clean GPU memory (especially transformer weights) between stages. If you have enough memory, you can skip this cleanup to reduce running time:

```python
# In pipeline implementations, memory cleanup happens automatically
# between stages. For custom pipelines, you can skip:
# utils.cleanup_memory()  # Comment out if you have enough VRAM
```

## Compilation (`torch.compile`)

Compiling the transformer blocks with `torch.compile` speeds up inference. It is **opt-in and off by default**. The blocks are compiled shape-polymorphically (the sequence dimension is marked dynamic), so one compiled artifact serves any token count without recompiling.

**CLI** — the `--compile` flag maps directly to `CompilationConfig`:

| Form | Result |
| ---- | ------ |
| *(flag absent)* | eager, no compilation |
| `--compile` | compile with defaults |
| `--compile KEY=VALUE ...` | compile, overriding individual fields |

```bash
# Defaults
python -m ltx_pipelines.ti2vid_two_stages --compile --checkpoint-path=...

# reduce-overhead captures CUDA graphs -- the main latency lever for the denoising loop.
# Off by default because graph capture reserves static memory pools (extra VRAM), so it
# trades memory for speed; enable it when you have headroom.
python -m ltx_pipelines.ti2vid_two_stages --compile mode=reduce-overhead --checkpoint-path=...

# Several overrides at once
python -m ltx_pipelines.ti2vid_two_stages \
    --compile mode=max-autotune fullgraph=true dynamic=true --checkpoint-path=...
```

| Field | Values | Default | Notes |
| ----- | ------ | ------- | ----- |
| `mode` | `none`, `reduce-overhead`, `max-autotune`, … | `none` | `reduce-overhead`/`max-autotune` enable CUDA graphs |
| `backend` | `inductor`, `eager`, … | `inductor` | |
| `fullgraph` | `true`/`false` | `false` | |
| `dynamic` | `auto`/`true`/`false` | `auto` | the seq dim is marked dynamic regardless |
| `inductor_config` | JSON object or path to a `.json` | `{}` | `torch._inductor.config` overrides |
| `dynamo_config` | JSON object or path to a `.json` | `{"inline_inbuilt_nn_modules": true, "cache_size_limit": 256}` | `torch._dynamo.config` overrides |

**Controlling inductor / dynamo configs.** `inductor_config` and `dynamo_config` take either an inline JSON object or a path to a `.json` file, applied via `torch._inductor.config.patch(...)` / `torch._dynamo.config.patch(...)` around the compiled forward. They **replace the defaults wholesale — they do not merge**, so when overriding `dynamo_config` re-include any defaults you want to keep:

```bash
python -m ltx_pipelines.ti2vid_two_stages \
    --compile 'inductor_config={"max_autotune": true}' \
              'dynamo_config={"inline_inbuilt_nn_modules": true, "cache_size_limit": 256, "recompile_limit": 32}' \
    --checkpoint-path=...
```

**Programmatically**, pass a `CompilationConfig` to the pipeline:

```python
from ltx_core.model.transformer.compiling import CompilationConfig

pipeline = TI2VidTwoStagesPipeline(
    ...,
    compilation_config=CompilationConfig(mode="reduce-overhead"),
)
```

**Faster cache loads: `unsafe_skip_cache_dynamic_shape_guards` (unsafe, opt-in).** Inductor's FX-graph cache re-checks the dynamic-shape guards stored with each entry on every lookup. Setting this flag skips that re-check (every entry is treated as a guard hit), which speeds up warm and cross-process cache loads. It is **not enabled by default** because it is a correctness hazard: a kernel first compiled at a small sequence length keeps int32 address arithmetic, and reusing it at a larger sequence length (roughly **>58k tokens/rank**) overflows int32 and reads out of bounds — surfacing as a CUDA illegal memory access or silently corrupted output. Only enable it when your token counts stay within the range the cached kernels were compiled for:

```bash
python -m ltx_pipelines.ti2vid_two_stages \
    --compile 'inductor_config={"unsafe_skip_cache_dynamic_shape_guards": true}' \
    --checkpoint-path=...
```

## Denoising Loop Optimization

**Gradient Estimation Denoising Loop:**

Instead of the standard Euler denoising loop, you can use gradient estimation for fewer steps (~20-30 instead of 40):

```python
from ltx_pipelines.utils import gradient_estimating_euler_denoising_loop

# Use gradient estimation denoising loop
def denoising_loop(sigmas, video_state, audio_state, stepper):
    return gradient_estimating_euler_denoising_loop(
        sigmas=sigmas,
        video_state=video_state,
        audio_state=audio_state,
        stepper=stepper,
        transformer=transformer,
        denoiser=denoiser,
        ge_gamma=2.0,  # Gradient estimation coefficient
    )
```

This allows you to use **20-30 steps instead of 40** while maintaining quality. The gradient estimation function is defined in [`samplers.py`](../src/ltx_pipelines/utils/samplers.py).
