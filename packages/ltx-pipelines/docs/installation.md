# Installation & Usage

## Installation

```bash
# From the repository root
uv sync --frozen

# Or install as a package
pip install -e packages/ltx-pipelines
```

## Requirements

- **LTX-2 Model Checkpoint** - Local `.safetensors` file
- **Gemma Text Encoder** - Local Gemma model directory
- **Spatial Upscaler** - Required by two-stage pipelines, for the upsampling stage
- **Distilled LoRA** - Required by two-stage non-distilled pipelines, used for the stage-2 refinement

## Running Pipelines

All pipelines can be run directly from the command line. Each pipeline module is executable:

```bash
# Run a pipeline (example: two-stage text-to-video)
python -m ltx_pipelines.ti2vid_two_stages \
    --checkpoint-path path/to/checkpoint.safetensors \
    --distilled-lora path/to/distilled_lora.safetensors 0.8 \
    --spatial-upsampler-path path/to/upsampler.safetensors \
    --gemma-root path/to/gemma \
    --prompt "A beautiful sunset over the ocean" \
    --output-path output.mp4

# View all available options for any pipeline
python -m ltx_pipelines.ti2vid_two_stages --help
```

### Available pipeline modules

- `ltx_pipelines.ti2vid_two_stages` - Two-stage text/image-to-video (recommended). ([docs](pipelines.md#1-ti2vidtwostagespipeline), [source](../src/ltx_pipelines/ti2vid_two_stages.py))
- `ltx_pipelines.ti2vid_two_stages_hq` - Two-stage text/image-to-video (different sampler, better quality). ([docs](pipelines.md#2-ti2vidtwostageshqpipeline), [source](../src/ltx_pipelines/ti2vid_two_stages_hq.py))
- `ltx_pipelines.ti2vid_one_stage` - Single-stage text/image-to-video. ([docs](pipelines.md#3-ti2vidonestagepipeline), [source](../src/ltx_pipelines/ti2vid_one_stage.py))
- `ltx_pipelines.t2a_one_stage` - Single-stage text-to-audio (audio-only output). ([docs](pipelines.md#11-t2aonestagepipeline), [source](../src/ltx_pipelines/t2a_one_stage.py))
- `ltx_pipelines.distilled` - Fast text/image-to-video pipeline using only the distilled model. ([docs](pipelines.md#4-distilledpipeline), [source](../src/ltx_pipelines/distilled.py))
- `ltx_pipelines.ic_lora` - Video-to-video with IC-LoRA. ([docs](pipelines.md#5-iclorapipeline), [source](../src/ltx_pipelines/ic_lora.py))
- `ltx_pipelines.keyframe_interpolation` - Keyframe interpolation. ([docs](pipelines.md#6-keyframeinterpolationpipeline), [source](../src/ltx_pipelines/keyframe_interpolation.py))
- `ltx_pipelines.a2vid_two_stage` - Audio-to-video generation conditioned on an input audio. ([docs](pipelines.md#7-a2vidpipelinetwostage), [source](../src/ltx_pipelines/a2vid_two_stage.py))
- `ltx_pipelines.retake` - Regenerate a time region of an existing video. ([docs](pipelines.md#8-retakepipeline), [source](../src/ltx_pipelines/retake.py))
- `ltx_pipelines.hdr_ic_lora` - Video-to-video with HDR output (linear float via LogC3 inverse decode). ([docs](pipelines.md#9-hdriclorapipeline), [source](../src/ltx_pipelines/hdr_ic_lora.py))
- `ltx_pipelines.lipdub` - Lip dubbing / re-voicing with IC-LoRA and audio reference conditioning. ([docs](pipelines.md#10-lipdubpipeline), [source](../src/ltx_pipelines/lipdub.py))

Use `--help` with any pipeline module to see all available options and parameters.

## Common CLI flags

These flags are shared across the pipeline CLIs (they come from a common base parser); run a module with `--help` for its full set.

- `--seed <int>` - random seed for reproducible generation (default 10).
- `--offload {none,cpu,disk}` - offload transformer weights to reduce peak GPU memory. `cpu` holds them in system RAM; `disk` streams them from disk when RAM is also limited (slower). Default `none`.
- `--quantization {fp8-cast,fp8-scaled-mm}` - run the transformer in FP8 to cut memory. `fp8-cast` downcasts a bf16 checkpoint on the fly (any FP8-capable GPU); `fp8-scaled-mm` expects an fp8 checkpoint and native FP8 support (best on Hopper+).
- `--max-batch-size <int>` - max batch per transformer forward pass (default 1). Higher values reduce layer-streaming transfers at the cost of peak memory.
- `--compile [key=value ...]` - enable `torch.compile`, optionally overriding the compilation config.
- `--lora <path> [strength]` - apply a LoRA (repeatable; default strength 1.0).
- `--enhance-prompt` - rewrite the prompt with the built-in enhancer before generation.
