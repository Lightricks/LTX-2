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
- **Distilled LoRA** - Required by two-stage non-distilled pipelines (TI2Vid, Keyframe, A2Vid), used for the stage-2 refinement. Not used by DFR.
- **Detailing IC-LoRA** - Required by DFR (`--detailing-lora`)
- **Temporal Upscaler** - Required by DFR when `--temporal-upscalings` is 1 or 2

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

### Checkpoint layouts: monolith vs split

Every pipeline CLI accepts one of two mutually exclusive path sets.

**Monolith** (legacy) -- one fat `.safetensors` plus a HuggingFace Gemma directory:

```bash
--checkpoint-path path/to/checkpoint.safetensors --gemma-root path/to/gemma
```

DistilledPipeline and DFR use `--distilled-checkpoint-path` instead of `--checkpoint-path`.

**Split** (Comfy-aligned) -- one file per component. Pass the subset this pipeline needs
(unused flags may be omitted; the pipeline validates required slots on `ModelPaths`):

```bash
python -m ltx_pipelines.distilled \
    --transformer-path      path/to/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
    --text-encoder-path   path/to/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
    --video-vae-path      path/to/vae/ltx-2.5-video-vae-bf16.safetensors \
    --audio-vae-path      path/to/vae/ltx-2.5-audio-vae-bf16.safetensors \
    --duration-head-path  path/to/duration_head/ltx-2.5-duration-head-bf16.safetensors \
    ...
```

Mixing the two sets is an error. In split mode `--text-encoder-path` is a single file -- the
HuggingFace sidecars (config, tokenizer, processor) are embedded in it and loaded in memory,
so no `--gemma-root` is needed. Both layouts resolve to the same `ModelPaths` object, which is
what pipeline constructors take.

### Available pipeline modules

- `ltx_pipelines.distilled` - Fast text/image-to-video pipeline using only the distilled model (starting point). ([docs](pipelines.md#4-distilledpipeline), [source](../src/ltx_pipelines/distilled.py))
- `ltx_pipelines.dfr_pipeline` - Production-quality text/image-to-video: distilled checkpoint, generated keyframes, spatial detailing, optional temporal 2x/4x. ([docs](pipelines.md#12-dfrpipeline), [how to run](pipelines.md#running-dfr), [source](../src/ltx_pipelines/dfr_pipeline.py))
- `ltx_pipelines.dfr_mgpu` - Multi-GPU DFR: sequence parallelism on both stages. ([docs](multigpu/README.md), [source](../src/ltx_pipelines/dfr_mgpu.py))
- `ltx_pipelines.ti2vid_two_stages` - Two-stage text/image-to-video. ([docs](pipelines.md#1-ti2vidtwostagespipeline), [source](../src/ltx_pipelines/ti2vid_two_stages.py))
- `ltx_pipelines.ti2vid_two_stages_hq` - Two-stage text/image-to-video (different sampler, better quality). ([docs](pipelines.md#2-ti2vidtwostageshqpipeline), [source](../src/ltx_pipelines/ti2vid_two_stages_hq.py))
- `ltx_pipelines.ti2vid_one_stage` - Single-stage text/image-to-video. ([docs](pipelines.md#3-ti2vidonestagepipeline), [source](../src/ltx_pipelines/ti2vid_one_stage.py))
- `ltx_pipelines.t2a_one_stage` - Single-stage text-to-audio (audio-only output). ([docs](pipelines.md#11-t2aonestagepipeline), [source](../src/ltx_pipelines/t2a_one_stage.py))
- `ltx_pipelines.ic_lora` - Video-to-video with IC-LoRA. ([docs](pipelines.md#5-iclorapipeline), [source](../src/ltx_pipelines/ic_lora.py))
- `ltx_pipelines.a2vid_two_stage` - Audio-to-video generation conditioned on an input audio. ([docs](pipelines.md#7-a2vidpipelinetwostage), [source](../src/ltx_pipelines/a2vid_two_stage.py))
- `ltx_pipelines.retake` - Regenerate a time region of an existing video. ([docs](pipelines.md#8-retakepipeline), [source](../src/ltx_pipelines/retake.py))
- `ltx_pipelines.hdr_ic_lora` - Video-to-video SDR→HDR: writes a BT.2020/HLG master plus an EXR sequence (`--exr-colorspace`, default ACEScg). ([docs](pipelines.md#9-hdriclorapipeline), [source](../src/ltx_pipelines/hdr_ic_lora.py))
- `ltx_pipelines.dubit` - Dub-It / re-voicing with IC-LoRA and audio reference conditioning. ([docs](pipelines.md#10-dubitpipeline), [source](../src/ltx_pipelines/dubit.py))

Use `--help` with any pipeline module to see all available options and parameters.

For native EXR conditioning and HDR encode on the standard pipelines, see [HDR Support](hdr.md).

## Calling pipelines from Python

Pipelines are importable classes as well as CLI modules. The CLI `main()` functions run under
`torch.inference_mode()`; pipeline `__call__` methods do not, so a pipeline can also run inside a
training loop that records gradients. For inference from Python, wrap the whole call, including
prompt encoding, in `torch.inference_mode()`:

```python
import torch

from ltx_pipelines import DistilledPipeline
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.model_paths import ModelPaths

pipeline = DistilledPipeline(
    model_paths=ModelPaths.from_monolith("path/to/distilled.safetensors", "path/to/gemma"),
    spatial_upsampler_path="path/to/upsampler.safetensors",
    loras=[],
)
with torch.inference_mode():
    output = pipeline(
        prompt="A beautiful sunset over the ocean",
        seed=10,
        height=512,
        width=768,
        frame_rate=24.0,
        images=[],
        num_frames=121,
    )
    encode_video(output.video, 24.0, output.audio, "output.mp4", video_chunks_number=None)
```

Without `torch.inference_mode()` (or `torch.no_grad()`), autograd records the forward passes:
the text encoder's outputs keep a reference to the encoder and its activations after prompt
encoding, which can hold tens of GiB of device memory and run out of memory later in the
pipeline. `model.eval()` does not disable autograd.

## Common CLI flags

These flags are shared across the pipeline CLIs (they come from a common base parser); run a module with `--help` for its full set.

- `--seed <int>` - random seed for reproducible generation (default 10).
- `--offload {none,cpu,disk}` - offload transformer weights to reduce peak GPU memory. `cpu` holds them in system RAM; `disk` streams them from disk when RAM is also limited (slower). Default `none`.
- `--quantization {fp8-cast,fp8-scaled-mm}` - run the transformer in FP8 to cut memory. `fp8-cast` downcasts a bf16 checkpoint on the fly (any FP8-capable GPU); `fp8-scaled-mm` expects an fp8 checkpoint and native FP8 support (best on Hopper+).
- `--max-batch-size <int>` - max batch per transformer forward pass (default 1). Higher values reduce layer-streaming transfers at the cost of peak memory.
- `--compile [key=value ...]` - enable `torch.compile`. Bare `--compile` compiles without CUDA graphs. CUDA-graph modes (`capture=true`, `mode=reduce-overhead`, `mode=max-autotune`) need `--offload cpu` or `disk` on single GPU (streaming reuses GPU weight slots; a registry by itself does not). `capture=true` captures one graph per (shape, perturbation); `max_video_tokens` / `max_audio_tokens` size a shared **static input** pool so those graphs do not each allocate their own activation buffers. See [Compilation](optimization.md#compilation-torchcompile).
- `--lora <path> [strength]` - apply a LoRA (repeatable; default strength 1.0).
- `--enhance-prompt` - rewrite the prompt with the built-in enhancer before generation.
- `--hdr {SRGB_LINEAR,ACESCG,ACESCCT,ACES2065_1}` - on pipelines that accept image / video conditioning (not Dub-It, HDR IC-LoRA, or T2A): declare the source colour space for EXR stills / EXR-frame folders. Required whenever any EXR input is passed; omit for SDR. With `--hdr` set, encode writes half EXR frames plus a BT.2020/HLG master. See [HDR Support](hdr.md).
- `--marker-root-path <dir>` - optional local `path/to/marking` pack for post-hoc VideoSeal (SDR pixels) and AudioSeal (stereo PCM). Requires `uv sync --extra videoseal --extra audioseal`. HDR encode skips video stamping and still stamps audio. See [VideoSeal marking](marking.md).
- `--video-vae-path <path>` - video VAE `.safetensors` (encoder + decoder). Split: the `vae/` component. Monolith: optional override of the VAE bundled in `--checkpoint-path`/`--distilled-checkpoint-path`. When it is a diffusion VAE, decoding builds a `DiffusionVideoDecoder`, which for best performance requires the `natten` extra. See [Optimization Tips](optimization.md#diffusion-vae-decoder).
- `--diffvae-optimization {chunked_eager,chunked_compile,combined_compile,blackwell_dsl}` - DiffVAE decode preset (ignored for convolutional VAEs). `blackwell_dsl` is the fast path on datacenter Blackwell.
- `--chunk-pixel-frames` / `--chunk-carry-frames` - on `distilled`, `ti2vid_two_stages` (and their MGPU runners), `a2vid_two_stage`, and `dubit`: pass either flag to generate in windows instead of one full-clip chunk. Defaults are 97 pixel frames and 25-pixel carry when the other is set. Pixel frame counts must be on the causal grid (`8k+1`). Off-grid `--num-frames` is floored to the causal grid.
