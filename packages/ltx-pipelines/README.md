# LTX-2 Pipelines

High-level pipeline implementations for generating audio-video content with Lightricks' **LTX-2** model. This package provides ready-to-use pipelines for text-to-video, image-to-video, video-to-video, and keyframe interpolation tasks.

Pipelines are built using building blocks from [`ltx-core`](../ltx-core/) (schedulers, guiders, noisers, patchifiers) and handle the complete inference flow including model loading, encoding, decoding, and file I/O.

---

## 📋 Overview

LTX-2 Pipelines provides production-ready implementations that abstract away the complexity of the diffusion process, model loading, and memory management. Each pipeline is optimized for specific use cases and offers different trade-offs between speed, quality, and memory usage.

**Key Features:**

- 🎬 **Multiple Pipeline Types**: Text-to-video, image-to-video, video-to-video, and keyframe interpolation
- ⚡ **Optimized Performance**: Support for FP8 transformers, gradient estimation, and memory optimization
- 🎯 **Production Ready**: Two-stage pipelines for best quality output
- 🔧 **LoRA Support**: Easy integration with trained LoRA adapters
- 📦 **Self-Contained**: Handles model loading, encoding, decoding, and file I/O
- 🚀 **CLI Support**: All pipelines can be run as command-line scripts

---

## 🚀 Quick Start

`ltx-pipelines` provides ready-made inference pipelines for text-to-video, image-to-video, video-to-video, and keyframe interpolation. Built using building blocks from [`ltx-core`](../ltx-core/), these pipelines handle the complete inference flow including model loading, encoding, decoding, and file I/O.

## 🔧 Installation

```bash
# From the repository root
uv sync --frozen

# Or install as a package
pip install -e packages/ltx-pipelines
```

### Running Pipelines

All pipelines can be run directly from the command line. Each pipeline module is executable:

```bash
# Run a pipeline (example: two-stage text-to-video)
python -m ltx_pipelines.ti2vid_two_stages \
    --checkpoint-path path/to/checkpoint.safetensors \
    --distilled-lora-path path/to/distilled_lora.safetensors \
    --spatial-upsampler-path path/to/upsampler.safetensors \
    --gemma-root path/to/gemma \
    --prompt "A beautiful sunset over the ocean" \
    --output-path output.mp4

# View all available options for any pipeline
python -m ltx_pipelines.ti2vid_two_stages --help
```

Available pipeline modules:

- `ltx_pipelines.ti2vid_two_stages` - Two-stage text-to-video (recommended)
- `ltx_pipelines.ti2vid_one_stage` - Single-stage text-to-video
- `ltx_pipelines.distilled` - Fast distilled pipeline
- `ltx_pipelines.ic_lora` - Video-to-video with IC-LoRA
- `ltx_pipelines.keyframe_interpolation` - Keyframe interpolation

Use `--help` with any pipeline module to see all available options and parameters.

---

## 🎯 Pipeline Selection Guide

### Quick Decision Tree

```text
Do you need to condition on existing images/videos?
├─ YES → Do you have reference videos for video-to-video?
│  ├─ YES → Use ICLoraPipeline
│  └─ NO → Do you have keyframe images to interpolate?
│     ├─ YES → Use KeyframeInterpolationPipeline
│     └─ NO → Use ICLoraPipeline (image conditioning only)
│
└─ NO → Text-to-video only
   ├─ Do you need best quality?
   │  └─ YES → Use TI2VidTwoStagesPipeline (recommended for production)
   │
   └─ Do you need fastest inference?
      └─ YES → Use DistilledPipeline (with 8 predefined sigmas)
```

> **Note:** [`TI2VidOneStagePipeline`](src/ltx_pipelines/ti2vid_one_stage.py) is primarily for educational purposes. For best quality, use two-stage pipelines ([`TI2VidTwoStagesPipeline`](src/ltx_pipelines/ti2vid_two_stages.py), [`ICLoraPipeline`](src/ltx_pipelines/ic_lora.py), [`KeyframeInterpolationPipeline`](src/ltx_pipelines/keyframe_interpolation.py), or [`DistilledPipeline`](src/ltx_pipelines/distilled.py)).

### Features Comparison

| Pipeline | Stages | CFG | Upsampling | Conditioning | Best For |
| -------- | ------ | --- | ---------- | ------------- | -------- |
| **TI2VidTwoStagesPipeline** | 2 | ✅ | ✅ | Image | **Production quality** (recommended) |
| **TI2VidOneStagePipeline** | 1 | ✅ | ❌ | Image | Educational, prototyping |
| **DistilledPipeline** | 2 | ❌ | ✅ | Image | Fastest inference (8 sigmas) |
| **ICLoraPipeline** | 2 | ✅ | ✅ | Image + Video | Video-to-video transformations |
| **KeyframeInterpolationPipeline** | 2 | ✅ | ✅ | Keyframes | Animation, interpolation |

---

## 📦 Available Pipelines

### 1. TI2VidTwoStagesPipeline

**Best for:** High-quality text-to-video generation with upsampling. **Recommended for production use.**

**Source**: [`src/ltx_pipelines/ti2vid_two_stages.py`](src/ltx_pipelines/ti2vid_two_stages.py)

Two-stage generation: Stage 1 generates low-resolution video with CFG guidance, Stage 2 upsamples to 2x resolution with distilled LoRA refinement. Supports image conditioning. Highest quality output, slower than one-stage but significantly better quality.

**Use when:** Production-quality video generation, higher resolution needed, quality over speed, text-to-video with image conditioning.

---

### 2. TI2VidOneStagePipeline

**Best for:** Educational purposes and quick prototyping.

**Source**: [`src/ltx_pipelines/ti2vid_one_stage.py`](src/ltx_pipelines/ti2vid_one_stage.py)

> **⚠️ Important:** This pipeline is primarily for educational purposes. For production-quality results, use `TI2VidTwoStagesPipeline` or other two-stage pipelines.

Single-stage generation (no upsampling) with CFG guidance and image conditioning support. Faster inference but lower resolution output (typically 512x768).

**Use when:** Learning how the pipeline works, quick prototyping, testing, or when high resolution is not needed.

---

### 3. DistilledPipeline

**Best for:** Fastest inference with good quality using a distilled model with predefined sigma schedule.

**Source**: [`src/ltx_pipelines/distilled.py`](src/ltx_pipelines/distilled.py)

Two-stage generation with 8 predefined sigmas (8 steps in stage 1, 4 steps in stage 2). No CFG guidance required. Fastest inference among all pipelines. Supports image conditioning. Requires spatial upsampler.

**Use when:** Fastest inference is critical, batch processing many videos, or when you have a distilled model checkpoint.

---

### 4. ICLoraPipeline

**Best for:** Video-to-video and image-to-video transformations using IC-LoRA.

**Source**: [`src/ltx_pipelines/ic_lora.py`](src/ltx_pipelines/ic_lora.py)

Two-stage generation with IC-LoRA support. Can condition on reference videos (video-to-video) or images at specific frames. CFG guidance in stage 1, upsampling in stage 2. Requires IC-LoRA trained model.

**Use when:** Video-to-video transformations, image-to-video with strong control, or when you have reference videos to guide generation.

---

### 5. KeyframeInterpolationPipeline

**Best for:** Generating videos by interpolating between keyframe images.

**Source**: [`src/ltx_pipelines/keyframe_interpolation.py`](src/ltx_pipelines/keyframe_interpolation.py)

Two-stage generation with keyframe interpolation. Uses guiding latents (additive conditioning) instead of replacing latents for smoother transitions. CFG guidance in stage 1, upsampling in stage 2.

**Use when:** You have keyframe images and want to interpolate between them, creating smooth transitions, or animation/motion interpolation tasks.

---

## 🎨 Conditioning Types

Pipelines use different conditioning methods from [`ltx-core`](../ltx-core/) for controlling generation. See the [ltx-core conditioning documentation](../ltx-core/README.md#conditioning--control) for details.

### Image Conditioning

All pipelines support image conditioning, but with different methods:

- **Replacing Latents** ([`image_conditionings_by_replacing_latent`](src/ltx_pipelines/utils/helpers.py)):
  - Used by: `TI2VidOneStagePipeline`, `TI2VidTwoStagesPipeline`, `DistilledPipeline`, `ICLoraPipeline`
  - Replaces the latent at a specific frame with the encoded image
  - Strong control over specific frames

- **Guiding Latents** ([`image_conditionings_by_adding_guiding_latent`](src/ltx_pipelines/utils/helpers.py)):
  - Used by: `KeyframeInterpolationPipeline`
  - Adds the image as a guiding signal rather than replacing
  - Better for smooth interpolation between keyframes

### Video Conditioning

- **Video Conditioning** (ICLoraPipeline only):
  - Conditions on entire reference videos
  - Useful for video-to-video transformations
  - Uses `VideoConditionByKeyframeIndex` from [`ltx-core`](../ltx-core/)

---

## ⚡ Optimization Tips


### Memory Optimization

**FP8 Transformer (Lower Memory Footprint):**

For smaller GPU memory footprint, use the `enable-fp8` flag and use the `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` environment variable.

**CLI:**

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m ltx_pipelines.ti2vid_one_stage --enable-fp8 --checkpoint-path=...
```

**Programmatically:**

When authoring custom scripts, pass the `fp8transformer` flag to pipeline classes or construct your own by analogy:

```python
pipeline = TI2VidTwoStagesPipeline(
    checkpoint_path=ltx_model_path,
    distilled_lora_path=distilled_lora_path,
    distilled_lora_strength=0.6,
    spatial_upsampler_path=upsampler_path,
    gemma_root=gemma_root_path,
    loras=[],
    fp8transformer=True,
)
pipeline(...)
```

You still need to use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` when launching:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python my_denoising_pipeline.py
```

**Memory Cleanup Between Stages:**

By default, pipelines clean GPU memory (especially transformer weights) between stages. If you have enough memory, you can skip this cleanup to reduce running time:

```python
# In pipeline implementations, memory cleanup happens automatically
# between stages. For custom pipelines, you can skip:
# utils.cleanup_memory()  # Comment out if you have enough VRAM
```

### Denoising Loop Optimization

**Gradient Estimation Denoising Loop:**

Instead of the standard Euler denoising loop, you can use gradient estimation for fewer steps (~20-30 instead of 40):

```python
from ltx_pipelines.utils.helpers import gradient_estimating_euler_denoising_loop

# Use gradient estimation denoising loop
def denoising_loop(sigmas, video_state, audio_state, stepper):
    return gradient_estimating_euler_denoising_loop(
        sigmas=sigmas,
        video_state=video_state,
        audio_state=audio_state,
        stepper=stepper,
        denoise_fn=your_denoise_function,
        ge_gamma=2.0,  # Gradient estimation coefficient
    )
```

This allows you to use **20-30 steps instead of 40** while maintaining quality. The gradient estimation function is available in [`pipeline_utils.py`](src/ltx_pipelines/utils/helpers.py).

---

## 🔧 Requirements

- **LTX-2 Model Checkpoint** - Local `.safetensors` file
- **Gemma Text Encoder** - Local Gemma model directory
- **Spatial Upscaler** - Required for two-stage pipelines (except one-stage)
- **Distilled LoRA** - Required for two-stage pipelines (except one-stage and distilled)

---

## 📖 Example: Image-to-Video

```python
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

pipeline = TI2VidTwoStagesPipeline(
    checkpoint_path="/path/to/checkpoint.safetensors",
    distilled_lora_path="/path/to/distilled_lora.safetensors",
    spatial_upsampler_path="/path/to/upsampler.safetensors",
    gemma_root="/path/to/gemma",
    loras=[],
)

# Generate video from image
pipeline(
    prompt="A serene landscape with mountains in the background",
    output_path="output.mp4",
    seed=42,
    height=512,
    width=768,
    num_frames=121,
    frame_rate=25.0,
    num_inference_steps=40,
    cfg_guidance_scale=3.0,
    images=[("input_image.jpg", 0, 1.0)],  # Image at frame 0, strength 1.0
)
```

---

## 🔗 Related Projects

- **[LTX-Core](../ltx-core/)** - Core model implementation and inference components (schedulers, guiders, noisers, patchifiers)
- **[LTX-Trainer](../ltx-trainer/)** - Training and fine-tuning tools
