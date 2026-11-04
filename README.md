# LTX-2

[![Website](https://img.shields.io/badge/Website-LTX-181717?logo=google-chrome)](https://ltx.io)
[![Model](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/Lightricks/LTX-2.5)
[![Demo](https://img.shields.io/badge/Demo-Try%20Now-brightgreen?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAYAAACNiR0NAAAAAXNSR0IArs4c6QAAAERlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAA6ABAAMAAAABAAEAAKACAAQAAAABAAAAFKADAAQAAAABAAAAFAAAAACy3fD9AAACmElEQVQ4Ea1VP2haYRA/fRo0mESRIIqb2IwxuNUl0CGFQBC6OAWcikMottCpqYtDQIgdQsBFhAjZqiQhbhmySJBOgmNU0EGCg9r61Bivd0ffoykE0iQH37/77n7f3e/uqQFIPB7P/N3d3QeDwfAFEedZ91ghnyH5JM1m87dWq6UavF6vdTKZfDcajW/p4rE49+wIFMj33Gq1vlNo+kxg758KpiETqP/29vaXweVyqaS0aBfPXEfGFwTjWCwM+KBQoWA4HAJx/KDNvxcmTTGbzYAH8SljOp2C2+2GjY0NqNfrcHFxAXNzc2LDfCuKIq78KBdFOwsgGzidTnA4HHBzcwO9Xg8sFgtsbm7C3t4eVCoVaDQa0O12YXl5GUwmk5z5cZ/PB6PRCNrttgADFQUXFhbw8PAQVVXF3d1dJAeMx+P0zn0Jh8OYz+eRADCRSGAqlcLxeIz7+/u4tLSEjKUDZrNZ8U4mk0jR4fr6Op6enoru+voa0+k0rq2tYTAYxE6ng9QiSLRgrVZDv9+PFLkA6kUhT+GEC8C8XF5ewtHRkejICShiaDabwPvj42NJm3k7ODiQdDl9Fr0ocqJpdXUVIpEIdz7Y7XZRr6ysQDQahXK5LORvbW1p5rC9vQ2UifAooBqHuVxO0vt72tnZwWq1qqtisRgWCgU5ZzIZPDk50fdUUEmZvxTmAgKBgAxunT/fJpRKJWmhUCgEVDi4uroSG46kWCzC4uKitNVgMICzszOhSgA5fiJZhp4Lbbh1KARpbF65D/lx3vMdP05Vlkf5zKIDyukFJi7N6AVwNAhVsdlsM+LsjaZ56sq8kyQUqs4P6rsAKV49B4x4Padf7Y9Kv9+fEmiBQH8S4Gsa5v8EHpL9VwL7xH8BvwEcd4ccVf02KQAAAABJRU5ErkJggg==)](https://console.ltx.video/playground)
[![Paper](https://img.shields.io/badge/Paper-PDF-EC1C24?logo=adobeacrobatreader&logoColor=white)](https://arxiv.org/abs/2601.03233)
[![Discord](https://img.shields.io/badge/Join-Discord-5865F2?logo=discord)](https://discord.gg/ltxplatform)

**LTX-2** is the first DiT-based audio-video foundation model that contains all core capabilities of modern video generation in one model: synchronized audio and video, high fidelity, multiple performance modes, production-ready outputs, API access, and open access.

<div align="center">
  <video src="https://github.com/user-attachments/assets/4414adc0-086c-43de-b367-9362eeb20228" width="70%" poster=""> </video>
</div>

## 🚀 Quick Start

Clone the repo

```bash
git clone https://github.com/Lightricks/LTX-2.git
cd LTX-2
```

Install the dependencies. The `natten` extra is the fastest backend for the diffusion video VAE below, and is Linux + CUDA only -- on Windows and macOS it is skipped automatically and decoding falls back to a Triton or eager implementation, so the same command works everywhere (see [neighborhood attention backends](packages/ltx-pipelines/docs/optimization.md#diffusion-vae-decoder))

```bash
uv sync --extra natten
```

Download the [models](https://huggingface.co/Lightricks/LTX-2.5) or use the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)

```bash
hf auth login
hf download Lightricks/LTX-2.5 \
    diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
    text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
    vae/ltx-2.5-video-vae-bf16.safetensors \
    vae/ltx-2.5-audio-vae-bf16.safetensors \
    latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
    --local-dir models/ltx-2.5
```

That is roughly 66 GiB. The CLI keeps the repository's folder layout under `--local-dir`, which is why the paths below include `diffusion_models/`, `vae/` and so on.

If you get a 401/403, accept the model terms on Hugging Face and log in with a **Read** token (fine-grained tokens need the "read gated repos" scope enabled).

Generate

```bash
uv run python -m ltx_pipelines.distilled \
    --transformer-path       models/ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
    --text-encoder-path      models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
    --video-vae-path         models/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors \
    --audio-vae-path         models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors \
    --spatial-upsampler-path models/ltx-2.5/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
    --num-frames 121 \
    --seed 42 \
    --output-path output.mp4 \
    --prompt "A medium close-up shot features a Caucasian man with a beard, wearing a green and white baseball cap without any letters on the front, and a light blue shirt over a white t-shirt. He is positioned in the center of the frame, looking intently directly at the camera, his eyes focused on camera. His facial expression is one of deep concentration, with his brow slightly raised. As he looks straight at the camera, a quick sniff sound is heard, and then he speaks with a deep male voice and a satisfied tone, saying, 'I think it's so good.' The camera remains static throughout, maintaining a shallow depth of field, which keeps the man in sharp focus while the background is softly blurred, showing a beige wall behind him. After a brief pause, another short, audible sniff is heard. The man then continues to speak, his voice maintaining the same quality, as he states, 'So good. So good.' He elaborates further, emphasizing his point with a final statement, 'This got to be, it's got to be the best tool I've ever seen.'"
```

In cases of GPU memory constraints, consider `--quantization fp8-cast --offload {cpu, disk}`. See [additional flags](packages/ltx-pipelines/docs/installation.md#common-cli-flags).

This is **DistilledPipeline**: the fast starting point. For **production quality** (slower, more VRAM), run [DFR](#dfr-production-quality) below. For other capabilities, see [Models](#full-model-list) and [Pipelines](#available-pipelines).

### DFR (production quality)

**DFR** (Diffusion Fidelity Rendering) is the production-quality text/image-to-video path. It uses the **same distilled transformer** as the command above, plus a detailing IC-LoRA — extra generated keyframes and a spatial detailing pass. Expect longer runtime and more VRAM, not a different prompting style. Do not pass the full (dev) transformer.

Reuse the text encoder, VAEs, spatial upscaler, and distilled transformer from the first download; add the detailing IC-LoRA (separate repository):

```bash
hf download Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler \
    ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors \
    --local-dir models/ltx-2.5/loras
```

```bash
uv run python -m ltx_pipelines.dfr_pipeline \
    --transformer-path       models/ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
    --text-encoder-path      models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
    --video-vae-path         models/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors \
    --audio-vae-path         models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors \
    --detailing-lora         models/ltx-2.5/loras/ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors \
    --spatial-upsampler-path models/ltx-2.5/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
    --num-frames 121 \
    --seed 42 \
    --output-path output_dfr.mp4 \
    --prompt "A medium close-up shot features a Caucasian man with a beard, wearing a green and white baseball cap without any letters on the front, and a light blue shirt over a white t-shirt. He is positioned in the center of the frame, looking intently directly at the camera, his eyes focused on camera. His facial expression is one of deep concentration, with his brow slightly raised. As he looks straight at the camera, a quick sniff sound is heard, and then he speaks with a deep male voice and a satisfied tone, saying, 'I think it's so good.' The camera remains static throughout, maintaining a shallow depth of field, which keeps the man in sharp focus while the background is softly blurred, showing a beige wall behind him. After a brief pause, another short, audible sniff is heard. The man then continues to speak, his voice maintaining the same quality, as he states, 'So good. So good.' He elaborates further, emphasizing his point with a final statement, 'This got to be, it's got to be the best tool I've ever seen.'"
```

Defaults are 1024×1536 at 24 fps (`--temporal-upscalings 0`). UHD 4K is `--width 3840 --height 2176` (not 2160). `--temporal-upscalings 1` or `2` needs the [temporal upscaler](#full-model-list). Size, fps, and memory notes: [Running DFR](packages/ltx-pipelines/docs/pipelines.md#running-dfr).

### Full Model List

| Model | Model license |
| :--- | :--- |
| [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) | [LTX-2.x Community License](LICENSE-2_x) |
| [LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) | [LTX-2 Community License](LICENSE-2) |

The code in this repository is open source under the [Apache License 2.0](LICENSE). The LTX models are not part of that: they are released under their own licenses, listed above, which apply whenever you use the models, including with this code.

LTX-2.5 is the recommended model, and what the [Quick Start](#-quick-start) uses. Its weights are published as one file per component, so you download only the parts your pipeline needs.

Download from the [LTX-2.5 HuggingFace repository](https://huggingface.co/Lightricks/LTX-2.5):

**Transformer** (choose and download one of the following)
  * [`ltx-2.5-22b-dev-transformer-bf16.safetensors`](https://huggingface.co/Lightricks/LTX-2.5/blob/main/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2.5/resolve/main/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors) - the full model; used by the guided two-stage pipelines (TI2Vid, Keyframe, A2Vid)
  * [`ltx-2.5-22b-distilled-transformer-bf16.safetensors`](https://huggingface.co/Lightricks/LTX-2.5/blob/main/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2.5/resolve/main/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors) - runs in far fewer steps; what `DistilledPipeline`, [`DFRPipeline`](packages/ltx-pipelines/src/ltx_pipelines/dfr_pipeline.py), `ICLoraPipeline` and `DubItPipeline` expect

**Text Encoder** - Gemma 4 12B, fine-tuned for LTX, with the text projection bundled in; required by every pipeline. It is bundled with the model, so no separate Gemma download is needed. Google's stock Gemma 4 release is not a substitute: loading checks the encoder's version against the one the checkpoint was trained with (`gemma4-12b-ltx-v1`)
  * [`gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`](https://huggingface.co/Lightricks/LTX-2.5/blob/main/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2.5/resolve/main/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors)

**Video VAE** (choose and download one of the following)
  * [`ltx-2.5-video-vae-bf16.safetensors`](https://huggingface.co/Lightricks/LTX-2.5/blob/main/vae/ltx-2.5-video-vae-bf16.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2.5/resolve/main/vae/ltx-2.5-video-vae-bf16.safetensors) - diffusion decoder (`NADiffusionDecoder`); improved quality at the cost of longer decode time and more VRAM. Fastest with the `natten` extra, and falls back to Triton or eager neighborhood attention without it
  * [`ltx-2.5-video-vae-conv-bf16.safetensors`](https://huggingface.co/Lightricks/LTX-2.5/blob/main/vae/ltx-2.5-video-vae-conv-bf16.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2.5/resolve/main/vae/ltx-2.5-video-vae-conv-bf16.safetensors) - convolutional decoder; lighter and needs no extra dependencies

**Audio VAE** - required by the pipelines that generate or decode audio
  * [`ltx-2.5-audio-vae-bf16.safetensors`](https://huggingface.co/Lightricks/LTX-2.5/blob/main/vae/ltx-2.5-audio-vae-bf16.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2.5/resolve/main/vae/ltx-2.5-audio-vae-bf16.safetensors)

**Spatial Upscaler** - required by the two-stage pipeline implementations in this repository
  * [`ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors`](https://huggingface.co/Lightricks/LTX-2.5/blob/main/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2.5/resolve/main/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors)

**Temporal Upscaler** - required by [`DFRPipeline`](packages/ltx-pipelines/src/ltx_pipelines/dfr_pipeline.py) when running temporal refine rounds (`--temporal-upscalings`)
  * [`ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors`](https://huggingface.co/Lightricks/LTX-2.5/blob/main/latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2.5/resolve/main/latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors)

**Distilled LoRA** - required by the two-stage pipeline implementations that run the full model in stage 1 (TI2Vid two-stage / HQ, Keyframe, A2Vid; not DistilledPipeline, DFRPipeline, ICLoraPipeline, or DubItPipeline)
  * [`ltx-2.5-22b-distilled-lora-450-bf16.safetensors`](https://huggingface.co/Lightricks/LTX-2.5/blob/main/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2.5/resolve/main/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors)

**Detailing IC-LoRA** - required by [`DFRPipeline`](packages/ltx-pipelines/src/ltx_pipelines/dfr_pipeline.py)'s refinement stage (`--detailing-lora`). It lives in its own repository, [`LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler`](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler)
  * [`ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors`](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler/blob/main/ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler/resolve/main/ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors)

**Duration Head** - optional; lets you omit `--num-frames` and have the clip length predicted from the prompt
  * [`ltx-2.5-duration-head-bf16.safetensors`](https://huggingface.co/Lightricks/LTX-2.5/blob/main/model_patches/ltx-2.5-duration-head-bf16.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2.5/resolve/main/model_patches/ltx-2.5-duration-head-bf16.safetensors)

#### Legacy: LTX-2.3

Every pipeline in this repository also runs on LTX-2.3. Its checkpoints are single files bundling
the transformer, VAEs and text projection, under the [LTX-2 Community License](LICENSE-2). The Gemma 3
text encoder is downloaded separately and stays under Google's license. Files are not interchangeable
between the two models, and a LoRA only works with the model it was trained on.

See **[LTX-2.3 models](MODELS-LTX-2.3.md)** for the full list.

### Available Pipelines

* **[DistilledPipeline](packages/ltx-pipelines/src/ltx_pipelines/distilled.py)** - Fastest text/image-to-video (starting point)
* **[DFRPipeline](packages/ltx-pipelines/src/ltx_pipelines/dfr_pipeline.py)** - Production-quality text/image-to-video (slower, more VRAM): same distilled transformer, generated keyframes, spatial detailing, optional 2x/4x fps. How to run: [DFR in Quick Start](#dfr-production-quality) and [Running DFR](packages/ltx-pipelines/docs/pipelines.md#running-dfr)
* **[TI2VidTwoStagesPipeline](packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py)** - Guided two-stage text/image-to-video and keyframe interpolation with CFG/STG and 2x upsampling
* **[TI2VidTwoStagesHQPipeline](packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages_hq.py)** - Same guided two-stage flow with the res_2s sampler (fewer steps)
* **[TI2VidOneStagePipeline](packages/ltx-pipelines/src/ltx_pipelines/ti2vid_one_stage.py)** - Single-stage generation for quick prototyping
* **[ICLoraPipeline](packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py)** - Video-to-video and image-to-video transformations (uses distilled model.)
* **[A2VidPipelineTwoStage](packages/ltx-pipelines/src/ltx_pipelines/a2vid_two_stage.py)** - Audio-to-video generation conditioned on an input audio file
* **[RetakePipeline](packages/ltx-pipelines/src/ltx_pipelines/retake.py)** - Regenerate a specific time region of an existing video
* **[HDRICLoraPipeline](packages/ltx-pipelines/src/ltx_pipelines/hdr_ic_lora.py)** - Video-to-video SDR→HDR with an HDR IC-LoRA, writing a BT.2020/HLG master plus an ACEScct EXR sequence for EXR export and grading
* **[DubItPipeline](packages/ltx-pipelines/src/ltx_pipelines/dubit.py)** - Dub-It: rephrasing while matching speaker identity and lip movements (distilled model, single IC-LoRA, two stages).
* **Native HDR / EXR** — standard pipelines accept EXR stills and EXR-frame folders with `--hdr {SRGB_LINEAR,ACESCG,ACESCCT,ACES2065_1}` and write half EXR frames plus a BT.2020/HLG master. See [HDR Support](packages/ltx-pipelines/docs/hdr.md).

### ⚡ Optimization Tips

* **Use DistilledPipeline for speed** - Fastest inference with only 8 predefined sigmas (8 steps stage 1, 4 steps stage 2). For production quality, use [DFR](#dfr-production-quality) instead.
* **Enable FP8 quantization** - Enables lower memory footprint: `--quantization fp8-cast` (CLI) or `quantization=QuantizationPolicy.fp8_cast()` (Python). Fp8-cast should be used with bf16 checkpoints, it shall downcast them on the fly. On Hopper+ GPUs with native FP8 support, use `--quantization fp8-scaled-mm` for FP8 scaled matrix multiplication. Fp8-scaled-mm should be used with fp8 checkpoints.
* **Install attention optimizations** - On datacenter Blackwell GPUs (B200), install FlashAttention 4 manually: `uv pip install 'flash-attn-4==4.0.0b9'` (this specific revision is the one we have verified against torch 2.9.1+cu128; newer betas have known issues on consumer Blackwell). On Hopper GPUs, install the FlashAttention 3 wheel. On other CUDA GPUs, PyTorch SDPA is used automatically. An installed backend is selected automatically at runtime; forcing a specific one is a Python-API option (`AttentionFunction.FLASH_ATTENTION_3`/`FLASH_ATTENTION_4`), not a CLI flag.
* **Use gradient estimation** - Reduce inference steps from 40 to 20-30 while maintaining quality (see [pipeline documentation](packages/ltx-pipelines/docs/optimization.md#denoising-loop-optimization))
* **Skip memory cleanup** - If you have sufficient VRAM, disable automatic memory cleanup between stages for faster processing
* **Choose single-stage pipeline** - Use `TI2VidOneStagePipeline` for faster generation when high resolution isn't required

## ✍️ Prompting for LTX-2

When writing prompts, focus on detailed, chronological descriptions of actions and scenes. Include specific movements, appearances, camera angles, and environmental details - all in a single flowing paragraph. Start directly with the action, and keep descriptions literal and precise. Think like a cinematographer describing a shot list. Keep within 200 words. For best results, build your prompts using this structure:

- Start with main action in a single sentence
- Add specific details about movements and gestures
- Describe character/object appearances precisely
- Include background and environment details
- Specify camera angles and movements
- Describe lighting and colors
- Note any changes or sudden events

For additional guidance on writing a prompt please refer to <https://ltx.io/blog/prompting-guide-for-ltx-2>

### Automatic Prompt Enhancement

LTX-2 pipelines support automatic prompt enhancement via an `enhance_prompt` parameter.

## 🔌 ComfyUI Integration

To use our model with ComfyUI, please follow the instructions at <https://github.com/Lightricks/ComfyUI-LTXVideo/>.

## 📦 Packages

This repository is organized as a monorepo with three main packages:

* **[ltx-core](packages/ltx-core/)** - Core model implementation, inference stack, and utilities
* **[ltx-pipelines](packages/ltx-pipelines/)** - High-level pipeline implementations for text-to-video, image-to-video, and other generation modes
* **[ltx-trainer](packages/ltx-trainer/)** - Training and fine-tuning tools for LoRA, full fine-tuning, and IC-LoRA

Each package has its own README and documentation. See the [Documentation](#-documentation) section below.

## 📚 Documentation

Each package includes comprehensive documentation:

* **[LTX-Core README](packages/ltx-core/README.md)** - Core model implementation, inference stack, and utilities
* **[LTX-Pipelines README](packages/ltx-pipelines/README.md)** - High-level pipeline implementations and usage guides
* **[LTX-Trainer README](packages/ltx-trainer/README.md)** - Training and fine-tuning documentation with detailed guides
