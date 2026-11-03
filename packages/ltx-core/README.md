# LTX-Core

The foundational library for the LTX-2 Audio-Video generation model. This package contains the raw model definitions, component implementations, and loading logic used by `ltx-pipelines` and `ltx-trainer`.

## 📦 What's Inside?

- **`components/`**: Modular diffusion components (Schedulers, Guiders, Noisers, Patchifiers) following standard protocols
- **`conditioning/`**: Tools for preparing latent states and applying conditioning (image, video, keyframes, generated keyframe slots)
- **`guidance/`**: Perturbation system for fine-grained control over attention mechanisms
- **`loader/`**: Utilities for loading weights from `.safetensors`, fusing LoRAs, and managing memory
- **`loader/` streaming**: Memory-efficient inference that streams transformer units through reusable device buffers (from CPU memory or disk)
- **`model/`**: PyTorch implementations of the LTX-2 Transformer, Video VAE, Audio VAE, Vocoder and Upscaler
- **`text_encoders/gemma`**: Gemma text encoder implementation with tokenizers, feature extractors, and separate encoders for audio-video and video-only generation
- **`quantization/`**: FP8 (scaled MM, cast), blockwise FP8/FP6, and NVFP4 Linear (via `ltx-kernels`) for reduced memory / faster GEMM.

## 🚀 Quick Start

`ltx-core` provides the building blocks (models, components, and utilities) needed to construct inference flows. For ready-made inference pipelines use [`ltx-pipelines`](../ltx-pipelines/) or [`ltx-trainer`](../ltx-trainer/) for training.

## 🔧 Installation

```bash
# From the repository root
uv sync --frozen

# Or install as a package
pip install -e packages/ltx-core
```

## Building Blocks Overview

`ltx-core` provides modular components that can be combined to build custom inference flows:

### Core Models

- **Transformer** ([`model/transformer/`](src/ltx_core/model/transformer/)): The asymmetric dual-stream LTX-2 transformer (14B-parameter video stream, 5B-parameter audio stream) with bidirectional cross-modal attention for joint audio-video processing. Expects inputs in [`Modality`](src/ltx_core/model/transformer/modality.py) format
- **Video VAE** ([`model/video_vae/`](src/ltx_core/model/video_vae/)): Encodes/decodes video pixels to/from latent space with temporal and spatial compression
- **Audio VAE** ([`model/audio_vae/`](src/ltx_core/model/audio_vae/)): Encodes/decodes audio spectrograms to/from latent space
- **Vocoder** ([`model/audio_vae/`](src/ltx_core/model/audio_vae/)): Neural vocoder that converts mel spectrograms to audio waveforms
- **Text Encoder** ([`text_encoders/`](src/ltx_core/text_encoders/)): Gemma 3-based multilingual encoder with multi-layer feature extraction and thinking tokens that produces separate embeddings for video and audio conditioning
- **Spatial Upscaler** ([`model/upsampler/`](src/ltx_core/model/upsampler/)): Upsamples latent representations for higher-resolution generation

### Diffusion Components

- **Schedulers** ([`components/schedulers.py`](src/ltx_core/components/schedulers.py)): Noise schedules (LTX2Scheduler, LinearQuadratic, Beta) that control the denoising process
- **Guiders** ([`components/guiders.py`](src/ltx_core/components/guiders.py)): Guidance strategies (CFG, STG, APG) for controlling generation quality and adherence to prompts
- **Noisers** ([`components/noisers.py`](src/ltx_core/components/noisers.py)): Add noise to latents according to the diffusion schedule
- **Patchifiers** ([`components/patchifiers.py`](src/ltx_core/components/patchifiers.py)): Convert between spatial latents `[B, C, F, H, W]` and sequence format `[B, seq_len, dim]` for transformer processing

### Conditioning & Control

- **Conditioning** ([`conditioning/`](src/ltx_core/conditioning/)): Tools for preparing and applying various conditioning types (image, video, keyframes, and generated keyframe slots via `VideoGeneratedKeyframeSlots` -- extra model-generated frames at interior positions, requiring a checkpoint whose transformer config sets `use_keyframes_abs_pos_embedding`)
- **Guidance** ([`guidance/`](src/ltx_core/guidance/)): Perturbation system for fine-grained control over attention mechanisms (e.g., skipping specific attention layers)

### Utilities

- **Loader** ([`loader/`](src/ltx_core/loader/)): Model loading from `.safetensors`, LoRA fusion, weight remapping, and memory management
- **Quantization** ([`quantization/`](src/ltx_core/quantization/)): FP8, blockwise FP8/FP6, and NVFP4 Linear policies for reduced memory footprint and faster inference
- **Unit Streaming** ([`loader/`](src/ltx_core/loader/)): Streams transformer units through device buffers so a full model can run without keeping every weight on the accelerator
- **Colour / HDR** ([`color/`](src/ltx_core/color/), [`hdr.py`](src/ltx_core/hdr.py)): Primaries matrices, YUV/HLG helpers, and working-space transfers used by pipeline EXR/`--hdr` I/O. Pipeline-level docs: [HDR Support](../ltx-pipelines/docs/hdr.md).

### Loader

The `loader/` module builds models from `.safetensors` checkpoints in weight units. `SimpleBuilder` is a frozen
dataclass holding the recipe: a model configurator, checkpoint shards, `SDOps` stages that rename and transform
checkpoint keys, `ModuleOps` that rewrite the model shell, and ordered LoRA requests.

#### Basic usage

```python
from pathlib import Path

import torch
from ltx_core.loader.simple_builder import SimpleBuilder

builder = SimpleBuilder(MyModelConfigurator, (Path("/path/to/model.safetensors"),))
model = builder.build(device=torch.device("cuda"))
```

#### Loading LoRA adapters

Use `.lora()` to append adapters before calling `.build()`; they are fused in request order:

```python
from ltx_core.loader import SDOps

lora_sd_ops = (SDOps("identity").with_matching(),)  # or model-specific key-renaming SDOps

builder = (
    SimpleBuilder(MyModelConfigurator, (Path("/path/to/model.safetensors"),))
    .lora((Path("/path/to/lora_a.safetensors"),), 0.8, lora_sd_ops)
    .lora((Path("/path/to/lora_b.safetensors"),), 0.5, lora_sd_ops)
)
model = builder.build(device=torch.device("cuda"))
```

#### Registry and retention

Builders share a `ModelRegistry`. It caches model shells and checkpoint metadata, and keeps unit states only where a
placement policy places them: the registry's default `placement_policy`, or a builder's own
`with_placement_policy(...)`. With the default policy nothing is retained, so every build reads the checkpoint again.
To keep clean weights on the GPU across builds, and LoRA factors in pinned host memory:

```python
from ltx_core.loader.placement import Placement, RetentionPolicy
from ltx_core.loader.registry import ModelRegistry

registry = ModelRegistry(
    placement_policy=RetentionPolicy(
        clean=Placement(torch.device("cuda")),
        lora=Placement(torch.device("cpu"), pin_memory=True),
    )
)
builder = builder.with_registry(registry)
```

### Quantization

The `quantization/` module provides FP8 quantization support for the LTX-2 transformer, significantly reducing memory usage while maintaining quality. Two backends are available:

#### FP8 Scaled MM

Uses PyTorch's `torch._scaled_mm` for efficient FP8 matrix multiplication. Weights are stored in FP8 format with per-tensor scaling, and inputs are quantized dynamically.

**Usage with QuantizationPolicy:**

```python
from ltx_core.quantization.fp8_scaled_mm import build_policy as build_fp8_scaled_mm_policy

# Discovers the layer set from the checkpoint's .weight_scale tensors
policy = build_fp8_scaled_mm_policy("/path/to/checkpoint.safetensors")
```

The policy carries `sd_ops`, `module_ops`, and the LoRA `fuse_rule`; `compose_quantization` applies them to a builder:

```python
import torch
from ltx_core.loader.simple_builder import SimpleBuilder
from ltx_core.quantization.policy import compose_quantization

builder = compose_quantization(SimpleBuilder(MyModelConfigurator, (Path("/path/to/checkpoint.safetensors"),)), policy)
model = builder.build(device=torch.device("cuda"))
```

#### FP8 Cast

A simpler approach that casts weights to FP8 for storage and upcasts during inference:

```python
from ltx_core.quantization.fp8_cast import build_policy as build_fp8_cast_policy

policy = build_fp8_cast_policy("/path/to/checkpoint.safetensors")
```

### Unit Streaming

The [`loader/`](src/ltx_core/loader/) streaming builder keeps resident weights on the compute device and delivers each selected transformer unit through reusable device buffers just before its forward hook runs. `StreamingBuilder` wraps a `SimpleBuilder` checkpoint recipe and returns a `StreamingModel`, an `nn.Module` wrapper with explicit `close()` cleanup.

#### Strategies

Choose a backing mode in `StreamingConfig`:

- **RAM streaming** (`mode="cpu"`): selected units are loaded into CPU buffers during the build, then copied to the accelerator on demand.
- **Disk streaming** (`mode="disk"`): selected units are read from checkpoint shards on demand into reusable host buffers.

The streamed model owns its RAM-streamed host units and frees them on `close()`. Units outlive it only where a
placement policy retains them: the base builder's own (`with_placement_policy`) or its registry's default
(`ModelRegistry(placement_policy=...)`).

#### Basic usage

```python
import torch
from pathlib import Path
from ltx_core.loader.simple_builder import SimpleBuilder
from ltx_core.loader.streaming_builder import StreamingBuilder, StreamingConfig

base = SimpleBuilder(
    configurator=MyModelConfigurator,
    paths=(Path("/path/to/model.safetensors"),),
    sd_ops=(),  # add checkpoint-to-model key operations when needed
)
builder = StreamingBuilder(
    base,
    StreamingConfig(
        blocks_prefix="transformer_blocks",
        mode="disk",  # use "cpu" for RAM streaming
        prefetch_depth=2,
        device_buffers=2,
        host_buffers=2,
    ),
)
model = builder.build(device=torch.device("cuda"), dtype=torch.bfloat16)
try:
    output = model(inputs)
finally:
    model.close()
```

For complete, production-ready pipeline implementations that combine these building blocks, see the [`ltx-pipelines`](../ltx-pipelines/) package.

---

# Architecture Overview

This section provides a deep dive into the internal architecture of the LTX-2 Audio-Video generation model.

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [The Transformer](#the-transformer)
3. [Video VAE](#video-vae)
4. [Audio VAE](#audio-vae)
5. [Text Encoding (Gemma)](#text-encoding-gemma)
6. [Spatial Upscaler](#spatial-upsampler)
7. [Data Flow](#data-flow)

---

## High-Level Architecture

LTX-2 is an **asymmetric dual-stream diffusion transformer** that jointly models the text-conditioned distribution of video and audio signals, capturing true joint dependencies (unlike sequential T2V→V2A pipelines).

### Key Design Principles

- **Decoupled Latent Representations**: Separate modality-specific VAEs enable 3D RoPE (video) vs 1D RoPE (audio), independent compression optimization, and native V2A/A2V editing workflows
- **Asymmetric Dual-Stream**: 14B-parameter video stream (spatiotemporal dynamics) + 5B-parameter audio stream (1D temporal), sharing 48 transformer blocks but differing in width
- **Bidirectional Cross-Modal Attention**: 1D temporal RoPE enables sub-frame alignment, mapping visual cues to auditory events (lip-sync, foley, environmental acoustics)
- **Cross-Modality AdaLN**: Scaling/shift parameters conditioned on the other modality's hidden states for synchronization across differing diffusion timesteps/temporal resolutions

```text
┌─────────────────────────────────────────────────────────────┐
│                    INPUT PREPARATION                        │
│                                                             │
│  Video Pixels → Video VAE Encoder → Video Latents           │
│  Audio Waveform → Audio VAE Encoder → Audio Latents         │
│  Text Prompt → Gemma 3 Encoder → Text Embeddings            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│     LTX-2 ASYMMETRIC DUAL-STREAM TRANSFORMER (48 Blocks)    │
│                                                             │
│  ┌──────────────────────┐      ┌──────────────────────┐     │
│  │  Video Stream (14B)  │      │  Audio Stream (5B)   │     │
│  │                      │      │                      │     │
│  │  3D RoPE (x,y,t)     │      │  1D RoPE (temporal)  │     │
│  │                      │      │                      │     │
│  │  Self-Attn           │      │  Self-Attn           │     │
│  │  Text Cross-Attn     │      │  Text Cross-Attn     │     │
│  │                      │◄────►│                      │     │
│  │  A↔V Cross-Attn      │      │  A↔V Cross-Attn      │     │
│  │  (1D temporal RoPE)  │      │  (1D temporal RoPE)  │     │
│  │  Cross-modality      │      │  Cross-modality      │     │
│  │  AdaLN               │      │  AdaLN               │     │
│  │  Feed-Forward        │      │  Feed-Forward        │     │
│  └──────────────────────┘      └──────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT DECODING                          │
│                                                             │
│  Video Latents → Video VAE Decoder → Video Pixels           │
│  Audio Latents → Audio VAE Decoder → Mel Spectrogram        │
│  Mel Spectrogram → Vocoder → Audio Waveform (24 kHz)        │
└─────────────────────────────────────────────────────────────┘
```

---

## The Transformer

The core of LTX-2 is an **asymmetric dual-stream diffusion transformer** with 48 layers that processes both video and audio tokens simultaneously. The architecture allocates 14B parameters to the video stream and 5B parameters to the audio stream, reflecting the different information densities of the two modalities.

### Model Structure

**Source**: [`src/ltx_core/model/transformer/model.py`](src/ltx_core/model/transformer/model.py)

The `LTXModel` class implements the transformer. It supports both video-only and audio-video generation modes. For actual usage, see the [`ltx-pipelines`](../ltx-pipelines/) package which handles model loading and initialization.

### Transformer Block Architecture

**Source**: [`src/ltx_core/model/transformer/transformer.py`](src/ltx_core/model/transformer/transformer.py)

Each dual-stream block performs four operations sequentially:

1. **Self-Attention**: Within-modality attention for each stream
2. **Text Cross-Attention**: Textual prompt conditioning for both streams
3. **Audio-Visual Cross-Attention**: Bidirectional inter-modal exchange
4. **Feed-Forward Network (FFN)**: Feature refinement

```text
┌─────────────────────────────────────────────────────────────┐
│                    TRANSFORMER BLOCK                        │
│                                                             │
│  VIDEO (14B): Input → RMSNorm → AdaLN → Self-Attn →         │
│              RMSNorm → Text Cross-Attn →                    │
│              RMSNorm → AdaLN → A↔V Cross-Attn (1D RoPE) →   │
│              RMSNorm → AdaLN → FFN → Output                 │
│                                                             │
│  AUDIO (5B):  Input → RMSNorm → AdaLN → Self-Attn →         │
│              RMSNorm → Text Cross-Attn →                    │
│              RMSNorm → AdaLN → A↔V Cross-Attn (1D RoPE) →   │
│              RMSNorm → AdaLN → FFN → Output                 │
│                                                             │
│  RoPE: Video=3D (x,y,t), Audio=1D (t), Cross-Attn=1D (t)    │
│  AdaLN: Timestep-conditioned, cross-modality for A↔V CA     │
└─────────────────────────────────────────────────────────────┘
```

### Audio-Visual Cross-Attention Details

Bidirectional cross-attention enables tight temporal alignment: video and audio streams exchange information bidirectionally using 1D temporal RoPE (synchronization only, no spatial alignment). AdaLN gates condition on each modality's timestep for cross-modal synchronization.

### Perturbations

The transformer supports [**perturbations**](src/ltx_core/guidance/perturbations.py) that selectively skip attention operations.

Perturbations allow you to disable specific attention mechanisms during inference, which is useful for guidance techniques like STG (Spatio-Temporal Guidance).

**Supported Perturbation Types**:

- `SKIP_VIDEO_SELF_ATTN`: Skip video self-attention
- `SKIP_AUDIO_SELF_ATTN`: Skip audio self-attention
- `SKIP_A2V_CROSS_ATTN`: Skip audio-to-video cross-attention
- `SKIP_V2A_CROSS_ATTN`: Skip video-to-audio cross-attention

Perturbations are used internally by guidance mechanisms like STG (Spatio-Temporal Guidance). For usage examples, see the [`ltx-pipelines`](../ltx-pipelines/) package.

---

## Video VAE

The Video VAE ([`src/ltx_core/model/video_vae/`](src/ltx_core/model/video_vae/)) encodes video pixels into latent representations and decodes them back.

### Architecture

- **Encoder**: Compresses `[B, 3, F, H, W]` pixels → `[B, 128, F', H/32, W/32]` latents
  - Where `F' = 1 + (F-1)/8` (frame count must satisfy `(F-1) % 8 == 0`)
  - Example: `[B, 3, 33, 512, 512]` → `[B, 128, 5, 16, 16]`
- **Decoder**: Expands `[B, 128, F, H, W]` latents → `[B, 3, F', H*32, W*32]` pixels
  - Where `F' = 1 + (F-1)*8`
  - Example: `[B, 128, 5, 16, 16]` → `[B, 3, 33, 512, 512]`

`ConvVideoDecoder` (above) is the default, single-forward-pass decoder. `model/video_vae/` also has `DiffusionVideoDecoder` -- a neighborhood-attention decoder (for best performance requires the `natten` extra: `uv sync --package ltx-core --extra natten`; without it, Triton or eager fallbacks are used) that iteratively denoises pixels via Euler steps (`default_num_inference_steps=2` for distilled) instead of a single deterministic pass. `VideoDecoderConfigurator.from_config` picks between the two based on the checkpoint's own `vae._class_name`. `DIFFUSION_VAE_DECODER_COMFY_KEYS_FILTER` is its checkpoint key-remapping `SDOps`. Pipeline-side DiffVAE presets live in `DiffVAEMode` / `apply_diffvae_mode` (default `CHUNKED_EAGER`; CLI `--diffvae-optimization`). See `ltx-pipelines` docs for relative compile/runtime/VRAM tradeoffs between modes.

> **DiffVAE + CUDA illegal memory access:** the `natten` extra pins `natten==0.21.7+torch2130cu132` with `torch==2.13.0` (cu132). Older PyTorch/NVIDIA stacks can IMA inside NATTEN TokPerm on large stage-5 volumes. If that happens, upgrade CUDA / PyTorch / natten to those pins (see also [pipelines optimization docs](../ltx-pipelines/docs/optimization.md#diffusion-vae-decoder)).

The Video VAE is used internally by pipelines for encoding video pixels to latents and decoding latents back to pixels. For usage examples, see the [`ltx-pipelines`](../ltx-pipelines/) package.

---

## Audio VAE

The Audio VAE ([`src/ltx_core/model/audio_vae/`](src/ltx_core/model/audio_vae/)) processes audio spectrograms.

### Audio VAE Architecture

Compact neural audio representation optimized for diffusion-based training. Natively supports stereo: processes two-channel mel-spectrograms (16 kHz input) with channel concatenation before encoding.

- **Encoder**: `[B, mel_bins, T]` → `[B, 8, T/4, 16]` latents (4× temporal downsampling, 8 channels, 16 mel bins in latent space, ~1/25s per token, 128-dim feature vector)
- **Decoder**: `[B, 8, T, 16]` → `[B, mel_bins, T*4]` mel spectrogram
- **Vocoder**: HiFi-GAN-based, modified for stereo synthesis and upsampling (16 kHz mel → 24 kHz waveform, doubled generator capacity for stereo)

**Downsampling**:

- Temporal: 4× (time steps)
- Frequency: Variable (input mel_bins → fixed 16 in latent space)

The Audio VAE is used internally by pipelines for encoding mel spectrograms to latents and decoding latents back to mel spectrograms. The vocoder converts mel spectrograms to audio waveforms. For usage examples, see the [`ltx-pipelines`](../ltx-pipelines/) package.

---

## Text Encoding (Gemma)

LTX-2 uses Gemma as the multilingual text encoder backbone, located in [`src/ltx_core/text_encoders/gemma/`](src/ltx_core/text_encoders/gemma/). LTX-2.3 uses stock **Gemma 3** (12B), downloaded separately; LTX-2.5 uses a **Gemma 4** (unified) fine-tuned for LTX and shipped with the model. Advanced text understanding is critical not only for global language support but for the phonetic and semantic accuracy of generated speech.

### Tokenization and BOS

Encoding goes through [`LTXGemmaTokenizer`](src/ltx_core/text_encoders/gemma/tokenizer.py), which always ensures a leading `<bos>`:

| Family | Tokenizer assets prepend `<bos>`? | Encode path |
|--------|-----------------------------------|-------------|
| Gemma 3 | Yes (`tokenizer.json` post_processor) | Leading `<bos>` already from assets |
| Gemma 4 | No | We prepend `<bos>` so both families match |

EOS is not appended on the encode path.

### Text Encoder Architecture

The text conditioning pipeline consists of three stages:

1. **Gemma 3 Backbone**: Decoder-only LLM processes text tokens → embeddings across all layers `[B, T, D, L]`
2. **Multi-Layer Feature Extractor**: Aggregates features from all decoder layers (not just final layer), applies mean-centered scaling, flattens to `[B, T, D×L]`, and projects via learnable matrix W (jointly optimized with LTX-2, LLM weights frozen)
3. **Text Connector**: Bidirectional transformer blocks with learnable registers (replacing padded positions, also referred to as "thinking tokens" in the paper) for contextual mixing. Separate connectors for video and audio streams (`Embeddings1DConnector`)

**Encoders**:

- `AVGemmaTextEncoderModel`: Audio-video generation (two connectors → `AVGemmaEncoderOutput` with separate video/audio contexts)
- `VideoGemmaTextEncoderModel`: Video-only generation (single connector → `VideoGemmaEncoderOutput`)

### System Prompts

System prompts are also used to enhance user's prompts.

- **Text-to-Video**: [`gemma_t2v_system_prompt.txt`](src/ltx_core/text_encoders/gemma/encoders/prompts/gemma_t2v_system_prompt.txt)
- **Image-to-Video**: [`gemma_i2v_system_prompt.txt`](src/ltx_core/text_encoders/gemma/encoders/prompts/gemma_i2v_system_prompt.txt)

**Important**: Video and audio receive **different** context embeddings, even from the same prompt. This allows better modality-specific conditioning and enables the model to synthesize speech that is synchronized with visual lip movement while being natural in cadence, accent, and emotional tone.

**Output Format**:

- Video context: `[B, seq_len, 4096]` - Video-specific text embeddings
- Audio context: `[B, seq_len, 2048]` - Audio-specific text embeddings

The text encoder is used internally by pipelines. For usage examples, see the [`ltx-pipelines`](../ltx-pipelines/) package.

---

## Upscaler

The Upscaler ([`src/ltx_core/model/upsampler/`](src/ltx_core/model/upsampler/)) upsamples latent representations for higher-resolution output.

The spatial upsampler is used internally by two-stage pipelines (e.g., [`TI2VidTwoStagesPipeline`](../ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py), [`ICLoraPipeline`](../ltx-pipelines/src/ltx_pipelines/ic_lora.py)) to upsample low-resolution latents before final VAE decoding. For usage examples, see the [`ltx-pipelines`](../ltx-pipelines/) package.

---

## Data Flow

### Complete Generation Pipeline

Here's how all the components work together conceptually ([`src/ltx_core/components/`](src/ltx_core/components/)):

**Pipeline Steps**:

1. **Text Encoding**: Text prompt → Gemma encoder → separate video/audio embeddings
2. **Latent Initialization**: Initialize noise latents in spatial format `[B, C, F, H, W]`
3. **Patchification**: Convert spatial latents to sequence format `[B, seq_len, dim]` for transformer
4. **Sigma Schedule**: Generate noise schedule (adapts to token count)
5. **Denoising Loop**: Iteratively denoise using transformer predictions
   - Create Modality inputs with per-token timesteps and RoPE positions
   - Forward pass through transformer (conditional and unconditional for CFG)
   - Apply guidance (CFG, STG, etc.)
   - Update latents using diffusion step (Euler, etc.)
6. **Unpatchification**: Convert sequence back to spatial format
7. **VAE Decoding**: Decode latents to pixel space (with optional upsampling for two-stage)

- [`DFRPipeline`](../ltx-pipelines/src/ltx_pipelines/dfr_pipeline.py) - Production-quality text/image-to-video
- [`TI2VidTwoStagesPipeline`](../ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py) - Guided two-stage text/image-to-video and keyframe interpolation
- [`ICLoraPipeline`](../ltx-pipelines/src/ltx_pipelines/ic_lora.py) - Video-to-video with IC-LoRA control
- [`DistilledPipeline`](../ltx-pipelines/src/ltx_pipelines/distilled.py) - Fast inference with distilled model

See the [ltx-pipelines README](../ltx-pipelines/README.md) for usage examples.

## 🔗 Related Projects

- **[ltx-pipelines](../ltx-pipelines/)** - High-level pipeline implementations for text-to-video, image-to-video, and video-to-video
- **[ltx-trainer](../ltx-trainer/)** - Training and fine-tuning tools
