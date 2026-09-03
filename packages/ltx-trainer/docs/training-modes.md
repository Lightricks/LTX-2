# Training Modes Guide

The trainer uses the **flexible** training strategy (`name: "flexible"`) — a unified conditioning framework that
supports all training modes through configuration. Every scenario is expressed by setting `is_generated` on each
modality and adding optional conditions, rather than choosing a separate strategy class.

## Key Concepts

Before diving into individual modes, here are the core ideas behind the flexible strategy:

- **`is_generated: true`** — the modality is denoised during training and contributes to the loss. This is the
  modality the model learns to generate.
- **`is_generated: false`** — the modality is frozen (sigma=0, no noise, no loss). It passes through the transformer
  clean and acts as cross-modal conditioning for the generated modality.
- **At least one modality must have `is_generated: true`.**
- **Conditions** are per-modality and can be composed (e.g., `reference` + `first_frame` together on the video
  modality).
- Audio does **not** support `first_frame` or `spatial_crop` conditions — only `prefix`, `suffix`, `mask`,
  and `reference`.

> [!TIP]
> If you are using an agent-enabled environment with repository skills and are unsure which mode to choose,
> ask for the [`train-model`](../../../.claude/skills/train-model/SKILL.md) skill. It maps your intent to one of
> these configs and walks through dataset preparation, preprocessing, launch, and monitoring.

## 📊 Quick Reference

| Mode                  | Video     | Audio     | Conditions          | Config |
|-----------------------|-----------|-----------|---------------------|--------|
| **T2V**               | Generated | Generated | —                   | [`t2v_lora`](../configs/t2v_lora.yaml) |
| **I2V**               | Generated | Generated | `first_frame`       | [`i2v_lora`](../configs/i2v_lora.yaml) |
| **Video Extension**   | Generated | Generated | `prefix`/`suffix`   | [`video_extend_lora`](../configs/video_extend_lora.yaml) |
| **V2V IC-LoRA**       | Generated | —         | `reference`         | [`v2v_ic_lora`](../configs/v2v_ic_lora.yaml) |
| **Reference + source phase** | Generated | —  | `reference`         | [`v2v_ic_lora_reference_phase`](../configs/v2v_ic_lora_reference_phase.yaml) |
| **A2V**               | Generated | Frozen    | —                   | [`a2v_lora`](../configs/a2v_lora.yaml) |
| **V2A (Foley)**       | Frozen    | Generated | —                   | [`v2a_lora`](../configs/v2a_lora.yaml) |
| **Video Inpainting**  | Generated | —         | `mask`              | [`video_inpainting_lora`](../configs/video_inpainting_lora.yaml) |
| **Video Outpainting** | Generated | —         | `spatial_crop`      | [`video_outpainting_lora`](../configs/video_outpainting_lora.yaml) |
| **T2A**               | —         | Generated | —                   | [`t2a_lora`](../configs/t2a_lora.yaml) |
| **Audio Extension**   | —         | Generated | `prefix`/`suffix`   | [`audio_extend_lora`](../configs/audio_extend_lora.yaml) |
| **Audio Inpainting**  | —         | Generated | `mask`              | [`audio_inpainting_lora`](../configs/audio_inpainting_lora.yaml) |
| **A2A IC-LoRA**       | —         | Generated | `reference`         | [`a2a_ic_lora`](../configs/a2a_ic_lora.yaml) |
| **AV2AV IC-LoRA**     | Generated | Generated | `reference` (both)  | [`av2av_ic_lora`](../configs/av2av_ic_lora.yaml) |

---

## 🎯 Text-to-Video (T2V)

Generate video and audio from text prompts. Both modalities are denoised with no additional conditions.

```yaml
training_strategy:
  name: "flexible"
  video:
    is_generated: true
    latents_dir: "latents"
  audio:
    is_generated: true
    latents_dir: "audio_latents"
```

**Example config:** 📄 [t2v_lora.yaml](../configs/t2v_lora.yaml)

---

## 🖼️ Image-to-Video (I2V)

Generate video conditioned on a starting image. The first frame is provided as a clean conditioning signal — no noise,
timestep=0, excluded from loss. The `probability` parameter controls how often first-frame conditioning is applied;
remaining samples train in pure T2V mode.

```yaml
training_strategy:
  name: "flexible"
  video:
    is_generated: true
    latents_dir: "latents"
    conditions:
      - type: first_frame
        probability: 0.5
  audio:
    is_generated: true
    latents_dir: "audio_latents"
```

**Example config:** 📄 [i2v_lora.yaml](../configs/i2v_lora.yaml)

---

## ⏩ Video Extension

Extend a video forward (or backward) in time. Prefix or suffix conditioning provides a span of existing latent frames
as clean conditioning. The `temporal_boundary` sets the number of **latent frames** used as context. For the default
VAE, each latent frame represents 8 pixel frames; the trainer derives the actual factor from the checkpoint.

```yaml
training_strategy:
  name: "flexible"
  video:
    is_generated: true
    latents_dir: "latents"
    conditions:
      - type: prefix            # or "suffix" for backward extension
        temporal_boundary: 8    # 8 latent frames = 64 pixel frames for the default VAE
        probability: 1.0
  audio:
    is_generated: true
    latents_dir: "audio_latents"
```

> [!NOTE]
> The `prefix` and `suffix` conditions also work on the audio modality for audio extension.
> Set `temporal_boundary` on the audio modality's conditions list to condition on a prefix or suffix
> of the audio latents.

**Example configs:** 📄 [video_extend_lora.yaml](../configs/video_extend_lora.yaml) (forward), 📄 [video_suffix_lora.yaml](../configs/video_suffix_lora.yaml) (backward)

---

## 🔄 IC-LoRA / Video-to-Video (V2V)

In-Context LoRA learns transformations from paired videos. Pre-encoded reference latents are concatenated to the target
sequence — reference tokens participate in bidirectional self-attention but receive no noise and are excluded from loss.
This enables control adapters (depth, pose), style transfer, deblurring, colorization, and more.

```yaml
training_strategy:
  name: "flexible"
  video:
    is_generated: true
    latents_dir: "latents"
    conditions:
      - type: reference
        latents_dir: "reference_latents"
        probability: 1.0
      - type: first_frame       # optional — composable with reference
        probability: 0.2
```

> [!NOTE]
> IC-LoRA is video-only by default (no audio modality block). Conditions can be composed — the example above also
> applies first-frame conditioning with 20% probability alongside the reference.
> Use [AV2AV IC-LoRA](#av2av-ic-lora) when both video and audio references should be trained jointly.

**Example config:** 📄 [v2v_ic_lora.yaml](../configs/v2v_ic_lora.yaml)

### Reference RoPE layouts and source phase

The `reference` condition can additionally control *where* the reference sits in the RoPE grid
(`layout`) and tag each source with an independent rotary phase (`source_phase`). Both are
opt-in; the defaults reproduce the behaviour above exactly.

The recommended starting point for reference/identity transfer is `layout: overlap` with
`source_phase: true` — the reference keeps the target's coordinates and the sources are told
apart by phase.

**Example config:** 📄 [v2v_ic_lora_reference_phase.yaml](../configs/v2v_ic_lora_reference_phase.yaml)

See [reference-conditioning-layouts.md](./reference-conditioning-layouts.md) for the full
description, the ST-DRC reference, multi-source configuration, and the training/inference
parity requirements.

### Dataset Requirements

- **Paired videos** — each target video has a corresponding reference video
- **Same frame count** between reference and target
- Reference videos can optionally be at **lower spatial resolution** (see [Scaled Reference](#scaled-reference-conditioning) below)
- Both must be **preprocessed** before training

**Dataset structure:**

```
preprocessed_data_root/
├── latents/            # Target video latents
├── conditions/         # Text embeddings
└── reference_latents/  # Reference video latents (conditioning input)
```

### Generating Reference Videos

Use the `compute_reference.py` script to generate reference videos (e.g., Canny edge maps) for a dataset:

```bash
uv run python scripts/compute_reference.py scenes_output_dir/ \
    --output scenes_output_dir/dataset.json
```

To compute a different condition (depth maps, pose skeletons, etc.), modify the `compute_reference()` function in the
script.

> [!NOTE]
> `compute_reference.py` writes generated references to the `reference_video` column, which
> `process_dataset.py` detects automatically. The legacy `ref_media_path` column is also accepted.

### Scaled Reference Conditioning

For more efficient training and inference, use **downscaled reference videos** while keeping targets at full
resolution. During training, the strategy infers the spatial and temporal scale factors from the preprocessed
reference and target latents and adjusts positional encodings accordingly. This reduces conditioning tokens, leading to:

- **Faster training** — shorter sequence lengths
- **Faster inference** — reduced memory usage
- **Same aspect ratio** maintained between reference and target

Preprocess with the `--reference-downscale-factor` option:

```bash
uv run python scripts/process_dataset.py dataset.json \
    --resolution-buckets 768x768x25 \
    --model-path /path/to/ltx-checkpoint.safetensors \
    --text-encoder-path /path/to/gemma-root \
    --reference-downscale-factor 2
```

> [!NOTE]
> This example uses a unified checkpoint. On a split pack (LTX 2.5) also pass `--video-vae-path` and
> `--audio-vae-path` — see [Dataset Preparation](dataset-preparation.md#basic-usage).

> [!NOTE]
> The `reference_video` column is auto-detected by convention — no `--reference-column` flag needed.

Validation encodes reference media on the fly, so set `downscale_factor` and `temporal_scale_factor`
on each `reference` validation condition to match the preprocessing factors:

```yaml
validation:
  samples:
    - prompt: "..."
      conditions:
        - type: reference
          video: "/path/to/reference.mp4"
          downscale_factor: 2
          temporal_scale_factor: 1
          include_in_output: true
```

> [!NOTE]
> The scale factor must be a positive integer, and all dimensions must be divisible by the VAE
> spatial factor (32 for the default VAE). Common values are 1 (no scaling), 2 (half resolution),
> or 4 (quarter resolution).

---

## 🔊 Audio-to-Video (A2V)

Generate video conditioned on frozen audio. Audio passes through the transformer clean (sigma=0) and influences video
via the built-in cross-modal attention. Only video is denoised.

```yaml
training_strategy:
  name: "flexible"
  video:
    is_generated: true
    latents_dir: "latents"
  audio:
    is_generated: false
    latents_dir: "audio_latents"
```

**Example config:** 📄 [a2v_lora.yaml](../configs/a2v_lora.yaml)

---

## 🎵 Video-to-Audio / Foley (V2A)

Generate audio (Foley) conditioned on frozen video. Video passes through the transformer clean (sigma=0) and
conditions audio via cross-modal attention. Only audio is denoised.

```yaml
training_strategy:
  name: "flexible"
  video:
    is_generated: false
    latents_dir: "latents"
  audio:
    is_generated: true
    latents_dir: "audio_latents"
```

**Example config:** 📄 [v2a_lora.yaml](../configs/v2a_lora.yaml)

---

## 🎭 Video Inpainting

Fill in masked regions of a video. Per-sample masks loaded from disk define which tokens are conditioning and which
must be generated. Masks are thresholded at `0.5` to match validation/inference: tokens with `mask > 0.5` receive clean
latents and timestep=0 and are excluded from loss; tokens with `mask <= 0.5` are denoised normally and contribute to
loss.

```yaml
training_strategy:
  name: "flexible"
  video:
    is_generated: true
    latents_dir: "latents"
    conditions:
      - type: mask
        mask_dir: "video_masks"
        probability: 1.0
```

**Dataset structure:**

```
preprocessed_data_root/
├── latents/        # Video latents
├── conditions/     # Text embeddings
└── video_masks/    # Per-sample binary masks (1 → conditioning, 0 → generate)
```

In dataset metadata, provide mask media via the `video_mask` column; preprocessing converts it into `video_masks/`.

**Example config:** 📄 [video_inpainting_lora.yaml](../configs/video_inpainting_lora.yaml)

---

## 🌅 Video Outpainting

Extend a video spatially beyond its original boundaries. A rectangular pixel region is provided as clean conditioning
(no noise, timestep=0, excluded from loss) — the model learns to generate the surrounding content. The `spatial_region`
is specified in pixel coordinates `[y1, x1, y2, x2]` and automatically converted to latent space.

```yaml
training_strategy:
  name: "flexible"
  video:
    is_generated: true
    latents_dir: "latents"
    conditions:
      - type: spatial_crop
        spatial_region: [0, 0, 288, 576]  # y1, x1, y2, x2 in pixels
        probability: 1.0
```

> [!NOTE]
> `spatial_crop` is a video-only condition — it is not supported on the audio modality.

**Example config:** 📄 [video_outpainting_lora.yaml](../configs/video_outpainting_lora.yaml)

---

## 🔈 Text-to-Audio (T2A)

Generate audio from text prompts with no video modality. Only the audio branch of the transformer is denoised. Since
no video modality is configured, this mode uses **audio-only LoRA targets** — explicitly targeting `audio_attn1`,
`audio_attn2`, and `audio_ff` modules.

```yaml
training_strategy:
  name: "flexible"
  audio:
    is_generated: true
    latents_dir: "audio_latents"
```

> [!NOTE]
> With no `video` block in the strategy, the trainer only loads audio latents and text embeddings. LoRA adapters
> should explicitly target audio modules (e.g., `audio_attn1.to_k`) rather than short patterns like `to_k` which
> would also match video modules. See [LoRA Target Modules Guidance](#lora-target-modules-guidance) below.

**Example config:** 📄 [t2a_lora.yaml](../configs/t2a_lora.yaml)

---

## 🔊 Audio Extension

Extend audio forward (prefix) or backward (suffix) in time — the audio equivalent of Video Extension. A span of
existing audio latent frames is provided as clean conditioning, and the model generates the continuation. The
`temporal_boundary` sets the number of latent frames used as context. This mode uses **audio-only LoRA targets**.

```yaml
training_strategy:
  name: "flexible"
  audio:
    is_generated: true
    latents_dir: "audio_latents"
    conditions:
      - type: prefix            # or "suffix" for backward extension
        temporal_boundary: 8
        probability: 1.0
```

**Example configs:** 📄 [audio_extend_lora.yaml](../configs/audio_extend_lora.yaml), 📄 [audio_suffix_lora.yaml](../configs/audio_suffix_lora.yaml)

---

## 🎭 Audio Inpainting

Fill in masked regions of audio. Per-sample masks loaded from disk define which audio tokens are conditioning and
which must be generated — the audio equivalent of Video Inpainting. Masks are thresholded at `0.5` with the same
binary semantics as video inpainting. This mode uses **audio-only LoRA targets**.

```yaml
training_strategy:
  name: "flexible"
  audio:
    is_generated: true
    latents_dir: "audio_latents"
    conditions:
      - type: mask
        mask_dir: "audio_masks"
        probability: 1.0
```

**Dataset structure:**

```
preprocessed_data_root/
├── conditions/      # Text embeddings
├── audio_latents/   # Audio latents
└── audio_masks/     # Per-sample binary masks (1 → conditioning, 0 → generate)
```

In dataset metadata, provide mask media via the `audio_mask` column; preprocessing converts it into `audio_masks/`.

**Example config:** 📄 [audio_inpainting_lora.yaml](../configs/audio_inpainting_lora.yaml)

---

## 🔄 IC-LoRA / Audio-to-Audio (A2A)

In-Context LoRA for audio-to-audio transformations. Pre-encoded reference audio latents are concatenated to the target
sequence — reference tokens participate in bidirectional self-attention but receive no noise and are excluded from loss.
This enables audio style transfer, voice conversion, sound effect transformation, and more. This mode uses
**audio-only LoRA targets**.

```yaml
training_strategy:
  name: "flexible"
  audio:
    is_generated: true
    latents_dir: "audio_latents"
    conditions:
      - type: reference
        latents_dir: "reference_audio_latents"
        probability: 1.0
```

**Dataset structure:**

```
preprocessed_data_root/
├── conditions/              # Text embeddings
├── audio_latents/           # Target audio latents
└── reference_audio_latents/ # Reference audio latents (conditioning input)
```

**Example config:** 📄 [a2a_ic_lora.yaml](../configs/a2a_ic_lora.yaml)

---

## 🔄 AV2AV IC-LoRA

Joint audio-video In-Context LoRA — both modalities have reference conditioning. Pre-encoded reference latents are
concatenated to each modality's target sequence independently. This enables joint audiovisual transformations such as
synchronized style transfer across both video and audio.

```yaml
training_strategy:
  name: "flexible"
  video:
    is_generated: true
    latents_dir: "latents"
    conditions:
      - type: reference
        latents_dir: "reference_latents"
        probability: 1.0
  audio:
    is_generated: true
    latents_dir: "audio_latents"
    conditions:
      - type: reference
        latents_dir: "reference_audio_latents"
        probability: 1.0
```

> [!NOTE]
> Unlike audio-only IC-LoRA (A2A), AV2AV uses short LoRA target patterns like `"to_k"` to match all branches
> (video, audio, and cross-modal attention), since both modalities are trained.

**Dataset structure:**

```
preprocessed_data_root/
├── latents/                 # Target video latents
├── audio_latents/           # Target audio latents
├── conditions/              # Text embeddings
├── reference_latents/       # Reference video latents (conditioning input)
└── reference_audio_latents/ # Reference audio latents (conditioning input)
```

**Example config:** 📄 [av2av_ic_lora.yaml](../configs/av2av_ic_lora.yaml)

---

## 🔥 Full Model Fine-tuning

All modes above default to `training_mode: "lora"`. For full fine-tuning, set `training_mode: "full"` — this updates
all model parameters rather than adding LoRA adapters.

```yaml
model:
  training_mode: "full"

training_strategy:
  name: "flexible"
  video:
    is_generated: true
    latents_dir: "latents"
  audio:
    is_generated: true
    latents_dir: "audio_latents"
```

> [!IMPORTANT]
> Full fine-tuning requires multiple high-end GPUs (e.g., 4-8× H100 80GB) and distributed training with FSDP.
> See [Training Guide](training-guide.md) for multi-GPU setup instructions.

---

## 🎛️ LoRA Target Modules Guidance

The `target_modules` configuration determines which transformer modules receive LoRA adapters. The right choice depends
on whether your training involves cross-modal (audio ↔ video) interaction.

**For T2V, I2V, A2V, V2A, or any mode involving both modalities** — use short patterns to match all branches
(video, audio, and cross-modal attention):

```yaml
target_modules:
  - "to_k"
  - "to_q"
  - "to_v"
  - "to_out.0"
```

> [!IMPORTANT]
> Short patterns like `"to_k"` match video modules (`attn1.to_k`, `attn2.to_k`), audio modules
> (`audio_attn1.to_k`, `audio_attn2.to_k`), and cross-modal modules (`audio_to_video_attn.to_k`,
> `video_to_audio_attn.to_k`). The cross-modal attention modules enable bidirectional information flow between
> audio and video, which is critical for synchronized audiovisual generation.
> See [Understanding Target Modules](configuration-reference.md#understanding-target-modules) for detailed guidance.

**For video-only IC-LoRA** — explicitly target video modules (including FFN layers for better transformation quality):

```yaml
target_modules:
  - "attn1.to_k"
  - "attn1.to_q"
  - "attn1.to_v"
  - "attn1.to_out.0"
  - "attn2.to_k"
  - "attn2.to_q"
  - "attn2.to_v"
  - "attn2.to_out.0"
  - "ff.net.0.proj"
  - "ff.net.2"
```

**For audio-only modes (T2A, Audio Extension, Audio Inpainting, A2A IC-LoRA)** — explicitly target audio modules:

```yaml
target_modules:
  - "audio_attn1.to_k"
  - "audio_attn1.to_q"
  - "audio_attn1.to_v"
  - "audio_attn1.to_out.0"
  - "audio_attn2.to_k"
  - "audio_attn2.to_q"
  - "audio_attn2.to_v"
  - "audio_attn2.to_out.0"
  - "audio_ff.net.0.proj"
  - "audio_ff.net.2"
```

> [!NOTE]
> Audio-only modes have no `video` block in the strategy, so there is no need to train video or cross-modal
> attention modules. Targeting only `audio_*` modules keeps the LoRA small and focused.

---

## 🎬 Using Trained Models for Inference

After training, use the [`ltx-pipelines`](../../ltx-pipelines/) package for production inference with your trained
LoRAs:

| Training Mode           | Recommended Pipeline                                  |
|-------------------------|-------------------------------------------------------|
| T2V / I2V / A2V / Extension / Inpainting / Outpainting | `TI2VidOneStagePipeline` or `TI2VidTwoStagesPipeline` |
| IC-LoRA (V2V / A2A / AV2AV) | `ICLoraPipeline`                                 |
| V2A (Foley) / T2A / Audio Extension / Audio Inpainting | `TI2VidOneStagePipeline` or `TI2VidTwoStagesPipeline` |

All pipelines support loading custom LoRAs via the `loras` parameter. See the [`ltx-pipelines`](../../ltx-pipelines/)
package documentation for detailed usage instructions.

> [!NOTE]
> You can generate audio during validation even if you're not training the audio branch.
> Set `validation.generate_audio: true` independently of whether audio has `is_generated: true`.

---

## 🔄 Migration from Legacy Strategies

Legacy `text_to_video` and `video_to_video` strategy configs are forward-compatible and will continue to work (with a
deprecation warning). We recommend migrating to `flexible` for access to all conditioning modes.

---

## 🚀 Next Steps

Once you've chosen your training mode:

- Set up your dataset using [Dataset Preparation](dataset-preparation.md)
- Configure your training parameters in [Configuration Reference](configuration-reference.md)
- Start training with the [Training Guide](training-guide.md)

> [!TIP]
> Need a training mode that's not covered here?
> First check whether it can be expressed by composing existing `flexible` conditions. Use
> [Implementing Custom Training Strategies](custom-training-strategies.md) only for custom losses,
> noising rules, model outputs, or preprocessing that cannot be represented by configuration.
