# Configuration Reference

The trainer uses structured Pydantic models for configuration, making it easy to customize training parameters.
This guide covers all available configuration options and their usage.

## 📋 Overview

The main configuration class is [`LtxTrainerConfig`](../src/ltx_trainer/config.py), which includes the following
sub-configurations:

- **ModelConfig**: Base model and training mode settings
- **LoraConfig**: LoRA training parameters
- **TrainingStrategyConfig**: Training strategy settings (flexible conditioning framework)
- **OptimizationConfig**: Learning rate, batch sizes, and scheduler settings
- **AccelerationConfig**: Mixed precision and quantization settings
- **DataConfig**: Data loading parameters
- **ValidationConfig**: Validation and inference settings
- **CheckpointsConfig**: Checkpoint saving frequency and retention settings
- **HubConfig**: Hugging Face Hub integration settings
- **WandbConfig**: Weights & Biases logging settings
- **FlowMatchingConfig**: Timestep sampling parameters

## 📄 Example Configuration Files

Check out our example configurations in the `configs` directory:

- 📄 [Text-to-Video LoRA](../configs/t2v_lora.yaml) - Text-to-video LoRA training
- 📄 [Image-to-Video LoRA](../configs/i2v_lora.yaml) - Image-to-video LoRA training
- 📄 [IC-LoRA Video-to-Video](../configs/v2v_ic_lora.yaml) - IC-LoRA video-to-video training
- 📄 [Audio-to-Video LoRA](../configs/a2v_lora.yaml) - Audio-to-video LoRA training
- 📄 [Video-to-Audio LoRA](../configs/v2a_lora.yaml) - Video-to-audio (Foley) LoRA training
- 📄 [Video Extension LoRA](../configs/video_extend_lora.yaml) - Video extension (forward) LoRA training
- 📄 [Video Suffix LoRA](../configs/video_suffix_lora.yaml) - Video extension (backward) LoRA training
- 📄 [Video Inpainting LoRA](../configs/video_inpainting_lora.yaml) - Video inpainting LoRA training
- 📄 [Video Outpainting LoRA](../configs/video_outpainting_lora.yaml) - Video outpainting (spatial crop) LoRA training
- 📄 [Text-to-Audio LoRA](../configs/t2a_lora.yaml) - Text-to-audio LoRA training
- 📄 [Audio Extension LoRA](../configs/audio_extend_lora.yaml) - Audio extension (forward) LoRA training
- 📄 [Audio Suffix LoRA](../configs/audio_suffix_lora.yaml) - Audio extension (backward) LoRA training
- 📄 [Audio Inpainting LoRA](../configs/audio_inpainting_lora.yaml) - Audio inpainting LoRA training
- 📄 [Audio-to-Audio IC-LoRA](../configs/a2a_ic_lora.yaml) - Audio IC-LoRA transformation training
- 📄 [AV2AV IC-LoRA](../configs/av2av_ic_lora.yaml) - Audio+video IC-LoRA transformation training
- 📄 [T2V LoRA (Low VRAM)](../configs/t2v_lora_low_vram.yaml) - Memory-optimized config for 32GB GPUs

## ⚙️ Configuration Sections

> [!NOTE]
> The YAML snippets below show **recommended starting values**, not necessarily the code defaults.
> Fields you omit from your config file will use the code defaults from [`config.py`](../src/ltx_trainer/config.py).

## Supported Checkpoints and Text Encoders

The same configuration schema supports LTX-2, LTX-2.3, and **LTX 2.5**.
The trainer detects the architecture from checkpoint metadata; no model-version field is needed.

All model paths must be local. Always use the text encoder that the checkpoint's own metadata asks for: LTX 2.5
requires the LTX-specific fine-tuned Gemma 4 (for example `gemma4-12b-ltx-v1`), not Google's vanilla Gemma 4, and
older checkpoints require their matching Gemma 3.

### ModelConfig

Controls the base model and training mode settings.

#### Choosing your checkpoint layout

LTX models are published in two layouts, and which fields you set depends on which one you downloaded. Look at what
you have on disk:

**A single `.safetensors` file plus a Gemma folder** — this is the *unified* layout, used by LTX-2, LTX-2.3, and
LTX-2.5. One file holds the transformer, both VAEs, and the vocoder together.

```yaml
model:
  # The one big checkpoint file. Everything except the text encoder lives inside it.
  model_path: "/models/checkpoints/ltx-2.3-22b-dev.safetensors"
  # The folder holding the matching Gemma model (config.json, tokenizer, weights).
  text_encoder_path: "/models/text_encoders/gemma-3-12b-it/"
  training_mode: "lora"
  load_checkpoint: null
```

Leave `video_vae_path` and `audio_vae_path` unset — the trainer reads both VAEs out of `model_path`.

**A folder tree with one file per component** — this is the *split* layout, used by LTX 2.5. It matches the ComfyUI
directory structure (`diffusion_models/`, `text_encoders/`, `vae/`), and every component must be named
individually because no single file contains them all:

```yaml
model:
  # The transformer only. It holds no VAE and no text encoder.
  model_path: "/models/ltx-2.5/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
  # One file containing the Gemma weights, its tokenizer/config, and the text projections.
  text_encoder_path: "/models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
  # Required: the transformer above carries no VAE weights.
  video_vae_path: "/models/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors"
  audio_vae_path: "/models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors"
  training_mode: "lora"
  load_checkpoint: null
```

If you point `model_path` at a split transformer and forget a VAE path, training stops with an error naming the field
to set. Each component is only demanded when it is actually loaded, so a video-only run never asks for the audio VAE.

#### Key parameters

Path fields, which differ per layout:

| Parameter | Unified layout | Split layout |
|---|---|---|
| `model_path` | **Required.** The single checkpoint file. | **Required.** The transformer file. |
| `text_encoder_path` | **Required.** The Gemma model directory. | **Required.** The packed text-encoder `.safetensors`. |
| `video_vae_path` | Leave unset — read from `model_path`. | **Required.** The standalone video VAE. |
| `audio_vae_path` | Leave unset — read from `model_path`. | **Required when the run touches audio** (audio latents, or validation that generates audio). The standalone audio VAE, which also holds the vocoder. A video-only run may omit it. |

URLs are not supported for any of them.

The remaining fields are the same in both layouts:

| Parameter | Description |
|---|---|
| `training_mode` | `"lora"` for LoRA training, `"full"` for full-rank fine-tuning. |
| `load_checkpoint` | Optional path to resume training from a checkpoint file or directory. |

> [!NOTE]
> The same distinction applies to the preprocessing scripts, which accept `--video-vae-path` and `--audio-vae-path`
> alongside `--model-path` and `--text-encoder-path`.

### LoraConfig

LoRA-specific fine-tuning parameters (only used when `training_mode: "lora"`).

```yaml
lora:
  rank: 32         # LoRA rank (higher = more parameters)
  alpha: 32        # LoRA alpha scaling factor
  dropout: 0.0     # Dropout probability (0.0-1.0)
  target_modules: # Modules to apply LoRA to
    - "to_k"
    - "to_q"
    - "to_v"
    - "to_out.0"
```

**Key parameters:**

| Parameter        | Description                                                                     |
|------------------|---------------------------------------------------------------------------------|
| `rank`           | LoRA rank - higher values mean more trainable parameters (typical range: 8-128) |
| `alpha`          | Alpha scaling factor - typically set equal to rank                              |
| `dropout`        | Dropout probability for regularization                                          |
| `target_modules` | List of transformer modules to apply LoRA adapters to (see below)               |

#### Understanding Target Modules

The LTX-2 transformer has separate attention and feed-forward blocks for video and audio, as well as cross-attention
modules that enable the two modalities to exchange information. Choosing the right `target_modules` is critical for
achieving good results, especially when training with audio.

**Video-only modules:**

| Module Pattern                                             | Description                     |
|------------------------------------------------------------|---------------------------------|
| `attn1.to_k`, `attn1.to_q`, `attn1.to_v`, `attn1.to_out.0` | Video self-attention            |
| `attn2.to_k`, `attn2.to_q`, `attn2.to_v`, `attn2.to_out.0` | Video cross-attention (to text) |
| `ff.net.0.proj`, `ff.net.2`                                | Video feed-forward network      |

**Audio-only modules:**

| Module Pattern                                                                     | Description                     |
|------------------------------------------------------------------------------------|---------------------------------|
| `audio_attn1.to_k`, `audio_attn1.to_q`, `audio_attn1.to_v`, `audio_attn1.to_out.0` | Audio self-attention            |
| `audio_attn2.to_k`, `audio_attn2.to_q`, `audio_attn2.to_v`, `audio_attn2.to_out.0` | Audio cross-attention (to text) |
| `audio_ff.net.0.proj`, `audio_ff.net.2`                                            | Audio feed-forward network      |

**Audio-video cross-attention modules:**

These modules enable bidirectional information flow between the audio and video modalities:

| Module Pattern                                                                                                     | Description                                           |
|--------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| `audio_to_video_attn.to_k`, `audio_to_video_attn.to_q`, `audio_to_video_attn.to_v`, `audio_to_video_attn.to_out.0` | Video attends to audio (Q from video, K/V from audio) |
| `video_to_audio_attn.to_k`, `video_to_audio_attn.to_q`, `video_to_audio_attn.to_v`, `video_to_audio_attn.to_out.0` | Audio attends to video (Q from audio, K/V from video) |

**Recommended configurations:**

For **video-only training**, target the video attention layers:

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
```

For **audio-video training**, use patterns that match both branches:

```yaml
target_modules:
  - "to_k"
  - "to_q"
  - "to_v"
  - "to_out.0"
```

> [!NOTE]
> Using shorter patterns like `"to_k"` will match all attention modules including `attn1.to_k`, `audio_attn1.to_k`,
> `audio_to_video_attn.to_k`, and `video_to_audio_attn.to_k`, effectively training video, audio, and cross-modal
> attention branches together.

> [!TIP]
> You can also target the feed-forward (FFN) modules (`ff.net.0.proj`, `ff.net.2` for video,
> `audio_ff.net.0.proj`, `audio_ff.net.2` for audio) to increase the LoRA's capacity and potentially
> help it capture the target distribution better.

### TrainingStrategyConfig

Configures the training strategy. The recommended strategy is `"flexible"`, which supports all conditioning scenarios through configuration.

#### Flexible Strategy

The flexible strategy provides a unified conditioning framework. Each modality (video, audio) is configured
independently with its own latents directory, generation flag, and list of conditions.

```yaml
training_strategy:
  name: "flexible"
  video:
    is_generated: true                 # Video is denoised during training
    latents_dir: "latents"             # Directory containing precomputed video latents
    conditions:
      - type: first_frame              # Use first frame as conditioning
        probability: 0.5               # Apply this condition 50% of the time
  audio:
    is_generated: true                 # Audio is denoised during training
    latents_dir: "audio_latents"       # Directory containing precomputed audio latents
    conditions: []                     # No additional audio conditions (text-only)
```

**ModalityConfig parameters:**

| Parameter      | Description                                                                                                      |
|----------------|------------------------------------------------------------------------------------------------------------------|
| `is_generated` | `true` = modality is denoised (contributes to loss). `false` = frozen conditioning (sigma=0, no loss).           |
| `latents_dir`  | Directory name within `preprocessed_data_root` containing precomputed latents for this modality.                 |
| `conditions`   | List of conditioning configs applied during training (see condition types below). Text conditioning is implicit. |

**Condition types:**

| Type           | Parameters                                          | Description                                                                           |
|----------------|-----------------------------------------------------|---------------------------------------------------------------------------------------|
| `first_frame`  | `probability`                                       | First latent frame is clean, excluded from loss. **Video only.**                      |
| `prefix`       | `temporal_boundary`, `probability`                  | First N latent temporal units are clean. For extension forward.                       |
| `suffix`       | `temporal_boundary`, `probability`                  | Last N latent temporal units are clean. For extension backward.                       |
| `spatial_crop` | `spatial_region` (y1, x1, y2, x2 in px), `probability` | Rectangular region is clean, excluded from loss. For outpainting. **Video only.** |
| `mask`         | `mask_dir`, `probability`                           | Per-sample mask directory. Masks are thresholded at `0.5`; `1` means conditioning, `0` means generate. |
| `reference`    | `latents_dir`, `probability`                        | IC-LoRA style concatenation. Reference tokens are prepended, clean (timestep=0), no loss. |

> [!NOTE]
> The `prefix`, `suffix`, `mask`, and `reference` condition types work on both video and audio modalities —
> place them in the `video.conditions` or `audio.conditions` list as appropriate.
> `first_frame` and `spatial_crop` are video-only conditions.

> [!NOTE]
> Training conditions reference **directories** of precomputed data (within `preprocessed_data_root`),
> while validation conditions reference **individual files** (images, videos, masks) that are encoded
> on-the-fly during validation. The condition `type` names are the same, but the fields differ.

> [!NOTE]
> The legacy `text_to_video` and `video_to_video` strategies are deprecated but remain forward-compatible.
> New configs should use `name: "flexible"`.

### OptimizationConfig

Training optimization parameters including learning rates, batch sizes, and schedulers.

```yaml
optimization:
  learning_rate: 1e-4                  # Learning rate
  steps: 2000                          # Total training steps
  batch_size: 1                        # Batch size per GPU
  gradient_accumulation_steps: 1       # Steps to accumulate gradients
  max_grad_norm: 1.0                   # Gradient clipping threshold
  optimizer_type: "adamw"              # "adamw", "adamw8bit" or "prodigy"
  # prodigy_d_coef: 1.0                # prodigy only: step-size growth rate
  # prodigy_weight_decay: 0.0          # prodigy only: decoupled weight decay
  scheduler_type: "linear"             # Scheduler type
  scheduler_params: { }                # Additional scheduler parameters
  enable_gradient_checkpointing: true  # Memory optimization
```

**Key parameters:**

| Parameter                       | Description                                                                                  |
|---------------------------------|----------------------------------------------------------------------------------------------|
| `learning_rate`                 | Learning rate for optimization (typical range: 1e-5 to 1e-3)                                 |
| `steps`                         | Total number of training steps                                                               |
| `batch_size`                    | Batch size per GPU (reduce if running out of memory)                                         |
| `gradient_accumulation_steps`   | Accumulate gradients over multiple steps                                                     |
| `scheduler_type`                | LR scheduler: `"constant"`, `"linear"`, `"cosine"`, `"cosine_with_restarts"`, `"polynomial"`, `"step"` |
| `enable_gradient_checkpointing` | Trade training speed for GPU memory savings (recommended for large models)                   |

### AccelerationConfig

Hardware acceleration and compute optimization settings.

```yaml
acceleration:
  mixed_precision_mode: "bf16"                  # "no", "fp16", or "bf16"
  quantization: null                            # Quantization options
  load_text_encoder_in_8bit: false              # Load text encoder in 8-bit
  offload_optimizer_during_validation: false    # Offload optimizer state to CPU during validation
```

**Key parameters:**

| Parameter                             | Description                                                                                                                                                                              |
|---------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mixed_precision_mode`                | Precision mode - `"bf16"` recommended for modern GPUs                                                                                                                                    |
| `quantization`                        | Model quantization: `null`, `"int8-quanto"`, `"int4-quanto"`, `"int2-quanto"`, `"fp8-quanto"`, or `"fp8uz-quanto"`                                                                       |
| `load_text_encoder_in_8bit`           | Quantize the Gemma text encoder to 8-bit, saving ~8 GiB of VRAM while caption embeddings are computed. Works with both checkpoint layouts. Requires `bitsandbytes` and a CUDA device, and materializes the full-precision weights in host RAM first (~26 GB for LTX 2.5). **This can potentially degrade the cached caption features.** |
| `offload_optimizer_during_validation` | Move optimizer state to CPU before validation video sampling and back afterwards. Useful when validation OOMs because VAE decoder + transformer + optimizer state can't coexist on the GPU (full fine-tune, high-rank LoRA). No effect for FSDP. |

### DataConfig

Data loading and processing configuration.

```yaml
data:
  preprocessed_data_root: "/path/to/preprocessed/data"  # Path to precomputed dataset
  num_dataloader_workers: 2                             # Background data loading workers
```

**Key parameters:**

| Parameter                | Description                                                                                |
|--------------------------|--------------------------------------------------------------------------------------------|
| `preprocessed_data_root` | Path to your preprocessed dataset directory produced by `process_dataset.py` (contains `latents/`, `conditions/`, etc.) |
| `num_dataloader_workers` | Number of parallel data loading processes (0 = synchronous loading, useful when debugging) |

### ValidationConfig

Validation and inference settings for monitoring training progress. Validation samples use a self-describing
format where each sample specifies its own prompt and conditions.

```yaml
validation:
  samples:
    - prompt: "A cat playing with a ball"
      conditions:
        - type: first_frame
          image_or_video: "/path/to/image.png"
    - prompt: "A dog running in a field"
  video_dims: [960, 544, 89]                # Output dimensions: [width, height, frames]
  negative_prompt: "..."                    # Negative prompt for all samples (defaults to ltx-pipelines DEFAULT_NEGATIVE_PROMPT)
  frame_rate: 24.0                          # Output video frame rate (fps)
  seed: 42                                  # Random seed for reproducibility
  inference_steps: 30                       # Number of denoising steps
  interval: 100                             # Run validation every N steps (null to disable)
  video_cfg_scale: 3.0                      # Video CFG scale
  audio_cfg_scale: 7.0                      # Audio CFG scale
  video_stg_scale: 1.0                      # Video STG scale (0.0 to disable)
  audio_stg_scale: 1.0                      # Audio STG scale (0.0 to disable)
  stg_blocks: [28]                          # Transformer blocks to apply STG perturbation
  guidance_rescale: 0.7                     # Guidance variance rescale factor (0.0 disables)
  video_modality_guidance_scale: 3.0        # Video modality guidance scale
  audio_modality_guidance_scale: 3.0        # Audio modality guidance scale
  generate_audio: true                      # Whether to generate audio during validation
  generate_video: true                      # Whether to generate video during validation
  skip_initial_validation: false            # Skip validation at step 0
```

**Key parameters:**

| Parameter                | Description                                                                                                              |
|--------------------------|--------------------------------------------------------------------------------------------------------------------------|
| `samples`                | List of `ValidationSample` objects (see below). Replaces the legacy `prompts`/`images`/`reference_videos` fields.        |
| `video_dims`             | Output dimensions `[width, height, frames]`. Width/height must be divisible by the VAE spatial factor (32 by default), frames must satisfy `frames % T == 1` where `T` is the VAE temporal factor (8 by default) |
| `interval`               | Steps between validation runs (set to `null` to disable)                                                                 |
| `video_cfg_scale`        | Video CFG (Classifier-Free Guidance) scale. LTX 2.5 default: `3.0` |
| `audio_cfg_scale`        | Audio CFG (Classifier-Free Guidance) scale. LTX 2.5 default: `7.0` |
| `video_stg_scale`        | Video STG (Spatio-Temporal Guidance) scale. LTX 2.5 default: `1.0`; `0.0` disables STG |
| `audio_stg_scale`        | Audio STG (Spatio-Temporal Guidance) scale. LTX 2.5 default: `1.0`; `0.0` disables STG |
| `stg_blocks`             | Transformer blocks to perturb for STG. LTX 2.5 default: `[28]`; `None` perturbs all blocks |
| `guidance_rescale`       | Guidance variance rescale factor. LTX 2.5 default: `0.7`; `0.0` disables rescaling |
| `video_modality_guidance_scale` | Video modality guidance scale for audio-to-video isolation guidance. LTX 2.5 default: `3.0` |
| `audio_modality_guidance_scale` | Audio modality guidance scale for video-to-audio isolation guidance. LTX 2.5 default: `3.0` |
| `generate_audio`         | Whether to generate audio in validation samples                                                                          |
| `generate_video`         | Whether to generate video in validation samples. Set to `false` for V2A (video-to-audio) validation. Default: `true`     |
| `skip_initial_validation`| Skip validation video sampling at step 0 (beginning of training)                                                         |

#### ValidationSample

Each sample in the `samples` list has:

| Field        | Description                                                                                     |
|--------------|-------------------------------------------------------------------------------------------------|
| `prompt`     | Text prompt for this validation sample.                                                         |
| `conditions` | List of validation conditions (see types below). Empty list = text-only generation.             |
| `video_dims` | Optional per-sample override for `(width, height, frames)`. Inherits from `ValidationConfig` if not set. |
| `seed`       | Optional per-sample override for random seed. Inherits from `ValidationConfig` if not set.      |

#### Validation Condition Types

| Type             | Parameters                                                 | Description                                                             |
|------------------|------------------------------------------------------------|-------------------------------------------------------------------------|
| `first_frame`    | `image_or_video` (path)                                    | Use the first frame of the image/video as conditioning.                 |
| `prefix`         | `video` or `audio` (path), optional `num_frames`/`duration`| Use a video/audio clip as temporal prefix (for extension forward).      |
| `suffix`         | `video` or `audio` (path), optional `num_frames`/`duration`| Use a video/audio clip as temporal suffix (for extension backward).     |
| `spatial_crop`   | `video` (path), `spatial_region` (y1, x1, y2, x2)         | Provide spatial context for outpainting. Video only.                    |
| `mask`           | `video` or `audio` (path), `mask` (path)                   | Mask-based inpainting with a binary mask file.                          |
| `reference`      | `video` or `audio` (path), optional video-reference `downscale_factor`, `temporal_scale_factor`, `include_in_output` | IC-LoRA style reference conditioning. |
| `video_to_audio` | `video` (path)                                             | Freeze video, generate audio. For Foley/V2A tasks.                      |
| `audio_to_video` | `audio` (path)                                             | Freeze audio, generate video. For audio-driven generation.              |

For video `reference` validation conditions, `downscale_factor` is the spatial reference scale and
`temporal_scale_factor` is the temporal reference scale. Set both to match the factors used when
preprocessing video reference latents for training; validation media is encoded on the fly and cannot infer
those factors from the training dataset.

> [!NOTE]
> The legacy fields `prompts`, `images`, and `reference_videos` are deprecated but auto-converted to `samples`
> internally. New configs should use the `samples` format.

### CheckpointsConfig

Model checkpointing configuration.

```yaml
checkpoints:
  interval: 250       # Steps between checkpoint saves (null = disabled)
  keep_last_n: 3      # Number of recent checkpoints to retain
  precision: bfloat16 # Precision for saved weights (bfloat16 or float32)
  no_resume: false            # Ignore saved state, start from step 0
  save_training_state: "minimal"  # "full", "minimal", or "off"
```

**Key parameters:**

| Parameter     | Description                                                                   |
|---------------|-------------------------------------------------------------------------------|
| `interval`    | Steps between intermediate checkpoint saves (set to `null` to disable)        |
| `keep_last_n` | Number of most recent checkpoints to keep (-1 = keep all)                     |
| `precision`   | Precision for saved checkpoint weights: `"bfloat16"` (default) or `"float32"` |
| `no_resume` | When `true`, ignore saved training state and start from step 0. Model weights from `load_checkpoint` are still loaded. |
| `save_training_state` | Save training state for resume: `"full"` (optimizer + scheduler + RNG), `"minimal"` (scheduler + RNG only, sufficient for LoRA), `"off"` (no resume). |

### HubConfig

Hugging Face Hub integration for automatic model uploads.

```yaml
hub:
  push_to_hub: false                   # Enable Hub uploading
  hub_model_id: "username/model-name"  # Hub repository ID
```

**Key parameters:**

| Parameter      | Description                                                      |
|----------------|------------------------------------------------------------------|
| `push_to_hub`  | Whether to automatically push trained models to Hugging Face Hub |
| `hub_model_id` | Repository ID in format `"username/repository-name"`             |

### WandbConfig

Weights & Biases logging configuration.

```yaml
wandb:
  enabled: false               # Enable W&B logging
  project: "ltx-2-trainer"     # W&B project name
  entity: null                 # W&B username or team
  tags: [ ]                    # Tags for the run
  log_validation_videos: true  # Log validation videos to W&B
```

**Key parameters:**

| Parameter               | Description                                      |
|-------------------------|--------------------------------------------------|
| `enabled`               | Whether to enable W&B logging                    |
| `project`               | W&B project name                                 |
| `entity`                | W&B username or team (null uses default account) |
| `log_validation_videos` | Whether to log validation videos to W&B          |

### FlowMatchingConfig

Flow matching training configuration for timestep sampling.

```yaml
flow_matching:
  timestep_sampling_mode: "shifted_logit_normal"  # Timestep sampling strategy
  timestep_sampling_params: { }                   # Additional sampling parameters
```

**Key parameters:**

| Parameter                  | Description                                                |
|----------------------------|------------------------------------------------------------|
| `timestep_sampling_mode`   | Sampling strategy: `"uniform"` or `"shifted_logit_normal"` |
| `timestep_sampling_params` | Additional parameters for the sampling strategy            |

### General Configuration

Top-level settings for the training run.

```yaml
seed: 42                                  # Random seed for reproducibility
output_dir: "outputs/my_training_run"     # Directory for outputs (checkpoints, validation videos, logs)
```

| Parameter    | Description                                              |
|--------------|----------------------------------------------------------|
| `seed`       | Random seed for reproducibility (default: `42`)          |
| `output_dir` | Directory to save outputs (default: `"outputs"`)         |

## 🚀 Next Steps

Once you've configured your training parameters:

- Set up your dataset using [Dataset Preparation](dataset-preparation.md)
- Choose your training approach in [Training Modes](training-modes.md)
- Start training with the [Training Guide](training-guide.md)
