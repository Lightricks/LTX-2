# Configuration Reference

Start from a [complete example config](../configs/README.md) for your [training mode](training-modes.md), then use this
page to look up individual fields. The trainer rejects unknown fields, so spelling and nesting matter.

## 📄 Example Configuration Files

Choose the task closest to your run, then edit its model/data paths and validation inputs:

- **Text/image-to-video:** [text-to-video](../configs/t2v_lora.yaml), [low VRAM](../configs/t2v_lora_low_vram.yaml),
  [image-to-video](../configs/i2v_lora.yaml).
- **Audio/video conditioning:** [audio-to-video](../configs/a2v_lora.yaml), [video-to-audio / Foley](../configs/v2a_lora.yaml).
- **Text-to-audio:** [audio generation](../configs/t2a_lora.yaml).
- **Video extension:** [forward](../configs/video_extend_lora.yaml), [backward](../configs/video_suffix_lora.yaml).
- **Audio extension:** [forward](../configs/audio_extend_lora.yaml), [backward](../configs/audio_suffix_lora.yaml).
- **Inpainting:** [video](../configs/video_inpainting_lora.yaml), [audio](../configs/audio_inpainting_lora.yaml).
- **Outpainting:** [video](../configs/video_outpainting_lora.yaml).
- **IC-LoRA references:** [video](../configs/v2v_ic_lora.yaml), [audio](../configs/a2a_ic_lora.yaml),
  [audio + video](../configs/av2av_ic_lora.yaml).

All mode configs use LoRA. For full transformer training, use the same modality/condition setup and
[change `training_mode`](#training_mode-lora-quantization).

## 🔎 Find a Setting

- **Model files:** [unified and split model layouts](#model).
- **LoRA and memory:** [rank, targets, quantization](#training_mode-lora-quantization) and [target-module examples](#understanding-target-modules).
- **Dataset setup:** [paths, image/video buckets, audio, and reference scaling](#data).
- **What to generate:** [video/audio and training conditions](#video--audio-modalities).
- **Learning rate and batch size:** [optimizer/scheduler](#optimizer) and [runtime optimization](#optimizationconfig-optimization).
- **Validation:** [prompts, media inputs, dimensions, and guidance](#validation-content) versus [when to sample](#validation-cadence-validation).
- **Run management:** [checkpoints/resume](#checkpointsconfig-checkpoints), [W&B](#wandbconfig-wandb), and [output/seed](#general).

> [!NOTE]
> **Relative paths have different bases:**
>
> - Paths **in the YAML** resolve from the command's working directory, not the YAML's directory.
> - Media paths **in dataset metadata** resolve from the metadata file's directory.

## 📋 Overview

Most settings go in one of two places:

- **`training_strategy`** describes what to train and how: models, LoRA, quantization, data/preprocessing, video/audio,
  optimizer/scheduler, noise sampling, and what to generate during validation.
- **Top-level keys** control the run: step budget, batch size, mixed precision, when to validate, preprocessing
  compatibility checks, checkpoints, W&B, seed, and output directory.

> [!NOTE]
> **Example values versus defaults:** This example shows starting choices; the tables give **code defaults when fields
> are omitted**, which may differ. Section examples are fragments to merge into a complete run config. Keep the shown
> parent keys: for example, `optimizer` belongs under `training_strategy`, while `optimization` is top-level.

```yaml
training_strategy:
  name: "flexible"
  model:
    model_path: "/path/to/model.safetensors"
    text_encoder_path: "/path/to/matching-text-encoder"
  training_mode: "lora"
  lora:
    rank: 32
    alpha: 32
  data:
    dataset_file: "/path/to/dataset.json"
    preprocessed_data_root: "/path/to/.precomputed"
    resolution_buckets: [[960, 544, 49]]
  video:
    is_generated: true
    latents_dir: "latents"
  audio:
    is_generated: true
    latents_dir: "audio_latents"
  optimizer:
    learning_rate: 1e-4
    gradient_checkpointing: true
  validation:  # What to generate for comparison
    samples:
      - prompt: "A cat playing with a ball"

optimization:
  steps: 2000
  batch_size: 1
validation:  # How often to generate it
  interval: 100
checkpoints:
  interval: 250
  keep_last_n: 3
output_dir: "outputs/my_run"
```

The schema is defined by strict Pydantic models (`extra="forbid"`): [`LtxTrainerConfig`](../src/ltx_trainer/config.py),
[`FlexibleStrategyConfig`](../src/ltx_trainer/training_strategies/flexible/config.py),
[shared config blocks](../src/ltx_trainer/training_strategies/shared_config.py), and
[validation content](../src/ltx_trainer/validation/config.py).
Legacy flat-layout configs (top-level `model`, `optimization.learning_rate`, `acceleration`, `flow_matching`, `hub`)
are migrated at load time with a deprecation warning; use the current layout for new runs.

## `training_strategy` (strategy-owned)

`name: "flexible"` selects the built-in strategy. Each custom strategy owns its own schema; the fields below describe
`flexible`. See [Custom Training Strategies](custom-training-strategies.md) for the extension contract.

### `model`

Unified checkpoints bundle the transformer and media encoders/decoders (VAEs). Split packs keep these in separate
files, so you must supply the component paths for the video/audio your run uses.

**Unified LTX-2 / LTX-2.3 checkpoint:**

```yaml
training_strategy:
  model:
    model_path: "/path/to/ltx-2-model.safetensors"
    text_encoder_path: "/path/to/gemma-3-directory"
    load_text_encoder_in_8bit: false
```

| Field | Code default | Meaning |
|-------|--------------|---------|
| `model_path` | Required | Existing local unified LTX-2/LTX-2.3 checkpoint or split-pack transformer `.safetensors`. URLs are unsupported. |
| `text_encoder_path` | `null` | Matching text encoder: Gemma 3 directory for unified LTX-2/LTX-2.3, or packed LTX-specific Gemma 4 safetensors for LTX-2.5. Needed for caption encoding and validation prompts. Strategies with embeddings connectors may also require it when training on precomputed text features. |
| `video_vae_path` | `null` | Video VAE safetensors. Required for video processing with a split pack; unified checkpoints supply it through `model_path`. |
| `audio_vae_path` | `null` | Audio VAE/vocoder safetensors. Required for audio processing with a split pack; unified checkpoints supply it through `model_path`. |
| `load_text_encoder_in_8bit` | `false` | Use bitsandbytes for text-encoder memory savings during preprocessing and validation prompt caching; no per-step cost. |

**Split LTX-2.5 pack with video and audio:**

```yaml
training_strategy:
  model:
    model_path: "/path/to/ltx-2.5-transformer.safetensors"
    text_encoder_path: "/path/to/ltx-2.5-text-encoder.safetensors"
    video_vae_path: "/path/to/video-vae.safetensors"
    audio_vae_path: "/path/to/audio-vae.safetensors"
```

> [!IMPORTANT]
> **LTX-2.5 needs the LTX-specific Gemma 4 file**, rather than Google's vanilla Gemma 4 weights. The packed file also
> supplies the text-projection weights used with the split transformer. Configure VAE paths for the modalities your run uses.

### `training_mode`, `lora`, `quantization`

**LoRA starting point for audio-video training** ([complete T2V config](../configs/t2v_lora.yaml)):

```yaml
training_strategy:
  training_mode: "lora"
  lora:
    rank: 32
    alpha: 32
    dropout: 0.0
    target_modules: ["to_k", "to_q", "to_v", "to_out.0"]
  quantization: null  # For example, "int8-quanto" for LoRA memory savings
```

Choose branch-specific targets for [video-only](../configs/v2v_ic_lora.yaml) or [audio-only](../configs/t2a_lora.yaml)
training; see [target-module examples](#understanding-target-modules). For a combined memory-saving setup, use
the [low VRAM config](../configs/t2v_lora_low_vram.yaml).

| Field | Code default | Meaning |
|-------|--------------|---------|
| `training_mode` | `"lora"` | `"lora"` trains adapters; `"full"` trains all transformer parameters. Text encoder and VAEs remain frozen. |
| `lora` | `null` | Required in LoRA mode. Defaults below apply to fields inside a supplied `lora` block. |
| `lora.rank` | `64` | Integer ≥ 2; higher rank adds parameters (typical range 8–128). Examples may start at 32. |
| `lora.alpha` | `64` | LoRA scaling factor, integer ≥ 1; commonly set equal to rank. |
| `lora.dropout` | `0.0` | Dropout probability in `[0, 1]`. |
| `lora.target_modules` | `["to_k", "to_q", "to_v", "to_out.0"]` | Adapter targets; see [module selection](#understanding-target-modules). |
| `quantization` | `null` | Transformer quantization for LoRA memory savings: `"int8-quanto"`, `"int4-quanto"`, `"int2-quanto"`, `"fp8-quanto"`, or `"fp8uz-quanto"`. |

For **full fine-tuning**, remove the `lora` block, leave `quantization: null`, and change the mode:

```yaml
training_strategy:
  training_mode: "full"
  quantization: null
```

Retain the chosen recipe's video/audio and condition settings. See [full-training hardware requirements](training-modes.md#-full-transformer-fine-tuning).

### `data`

The same block specifies where training reads cached artifacts and how preprocessing encodes raw media.

**Video preprocessing and training** ([complete T2V config](../configs/t2v_lora.yaml)):

```yaml
training_strategy:
  data:
    dataset_file: "/path/to/dataset.json"
    preprocessed_data_root: "/path/to/.precomputed"
    resolution_buckets: [[960, 544, 49]]  # [width, height, frames]
    num_dataloader_workers: 2
    lora_trigger: null
    decode: false  # Set true to inspect reconstructed media after encoding
```

> [!TIP]
> **Images use one frame:** Set `resolution_buckets: [[960, 544, 1]]`. For mixed images/videos, include both one-frame
> and video buckets and set top-level `optimization.batch_size: 1`. See [image/mixed dataset guidance](dataset-preparation.md#-dataset-format).

**Data locations and loading**

| Field | Code default | Meaning |
|-------|--------------|---------|
| `preprocessed_data_root` | `null` | Training artifact root containing latents, conditions, etc. Required for training; preprocessing defaults to `<dataset_file parent>/.precomputed`. |
| `dataset_file` | `null` | Raw CSV/JSON/JSONL metadata; required for preprocessing. Training consumes the cached artifacts. |
| `num_dataloader_workers` | `2` | Training workers; `0` gives synchronous loading for debugging. |

**Encoding settings**

| Field | Code default | Meaning |
|-------|--------------|---------|
| `resolution_buckets` | `null` | Video buckets as `[width, height, frames]`, matching `validation.video_dims` order. Required for video preprocessing. **Multiple buckets require `optimization.batch_size: 1`.** |
| `lora_trigger` | `null` | Token automatically prepended to captions during preprocessing; include it in inference prompts. |
| `audio_durations` | `null` | Positive audio-only duration buckets in seconds. Without buckets, variable-length audio requires batch size 1. |
| `skip_audio` | `false` | Skip automatic audio extraction from video. |
| `decode` | `false` | Decode encoded latents for verification. |
| `vae_tiling` | `false` | Enable VAE tiling during video preprocessing. |
| `encode_batch_size` | `1` | Offline encoder batch size. |
| `encode_num_workers` | `4` | Video preprocessing workers. |
| `reshape_mode` | `"center"` | Video crop mode: `"center"` or `"random"`. |

**Reference scaling**

| Field | Code default | Meaning |
|-------|--------------|---------|
| `reference_spatial_scale_factor` | `1` | Positive integer target/reference spatial ratio. Validation inherits it. Legacy alias: `reference_downscale_factor`. |
| `reference_temporal_scale_factor` | `1` | Positive integer target/reference frame-rate ratio. For factor `K`, bucket frames satisfy `(frames - 1) % (K * T) == 0`, where `T` is the VAE temporal factor. |

Reference factors must match the cached data, including when reusing an existing cache. **If either exceeds 1,
preprocessing requires one resolution bucket.** Validation inherits both factors; setting them under validation is
rejected. See [reference scaling](dataset-preparation.md#scaled-reference-conditioning) for alignment and source-length
constraints, or [Dataset Preparation](dataset-preparation.md) for metadata columns, output layout, and reruns.

For example, to encode references at half the target width/height while keeping the same frame count:

```yaml
training_strategy:
  data:
    resolution_buckets: [[960, 576, 49]]
    reference_spatial_scale_factor: 2
    reference_temporal_scale_factor: 1
  validation:
    video_dims: [960, 576, 49]  # Reference dimensions stay VAE-aligned here too
```

Also declare a `reference` condition as shown [below](#video--audio-modalities). Start from the
[video IC-LoRA config](../configs/v2v_ic_lora.yaml), or the [audio-only config](../configs/t2a_lora.yaml) for an
example using `data.audio_durations` instead of video buckets.

### `video` / `audio` modalities

A *modality* is video or audio. Both blocks default to `null`; omit a block to exclude that modality.
At least one must have `is_generated: true`. During training, generated modalities receive noise and the model learns
to remove it. Frozen modalities stay unchanged and provide context for generation.

**Generate video and audio, optionally conditioned on the first frame** ([complete I2V config](../configs/i2v_lora.yaml)):

```yaml
training_strategy:
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

| Field | Code default | Meaning |
|-------|--------------|---------|
| `is_generated` | Required | `true`: denoised and contributes to loss. `false`: frozen conditioning, sigma zero, no loss. |
| `latents_dir` | Required | Modality directory under `preprocessed_data_root`. |
| `conditions` | `[]` | Per-modality conditioning; text conditioning is implicit. |

**Generate video from source audio** ([complete A2V config](../configs/a2v_lora.yaml)):

```yaml
training_strategy:
  video:
    is_generated: true
    latents_dir: "latents"
  audio:
    is_generated: false  # Input audio stays clean
    latents_dir: "audio_latents"
```

For [video-to-audio / Foley](../configs/v2a_lora.yaml), reverse the flags: video is `false`, audio is `true`.
For [text-to-audio](../configs/t2a_lora.yaml), omit the video block and disable video output during validation:

```yaml
training_strategy:
  audio:
    is_generated: true
    latents_dir: "audio_latents"
  validation:
    generate_video: false
```

Use the audio-only config's LoRA targets too; omitting video input does not change adapter placement.

*Sigma* is the noise level. `training_strategy.share_sigmas` defaults to `true`: when both video and audio are generated,
they share the noise level sampled from video. Set it to `false` to sample their noise levels independently.
Frozen modalities always use sigma zero; a sole generated modality samples its own noise level.
All example YAMLs explicitly enable shared sigmas.

**Training condition types:** `probability` defaults to `1.0`, in `[0, 1]`. Other listed parameters are required.
Clean conditioning tokens receive no noise, use timestep zero, and are excluded from the loss (the prediction error
the model learns to reduce).

| Type | Parameters | Effect |
|------|------------|--------|
| `first_frame` | `probability` | First latent frame is clean. **Video only.** |
| `prefix` | `temporal_boundary`, `probability` | First N latent frames/timesteps are clean; forward extension. |
| `suffix` | `temporal_boundary`, `probability` | Last N latent frames/timesteps are clean; backward extension. |
| `spatial_crop` | `spatial_region`, `probability` | Clean rectangle `[y1, x1, y2, x2]` in pixel coordinates. **Video only.** |
| `mask` | `mask_dir`, `probability` | Per-sample masks: values `> 0.5` condition, `<= 0.5` generate. |
| `reference` | `latents_dir`, `probability` | Append clean reference tokens for IC-LoRA; bidirectional self-attention includes them. |

Keep these rules in mind when composing conditions:

- `prefix`, `suffix`, `mask`, and `reference` work for both modalities. `temporal_boundary` counts **latent** temporal
  units and must be ≥ 1; see the [video extension example](training-modes.md#-video-extension) for conversion to pixel frames.
- `first_frame` is skipped for single-frame image samples so there are still video tokens to learn from.
- Video positions use each sample's saved FPS, including batches with different frame rates.
- Conditions on the target itself apply before reference tokens are appended, regardless of config order.

> [!NOTE]
> **Training conditions use precomputed directories; validation conditions use individual media files.**
> The type names overlap but their fields differ; see [Validation Condition Types](#validation-condition-types).

**Reference-guided video** ([complete IC-LoRA config](../configs/v2v_ic_lora.yaml)):

```yaml
training_strategy:
  video:
    is_generated: true
    latents_dir: "latents"
    conditions:
      - type: reference
        latents_dir: "reference_latents"
        probability: 1.0
```

For [inpainting](../configs/video_inpainting_lora.yaml), use `type: mask` with `mask_dir: "video_masks"` instead of
the reference condition. For [forward extension](../configs/video_extend_lora.yaml), use `type: prefix` with
`temporal_boundary: 8`; change `prefix` to `suffix` for [backward extension](../configs/video_suffix_lora.yaml).
See [Training Modes](training-modes.md) for copyable recipes for every condition.

### `optimizer`

These strategy-owned fields are separate from the global `optimization` block.

```yaml
training_strategy:
  optimizer:
    learning_rate: 1e-4
    optimizer_type: "adamw"  # Or "adamw8bit" for lower optimizer memory
    scheduler_type: "linear"
    scheduler_params: {}
    gradient_checkpointing: true
```

| Field | Code default | Meaning |
|-------|--------------|---------|
| `learning_rate` | `5e-4` | Learning rate; example LoRA starting choices commonly fall in `1e-5`–`1e-4`. |
| `optimizer_type` | `"adamw"` | `"adamw"` or memory-saving `"adamw8bit"`. |
| `scheduler_type` | `"linear"` | `"constant"`, `"linear"`, `"cosine"`, `"cosine_with_restarts"`, `"polynomial"`, or `"step"`. |
| `scheduler_params` | `{}` | Parameters for the selected scheduler. |
| `gradient_checkpointing` | `false` | Trade speed for GPU memory; recommended for large models. |

### `timestep_sampling`

This controls which noise levels the model sees during training. A starting setup is:

```yaml
training_strategy:
  timestep_sampling:
    timestep_sampling_mode: "shifted_logit_normal"
    timestep_sampling_params: {}
```

| Field | Code default | Meaning |
|-------|--------------|---------|
| `timestep_sampling_mode` | `"shifted_logit_normal"` | Flow-matching sampler: `"uniform"` or `"shifted_logit_normal"` (recommended for LTX-2). |
| `timestep_sampling_params` | `{}` | Sampler-specific parameters. |

### `validation` (content)

Validation generates samples so you can compare checkpoints during training. Configure their prompts, inputs, and
sampling settings here; top-level [`validation`](#validation-cadence-validation) controls when they are generated.

**Image-conditioned and text-only samples** ([complete I2V config](../configs/i2v_lora.yaml)):

```yaml
training_strategy:
  validation:
    samples:
      - prompt: "A cat playing with a ball"
        conditions:
          - type: first_frame
            image_or_video: "/path/to/image.png"
      - prompt: "A dog running in a field"
    video_dims: [576, 576, 89]
    generate_audio: true
```

**Samples and output**

| Field | Code default | Meaning |
|-------|--------------|---------|
| `samples` | `[]` | List of [ValidationSample](#validationsample) entries; empty means no validation content. |
| `negative_prompt` | `"worst quality, inconsistent motion, blurry, jittery, distorted"` | Negative prompt shared by all samples. |
| `video_dims` | `[960, 544, 97]` | Output `[width, height, frames]`; must follow the VAE alignment rules below. |
| `frame_rate` | `24.0` | Output frames per second (FPS), > 0. |
| `seed` | `42` | Validation sampling seed; independent of the top-level training seed. |
| `inference_steps` | `30` | Denoising steps, > 0. |
| `generate_audio` | `true` | Audio generation independent of which modalities are trained. |
| `generate_video` | `true` | Set `false` for audio-only/V2A output to skip the video VAE decoder. |

Width and height must be divisible by the VAE spatial factor (32 by default); frames must satisfy
`(frames - 1) % T == 0`, where `T` is the temporal factor (8 by default). See
[resolution buckets](dataset-preparation.md#-resolution-buckets) for examples. At least one output modality must be
enabled when samples are configured.

**Guidance**

Guidance changes how predictions are combined during sampling. CFG (classifier-free guidance) strengthens the effect
of the text prompt; STG (spatiotemporal guidance) compares against a pass with selected attention blocks perturbed.
Start with the defaults and compare the same prompts and seeds when changing them.

```yaml
training_strategy:
  validation:
    inference_steps: 30
    video_cfg_scale: 3.0
    audio_cfg_scale: 7.0
    video_stg_scale: 1.0
    audio_stg_scale: 1.0
    stg_blocks: [28]
    guidance_rescale: 0.7
    video_modality_guidance_scale: 3.0
    audio_modality_guidance_scale: 3.0
```

| Field | Code default | Meaning |
|-------|--------------|---------|
| `video_cfg_scale` / `audio_cfg_scale` | `3.0` / `7.0` | Per-modality CFG, each ≥ 1. |
| `video_stg_scale` / `audio_stg_scale` | `1.0` / `1.0` | Per-modality STG; `0` disables it. |
| `stg_blocks` | `[28]` | Blocks to perturb for STG; `null` perturbs all. Separate from LoRA adapter placement. |
| `guidance_rescale` | `0.7` | Variance rescale in `[0, 1]`; `0` disables it. |
| `video_modality_guidance_scale` / `audio_modality_guidance_scale` | `3.0` / `3.0` | Cross-modality guidance, each ≥ 1. |

Validation uses simplified sampling for quick comparisons. Use [ltx-pipelines](../../ltx-pipelines/) for final inference.

#### ValidationSample

| Field | Code default | Meaning |
|-------|--------------|---------|
| `prompt` | Required | Text prompt. |
| `conditions` | `[]` | Validation conditions below; empty means text-only generation. |
| `video_dims` | `null` | Per-sample `[width, height, frames]` override; otherwise inherits from validation content. |
| `seed` | `null` | Per-sample seed override; otherwise inherits from validation content. |

#### Validation Condition Types

| Type | Parameters | Effect |
|------|------------|--------|
| `first_frame` | `image_or_video` | First frame from an image/video. Video only. |
| `prefix` | `video` or `audio`; optional `num_frames` / `duration` | Temporal prefix for forward extension. |
| `suffix` | `video` or `audio`; optional `num_frames` / `duration` | Temporal suffix for backward extension. |
| `spatial_crop` | `video`, `spatial_region` | Clean pixel rectangle `[y1, x1, y2, x2]`. Video only. |
| `mask` | `video` or `audio`, `mask` | Binary mask conditioning from media files. |
| `reference` | `video` or `audio`; optional `include_in_output` | IC-LoRA reference; `include_in_output: true` displays it alongside the generated sample (default `false`). |
| `video_to_audio` | `video` | Frozen video conditioning for Foley/V2A. |
| `audio_to_video` | `audio` | Frozen audio conditioning for A2V. |

**Reference and inpainting samples use file paths**, unlike the training condition's directory name:

```yaml
training_strategy:
  validation:
    samples:
      - prompt: "A cat playing with yarn"
        conditions:
          - type: reference
            video: "/path/to/reference.mp4"
            include_in_output: true
      - prompt: "A cat playing under a blue sky"
        conditions:
          - type: mask
            video: "/path/to/source.mp4"
            mask: "/path/to/mask.mp4"
```

See the matching [IC-LoRA](../configs/v2v_ic_lora.yaml) and [video inpainting](../configs/video_inpainting_lora.yaml)
configs for complete runs. For extension samples using `prefix` or `suffix`, see the
[forward](../configs/video_extend_lora.yaml) and [backward](../configs/video_suffix_lora.yaml) configs.

All media values are paths. Conditions accepting `video` or `audio` require **exactly one**.

> [!IMPORTANT]
> **Extension lengths use different units:** Training's `temporal_boundary` counts latent temporal units.
> Validation uses **pixel frames** (`num_frames`) or **seconds** (`duration`).

Reference scale factors come from `training_strategy.data`; setting them here raises an error.
See [scaled reference validation](dataset-preparation.md#scaled-reference-conditioning).

## Global runtime keys

### OptimizationConfig (`optimization`)

These settings control the training loop. Learning rate and optimizer choice belong in
[`training_strategy.optimizer`](#optimizer).

```yaml
optimization:
  steps: 2000
  batch_size: 1
  gradient_accumulation_steps: 4
  max_grad_norm: 1.0
  mixed_precision_mode: "bf16"
  offload_optimizer_during_validation: false
```

On two GPUs, this example gives an effective batch of `1 × 4 × 2 = 8` samples per update.
See the [low VRAM config](../configs/t2v_lora_low_vram.yaml) for a complete memory-saving setup.

| Field | Code default | Meaning |
|-------|--------------|---------|
| `steps` | `3000` | Total training steps, including steps restored on resume. Normally one parameter update per step; custom strategies can define their own update cycle. |
| `batch_size` | `2` | Per-GPU batch; must be `1` for multiple resolution buckets. |
| `gradient_accumulation_steps` | `1` | Effective batch = batch size × accumulation × process count. |
| `max_grad_norm` | `1.0` | Gradient clipping threshold. |
| `mixed_precision_mode` | `"bf16"` | `"no"`, `"fp16"`, `"bf16"`, or `null`; BF16 recommended for modern GPUs. |
| `offload_optimizer_during_validation` | `false` | Move optimizer state to CPU for sampling, then reload; useful for full fine-tunes/high-rank LoRA validation OOMs. No effect for FSDP. |

### Validation cadence (`validation`)

These settings control **when** to generate samples, rather than what to generate.

```yaml
validation:
  interval: 100  # null disables validation during training
  skip_initial_validation: false
```

| Field | Code default | Meaning |
|-------|--------------|---------|
| `interval` | `100` | Validate every N steps; positive integer or `null` to disable. |
| `skip_initial_validation` | `false` | Skip validation at step 0. |

Validation [content](#validation-content) lives under `training_strategy.validation`.

### `allow_stale_preprocessed`

Default `false`. Set `true` to intentionally train despite a preprocessing fingerprint mismatch; it becomes a warning.
This does not change `preprocess.py --check` behavior. See
[manifest checks](dataset-preparation.md#preprocessing-manifest-and-staleness-guard) for legacy data and rebuild semantics.

```yaml
allow_stale_preprocessed: false
```

### CheckpointsConfig (`checkpoints`)

**Save every 250 steps and retain the latest three checkpoints:**

```yaml
checkpoints:
  load_checkpoint: null
  interval: 250
  keep_last_n: 3
  precision: "bfloat16"
  no_resume: false
  save_training_state: "minimal"
```

| Field | Code default | Meaning |
|-------|--------------|---------|
| `load_checkpoint` | `null` | Run/checkpoints directory (latest checkpoint), specific `step_NNNNN/`, or external `.safetensors` initial weights. |
| `interval` | `null` | Positive save interval; `null` disables intermediate saves. A final checkpoint is saved on normal completion. |
| `keep_last_n` | `1` | Recent checkpoints to retain; `-1` keeps all, `0` is invalid. |
| `precision` | `"bfloat16"` | Saved weight precision: `"bfloat16"` or `"float32"`. |
| `no_resume` | `false` | Load weights but ignore saved training state and restart at step 0. |
| `save_training_state` | `"minimal"` | `"full"`: optimizer/scheduler/RNG/step; `"minimal"`: scheduler/RNG/step with fresh optimizer; `"off"`: weights only. |

Each completed checkpoint directory contains `weights.safetensors`; `state.pt` and `metadata.json` are omitted in `"off"`
mode. **FSDP `"full"` saves omit optimizer state and warn; the optimizer starts fresh on resume.**
Random-number-generator (RNG) state is restored only in single-process runs.
See [resume behavior and output files](training-guide.md#resume-training).

**Continue from step 1,000 for another 500 steps:** Point to the checkpoint directory, and set the **total** step target:

```yaml
checkpoints:
  load_checkpoint: "outputs/my_run/checkpoints/step_01000"
optimization:
  steps: 1500
```

The saved state must support resume. Loading a `.safetensors` file directly loads only weights and starts at step 0;
check the log for **“Resuming from step …”**. See [resume limitations](training-guide.md#resume-training).

### WandbConfig (`wandb`)

```yaml
wandb:
  enabled: true
  project: "ltx-2-trainer"
  entity: null  # Your username/team, or null for the default account
  tags: ["ltx2", "lora"]
  log_validation_videos: true
```

| Field | Code default | Meaning |
|-------|--------------|---------|
| `enabled` | `false` | Enable W&B logging. |
| `project` | `"ltxv-trainer"` | Project name; examples may choose `"ltx-2-trainer"`. |
| `entity` | `null` | Username/team; `null` uses the default account. |
| `tags` | `[]` | Run tags. |
| `log_validation_videos` | `true` | Log validation media. |

### General

```yaml
seed: 42
output_dir: "outputs/my_run"
```

| Field | Code default | Meaning |
|-------|--------------|---------|
| `seed` | `42` | Training random seed. |
| `output_dir` | `"outputs"` | Checkpoints, validation media, and resolved run config. |

## Understanding Target Modules

`training_strategy.lora.target_modules` is a list of literal module names or dot-separated suffixes, combined as
alternatives. `"to_q"` matches every attention query projection; `"attn1.to_q"` selects video self-attention queries
across all blocks; `"transformer_blocks.0.attn1.to_q"` selects that projection in the first block only.
Wildcards and regular expressions are not expanded within the list.

For example, adapt only video self-attention queries and values in the first two blocks:

```yaml
training_strategy:
  lora:
    target_modules:
      - "transformer_blocks.0.attn1.to_q"
      - "transformer_blocks.0.attn1.to_v"
      - "transformer_blocks.1.attn1.to_q"
      - "transformer_blocks.1.attn1.to_v"
```

Replace the existing target list when restricting blocks; keeping a broad entry such as `"to_q"` would still match
every block. For starting target lists, open the [audio-video](../configs/t2v_lora.yaml),
[video-only](../configs/v2v_ic_lora.yaml), or [audio-only](../configs/t2a_lora.yaml) config.

Each attention module below has `to_k`, `to_q`, `to_v`, and `to_out.0` projections:

| Module | Role |
|--------|------|
| `attn1` / `attn2` | Video self-attention / attention to text. |
| `audio_attn1` / `audio_attn2` | Audio self-attention / attention to text. |
| `audio_to_video_attn` | Video attends to audio: queries from video, keys/values from audio. |
| `video_to_audio_attn` | Audio attends to video: queries from audio, keys/values from video. |
| `ff.net.0.proj`, `ff.net.2` | Video feed-forward layers; optional additional LoRA capacity. |
| `audio_ff.net.0.proj`, `audio_ff.net.2` | Audio feed-forward layers; optional additional LoRA capacity. |

Gated-attention checkpoints additionally expose `to_gate_logits` in their attention modules, for example
`"attn1.to_gate_logits"`. Standard `to_k` / `to_q` / `to_v` / `to_out.0` targets do not include these gates.

See [LoRA Target Modules Guidance](training-modes.md#-lora-target-modules-guidance) for copyable YAML covering all
attention, individual branches/projections, feed-forward layers, specific blocks, and adapter verification.
The [mode example configs](../configs/README.md) include starting target lists.

## 🚀 Next Steps

Once your config is ready:

1. Confirm your modality and conditioning choices in [Training Modes](training-modes.md).
2. Encode the dataset using [Dataset Preparation](dataset-preparation.md).
3. Launch your run with the [Training Guide](training-guide.md).
