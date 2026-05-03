# Configuration Reference

The trainer uses structured Pydantic models for configuration, making it easy to customize training parameters.
This guide covers all available configuration options and their usage.

## 📋 Overview

The main configuration class is [`LtxTrainerConfig`](../src/ltx_trainer/config.py), which includes the following
sub-configurations:

- **ModelConfig**: Base model and training mode settings
- **LoraConfig**: LoRA training parameters
- **TrainingStrategyConfig**: Training strategy settings (text-to-video or video-to-video)
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

- 📄 [Audio-Video LoRA Training](../configs/ltx2_av_lora.yaml) - Joint audio-video generation training
- 📄 [Audio-Video LoRA Training (Low VRAM)](../configs/ltx2_av_lora_low_vram.yaml) - Memory-optimized config for 32GB
  GPUs (uses 8-bit optimizer, INT8 quantization, and reduced LoRA rank)
- 📄 [IC-LoRA Training](../configs/ltx2_v2v_ic_lora.yaml) - Video-to-video transformation training

## ⚙️ Configuration Sections

### ModelConfig

Controls the base model and training mode settings.

```yaml
model:
  model_path: "/path/to/ltx-2-model.safetensors"  # Local path to model checkpoint
  text_encoder_path: "/path/to/gemma-model"       # Path to Gemma text encoder directory
  training_mode: "lora"                           # "lora" or "full"
  load_checkpoint: null                           # Path to checkpoint to resume from
```

**Key parameters:**

| Parameter           | Description                                                                                                                                                    |
|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `model_path`        | **Required.** Local path to the LTX-2 model checkpoint (`.safetensors` file). URLs are not supported.                                                          |
| `text_encoder_path` | **Required.** Path to the Gemma text encoder model directory. Download from [HuggingFace](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/). |
| `training_mode`     | Training approach - `"lora"` for LoRA training or `"full"` for full-rank fine-tuning.                                                                          |
| `load_checkpoint`   | Optional path to resume training from a checkpoint file or directory.                                                                                          |

> [!NOTE]
> LTX-2 requires both a model checkpoint and a Gemma text encoder. Both must be local paths.

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

Configures the training strategy. The trainer includes two built-in strategies described below.
For custom use cases, see [Implementing Custom Training Strategies](custom-training-strategies.md).

#### Text-to-Video Strategy

```yaml
training_strategy:
  name: "text_to_video"
  first_frame_conditioning_p: 0.1     # Probability of first-frame conditioning
  with_audio: false                   # Enable joint audio-video training
  audio_latents_dir: "audio_latents"  # Directory for audio latents (when with_audio: true)
```

#### Video-to-Video Strategy (IC-LoRA)

```yaml
training_strategy:
  name: "video_to_video"
  first_frame_conditioning_p: 0.1
  reference_latents_dir: "reference_latents"  # Directory for reference video latents
```

**Key parameters:**

| Parameter                    | Description                                                      |
|------------------------------|------------------------------------------------------------------|
| `name`                       | Strategy type: `"text_to_video"` or `"video_to_video"`           |
| `first_frame_conditioning_p` | Probability of using first frame as conditioning (0.0-1.0)       |
| `with_audio`                 | (text_to_video only) Enable joint audio-video training           |
| `audio_latents_dir`          | (text_to_video only) Directory name for audio latents            |
| `reference_latents_dir`      | (video_to_video only) Directory name for reference video latents |

### OptimizationConfig

Training optimization parameters including learning rates, batch sizes, and schedulers.

```yaml
optimization:
  learning_rate: 1e-4                  # Learning rate
  steps: 2000                          # Total training steps
  batch_size: 1                        # Batch size per GPU
  gradient_accumulation_steps: 1       # Steps to accumulate gradients
  max_grad_norm: 1.0                   # Gradient clipping threshold
  optimizer_type: "adamw"              # "adamw" or "adamw8bit"
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
| `scheduler_type`                | LR scheduler: `"constant"`, `"linear"`, `"cosine"`, `"cosine_with_restarts"`, `"polynomial"` |
| `enable_gradient_checkpointing` | Trade training speed for GPU memory savings (recommended for large models)                   |

### AccelerationConfig

Hardware acceleration and compute optimization settings.

```yaml
acceleration:
  mixed_precision_mode: "bf16"      # "no", "fp16", or "bf16"
  quantization: null                # Quantization options
  load_text_encoder_in_8bit: false  # Load text encoder in 8-bit
```

**Key parameters:**

| Parameter                   | Description                                                                        |
|-----------------------------|------------------------------------------------------------------------------------|
| `mixed_precision_mode`      | Precision mode - `"bf16"` recommended for modern GPUs                              |
| `quantization`              | Model quantization: `null`, `"int8-quanto"`, `"int4-quanto"`, `"fp8-quanto"`, etc. |
| `load_text_encoder_in_8bit` | Load the Gemma text encoder in 8-bit to save GPU memory                            |

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
| `preprocessed_data_root` | Path to your preprocessed dataset (contains `latents/`, `conditions/`, etc.)               |
| `num_dataloader_workers` | Number of parallel data loading processes (0 = synchronous loading, useful when debugging) |

### ValidationConfig

Validation and inference settings for monitoring training progress.

```yaml
validation:
  prompts: # Validation prompts
    - "A cat playing with a ball"
    - "A dog running in a field"
  negative_prompt: "worst quality, inconsistent motion, blurry, jittery, distorted"
  images: null                        # Optional image paths for image-to-video
  reference_videos: null              # Reference video paths (IC-LoRA only)
  video_dims: [ 576, 576, 89 ]        # Video dimensions [width, height, frames]
  frame_rate: 25.0                    # Frame rate for generated videos
  seed: 42                            # Random seed for reproducibility
  inference_steps: 30                 # Number of inference steps
  interval: 100                       # Steps between validation runs
  guidance_scale: 4.0                 # CFG guidance strength
  stg_scale: 1.0                      # STG guidance strength (0.0 to disable)
  stg_blocks: [ 29 ]                  # Transformer blocks to perturb for STG
  stg_mode: "stg_av"                  # "stg_av" or "stg_v" (video only)
  generate_audio: true                # Whether to generate audio
  skip_initial_validation: false      # Skip validation at step 0
  include_reference_in_output: false  # Include reference video side-by-side (IC-LoRA)
```

**Key parameters:**

| Parameter                     | Description                                                                                                              |
|-------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| `prompts`                     | List of text prompts for validation video generation                                                                     |
| `images`                      | List of image paths for image-to-video validation (must match number of prompts)                                         |
| `reference_videos`            | List of reference video paths for IC-LoRA validation (must match number of prompts)                                      |
| `video_dims`                  | Output dimensions `[width, height, frames]`. Width/height must be divisible by 32, frames must satisfy `frames % 8 == 1` |
| `interval`                    | Steps between validation runs (set to `null` to disable)                                                                 |
| `guidance_scale`              | CFG (Classifier-Free Guidance) scale. Recommended: 4.0                                                                   |
| `stg_scale`                   | STG (Spatio-Temporal Guidance) scale. 0.0 disables STG. Recommended: 1.0                                                 |
| `stg_blocks`                  | Transformer blocks to perturb for STG. Recommended: `[29]` (single block)                                                |
| `stg_mode`                    | STG mode: `"stg_av"` perturbs both audio and video, `"stg_v"` perturbs video only                                        |
| `generate_audio`              | Whether to generate audio in validation samples                                                                          |
| `include_reference_in_output` | For IC-LoRA: concatenate reference video side-by-side with output                                                        |

### CheckpointsConfig

Model checkpointing configuration.

```yaml
checkpoints:
  interval: 250       # Steps between checkpoint saves (null = disabled)
  keep_last_n: 3      # Number of recent checkpoints to retain
  precision: bfloat16 # Precision for saved weights (bfloat16 or float32)
```

**Key parameters:**

| Parameter     | Description                                                                   |
|---------------|-------------------------------------------------------------------------------|
| `interval`    | Steps between intermediate checkpoint saves (set to `null` to disable)        |
| `keep_last_n` | Number of most recent checkpoints to keep (-1 = keep all)                     |
| `precision`   | Precision for saved checkpoint weights: `"bfloat16"` (default) or `"float32"` |

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

## 🚀 Next Steps

Once you've configured your training parameters:

- Set up your dataset using [Dataset Preparation](dataset-preparation.md)
- Choose your training approach in [Training Modes](training-modes.md)
- Start training with the [Training Guide](training-guide.md)
