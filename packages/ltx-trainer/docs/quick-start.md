# Quick Start Guide

Get up and running with LTX-2 training in just a few steps!

## 📋 Prerequisites

Before you begin, ensure you have:

1. **LTX-2 Model Checkpoint** - A local `.safetensors` file containing the LTX-2 model weights.
   Download `ltx-2-19b-dev.safetensors` from: [HuggingFace Hub](https://huggingface.co/Lightricks/LTX-2)
2. **Gemma Text Encoder** - A local directory containing the Gemma model (required for LTX-2).
   Download from: [HuggingFace Hub](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/)
3. **CUDA or Apple Silicon MPS** - CUDA is recommended for full training performance; MPS is supported for local
   preprocessing, validation, and smaller single-process runs
4. **GPU with sufficient memory** - 80GB recommended for the standard config. For GPUs with 32GB VRAM (e.g., RTX 5090),
   use the [low VRAM config](../configs/ltx2_av_lora_low_vram.yaml) which enables INT8 quantization and other
   memory optimizations

## ⚡ Installation

First, install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already.
Then clone the repository and install the dependencies:

```bash
git clone https://github.com/Lightricks/LTX-2
```

The `ltx-trainer` package is part of the `LTX-2` monorepo. Install the dependencies from the repository root,
then navigate to the trainer package:

```bash
# From the repository root
uv sync
cd packages/ltx-trainer
```

> [!NOTE]
> The trainer depends on [`ltx-core`](../../ltx-core/) and [`ltx-pipelines`](../../ltx-pipelines/)
> packages which are automatically installed from the monorepo.

## 🏋 Training Workflow

### 1. Prepare Your Dataset

Organize your videos and captions, then preprocess them:

```bash
# Split long videos into scenes (optional)
uv run python scripts/split_scenes.py input.mp4 scenes_output_dir/ --filter-shorter-than 5s

# Generate captions for videos (optional)
uv run python scripts/caption_videos.py scenes_output_dir/ --output dataset.json

# Preprocess the dataset (compute latents and embeddings)
uv run python scripts/process_dataset.py dataset.json \
    --resolution-buckets "960x544x49" \
    --model-path /path/to/ltx-2-model.safetensors \
    --text-encoder-path /path/to/gemma-model
```

See [Dataset Preparation](dataset-preparation.md) for detailed instructions.

### 2. Configure Training

Create or modify a configuration YAML file. Start with one of the example configs:

- [`configs/ltx2_av_lora.yaml`](../configs/ltx2_av_lora.yaml) - Audio-video LoRA training
- [`configs/ltx2_av_lora_low_vram.yaml`](../configs/ltx2_av_lora_low_vram.yaml) - Audio-video LoRA training (optimized for 32GB VRAM)
- [`configs/ltx2_v2v_ic_lora.yaml`](../configs/ltx2_v2v_ic_lora.yaml) - IC-LoRA video-to-video

Key settings to update:

```yaml
model:
  model_path: "/path/to/ltx-2-model.safetensors"
  text_encoder_path: "/path/to/gemma-model"

data:
  preprocessed_data_root: "/path/to/preprocessed/data"

output_dir: "outputs/my_training_run"
```

See [Configuration Reference](configuration-reference.md) for all available options.

### 3. Start Training

```bash
uv run python scripts/train.py configs/ltx2_av_lora.yaml
```

For multi-GPU training:

```bash
uv run accelerate launch scripts/train.py configs/ltx2_av_lora.yaml
```

See [Training Guide](training-guide.md) for distributed training and advanced options.

## 🎯 Training Modes

The trainer supports several training modes:

| Mode                 | Description                    | Config Example                             |
|----------------------|--------------------------------|--------------------------------------------|
| **LoRA**             | Efficient adapter training     | `training_strategy.name: "text_to_video"`  |
| **Audio-Video LoRA** | Joint audio-video training     | `training_strategy.with_audio: true`       |
| **IC-LoRA**          | Video-to-video transformations | `training_strategy.name: "video_to_video"` |
| **Full Fine-tuning** | Full model training            | `model.training_mode: "full"`              |

See [Training Modes](training-modes.md) for detailed explanations,
or [Custom Training Strategies](custom-training-strategies.md) if you need to implement your own training recipe.

## Next Steps

Once you've completed your first training run, you can:

- **Use your trained LoRA for inference** - The [`ltx-pipelines`](../../ltx-pipelines/) package provides
  production-ready inference
  pipelines for various use cases (T2V, I2V, IC-LoRA, etc.). See the package documentation for details.
- Learn more about [Dataset Preparation](dataset-preparation.md) for advanced preprocessing
- Explore different [Training Modes](training-modes.md) (LoRA, Audio-Video, IC-LoRA)
- Dive deeper into [Training Configuration](configuration-reference.md)
- Understand the model architecture in [LTX-Core Documentation](../../ltx-core/README.md)

## Need Help?

If you run into issues at any step, see the [Troubleshooting Guide](troubleshooting.md) for solutions to common
problems.

Join our [Discord community](https://discord.gg/ltxplatform) for real-time help and discussion!
