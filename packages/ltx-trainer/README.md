# LTX Trainer

Fine-tune Lightricks' **LTX** audio-video models. Train small LoRA adapters to customize a base model, or use full
fine-tuning to update all transformer weights. Choose from text-to-video, image-to-video, audio generation,
reference-guided transformations, extension, inpainting, and outpainting in [Training Modes](docs/training-modes.md).

## 📖 Documentation

New to the trainer? Follow **Quick Start** for your first run. For a specific task, choose a recipe in **Training Modes**,
then follow **Dataset Preparation** and the **Training Guide**. Keep **Configuration Reference** nearby for field details.

- [⚡ Quick Start](docs/quick-start.md) — Complete your first training run.
- [🛠️ Training Modes](docs/training-modes.md) — Choose a recipe and example YAML from the [config index](configs/README.md).
- [🎬 Dataset Preparation](docs/dataset-preparation.md) — Prepare metadata, encode media, and rebuild cached data.
- [⚙️ Configuration Reference](docs/configuration-reference.md) — Look up fields, defaults, and condition types.
- [🚀 Training Guide](docs/training-guide.md) — Launch, distribute, monitor, and resume training.
- [🔧 Utility Scripts](docs/utility-scripts.md) — Split scenes, generate captions, and create reference videos.
- [🧩 Custom Training Strategies](docs/custom-training-strategies.md) — Implement a different training algorithm.
- [🛡️ Troubleshooting](docs/troubleshooting.md) — Diagnose errors and memory issues.
- [🧪 ltx-pipelines](../ltx-pipelines/README.md) — Use trained weights for inference.
- [📚 ltx-core](../ltx-core/README.md) — Understand model components.

### 🤖 Agent-Assisted Training

In environments with repository skills, [`/train`](../../.claude/skills/train/SKILL.md) guides mode selection,
data preparation, preprocessing, launch, and monitoring using these docs.

## 🔧 Requirements

- **LTX Model** - Local files, in either published layout:
    - *Unified* (LTX-2, LTX-2.3) — one `.safetensors` holding the transformer, both VAEs, and the vocoder, used with
      a separate matching Gemma text-encoder directory.
    - *Split* (LTX-2.5) — separate transformer, packed text encoder, video VAE, and audio VAE files. Configure VAE
      paths for the modalities used; see [Model configuration](docs/configuration-reference.md#model).
- **Matching text encoder** - LTX 2.5 requires the LTX-specific fine-tuned Gemma 4 text encoder; do not substitute
  Google's vanilla Gemma 4 model. LTX-2/LTX-2.3 use Gemma 3.
- **Linux with CUDA** - CUDA 13+ recommended for optimal performance
- **Nvidia GPU with 80GB+ VRAM** - Recommended for the standard config. For GPUs with 32GB VRAM (e.g., RTX 5090),
  use the [low VRAM config](configs/t2v_lora_low_vram.yaml) which enables INT8 quantization and other
  memory optimizations.

## 🤝 Contributing

Report bugs and request features through GitHub issues; contributions are welcome through pull requests.
For questions, sharing training results, and announcements, join the [community Discord](https://discord.gg/ltxplatform).
