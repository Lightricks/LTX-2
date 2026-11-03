# Quick Start Guide

Run a text-to-video LoRA training job using one YAML for preprocessing and training.
There are two stages: **preprocessing** encodes your media and captions once, then **training** learns from those
cached results. The encoded media are called *latents*: compact representations of the video and audio.

## 📋 Prerequisites

- **Model and matching text encoder:** a unified LTX-2/LTX-2.3 checkpoint with a Gemma 3 directory
  ([download](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/)), or an LTX-2.5 split pack with
  its packed LTX-specific Gemma 4 text encoder and VAE components. Architecture is detected from checkpoint metadata.
  See [model configuration](configuration-reference.md#model) for the path fields.
- **Linux with CUDA:** the trainer depends on `triton`; CUDA 13+ is recommended.
- **GPU memory:** 80GB recommended for the standard config. For ~32GB GPUs, start from
  [the low VRAM config](../configs/t2v_lora_low_vram.yaml), which uses INT8 quantization and memory optimizations.

## ⚡ Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/Lightricks/LTX-2
cd LTX-2
uv sync
cd packages/ltx-trainer
```

`uv sync` installs the workspace packages and their dependencies. The trainer depends on `ltx-core` for model components.

> [!NOTE]
> **Working directory:** Run the commands in these guides from `packages/ltx-trainer` unless stated otherwise.

## 🏋 Training Workflow

In environments with repository skills, [`/train`](../../../.claude/skills/train/SKILL.md) can guide this workflow,
including dataset and hardware checks, launch, and monitoring. It requests approval before expensive steps.

### 1. Choose a Training Mode

Start with [t2v_lora.yaml](../configs/t2v_lora.yaml), or
[t2v_lora_low_vram.yaml](../configs/t2v_lora_low_vram.yaml) for ~32GB GPUs. Copy the example for your run:

```bash
cp configs/t2v_lora.yaml configs/my_run.yaml
```

> [!NOTE]
> **Use clips with audio:** This starter recipe trains both video and audio.

For other tasks, choose a config from [Training Modes](training-modes.md) first. Reference conditioning and inpainting
need extra metadata columns such as `reference_video`, `video_mask`, or `audio_mask` before preprocessing.

### 2. Prepare Your Dataset Metadata

Create CSV, JSON, or JSONL metadata with `video` and `caption` fields. For example, `dataset.json`:

```json
[
  {"video": "videos/clip.mp4", "caption": "A cat walks across a sunlit room."}
]
```

The `video` path is relative to the directory containing `dataset.json`; absolute paths also work.

#### Optional: Split Scenes and Generate Captions

If your clips and captions are ready, continue to step 3. Otherwise, split long videos into scenes:

```bash
uv run python scripts/split_scenes.py input.mp4 scenes_output_dir/ --filter-shorter-than 5s
```

The default captioner needs its own GPU memory and a running local server. For captioning without a local GPU, see
the [Gemini backend](utility-scripts.md#automatic-video-captioning). To use the default, start the server in another
terminal and leave it running:

```bash
# Separate terminal, from packages/ltx-trainer; leave running while captioning
uv run python scripts/serve_captioner.py
```

Then, back in your original terminal:

```bash
uv run python scripts/caption_videos.py scenes_output_dir/ --output dataset.json
```

Review generated captions before encoding. See [Dataset Preparation](dataset-preparation.md) for media formats,
audio handling, and references, or [captioner setup](utility-scripts.md#automatic-video-captioning) for other backends.

### 3. Configure Your Run

Edit these fields in `configs/my_run.yaml`, keeping the example's remaining settings:

```yaml
training_strategy:
  name: "flexible"
  model:
    model_path: "/path/to/ltx-2-model.safetensors"
    text_encoder_path: "/path/to/gemma-model"
    video_vae_path: null  # Set the component path for an LTX-2.5 split pack
    audio_vae_path: null  # Set the component path when the split-pack run uses audio
  data:
    dataset_file: "/path/to/dataset.json"
    preprocessed_data_root: "/path/to/preprocessed/data"
    resolution_buckets:
      - [960, 544, 49]  # [width, height, frames]

output_dir: "outputs/my_training_run"
```

Use the matching model/text-encoder layout from [model configuration](configuration-reference.md#model).
Also update any validation media paths and prompts in the copied config. See
[Configuration Reference](configuration-reference.md) for the remaining fields.

### 4. Preprocess the Dataset

Encode media into latents and captions into text embeddings (the model's representation of text).
Use the same config you will train with:

```bash
uv run python scripts/preprocess.py configs/my_run.yaml
```

Preprocessing writes to `training_strategy.data.preprocessed_data_root`. If unset, preprocessing uses
`<dataset_file parent>/.precomputed`; set that path explicitly in the config before training.
A settings record, `preprocess_manifest.json`, lets the trainer check that the cache matches your run config.

> [!IMPORTANT]
> Reruns skip existing outputs. **After changing encoding settings or source data, use `--overwrite` to recompute them.**
> See [rerun and manifest behavior](dataset-preparation.md#preprocessing-manifest-and-staleness-guard) for details.

### 5. Start Training

```bash
uv run python scripts/train.py configs/my_run.yaml
```

Before a long run, try a small subset and a few hundred steps; confirm validation samples and checkpoints appear in
`output_dir`. See the [Training Guide](training-guide.md) for outputs, multi-GPU launch, W&B, and resume.

## 🚀 Next Steps

After your first run:

- Try another task using the recipes in [Training Modes](training-modes.md).
- Tune your run with the fields and defaults in [Configuration Reference](configuration-reference.md).
- Use your trained weights with the [inference entry points](training-modes.md#-using-trained-models-for-inference).

If a step fails, see [Troubleshooting](troubleshooting.md).
