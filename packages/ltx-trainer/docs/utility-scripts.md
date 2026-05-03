# Utility Scripts Reference

This guide covers the various utility scripts available for preprocessing, conversion, and debugging tasks.

## 🎬 Dataset Processing Scripts

### Video Scene Splitting

The `scripts/split_scenes.py` script automatically splits long videos into shorter, coherent scenes.

```bash
# Basic scene splitting
uv run python scripts/split_scenes.py input.mp4 output_dir/ --filter-shorter-than 5s
```

**Key features:**

- **Automatic scene detection**: Uses PySceneDetect for intelligent splitting
- **Multiple algorithms**: Content-based, adaptive, threshold, and histogram detection
- **Filtering options**: Remove scenes shorter than specified duration
- **Customizable parameters**: Thresholds, window sizes, and detection modes

**Common options:**

```bash
# See all available options
uv run python scripts/split_scenes.py --help

# Use adaptive detection with custom threshold
uv run python scripts/split_scenes.py video.mp4 scenes/ --detector adaptive --threshold 30.0

# Limit to maximum number of scenes
uv run python scripts/split_scenes.py video.mp4 scenes/ --max-scenes 50
```

### Automatic Video Captioning

The `scripts/caption_videos.py` script generates captions for videos (with audio) using multimodal models.

```bash
# Generate captions for all videos in a directory (uses Qwen2.5-Omni by default)
uv run python scripts/caption_videos.py videos_dir/ --output dataset.json

# Use 8-bit quantization to reduce VRAM usage
uv run python scripts/caption_videos.py videos_dir/ --output dataset.json --use-8bit

# Use Gemini Flash API instead (requires API key)
uv run python scripts/caption_videos.py videos_dir/ --output dataset.json \
    --captioner-type gemini_flash --api-key YOUR_API_KEY

# Use Gemini Flash with parallel workers for faster throughput
uv run python scripts/caption_videos.py videos_dir/ --output dataset.json \
    --captioner-type gemini_flash --num-workers 5

# Caption without audio processing (video-only)
uv run python scripts/caption_videos.py videos_dir/ --output dataset.json --no-audio

# Force re-caption all files
uv run python scripts/caption_videos.py videos_dir/ --output dataset.json --override
```

**Key features:**

- **Audio-visual captioning**: Processes both video and audio content, including speech transcription
- **Multiple backends**:
  - `qwen_omni` (default): Local Qwen2.5-Omni model - processes video + audio locally
  - `gemini_flash`: Google Gemini Flash API - cloud-based, requires API key
- **Parallel captioning** (Gemini Flash only): Use `--num-workers` to run multiple API calls concurrently for faster throughput on large datasets
- **Structured output**: Captions include visual description, speech transcription, sounds, and on-screen text
- **Memory optimization**: 8-bit quantization option for limited VRAM
- **Incremental processing**: Skips already-captioned files by default; progress is saved every 5 videos
- **Multiple output formats**: JSON, JSONL, CSV, or TXT

**Caption format:**

The captioner produces structured captions with four sections:
- `[VISUAL]`: Detailed description of visual content
- `[SPEECH]`: Word-for-word transcription of spoken content
- `[SOUNDS]`: Description of music, ambient sounds, sound effects
- `[TEXT]`: Any on-screen text visible in the video

**Parallel captioning with Gemini Flash:**

When using `--captioner-type gemini_flash`, you can speed up large dataset captioning by running multiple API calls at the same time using `--num-workers` (accepts 1–10, default is 1):

```bash
export GEMINI_API_KEY="your-key-here"

# Caption a large dataset with 5 workers running concurrently
uv run python scripts/caption_videos.py videos_dir/ \
    --output dataset.json \
    --captioner-type gemini_flash \
    --num-workers 5
```

> [!NOTE]
> `--num-workers` is only supported with `gemini_flash`. Using it with `qwen_omni` or any other local model will raise an error, because local GPU models are not thread-safe.

> [!TIP]
> Keep `--num-workers` between 3–5 for most use cases. Very high values (8–10) may hit Gemini API rate limits depending on your quota tier.

**Environment variables (for Gemini Flash):**

Set one of these to use Gemini Flash without passing `--api-key`:
- `GOOGLE_API_KEY`
- `GEMINI_API_KEY`

### Dataset Preprocessing

The `scripts/process_dataset.py` script processes videos and caches latents for training.

```bash
# Basic preprocessing
uv run python scripts/process_dataset.py dataset.json \
    --resolution-buckets "960x544x49" \
    --model-path /path/to/ltx-2-model.safetensors \
    --text-encoder-path /path/to/gemma-model

# With audio processing
uv run python scripts/process_dataset.py dataset.json \
    --resolution-buckets "960x544x49" \
    --model-path /path/to/ltx-2-model.safetensors \
    --text-encoder-path /path/to/gemma-model \
    --with-audio

# With video decoding for verification
uv run python scripts/process_dataset.py dataset.json \
    --resolution-buckets "960x544x49" \
    --model-path /path/to/ltx-2-model.safetensors \
    --text-encoder-path /path/to/gemma-model \
    --decode
```

Multiple resolution buckets can be specified, separated by `;`:

```bash
uv run python scripts/process_dataset.py dataset.json \
    --resolution-buckets "960x544x49;512x512x81" \
    --model-path /path/to/ltx-2-model.safetensors \
    --text-encoder-path /path/to/gemma-model
```

> [!NOTE]
> When training with multiple resolution buckets, set `optimization.batch_size: 1`.

For detailed usage, see the [Dataset Preparation Guide](dataset-preparation.md).

### Reference Video Generation

The `scripts/compute_reference.py` script provides a template for creating reference videos needed for IC-LoRA training.
The default implementation generates Canny edge reference videos.

```bash
# Generate Canny edge reference videos
uv run python scripts/compute_reference.py videos_dir/ --output dataset.json
```

**Key features:**

- **Canny edge detection**: Creates edge-based reference videos
- **In-place editing**: Updates existing dataset JSON files
- **Customizable**: Modify the `compute_reference()` function for different conditions (depth, pose, etc.)

> [!TIP]
> You can edit this script to generate other types of reference videos for IC-LoRA training,
> such as depth maps, segmentation masks, or any custom video transformation.

## 🔍 Debugging and Verification Scripts

### Latents Decoding

The `scripts/decode_latents.py` script decodes precomputed video latents back into video files for visual inspection.

```bash
# Basic usage
uv run python scripts/decode_latents.py /path/to/latents/dir \
    --output-dir /path/to/output \
    --model-path /path/to/ltx-2-model.safetensors

# With VAE tiling for large videos
uv run python scripts/decode_latents.py /path/to/latents/dir \
    --output-dir /path/to/output \
    --model-path /path/to/ltx-2-model.safetensors \
    --vae-tiling

# Decode both video and audio latents
uv run python scripts/decode_latents.py /path/to/latents/dir \
    --output-dir /path/to/output \
    --model-path /path/to/ltx-2-model.safetensors \
    --with-audio
```

**The script will:**

1. **Load the VAE model** from the specified path
2. **Process all `.pt` latent files** in the input directory
3. **Decode each latent** back into a video using the VAE
4. **Save resulting videos** as MP4 files in the output directory

**When to use:**

- **Verify preprocessing quality**: Check that your videos were encoded correctly
- **Debug training data**: Visualize what the model actually sees during training
- **Quality assessment**: Ensure latent encoding preserves important visual details


### Inference Script

The `scripts/inference.py` script runs inference with a trained model.

> [!TIP]
> For production inference, consider using the [`ltx-pipelines`](../../ltx-pipelines/) package which provides optimized,
> feature-rich pipelines for various use cases:
> - **Text/Image-to-Video**: `TI2VidOneStagePipeline`, `TI2VidTwoStagesPipeline`
> - **Distilled (fast) inference**: `DistilledPipeline`
> - **IC-LoRA video-to-video**: `ICLoraPipeline`
> - **Keyframe interpolation**: `KeyframeInterpolationPipeline`
>
> All pipelines support loading custom LoRAs trained with this trainer.

```bash
# Text-to-video inference (with audio by default)
# By default, uses CFG scale 4.0 and STG scale 1.0 with block 29
uv run python scripts/inference.py \
    --checkpoint /path/to/model.safetensors \
    --text-encoder-path /path/to/gemma \
    --prompt "A cat playing with a ball" \
    --output output.mp4

# Video-only (skip audio generation)
uv run python scripts/inference.py \
    --checkpoint /path/to/model.safetensors \
    --text-encoder-path /path/to/gemma \
    --prompt "A cat playing with a ball" \
    --skip-audio \
    --output output.mp4

# Image-to-video with conditioning image
uv run python scripts/inference.py \
    --checkpoint /path/to/model.safetensors \
    --text-encoder-path /path/to/gemma \
    --prompt "A cat walking" \
    --condition-image first_frame.png \
    --output output.mp4

# Custom guidance settings
uv run python scripts/inference.py \
    --checkpoint /path/to/model.safetensors \
    --text-encoder-path /path/to/gemma \
    --prompt "A cat playing with a ball" \
    --guidance-scale 4.0 \
    --stg-scale 1.0 \
    --stg-blocks 29 \
    --output output.mp4

# Disable STG (CFG only)
uv run python scripts/inference.py \
    --checkpoint /path/to/model.safetensors \
    --text-encoder-path /path/to/gemma \
    --prompt "A cat playing with a ball" \
    --stg-scale 0.0 \
    --output output.mp4
```

**Guidance parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--guidance-scale` | 4.0 | CFG (Classifier-Free Guidance) scale |
| `--stg-scale` | 1.0 | STG (Spatio-Temporal Guidance) scale. 0.0 disables STG |
| `--stg-blocks` | 29 | Transformer block(s) to perturb for STG |
| `--stg-mode` | stg_av | `stg_av` perturbs both audio and video, `stg_v` video only |

## 🚀 Training Scripts

### Basic and Distributed Training

Use `scripts/train.py` for both single GPU and multi-GPU runs:

```bash
# Single-GPU training
uv run python scripts/train.py configs/ltx2_av_lora.yaml

# Multi-GPU (uses your accelerate config)
uv run accelerate launch scripts/train.py configs/ltx2_av_lora.yaml

# Override number of processes
uv run accelerate launch --num_processes 4 scripts/train.py configs/ltx2_av_lora.yaml
```

For detailed usage, see the [Training Guide](training-guide.md).

## 💡 Tips for Using Utility Scripts

- **Start with `--help`**: Always check available options for each script
- **Test on small datasets**: Verify workflows with a few files before processing large datasets
- **Use decode verification**: Always decode a few samples to verify preprocessing quality
- **Monitor VRAM usage**: Use `--use-8bit` or quantization flags when running into memory issues
- **Keep backups**: Make copies of important dataset files before running conversion scripts
