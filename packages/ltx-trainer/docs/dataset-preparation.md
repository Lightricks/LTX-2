# Dataset Preparation Guide

This guide covers the complete workflow for preparing and preprocessing your dataset for training.

## 📋 Overview

The general dataset preparation workflow is:

1. **(Optional)** Split long videos into scenes using `split_scenes.py`
2. **(Optional)** Generate captions for your videos using `caption_videos.py`
3. **Preprocess your dataset** using `process_dataset.py` to compute and cache video/audio latents and text embeddings
4. **Run the trainer** with your preprocessed dataset

## 🎬 Step 1: Split Scenes

If you're starting with raw, long-form videos (e.g., downloaded from YouTube), you should first split them into shorter, coherent scenes.

```bash
uv run python scripts/split_scenes.py input.mp4 scenes_output_dir/ \
    --filter-shorter-than 5s
```

This will create multiple video clips in `scenes_output_dir`.
These clips will be the input for the captioning step, if you choose to use it.

The script supports many configuration options for scene detection (detector algorithms, thresholds, minimum scene lengths, etc.):

```bash
uv run python scripts/split_scenes.py --help
```

## 📝 Step 2: Caption Videos

If your dataset doesn't include captions, you can automatically generate them using multimodal models that understand both video and audio.

```bash
uv run python scripts/caption_videos.py scenes_output_dir/ \
    --output scenes_output_dir/dataset.json
```

If you're running into VRAM issues, try enabling 8-bit quantization to reduce memory usage:

```bash
uv run python scripts/caption_videos.py scenes_output_dir/ \
    --output scenes_output_dir/dataset.json \
    --use-8bit
```

This will create a `dataset.json` file containing video paths and their captions.

**Captioning options:**


| Option             | Description                                                |
| ------------------ | ---------------------------------------------------------- |
| `--captioner-type` | `qwen_omni` (default, local) or `gemini_flash` (API)       |
| `--use-8bit`       | Enable 8-bit quantization for lower VRAM usage             |
| `--no-audio`       | Disable audio processing (video-only captions)             |
| `--override`       | Re-caption files that already have captions                |
| `--api-key`        | API key for Gemini Flash (or set `GOOGLE_API_KEY` env var) |


**Caption format:**

The captioner produces structured captions with sections for:

- **Visual content**: People, objects, actions, settings, colors, movements
- **Speech transcription**: Word-for-word transcription of spoken content
- **Sounds**: Music, ambient sounds, sound effects
- **On-screen text**: Any visible text overlays

> [!NOTE]
> The automatically generated captions may contain inaccuracies or hallucinated content.
> We recommend reviewing and correcting the generated captions in your `dataset.json` file before proceeding to preprocessing.

## ⚡ Step 3: Dataset Preprocessing

This step preprocesses your video dataset by:

1. Resizing and cropping videos to fit specified resolution buckets
2. Computing and caching video latent representations
3. Computing and caching text embeddings for captions
4. (Optional) Computing and caching audio latents

> [!WARNING]
> Very large videos (especially high spatial resolution and/or many frames) can cause GPU out-of-memory (OOM)
> during preprocessing/encoding.
> The simplest fix is to reduce the target resolution (spatially: width/height) and/or the number of frames
> (temporally) by using `--resolution-buckets` with smaller dimensions (lower width/height and/or fewer frames).

### Basic Usage

```bash
uv run python scripts/process_dataset.py dataset.json \
    --resolution-buckets "960x544x49" \
    --model-path /path/to/ltx-2-model.safetensors \
    --text-encoder-path /path/to/gemma-model
```

### With Audio Processing

For audio-video training, add the `--with-audio` flag:

```bash
uv run python scripts/process_dataset.py dataset.json \
    --resolution-buckets "960x544x49" \
    --model-path /path/to/ltx-2-model.safetensors \
    --text-encoder-path /path/to/gemma-model \
    --with-audio
```

### 🚀 Multi-GPU Preprocessing

Preprocessing large datasets can take a while. To run it across multiple GPUs in parallel, wrap the command with
`accelerate launch` (for example `--num_processes 4`). Each process handles an interleaved slice of the dataset.
The same approach applies to `process_videos.py` and `process_captions.py` when you run them standalone.

```bash
uv run accelerate launch --num_processes 4 scripts/process_dataset.py dataset.json \
    --resolution-buckets "960x544x49" \
    --model-path /path/to/ltx-2-model.safetensors \
    --text-encoder-path /path/to/gemma-model
```

Outputs are written atomically (via a per-process temporary file, then renamed), so an interrupted run leaves no
corrupt files. By default a rerun **resumes** — items whose output `.pt` already exists are skipped.

> [!IMPORTANT]
> Pass `**--overwrite`** when rerunning with changed parameters (different model checkpoint, resolution buckets,
> text encoder, `--lora-trigger`, etc.). Without it the script keeps the stale outputs from the previous run.
>
> ```bash
> uv run accelerate launch --num_processes 4 scripts/process_dataset.py dataset.json \
>     --resolution-buckets "960x544x49" \
>     --model-path /path/to/ltx-2.3-model.safetensors \
>     --text-encoder-path /path/to/gemma-model \
>     --overwrite
> ```

### 📊 Dataset Format

The trainer supports videos, single images, or a mix of both in the same dataset.

> [!TIP]
> **Image Datasets:** When using images, follow the same preprocessing steps and format requirements as with videos,
> but use `1` for the frame count in the resolution bucket (e.g., `960x544x1`).

> [!NOTE]
> **Mixed image + video datasets:** Mixing stills and videos in a single dataset is supported, but requires some care:
>
> - Preprocess with **multiple resolution buckets** covering both frame counts — e.g.
> `--resolution-buckets "960x544x1;960x544x49"`. Images are automatically assigned to the `F=1` bucket and
> videos to an `F>1` bucket.
> - You **must** set `optimization.batch_size: 1` in your training config (see the warning under
> [Resolution Buckets](#-resolution-buckets)), since samples with different shapes cannot be collated into a
> single batch. Use `gradient_accumulation_steps` if you need a larger effective batch.
> - Per-step cost differs substantially between a single-frame sample and a many-frame sample, which can lead to
> uneven gradient magnitudes across steps. Consider weighting the two subsets or tuning the learning rate if
> you observe instability.
> - If you prefer a fully officially-supported path, train two separate LoRAs (one on stills, one on video) and
> stack them at inference.

The dataset must be a CSV, JSON, or JSONL metadata file with columns for captions and video paths:

**JSON format example:**

```json
[
  {
    "caption": "A cat playing with a ball of yarn",
    "media_path": "videos/cat_playing.mp4"
  },
  {
    "caption": "A dog running in the park",
    "media_path": "videos/dog_running.mp4"
  }
]
```

**JSONL format example:**

```jsonl
{"caption": "A cat playing with a ball of yarn", "media_path": "videos/cat_playing.mp4"}
{"caption": "A dog running in the park", "media_path": "videos/dog_running.mp4"}
```

**CSV format example:**

```csv
caption,media_path
"A cat playing with a ball of yarn","videos/cat_playing.mp4"
"A dog running in the park","videos/dog_running.mp4"
```

### 📐 Resolution Buckets

Videos are organized into "buckets" of specific dimensions (width × height × frames).
Each video is assigned to the nearest matching bucket.
You can preprocess with one or multiple resolution buckets.
When training with multiple resolution buckets, you must use a batch size of 1.

The dimensions of each bucket must follow these constraints due to LTX-2's VAE architecture:

- **Spatial dimensions** (width and height) must be multiples of 32
- **Number of frames** must satisfy `frames % 8 == 1` (e.g., 1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 121, etc.)

**Guidelines for choosing training resolution:**

- For high-quality, detailed videos: use larger spatial dimensions (e.g. 768x448) with fewer frames (e.g. 89)
- For longer, motion-focused videos: use smaller spatial dimensions (512×512) with more frames (121)
- Memory usage increases with both spatial and temporal dimensions

**Example usage:**

```bash
uv run python scripts/process_dataset.py dataset.json \
    --resolution-buckets "960x544x49" \
    --model-path /path/to/ltx-2-model.safetensors \
    --text-encoder-path /path/to/gemma-model
```

Multiple buckets are supported by separating entries with `;`:

```bash
uv run python scripts/process_dataset.py dataset.json \
    --resolution-buckets "960x544x49;512x512x49" \
    --model-path /path/to/ltx-2-model.safetensors \
    --text-encoder-path /path/to/gemma-model
```

**Video processing workflow:**

1. Videos are **resized** maintaining aspect ratio until either width or height matches the target
2. The larger dimension is **center cropped** to match the bucket's dimensions
3. Only the **first X frames are taken** to match the bucket's frame count, remaining frames are ignored

> [!NOTE]
> The sequence length processed by the transformer model can be calculated as:
>
> ```
> sequence_length = (H/32) * (W/32) * ((F-1)/8 + 1)
> ```
>
> Where:
>
> - H = Height of video
> - W = Width of video
> - F = Number of frames
> - 32 = VAE's spatial downsampling factor
> - 8 = VAE's temporal downsampling factor
>
> For example, a 768×448×89 video would have sequence length:
>
> ```
> (768/32) * (448/32) * ((89-1)/8 + 1) = 24 * 14 * 12 = 4,032
> ```
>
> Keep this in mind when choosing video dimensions, as longer sequences require more GPU memory.

> [!WARNING]
> When training with multiple resolution buckets, you must use a batch size of 1
> (i.e., set `optimization.batch_size: 1` in your training config).

### 📁 Output Structure

The preprocessed data is saved in a `.precomputed` directory:

```
dataset/
└── .precomputed/
    ├── latents/            # Cached video latents
    ├── conditions/         # Cached text embeddings
    ├── audio_latents/      # (only if --with-audio) Cached audio latents
    └── reference_latents/  # (only for IC-LoRA) Cached reference video latents
```

## 🪄 IC-LoRA Reference Video Preprocessing

For IC-LoRA training, you need to preprocess datasets that include reference videos.
Reference videos provide the conditioning input while target videos represent the desired transformed output.

### Dataset Format with Reference Videos

**JSON format:**

```json
[
  {
    "caption": "A cat playing with a ball of yarn",
    "media_path": "videos/cat_playing.mp4",
    "reference_path": "references/cat_playing_depth.mp4"
  }
]
```

**JSONL format:**

```jsonl
{"caption": "A cat playing with a ball of yarn", "media_path": "videos/cat_playing.mp4", "reference_path": "references/cat_playing_depth.mp4"}
{"caption": "A dog running in the park", "media_path": "videos/dog_running.mp4", "reference_path": "references/dog_running_depth.mp4"}
```

### Preprocessing with Reference Videos

To preprocess a dataset with reference videos, add the `--reference-column` argument specifying the name of the field
in your dataset JSON/JSONL/CSV that contains the reference video paths:

```bash
uv run python scripts/process_dataset.py dataset.json \
    --resolution-buckets "960x544x49" \
    --model-path /path/to/ltx-2-model.safetensors \
    --text-encoder-path /path/to/gemma-model \
    --reference-column "reference_path"
```

This will create an additional `reference_latents/` directory containing the preprocessed reference video latents.

### Generating Reference Videos

**Dataset Requirements for IC-LoRA:**

- Your dataset must contain paired videos where each target video has a corresponding reference video
- Reference and target videos must have *identical* resolution and length
- Both reference and target videos should be preprocessed together using the same resolution buckets

We provide an example script, `[scripts/compute_reference.py](../scripts/compute_reference.py)`, to generate reference
videos for a given dataset. The default implementation generates Canny edge reference videos.

```bash
uv run python scripts/compute_reference.py scenes_output_dir/ \
    --output scenes_output_dir/dataset.json
```

The script accepts a JSON file as the dataset configuration and updates it in-place by adding the filenames of the generated reference videos.

If you want to generate a different type of condition (depth maps, pose skeletons, etc.), modify or replace the `compute_reference()` function within this script.

### Example Dataset

For reference, see our **[Canny Control Dataset](https://huggingface.co/datasets/Lightricks/Canny-Control-Dataset)** which demonstrates proper IC-LoRA dataset structure with paired videos and Canny edge maps.

## 🎯 LoRA Trigger Words

When training a LoRA, you can specify a trigger token that will be prepended to all captions:

```bash
uv run python scripts/process_dataset.py dataset.json \
    --resolution-buckets "960x544x49" \
    --model-path /path/to/ltx-2-model.safetensors \
    --text-encoder-path /path/to/gemma-model \
    --lora-trigger "MYTRIGGER"
```

This acts as a trigger word that activates the LoRA during inference when you include the same token in your prompts.

> [!NOTE]
> There is no need to manually insert the trigger word into your dataset JSON/JSONL/CSV file.
> The trigger word specified with `--lora-trigger` is automatically prepended to each caption during preprocessing.

## 🔍 Decoding Videos for Verification

If you add the `--decode` flag, the script will VAE-decode the precomputed latents and save the resulting videos
in `.precomputed/decoded_videos`. When audio preprocessing is enabled (`--with-audio`), audio latents will also be
decoded and saved to `.precomputed/decoded_audio`. This allows you to visually and audibly inspect the processed data.

```bash
uv run python scripts/process_dataset.py dataset.json \
    --resolution-buckets "960x544x49" \
    --model-path /path/to/ltx-2-model.safetensors \
    --text-encoder-path /path/to/gemma-model \
    --decode
```

For single-frame images, the decoded latents will be saved as PNG files rather than MP4 videos.

## 🚀 Next Steps

Once your dataset is preprocessed, you can proceed to:

- Configure your training parameters in [Configuration Reference](configuration-reference.md)
- Choose your training approach in [Training Modes](training-modes.md)
- Start training with the [Training Guide](training-guide.md)

> [!TIP]
> If your training recipe requires additional preprocessed data (e.g., masks, conditioning signals), see
> [Implementing Custom Training Strategies](custom-training-strategies.md) for guidance on extending the
> preprocessing pipeline.
