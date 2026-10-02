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

The default `qwen_omni` backend talks to a local vLLM server, which you launch once in a separate terminal:

```bash
# Terminal 1: start the captioner server (stays running)
uv run python scripts/serve_captioner.py
```

```bash
# Terminal 2: caption your videos
uv run python scripts/caption_videos.py scenes_output_dir/ \
    --output scenes_output_dir/dataset.json
```

This will create a `dataset.json` file containing video paths and their captions.

**Captioning options:**

| Option             | Description                                                      |
| ------------------ | --------------------------------------------------------------- |
| `--captioner-type` | `qwen_omni` (default, local vLLM server) or `gemini_flash` (API) |
| `--vllm-url`       | Base URL of the vLLM server (default `http://127.0.0.1:8001/v1`) |
| `--override`       | Re-caption files that already have captions                     |
| `--api-key`        | Gemini API key (else `GEMINI_API_KEY`/`GOOGLE_API_KEY`; with no key, uses gcloud/Vertex AI auth) |

**Caption format:**

Each caption is a single, detailed paragraph describing both the visual content and the audio (speech, music, ambient sounds) of the clip. See the [Utility Scripts Reference](utility-scripts.md#automatic-video-captioning) for backend setup and the full list of options.

> [!NOTE]
> The automatically generated captions may contain inaccuracies or hallucinated content.
> We recommend reviewing and correcting the generated captions in your `dataset.json` file before proceeding to preprocessing.

## ⚡ Step 3: Dataset Preprocessing

This step preprocesses your video dataset by:

1. Resizing and cropping videos to fit specified resolution buckets
2. Computing and caching video latent representations
3. Computing and caching text embeddings for captions
4. Extracting and caching audio latents from videos (automatic, use `--skip-audio` to disable)

> [!WARNING]
> Very large videos (especially high spatial resolution and/or many frames) can cause GPU out-of-memory (OOM)
> during preprocessing/encoding.
> The simplest fix is to reduce the target resolution (spatially: width/height) and/or the number of frames
> (temporally) by using `--resolution-buckets` with smaller dimensions (lower width/height and/or fewer frames).

### Basic Usage

Which model flags you pass depends on how your checkpoint was published.

**Unified checkpoint** (LTX-2, LTX-2.3, LTX-2.5) — one `.safetensors` holding every component, plus a Gemma folder:

```bash
uv run python scripts/process_dataset.py dataset.json \
    --resolution-buckets "960x544x49" \
    --model-path /path/to/ltx-2.3-22b-dev.safetensors \
    --text-encoder-path /path/to/gemma-root
```

**Split pack** (LTX 2.5) — one file per component, so the VAEs must be named explicitly. Preprocessing needs the
video VAE to encode video latents and the audio VAE to encode audio latents; neither lives in the transformer file:

```bash
uv run python scripts/process_dataset.py dataset.json \
    --resolution-buckets "960x544x49" \
    --model-path /path/to/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors \
    --text-encoder-path /path/to/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
    --video-vae-path /path/to/vae/ltx-2.5-video-vae-bf16.safetensors \
    --audio-vae-path /path/to/vae/ltx-2.5-audio-vae-bf16.safetensors
```

> [!IMPORTANT]
> The remaining examples in this document are written for a unified checkpoint, to keep them readable. If you are on
> a split pack, add `--video-vae-path` and `--audio-vae-path` to every one of them. Leaving them off is not silently
> tolerated — the script stops with an error naming the missing flag.

Audio latents are automatically extracted from video files — no extra flag is needed. Use `--skip-audio`
to disable this. For standalone audio files (`.wav`), use the `audio` column in your dataset instead
(see [Convention-Based Column Detection](#convention-based-column-detection) below).

### 🚀 Multi-GPU Preprocessing

Preprocessing large datasets can take a while. To run it across multiple GPUs in parallel, wrap the command with
`accelerate launch` (for example `--num_processes 4`). Each process handles an interleaved slice of the dataset.
The same approach applies to `process_videos.py` and `process_captions.py` when you run them standalone.

```bash
uv run accelerate launch --num_processes 4 scripts/process_dataset.py dataset.json \
    --resolution-buckets "960x544x49" \
    --model-path /path/to/ltx-2.x-checkpoint.safetensors \
    --text-encoder-path /path/to/gemma-root
```

Outputs are written atomically (via a per-process temporary file, then renamed), so an interrupted run leaves no
corrupt files. By default a rerun **resumes** — items whose output `.pt` already exists are skipped.

> [!IMPORTANT]
> Pass **`--overwrite`** when rerunning with changed parameters (different model checkpoint, resolution buckets,
> text encoder, `--lora-trigger`, etc.). Without it the script keeps the stale outputs from the previous run.
>
> ```bash
> uv run accelerate launch --num_processes 4 scripts/process_dataset.py dataset.json \
>     --resolution-buckets "960x544x49" \
>     --model-path /path/to/ltx-2.x-checkpoint.safetensors \
>     --text-encoder-path /path/to/gemma-root \
>     --overwrite
> ```

### Switching between LTX model versions

The `conditions/` files contain features generated by the selected Gemma text encoder. They are not interchangeable
between LTX-2.3 and LTX 2.5: LTX 2.5 uses Gemma 4 and its embeddings differ from those of the older Gemma 3 model.

When changing the checkpoint, Gemma root, trigger word, or any other encoding-affecting option:

1. Prefer a new `.precomputed/` output directory.
2. If reusing an existing directory, pass `--overwrite` to `process_dataset.py`.
3. When running `process_captions.py` or `process_videos.py` directly, pass `--overwrite` to each relevant script.

The checkpoint metadata validates that the checkpoint and Gemma root are a compatible pair, but existing `.pt` files
are skipped by default and are not automatically re-encoded. Use the checkpoint's supplied metadata rather than
adding a version flag.

### Caption embedding size

Each caption is still encoded padded to 1024 tokens, but only its real tokens are stored. `process_captions.py`
keeps the last `mask.sum()` rows, because padding is on the left, and `PrecomputedDataset` pads them back to 1024 on
load. The connector replaces every pad row with a learnable register. What the model sees is therefore bit-identical
to a full 1024-token file, and caches written before this change still load as they are.

For example, one LTX 2.5 dataset holds 108,056 captions, with a median of 242 real tokens and a 95th percentile of 367.
Its caption latents went from 1,267.4 GiB to 310.0 GiB (−75.5%), or 12.59 MB to 3.08 MB per file. A set of 268,382
shorter LTX-2.3 captions, with a median of 126 tokens, went from 3,147.9 GiB to 417.6 GiB (−86.7%). The saving
depends on how much shorter your captions are than 1024 tokens.

Both sets were checked after trimming, 376,438 files in all. Padding each trimmed file back gave exactly the
stored mask and real-token rows. We also ran 254 of them, against their originals, through the real LTX 2.5 and
LTX-2.3 embeddings processors, the same `create_embeddings` call the trainer makes. The embeddings were
bit-equal, with 0 mismatches.

Loading gets faster wherever disk or network is the limit. We timed 2,000 of those captions through
`PrecomputedDataset` and a `DataLoader` with batch size 4, on a disk that reads about 205 MB/s:

| | 1024-token files | trimmed files |
| --- | --- | --- |
| Batch with a cold page cache | 245.5 ms | 60.4 ms (4.1× faster) |
| Batch with a cold page cache, 16 workers | 245.9 ms | 61.0 ms (4.0× faster) |
| Batch with a warm page cache, 16 workers | 8.1 ms | 8.4 ms (no change) |
| Copying the 2,000 files from cloud storage | 135.9 s | 60.7 s (2.2× faster) |

With the files already in memory there is no difference.

To shrink an existing cache without re-encoding:

```python
from pathlib import Path
import torch
from ltx_trainer.datasets import trim_prompt_latents

for p in Path("/path/to/.precomputed/conditions").rglob("*.pt"):
    d = torch.load(p, weights_only=True)
    if d["prompt_attention_mask"].shape[-1] == 1024 and not d["prompt_attention_mask"].all():
        tmp = p.with_suffix(".tmp")
        torch.save(trim_prompt_latents(d), tmp)
        tmp.replace(p)
```

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
> [Resolution Buckets](#resolution-buckets)), since samples with different shapes cannot be collated into a
> single batch. Use `gradient_accumulation_steps` if you need a larger effective batch.
> - Per-step cost differs substantially between a single-frame sample and a many-frame sample, which can lead to
> uneven gradient magnitudes across steps. Consider weighting the two subsets or tuning the learning rate if
> you observe instability.
> - If you prefer a fully officially-supported path, train two separate LoRAs (one on stills, one on video) and
> stack them at inference.

The dataset must be a CSV, JSON, or JSONL metadata file with columns for captions and media paths.

#### Convention-Based Column Detection

The preprocessing script automatically detects and processes columns based on their names. The following columns are recognized:

| Column | Output Dir | Description |
|--------|-----------|-------------|
| `video` (or legacy `media_path`) | `latents/` | Target video to encode |
| `audio` | `audio_latents/` | Explicit audio file (overrides auto-extraction from video) |
| `caption` | `conditions/` | Text caption for the sample |
| `reference_video` (or legacy `ref_media_path`) | `reference_latents/` | IC-LoRA reference video |
| `reference_audio` | `reference_audio_latents/` | IC-LoRA reference audio |
| `video_mask` | `video_masks/` | Binary mask for video inpainting |
| `audio_mask` | `audio_masks/` | Binary mask for audio inpainting |

> [!NOTE]
> **Legacy column names:** `media_path` and `ref_media_path` are accepted as aliases for `video` and `reference_video` respectively. Existing datasets using these names will continue to work without modification.

**JSON format example:**

```json
[
  {
    "caption": "A cat playing with a ball of yarn",
    "video": "videos/cat_playing.mp4"
  },
  {
    "caption": "A dog running in the park",
    "video": "videos/dog_running.mp4"
  }
]
```

**JSONL format example:**

```jsonl
{"caption": "A cat playing with a ball of yarn", "video": "videos/cat_playing.mp4"}
{"caption": "A dog running in the park", "video": "videos/dog_running.mp4"}
```

**CSV format example:**

```csv
caption,video
"A cat playing with a ball of yarn","videos/cat_playing.mp4"
"A dog running in the park","videos/dog_running.mp4"
```

**Additional dataset format examples:**

Audio-only dataset:
```json
{"audio": "song.wav", "caption": "piano melody"}
```

V2V IC-LoRA with reference video:
```json
{"video": "clip.mp4", "reference_video": "depth.mp4", "caption": "depth to video"}
```

A2A IC-LoRA with reference audio:
```json
{"video": "clip.mp4", "reference_audio": "ref.wav", "caption": "match this style"}
```
This form auto-extracts the target audio from `clip.mp4`. For pure audio datasets, use `audio` plus
`reference_audio` columns and preprocess with `--audio-durations`.

Video inpainting with mask:
```json
{"video": "clip.mp4", "video_mask": "mask.mp4", "caption": "fill the sky"}
```

### 📐 Resolution Buckets

Videos are organized into "buckets" of specific dimensions (width × height × frames).
Each video is assigned to the nearest matching bucket.
You can preprocess with one or multiple resolution buckets.
When training with multiple resolution buckets, you must use a batch size of 1.

The dimensions of each bucket must follow these constraints, set by the VAE's compression
factors. The values below are the **default** VAE's factors (spatial 32, temporal 8); the
preprocessing scripts derive the actual factors from the checkpoint, so a less-compressed VAE
relaxes them (e.g. a 16x16x4 VAE allows multiples of 16 and `frames % 4 == 1`):

- **Spatial dimensions** (width and height) must be multiples of the VAE spatial factor (32 by default)
- **Number of frames** must satisfy `frames % T == 1`, where `T` is the VAE temporal factor
  (8 by default, giving 1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 121, etc.)

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
> sequence_length = (H/S) * (W/S) * ((F-1)/T + 1)
> ```
>
> Where:
>
> - H = Height of video
> - W = Width of video
> - F = Number of frames
> - S = VAE's spatial downsampling factor (32 for the default VAE)
> - T = VAE's temporal downsampling factor (8 for the default VAE)
>
> For example, a 768×448×89 video through the default VAE (S=32, T=8) would have sequence length:
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
    ├── latents/                  # Video latents
    ├── conditions/               # Text embeddings
    ├── audio_latents/            # Audio latents (auto-extracted or explicit)
    ├── reference_latents/        # Reference video latents (IC-LoRA)
    ├── reference_audio_latents/  # Reference audio latents (audio IC-LoRA)
    ├── video_masks/              # Video masks (inpainting)
    └── audio_masks/              # Audio masks (audio inpainting)
```

Set `data.preprocessed_data_root` in your training config to this `.precomputed` directory — the parent directory that
contains `latents/`, `conditions/`, and any mode-specific audio/reference/mask directories.

## 🔊 Audio-Only Dataset Preprocessing

For datasets containing only audio files (no `video` column), use `--audio-durations` to specify duration buckets:

```bash
uv run python scripts/process_dataset.py dataset.json \
    --audio-durations "2.0;4.0;8.0" \
    --model-path /path/to/ltx-2-model.safetensors \
    --text-encoder-path /path/to/gemma-model
```

The `--audio-durations` flag provides duration buckets (in seconds) for audio-only datasets. Since there is no video column to derive timing from, explicit duration buckets are required.

## 🪄 IC-LoRA Reference Video Preprocessing

For IC-LoRA training, you need to preprocess datasets that include reference videos.
Reference videos provide the conditioning input while target videos represent the desired transformed output.

### Dataset Format with Reference Videos

The `reference_video` column is automatically detected by convention — no extra CLI flags are needed.

**JSON format:**

```json
[
  {
    "caption": "A cat playing with a ball of yarn",
    "video": "videos/cat_playing.mp4",
    "reference_video": "references/cat_playing_depth.mp4"
  }
]
```

**JSONL format:**

```jsonl
{"caption": "A cat playing with a ball of yarn", "video": "videos/cat_playing.mp4", "reference_video": "references/cat_playing_depth.mp4"}
{"caption": "A dog running in the park", "video": "videos/dog_running.mp4", "reference_video": "references/dog_running_depth.mp4"}
```

### Preprocessing with Reference Videos

Convention-based detection means you just need the `reference_video` column in your dataset, and `process_dataset.py` will automatically detect and process it. No `--reference-column` flag is needed:

```bash
uv run python scripts/process_dataset.py dataset.json \
    --resolution-buckets "960x544x49" \
    --model-path /path/to/ltx-2-model.safetensors \
    --text-encoder-path /path/to/gemma-model \
    --reference-downscale-factor 2 \
    --reference-temporal-scale-factor 1
```

This will create an additional `reference_latents/` directory containing the preprocessed reference video latents.
Use `--reference-downscale-factor` for spatial subsampling and `--reference-temporal-scale-factor` for temporal
subsampling. Validation reference conditions should use matching `downscale_factor` and `temporal_scale_factor` values.

> [!NOTE]
> **Legacy column names:** If your dataset uses `ref_media_path`, it is accepted as an alias for `reference_video`.

### Generating Reference Videos

**Dataset Requirements for IC-LoRA:**

- Your dataset must contain paired videos where each target video has a corresponding reference video
- Reference and target videos should cover the same content. Reference videos can optionally be lower spatial
  resolution or temporally subsampled (see Scaled Reference Conditioning in [Training Modes](training-modes.md)).
- Both reference and target videos should be preprocessed together using the same target resolution buckets, plus any
  reference scale factors you choose.

We provide an example script, `[scripts/compute_reference.py](../scripts/compute_reference.py)`, to generate reference
videos for a given dataset. The default implementation generates Canny edge reference videos.

```bash
uv run python scripts/compute_reference.py scenes_output_dir/ \
    --output scenes_output_dir/dataset.json
```

The script accepts a JSON file as the dataset configuration and updates it in-place by adding the filenames of the generated reference videos.

> [!NOTE]
> `compute_reference.py` writes generated references to the `reference_video` column, which `process_dataset.py`
> detects automatically. The legacy `ref_media_path` column is also accepted.

If you want to generate a different type of condition (depth maps, pose skeletons, etc.), modify or replace the `compute_reference()` function within this script.

### Example Dataset

For reference, see our **[Canny Control Dataset](https://huggingface.co/datasets/Lightricks/Canny-Control-Dataset)** which demonstrates proper IC-LoRA dataset structure with paired videos and Canny edge maps.

## 🎭 Mask Preprocessing for Inpainting

For inpainting training with the `mask` condition type, provide `video_mask` or `audio_mask` columns in your dataset
metadata. These columns point to mask media files (for example a mask image/video for video inpainting, or a waveform or
`.pt` tensor for audio inpainting). `process_dataset.py` downsamples and thresholds them into per-sample `.pt` tensors
under `video_masks/` or `audio_masks/`.

### Processed Video Mask Format

If you create masks manually instead of using `process_dataset.py`, save them as `.pt` files with the key `"mask"`
containing a tensor of shape `[F, H, W]` where:

- `F` = number of latent frames (temporal dimension)
- `H` = latent height (pixel height / 32)
- `W` = latent width (pixel width / 32)
- Values are thresholded at `0.5`: values `> 0.5` are conditioning tokens (clean, excluded from loss),
  and values `<= 0.5` are generated tokens (noised, contributes to loss).

### Audio Mask Format

Audio masks follow the same thresholding pattern as video masks but with shape `[T]` (temporal dimension only), where `T` is the number of audio latent frames. They are stored in `audio_masks/`.

### Directory Structure

Place masks in a directory within your preprocessed data root:

```
preprocessed_data_root/
├── latents/          # Video latents
├── conditions/       # Text embeddings
├── video_masks/      # Video masks (one .pt per sample, matching latent filenames)
└── audio_masks/      # Audio masks (one .pt per sample, matching latent filenames)
```

Then reference the mask directory in your training config:

```yaml
training_strategy:
  name: "flexible"
  video:
    is_generated: true
    latents_dir: "latents"
    conditions:
      - type: mask
        mask_dir: "video_masks"
```

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

If you add the `--decode` flag, the script will VAE-decode the precomputed video latents and save the resulting videos
in `.precomputed/decoded_videos`. Reference video latents are decoded to `.precomputed/decoded_reference_videos` when
present. To inspect audio latents, run `scripts/decode_latents.py` with `--with-audio`.

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
> The `flexible` strategy supports masks for inpainting (`mask` condition type) and spatial crop regions for outpainting (`spatial_crop` condition type) out of the box. For other custom preprocessing needs, see [Custom Training Strategies](custom-training-strategies.md).
