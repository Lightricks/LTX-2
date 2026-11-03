# Dataset Preparation Guide

Prepare a metadata file listing media paths and captions, then preprocess it once before training.
Preprocessing turns video/audio into compact representations called **latents**, and captions into **text embeddings**.
Training reads these cached files instead of encoding the original media at every step.
Commands assume `packages/ltx-trainer` as the working directory.

## 📋 Overview

1. Optionally split long videos into scenes and generate captions.
2. Choose a [training mode](training-modes.md) and prepare its metadata columns.
3. Run `scripts/preprocess.py CONFIG` to cache video/audio latents and text embeddings.
4. [Train](training-guide.md) using the same YAML.

If your clips already have captions and metadata, skip steps 1 and 2 below and start with preprocessing.

Preprocessing settings live under `training_strategy.data` in the training config: dataset location, resolution
buckets, reference/audio settings, and LoRA trigger. CLI flags control execution, such as overwrite and device selection.

## 🎬 Step 1: Split Scenes

Split long videos into coherent clips if needed:

```bash
uv run python scripts/split_scenes.py input.mp4 scenes_output_dir/ --filter-shorter-than 5s
```

Use the resulting clips for captioning. See [scene-splitting options](utility-scripts.md#video-scene-splitting)
for detectors, thresholds, and filtering.

## 📝 Step 2: Caption Videos

If captions are missing, the default `qwen_omni` backend generates combined visual/audio descriptions through a local
vLLM server. This needs additional GPU memory; the [Gemini backend](utility-scripts.md#automatic-video-captioning)
is an alternative without a local GPU. For the default backend, start the server in a separate terminal and leave it running:

```bash
# Terminal 1; leave running while captioning
uv run python scripts/serve_captioner.py
```

```bash
# Terminal 2
uv run python scripts/caption_videos.py scenes_output_dir/ --output scenes_output_dir/dataset.json
```

> [!TIP]
> **Review generated captions** for accuracy before encoding; correct any invented or missing details.

Each caption is a detailed paragraph covering visual content and audio, including speech, music, and ambient sounds. See
[captioner setup and options](utility-scripts.md#automatic-video-captioning) for the Gemini backend, remote servers,
authentication, and re-captioning.

## ⚡ Step 3: Dataset Preprocessing

Preprocessing resizes/crops media to resolution buckets and caches video latents, caption embeddings, and audio
latents for configured modalities.

> [!WARNING]
> **Encoding out of memory?** Reduce bucket width, height, or frame count; use `--overwrite` when rebuilding
> existing artifacts with the new settings.

### Basic Usage

Edit these fields in a complete [example config](../configs/README.md), then run preprocessing.
For a split checkpoint pack, also set the [component model paths](configuration-reference.md#model).

```yaml
training_strategy:
  name: "flexible"
  model:
    model_path: "/path/to/ltx-2-model.safetensors"
    text_encoder_path: "/path/to/gemma-model"
  data:
    dataset_file: "/path/to/dataset.json"      # raw metadata (CSV/JSON/JSONL)
    resolution_buckets:
      - [960, 544, 49]                          # [width, height, frames]
```

```bash
uv run python scripts/preprocess.py configs/t2v_lora.yaml
```

> [!NOTE]
> **Encoded outputs follow your config.** Only declared modalities and conditions are encoded:
>
> - **Audio:** With `training_strategy.audio`, audio is extracted from video into that modality's `latents_dir` unless
>   `data.skip_audio: true`. An explicit `audio` column takes precedence; use it for standalone audio files (`.wav`).
> - **References:** `reference_latents/` is produced only when a `reference` condition (with a `latents_dir`) is present
>   on a modality **and** the dataset has a `reference_video`/`reference_audio` column.
>
> See [column detection](#convention-based-column-detection) for metadata names and output directories.

### 🚀 Multi-GPU Preprocessing

Preprocessing large datasets can take a while. To run it across multiple GPUs in parallel, wrap the command with
`accelerate launch` (for example `--num_processes 4`). Each process handles an interleaved slice of the dataset.

```bash
uv run accelerate launch --num_processes 4 scripts/preprocess.py configs/t2v_lora.yaml
```

Each process writes to a temporary file, then renames it when complete. An interruption therefore does not leave a
partially written file at the final output path. By default a rerun **resumes**: items whose output `.pt` already exists
are skipped.

> [!IMPORTANT]
> Pass **`--overwrite`** after changing encoding settings (model, buckets, text encoder, trigger, etc.).
> The manifest guard rejects incremental runs with incompatible settings. Source edits at the same paths are not
> fingerprinted, so also overwrite after changing media, captions, or checkpoint contents in place.
>
> ```bash
> uv run accelerate launch --num_processes 4 scripts/preprocess.py configs/t2v_lora.yaml --overwrite
> ```
>
> A specific device can be selected with `--device` (default `cuda`).

### 📊 Dataset Format

Use CSV, JSON, or JSONL metadata with captions and media paths. Videos, single images, and mixed datasets are supported.
Relative media paths resolve from the metadata file's directory; absolute paths are also accepted.

> [!TIP]
> **Image datasets:** Use the same metadata and preprocessing as video, with a one-frame bucket such as `[960, 544, 1]`.

> [!NOTE]
> **Mixed image + video datasets:** Include buckets for both frame counts, for example
> `resolution_buckets: [[960, 544, 1], [960, 544, 49]]`. Images use `F=1`; videos use `F>1`.
>
> - **Set `optimization.batch_size: 1`** because shapes differ; use gradient accumulation for a larger effective batch.
> - Per-step cost differs between stills and video; evaluate the subsets separately if training becomes unstable.
>   Separate image/video LoRAs stacked at inference are an alternative.

#### Convention-Based Column Detection

Preprocessing detects dataset columns by name and maps them to roles; which roles are actually encoded also depends
on the modalities/conditions in your config (see the note in [Basic Usage](#basic-usage)). The following columns are
recognized:

| Column | Output Dir | Description |
|--------|-----------|-------------|
| `video` (or legacy `media_path`) | `latents/` | Target video to encode |
| `audio` | `audio_latents/` | Explicit audio file (overrides auto-extraction from video) |
| `caption` | `conditions/` | Text caption for the sample |
| `reference_video` (or legacy `ref_media_path`) | `reference_latents/` | IC-LoRA reference video |
| `reference_audio` | `reference_audio_latents/` | IC-LoRA reference audio |
| `video_mask` | `video_masks/` | Binary mask for video inpainting |
| `audio_mask` | `audio_masks/` | Binary mask for audio inpainting |

Directory names in the table are conventions from the example configs. Actual outputs use each modality/reference
condition's `latents_dir` and mask condition's `mask_dir`, under `preprocessed_data_root`.

**JSON format example:**

```json
[
  {
    "caption": "A cat playing with a ball of yarn",
    "video": "videos/cat_playing.mp4"
  }
]
```

**JSONL format example:**

```jsonl
{"caption": "A cat playing with a ball of yarn", "video": "videos/cat_playing.mp4"}
```

**CSV format example:**

```csv
caption,video
"A cat playing with a ball of yarn","videos/cat_playing.mp4"
```

**Additional metadata rows:** These are JSONL examples; wrap rows in an array for a `.json` file.

**Audio-only dataset:**

```jsonl
{"audio": "song.wav", "caption": "piano melody"}
```

**V2V IC-LoRA with reference video:**

```jsonl
{"video": "clip.mp4", "reference_video": "depth.mp4", "caption": "depth to video"}
```

**A2A IC-LoRA with reference audio:**

```jsonl
{"video": "clip.mp4", "reference_audio": "ref.wav", "caption": "match this style"}
```

This form auto-extracts the target audio from `clip.mp4`. For pure audio datasets, use `audio` plus
`reference_audio` columns and set `data.audio_durations` in your config.

**Video inpainting with mask:**

```jsonl
{"video": "clip.mp4", "video_mask": "mask.mp4", "caption": "fill the sky"}
```

### 📐 Resolution Buckets

A **bucket** is a fixed output size: `[width, height, frames]`. Preprocessing assigns media to the nearest matching
entry in `data.resolution_buckets`, then resizes and crops it to fit.

The VAE is the model component that compresses media into latents. Its compression factors determine which sizes are
valid. The trainer reads these factors from the checkpoint; the default VAE has spatial factor `S=32` and temporal
factor `T=8`:

- Width and height must be multiples of `S`.
- Frames must satisfy `(frames - 1) % T == 0`: for `T=8`, examples are 1, 9, 17, 25, 49, 89, and 121.
  A 16×16×4 VAE instead allows multiples of 16 and frame counts 1, 5, 9, 13, etc.

For example, a 960×544 video with 49 pixel frames becomes a 30×17 latent grid with 7 latent frames.

Use larger spatial dimensions for detail, or fewer pixels and more frames for longer motion sequences.
Both spatial and temporal dimensions increase memory use. Add entries for multiple buckets:

```yaml
training_strategy:
  data:
    resolution_buckets:
      - [960, 544, 49]
      - [512, 512, 49]
```

The shorthand `resolution_buckets: "960x544x49;512x512x49"` is also accepted.

> [!WARNING]
> **Multiple buckets require `optimization.batch_size: 1`**, since different shapes cannot share a batch.

**Video processing workflow:**

1. Resize the video while preserving its aspect ratio until either width or height matches the target.
2. Crop to bucket dimensions (`reshape_mode: "center"` by default; `"random"` is also supported).
3. Take the bucket's frame count from the **start of the video**; ignore remaining frames.

Video token count is `(H/S) × (W/S) × ((F-1)/T + 1)`, where `H`, `W`, and `F` are pixel height, width, and frame count;
`S`/`T` are the VAE compression factors. With `S=32`, `T=8`, a 768×448×89 clip produces
`24 × 14 × 12 = 4,032` tokens. Larger token counts require more GPU memory.

### 📁 Output Structure

Outputs go to `data.preprocessed_data_root`, defaulting to `<dataset_file parent>/.precomputed` during preprocessing.
Only configured modalities and conditions produce directories:

```
dataset/
└── .precomputed/
    ├── preprocess_manifest.json  # Settings used to build this cache
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

Training discovers cached `.pt` files and uses samples with matching relative paths in **every required directory**.
For example, `latents/clip.pt` needs a matching `conditions/clip.pt` and any required audio/reference/mask files.
Check the logs for skipped samples: missing files can exclude items.

> [!IMPORTANT]
> Training reads the cache, not the raw metadata. Removing a metadata row does not remove its cached files.
> Use a fresh `preprocessed_data_root` when you need a clean rebuild without those samples.

#### Preprocessing Manifest and Staleness Guard

The **manifest**, `preprocess_manifest.json`, records how a cache was built. It stores a `format_version` (currently `1`)
and a `fingerprint` object: the settings that affect the encoded results. Training compares these settings with the
current config before loading models.

For the flexible strategy, the fingerprint includes:

- Model/text-encoder paths, text-encoder 8-bit mode, and trigger word.
- Resolution buckets (stored in canonical `[frames, height, width]` order), crop mode, and VAE tiling.
- Reference scale factors, audio duration/extraction settings, and the mapping from cache directories to batch keys.

Execution settings such as encoding batch size, worker counts, decode verification, and optimization parameters are
excluded because they do not describe the encoded data.

The guard runs at two points:

- **Before preprocessing:** an incremental run over a non-empty root needs a matching manifest; a missing or
  mismatching one requires `--overwrite`, so existing files can't be skipped and then marked current.
- **Before training loads models:** a mismatch fails with a field-level diff. Missing manifests warn and continue
  (legacy compatibility); corrupt or unsupported manifests fail.

The fingerprint compares settings and resolved model paths; it does **not** compare the contents of checkpoint files,
metadata, captions, or source media. Choose how to rerun based on what changed:

| Change | What to do |
|--------|------------|
| Add new samples with the same encoding settings | Rerun preprocessing; existing `.pt` outputs are skipped. |
| Edit captions/media, replace checkpoint contents at the same path, or change encoding settings | Use `--overwrite` or a fresh output root to recompute the cache. |
| Remove/rename samples or stop using a cached role | Use a fresh output root for a clean rebuild; `--overwrite` does not delete old files. |

Use `--check` for a strict, read-only comparison (no encoding or model loading); unlike training, it also exits nonzero
for a missing manifest, since compatibility can't be proven. It validates the config in training mode, so set
`data.preprocessed_data_root` explicitly. It cannot be combined with `--overwrite`:

```bash
uv run python scripts/preprocess.py configs/your_config.yaml --check
```

To intentionally train despite a mismatch, set
[`allow_stale_preprocessed`](configuration-reference.md#allow_stale_preprocessed) to `true` — this downgrades the
training failure to a warning but does not affect `--check`.

In multi-GPU runs, the main process writes the manifest only after every GPU process finishes.

## 🔊 Audio-Only Dataset Preprocessing

For datasets containing only audio files (no `video` column), set `data.audio_durations` to specify duration buckets:

```yaml
training_strategy:
  data:
    audio_durations: [2.0, 4.0, 8.0]   # duration buckets in seconds
```

```bash
uv run python scripts/preprocess.py configs/t2a_lora.yaml
```

> [!NOTE]
> **Variable-length audio requires `optimization.batch_size: 1`.** Without `data.audio_durations`, each file is encoded
> at its full duration. Set duration buckets (in seconds) to trim longer files into uniform lengths or skip files shorter
> than the smallest bucket.

## 🪄 IC-LoRA Reference Video Preprocessing

Add `reference_video` to each metadata row and declare a video `reference` condition. Reference media is encoded into
that condition's `latents_dir`. See [column detection](#convention-based-column-detection) for audio references and
legacy column aliases. For example, a JSONL row:

```jsonl
{"video": "clip.mp4", "reference_video": "depth.mp4", "caption": "A cat playing with yarn"}
```

Preprocess targets and references together; they should cover the same content.
The [Canny Control Dataset](https://huggingface.co/datasets/Lightricks/Canny-Control-Dataset) demonstrates paired media.

> [!WARNING]
> **Reference sources need the full target frame count.** They must contain at least the target bucket's frame count;
> shorter sources are skipped even when temporal subsampling is enabled.

### Scaled Reference Conditioning

Lower-resolution or temporally subsampled references reduce conditioning token count while targets keep their configured
resolution. Declare target/reference ratios under `training_strategy.data`, including when training an existing
precomputed dataset.

For example, spatial factor `2` turns a 960×576 target into a 480×288 reference. Temporal factor `2` would reduce
49 target frames to 25 reference frames. The config below scales only space and keeps the same frame count:

```yaml
training_strategy:
  data:
    resolution_buckets:
      - [960, 576, 49]
    reference_spatial_scale_factor: 2
    reference_temporal_scale_factor: 1
  video:
    is_generated: true
    latents_dir: "latents"
    conditions:
      - type: reference
        latents_dir: "reference_latents"
```

```bash
uv run python scripts/preprocess.py configs/v2v_ic_lora.yaml
```

- Scale factors are positive integers; `1` means no scaling. The spatial factor applies equally to height and width,
  preserving aspect ratio. Target and encoded reference dimensions must be VAE-aligned (multiples of 32 by default).
- For temporal factor `K`, bucket frames must satisfy `(frames - 1) % (K * T) == 0`, where `T` is the VAE temporal
  factor (8 by default).
- **If either factor exceeds 1, preprocessing requires exactly one resolution bucket.**
- Training verifies declared ratios against every reference/target batch before adjusting positional encodings.
  Single-frame pairs retain the configured temporal factor because frame counts cannot determine the ratio.
- `reference_downscale_factor` is a legacy alias for `reference_spatial_scale_factor`.

> [!IMPORTANT]
> **Validation inherits reference scale factors from `training_strategy.data`.** Do not set them on validation
> conditions; that is rejected.

Validation encodes reference media on the fly. Supply only the reference media and optional output display setting:

```yaml
training_strategy:
  validation:
    samples:
      - prompt: "A cat playing with yarn"
        conditions:
          - type: reference
            video: "/path/to/reference.mp4"
            include_in_output: true
```

### Generating Reference Videos

Generate Canny references and add the `reference_video` column to your metadata:

```bash
uv run python scripts/compute_reference.py scenes_output_dir/ --output scenes_output_dir/dataset.json
```

See [reference generation](utility-scripts.md#reference-video-generation) to customize `compute_reference()` for depth,
pose, or other transformations.

## 🎭 Mask Preprocessing for Inpainting

For inpainting training with the `mask` condition type, provide `video_mask` or `audio_mask` columns in your dataset
metadata. These columns point to mask media files (for example a mask image/video for video inpainting, or a waveform or
`.pt` tensor for audio inpainting). Preprocessing downsamples and thresholds them into per-sample `.pt` tensors
under `video_masks/` or `audio_masks/` (the `mask_dir` declared on the mask condition).

**A mask marks what to keep.** For a black-and-white video mask, white keeps the source and black asks the model to
generate that region:

| Mask value | Meaning |
|------------|---------|
| `> 0.5` (white / `1`) | Keep as clean conditioning; exclude from loss. |
| `<= 0.5` (black / `0`) | Add noise and train the model to generate it. |

### Processed Video Mask Format

If you create masks manually instead of running preprocessing, save them as `.pt` files with the key `"mask"`
containing a tensor of shape `[F, H, W]` where:

- `F` = number of latent frames (temporal dimension)
- `H` = latent height (pixel height / VAE spatial factor, 32 by default)
- `W` = latent width (pixel width / VAE spatial factor, 32 by default)
- Values are thresholded at `0.5`: values `> 0.5` are conditioning tokens (clean, excluded from loss),
  and values `<= 0.5` are generated tokens (noised, contributes to loss).

### Audio Mask Format

Audio masks follow the same thresholding pattern as video masks but with shape `[T]`
(temporal dimension only), where `T` is the number of audio latent frames. They are stored in `audio_masks/`.

### Directory Structure

Save one `.pt` mask per sample under the preprocessed root, matching latent filenames. The conventional directories
are `video_masks/` and `audio_masks/`; see [output structure](#-output-structure).

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

Set a trigger token to prepend to captions during preprocessing:

```yaml
training_strategy:
  data:
    lora_trigger: "MYTRIGGER"
```

> [!NOTE]
> **Trigger words:** Preprocessing inserts the token automatically; don't add it manually to metadata captions.
> Include it in inference prompts. If adding or changing it after preprocessing, rerun with `--overwrite`.

## 🔍 Decoding Videos for Verification

Set `data.decode: true` to turn latents back into viewable media right after encoding. These are reconstructions of
your dataset, useful for checking cropping, frame selection, and audio before training; they are not trained model samples.

```yaml
training_strategy:
  data:
    decode: true
```

```bash
uv run python scripts/preprocess.py configs/t2v_lora.yaml
```

Decoded outputs are written under the preprocessed root:

- `decoded_videos/` — target video latents
- `decoded_reference_videos/` — reference video latents (when present)
- `decoded_audio/` — produced audio latents
- `decoded_reference_audio/` — reference audio latents (when present)

For single-frame images, the decoded latents are saved as PNG files rather than MP4 videos.

## 🚀 Next Steps

Once you've checked the preprocessed outputs:

1. Review the remaining run settings in [Configuration Reference](configuration-reference.md).
2. Launch training and monitor validation samples with the [Training Guide](training-guide.md).

Built-in mask and spatial-crop conditions cover inpainting/outpainting. For other preprocessing algorithms, see
[Custom Training Strategies](custom-training-strategies.md).
