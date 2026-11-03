# Utility Scripts Reference

Commands assume `packages/ltx-trainer` as the working directory. Use each script's `--help` for all options.
For the complete encoding workflow and metadata schema, see [Dataset Preparation](dataset-preparation.md).

## Video Scene Splitting

`scripts/split_scenes.py` uses PySceneDetect to split videos into coherent clips. Available detectors include content,
adaptive, threshold, and histogram detection.

Split a video and discard clips shorter than five seconds:

```bash
uv run python scripts/split_scenes.py input.mp4 output_dir/ --filter-shorter-than 5s
```

Use adaptive detection with a custom threshold:

```bash
uv run python scripts/split_scenes.py video.mp4 scenes/ --detector adaptive --threshold 3.0
```

Limit the output to 50 scenes:

```bash
uv run python scripts/split_scenes.py video.mp4 scenes/ --max-scenes 50
```

List all detector and filtering options:

```bash
uv run python scripts/split_scenes.py --help
```

## Automatic Video Captioning

`scripts/caption_videos.py` generates one detailed paragraph per video describing its visual content and audio.
Two backends are available:

- **`qwen_omni` (default)** — Qwen3-Omni-30B-A3B-Thinking served via a local
  [vLLM](https://docs.vllm.ai/) HTTP server. Runs offline once the model is downloaded.
- **`gemini_flash`** — Google Gemini (cloud, `gemini-3.5-flash`). No local GPU or captioner server required.

### Local Qwen Captioning

`scripts/serve_captioner.py` runs vLLM in a separate `uvx` environment to isolate its CUDA dependencies from the
trainer. The default FP8 quantization uses about 31 GiB for weights and fits on a 40 GB GPU.

1. Start the captioner server and leave it running:

   ```bash
   uv run python scripts/serve_captioner.py
   ```

2. Run captioning in another terminal:

   ```bash
   uv run python scripts/caption_videos.py videos_dir/ --output dataset.json
   ```

   The client connects to `http://127.0.0.1:8001/v1` by default.

#### Server Options

- Preview the vLLM command without starting the server:

  ```bash
  uv run python scripts/serve_captioner.py --print-cmd
  ```

- Use BF16 instead of FP8 if you have about 66 GiB of free VRAM:

  ```bash
  uv run python scripts/serve_captioner.py --quantization bf16
  ```

- Choose a download location with room for the roughly 65 GB model:

  ```bash
  uv run python scripts/serve_captioner.py --hf-home /mnt/disk
  ```

You can combine these server flags as needed. To connect to a reachable server on another machine, supply its URL
when running the client:

```bash
uv run python scripts/caption_videos.py videos_dir/ \
  --output dataset.json \
  --vllm-url http://other-host:8001/v1
```

#### Thinking Mode

Qwen3-Omni can emit a `<think>...</think>` reasoning trace before the caption. It is off by default;
leave it off for faster bulk captioning. To enable it:

```bash
uv run python scripts/caption_videos.py videos_dir/ --output dataset.json --enable-thinking
```

### Cloud Gemini Captioning

1. Choose an authentication method. For the Developer API, set an API key (`GOOGLE_API_KEY` also works):

   ```bash
   export GEMINI_API_KEY="<your-api-key>"
   ```

   Alternatively, use Google Cloud Application Default Credentials for Vertex AI, through an attached service account
   or an interactive login:

   ```bash
   gcloud auth application-default login
   ```

   The script also accepts a key through `--api-key`. It uses Vertex AI credentials when no API key is supplied.

2. Run captioning directly; no local captioner server is needed:

   ```bash
   uv run python scripts/caption_videos.py videos_dir/ \
     --output dataset.json \
     --captioner-type gemini_flash \
     --num-workers 5
   ```

Keep `--num-workers` at 3–5 for Gemini; higher values may hit API rate limits.

### Output and Reruns

Captioning is incremental (already-captioned files are skipped, progress saves every 5 videos)
and writes JSON, JSONL, CSV, or TXT based on the output extension.
To replace existing captions, add `--override` to your command, for example:

```bash
uv run python scripts/caption_videos.py videos_dir/ --output dataset.json --override
```

## Reference Video Generation

`scripts/compute_reference.py` generates Canny edge reference videos and writes their paths to the `reference_video`
metadata column:

```bash
uv run python scripts/compute_reference.py videos_dir/ --output dataset.json
```

> [!WARNING]
> **This edits existing dataset JSON in place.** Copy important metadata before running the script.

Customize `compute_reference()` in [the script](../scripts/compute_reference.py) to generate depth maps, poses,
segmentation masks, or other conditions.
See [reference preprocessing](dataset-preparation.md#-ic-lora-reference-video-preprocessing) to encode paired media.

## Dataset Preprocessing

`scripts/preprocess.py` encodes media and captions using the training YAML. Use your run config in place of
`configs/t2v_lora.yaml` in these examples.

### Encode and Check

Encode the dataset using your training config:

```bash
uv run python scripts/preprocess.py configs/t2v_lora.yaml
```

To check an existing cache, compare the config with the saved preprocessing manifest without encoding or loading models:

```bash
uv run python scripts/preprocess.py configs/t2v_lora.yaml --check
```

This requires a config valid for training and an existing compatible manifest. For the built-in flexible strategy,
set `training_strategy.data.preprocessed_data_root` to the cache location. `--check` cannot be combined with `--overwrite`.

### Rebuild Cached Data

Recompute the current dataset after changing encoding settings or source contents:

```bash
uv run python scripts/preprocess.py configs/t2v_lora.yaml --overwrite
```

> [!IMPORTANT]
> **Rebuild after changing encoding settings or source contents.** `--overwrite` recomputes current dataset items but
> leaves old `.pt` files for removed samples. Use a fresh output root for a clean rebuild.

### Select an Encoding Device

The default device is `cuda`. For example, select the second GPU with `cuda:1`:

```bash
uv run python scripts/preprocess.py configs/t2v_lora.yaml --device cuda:1
```

For the flexible strategy, media settings live under `training_strategy.data`. See
[manifest and rerun semantics](dataset-preparation.md#preprocessing-manifest-and-staleness-guard),
[multi-GPU preprocessing](dataset-preparation.md#-multi-gpu-preprocessing), and
[decode verification](dataset-preparation.md#-decoding-videos-for-verification).

> [!TIP]
> **Check a few samples before processing the full dataset.** Set `training_strategy.data.decode: true` and run
> preprocessing to inspect decoded video/audio and references. Review captions too; see
> [decode outputs](dataset-preparation.md#-decoding-videos-for-verification) for directory names and image handling.

## Training Scripts

Start a single-GPU run with your training config:

```bash
uv run python scripts/train.py configs/t2v_lora.yaml
```

For a two-GPU DDP run, launch through Accelerate with the supplied profile:

```bash
uv run accelerate launch --config_file configs/accelerate/ddp.yaml --num_processes 2 \
  scripts/train.py configs/t2v_lora.yaml
```

See the [Training Guide](training-guide.md) for GPU selection, other profiles, output files, and resume behavior.

## Inference with Trained Models

Use the separate [ltx-pipelines](../../ltx-pipelines/README.md) package to load trained LoRAs. See
[inference entry points and limitations](training-modes.md#-using-trained-models-for-inference) to choose a pipeline
that supports your checkpoint and conditioning inputs.
