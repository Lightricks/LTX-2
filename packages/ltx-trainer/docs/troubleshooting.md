# Troubleshooting Guide

Expand the issue that matches your symptoms, apply the relevant change, and retry on a small subset.
Commands assume `packages/ltx-trainer` as the working directory.

## ⚠️ Common Usage Issues

<details>
<summary>Out of memory during preprocessing, training, or validation</summary>

> 💡 **Low-VRAM starting point**
> For GPUs with 32GB VRAM, use the pre-configured low VRAM config:
> [`configs/t2v_lora_low_vram.yaml`](../configs/t2v_lora_low_vram.yaml)
> which combines 8-bit optimizer, INT8 quantization, and reduced LoRA rank.

### Memory Optimization Techniques

First check **where** you run out of memory; different stages need different fixes:

- **Encoding captions:** try the 8-bit text encoder below.
- **Encoding video:** reduce resolution/frame count or enable `training_strategy.data.vae_tiling`.
- **Training updates:** try gradient checkpointing, a smaller batch, quantization, or the 8-bit optimizer.
- **Validation sampling:** reduce `training_strategy.validation.video_dims` or try optimizer-state offloading below.

Apply the relevant options rather than treating the list below as a required sequence.

#### 1. Enable Gradient Checkpointing

Gradient checkpointing saves memory by recomputing intermediate activations during backward rather than keeping them
all in memory. It makes training slower but is recommended for most runs. This is separate from saving model
checkpoints to disk.

```yaml
training_strategy:
  optimizer:
    gradient_checkpointing: true
```

#### 2. Enable 8-bit Text Encoder

If preprocessing or validation prompt caching runs out of memory, load Gemma in 8-bit. This affects text-encoding
phases; it does not reduce the transformer's per-step training memory:

```yaml
training_strategy:
  model:
    load_text_encoder_in_8bit: true
```

#### 3. Reduce Batch Size

Reduce per-GPU batch size and use gradient accumulation to retain a larger effective batch:

```yaml
optimization:
  batch_size: 1
  gradient_accumulation_steps: 4  # Effective global batch = 4 × num_processes
```

#### 4. Use Lower Resolution

Reduce bucket width, height, or frame count and rebuild the cached artifacts. For example, use one smaller bucket:

```yaml
training_strategy:
  data:
    resolution_buckets:
      - [512, 512, 49]
```

```bash
uv run python scripts/preprocess.py configs/t2v_lora.yaml --overwrite
```

Alternatively, reduce duration, e.g. `[960, 544, 25]`. `--overwrite` recomputes current items but leaves files for removed samples;
use a fresh `preprocessed_data_root` for a clean rebuild. See
[manifest behavior](dataset-preparation.md#preprocessing-manifest-and-staleness-guard).

#### 5. Enable Model Quantization

Quantization stores the base transformer's weights at lower precision to save memory. This is supported in LoRA mode only:

```yaml
training_strategy:
  quantization: "int8-quanto"
```

Other choices are `int4-quanto`, `int2-quanto`, `fp8-quanto`, and `fp8uz-quanto`.

#### 6. Use 8-bit Optimizer

AdamW keeps running statistics for trainable parameters. Its 8-bit variant stores these more compactly to save memory:

```yaml
training_strategy:
  optimizer:
    optimizer_type: "adamw8bit"
```

#### 7. Offload Optimizer State During Validation

If you run out of memory specifically during validation video sampling, move the optimizer's running statistics
to CPU while sampling. This can help full fine-tunes or high-rank LoRA runs where those statistics and the VAE decoder
do not fit on the GPU together:

```yaml
optimization:
  offload_optimizer_during_validation: true
```

The trainer moves the state back to GPU after sampling. This happens once per validation interval, not per step,
and has no effect for FSDP, which uses sharded state.

</details>

<details>
<summary>"No module named 'ltx_trainer'" Error</summary>

**Solution:**
Ensure you've installed the dependencies and are using `uv run` to execute scripts:

```bash
# From the repository root
uv sync
cd packages/ltx-trainer
uv run python scripts/train.py configs/t2v_lora.yaml
```

`uv run` selects the workspace virtual environment without manual activation.

</details>

<details>
<summary>The text encoder path format does not match the checkpoint</summary>

**Solution:** Set `text_encoder_path` to the matching local text encoder. Unified LTX-2/LTX-2.3 checkpoints use a
Gemma model directory; an LTX-2.5 split pack uses the packed LTX-specific text-encoder safetensors file. For example:

```yaml
training_strategy:
  model:
    model_path: "/path/to/ltx-2-model.safetensors"  # Unified checkpoint
    text_encoder_path: "/path/to/gemma-model/"      # Gemma model directory
```

For an LTX-2.5 split pack, use its transformer and packed text-encoder files:

```yaml
training_strategy:
  model:
    model_path: "/path/to/ltx-2.5-transformer.safetensors"
    text_encoder_path: "/path/to/ltx-2.5-text-encoder.safetensors"
    video_vae_path: "/path/to/video-vae.safetensors"
    audio_vae_path: "/path/to/audio-vae.safetensors"
```

Set VAE paths for the modalities used, and use the LTX-specific text encoder. See
[model configuration](configuration-reference.md#model).

</details>

<details>
<summary>"Model path does not exist" Error</summary>

Use an existing local `.safetensors` path for `training_strategy.model.model_path`. URLs are unsupported.
For split packs, also check the matching text-encoder and VAE component paths; see
[model configuration](configuration-reference.md#model).

</details>

<details>
<summary>"frames must satisfy (frames - 1) % T == 0" Error</summary>

**Solution:**
The number of frames must be VAE-aligned: `(frames - 1) % T == 0`, where `T` is the VAE temporal
compression factor read from the checkpoint. For the **default** VAE, `T = 8`:

- ✅ Valid: 1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 121
- ❌ Invalid: 24, 32, 48, 64, 100

A less-compressed VAE lowers `T` (e.g. a 16x16x4 VAE uses `T = 4`, allowing 1, 5, 9, 13, ...).
The spatial dimensions must likewise be divisible by the VAE spatial factor (32 by default).

</details>

<details>
<summary>Preprocessed Data Does Not Match the Current Configuration</summary>

**Cause:** `preprocess_manifest.json` records the encode-affecting settings from the last successful preprocessing
run. The current config differs in a field such as `lora_trigger`, resolution buckets, model paths, reference scale
factors, audio settings, or a modality/reference/mask directory.

**Solution:** Align the config with the saved fingerprint, use a fresh output root, or recompute current items:

```bash
uv run python scripts/preprocess.py configs/your_config.yaml --overwrite
```

If the mismatch is intentional, set `allow_stale_preprocessed: true`; training continues with a
warning. The check runs before model loading.

To diagnose compatibility without starting training or encoding, run:

```bash
uv run python scripts/preprocess.py configs/your_config.yaml --check
```

`--check` reports failure if the manifest is missing. Training allows older caches without a manifest and warns,
but that warning does not mean the config matches the cache.

</details>

<details>
<summary>Preprocessing Fails Before Encoding with a Manifest Error</summary>

An incremental run is rejected when the output root is populated but its manifest is missing or differs from the
current fingerprint. Existing files would otherwise be skipped and mixed with newly encoded artifacts. Use a new
`preprocessed_data_root`, restore the matching config, or re-run with `--overwrite`.

Corrupt JSON or an unsupported manifest `format_version` also stops preprocessing; use a fresh
`preprocessed_data_root`, restore a valid matching manifest, or rebuild with `--overwrite`. See
[Preprocessing Manifest and Staleness Guard](dataset-preparation.md#preprocessing-manifest-and-staleness-guard) for
details on rebuilding and removing old cached files.

</details>

<details>
<summary>Data loading fails or no cached samples are found</summary>

Errors such as **"No data files found"**, **"Required … directory does not exist"**, or **"No valid samples found"**
usually point to the cache location or missing artifacts. For the built-in flexible strategy:

- Set `training_strategy.data.preprocessed_data_root` to the folder containing the configured source directories,
  such as `latents/` and `conditions/`, rather than to one of those directories.
- Check that all sources required by your mode exist, including any audio, reference, or mask directories.
  Samples need matching cached files across sources; preserve their relative paths when moving data.
- If the error is hidden behind a dataloader worker traceback, set the worker count to zero and retry to see the
  underlying error in the training process:

```yaml
training_strategy:
  data:
    preprocessed_data_root: "/path/to/dataset/.precomputed"
    num_dataloader_workers: 0
```

When the selected root contains a `.precomputed/` subdirectory, the dataset loader uses that subdirectory.
Check the [dataset layout for your mode](training-modes.md) and the
[preprocessing workflow](dataset-preparation.md) before regenerating data.

</details>

<details>
<summary>A checkpoint loads, but training starts again at step 0</summary>

**Loading weights and resuming training are different operations.** Point `checkpoints.load_checkpoint` at the
checkpoint directory to restore the step counter, and keep `checkpoints.no_resume: false`:

```yaml
checkpoints:
  load_checkpoint: "outputs/my_run/checkpoints/step_01000"
  no_resume: false
optimization:
  steps: 1500  # Total target: continue from step 1,000 to 1,500
```

- Pointing directly at `weights.safetensors`, or setting `no_resume: true`, loads weights and starts a new run at step 0.
- A resumable checkpoint needs `state.pt` and `metadata.json` beside `weights.safetensors`. Saves made with
  `save_training_state: "off"` contain weights only.
- Missing or incompatible resume state/metadata can cause a fallback to step 0. Read the preceding warning for the
  reason, and confirm **"Resuming training from step …"** in the logs before continuing a long run.

If you instead see **"No remaining training steps"**, increase `optimization.steps` beyond the restored step;
the value is the total target, not the number of additional updates. See
[resume behavior and limitations](training-guide.md#resume-training), including which optimizer/RNG state is restored.

</details>

<details>
<summary>Slow Training Speed</summary>

**Optimizations:**

1. **Disable gradient checkpointing** (if you have enough VRAM):

   ```yaml
   training_strategy:
     optimizer:
       gradient_checkpointing: false
   ```

2. **Use torch.compile** via Accelerate:

   ```bash
   uv run accelerate launch --config_file configs/accelerate/ddp_compile.yaml \
     scripts/train.py configs/t2v_lora.yaml
   ```

</details>

<details>
<summary>Poor Quality Validation Outputs</summary>

**Solutions:**

1. **Match validation to your task:** For image-to-video training, use a starting image to test the behavior you intend
   to use. For text-to-video training, keep text-only samples so you can evaluate generation without image guidance.
   An image-conditioned sample looks like this:

   ```yaml
   training_strategy:
     validation:
       samples:
         - prompt: "a professional portrait video of a person"
           conditions:
             - type: first_frame
               image_or_video: "/path/to/first_frame.png"
   ```

2. **Increase inference steps:**

   ```yaml
   training_strategy:
     validation:
       inference_steps: 40  # Default is 30
   ```

3. **Restore the default guidance settings if you changed them:**

   ```yaml
   training_strategy:
     validation:
       video_cfg_scale: 3.0
       audio_cfg_scale: 7.0
       video_stg_scale: 1.0
       audio_stg_scale: 1.0
       stg_blocks: [28]
       guidance_rescale: 0.7
   ```

4. **Check caption quality:**
   Review auto-generated captions for accurate visual/audio descriptions (ambient sounds, speech, music).
   After editing captions, rerun preprocessing with `--overwrite` to refresh text embeddings.

5. **Check target modules:**
   Ensure your `target_modules` configuration matches your training goals. For audio-video training,
   use patterns that match both branches (e.g., `"to_k"` instead of `"attn1.to_k"`).
   See [Understanding Target Modules](configuration-reference.md#understanding-target-modules) for details.

6. **Check adapter capacity:**
   If a lower-rank adapter underfits, try rank 64 or 128 and compare the same validation prompts:

   ```yaml
   training_strategy:
     lora:
       rank: 64  # Or 128 for more capacity
   ```

7. **Check training duration:** If validation is still improving, try a larger step budget:

   ```yaml
   optimization:
     steps: 3000
   ```

</details>

---

## 🔍 Debugging Tools

<details>
<summary>Monitor GPU Memory Usage</summary>

Track memory usage during training:

```bash
# Watch GPU memory in real-time
watch -n 1 nvidia-smi

# Log memory usage to file
nvidia-smi --query-gpu=memory.used,memory.total --format=csv --loop=5 > memory_log.csv
```

</details>

<details>
<summary>Verify Preprocessed Data</summary>

To sanity-check preprocessing, set `training_strategy.data.decode: true` in your config and run the preprocessor.
It VAE-decodes the latents right after encoding so you can compare them against the originals:

```yaml
training_strategy:
  data:
    decode: true
```

```bash
uv run python scripts/preprocess.py configs/t2v_lora.yaml
```

Compare decoded media with the originals. See
[decode output directories](dataset-preparation.md#-decoding-videos-for-verification) for video, audio, reference,
and single-frame image outputs.

</details>

## 🆘 Getting Help

Check the full console error and search existing GitHub issues. When reporting a problem, include the GPU model/VRAM,
resolved training config, complete error/traceback, and reproduction steps. Preserve the run config and relevant logs
so another user or agent can reproduce the failure. For discussion, use the
[community Discord](https://discord.gg/ltxplatform).
