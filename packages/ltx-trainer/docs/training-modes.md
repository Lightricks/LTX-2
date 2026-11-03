# Training Modes Guide

Choose what the model should generate and what inputs it should use. The built-in **flexible** strategy
(`name: "flexible"`) supports the recipes below through YAML configuration. Set `is_generated` for video and audio,
then add conditions such as a starting image, a reference clip, or a mask.

## Key Concepts

The YAML snippets below are **sections of a run config**, not complete configs. Copy the linked example YAML for a
runnable starting point; edit model/data paths and validation samples before preprocessing.

Dataset trees use the linked configs' directory names under `training_strategy.data.preprocessed_data_root`.
Each sample needs matching relative `.pt` filenames in every required folder; see [cached data layout](dataset-preparation.md#-output-structure).

A **modality** is video or audio. **Conditioning** is information provided to guide generation. During training, the
model removes noise from the generated content; the **loss** measures its prediction error. Content described as
**clean** has no added noise.

- **`is_generated: true`** — the modality is denoised during training and contributes to the loss. This is the
  modality the model learns to generate.
- **`is_generated: false`** — the modality is frozen (sigma=0, no noise, no loss). It passes through the transformer
  clean and acts as cross-modal conditioning for the generated modality. Here, “frozen” describes the input, not
  which transformer weights are trainable; [LoRA targets](#-lora-target-modules-guidance) control adapter placement.
- **At least one modality must have `is_generated: true`.** Omit a modality block to exclude it.
- **Conditions** are per-modality and can be composed (e.g., `reference` + `first_frame` together on the video
  modality).
- Intrinsic conditions (`first_frame`, `prefix`, `suffix`, `mask`, `spatial_crop`) keep selected target tokens clean:
  no noise, timestep zero, excluded from loss. `reference` instead appends clean reference tokens to the sequence.
- `probability` is the chance of applying a condition to a sample; unapplied conditions leave ordinary generation.
- Audio supports `prefix`, `suffix`, `mask`, and `reference`; `first_frame` and `spatial_crop` are video-only.

For example, audio-to-video training sets video to `is_generated: true` and audio to `false`: the model learns to
generate video while listening to unchanged source audio.

> [!TIP]
> If you are using an agent-enabled environment with repository skills and are unsure which mode to choose,
> ask for the [`/train`](../../../.claude/skills/train/SKILL.md) skill. It maps your intent to one of
> these configs and walks through dataset preparation, preprocessing, launch, and monitoring.

## 📊 Quick Reference

In mode names, **T = text, I = image, V = video, A = audio**. In the table, **Generated** means the model learns to
produce that modality; **Frozen** means it is provided as input; **—** means it is omitted. Conditions are additional
inputs beyond the text prompt. The T2V and I2V recipes below generate audio as well as video.

| Mode                  | Video     | Audio     | Conditions          | Config |
|-----------------------|-----------|-----------|---------------------|--------|
| **T2V**               | Generated | Generated | —                   | [`t2v_lora`](../configs/t2v_lora.yaml), [low VRAM](../configs/t2v_lora_low_vram.yaml) |
| **I2V**               | Generated | Generated | `first_frame`       | [`i2v_lora`](../configs/i2v_lora.yaml) |
| **Video Extension**   | Generated | Generated | `prefix`/`suffix`   | [`video_extend_lora`](../configs/video_extend_lora.yaml) |
| **V2V IC-LoRA**       | Generated | —         | `reference`         | [`v2v_ic_lora`](../configs/v2v_ic_lora.yaml) |
| **A2V**               | Generated | Frozen    | —                   | [`a2v_lora`](../configs/a2v_lora.yaml) |
| **V2A (Foley)**       | Frozen    | Generated | —                   | [`v2a_lora`](../configs/v2a_lora.yaml) |
| **Video Inpainting**  | Generated | —         | `mask`              | [`video_inpainting_lora`](../configs/video_inpainting_lora.yaml) |
| **Video Outpainting** | Generated | —         | `spatial_crop`      | [`video_outpainting_lora`](../configs/video_outpainting_lora.yaml) |
| **T2A**               | —         | Generated | —                   | [`t2a_lora`](../configs/t2a_lora.yaml) |
| **Audio Extension**   | —         | Generated | `prefix`/`suffix`   | [`audio_extend_lora`](../configs/audio_extend_lora.yaml) |
| **Audio Inpainting**  | —         | Generated | `mask`              | [`audio_inpainting_lora`](../configs/audio_inpainting_lora.yaml) |
| **A2A IC-LoRA**       | —         | Generated | `reference`         | [`a2a_ic_lora`](../configs/a2a_ic_lora.yaml) |
| **AV2AV IC-LoRA**     | Generated | Generated | `reference` (both)  | [`av2av_ic_lora`](../configs/av2av_ic_lora.yaml) |

---

## 🎯 Text-to-Video (T2V)

Generate video and audio from text prompts. Both modalities are denoised with no additional conditions.

```yaml
training_strategy:
  name: "flexible"
  video:
    is_generated: true
    latents_dir: "latents"
  audio:
    is_generated: true
    latents_dir: "audio_latents"
```

**Example config:** 📄 [t2v_lora.yaml](../configs/t2v_lora.yaml)

---

## 🖼️ Image-to-Video (I2V)

Generate video from a starting image. With `probability: 0.5`, about half the samples use first-frame conditioning and
remaining samples train in T2V mode. Single-frame image samples skip this condition so they retain a training loss.

```yaml
training_strategy:
  name: "flexible"
  video:
    is_generated: true
    latents_dir: "latents"
    conditions:
      - type: first_frame
        probability: 0.5
  audio:
    is_generated: true
    latents_dir: "audio_latents"
```

**Example config:** 📄 [i2v_lora.yaml](../configs/i2v_lora.yaml)

---

## ⏩ Video Extension

Extend a video forward (or backward) in time. Prefix or suffix conditioning provides a span of existing latent frames
as clean conditioning. The `temporal_boundary` sets the number of **latent frames** used as context. With the default
VAE temporal factor of 8, N latent frames correspond to `(N - 1) × 8 + 1` pixel frames because the first frame is
preserved by the causal encoder.

```yaml
training_strategy:
  name: "flexible"
  video:
    is_generated: true
    latents_dir: "latents"
    conditions:
      - type: prefix            # or "suffix" for backward extension
        temporal_boundary: 8    # 8 latent frames = 57 pixel frames with the default VAE
        probability: 1.0
  audio:
    is_generated: true
    latents_dir: "audio_latents"
```

> [!NOTE]
> The `prefix` and `suffix` conditions also work on the audio modality for audio extension.
> Set `temporal_boundary` on the audio modality's conditions list to condition on a prefix or suffix
> of the audio latents.

**Example configs:** 📄 [video_extend_lora.yaml](../configs/video_extend_lora.yaml) (forward), 📄 [video_suffix_lora.yaml](../configs/video_suffix_lora.yaml) (backward)

---

## 🔄 IC-LoRA / Video-to-Video (V2V)

In-Context LoRA learns transformations from paired videos: a reference input and the desired target output.
For example, a depth-map video can guide the layout of a generated video. This supports control adapters (depth, pose),
style transfer, deblurring, colorization, and more.

The model sees both videos together. Pre-encoded reference tokens are appended to the target sequence and participate
in bidirectional self-attention, so the model can use them as context. They receive no noise and are excluded from loss.

```yaml
training_strategy:
  name: "flexible"
  video:
    is_generated: true
    latents_dir: "latents"
    conditions:
      - type: reference
        latents_dir: "reference_latents"
        probability: 1.0
      - type: first_frame       # optional — composable with reference
        probability: 0.2
```

> [!NOTE]
> IC-LoRA is video-only by default (no audio modality block). Conditions can be composed — the example above also
> applies first-frame conditioning with 20% probability alongside the reference.
> Use [AV2AV IC-LoRA](#-av2av-ic-lora) when both video and audio references should be trained jointly.

**Example config:** 📄 [v2v_ic_lora.yaml](../configs/v2v_ic_lora.yaml)

### Dataset Requirements

Provide paired target/reference videos through the `video` and `reference_video` metadata columns, and preprocess them
together. **Source references must contain at least the bucket's frame count**, even when temporally subsampled;
shorter sources are skipped. See [reference preprocessing](dataset-preparation.md#-ic-lora-reference-video-preprocessing)
for output layout, aliases, and generating Canny references or other controls.

**Dataset structure:**

```text
preprocessed_data_root/
├── latents/            # Target video latents
├── conditions/         # Text embeddings
└── reference_latents/  # Reference video latents (conditioning input)
```

### Scaled Reference Conditioning

Reduce reference token count by declaring spatial/temporal scale factors under `training_strategy.data`:

```yaml
training_strategy:
  data:
    resolution_buckets:
      - [768, 768, 25]
    reference_spatial_scale_factor: 2
    reference_temporal_scale_factor: 1
```

Use the same factors when training existing precomputed data. Training checks reference/target latent ratios;
validation inherits the factors automatically, so **do not set them on validation reference conditions**.
If either factor exceeds 1, preprocessing requires one resolution bucket. See
[scaling constraints and validation example](dataset-preparation.md#scaled-reference-conditioning) for VAE alignment,
frame-count rules, and single-frame behavior.

---

## 🔊 Audio-to-Video (A2V)

Generate video conditioned on frozen audio. Audio passes through the transformer clean (sigma=0) and influences video
via the built-in cross-modal attention. Only video is denoised.

```yaml
training_strategy:
  name: "flexible"
  video:
    is_generated: true
    latents_dir: "latents"
  audio:
    is_generated: false
    latents_dir: "audio_latents"
```

**Example config:** 📄 [a2v_lora.yaml](../configs/a2v_lora.yaml)

---

## 🎵 Video-to-Audio / Foley (V2A)

Generate audio (Foley) conditioned on frozen video. Video passes through the transformer clean (sigma=0) and
conditions audio via cross-modal attention. Only audio is denoised.

```yaml
training_strategy:
  name: "flexible"
  video:
    is_generated: false
    latents_dir: "latents"
  audio:
    is_generated: true
    latents_dir: "audio_latents"
```

**Example config:** 📄 [v2a_lora.yaml](../configs/v2a_lora.yaml)

---

## 🎭 Video Inpainting

Fill in selected regions of a video using a mask for each sample.

> [!IMPORTANT]
> **Mask convention: white (`1`) keeps; black (`0`) generates.** Values above `0.5` provide clean conditioning with
> timestep zero and no loss. Values at or below `0.5` are noised and contribute to loss. Validation/inference use the same threshold.

```yaml
training_strategy:
  name: "flexible"
  video:
    is_generated: true
    latents_dir: "latents"
    conditions:
      - type: mask
        mask_dir: "video_masks"
        probability: 1.0
```

**Dataset structure:**

```text
preprocessed_data_root/
├── latents/      # Video latents
├── conditions/   # Text embeddings
└── video_masks/  # Per-sample binary masks (1 → conditioning, 0 → generate)
```

In dataset metadata, provide mask media via the `video_mask` column; preprocessing converts it into `video_masks/`.
Preprocessing and validation use the same image-mask resizing and causal temporal grouping. If a mask video is
shorter than the target, missing mask frames are treated as zero (generate) so the latent mask still covers the
full target sequence.

**Example config:** 📄 [video_inpainting_lora.yaml](../configs/video_inpainting_lora.yaml)

---

## 🌅 Video Outpainting

Extend a video spatially beyond its original boundaries. A rectangular pixel region is provided as clean conditioning
(no noise, timestep=0, excluded from loss) — the model learns to generate the surrounding content. The `spatial_region`
is specified in pixel coordinates `[y1, x1, y2, x2]` and automatically converted to latent space.

```yaml
training_strategy:
  name: "flexible"
  video:
    is_generated: true
    latents_dir: "latents"
    conditions:
      - type: spatial_crop
        spatial_region: [0, 0, 288, 576]  # y1, x1, y2, x2 in pixels
        probability: 1.0
```

> [!NOTE]
> `spatial_crop` is a video-only condition — it is not supported on the audio modality.

**Example config:** 📄 [video_outpainting_lora.yaml](../configs/video_outpainting_lora.yaml)

---

## 🔈 Text-to-Audio (T2A)

Generate audio from text without a video modality. Use audio-only LoRA targets (`audio_attn1`, `audio_attn2`,
and optionally `audio_ff`); see [target-module guidance](#-lora-target-modules-guidance).

```yaml
training_strategy:
  name: "flexible"
  audio:
    is_generated: true
    latents_dir: "audio_latents"
```

**Example config:** 📄 [t2a_lora.yaml](../configs/t2a_lora.yaml)

---

## 🔊 Audio Extension

Extend audio forward (prefix) or backward (suffix) in time — the audio equivalent of Video Extension. A span of
existing audio latent frames is provided as clean conditioning, and the model generates the continuation. The
`temporal_boundary` sets the number of latent frames used as context. This mode uses **audio-only LoRA targets**.

```yaml
training_strategy:
  name: "flexible"
  audio:
    is_generated: true
    latents_dir: "audio_latents"
    conditions:
      - type: prefix            # or "suffix" for backward extension
        temporal_boundary: 8
        probability: 1.0
```

**Example configs:** 📄 [audio_extend_lora.yaml](../configs/audio_extend_lora.yaml), 📄 [audio_suffix_lora.yaml](../configs/audio_suffix_lora.yaml)

---

## 🎭 Audio Inpainting

Fill in masked regions of audio. Per-sample masks loaded from disk define which audio tokens are conditioning and
which must be generated — the audio equivalent of Video Inpainting. Masks are thresholded at `0.5` with the same
binary semantics as video inpainting. This mode uses **audio-only LoRA targets**.

```yaml
training_strategy:
  name: "flexible"
  audio:
    is_generated: true
    latents_dir: "audio_latents"
    conditions:
      - type: mask
        mask_dir: "audio_masks"
        probability: 1.0
```

**Dataset structure:**

```text
preprocessed_data_root/
├── conditions/     # Text embeddings
├── audio_latents/  # Audio latents
└── audio_masks/    # Per-sample binary masks (1 → conditioning, 0 → generate)
```

In dataset metadata, provide mask media via the `audio_mask` column; preprocessing converts it into `audio_masks/`.

**Example config:** 📄 [audio_inpainting_lora.yaml](../configs/audio_inpainting_lora.yaml)

---

## 🔄 IC-LoRA / Audio-to-Audio (A2A)

In-Context LoRA for audio-to-audio transformations. Pre-encoded reference audio latents are concatenated to the target
sequence — reference tokens participate in bidirectional self-attention but receive no noise and are excluded from loss.
This enables audio style transfer, voice conversion, sound effect transformation, and more. This mode uses
**audio-only LoRA targets**.

```yaml
training_strategy:
  name: "flexible"
  audio:
    is_generated: true
    latents_dir: "audio_latents"
    conditions:
      - type: reference
        latents_dir: "reference_audio_latents"
        probability: 1.0
```

Provide `audio` (or extract target audio from `video`) and `reference_audio` metadata columns.
See [audio/reference column handling](dataset-preparation.md#convention-based-column-detection).

**Dataset structure:**

```text
preprocessed_data_root/
├── conditions/               # Text embeddings
├── audio_latents/            # Target audio latents
└── reference_audio_latents/  # Reference audio latents (conditioning input)
```

**Example config:** 📄 [a2a_ic_lora.yaml](../configs/a2a_ic_lora.yaml)

---

## 🔄 AV2AV IC-LoRA

Joint audio-video In-Context LoRA — both modalities have reference conditioning. Pre-encoded reference latents are
concatenated to each modality's target sequence independently. This enables joint audiovisual transformations such as
synchronized style transfer across both video and audio.

```yaml
training_strategy:
  name: "flexible"
  video:
    is_generated: true
    latents_dir: "latents"
    conditions:
      - type: reference
        latents_dir: "reference_latents"
        probability: 1.0
  audio:
    is_generated: true
    latents_dir: "audio_latents"
    conditions:
      - type: reference
        latents_dir: "reference_audio_latents"
        probability: 1.0
```

> [!NOTE]
> Unlike audio-only IC-LoRA (A2A), AV2AV uses short LoRA target patterns like `"to_k"` to match all branches
> (video, audio, and cross-modal attention), since both modalities are trained.

Provide paired `reference_video` and `reference_audio` metadata alongside the targets. See
[preprocessed output layout](dataset-preparation.md#-output-structure) for the conventional directories.

**Dataset structure:**

```text
preprocessed_data_root/
├── latents/                  # Target video latents
├── audio_latents/            # Target audio latents
├── conditions/               # Text embeddings
├── reference_latents/        # Reference video latents (conditioning input)
└── reference_audio_latents/  # Reference audio latents (conditioning input)
```

**Example config:** 📄 [av2av_ic_lora.yaml](../configs/av2av_ic_lora.yaml)

---

## 🔥 Full Transformer Fine-tuning

All modes above default to `training_mode: "lora"`. For full fine-tuning, set `training_mode: "full"` — this updates
all transformer parameters rather than adding LoRA adapters. The text encoder and VAEs remain frozen.

Set this in any of the mode configs above, retaining its modality/condition settings:

```yaml
training_strategy:
  training_mode: "full"
```

> [!IMPORTANT]
> Full fine-tuning of these model sizes typically needs multiple high-end GPUs (e.g., 4-8× H100 80GB).
> Use distributed training with FSDP to shard the model; see [Training Guide](training-guide.md) for setup instructions.

---

## 🎛️ LoRA Target Modules Guidance

`training_strategy.lora.target_modules` selects the linear layers that receive LoRA adapters. Start with the selected
example config, then use the recipes below to change which projections, branches, or transformer blocks are adapted.
All YAML snippets are fragments to merge into your run config.

### How Target Strings Match

The trainer passes a **list of strings** to PEFT. Each entry matches a complete module name or a **dot-separated
suffix** of that name. Entries are combined: a module receives an adapter if **any** entry matches.

| Target string | What it selects |
|---------------|-----------------|
| `"to_q"` | Query projections in every attention branch, across all transformer blocks. |
| `"attn1.to_q"` | Video self-attention query projections across all blocks; does not match `audio_attn1.to_q`. |
| `"attn2.to_v"` | Video attention-to-text value projections across all blocks. |
| `"audio_attn2.to_v"` | Audio attention-to-text value projections across all blocks. |
| `"audio_to_video_attn.to_k"` | Keys from audio in video-attends-to-audio attention, across all blocks. |
| `"ff.net.0.proj"` | Video feed-forward input projections across all blocks; does not match `audio_ff.net.0.proj`. |
| `"transformer_blocks.0.attn1.to_q"` | Only the video self-attention query projection in block 0. |

`to_q`, `to_k`, and `to_v` are the query, key, and value projections; `to_out.0` is the attention output projection.
Blocks use **zero-based indices**: block `0` is the first. Available block indices depend on the checkpoint.
Use paths from the unwrapped transformer; omit checkpoint or PEFT prefixes such as `diffusion_model.` and
`base_model.model.` from YAML targets.

Checkpoints with gated attention also have `to_gate_logits` projections. For example, `"attn1.to_gate_logits"`
selects video self-attention gates across all blocks. These layers are absent on checkpoints without gated attention.

> [!IMPORTANT]
> **Target strings are literal names or suffixes.** Entries such as `"attn1.*"` or `"transformer_blocks.[0-3].attn1.to_q"`
> are **not** expanded as globs or regular expressions. Target leaf projections (`"attn1.to_q"`), rather than containers
> such as `"attn1"` or `"transformer_blocks.0"`.

### All Audio-Video Attention

For T2V, I2V, A2V, V2A, or AV2AV, the supplied configs use these short suffixes to cover the four standard attention
projections in video, audio, and both cross-modal directions in every block:

```yaml
training_strategy:
  lora:
    target_modules: ["to_k", "to_q", "to_v", "to_out.0"]
```

This does not include feed-forward layers or optional attention gates.

> [!NOTE]
> **Choose adapter targets separately from input conditioning.** `is_generated: false` keeps an input clean;
> `target_modules` controls LoRA placement. Broad targets can attach adapters to branches your forward pass does not use.

### Video Attention and Feed-Forward Layers

For video-only adapters, such as V2V IC-LoRA, name the video branches explicitly. `attn1` is video self-attention,
and `attn2` attends to text. The two `ff` entries add the video feed-forward projections; remove them for attention only.

```yaml
training_strategy:
  lora:
    target_modules:
      - "attn1.to_k"
      - "attn1.to_q"
      - "attn1.to_v"
      - "attn1.to_out.0"
      - "attn2.to_k"
      - "attn2.to_q"
      - "attn2.to_v"
      - "attn2.to_out.0"
      - "ff.net.0.proj"  # Feed-forward input projection
      - "ff.net.2"       # Feed-forward output projection
```

For a narrower experiment, `target_modules: ["attn1.to_q", "attn1.to_v"]` adapts only video self-attention queries
and values in every block. It leaves keys, output projections, text attention, and feed-forward layers unchanged.

### Audio Attention and Feed-Forward Layers

For T2A, audio extension/inpainting, or A2A IC-LoRA, use the audio branches. `audio_attn1` is audio self-attention,
and `audio_attn2` attends to text. Remove the two `audio_ff` entries for attention only.

```yaml
training_strategy:
  lora:
    target_modules:
      - "audio_attn1.to_k"
      - "audio_attn1.to_q"
      - "audio_attn1.to_v"
      - "audio_attn1.to_out.0"
      - "audio_attn2.to_k"
      - "audio_attn2.to_q"
      - "audio_attn2.to_v"
      - "audio_attn2.to_out.0"
      - "audio_ff.net.0.proj"
      - "audio_ff.net.2"
```

### Cross-Modal Attention Only

To experiment with adapting only the interaction between video and audio, select these projections:

```yaml
training_strategy:
  lora:
    target_modules:
      - "audio_to_video_attn.to_k"
      - "audio_to_video_attn.to_q"
      - "audio_to_video_attn.to_v"
      - "audio_to_video_attn.to_out.0"
      - "video_to_audio_attn.to_k"
      - "video_to_audio_attn.to_q"
      - "video_to_audio_attn.to_v"
      - "video_to_audio_attn.to_out.0"
```

- `audio_to_video_attn`: **video attends to audio** — queries come from video, keys/values from audio.
- `video_to_audio_attn`: **audio attends to video** — queries come from audio, keys/values from video.

Keep only one group of four to adapt one direction. Both modalities must participate in the forward pass for these
layers to be used. For A2V/V2A, start with the broader supplied targets; cross-modal-only adaptation is a narrower experiment.

### Specific Transformer Blocks

Include `transformer_blocks.<index>.` to restrict placement to particular blocks. This example adapts **video
self-attention only in blocks 0 and 1**, leaving the other branches and blocks without adapters:

```yaml
training_strategy:
  lora:
    target_modules:
      - "transformer_blocks.0.attn1.to_k"
      - "transformer_blocks.0.attn1.to_q"
      - "transformer_blocks.0.attn1.to_v"
      - "transformer_blocks.0.attn1.to_out.0"
      - "transformer_blocks.1.attn1.to_k"
      - "transformer_blocks.1.attn1.to_q"
      - "transformer_blocks.1.attn1.to_v"
      - "transformer_blocks.1.attn1.to_out.0"
```

The same prefix works for other layers: `"transformer_blocks.1.audio_attn2.to_v"` selects one audio-to-text value
projection, `"transformer_blocks.1.audio_to_video_attn.to_q"` selects one cross-modal query projection, and
`"transformer_blocks.1.ff.net.2"` selects one video feed-forward output projection.

> [!WARNING]
> **Replace the existing target list when restricting blocks.** Keeping a broad entry such as `"to_q"` alongside a
> block-specific entry still adapts query projections across all blocks. Do not add PEFT's `layers_to_transform` field
> to the trainer YAML; it is not exposed by the trainer schema.

### Check Your Selection

A typo or nonexistent block can leave an entry unmatched. PEFT raises if no targets match, but **a successful setup
does not prove that every entry matched**. For custom selections, inspect the adapter keys after a short run:

```bash
uv run python - <<'PY'
from safetensors import safe_open

from ltx_trainer import logger

with safe_open("/path/to/checkpoints/step_00100/weights.safetensors", framework="pt", device="cpu") as checkpoint:
    for key in checkpoint.keys():
        if key.endswith(".lora_A.weight"):
            logger.info(key.removeprefix("diffusion_model.").removesuffix(".lora_A.weight"))
PY
```

This reads the LoRA checkpoint's tensor names without loading the base model. Expect block-specific names such as
`transformer_blocks.0.attn1.to_q`; check that the blocks and branches match your intent. See
[the module reference](configuration-reference.md#understanding-target-modules) for the complete branch map.

---

## 🎬 Using Trained Models for Inference

The separate [`ltx-pipelines`](../../ltx-pipelines/) package can load trained LoRAs. Choose an entry point that
supports your conditioning inputs and base checkpoint:

- **Text/image-to-video:** [`TI2VidOneStagePipeline`](../../ltx-pipelines/src/ltx_pipelines/ti2vid_one_stage.py) or
  [`TI2VidTwoStagesPipeline`](../../ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py) accepts text and image conditioning.
- **Audio-to-video:** [`A2VidPipelineTwoStage`](../../ltx-pipelines/src/ltx_pipelines/a2vid_two_stage.py) accepts source audio and optional images.
- **Text-to-audio:** [`T2AOneStagePipeline`](../../ltx-pipelines/src/ltx_pipelines/t2a_one_stage.py) generates audio without the video branch.
- **Video-reference IC-LoRA:** [`ICLoraPipeline`](../../ltx-pipelines/src/ltx_pipelines/ic_lora.py) accepts reference video conditioning on a distilled checkpoint.
- **Temporal editing or video-to-audio:** [`RetakePipeline`](../../ltx-pipelines/src/ltx_pipelines/retake.py) regenerates a time window. Its Python interface controls
  video/audio separately; for V2A, set `regenerate_video=False, regenerate_audio=True` to preserve video and regenerate audio.

> [!WARNING]
> **Check inference support before training.** Audio-reference IC-LoRA (A2A/AV2AV), arbitrary masks, outpainting, and
> prefix/suffix conditioning may require custom inference wiring. Match the training base checkpoint and reference
> scaling; a pipeline name alone does not make an adapter compatible with a different base model.

The trainer's [validation conditions](configuration-reference.md#validation-condition-types) can evaluate these inputs during training.
Check the [pipeline reference](../../ltx-pipelines/docs/pipelines.md) for constructor/CLI arguments and additional
weights such as distilled LoRAs and upsamplers.

> [!NOTE]
> You can generate audio during validation even if you're not training the audio branch.
> Set `training_strategy.validation.generate_audio: true` independently of whether audio has `is_generated: true`.

---

## 🔄 Legacy Strategies

New configs should use `name: "flexible"` and express the training mode through modality and condition configuration.
Legacy `text_to_video` and `video_to_video` configs are migrated automatically when loaded.

---

## 🚀 Next Steps

Once you've chosen a training mode:

1. Prepare its metadata and cached artifacts using [Dataset Preparation](dataset-preparation.md).
2. Set the remaining run fields with [Configuration Reference](configuration-reference.md).
3. Launch and monitor training with the [Training Guide](training-guide.md).

For a new loss, noising rule, or multi-model algorithm, see [Custom Training Strategies](custom-training-strategies.md);
check whether existing conditions can express the task first.
