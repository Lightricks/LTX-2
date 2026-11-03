# Available Pipelines

Full reference for each pipeline. See the [Pipeline Selection Guide](pipeline-selection.md) to pick one.

---

## 1. TI2VidTwoStagesPipeline

**Best for:** Guided two-stage text/image-to-video and keyframe interpolation with CFG/STG and 2x upsampling.

**Source**: [`src/ltx_pipelines/ti2vid_two_stages.py`](../src/ltx_pipelines/ti2vid_two_stages.py)

Two-stage generation: Stage 1 generates low-resolution video with [multimodal guidance](multimodal-guidance.md),
Stage 2 upsamples to 2x resolution with distilled LoRA refinement. Supports image conditioning, including
[keyframe interpolation](#6-keyframe-interpolation) through explicit keyframe-guide bindings. Slower than
DistilledPipeline; for production quality use [`DFRPipeline`](#12-dfrpipeline).

**Extra CLI arguments:** `--chunk-pixel-frames` (default 97) and/or `--chunk-carry-frames` (default 25) enable
long-video windowing; `--chunk-blend-frames` controls the decoded seam crossfade (default: full carry, `0` for a hard cut).
`--refine-audio` denoises audio again in stage 2 instead of preserving stage-1 audio. These flags are also available on
`ltx_pipelines.ti2vid_two_stages_mgpu`.

**Use when:** You want CFG/STG-guided two-stage text/image-to-video or keyframe interpolation, not the DFR detailing
path.

---

## 2. TI2VidTwoStagesHQPipeline

**Best for:** Same two-stage text/image-to-video as TI2VidTwoStagesPipeline but with a different sampler and step count.

**Source**: [`src/ltx_pipelines/ti2vid_two_stages_hq.py`](../src/ltx_pipelines/ti2vid_two_stages_hq.py)

Uses the **res_2s** second-order sampler instead of Euler. Same stage structure (stage 1 at target resolution with CFG, stage 2 upsampling with distilled LoRA) and image conditioning support. Typically allows fewer steps for comparable quality; trade-offs differ from the default Euler-based pipeline.

**Use when:** You want the same two-stage workflow with fewer steps or prefer the res_2s sampling behavior.

---

## 3. TI2VidOneStagePipeline

**Best for:** Educational purposes and quick prototyping.

**Source**: [`src/ltx_pipelines/ti2vid_one_stage.py`](../src/ltx_pipelines/ti2vid_one_stage.py)

> **⚠️ Important:** This pipeline is primarily for educational purposes. For production-quality results, use [`DFRPipeline`](#12-dfrpipeline). For guided CFG/STG two-stage, use `TI2VidTwoStagesPipeline`.

Single-stage generation (no upsampling) with [multimodal guidance](multimodal-guidance.md) and image conditioning support. Faster inference but lower resolution output (typically 512x768).

**Use when:** Learning how the pipeline works, quick prototyping, testing, or when high resolution is not needed.

---

## 4. DistilledPipeline

**Best for:** Fastest inference with good quality using a distilled model with predefined sigma schedule. **Starting point.**

**Source**: [`src/ltx_pipelines/distilled.py`](../src/ltx_pipelines/distilled.py)

Two-stage generation with 8 predefined sigmas (8 steps in stage 1, 4 steps in stage 2). Each stage samples with Euler ancestral on an LTX-2.5+ checkpoint and with deterministic Euler below that. No guidance required. Fastest inference among all pipelines. Supports image conditioning. Requires spatial upsampler.

**Extra CLI arguments:** `--chunk-pixel-frames` (default 97) and/or `--chunk-carry-frames` (default 25) enable long-video windowing; `--chunk-blend-frames` controls the decoded seam crossfade (default: full carry, `0` for a hard cut). Also on `ltx_pipelines.distilled_mgpu`.

**Use when:** Fastest inference is critical, batch processing many videos, or when you have a distilled model checkpoint.

---

## 5. ICLoraPipeline

**Best for:** Video-to-video and image-to-video transformations using IC-LoRA.

**Source**: [`src/ltx_pipelines/ic_lora.py`](../src/ltx_pipelines/ic_lora.py)

The **CLI is always two stages** on a distilled checkpoint. Stage 1 generates at half of `--height` / `--width` with the IC-LoRA and `--video-conditioning` reference. Stage 2 upsamples 2x and refines. `--skip-stage-2` drops stage 2 for a half-res preview.

Transformer tiling is opt-in and is **not extra stages**. Without `--tile`, 1080-class sizes (for example 1088x1920) stay the untiled recipe even if they are larger than the default tile window. Raising `--height` / `--width` alone does not tile; 4K without `--tile` may OOM (the CLI re-raises that error with `--tile` / tile size and `--offload`).

**`--stage-2-ic-lora`.** Default stage 2 runs the bare checkpoint (the upsampled canvas can recover structure when it is **not** tiled). Pass this flag to keep the IC-LoRA and reference on stage 2 as well. Required when `--tile` makes stage 2 larger than the tile window; each tile is a crop at normalized positions and cannot follow the driving video without the reference.

**`--tile`.** Same two stages, each transformer step evaluated over a fixed window (default `--tile-height 1024` / `--tile-width 1536`, matching stock distilled `--height` / `--width`). An axis smaller than the window is left as one tile, so stage 1 often stays untiled while stage 2 tiles. Override the window with `--tile-height` / `--tile-width` (at least 64, divisible by 32). Those size flags require `--tile`.

**Python.** `ICLoraPipeline(stages=...)` takes a sequence of `ICLoraStageConfig`. Consecutive stages must keep the same `resolution_divisor` or drop by exactly 2x, and the last stage must be divisor 1. That is how you run more than two stages (same-res repeats or `4 -> 2 -> 1`). The CLI does not expose that ladder.

**`--audio-path` and `--refine-audio`.** `--audio-path` is optional. With it, the audio (or the audio track of a video) is read from `--audio-start-time` for `--num-frames / --frame-rate` seconds, encoded once, and held frozen as clean target audio in stage 1. Audio shorter than the clip raises an error. `--refine-audio` (default on) lets stages 2+ denoise the stage-1 audio; `--no-refine-audio` keeps it frozen. In Python, pass `audio=AudioInput(path, start_time)` and `refine_audio=` to `ICLoraPipeline.__call__`.

| Audio source | `refine_audio` | Stage 1 audio | Stage 2+ audio | Output audio |
|---|---|---|---|---|
| none | `True` (default) | generated | refined | VAE decode |
| none | `False` | generated | frozen | VAE decode |
| given | `False` | frozen source | frozen source | source waveform |
| given | `True` | frozen source | refined | VAE decode |

With a source and a single stage (`--skip-stage-2`), no stage denoises audio, so the output carries the source waveform regardless of `refine_audio`.

**Note:** Distilled checkpoint only. Pass the IC-LoRA with `--lora`.

```bash
# 1080-class, today's two-stage recipe (untiled; stage 2 bare checkpoint)
python -m ltx_pipelines.ic_lora \
    --distilled-checkpoint-path path/to/distilled.safetensors \
    --spatial-upsampler-path path/to/upsampler.safetensors \
    --gemma-root path/to/gemma \
    --lora path/to/ic-lora.safetensors \
    --video-conditioning path/to/guide.mp4 1.0 \
    --prompt "..." \
    --height 1088 --width 1920 \
    --output-path out.mp4

# Same size, keep the guide on stage 2, still untiled
python -m ltx_pipelines.ic_lora \
    ... \
    --height 1088 --width 1920 --stage-2-ic-lora \
    --output-path out.mp4

# 4K: two stages, default 1024x1536 transformer windows
python -m ltx_pipelines.ic_lora \
    ... \
    --height 3840 --width 2176 --tile --stage-2-ic-lora \
    --output-path out.mp4

# 4K with a smaller window
python -m ltx_pipelines.ic_lora \
    ... \
    --height 3840 --width 2176 --tile --tile-height 576 --tile-width 1024 --stage-2-ic-lora \
    --output-path out.mp4

# Keep the input audio (frozen in both stages; the output carries the source waveform)
python -m ltx_pipelines.ic_lora \
    ... \
    --audio-path path/to/speech.wav --no-refine-audio \
    --output-path out.mp4
```

**Extra CLI arguments:** `--video-conditioning` (required), `--conditioning-attention-mask`, `--skip-stage-2`, `--stage-2-ic-lora`, `--tile`, `--tile-height`, `--tile-width`, `--chunked` or `--chunk-pixel-frames` / `--chunk-carry-frames` for long clips, and `--chunk-blend-frames` for decoded seam crossfade length (same windowing as Distilled and Dub-It), `--audio-path`, `--audio-start-time`, `--refine-audio` / `--no-refine-audio`.

**Python.** ``stream_chunks()`` returns ``(Iterator[PlaybackSegment], num_frames, tiling_config)``. Pass ``chunk_config`` to ``__call__`` or ``stream_chunks()``; ``chunk_config=None`` is one window over the snapped clip (today's one-shot path). Temporal chunks are the outer loop; each stage still runs per-step spatial tiling when ``--tile`` or ``ICLoraStageConfig.tiling`` is set, so runtime scales roughly with ``temporal_chunks x spatial_tiles x denoising_steps``.

**Use when:** Video-to-video transformations, image-to-video with strong control, or when you have reference videos to guide generation.

---

## 6. Keyframe interpolation

**Best for:** Generating videos by interpolating between keyframe images.

Use [`TI2VidTwoStagesPipeline`](#1-ti2vidtwostagespipeline) with explicit keyframe-guide image bindings. In Python,
construct each image with `ImageConditioningInput.guide(path, frame_idx, strength)`, where `frame_idx` is a
zero-based pixel-frame index on the decoded output timeline; on the CLI, pass each image as
`--image PATH FRAME_IDX STRENGTH guide [CRF]` and add `--refine-audio`. The CLI index uses the same pixel-frame
unit.

The two-stage pipeline uses guiding latents (additive conditioning) for these inputs, with
[multimodal guidance](multimodal-guidance.md) in stage 1 and upsampling in stage 2. Pass `refine_audio=True` to
also refine audio in stage 2, preserving the former keyframe-interpolation behavior.

**Use when:** You have keyframe images and want to interpolate between them, creating smooth transitions, or animation/motion interpolation tasks.

---

## 7. A2VidPipelineTwoStage

**Best for:** Generating video driven by an input audio.

**Source**: [`src/ltx_pipelines/a2vid_two_stage.py`](../src/ltx_pipelines/a2vid_two_stage.py)

Two-stage audio-to-video generation. Stage 1 generates video at half resolution with audio conditioning (video-only denoising with the audio frozen), then Stage 2 upsamples by 2x and refines the video while keeping the audio fixed, using a distilled LoRA. The input audio is encoded via the audio VAE and used as the initial audio latent, but the original audio waveform is passed through and returned in the output to preserve fidelity. Supports image conditioning and prompt enhancement.

**Extra CLI arguments:** `--audio-path` (required), `--audio-start-time`, `--audio-max-duration`, `--chunk-pixel-frames` / `--chunk-carry-frames`, and `--chunk-blend-frames`. `--num-frames` and `--audio-max-duration` are mutually exclusive. With `--audio-max-duration`, video length is derived from the effective clip (`min(--audio-max-duration, remaining audio after --audio-start-time)`) at `--frame-rate`, snapped to the VAE temporal grid (`8k+1`). With `--num-frames` (or neither), audio is clipped to `--num-frames / --frame-rate`.

**Use when:** You have an audio clip and want to generate a matching video, audio-reactive video generation, or music visualization.

---

## 8. RetakePipeline

**Best for:** Regenerating a specific time region of an existing video while keeping the rest unchanged.

**Source**: [`src/ltx_pipelines/retake.py`](../src/ltx_pipelines/retake.py)

Single-stage generation that encodes the source video and audio into latents, applies a temporal region mask to mark `[start_time, end_time]` for regeneration, and denoises only the masked region from a text prompt. Content outside the time window is preserved. Supports independent control over video and audio regeneration (`regenerate_video`, `regenerate_audio` flags), and can use either the full model with CFG guidance or the distilled model with a fixed sigma schedule.

**Extra CLI arguments:** `--video-path` (required), `--start-time` (required), `--end-time` (required).

**Constraints:** Source video frame count must satisfy the 8k+1 format (e.g. 97, 193) and resolution must be multiples of 32.

**Use when:** You want to re-do a specific section of a generated video (e.g. fix a bad segment), selectively regenerate audio or video in a time window, or iterate on part of a result without re-generating the entire clip.

---

## 9. HDRICLoraPipeline

**Best for:** Video-to-video generation with HDR output for EXR export and offline tonemapping.

**Source**: [`src/ltx_pipelines/hdr_ic_lora.py`](../src/ltx_pipelines/hdr_ic_lora.py)

One-stage ACEScct SDR-to-HDR on a distilled model with an HDR IC-LoRA. The source is converted to ACEScct (`--input-colorspace`), VAE-encoded, and used as the IC-LoRA reference. Denoising is the distilled 8-step Euler schedule over the whole clip. Text embeddings are loaded from a `.safetensors` file. Output is a 10-bit BT.2020/HLG master plus an EXR sequence beside it (`--exr-colorspace`, default ACEScg).

Seam keyframes are on by default. Pass `--no-keyframes` for plain IC-LoRA. When keyframes are on, pass `--video-vae-path` to a keyframe-trained video VAE.

Native EXR `--hdr` conditioning on the other pipelines is a separate path; see [HDR Support](hdr.md).

**Extra CLI arguments:** `--input` (required), `--output-path` (required), `--exr-colorspace` (`acescg` / `acescct` / `srgb_linear`; default `acescg`), `--hdr-lora` (required), `--text-embeddings` (required), `--distilled-checkpoint-path` or `--transformer-path`, `--video-vae-path` (needed with keyframes, the default), `--input-colorspace`, `--frame-rate` (EXR directories only), `--seed`, `--high-quality`, `--offload`, `--no-quantization`, `--diffvae-optimization`, `--no-keyframes`, `--keyframe-strength`.

**Constraints:** Source frame count must be 8k+1. `--frame-rate` is required for EXR folders and must not be set for MP4/MOV.

**Use when:** You need ACEScct HDR from an SDR video (or linear Rec.709 / ACEScg / ACEScct EXR folder) for HLG masters and EXR grading.

---

## 10. DubItPipeline

**Best for:** Dub-It — rephrasing while keeping the same speaker identity and matching lip movements to new audio.

**Source**: [`src/ltx_pipelines/dubit.py`](../src/ltx_pipelines/dubit.py)

Uses IC-LoRA on a **distilled** checkpoint with a **single** Dub-It IC-LoRA applied in **both** stages. The reference clip provides video and audio reference tokens whose VAE latents are appended to the target audio sequence as frozen reference tokens. The frame count and frame rate are derived from the reference video (frame count is silently snapped to the nearest `8k+1`), so the CLI does not accept `--num-frames` or `--frame-rate`. Required: `--reference-video`. Optional: `--reference-strength`. LoRA: [`Lightricks/LTX-2.3-22b-IC-LoRA-DubIt`](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-DubIt).

**Extra CLI arguments:** `--chunk-pixel-frames` / `--chunk-carry-frames` and `--chunk-blend-frames`. Frame count still comes from the reference video.

**Note:** Requires a distilled model checkpoint and one Dub-It IC-LoRA (`--lora` exactly once).

**Use when:** Dubbing, rephrasing with matched lips and speaker identity.

---

## 11. T2AOneStagePipeline

**Best for:** Text-to-audio — generating speech/audio only (no video) from a text prompt, e.g. driving an audio-style LoRA such as an accent LoRA.

**Source**: [`src/ltx_pipelines/t2a_one_stage.py`](../src/ltx_pipelines/t2a_one_stage.py)

Single-stage, **audio-only** generation: the video branch is absent (`video=None`), so only the audio modality is denoised and decoded through the audio VAE + vocoder, producing a wave file. Audio duration is derived from `--num-frames` / `--frame-rate` (the same `8k+1` frame convention as video). Audio guidance (CFG/STG) is optional — the `--audio-*` flags default to the model's values; the video→audio cross-modal guidance is disabled since there is no video modality.

**Extra CLI arguments (all optional, with sensible defaults):** `--num-frames`, `--frame-rate`, `--negative-prompt`, `--audio-cfg-guidance-scale`, `--audio-stg-guidance-scale`, `--audio-stg-blocks`, `--audio-rescale-scale`, `--audio-skip-step`. No `--height/--width/--image` (audio has no spatial dimensions).

**Use when:** You need speech/audio from text alone, or to evaluate an audio-only LoRA (accent, voice style) without generating video.

---

## 12. DFRPipeline

**Best for:** Production-quality text/image-to-video — generating at half resolution with extra generated keyframes, then re-rendering at full resolution with a spatial detailing LoRA, optionally densifying time by 2x or 4x.

**Source**: [`src/ltx_pipelines/dfr_pipeline.py`](../src/ltx_pipelines/dfr_pipeline.py). Stage functions live in [`dfr_stages.py`](../src/ltx_pipelines/dfr_stages.py) (`denoise_stage1`, `denoise_stage2`, `run_one_temporal_round`, `run_spatial_epilogue`); layout and pure helpers live under [`dfr_helpers/`](../src/ltx_pipelines/dfr_helpers/). Other adapters import the stage functions instead of wrapping `DFRPipeline`.

Diffusion Fidelity Rendering runs the distilled sigma schedule on a **distilled checkpoint** (the same transformer as DistilledPipeline). Stage 1 generates video **and [generated keyframe slots](conditioning.md#generated-keyframe-slots)** at half resolution, placing slots on an 8-frame-border segment grid; the half-resolution result is kept as a reference while video and keyframes are upsampled in latent space. Stage 2 re-denoises at full resolution with a 2x spatial detailing IC-LoRA, conditioned on the stage-1 reference. Both stages and the spatial epilogue sample with Euler ancestral on an LTX-2.5+ checkpoint and with deterministic Euler below that.

Audio comes from **stage 1**. Stage 2 still runs an audio pass, because the video branch needs the cross-modal attention, but nothing refines audio after stage 1.

**Temporal refinement (optional, default `0`).** `--temporal-upscalings {0,1,2}` adds rounds that each double the **playback** frame rate: the canvas is upsampled temporally, split into `2**round` tiles, given fresh mid-segment slots, and densified with ancestral Euler. A non-first tile starts on the last keyframe plane before its seam; that prefix is pinned to the previous tile's output. Whatever padding the canvas needs internally, you always get `(num_frames - 1) * 2**rounds + 1` frames back. `--temporal-upsampler-path` is required when the count is greater than 0.

**Spatial epilogue (`--spatial-upscalings 2`).** Default `--spatial-upscalings` is `1`: stage 1 at `h/2`, stage 2 at the final `h×w`. With `2`, stage 1 is at `h/4` and stage 2 at `h/2`. The epilogue decodes each carry keyframe as its own latent, Lanczos-stretches the RGB x2, and encodes it again. Only the video latent is spatially upsampled. Time windows run sequentially (pinned prefix handover) and are 4×4-tiled spatially (10 latent cells of overlap); the 3-step detailing schedule runs with those keyframes as strength-1 conditions, the previous-stage video as the IC-LoRA reference, and frozen stage-1 audio so video-audio cross-attention still has a stream. The pipeline still ships the original stage-1 audio. Spatial tiles blend after every Euler step. Detailing LoRA strength is fixed at 0.5. The same encoded keyframes are used for keyframe-aware DiffVAE decode.

**Extra CLI arguments:** `--detailing-lora PATH [STRENGTH]` (required; strength is ignored and hardcoded to 0.5), `--spatial-upscalings {1,2}` (default `1`), `--temporal-upsampler-path` (required when `--temporal-upscalings` > 0), `--temporal-upscalings {0,1,2}` (default `0`). Unlike the other keyframe-capable pipelines, DFR does **not** take `--num-generated-keyframes` — it derives slot positions from its own segment grid. There is no `--distilled-lora`; the distilled weights are the checkpoint.

### Running DFR

`--height` / `--width` are the **final** output size. They must be divisible by **64** (by **128** when `--spatial-upscalings 2`). UHD 4K is `3840×2176`, not `3840×2160` (2160 is not on the 64-pixel grid).

`--num-frames` is the **stage-1** count on the VAE temporal grid (`8k+1`). `--num-frames 121` at the default 24 fps is about 5.04 s. Output length is still that wall-clock duration; extra temporal rounds add frames, not seconds:

| `--temporal-upscalings` | Frames from 121 | Playback fps | Duration |
| --- | --- | --- | --- |
| `0` (default) | 121 | 24 | ~5.04 s |
| `1` | 241 | 48 | ~5.02 s |
| `2` | 481 | 96 | ~5.01 s |

The file is encoded at that playback fps. The transformer independently snaps conditioning fps to 60 whenever playback fps is above 30 (so a 48 fps clip still ships at 48 fps). Some players treat unusual H.264 rates such as 96 fps as 24 fps and play the file in slow motion.

Shared memory flags (`--quantization`, `--offload`, `--compile`, `--diffvae-optimization`) work the same as on other pipelines; see [Optimization Tips](optimization.md). Defaults are no compile, no quant, no offload, and DiffVAE `chunked_eager`. CUDA-graph compile (`capture=true`, `reduce-overhead`, `max-autotune`) on single GPU needs `--offload cpu` or `disk` so stage rebuilds reuse the same GPU weight slots — that is not the static input pool. DFR hits several transformer shapes (half-res stage 1, full-res stage 2, temporal tiles). With `--compile capture=true`, each shape still gets its own CUDA graph, but `max_video_tokens` / `max_audio_tokens` can size **one** static input pool for all of them. Count the **full** stage-2 sequence: target tokens from the final `height`×`width` **plus** five keyframe planes **plus** the half-res IC-LoRA reference (text-only defaults: 38400 at 1024×1536, 204000 at 3840×2176; see [token budgets](optimization.md#compilation-torchcompile)).

**Cost.** Stage 2 at `--spatial-upscalings 1` denoises the full output resolution. `--spatial-upscalings 2` keeps stage 2 at half res and finishes 4K in the tiled epilogue — use that if stage 2 OOMs at 4K. Each temporal round splits the canvas into `2**round` tiles and **reloads the transformer per tile**, so wall-clock grows much faster than the four Euler steps suggest. After denoising, DFR always **keyframe-decodes**; that is usually the VRAM cliff, not the denoise. `AUTO_TILING` sizes decode tiles for the keyframe path automatically — there is no decode-tile CLI flag. Install the `natten` extra for production DiffVAE decode (`uv sync --package ltx-core --extra natten`); without it the decoder falls back to Triton. On datacenter Blackwell, `--diffvae-optimization blackwell_dsl` is the fast decode path.

Monolith (fat distilled checkpoint + Gemma directory):

```bash
uv run python -m ltx_pipelines.dfr_pipeline \
    --distilled-checkpoint-path path/to/distilled-checkpoint.safetensors \
    --gemma-root                path/to/gemma \
    --detailing-lora            models/ltx-2.5/loras/ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors \
    --spatial-upsampler-path    models/ltx-2.5/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
    --width 3840 --height 2176 --num-frames 121 \
    --output-path output.mp4 --prompt "..."
```

Split (Comfy-aligned files; omit `--gemma-root`). Same distilled transformer as DistilledPipeline. Add `--temporal-upsampler-path` and `--temporal-upscalings 1` or `2` for 48 / 96 fps:

```bash
uv run python -m ltx_pipelines.dfr_pipeline \
    --transformer-path        models/ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
    --text-encoder-path       models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
    --video-vae-path          models/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors \
    --audio-vae-path          models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors \
    --detailing-lora          models/ltx-2.5/loras/ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors \
    --spatial-upsampler-path  models/ltx-2.5/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
    --temporal-upsampler-path models/ltx-2.5/latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors \
    --temporal-upscalings 1 \
    --width 3840 --height 2176 --num-frames 121 \
    --output-path output.mp4 --prompt "..."
```

**Note:** Requires a distilled checkpoint that supports generated keyframe slots (LTX-2.5 and later). Do not pass the full (dev) transformer. Multi-GPU: `python -m ltx_pipelines.dfr_mgpu` with the same flags; both stages use sequence parallelism.

**Use when:** You want production-quality output, quality over wall-clock time, or a higher effective frame rate than the base model produces.
