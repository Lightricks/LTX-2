# Available Pipelines

Full reference for each pipeline. See the [Pipeline Selection Guide](pipeline-selection.md) to pick one.

---

## 1. TI2VidTwoStagesPipeline

**Best for:** High-quality text/image-to-video generation with upsampling. **Recommended for production use.**

**Source**: [`src/ltx_pipelines/ti2vid_two_stages.py`](../src/ltx_pipelines/ti2vid_two_stages.py)

Two-stage generation: Stage 1 generates low-resolution video with [multimodal guidance](multimodal-guidance.md), Stage 2 upsamples to 2x resolution with distilled LoRA refinement. Supports image conditioning. Highest quality output, slower than one-stage but significantly better quality.

**Use when:** Production-quality video generation, higher resolution needed, quality over speed, text-to-video with image conditioning.

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

> **⚠️ Important:** This pipeline is primarily for educational purposes. For production-quality results, use `TI2VidTwoStagesPipeline` or other two-stage pipelines.

Single-stage generation (no upsampling) with [multimodal guidance](multimodal-guidance.md) and image conditioning support. Faster inference but lower resolution output (typically 512x768).

**Use when:** Learning how the pipeline works, quick prototyping, testing, or when high resolution is not needed.

---

## 4. DistilledPipeline

**Best for:** Fastest inference with good quality using a distilled model with predefined sigma schedule.

**Source**: [`src/ltx_pipelines/distilled.py`](../src/ltx_pipelines/distilled.py)

Two-stage generation with 8 predefined sigmas (8 steps in stage 1, 4 steps in stage 2). No guidance required. Fastest inference among all pipelines. Supports image conditioning. Requires spatial upsampler.

**Use when:** Fastest inference is critical, batch processing many videos, or when you have a distilled model checkpoint.

---

## 5. ICLoraPipeline

**Best for:** Video-to-video and image-to-video transformations using IC-LoRA.

**Source**: [`src/ltx_pipelines/ic_lora.py`](../src/ltx_pipelines/ic_lora.py)

Two-stage generation with IC-LoRA support. Can condition on reference videos (video-to-video) or images at specific frames. CFG guidance in stage 1, upsampling in stage 2. Requires IC-LoRA trained model.

**Note:** ICLoraPipeline can only be used with a distilled model.

**Use when:** Video-to-video transformations, image-to-video with strong control, or when you have reference videos to guide generation.

---

## 6. KeyframeInterpolationPipeline

**Best for:** Generating videos by interpolating between keyframe images.

**Source**: [`src/ltx_pipelines/keyframe_interpolation.py`](../src/ltx_pipelines/keyframe_interpolation.py)

Two-stage generation with keyframe interpolation. Uses guiding latents (additive conditioning) instead of replacing latents for smoother transitions. [Multimodal guidance](multimodal-guidance.md) in stage 1, upsampling in stage 2.

**Use when:** You have keyframe images and want to interpolate between them, creating smooth transitions, or animation/motion interpolation tasks.

---

## 7. A2VidPipelineTwoStage

**Best for:** Generating video driven by an input audio.

**Source**: [`src/ltx_pipelines/a2vid_two_stage.py`](../src/ltx_pipelines/a2vid_two_stage.py)

Two-stage audio-to-video generation. Stage 1 generates video at half resolution with audio conditioning (video-only denoising with the audio frozen), then Stage 2 upsamples by 2x and refines the video while keeping the audio fixed, using a distilled LoRA. The input audio is encoded via the audio VAE and used as the initial audio latent, but the original audio waveform is passed through and returned in the output to preserve fidelity. Supports image conditioning and prompt enhancement.

**Extra CLI arguments:** `--audio-path` (required), `--audio-start-time`, `--audio-max-duration`.

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

Two-stage video-to-video on the distilled model with an HDR IC-LoRA. Decoded latents pass through an HDR inverse transform (ARRI LogC3, auto-detected from LoRA metadata) to produce a **linear HDR float** tensor `[f, h, w, c]`. Video-only (audio skipped). Text embeddings are pre-computed externally and loaded from a `.safetensors` file. Tonemapping and EXR saving are the caller's responsibility. LoRA and embeddings: [`Lightricks/LTX-2.3-22b-IC-LoRA-HDR`](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-HDR).

**Extra CLI arguments:** `--input` (mp4 or directory, required), `--output-dir` (required), `--hdr-lora` (required), `--text-embeddings` (pre-computed `.safetensors`, required), `--num-frames`, `--spatial-tile` (tiled VAE decode tile size; reduce on lower-VRAM GPUs), `--skip-mp4` (EXR only, no H.264 preview), `--exr-half` (float16 EXR), `--high-quality` (generates 2x frames internally for smoother output, ~2x slower), `--offload {none,cpu,disk}` (weight offloading; disables FP8 quantization when not `none`).

**Use when:** You need linear HDR float output for EXR export, color grading, or custom tonemapping workflows.

---

## 10. LipDubPipeline

**Best for:** Lip dubbing, rephrasing while keeping the same speaker identity and matching lip movements to new audio.

**Source**: [`src/ltx_pipelines/lipdub.py`](../src/ltx_pipelines/lipdub.py)

Uses IC-LoRA on a **distilled** checkpoint with a **single** lip-dub IC-LoRA applied in **both** stages. The reference clip provides video and audio reference tokens whose VAE latents are appended to the target audio sequence as frozen reference tokens. The frame count and frame rate are derived from the reference video (frame count is silently snapped to the nearest `8k+1`), so the CLI does not accept `--num-frames` or `--frame-rate`. Required: `--reference-video`. Optional: `--reference-strength`. LoRA: [`Lightricks/LTX-2.3-22b-IC-LoRA-LipDub`](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-LipDub).

**Note:** Requires a distilled model checkpoint and one lip-dub IC-LoRA (`--lora` exactly once).

**Use when:** Dubbing, rephrasing with matched lips and speaker identity.

---

## 11. T2AOneStagePipeline

**Best for:** Text-to-audio — generating speech/audio only (no video) from a text prompt, e.g. driving an audio-style LoRA such as an accent LoRA.

**Source**: [`src/ltx_pipelines/t2a_one_stage.py`](../src/ltx_pipelines/t2a_one_stage.py)

Single-stage, **audio-only** generation: the video branch is absent (`video=None`), so only the audio modality is denoised and decoded through the audio VAE + vocoder, producing a wave file. Audio duration is derived from `--num-frames` / `--frame-rate` (the same `8k+1` frame convention as video). Audio guidance (CFG/STG) is optional — the `--audio-*` flags default to the model's values; the video→audio cross-modal guidance is disabled since there is no video modality.

**Extra CLI arguments (all optional, with sensible defaults):** `--num-frames`, `--frame-rate`, `--negative-prompt`, `--audio-cfg-guidance-scale`, `--audio-stg-guidance-scale`, `--audio-stg-blocks`, `--audio-rescale-scale`, `--audio-skip-step`. No `--height/--width/--image` (audio has no spatial dimensions).

**Use when:** You need speech/audio from text alone, or to evaluate an audio-only LoRA (accent, voice style) without generating video.
