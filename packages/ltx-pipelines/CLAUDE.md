<!--
MAINTENANCE: When modifying any pipeline class in src/ltx_pipelines/,
update this document to reflect changes to:
- __init__ / __call__ signatures
- sigma handling or step counts
- denoiser types or guidance
- new or removed pipelines
Run: ls src/ltx_pipelines/*.py to check for new pipeline files.
-->

# ltx-pipelines

Inference pipelines for LTX-2 audio-video generation. Depends on `ltx-core` for model definitions, diffusion components, and loading. All pipelines live in `packages/ltx-pipelines/src/ltx_pipelines/`.

## Pipeline selection

| Pipeline | File | Stages | Model | Sampler | Use case |
|----------|------|--------|-------|---------|----------|
| `TI2VidOneStagePipeline` | `ti2vid_one_stage.py` | 1 | Full | Euler | Simple text/image-to-video |
| `TI2VidTwoStagesPipeline` | `ti2vid_two_stages.py` | 2 | Full + distilled LoRA | Euler | Production quality |
| `TI2VidTwoStagesHQPipeline` | `ti2vid_two_stages_hq.py` | 2 | Full + distilled LoRA (both stages) | Res2s | Highest quality, fewer steps |
| `A2VidPipelineTwoStage` | `a2vid_two_stage.py` | 2 | Full + distilled LoRA | Euler | Audio-conditioned video |
| `KeyframeInterpolationPipeline` | `keyframe_interpolation.py` | 2 | Full + distilled LoRA | Euler | Keyframe interpolation |
| `DistilledPipeline` | `distilled.py` | 2 | Distilled only | Euler | Fastest inference |
| `ICLoraPipeline` | `ic_lora.py` | 2 | Distilled only | Euler | Video-to-video with IC-LoRA control |
| `RetakePipeline` | `retake.py` | 1 | Full or distilled | Euler | Video region regeneration |

## Guidance

- **CFG**: Blends conditioned/unconditioned predictions. Defaults: `cfg_scale=3.0` (video), `7.0` (audio).
- **STG**: Perturbs self-attention in transformer blocks. Default `stg_scale=1.0`, `stg_blocks=[28]` (LTX-2.3) / `[29]` (LTX-2). HQ disables STG (`stg_scale=0.0`).
- **Modality guidance**: Cross-modal attention scaling (`modality_scale=3.0`).
- All guidance is stage 1 only. Stage 2 always uses `SimpleDenoiser`.

## Sigma schedules and step counts

- **Scheduler-based** (full model): `self._scheduler = LTX2Scheduler()` with `execute(steps=N)` (HQ also passes `latent=` for token-count-dependent shift). Defaults: 30 steps (LTX-2.3), 40 (LTX-2), 15 (HQ).
- **Distilled**: Fixed 8-step `DISTILLED_SIGMA_VALUES` (9 values). Stage 2 uses 3-step `STAGE_2_DISTILLED_SIGMA_VALUES` (4 values). No `num_inference_steps` param.
- **Retake**: `num_inference_steps=40` default; ignored when `distilled=True` (fixed 8-step).
- **Overrides**: All pipelines accept optional sigma tensors in `__call__`: `sigmas` (one-stage), `stage_1_sigmas` + `stage_2_sigmas` (two-stage).

## LoRA conventions

- No default LoRAs. `loras` param defaults to empty list/tuple. `DEFAULT_LORA_STRENGTH = 1.0`.
- Two-stage non-distilled pipelines require `distilled_lora` (applied to stage 2 only in TI2Vid/A2Vid/Keyframe).
- HQ is unique: applies distilled LoRA to **both** stages with separate `distilled_lora_strength_stage_1` / `_stage_2` params.

## Shared building blocks (`utils/blocks.py`)

- `DiffusionStage` -- owns transformer lifecycle; builds model on call, frees on exit via `gpu_model()` context manager (moves params to meta device to release GPU/CPU memory). Accepts optional `stepper` and `loop` overrides.
- `PromptEncoder` -- Gemma text encoder + embeddings processor (video 4096-dim, audio 2048-dim).
- `ImageConditioner` / `AudioConditioner` -- temporary encoder scope; builds encoder, passes to callable, frees.
- `VideoUpsampler` -- 2x spatial upsampling via encoder + upsampler.
- `VideoDecoder` / `AudioDecoder` -- latent-to-pixel decoding (iterator for video, `Audio` for audio).

### Memory management

- **Model lifecycle**: All blocks build their model on call and free it on exit. `gpu_model()` moves params to `"meta"` device on exit, immediately releasing storage. No model persists between calls.
- **Block streaming**: When offloading is enabled, `DiffusionStage` wraps the transformer in `BlockStreamingWrapper`. Blocks live on pinned CPU memory; only 2 blocks are buffered on GPU at a time (one for compute, one for async H2D copy on a separate CUDA stream).
- **Batch splitting**: `BatchSplitAdapter` wraps the transformer and splits inputs exceeding `max_batch_size` into sequential chunks. If guidance needs B=4 but `max_batch_size=1`, it runs 4 sequential B=1 passes. Higher `max_batch_size` reduces layer-streaming PCIe transfers at the cost of peak memory.

## Denoisers (`utils/denoisers.py`)

- `SimpleDenoiser` -- single forward pass (B=1), no guidance. Used by distilled pipelines and all stage 2.
- `GuidedDenoiser` -- CFG/STG with static `MultiModalGuider` instances (HQ, A2Vid, Retake non-distilled).
- `FactoryGuidedDenoiser` -- per-step guider creation via factory (OneStageTI2Vid, TwoStagesTI2Vid, Keyframe).

Guided denoisers batch all guidance passes into a **single transformer call**: states are repeated along the batch dimension, contexts concatenated, and a `BatchedPerturbationConfig` controls which attention ops are skipped per sample. Pass count is dynamic: B=2 for CFG-only, up to B=4 with CFG+STG+modality isolation. Results are split back and blended by the guider.

## Per-pipeline unique features

- **HQ**: Res2s second-order sampler for **both** stages, latent-dependent sigma schedule, distilled LoRA on both stages with separate strengths.
- **A2Vid**: Audio frozen in both stages (`frozen=True, noise_scale=0.0`). Returns original audio (not VAE-decoded); no `AudioDecoder`.
- **IC-LoRA**: `VideoConditionByReferenceLatent`, `reference_downscale_factor` from LoRA metadata, `skip_stage_2`, attention mask downsampling. Stage 2 is LoRA-free and uses `combined_image_conditionings` (no IC-LoRA conditioning).
- **Keyframe**: Uses `image_conditionings_by_adding_guiding_latent` in both stages (all frames as keyframe guidance, no replacement) -- unlike TI2Vid which uses `combined_image_conditionings` (frame_idx=0 replaces, others guide).
- **Retake**: `TemporalRegionMask` for selective time-window regeneration. `regenerate_video`/`regenerate_audio` flags. Conditional distilled/full behavior.
- **Distilled**: Single `self.stage` reused for both stages (not `stage_1`/`stage_2`).

## Image conditioning helpers (`utils/helpers.py`)

- `combined_image_conditionings()` -- images with `frame_idx==0` replace latent (`VideoConditionByLatentIndex`), others guide (`VideoConditionByKeyframeIndex`).
- `image_conditionings_by_adding_guiding_latent()` -- all images become keyframe guidance regardless of `frame_idx`.
