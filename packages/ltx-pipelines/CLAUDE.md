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

## Chunk generation (`chunks/`)

Long-video framework: the `Chunk` / `PlaybackSegment` wire (`state.py` and `utils/media_io/types.py`), `uniform_chunk_layouts` (`layout.py`) and `generate_uniform_chunks` (`planner.py`), chunk operators in `operators.py` (`denoise_chunks`, `spatially_upsample_chunks`, `replace_video_conditionings`, `replace_audio_conditionings`, `decode_chunks`, `replace_chunks_audio`), and stream I/O in `export.py` (`pipeline_output_from_chunks`, `SequentialVideoFrameSource` for overlap-aware IC-LoRA reference decode). `encode_media_segments` writes ordered `PlaybackSegment` video/audio pairs incrementally through one PyAV container thread. `Chunk` carries user-provided `make_video_conditionings` / `make_audio_conditionings` callables that materialize `ConditioningItem` lists at the current latent shape when the chunk reaches denoise, so the same factory survives a spatial upsample between passes. Pipelines assemble those callables from per-item builders in `conditionings.py` (`image_conditionings_for_chunk`, `generated_keyframe_slots_for_chunk`, `reference_video_conditionings_for_chunk`, `audio_reference_conditionings_for_chunk`). Point items (images, generated-keyframe slots) map onto the chunk that owns the stitched `frame_idx` via `ChunkLayout.local_frame_index`. Span items (IC-LoRA reference video, audio-ref) apply to every window and slice to that window inside the factory. Stage-2 swaps a new factory through `replace_video_conditionings` / `replace_audio_conditionings` rather than filtering. `generate_uniform_chunks(audio_latent=)` seeds frozen source audio; `replace_chunks_audio` muxes the original waveform after decode instead of vocoding.

Chunked pipelines hold the same `DiffusionStage` the regular ones do, so every chunk operator wraps a core component through its `__call__`: `denoise_chunks` wraps `DiffusionStage` exactly as `spatially_upsample_chunks` wraps `VideoUpsampler`. `denoise_chunks` calls the stage once per chunk with a `VideoAudio` of `ModalitySpec`s carrying that chunk's own latents and carry-appended conditionings. Continuity across a seam is a `VideoConditionByLatentIndex` / `AudioConditionByLatentIndex` pin of the previous chunk's latent tail at index 0.

Pipelines stay at `ltx_pipelines/<name>.py`. Distilled, TI2Vid two-stage, A2Vid two-stage, Dub-It, and IC-LoRA expose long-video generation via `stream_chunks()`, which composes the chunk operators and returns `(Iterator[PlaybackSegment], num_frames, tiling_config)`. `__call__` on those pipelines delegates to `stream_chunks` and wraps the result in `PipelineOutput`; with `chunk_config=None` that is a single full-clip chunk. They also accept `chunk_pixel_frames` / `next_video_carry_frames` on `__call__` via `ChunkConfig`. Each pipeline's `_prepare_run` validates inputs, encodes prompts, and resolves frame count / tiling (plus pipeline-specific prep such as A2Vid source audio) before `stream_chunks` plans windows. DFR is one-shot only; chunked DFR is deferred.

CLI: on `python -m ltx_pipelines.distilled`, `python -m ltx_pipelines.ti2vid_two_stages` (and their MGPU runners), `python -m ltx_pipelines.a2vid_two_stage`, or `python -m ltx_pipelines.dubit`, pass `--chunk-pixel-frames` (default 97) and/or `--chunk-carry-frames` (default 25) via `add_chunk_layout_args` (`utils/args.py`); `--chunk-blend-frames` controls decoded video seam crossfade length (default: full carry, `0` for a hard cut). CLI drivers always pass `stream_chunks` directly to `encode_media_segments`; with no chunk flags, `stream_chunks` / `uniform_chunk_layouts` plans one chunk covering the snapped clip.

TI2Vid two-stage freezes stage-1 audio during stage 2 by default. Pass `refine_audio=True` (CLI: `--refine-audio`)
to denoise both modalities in stage 2, including when replacing the former keyframe-interpolation pipeline.

## Pipeline selection

| Pipeline | File | Stages | Model | Sampler | Use case |
|----------|------|--------|-------|---------|----------|
| `TI2VidOneStagePipeline` | `ti2vid_one_stage.py` | 1 | Full | Euler | Simple text/image-to-video |
| `T2AOneStagePipeline` | `t2a_one_stage.py` | 1 | Full | Euler | Text-to-audio (audio-only output, no video branch) |
| `TI2VidTwoStagesPipeline` | `ti2vid_two_stages.py` | 2 | Full + distilled LoRA | Euler | Guided two-stage (CFG/STG), including keyframe interpolation |
| `TI2VidTwoStagesHQPipeline` | `ti2vid_two_stages_hq.py` | 2 | Full + distilled LoRA (both stages) | Res2s | Guided two-stage, res_2s, fewer steps |
| `A2VidPipelineTwoStage` | `a2vid_two_stage.py` | 2 | Full + distilled LoRA | Euler | Audio-conditioned video |
| `DFRPipeline` | `dfr_pipeline.py` | 2 (+ optional tiled temporal) | Distilled checkpoint + detailing IC-LoRA stage 2 | Euler ancestral (2.5+) | Production quality (keyframes + detailing + optional temporal x2) |
| `DistilledPipeline` | `distilled.py` | 2 | Distilled only | Euler ancestral (2.5+) | Fastest inference |
| `ICLoraPipeline` | `ic_lora.py` | 2 | Distilled only | Euler | Video-to-video with IC-LoRA control |
| `DubItPipeline` | `dubit.py` | 2 | Distilled only | Euler | Dub-It with IC-LoRA + audio ref conditioning |
| `RetakePipeline` | `retake.py` | 1 | Full or distilled | Euler | Video region regeneration |
| `HDRICLoraPipeline` | `hdr_ic_lora.py` | 1 | Distilled + HDR IC-LoRA | Euler | ACEScct SDR-to-HDR; DFR seam keyframes on by default |
| `AlphaGenPipeline` | `alpha_gen.py` | 1 | Full + IC-LoRA | Euler | Guided one-stage IC-LoRA (video CFG 1, STG off) |

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
- Two-stage non-distilled pipelines require `distilled_lora` (applied to stage 2 only in TI2Vid/A2Vid).
- HQ applies distilled LoRA to **both** stages with separate `distilled_lora_strength_stage_1` / `_stage_2` params.
- DFR loads a distilled checkpoint (no distilled LoRA). Stage 2 adds a detailing IC-LoRA via `with_loras`.

## Shared building blocks (`utils/blocks.py`)

- `DiffusionStage` -- owns transformer lifecycle; builds model on call, frees on exit via `gpu_model()` context manager (moves params to meta device to release GPU/CPU memory), except that a builder whose weights the registry keeps GPU-resident (`keeps_gpu_resident_weights`) under `AllocatorTrimStrategy.DEFER` leaves its build assigned: every build reassigns those resident weights, so a dispose would free nothing. Stages given one `active_weights` dict (e.g. DFR's passes) share one transformer allocation: builds reuse the registry's bound model or switch LoRAs in place, and stay assigned until `release_transformer()`. Accepts an optional `loop` override. `__init__` takes a pre-built transformer builder; pipelines construct it via the `DiffusionStage.from_checkpoint(checkpoint_path, ..., loras=...)` classmethod, which builds the standard (and, when offloading, streaming) builders. `with_builder` / `with_loras` return a new stage with a swapped builder / LoRA set without re-specifying config. ``supports_generated_keyframes`` reads ``use_keyframes_abs_pos_embedding`` from the checkpoint config (no weights loaded). ``__call__`` re-checks slot conditionings via ``_assert_supports_conditionings`` as a backstop for callers that build items directly.
  `__call__` takes `modalities: VideoAudio[ModalitySpec]` and is a latent entry point: each `ModalitySpec` brings the latent to denoise, and that latent is the stage's only source of shape -- pixel dimensions never reach it, and it never sizes a latent itself. A pipeline's *first* stage seeds the latent with `create_initial_video_latent` / `create_initial_audio_latent` / `create_initial_av_latents` (`utils/helpers.py`); every later stage passes the previous stage's output, upscaled or not. So latents are allocated once per run, at the pipeline head. Sizing audio off a playback rate that differs from the video RoPE time base is done by building the audio latent at that rate, not by a stage argument. Returns `VideoAudio`, which unpacks as `(video, audio)`.
- `PromptEncoder` -- Gemma text encoder + embeddings processor (video 4096-dim, audio 2048-dim).
- `ImageConditioner` / `AudioConditioner` -- temporary encoder scope; builds encoder, passes to callable, frees. `ImageConditioner` is additionally the single owner of the image-conditioning CRF: `resolve_crf(images)` fills in the H.264 CRF of any `ImageConditioningInput` (`utils.types`) that left it unset, reading it from the checkpoint's `model_version` (`detect_params`, lazily and cached). So omitting `crf` means "use what matches this model" (33 through LTX-2.3, 18 from 2.4), while an explicit `crf` -- including `0` for no re-compression -- is always honoured. `load_conditionings(images, color_space=None)` (called near the top of every pipeline's `__call__`) resolves the CRF that way and decodes each distinct still once for the run (SDR re-compressed at its CRF; EXR with the run's `color_space`). It returns `LoadedImageConditioning`s (`utils.types`): the decoded still plus the input's `binding` and `strength`, with `frame_idx` / `assert_in_clip` / `map_frame_index` / `to_item`. Everything after it -- chunk and DFR rebasing, stage/tile/window plumbing, `image_conditionings()` -- takes only `LoadedImageConditioning`s; rebasing keeps the still, so encodes only resize per resolution. `ImageConditioningInput` is the user-facing request (path, strength, binding, CRF).
- `VideoUpsampler` -- 2x spatial upsampling via encoder + upsampler.
- `VideoDecoder` / `AudioDecoder` -- latent-to-pixel decoding (iterator for video, `Audio` for audio). `VideoDecoder` takes a single decoder `checkpoint_path` and `diffvae_optimization` (`DiffVAEMode`, default `CHUNKED_EAGER`); pipelines pass `model_paths.video_vae()`, the same file `ImageConditioner` and `VideoUpsampler` build their encoder from -- a video VAE checkpoint carries encoder and decoder together. Decoder kind (conv vs diffusion) is chosen from checkpoint metadata (`is_diffusion_video_vae`); distilled DiffVAE uses fixed 2 Euler steps. CLI: `--video-vae-path` is the video VAE slot in both monolith (optional override; defaults via `ModelPaths.from_monolith`) and split modes; `--diffvae-optimization` (see `docs/optimization.md#diffusion-vae-decoder` for mode meanings and relative compile/runtime/VRAM factors). Multi-GPU pipelines (`*_mgpu.py`) accept the same flags and wrap the loaded decoder in `DistributedVideoDecoder`.
- `DurationPredictor` -- predicts a frame count from `PromptEncoder`'s connector token outputs via `DurationHead` (`ltx_core.duration_head`), snapped to the VAE's temporal grid. Unlike other blocks, it holds the built model directly (`__init__` takes a `DurationHead`, not a builder) since the checkpoint is only a few MB -- no build-on-call/free-on-exit needed. Pipelines construct it via `DurationPredictor.from_checkpoint(checkpoint_path, dtype, device)`, which returns `None` instead of a predictor if the checkpoint has no `duration_head.*` weights (checkpoints predating LTX-2.5 / gemma4). Called with `(video_encoding, audio_encoding, frame_rate=...)` -- either may be `None` but not both. Wired into `TI2VidOneStagePipeline`, `TI2VidTwoStagesPipeline`, `TI2VidTwoStagesHQPipeline`, `DistilledPipeline`, `DFRPipeline`, and `T2AOneStagePipeline` (audio-only: `video_encoding=None`), plus their MGPU runners: when `num_frames` is omitted (`None`), `require_num_frames_source` (called at the top of `__call__`, before any work) raises immediately if no `DurationPredictor` is available, otherwise it's auto-predicted from the caption. Auto-duration resolves inside `__call__`; read the final count from ``PipelineOutput.num_frames`` (or the local ``num_frames`` variable before return).

### Memory management

- **Model lifecycle**: All blocks build their model on call and free it on exit. `gpu_model()` moves params to `"meta"` device on exit, immediately releasing storage. The exceptions are a `DiffusionStage` transformer with registry-resident weights under `DEFER`, and stages sharing `active_weights` until `release_transformer()`, which stay assigned (see `DiffusionStage` above); otherwise no model persists between calls.
- **Unit streaming**: When offloading is enabled, `DiffusionStage` builds a `StreamingModel`. RAM mode retains selected units on CPU; disk mode reads them into reusable host slots. A scheduler delivers them through accelerator slots, using CUDA or MPS completion rules.
- **Batch splitting**: `BatchSplitAdapter` wraps the transformer and splits inputs exceeding `max_batch_size` into sequential chunks. If guidance needs B=4 but `max_batch_size=1`, it runs 4 sequential B=1 passes. Higher `max_batch_size` reduces layer-streaming PCIe transfers at the cost of peak memory.

## Denoisers (`utils/denoisers.py`)

- `SimpleDenoiser` -- single forward pass (B=1), no guidance. Used by distilled pipelines and all stage 2.
- `GuidedDenoiser` -- CFG/STG with static `MultiModalGuider` instances (HQ, A2Vid, Retake non-distilled).
- `FactoryGuidedDenoiser` -- per-step guider creation via factory (OneStageTI2Vid, TwoStagesTI2Vid).

All denoisers return `VideoAudio` of `DenoisedLatentResult` (defined in `utils/types.py`). It still unpacks as `(video_result, audio_result)`; either side may be `None` for absent modalities. `DenoisedLatentResult.denoised` is the final blended tensor. Guided denoisers additionally populate per-pass fields (`.cond`, `.uncond`, `.ptb`, `.mod`) on each result; `SimpleDenoiser` leaves these `None`.

`GuidedDenoiser` and `FactoryGuidedDenoiser` accept `force_uncond_pass=True` to run the uncond pass even when `cfg_scale=1.0` (required by CFG++ when the guidance scale is 1 but the uncond prediction is still needed for the ODE derivative). Requires `negative_context` to be set on the guider. When enabled, `DenoisedLatentResult.uncond` will be a tensor instead of `None`.

Guided denoisers batch all guidance passes into a **single transformer call**: states are repeated along the batch dimension, contexts concatenated, and a `BatchedPerturbationConfig` controls which attention ops are skipped per sample. Pass count is dynamic: B=2 for CFG-only, up to B=4 with CFG+STG+modality isolation. Results are split back and blended by the guider.

## PipelineOutput (`utils/types.py`)

Every video pipeline ``__call__`` returns ``PipelineOutput``: ``video``, ``audio``, ``num_frames``, ``tiling_config``. Prefer attribute access over positional unpack. ``video`` is a lazy decode iterator; ``audio`` is an ``ltx_core.types.Audio``. Slot latents and the final video latent are **not** returned -- keyframe-aware decode is chosen inside the pipeline (``decode_with_keyframes`` on TI2Vid/Distilled, always on DFR). ``result.video`` already reflects that choice.

## Per-pipeline unique features

- **HQ**: Res2s second-order sampler for **both** stages, latent-dependent sigma schedule, distilled LoRA on both stages with separate strengths.
- **A2Vid**: Audio frozen in both stages (`frozen=True, noise_scale=0.0`). That zeros `denoise_mask` and forces audio `Modality.sigma=0` (prompt AdaLN / a2v gate). Returns original audio (not VAE-decoded); no `AudioDecoder`. Chunked mode seeds each chunk from the encoded source latent (`generate_uniform_chunks(audio_latent=)`) and muxes the original waveform after decode (`replace_chunks_audio`). CLI `--num-frames` and `--audio-max-duration` are mutually exclusive: with `--audio-max-duration`, `num_frames` is omitted (`None`) and derived from the decoded clip (capped by remaining audio after `--audio-start-time`); with `--num-frames` or neither, that frame count is used and audio is clipped to match. Returns ``PipelineOutput`` like the other two-stage pipelines.
- **IC-LoRA**: `VideoConditionByReferenceLatent`, `reference_downscale_factor` / `reference_temporal_scale_factor` from LoRA metadata, `skip_stage_2`, attention mask downsampling. Default stage 2 is LoRA-free (`image_conditionings` only); `apply_ic_lora` / `--stage-2-ic-lora` keeps the reference on a stage. CLI is two stages (`--tile` plus `--tile-height` / `--tile-width` for transformer windows). Longer ladders use `ICLoraPipeline(stages=...)`. Temporal chunking: `stream_chunks()` / `chunk_config` compose the shared chunk operators with per-stage LoRA and `diffusion_stage_with_tiling`; reference video uses `SequentialVideoFrameSource` and chunk-local mask/temporal scaling via `reference_video_conditionings_for_chunk`. CLI `--chunk-pixel-frames` / `--chunk-carry-frames` / `--chunk-blend-frames` stream encode like Distilled/Dub-It. Optional `audio: AudioInput` (`--audio-path` / `--audio-start-time`) is decoded and encoded once in `_prepare_run` into `EncodedAudio` (latent + clipped source waveform) on the run context; audio shorter than the clip raises before the prompt encoder. Stage 1 freezes audio iff a source is given (`frozen=True, noise_scale=0.0`, same `ModalitySpec` as A2Vid); stages 2+ freeze iff `refine_audio=False` (`--no-refine-audio`; default `True` keeps main's behavior). When a source is given and no stage denoises audio, the output carries the source waveform instead of the `AudioDecoder` output. Tiled stages pass the full audio stream to each window; the averaged audio prediction is discarded by the zero denoise mask.
- **Dub-It**: Standalone pipeline; IC reference **video** helpers and audio-reference patchify (negative RoPE) in `iclora_utils.py`. Appends frozen audio-reference tokens via `AudioConditionByReferenceLatent` (ltx-core), matching video token order (`[target | ref]`) while keeping reference RoPE positions negative (training-compatible). Single IC-LoRA on both stages; full IC-LoRA video conditioning at stage 1 and 2; stage-2 audio is frozen with S1 latent as initial state and uses S1-derived ref. Final audio decoded from stage 1 latent. Chunked mode walks the IC-LoRA reference video once through `SequentialVideoFrameSource` (overlap lookback) and encodes audio-reference tokens in the chunk factories (stage 2 swaps the audio factory to self-reference via `replace_audio_conditionings`). The Dub-It CLI does not expose `--conditioning-attention-mask`; use `ic_lora.py` if you need spatial IC attention masking.
- **TI2Vid two-stage (image conditioning)**: Preserves each caller-supplied binding through chunking and DFR rebasing. Use `ImageConditioningInput.replace()` / `.guide()` / `.at()` (or `--image` with optional trailing `replace` / `guide`) explicitly; the pipeline does not rewrite bindings. Keyframe interpolation uses `.guide()` at endpoint frames plus optional `refine_audio=True` (see `ti2vid_two_stages.py`).
- **Retake**: `TemporalRegionMask` for selective time-window regeneration. `regenerate_video`/`regenerate_audio` flags. Conditional distilled/full behavior.
- **Distilled**: Single `self.stage` reused for both stages (not `stage_1`/`stage_2`). The sampler is resolved in `__init__` from the checkpoint generation -- `should_use_ancestral_sampler(path)` (`detect_model_version(...) >= (2, 5)`) sets `self.use_ancestral_sampler`, so an LTX-2.5+ checkpoint samples every stage with `euler_ancestral_denoising_loop(..., eta=1.0, s_noise=1.0)` (ComfyUI's `sample_euler_ancestral_RF` defaults), and anything older samples with plain Euler. There is no per-call override; assign the attribute before calling to pin a sampler (e.g. to A/B both on one checkpoint). `distilled_mgpu.py` detects per rank from the same checkpoint, so ranks stay symmetric. Each pass has its own noise-seed offset (`ANCESTRAL_NOISE_SEED_OFFSET`, `ANCESTRAL_STAGE_2_NOISE_SEED_OFFSET`), so no two of them inject the same noise and stage 1's first draw is not bit-identical to the initial `GaussianNoiser` noise. Within a pass the seed advances by one on each denoise call, so sequential chunks do not reuse ancestral noise across a seam; a single `DiffusionStage` call is index 0, so one-shot output matches a fixed-seed partial.
- **DFR** (Diffusion Fidelity Rendering, `dfr_pipeline.py` orchestrator + `dfr_stages.py` +
  `dfr_helpers/` sidecar). Adapters import `denoise_stage1`, `denoise_stage2`, `run_one_temporal_round`,
  and `run_spatial_epilogue` from `dfr_stages` (and helpers from `dfr_helpers.ops` /
  `dfr_helpers.layout`) instead of wrapping `DFRPipeline`. Stages must not instantiate Gemma /
  `ModelPaths`. Operator how-to (resolution, fps, memory) is
  [Running DFR](docs/pipelines.md#running-dfr). The pipeline itself:
  distilled-schedule pipeline on a keyframe-slot-capable **distilled checkpoint**, plus an x2
  detailing IC-LoRA (required; ``--detailing-lora``). Stage 1, stage 2 and the spatial epilogue take
  their sampler from the same `use_ancestral_sampler` rule as DistilledPipeline (eta=1.0), each with
  its own noise-seed offset; temporal tiles densify with ancestral Euler at eta=0.5 regardless. Stage 1 (half-res) generates video + keyframe slots on an **x8-border segment
  grid**: pad ``(num_frames-1)`` up to a multiple of S (prefer S=32 unless S=24 pads strictly less);
  positions ``S, 2S, …, N'-1``. Half-res video is reserved for IC-LoRA; stage 2 spatially upsamples
  video and slot keyframes before denoising. Stage 2 jointly denoises with the distilled weights and the x2
  detailing IC-LoRA (``VideoConditionByReferenceLatent`` on the
  reserved half-res stage-1 video, ``STAGE_2_DISTILLED_SIGMAS``). Shipped audio is **stage 1's**:
  stage 2 still runs an audio pass (video needs the cross-modal attention) but re-noises audio under
  the detailing LoRA. ``denoise_stage1`` accepts an injectable audio ``ModalitySpec`` (default
  ``ModalitySpec(latent=..., context=audio_context)``); A2V callers must pass ``frozen=True, noise_scale=0.0``
  with ``latent=`` so the supplied audio is not re-noised before freezing. Temporal tiles pass frozen stage-1 audio sliced to the tile's playback
  *time* window (uncapped fps vs stage-1 duration) and retiled onto the snapped-fps token count
  for cross-attention only; they do not refine or ship audio.
  Optional ``temporal_upscalings`` (0–2): each round temporally x2-upsamples, partitions into
  ``2**round`` keyframe-seam tiles; a non-first tile starts on the last keyframe plane before
  its seam with a pinned prefix. Invents mid-segment slots per tile, and densifies with ancestral
  Euler (η=0.5). Partition/stitch uses
  ``dfr_helpers.layout.split_canvas_at_seams`` (remainder segments go to the leading tiles; core
  ``split_at_seams`` uses the same remainder rule). Optional
  ``spatial_upscalings`` (1 or 2, CLI ``--spatial-upscalings``): ``1`` is stage 1 at ``h/2`` and
  stage 2 at ``h``; ``2`` is stage 1 at ``h/4``, stage 2 at ``h/2``, then a full-res spatial
  detailing epilogue at ``h``. Carry keyframes are decoded one plane at a time, Lanczos-stretched
  x2 in RGB, and encoded as ``VideoConditionByKeyframeIndex`` (strength 1); only the video latent is
  spatially upsampled, and the previous-stage video is the IC-LoRA reference. Detailing LoRA
  strength is hardcoded to 0.5. The epilogue runs **one denoising pass per temporal window**,
  sequentially, each pass tiled *spatially* inside the transformer call by ``TiledDiffusionModel``
  (``frames=1``) -- the single-GPU twin of the multi-GPU ``TiledDataParallelModelWrapper``, sharing
  ``VideoModalityTilingHelper`` and its blend masks, installed via
  ``DiffusionStage.with_model_wrapper``. The first step runs on a 2x2 grid before any remaining
  steps retile to 4x4; a one-step schedule stays 2x2. Every Euler step reconciles the *spatial*
  overlap, and spatial conditionings are filtered per tile at the token level rather than cropped
  by hand. The
  carried planes are conditioning only -- they are Lanczos upscales, so anchoring the decode on them
  blurs what a keyframe should sharpen; the epilogue attaches a slot a latent frame beside each and
  **ships those full-res planes** instead. Running the temporal windows sequentially rather than per
  step costs the same -- they are seam-cut and never blended into each other -- and is what gives
  each window a finished predecessor to pin its prefix to (``run_spatial_epilogue``). Per window the
  driver rebases image conditionings, filters the keyframe bag and the full-res slots, slices the
  frozen stage-1 audio, and crops the IC-LoRA reference.
  Wrap the *model*, not the builder: a wrapping builder hides ``StreamingBuilder`` from
  ``_is_streaming``, so ``--offload`` would skip the streaming teardown. Temporal windows are cut on the last round's
  window seams and their lead-in is pinned to the previous window, so time windows never blend into
  each other; spatial tiles blend.
  ``height``/``width`` are always the **final** output; ``2`` requires they divide 128.

  Three invariants worth knowing before touching the temporal rounds:
  - **A non-first tile is a clip, not a slice, and its prefix is pinned.** It begins on the plane at
    the last keyframe position before its seam: cell 0 *is* that plane, cells
    ``1 .. (seam - keyframe)/scale`` are the video between them, and the tile resumes on the cell
    after the seam. The arithmetic closes -- cell ``k`` covers
    ``keyframe + scale*(k-1) + 1 ... keyframe + scale*k`` -- so the last pinned cell ends exactly
    *on* the seam and the seam keyframe needs no conditioning of its own: it is absorbed there.
    This matters because a tile is denoised as its own clip, whose cell 0 the model reads as a
    single pixel frame (causal convention); starting on a mid-canvas 8-frame cell put content and
    shape in disagreement and ran RoPE time 7 frames ahead of the canvas. Starting on a real
    one-frame plane makes the tile look like an i2v clip, which is in distribution.
    The whole prefix is pinned with ``VideoConditionByLatentIndex(strength=1.0)`` -- the plane, then
    the previous tile's *finished output* for those cells -- so ``denoise_mask`` is 0 there and
    ``lerp(clean, noised, mask)`` leaves them equal to it at every step (the mechanism A2Vid uses to
    freeze audio). The tiles are made to agree rather than told about the same frame and expected
    to. Keyframe conditionings before the resume point are dropped: the pinned cells hold this
    stage's content while a plane there holds an earlier stage's, and two contents for one moment is
    worse than none. Because the prefix follows plane positions rather than a cell count,
    ``split_at_seams`` takes **no overlap**: kept runs are disjoint and tile the canvas exactly once
    (``tile_prefix``).
  - **Conditioning fps snaps to 60 above 30** (``conditioning_fps``), independently of playback fps.
    RoPE time is ``pixel_frame / fps``, so a 48 or 120 fps time base is outside the trained
    distribution and the model can no longer lay out the 8 pixel frames inside a latent -- it
    decodes as a motion spike at each 8-frame latent border followed by a stall. 24 fps plus one
    temporal upsample still ships 48 fps; only the transformer uses 60. Decode/encode use playback fps.
    Audio duration is ``frames / fps`` too, so stages 1 and 2 size the audio latent with
    ``create_initial_audio_latent(video_latent, fps=frame_rate, video_scale_factors=...)`` -- otherwise a 48 fps run would ship 121/60 s of
    sound against a 121/48 s picture.
  - **Image conditioning is tile-local.** ``frame_idx`` is a pixel index on the canvas the caller
    requested; each temporal round maps it to ``frame_idx * 2**round`` before the window test.
    After re-basing, ``frame_idx=0`` means the *tile's* first frame, so only images that fall
    inside the window are re-attached. The spatial epilogue uses the same scale
    (``2**temporal_upscalings``).

  The canvas may pad the tail up to a whole segment, but the caller always gets
  ``(requested_frames - 1) * 2**rounds + 1`` frames: the excess is trimmed before decode (always on
  a latent boundary, since ``requested - 1`` is a multiple of the VAE temporal scale), and audio is
  cut to the video's duration. CLI: ``--detailing-lora``, ``--temporal-upsampler-path``,
  ``--temporal-upscalings``, ``--spatial-upscalings``.
  No ``--num-generated-keyframes``.

  Returns ``PipelineOutput``: ``video``, ``audio``, ``num_frames``, ``tiling_config``.
  ``video`` is decoded with ``decode_video(keyframes=)`` using the carry-forward slot planes
  (plain decode only if none survive the final trim).
- **HDR IC-LoRA**: One-stage ACEScct SDR-to-HDR. IC-LoRA `VideoConditionByReferenceLatent` on the VAE-encoded SDR clip; denoise is `DISTILLED_SIGMA_VALUES` (8 Euler steps) with `SimpleDenoiser`. Output is ACEScct `[0, 1]` FHWC plus an EXR sidecar dir (`--exr-colorspace`, default ACEScg). CLI default is `HDRKeyframeConfig` (opt out with `--no-keyframes`); the Python API stays opt-in (`keyframes=None`). A config adds `resolve_canvas` x8 seams, with **every** seam taking both roles: a generated HDR slot and, at the same position, a `VideoConditionByKeyframeIndex` SDR guide (strength 0.95, 1-frame VAE encode), then keyframe-aware DiffVAE decode. A slot per seam matches `dfr_pipeline` (it passes the whole `resolve_canvas` list to `VideoGeneratedKeyframeSlots`); the co-located SDR guide is ours, as DFR's `VideoConditionByKeyframeIndex` guides are round-to-round carry keyframes with fresh slots at segment midpoints. The earlier every-other-seam split is where this pipeline's seam artifacts showed up; pairing the roles costs a full latent frame of tokens per seam instead of per other seam. When using the DiffVAE decode with keyframes the decode VAE must carry a non-zero `decoder.type_emb` (`--video-vae-path .../ltx-2.5-video-vae-bf16.safetensors`); the plain distilled monolith has no such key, the loader synthesizes zeros, and the entire keyframe path then no-ops — clean run, plausible output, keyframes ignored, and nothing checks. Path flags come from the shared `add_model_args`, with the text-encoder, audio-VAE and duration-head components switched off — leaving `--gemma-root` undeclared is what lets a monolith run without a Gemma dir (see `model_paths_from_namespace`).

## Image conditioning helpers (`utils/helpers.py`)

- `image_conditionings()` -- encodes `LoadedImageConditioning`s using each `binding` (`ImageReplace` / `ImageGuide`). Both bindings carry pixel-frame indices; `ImageReplace` accepts causal latent starts (`0` or `1 + 8 * k`) and converts to a latent index only when lowered to `VideoConditionByLatentIndex`.
- `ImageConditioningInput.at()` -- combined i2v default: frame 0 replaces latent, later frames guide.
- `evenly_spaced_keyframe_positions()` -- evenly spaced **interior** positions (both endpoints excluded).
- `generated_keyframe_conditionings()` / `resolve_generated_keyframes()` -- turn the pipeline-level ``generated_keyframes`` arg into ``VideoGeneratedKeyframeSlots`` or pixel indices.
- `decode_keyframes_from_slots()` -- build ``DecodeKeyframes`` from carried slot latents, dropping planes past a trimmed canvas.
- `assert_generated_keyframes_request()` / `assert_stage_supports_generated_keyframes()` -- preamble validation before any model is built (capability gate + ``decode_with_keyframes`` requires slots).

## Generated keyframes

Optional, off by default on most pipelines, and requires a checkpoint whose transformer config sets
`use_keyframes_abs_pos_embedding`. Appends empty, fully-denoised single-pixel-frame token slots at
interior frame positions so the model generates extra frames there, relaxing the effective temporal
compression at those positions. Slots may optionally carry ``initial_keyframes`` latent seeds
(written into the appended ``latent`` tokens; ``denoise_mask=1`` still applies).

- **Where**: stage 1 on `TI2VidOneStagePipeline`, `TI2VidTwoStagesPipeline`,
  `TI2VidTwoStagesHQPipeline`, `DistilledPipeline`, `DFRPipeline`, and the three
  `*_mgpu` runners (`distilled_mgpu`, `ti2vid_two_stages_mgpu`, `ti2vid_two_stages_hq_mgpu`).
  ``decode_with_keyframes`` is supported on the four single-GPU pipelines above except DFR, plus
  those MGPU runners. One-stage runs slots and decode at final resolution (no stage-2 re-attach).
  Chunked two-stage pipelines keep **seeded** slots as conditioning at every stage, upsampling
  stage-1 slot latents with ``self.upsampler`` on the full ``(B, C, K, H, W)`` stack (same as DFR),
  then refining at full res. ``decode_with_keyframes`` controls only whether those final planes
  anchor video decode. `DFRPipeline` always
  re-attaches seeded slots in stage 2 and optional tiled temporal
  rounds attach mid-segment slots per tile; it always keyframe-decodes internally.
  `HDRICLoraPipeline` also supports generated slots (CLI on by default, `--no-keyframes` to disable).
- **API**: `generated_keyframes: int | Sequence[int] = 0` on `__call__` for TI2Vid/Distilled
  (an `int` requests evenly spaced **interior** keyframes; a sequence gives explicit indices).
  CLI: `--num-generated-keyframes` via `add_generated_keyframes_arg`. Pass
  ``decode_with_keyframes=True`` (CLI: ``--decode-with-keyframes`` via ``add_keyframe_decode_arg``)
  to decode through the keyframe-aware DiffVAE path using those slots as anchors.
  `DFRPipeline` derives slot positions from ``dfr_helpers.layout.resolve_canvas``, does **not** expose
  either flag, and calls ``assert_stage_supports_generated_keyframes`` unconditionally; CLI adds
  `--detailing-lora` (required), `--temporal-upsampler-path`, `--temporal-upscalings`.
  `HDRICLoraPipeline` uses `keyframes: HDRKeyframeConfig | None = None` (CLI default on, `--no-keyframes`, `--keyframe-strength`);
  seam indices come from `dfr_seam_roles`, not `--num-generated-keyframes`.
- **Validation**: ``DiffusionStage.supports_generated_keyframes`` reads the checkpoint config.
  Pipelines call ``assert_generated_keyframes_request(decode_with_keyframes, generated_keyframes, stage)``
  (or ``assert_stage_supports_generated_keyframes`` on DFR) in the ``__call__`` preamble, before
  the text encoder loads. ``DiffusionStage.__call__`` re-checks via ``_assert_supports_conditionings``.
  Each keyframe costs a full latent frame of tokens, so silently degrading would waste 16-31% of
  the token budget.
- **Cost**: one latent frame of tokens per keyframe, yielding 1 pixel frame instead of 8. At
  512x768/241f, 5 keyframes is +16% tokens (~1.35x attention); at 1088x1920/121f it is +31% (~1.72x).
- **On ``PipelineOutput``**: ``video`` already reflects whether the pipeline chose a
  keyframe-aware decode. Slot latents are not shipped on the return type.
- **Reading the keyframes back**: `LatentState.generated_keyframes` (`(B, C, K, H, W)`), extracted by
  `clear_conditioning` using the `generated_keyframe_layout` recorded on the state -- exact, not
  positional slicing. From outside a pipeline, substitute `RecordingDiffusionStage` for
  `pipeline.stage` to inspect per-stage states (including half-res leftovers). Decode each keyframe
  as a standalone one-frame clip; a K-frame causal decode would blend slots that were never adjacent.
  See `internal/scripts/generated_keyframe_diagnostics.py`.
- **Invariants**: slots are *appended* (which preserves target noise across a same-seed
  keyframes/no-keyframes A/B at B=1) and pass `attention_mask=None` (a dense `(B, T, T)` mask would
  be ~1.8 GB at 30k tokens and would disable FA3/FA4).
