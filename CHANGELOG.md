# Changelog

## 1.6.1 - 2026-11-04

### Added

- `EXRColorSpace.ACES2065_1` (`aces2065-1` / `ACES2065_1`) and `Primaries.AP0` add SMPTE ST 2065-1 (AP0 linear, D60 whitepoint) input and output support on `--hdr`, `--exr-colorspace`, and `--input-colorspace`. `ACES2065-1` EXR outputs are written to `<stem>_aces2065-1_exr/` as SMPTE ST 2065-4 compliant uncompressed half-float containers (`compression="none"`, `acesImageContainerFlag=1`, `adoptedNeutral`).

### Changed

- `DubItPipeline.__call__` and `TI2VidTwoStagesHQPipeline.__call__` no longer enter `torch.inference_mode()` themselves, like the other single-GPU pipelines, so they can run where gradients are recorded. The `ltx_pipelines.dubit` and `ltx_pipelines.ti2vid_two_stages_hq` CLIs still run under `inference_mode`; Python callers should wrap the call in `with torch.inference_mode():` (see [Calling pipelines from Python](packages/ltx-pipelines/docs/installation.md#calling-pipelines-from-python)).
- `to_hdr_linear` clamps sub-black decompressed `ACEScct` values at `0.0` in `AP1` before converting to `out_primaries` (so negative `AP1` undershoot below the `ACEScct` toe intercept no longer mixes across channels when converting to `Rec.709` for `srgb_linear` EXRs and HLG masters), and preserves negative out-of-`AP1`-gamut lobes when `out_primaries=Primaries.AP0`.
- The `ltx_kernels_inductor` compile backend also fuses bias-free linear projections followed by tanh GELU (new `LinearGELUNoBiasPass`), such as the LTX-2.5 video feed-forward. `ltx_kernels::linear_gelu` accepts `bias=None`.

## 1.6.0 - 2026-11-03

### Fixed

- STG no longer applies per-head attention gates to the value projection used to bypass attention. This applies to both fully perturbed and mixed batches for video and audio.

### Added

- `ICLoraPipeline` accepts optional `audio: AudioInput` (CLI `--audio-path` / `--audio-start-time`) and `refine_audio` (CLI `--refine-audio` / `--no-refine-audio`, default on). A given source is encoded once and held frozen as clean target audio in stage 1; `refine_audio=False` also freezes audio in stages 2+, and then the output carries the source waveform instead of a VAE decode. Audio shorter than `num_frames / frame_rate` raises. Defaults (`audio=None`, `refine_audio=True`) keep the previous behavior.
- `ltx_pipelines.utils.media_io.encode_media_segments` incrementally encodes ordered `PlaybackSegment` video/audio streams to SDR H.264 or HDR HLG through typed stream specs and a bounded, single-owner container writer.
- `encode_video(..., codec=VideoCodec.PRORES_4444)` writes ProRes 4444 from straight RGBA `(F, H, W, 4)`, tagging BT.709 and keeping alpha full range. A pixel marker stamps RGB only.
- Added custom trainer strategy registration through `register_strategy(name, config_cls)`, with hooks for model and
  optimizer construction, data loading, training steps, validation, and checkpoint state.
- Added `scripts/preprocess.py CONFIG` to preprocess video, audio, captions, and masks from the same YAML used for
  training. `--check` verifies cached data compatibility, and `--overwrite` rebuilds the cache. Set
  `training_strategy.data.decode: true` to decode cached latents for inspection.

### Changed

- The code in this repository is licensed under Apache 2.0. The LTX models are not part of that: LTX-2.5 remains under the LTX-2.x Community License and LTX-2.3 under the LTX-2 Community License.
- `DiffusionStage` takes an `active_weights` dict that stages share as one transformer allocation: their builds reuse the registry's bound model or switch LoRAs in place, and stay assigned until `release_transformer()`. The single-GPU DFR pipeline uses it for all its passes, tiles and the detailing LoRA, and releases the transformer before decoding.
- Distilled, TI2Vid two-stage, A2Vid, Dub-It and IC-LoRA CLIs encode the `stream_chunks` output directly, whether it contains one or multiple temporal chunks.
- All SDR, HDR and progressive encoding paths share the same pipeline-owned container lifecycle. Codec options now apply consistently to SDR and HDR, and HLG with audio uses zero-latency x265 ordering.
- Trainer model, LoRA, data, optimizer, timestep sampling, and validation content settings moved under
  `training_strategy`. Global `optimization` retains runtime settings, and global `validation` retains cadence.
  Legacy flat YAML configurations and legacy strategy names are migrated automatically with a deprecation warning.
- Preprocessing recorded a configuration manifest and rejected incompatible incremental updates. Training checked
  existing manifests before loading models; regenerate incompatible caches with `scripts/preprocess.py CONFIG --overwrite`,
  or explicitly bypass the training check with `allow_stale_preprocessed: true`. Training still accepted older caches
  without a manifest, while incremental preprocessing required rebuilding them with `--overwrite`.
- Trainer checkpoints moved to atomic `checkpoints/step_NNNNN/` directories containing `weights.safetensors`,
  optional `state.pt`, and resume metadata. `checkpoints.load_checkpoint` accepted a run output directory, its
  `checkpoints/` directory, or a specific step directory.
- Shared noise levels became the default for joint audio/video training with `flexible`; set
  `training_strategy.share_sigmas: false` to sample independent noise levels.
- Trainer dependencies switched to `opencv-python-headless>=4.11.0.86` and `scenedetect-headless>=0.7` for environments
  without a GUI.
- Model loading (`ltx_core.loader`) builds models in weight units. `SimpleBuilder(configurator, paths, sd_ops, module_ops, ...)` replaces `SingleGPUModelBuilder` (add LoRAs with `.lora(paths, strength, sd_ops)`); `StreamingBuilder(base, StreamingConfig(blocks_prefix, mode, prefetch_depth, device_buffers, host_buffers))` replaces the block-streaming builder; `ltx_core.loader.distributed_builders` (`SequenceParallelBuilder`, `TiledDataParallelBuilder`, `AccelerateGemmaBuilder`, `BatchParallelGemmaBuilder`) replace the `ltx_pipelines.multigpu` builders and tracker: stage builders given one shared `dst` dict switch LoRAs in place in one replicated allocation, and the registry returns a cached model still holding the requested selection. `ModelRegistry(cache_models=True, cache_metadata=True, placement_policy=...)` caches model shells and checkpoint metadata and keeps weights only where its placement policy, or a builder's `with_placement_policy`, places them; multi-GPU runners use `ltx_pipelines.multigpu.runner.rank_cuda_registry()`. `QuantizationPolicy(sd_ops, fuse_rule, module_ops)` is applied with `compose_quantization`. Every builder follows the `ltx_core.loader.model_builder.ModelBuilder` protocol: `build`, `registry_keys(dtype)` (the registry entries a build queries and retains), `model_config`, `keeps_gpu_resident_weights`, and `builds_per_shape`. `SDOps` value operations take `(key, inputs)` and return a dict of outputs (`with_kv_operation`), and `video_decoder_sd_ops_for_checkpoint` returns a tuple of stages.

### Removed

- The `v_context` / `a_context` constructor arguments of `SimpleDenoiser`, `GuidedDenoiser` and `FactoryGuidedDenoiser`; set `ModalitySpec.context`, or pass them per call. Custom denoisers and denoising loops must accept and forward `v_context` / `a_context`.
- The `width`, `height`, `frames` and `fps` arguments of `create_initial_video_latent` and `create_initial_av_latents`; pass `pixel_shape=VideoPixelShape(...)`.
- `DiffusionStage.resident()`: share `active_weights` between stages instead; distributed stage builders already return their bound model on repeated calls.
- `encode_video` no longer accepts `Iterator[Audio]`; pass one complete `Audio` track, or use `encode_media_segments` for chunk-aligned audio.
- The `scale_factors` argument of `ensure_tiling_config` and the multi-GPU `EnsureTilingConfig` protocol; scale factors are read from `vae_checkpoint_path`.
- `HlgPyAVEncoder`, `encode_linear_hdr_frames_to_hlg_mp4`, `encode_sdr_h264`, PyAV-backed `ltx_core.color.audio_mux`, and the direct `ltx-core` PyAV dependency; use `ltx_pipelines.utils.media_io.encode_video` or `encode_audio`. PyAV remains a direct dependency of `ltx-pipelines` and `ltx-trainer`.
- `ltx_pipelines.chunks.split_decoded_chunks` and `deferred_stitch_audio`; pass the `PlaybackSegment` iterator directly to `encode_media_segments`, or use `pipeline_output_from_chunks` when an in-memory `PipelineOutput` is required.
- Removed the separate `text_to_video` and `video_to_video` trainer strategies; use `training_strategy.name: flexible`
  with video/audio conditions. Legacy YAML configurations are migrated by the config loader.
- Removed the custom strategy hooks `prepare_training_inputs()`, `compute_loss()`, and `get_checkpoint_metadata()`;
  implement the new `TrainingStrategy` contract with `train_step()` (or `_compute_loss()` for its default training
  step), `collect_checkpoint()`, and model, optimizer, data, and state restoration hooks.
- Removed `scripts/process_dataset.py`, `scripts/process_videos.py`, `scripts/process_captions.py`, and
  `scripts/decode_latents.py`; use `scripts/preprocess.py CONFIG` and its configuration options instead.
- Removed `ltx_trainer.video_utils`, `ltx_trainer.validation_runner`, and `ltx_trainer.process_videos`; use
  `ltx_trainer.media_utils`, `ltx_trainer.validation.runner`, and `ltx_trainer.preprocessing`, respectively.
- Removed automatic Hugging Face Hub uploads and the trainer's `hub` configuration; publish saved checkpoints with
  an external upload workflow. Migrating a configuration with `hub.push_to_hub: true` emitted a warning.
- Removed support for resuming legacy flat trainer checkpoints; start a fresh run or resume from a new
  `checkpoints/step_NNNNN/` directory.
- `SingleGPUModelBuilder`, `SafetensorsModelStateDictLoader`, `ltx_core.block_streaming`, `ltx_core.multigpu.sharded_sd`, the `ltx_pipelines.multigpu` model builders and weight tracker, `ModelRegistry(cache_weights=...)`, and the legacy `SDOps` forms (`with_legacy_kv_operation`, single-tensor `apply_to_key_value`, `with_additional_allowed_keys`, `ContentMatching`/`ContentReplacement`/`KeyValueOperationResult`); see the loader entry under Changed.

### Fixed

- DiffVAE `AUTO_TILING` now scales its decode memory budget by the VAE dtype (`vae_dtype`), preventing 2x memory underestimation and Stage 5 out-of-memory errors when decoding in `float32` (such as `HDRICLoraPipeline` and `--hdr` EXR runs).
- Fixed first-frame conditioning suppressing image training loss, temporal positions using the first sample's FPS
  for an entire batch, and reference geometry checks overlooking later batches or additional references.
  Single-frame reference pairs retained their configured temporal scale.
- Fixed validation combinations of masks and reference conditions, audio masks being applied to video, and dropped
  attention masks. Validation disabled dropout while sampling, restored each module's previous mode, preserved
  seeded stochastic decoding including seed zero, and avoided NaNs during single-step sampling.
- Fixed inconsistent image-mask resizing between preprocessing and validation, missing frames in short mask videos,
  lost frame dimensions in single-frame videos, and caption pairing in nested latent caches. Trainer positions and
  mask downsampling also used the correct compression factors for nested VAE encoder configurations.
- Guidance rescaling (`rescale_scale`) estimates its factor only over tokens that are being denoised, so fixed conditioning and appended reference tokens no longer change the scale of the generated output. `MultiModalGuider.calculate` takes an optional `denoise_mask`; without it the factor is computed over all tokens as before. Outputs of conditioned runs with a non-zero `rescale_scale` change.
- HDR encoding publishes complete EXR sidecar sequences only after successful encoding, preventing partial or stale frames when encoding fails or replaces a longer clip.

## 1.5.0 - 2026-10-18

### Added

- `ImageConditioningInput` supports explicit latent-replacement and keyframe-guide bindings. On video CLIs, pass `--image PATH FRAME_IDX STRENGTH replace|guide [CRF]`; omit the binding keyword for the prior automatic binding (frame 0 replaces, later frames guide). Replacement frame indices are expressed in pixel space and must be `0` or a causal latent boundary (`1 + 8 * k`).
- `TI2VidTwoStagesPipeline` supports keyframe interpolation through `ImageConditioningInput.guide()` bindings, or
  `--image … guide` on the CLI. Set `refine_audio=True` or pass `--refine-audio` to preserve the former dedicated
  pipeline's stage-2 audio refinement; the default preserves stage-1 audio.
- `ImageConditioner.load_conditionings(images, color_space=None)` resolves each image conditioning's CRF (like `resolve_crf`), decodes each distinct still once (SDR, or EXR with `color_space`) and returns `LoadedImageConditioning`s: the decoded still plus the input's `binding` and `strength`.
- `MGPUController(cpu_threads_per_rank=8)` caps each rank's intra-op CPU threads (`torch.set_num_threads`) so the ranks' small host tensor ops do not stall on many-core hosts. Pass `None` to keep torch's default.
- `LTX_MGPU_DECODE_RECEIVER_TIMEOUT_SECONDS` sets how long a multi-GPU video decode waits for a rank that stopped streaming before it gives up (default 60).

### Changed

- Multi-GPU pipelines encode prompts faster with the default Accelerate-sharded Gemma: the other ranks now wait on the host until the source rank's Gemma has finished on every GPU, instead of spinning in an NCCL receive on a GPU Gemma is still using. `BroadcastCoordinator` moved from `ltx_pipelines.multigpu._broadcast` to `ltx_core.multigpu.broadcast`, and `BroadcastGemmaWrapper` (and so `AccelerateGemmaWrapper`) takes an optional `coordinator`. `BroadcastCoordinator.wait_for_signal_change` now consumes exactly one signal per call (a worker that polls late no longer skips a signal) and takes an optional `timeout`.
- `ImageConditioningInput` now takes an `ImageReplace` or `ImageGuide` `binding` instead of a `frame_idx`. Replace existing `ImageConditioningInput(path, frame_idx, strength, crf)` calls with `ImageConditioningInput.at(path, frame_idx, strength, crf)` to preserve the previous frame-0 replacement and later-frame guidance behavior; use `.replace(path, frame_idx, strength, crf)` or `.guide(path, frame_idx, strength, crf)` to select a binding explicitly. Both bindings use pixel-frame indices.
- `image_conditionings()` and `image_conditionings_for_chunk()` take `LoadedImageConditioning`s (from `ImageConditioner.load_conditionings`) instead of `ImageConditioningInput`s, and no longer take `color_space`.
- Performance improvements, in collaboration with doubleAI (https://doubleai.com): faster transformer passes (especially under multi-GPU sequence parallelism and with perturbation guidance) and LoRA switches, denoising samplers without per-step CPU synchronization, a DFR spatial epilogue (`--spatial-upscalings 2`) on fewer tiles (at most 1024 px per side, overlapping by 64 px, instead of a 2x2 first step and 4x4 at a 320 px overlap) that multiplies its long video projections in fp8 on GPUs with fp8 GEMMs (compute capability 8.9+), carry keyframes rebuilt on the rank that decodes them, a multi-GPU video decode that streams to MP4 encoding as it decodes, faster diffusion VAE decoding with lower reserved GPU memory, a faster-compiling `--diffvae-optimization blackwell_dsl` kernel, lower GPU memory during MP4 encoding, faster image conditioning, and faster stage starts with a warm model registry. Output is not bit-identical to previous releases, mainly because of the DFR spatial epilogue's new tiles and fp8 GEMMs; diffusion VAE decodes also differ by rounding, and under multi-GPU sequence parallelism audio cross-attention no longer attends to the video sequence's padding tokens.

### Removed

- `ImageConditioningInput.frame_idx`, `.assert_in_clip()`, `.map_frame_index()` and `.to_item()`; use the same members on the `LoadedImageConditioning` returned by `ImageConditioner.load_conditionings`, or `input.binding.frame_idx`.
- `KeyframeInterpolationPipeline` and the `ltx_pipelines.keyframe_interpolation` CLI; use `TI2VidTwoStagesPipeline` with `ImageConditioningInput.guide()` and `refine_audio=True`, or pass `--image … guide` with `--refine-audio`.

### Fixed

- DiffVAE 1-tile decode no longer produces gray frame tails from frame 128 on long 1080p clips under torch < 2.13. CUDA RMSNorm wrapped flattened offsets at 2^32 elements; last-dim norms are now T-chunked.
- Compiling a transformer that uses blockwise FP8/FP6 quantization with `seq_dim_dynamic` no longer pins the sequence length. Previously each new length recompiled the blocks, and a batch size divisible by 4 failed with a `ConstraintViolationError`.
- A warm multi-GPU DFR job (a second or later request in one process) could fail with "Cannot fit a DiffVAE decode tile". Cold and warm runs and every rank now plan the same decode tiles, and ranks with unequal GPUs no longer hang. `--diffvae-optimization blackwell_dsl` keyframe decodes plan smaller tiles.
- Diffusion VAE decode with `--diffvae-optimization blackwell_dsl` could hang indefinitely on datacenter Blackwell: the fused stage-5 attention kernel re-armed its per-tile barriers while some warps were still waiting on them.

## 1.4.2 - 2026-10-01

### Added

- `ltx_pipelines.alpha_gen`, a one-stage IC-LoRA pipeline on the full model with video CFG 1 and STG off.

## 1.4.1 - 2026-09-30

### Changed

- HDR video encoding streams decoded chunks directly into EXR and HLG outputs instead of retaining the full clip in memory.

## 1.4.0 - 2026-09-29

### Added

- `--chunked` on Distilled, TI2Vid two-stage, A2Vid, Dub-It, and IC-LoRA enables long-video generation with the default chunk layout (97-frame windows and 25-frame carry) without setting `--chunk-pixel-frames` or `--chunk-carry-frames`; `--chunk-blend-frames` controls decoded video seam crossfade length.
- `ltx-kernels` production `sm_100a` wheels now load precompiled LTX-2.5 diffusion-VAE CuTe DSL kernels, eliminating first-decode JIT for standard plain and keyframe decoding; custom kernel shapes still compile on first use.
- `ICLoraPipeline` accepts `stages`, a sequence of `ICLoraStageConfig` (resolution, IC-LoRA, sigmas, optional transformer tiling). Default remains half-res with IC-LoRA, then a full-res refinement without it. Consecutive stages must stay at the same resolution or upsample by exactly 2x.
- `spatial_tiled_ic_lora_stages()` builds that two-stage recipe with fixed-size spatial tiling so large canvases stay on a trained window; both stages keep the IC-LoRA and reference.
- `TiledDiffusionModel` and `FixedSizeSpatialTiling` in `ltx_pipelines.utils.tiled_diffusion` run sequential transformer tiles with trapezoidal blending. DFR's spatial epilogue uses the same wrapper.
- `--stage-2-ic-lora` on `ltx_pipelines.ic_lora` keeps the IC-LoRA and reference on stage 2. `--tile` enables transformer tiling (default window 1024x1536; override with `--tile-height` / `--tile-width`). The CLI is always the two-stage recipe; omit `--tile` at 1080-class sizes to keep today's untiled path. A GPU OOM from that CLI is re-raised with `--tile` / `--tile-height` / `--tile-width` and `--offload cpu`.
- `decode_with_keyframes` on `TI2VidOneStagePipeline`, `TI2VidTwoStagesPipeline`, `TI2VidTwoStagesHQPipeline`, and `DistilledPipeline`, plus the `distilled_mgpu`, `ti2vid_two_stages_mgpu`, and `ti2vid_two_stages_hq_mgpu` runners (CLI: `--decode-with-keyframes`). When set together with generated keyframe slots, the pipeline decodes through the keyframe-aware DiffVAE path using those slots as anchors. Two-stage pipelines re-attach seeded slots at stage 2 so the planes sit at final resolution before decode.
- `AudioConditionByLatentIndex` pins an audio latent at a given latent index, matching `VideoConditionByLatentIndex` for video.
- Added `ltx_pipelines.chunks`, a building block for long-video generation: plan a clip as temporal windows, run denoise → upsample → decode one window at a time, and stitch continuity across seams so VRAM stays bounded by window size rather than full clip length.
- Distilled, TI2Vid two-stage, A2Vid, Dub-It, and IC-LoRA can generate long clips in overlapping windows instead of holding the full timeline in memory. Pass `--chunk-pixel-frames` and optionally `--chunk-carry-frames` on the CLI (or `chunk_config` / `stream_chunks()` in Python) to tune window size and overlap; omit both flags for the existing one-shot path.
- Chunked long-video generation plans generated keyframes per window: an integer is a per-chunk slot budget with automatic stitch anchors; an explicit index list is merged with those anchors. Reference conditioning on continuation chunks appends only the suffix after the carried overlap while still encoding the full window for causal VAE alignment.
- DFR diffusion work is importable from `ltx_pipelines.dfr_stages` (`denoise_stage1`, `denoise_stage2`, `run_one_temporal_round`, `run_spatial_epilogue`) and helpers from `ltx_pipelines.dfr_helpers`, so an adapter can run the same path from a constructed `DiffusionStage` without wrapping `DFRPipeline` or instantiating Gemma.
- `--exr-colorspace` on `ltx_pipelines.hdr_ic_lora` (`acescg` / `acescct` / `srgb_linear`; default `acescg`). Only the EXR sidecar encoding changes; the HLG master stays BT.2020/HLG.
- `--transformer-path` on `ltx_pipelines.hdr_ic_lora`, which runs the pipeline from a split checkpoint pack instead of a distilled monolith. A split transformer carries no VAE weights, so `--video-vae-path` has to accompany it.
- `EXRColorSpace` / `VideoInput` in `ltx_pipelines` and an sRGB EOTF in `ltx_core.hdr`, which convert an SDR source into ACEScct.
- `TokenTimesteps` in `ltx_core`, which holds either dense per-token timesteps or a compact distinct-value table plus per-token indices. `TokenTimesteps.per_token()` expands either representation.
- `DenoiseMask` in `ltx_core.types`, which holds a per-token denoise mask and the table of its distinct values. Build the mask on the host; `DenoiseMask.compact(device)` tabulates its values and moves mask and table to the device before denoising.

### Fixed

- `FlashAttention4` accepts current `flash-attn-4` releases that return the output tensor directly while remaining compatible with older tuple-returning releases.

### Changed

- The `ltx_kernels_inductor` compile backend batches the fp8-cast policy's per-forward weight and bias upcasts into a few grouped kernels instead of one per tensor. Outputs are bit-identical.
- The `ltx_kernels_inductor` and `ltx_kernels_eager` compile backends run unmasked attention on whichever of cuDNN and flash attention is faster for each shape, which moves short-query attention (audio queries over video keys) off cuDNN's non-split fallback. The first call per attention shape times both kernels, so warmup takes slightly longer. This also applies when an `SDPA_*` attention backend is pinned.
- Chunk planning owns generated-keyframe placement and appends slots after pipeline conditionings. Generated slots remain transformer conditioning across every stage; `decode_with_keyframes` now controls only whether the final slots anchor video decode.
- `ChunkConfig.overlap_blend_frames` defaults to the full carry overlap (crossfade over all carried frames). Pass `0` for a hard cut at chunk seams.
- `ChunkConfig.overlap_blend_frames` linearly crossfades a configurable suffix of each decoded overlap instead of cutting directly between windows, buffering only that suffix during streaming.
- Under sequence parallelism, the `ltx_kernels_inductor` compile backend has the q/k RMS-norm + RoPE kernel and the v projection write each head straight into the all2all exchange buffer of the rank that owns it, instead of producing a local tensor for a separate exchange kernel. Outputs are bit-identical.
- DFR spatial epilogues run their first denoising step on a 2x2 tile grid before retiling any remaining steps to 4x4. A one-step schedule stays on the 2x2 grid.
- `create_initial_audio_latent` now takes a video latent and required `video_scale_factors`. Joint first stages use `create_initial_av_latents`, which also requires `video_scale_factors`. `create_initial_video_latent` requires `scale_factors` (no default). Pass the same video VAE factors used to size the video latent. Audio-only callers use `AudioLatentShape.from_duration`.
- Distilled and DFR sample with Euler ancestral (`eta=1.0`, `s_noise=1.0`) on LTX-2.5+ checkpoints -- distilled on both stages, DFR on stage 1, stage 2 and the spatial epilogue -- so generated output for those checkpoints differs from the previous release. Older generations sample with deterministic Euler. DFR's temporal densify tiles are unchanged at `eta=0.5`.
- `PipelineOutput` is again a four-field named tuple (`video`, `audio`, `num_frames`, `tiling_config`). Pipelines choose keyframe-aware decode internally; the returned `video` iterator reflects that choice.
- Conv, diffusion, and distributed `decode_video()` accept extra keyword arguments and ignore ones they do not use.
- Denoisers, denoising loops and `DiffusionStage.__call__` return `VideoAudio` instead of a plain `(video, audio)` tuple. It still unpacks as two values, and also exposes `.video` and `.audio`. Omit the absent side (`VideoAudio(video=...)`); constructing with both `None` raises.
- `DiffusionStage.__call__` is now a latent entry point. Pass `modalities=VideoAudio[ModalitySpec]` (either side may be omitted, but not both). Each `ModalitySpec` carries the latent to denoise, and that latent is the stage's only source of shape, so the stage never sizes a latent itself. Seed a pipeline's first stage with `create_initial_video_latent` / `create_initial_audio_latent` / `create_initial_av_latents` in `ltx_pipelines.utils.helpers`, and hand every later stage the previous stage's output, upscaled or not. Audio-only callers no longer need to pass placeholder video dimensions.
- `encode_video` and `encode_sdr_h264` accept streaming video frame iterators and can mux audio supplied as it becomes available, so long chunked runs can write a file without assembling the full clip first.
- One-shot `__call__` on Distilled, TI2Vid, A2Vid, and Dub-It still returns `PipelineOutput`; internally it uses the same chunked path with a single window. Reported `num_frames` is the stitched length on the causal grid.
- `ltx_pipelines.hdr_ic_lora` is a single-stage ACEScct SDR-to-HDR pipeline. It writes a 10-bit BT.2020/HLG master plus an EXR sequence whose colour space is `--exr-colorspace` (default ACEScg; `acescct` writes VAE log codes), where it previously ran two stages and returned linear float from a LogC3 inverse decode for the caller to tonemap and save. `--input-colorspace` selects the input transform, so the source may be an SDR MP4/MOV (`srgb_gamma`, `srgb`) or a directory of EXR frames (`srgb`, `acescg`, `acescct`); EXR directories also need `--frame-rate`, which a container source must not set.
- HDR EXR output directories are named for their colour space: `<output_stem>_acescct_exr`, `<output_stem>_acescg_exr` or `<output_stem>_srgb_linear_exr`. Every HDR run previously wrote `<output_stem>_exr`, so code that globs the old path will not find the frames.
- `--distilled-checkpoint-path` on `ltx_pipelines.hdr_ic_lora` is no longer required, since `--transformer-path` can supply the transformer instead. One of the two is still needed.
- DFR seam keyframes on `ltx_pipelines.hdr_ic_lora` are on by default. `--with-keyframes` is replaced by `--no-keyframes`. A 2.4 transformer fails the capability check unless `--no-keyframes` is passed; a 2.5 distilled monolith still needs `--video-vae-path` to a keyframe-trained VAE or the keyframe path is a silent no-op.
- `ltx-core` requires `colour-science`, which supplies the primaries, YUV matrix and HLG transfer definitions that were previously hand-written constants.
- The transformer projects AdaLN modulation once per distinct timestep value and reads it per token through `Modality.timesteps.indices` when a pipeline supplies a compact timestep table. Outputs are not bit-identical to previous releases. The saving is proportional to the token count while attention grows with its square, so its share of the forward shrinks as resolution rises: about 2-3% on one GPU at 16k tokens (544x960x241), and less per rank under sequence parallelism.
- `Modality.timesteps` takes a `TokenTimesteps` instead of a tensor; wrap per-token timesteps as `TokenTimesteps(timesteps)`.
- `LatentState.denoise_mask` takes a `DenoiseMask` instead of a tensor; wrap the mask as `DenoiseMask(mask)` with `mask` on the host, and call `compact(device)` on it before a denoising loop reads the state.

### Removed

- `stepper` on `DiffusionStage.__call__`, `denoise_chunks`, the DFR stage functions, and the denoising loops. Each loop constructs its own step. Pass `eta` and `s_noise` to `euler_ancestral_denoising_loop` and `euler_cfg_pp_denoising_loop`. `EulerAncestralDiffusionStep` and `EulerCfgPpDiffusionStep` no longer take those arguments; pass them to `step`.
- `TiledModelWrapper` in `ltx_pipelines.dfr_helpers.ops`; use `TiledDiffusionModel` from `ltx_pipelines.utils.tiled_diffusion`.
- `QuantizationPolicy.model_configurator`; pass the configurator directly to the model builder or `DiffusionStage.from_checkpoint(model_configurator=...)`.
- `All2All.set_rank_tokens`; All2All head exchange and AllGather now derive uniform token counts from their input tensors.
- `PipelineOutput.keyframes` and `PipelineOutput.video_latent`; use `decode_with_keyframes=True` (CLI: `--decode-with-keyframes`) on pipelines that support it.
- `width`, `height`, `frames` and `audio_fps` on `DiffusionStage.__call__`; size the latent yourself and pass it as `ModalitySpec(latent=...)`. Audio sized off a playback rate that differs from the video RoPE time base, which `audio_fps` covered in 1.3.0, is now done by building the audio latent at that rate with `create_initial_audio_latent(..., fps=...)`.
- Separate `video` and `audio` kwargs on `DiffusionStage.__call__`; pass `modalities=VideoAudio(...)`.
- `ModalitySpec.initial_latent`; use `ModalitySpec.latent`, which is required.
- `ImageConditioningInput` from `ltx_pipelines.utils.args`; import it from `ltx_pipelines.utils.types`.
- `--output-dir` on `ltx_pipelines.hdr_ic_lora`; use `--output-path`, which names the HLG master and places the EXR directory beside it.
- `--spatial-upsampler-path`, `--num-frames`, `--spatial-tile` and `--skip-mp4` on `ltx_pipelines.hdr_ic_lora`. The pipeline is single-stage and does not upsample, output length follows the source clip (whose frame count must be 8k+1), decode tiling is resolved automatically, and the HLG master is always written beside the EXR frames.

### Fixed

- Generated audio could outlast the decoded video when `num_frames` was not on the video VAE temporal grid (for example 87 frames at 24 fps decoded 81 picture frames against a longer soundtrack). Initial audio latents are now sized from the snapped video canvas.
- Chunked vocoded audio retains a 40 ms overlap and equal-power crossfades it at generation-window seams, avoiding clicks while keeping the causal decoder shortfall at stream startup.
- Multi-GPU sequence-parallel runs compiled with `torch.compile` could produce a corrupted, repeating texture instead of the prompted scene (seen with the `ltx_kernels_inductor` backend). The compiler could keep its own tensors in the All2All exchange buffer, which the next exchange overwrote. `torch.ops.ltx_kernels.send_recv_heads` and `gather_heads` now take that buffer as an argument they write in place and return nothing; `All2All.send_recv_heads` and `All2All.gather_heads` are unchanged.
- Pinned equal-size transformer windows reject an overlap above half the window and round the tile count down, so three tiles never cover one cell. This is the layout used by IC-LoRA `--tile`. The requested overlap is a target and the realized overlap may be smaller. Two tiles on a canvas only slightly larger than the window can still overlap by more than half, because there is no third tile.

## 1.3.0 - 2026-08-25

### Added

- DFR can finish 4K in a tiled spatial epilogue instead of denoising the full canvas in stage 2. `--spatial-upscalings 2` keeps stage 2 at half resolution and upscales in that epilogue, which is the path for better-quality 4K (use `3840x2176`; `3840x2160` is not on the size grid). Default `--spatial-upscalings 1` is unchanged.
- Added `ltx_pipelines.dfr_mgpu`, a multi-GPU DFR runner with the same flags as `ltx_pipelines.dfr_pipeline`.
- Added keyframe-aware diffusion-VAE decoding, which sharpens detail at the frames a generator anchored on. Pass keyframe latents into `VideoDecoder.decode_video(..., keyframes=)`. Needs a keyframe-trained video VAE; older checkpoints decode as before.
- `--compile max_video_tokens=N max_audio_tokens=N` (and the matching `CompilationConfig` fields) cap CUDA-graph input memory when `capture=true`, so several captured shapes share one buffer sized for the largest instead of paying for each separately.
- `DiffusionStage.with_model_wrapper()` runs a callable on each built transformer before denoising, and still composes with block streaming and quantization.
- `VideoDecoder.decode_single_frames()` decodes independent one-frame latents without neighbouring frames bleeding into each other.
- `DecodeKeyframes.for_frame_span()` and `.crop_spatial()` crop keyframe planes to a decode window, keeping the nearest anchors outside the window so a tiled decode matches a whole one.
- `DiffusionStage.__call__` accepts optional `audio_fps=` when audio should be sized from a different frame rate than the video RoPE time base.

### Changed

- `DFRPipeline` and `ltx_pipelines.dfr_mgpu` take a distilled checkpoint (`--distilled-checkpoint-path`) instead of a full checkpoint plus a distilled LoRA. Detailing IC-LoRA and optional user LoRAs are unchanged.
- DFR temporal rounds use `--temporal-upscalings` instead of `--temporal-upsample-rounds`.
- DFR snaps transformer conditioning fps to 60 whenever playback fps is above 30 (previously only values above 60 were capped), so 48 and 50 fps stay in the trained range. Playback fps is unchanged.
- `--diffvae-optimization combined_compile` no longer requires the `natten` extra. Without it the mode falls back to Triton, or to PyTorch when Triton is unavailable.
- Automatic decode tiling sizes tiles for keyframe-aware decoding as well as plain decoding, so a keyframe decode is less likely to run out of memory on a clip a plain decode fits.
- Video pipelines now return a named `PipelineOutput` instead of a 3- or 4-tuple. Extra fields are `keyframes` and `video_latent`. Existing unpacking must switch to `result.video`, `result.audio`, and so on.
- `VideoDecoder.decode_video()` takes `keyframes=` on every implementation. The diffusion decoder uses them; conv logs a warning and decodes without them; distributed splits them across ranks. Callers that never pass keyframes are unaffected; decoder implementations must accept the keyword.
- NVFP4 quantization and CuTe DSL DiffVAE kernels (`blackwell_dsl`) now support Jetson Thor. Build NVFP4 with the arch-specific `110a` target.

- The DFR spatial epilogue now generates the keyframes it ships instead of shipping upscales. The planes it carries in are Lanczos x2 stretches of the previous stage, so anchoring the VAE decode on them blurred the very frames a keyframe is meant to sharpen; they are now conditioning only, and the epilogue attaches a slot a latent frame to the side of each (a slot cannot occupy the frame the plane holds) and ships those full-resolution results. `PipelineOutput.keyframes.pixel_frame_indices` therefore move 8 frames from the carried positions. A candidate that would land inside a temporal window's pinned prefix is skipped, so a few carried positions ship no plane at all.
- A DFR temporal tile (and epilogue window) begins on a **keyframe plane** rather than on a mid-canvas latent cell. A tile is denoised as its own clip, whose first cell the model reads as a single pixel frame under the causal convention, so starting it on an 8-frame cell put content and shape in disagreement and ran RoPE time 7 frames ahead of the canvas. Cell 0 is now the plane at the last keyframe before the seam, the cells between it and the seam are pinned to the previous tile's output, and the tile resumes after the seam -- the last pinned cell ends exactly on it, so the seam keyframe is absorbed and needs no conditioning of its own. The prefix length follows the keyframe grid rather than a configured overlap, so the tile split takes no overlap at all and kept runs are disjoint.
- DFR temporal tiles and the spatial epilogue now hand over through a **pinned prefix** instead of synchronising on a shared keyframe. A non-first tile's lead-in cells are conditioned on the previous tile's finished output at strength 1, so they are that output at every step rather than a re-denoised approximation of it. Keyframe conditionings inside a pinned prefix are dropped, because the pinned cells hold the current stage's content while the plane holds an earlier stage's. Anchors from the resume point onward are unchanged, so the tile that owns a stretch still generates it against its keyframes.
- The DFR spatial epilogue runs one denoising pass per temporal window, sequentially, instead of one pass over the whole canvas with every temporal tile re-run inside each Euler step. Spatial tiling and its per-step blending are unchanged. Temporal windows were already seam-cut, rectangular and never blended into each other, so this costs the same and is what gives each window a finished predecessor to pin its lead-in to.

### Removed

- `DiffusionVideoDecoder.decode_video_with_keyframes`; use `decode_video(..., keyframes=)`.
- `DFROutput`; use `PipelineOutput`.
- `--checkpoint-path` and `--distilled-lora` on `ltx_pipelines.dfr_pipeline` and `ltx_pipelines.dfr_mgpu`; pass `--distilled-checkpoint-path` instead.
- `--num-generated-keyframes` on DFR; slot positions come from DFR's segment grid.
- `--temporal-upsample-rounds` on DFR; use `--temporal-upscalings`.

### Fixed

- Automatic decode tiling no longer OOMs in colour conversion on clips that looked like they had spare VRAM. The budget now reserves encoder memory for each chunk, so long videos decode in temporal pieces instead of as one oversize frame dump.
- A pipelines-only install (`ltx-core` / `ltx-pipelines`, no trainer) no longer fails to load Gemma 4 for missing torchvision.
- `uv sync` on macOS no longer fails looking for CUDA-only torch wheels. Macs use PyPI (MPS on Apple Silicon); Linux and Windows still use the CUDA 13.2 indexes.
- `--quantization fp8-cast` with a LoRA no longer crashes on pre-Ada GPUs such as RTX 3060 or A100.
- Blockwise FP8/FP6 quantization no longer fails on very long sequences or token counts that are not a multiple of 4.
- Multi-GPU DFR no longer crashes at tiling resolve, during carry-keyframe decode, or by dropping context at a temporal tile boundary.
- DFR image-to-video stills land on the correct frame after temporal upsampling (opening-frame `frame_idx=0` was already correct).
- `AUTO_TILING` is recognised after it crosses a process boundary, so multi-GPU pipelines resolve automatic tiling instead of rejecting the sentinel.

## 1.2.0 - 2026-08-11

Support for LTX 2.5

### Added

- Added support for newer LTX checkpoints, including Gemma 4 text encoders, checkpoint-driven architecture selection, compatibility checks between checkpoints and Gemma roots, and LTX 2.5 training workflows.
- Added diffusion-based video VAE decoding with single- and multi-GPU support, optional NATTEN acceleration, and `chunked_eager`, `chunked_compile`, `combined_compile`, and datacenter-Blackwell DSL optimization modes.
- Added caption-based automatic duration prediction for distilled, text-to-audio, and text/image-to-video pipelines. Use `--auto-duration MIN_SECONDS MAX_SECONDS`, or omit `--num-frames` with a compatible checkpoint.
- Added checkpoint-aware `--vae-checkpoint-path` overrides and `--diffvae-optimization` pipeline options.
- Added optional dedicated prompt-enhancement Gemma models through `--prompt-enhancer-gemma-root`, plus `--enhance-static-cache` for reusable enhancement KV caches.
- Added self-managed CUDA graph capture for compiled transformers with `capture=true`, alongside controls for dynamic sequence dimensions and perturbed-block recompilation.
- Added Euler ancestral diffusion sampling.
- Added checkpoint-aware size-, count-, and automatic-tiling APIs that support non-default VAE compression factors.
- Added reusable model-shell caching, configurable independently from checkpoint-weight caching, through `ModelRegistry`.
- Added disposable model support so cached module structure and non-persistent buffers can survive weight offloading.
- Added fused CuTe DSL diffusion-VAE kernels for datacenter Blackwell GPUs.
- Added model-version-aware image-conditioning compression defaults and expanded checkpoint metadata utilities.
- Added NVFP4 quantization, which cuts transformer memory use and speeds up inference on Blackwell GPUs. `--quantization nvfp4-cast` quantizes a BF16 checkpoint while loading it, and `--quantization nvfp4-prequant` loads an already-quantized NVFP4 checkpoint. Requires a Blackwell GPU and the `ltx-kernels` package.
- Added `DFRPipeline`, which raises output quality and holds detail together noticeably better on fast motion. It generates interior keyframes, adds a full-resolution detailing pass, and can run up to two temporal upsampling rounds for smoother movement. Adds `--detailing-lora`, `--temporal-upsampler-path`, `--temporal-upsample-rounds`, and `--num-generated-keyframes`.
- Added split-checkpoint loading, so you can download only the components a pipeline needs and mix them freely, such as a quantized transformer with a BF16 VAE. Each component has its own flag: `--transformer-path`, `--text-encoder-path`, `--video-vae-path`, `--audio-vae-path`, and `--duration-head-path`. Single-file checkpoints keep working through `--checkpoint-path` and `--distilled-checkpoint-path`.
- Added HDR conditioning and output, so generated video can go into a colour-grading or VFX pipeline without a lossy intermediate. `--hdr` selects the colour space for EXR conditioning and HDR encoding, which writes scene-linear EXR frames alongside a BT.2020/HLG video, and `--video-conditioning` accepts a conditioning video as either SDR or EXR frames.

### Changed

- Upgraded `ltx-core` to Transformers 5.8 or newer and added CUDA 13.2-compatible PyTorch, cuDNN, TorchCodec, NATTEN, and kernel-build dependency handling.
- Gemma loading now derives model structure from each local Hugging Face configuration, supports Gemma 3 and Gemma 4, and validates that the text encoder matches the LTX checkpoint.
- Gemma tokenization now consistently inserts a leading BOS token while avoiding duplicate BOS tokens for Gemma 3.
- Pipeline image-condition CRF defaults now come from checkpoint metadata: newer checkpoints use their trained value, while explicit CRF values remain unchanged.
- Pipeline constructors now accept a separate VAE checkpoint, diffusion-VAE optimization policy, prompt-enhancement model, and enhancement-cache setting.
- Direct construction of the former convolutional `VideoDecoder` now uses `ConvVideoDecoder`; `VideoDecoder` is now the common protocol implemented by convolutional and diffusion decoders.
- Trainer preprocessing, conditioning, validation, and latent decoding now derive spatial and temporal compression factors from checkpoint metadata instead of assuming 32x32x8.
- Trainer validation now uses separate video and audio CFG/STG controls, modality guidance, guidance rescaling, and checkpoint-aware frozen-modality handling.
- Trainer validation defaults now use 960x544x89 output, 24 fps, 30 inference steps, STG block 28, and a substantially expanded negative prompt.
- Legacy trainer validation guidance settings remain readable and are migrated automatically to their per-modality replacements.
- Model configurators now receive complete checkpoint metadata through `from_metadata`, enabling architecture and version-dependent construction.
- LoRA fusion and model reuse now preserve clean cached weights, avoid unnecessary tensor cloning, and support retained CPU weights.
- Transformer compilation modes that use CUDA graphs now require GPU-resident weights and fail early when used with incompatible offloading.

### Fixed

- Fixed multi-GPU video decoding applying pixel normalization twice, using a diffusion decoder's single-step forward path, and incorrectly rejecting temporal tiling configurations that do not actually split a worker's tile.
- Fixed multi-GPU prompt enhancement when encoding and enhancement use different Gemma roots, while retaining shared residency when they use the same root.
- Fixed Gemma 4 text encoding without a leading BOS token.
- Fixed trainer validation guidance for frozen audio or video, audio-only and video-only generation, cross-modal isolation, and modality-specific STG.
- Fixed trainer masks, spatial crops, reference-video alignment, and prefix/suffix validation for checkpoints with non-default VAE scale factors.
- Fixed LoRA fusion corrupting registry-cached weights and blockwise quantization failures when companion scale tensors remained on CPU.
- Fixed diffusion-VAE tiled decoding and multi-GPU blending to use checkpoint-specific geometry with lower peak host memory.
- Fixed 8-bit Gemma loading to resolve standard tokenizer assets and use architecture-agnostic Hugging Face model loading.
- Fixed CUDA builds using mismatched system toolkits or cuDNN sublibraries.
- Fixed diffusion-VAE decode tiling on Apple Silicon. The decode memory budget was probed only on CUDA, so on other backends it reported zero bytes available and no decode tile could ever fit, at any resolution. Automatic tiling now sizes itself from the Metal working set on MPS.

### Removed

- Removed `StateDictRegistry`; use `ModelRegistry`.
- Removed `SpatialTilingConfig` and `TemporalTilingConfig`; use `DimensionSizeConfig` with `TileSizeConfig`.
- Removed the old `TilingConfig` constructor fields `spatial_config` and `temporal_config`; use the new per-axis size/count tiling configuration.
- Removed `ltx_core.model.video_vae.tiling`; import tiling APIs from `ltx_core.tiling` or `ltx_core.model.video_vae`.
- Removed `GemmaTextEncoder`; use `LTXGemmaTextEncoder`.
- Removed `LTXVGemmaTokenizer`; use `LTXGemmaTokenizer`.
- Removed `GEMMA_LLM_KEY_OPS` and `GEMMA_MODEL_OPS`; use `get_gemma_ops()` for the selected Gemma root.
- Removed the hard-coded `Gemma3RopeScaling`, `Gemma3TextConfig`, `Gemma3VisionConfig`, `Gemma3ConfigData`, and `GEMMA3_CONFIG_FOR_LTX` definitions; Gemma configuration is now loaded from the model root through `gemma_model_config()`.
- Removed `ModelConfigurator.from_config()`; implement and call `from_metadata()`.
- Removed `DiffusionStage.model_context()` and `DiffusionStage.run()`; call the stage directly so it manages transformer construction and disposal.
- Removed `ltx_pipelines.utils.allocator_trim_strategy.AllocatorTrimStrategy`; import it from `ltx_core.allocator_trim_strategy`.
- Removed `ltx_trainer.training_strategies.VIDEO_SCALE_FACTORS`; use checkpoint-derived scale factors or `ltx_core.types.VIDEO_SCALE_FACTORS` when the legacy default is explicitly required.
- Removed trainer validation fields `guidance_scale`, `stg_scale`, and `stg_mode` from the current schema; use `video_cfg_scale`/`audio_cfg_scale` and `video_stg_scale`/`audio_stg_scale`. Legacy configuration files are migrated automatically.
- Removed the generic `gemma_i2v_system_prompt.txt` and `gemma_t2v_system_prompt.txt` names; use the `gemma3_*` or `gemma4_*` prompt files matching the encoder family.
- Removed `LipDubPipeline` and the `ltx_pipelines.lipdub` module; use `DubItPipeline` from `ltx_pipelines.dubit`.
