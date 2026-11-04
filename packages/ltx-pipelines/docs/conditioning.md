# Conditioning Types

Pipelines use different conditioning methods from [`ltx-core`](../../ltx-core/) for controlling generation. See the [ltx-core conditioning documentation](../../ltx-core/README.md#conditioning--control) for details.

## Image Conditioning

Stills are passed as [`ImageConditioningInput`](../src/ltx_pipelines/utils/types.py) values with an
[`ImageReplace`](../src/ltx_pipelines/utils/types.py) or [`ImageGuide`](../src/ltx_pipelines/utils/types.py)
binding. [`image_conditionings()`](../src/ltx_pipelines/utils/helpers.py) encodes each still and
emits the matching core conditioning item (`VideoConditionByLatentIndex` or
`VideoConditionByKeyframeIndex`).

`ImageConditioningInput` is a pixel-space API: every `frame_idx` is a zero-based index on the
decoded output video's frame timeline, never an index into the VAE latent. This applies equally to
`ImageConditioningInput.at()`, `.replace()`, `.guide()`, `ImageReplace`, `ImageGuide`, and the
`--image` (optional trailing `replace` / `guide` binding keyword) CLI flags. Chunking and temporal rebasing also
map these values in pixel space.

`ImageGuide` accepts any frame in the clip. `ImageReplace` must point to the start of a causal VAE
latent: pixel frame `0`, or `1 + 8 * k` (`1`, `9`, `17`, ...). The pipeline converts that
pixel-frame index to a latent index only when it creates the core conditioning item:

```python
opening = ImageConditioningInput.replace("opening.png", frame_idx=0, strength=1.0)
later_replacement = ImageConditioningInput.replace("later.png", frame_idx=9, strength=1.0)
guide = ImageConditioningInput.guide("guide.png", frame_idx=80, strength=0.8)
```

## Video Conditioning

- **Video Conditioning** (ICLoraPipeline only):
  - Conditions on entire reference videos
  - Useful for video-to-video transformations
  - Uses `VideoConditionByKeyframeIndex` from [`ltx-core`](../../ltx-core/)

## HDR / EXR Conditioning

Standard pipelines accept OpenEXR stills (`.exr`) and EXR-frame folders as image / video conditioning when you pass `--hdr {SRGB_LINEAR,ACESCG,ACESCCT,ACES2065_1}` to declare the source colour space. HDR runs decode to half-float EXR frames plus a BT.2020/HLG master.

See [HDR Support](hdr.md) for colour-space semantics, CLI constraints (no EXR/SDR mix, retake `--frame-rate` rules), and examples.
## Generated Keyframe Slots

Generated keyframes are extra frames the model produces *in addition* to the regular frame grid, at
interior positions you choose. Each slot is an empty, fully-denoised token slot appended to the
sequence; because a slot carries a single pixel frame rather than the eight a normal latent frame
carries, it relaxes the effective temporal compression at that position — useful where motion is too
fast for the base temporal resolution.

Uses `VideoGeneratedKeyframeSlots` from [`ltx-core`](../../ltx-core/).

**Requirements.** The checkpoint's transformer config must set `use_keyframes_abs_pos_embedding`.
LTX-2.5 checkpoints support this; on an older checkpoint the pipeline raises rather than silently
ignoring the request, since each slot costs a meaningful share of the token budget.

**Cost.** One latent frame of tokens per keyframe, yielding one pixel frame instead of eight. At
512x768 / 241 frames, 5 keyframes is about +16% tokens (~1.35x attention cost); at 1088x1920 / 121
frames it is about +31% (~1.72x). Budget for this before raising the count.

**Where it applies.** First stage only, on `TI2VidOneStagePipeline`, `TI2VidTwoStagesPipeline`,
`TI2VidTwoStagesHQPipeline`, `DistilledPipeline` and their multi-GPU runners. The second stage does
not need slots — the effect is already baked into the stage-1 latent.
[`DFRPipeline`](pipelines.md#12-dfrpipeline) is the exception: it re-attaches seeded slots in stage 2
and invents new ones per tile during temporal rounds.

**Usage.** Pass `generated_keyframes` to `__call__`: an `int` requests that many evenly spaced
interior keyframes (both endpoints excluded), a sequence gives explicit frame indices. On the CLI the
flag is `--num-generated-keyframes N` (default `0`, off). Pass `decode_with_keyframes=True` (CLI:
`--decode-with-keyframes`) to decode through the keyframe-aware DiffVAE path using those slots as
anchors. It is opt-in per pipeline rather than a shared flag, so only the CLIs listed above
advertise both flags -- `DFRPipeline` derives its own slot positions, always keyframe-decodes, and
does not accept either flag.

**Reading the keyframes back.** Generated keyframes land in `LatentState.generated_keyframes` with
shape `(B, C, K, H, W)`. Decode each one as a standalone single-frame clip: a K-frame causal decode
would blend slots that were never temporally adjacent.
