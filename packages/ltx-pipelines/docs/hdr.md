# HDR Support

Native **`--hdr`** support: pass EXR stills / EXR-frame folders into the standard generation and editing pipelines. Decode writes half-float EXR frames plus a BT.2020/HLG master.

For the separate video-to-video HDR IC-LoRA path (ACEScct SDR→HDR, optional DFR-style seam keyframes), see [HDRICLoraPipeline](pipelines.md#9-hdriclorapipeline).

## Native HDR (`--hdr`)

### Colour spaces

CLI flag: `--hdr {SRGB_LINEAR,ACESCG,ACESCCT,ACES2065_1}` (enum [`EXRColorSpace`](../src/ltx_pipelines/utils/media_io/color_config.py)).

| Value | Source meaning | Load behaviour |
| ----- | -------------- | -------------- |
| `SRGB_LINEAR` | Scene-linear Rec.709 / sRGB-tagged EXR | Compress to ACEScct for the VAE |
| `ACESCG` | Scene-linear ACEScg (AP1) EXR | Compress to ACEScct for the VAE |
| `ACESCCT` | Already ACEScct log codes | Pass through (no load-time transfer) |
| `ACES2065_1` | Scene-linear ACES2065-1 (AP0) EXR | Compress to ACEScct for the VAE |

`None` (omitting `--hdr`) means SDR: ordinary image/video inputs and H.264 output.

### Inputs

- EXR **still**: `--image path/to/plate.exr … --hdr SRGB_LINEAR` (same image-conditioning syntax as SDR; path ends in `.exr`).
- EXR **sequence**: a directory of `*.exr` frames for `--video-conditioning` / `--video-path` (retake). Frames are sorted lexicographically.
- Retake EXR folders require `--frame-rate` (EXR dirs have no container fps). Video-file retake forbids `--frame-rate`.
- Any EXR still or folder **requires** `--hdr`. Mixing EXR and non-EXR among `--image` / `--video-conditioning` is rejected.
- **Dub-It has no `--hdr` flag** — SDR video reference only. EXR folders / `.exr` files are rejected for `--reference-video` (fps and audio come from the container).

### Outputs

When `--hdr` is set, `encode_video` writes:

- **EXR**: half-float frames under `<output_stem>_<color_space>_exr/frame_XXXXX.exr`, tagged to match the chosen colour space (linear or ACEScct log). `SRGB_LINEAR`, `ACESCG`, and `ACESCCT` use ZIP compression; `ACES2065_1` writes uncompressed SMPTE ST 2065-4 ACES containers (`compression="none"`, `acesImageContainerFlag=1`, `adoptedNeutral`).
- **HLG master**: 10-bit BT.2020/HLG HEVC (`libx265`) at the `--output-path` (Rec.709 scene-linear → HLG).

VAE decode runs in **float32** for HDR (SDR stays bf16).

### Example — distilled I2V from an EXR still

Uses the LTX-2.5 split layout (same paths as the public README quick start). The diffusion video VAE needs the `natten` extra (`uv sync --extra natten`).

```bash
uv run python -m ltx_pipelines.distilled \
    --transformer-path       models/ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
    --text-encoder-path      models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
    --video-vae-path         models/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors \
    --audio-vae-path         models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors \
    --spatial-upsampler-path models/ltx-2.5/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
    --num-frames 121 \
    --prompt "Rain starts falling on the cow in the pasture" \
    --image path/to/cow_pasture.exr 0 1.0 \
    --hdr SRGB_LINEAR \
    --output-path output/cow_rain.mp4
```

Produces `output/cow_rain.mp4` (HLG HEVC) and `output/cow_rain_exr/*.exr`.

### Example — retake from an EXR sequence

Retake is single-stage (no spatial upsampler). Same LTX-2.5 split components otherwise:

```bash
uv run python -m ltx_pipelines.retake \
    --transformer-path  models/ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
    --text-encoder-path models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
    --video-vae-path    models/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors \
    --audio-vae-path    models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors \
    --video-path path/to/source_exr_frames/ \
    --frame-rate 24 \
    --start-time 0.67 --end-time 5.0 \
    --prompt "The cow morphs into a moose" \
    --hdr SRGB_LINEAR \
    --output-path output/retake.mp4
```

### Python API

```python
from ltx_pipelines.utils.media_io import EXRColorSpace, encode_video, resolve_hdr_color_space, vae_dtype_for_hdr

hdr = resolve_hdr_color_space(images=images, video_paths=[], hdr=EXRColorSpace.SRGB_LINEAR)
vae_dtype = vae_dtype_for_hdr(hdr, default=torch.bfloat16)

# After pipeline decode (iterator of [F,H,W,C] chunks):
encode_video(video_chunks, fps, audio, output_path, num_chunks, color_space=hdr)
```

Pipelines that accept image / video conditioning expose an optional `color_space: EXRColorSpace | None` on `__call__` (wired from CLI `--hdr`).

## Constraints (summary)

- EXR input ⇒ `--hdr` required.
- All conditioning media for a run must be all-EXR or all-SDR (no mix).
- Retake: `--frame-rate` required for EXR folders, forbidden for video files.
- Dub-It: no `--hdr`; reference must be an SDR video file (not EXR).
- DiffVAE / NATTEN: HDR decode still uses the configured VAE; install the `natten` extra when using a diffusion video VAE (same as SDR).
- Keyframe-aware decode (the IC-LoRA path's default; disable with `--no-keyframes`) needs a VAE whose `decoder.type_emb` is non-zero. A checkpoint without it loads and decodes fine — zeros are synthesized — but the keyframe stream becomes a silent no-op. See [HDRICLoraPipeline](pipelines.md#9-hdriclorapipeline).
