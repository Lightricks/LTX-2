# 🎛️ Multimodal Guidance

LTX-2 pipelines use **multimodal guidance** to steer the diffusion process for both video and audio modalities. Each modality (video, audio) has its own guider with independent parameters, allowing fine-grained control over generation quality and adherence to prompts.

## Guidance Parameters

The `MultiModalGuiderParams` dataclass controls guidance behavior:

| Parameter | Description |
| --------- | ----------- |
| `cfg_scale` | **Classifier-Free Guidance** scale. Higher values make the output adhere more strongly to the text prompt. Typical values: 2.0–5.0. Set to **1.0** to disable. |
| `stg_scale` | **Spatio-Temporal Guidance** scale. Controls perturbation-based guidance for improved temporal coherence. Typical values: 0.5–1.5. Set to **0.0** to disable. |
| `stg_blocks` | Which transformer blocks to perturb for STG (e.g., `[29]` for the last block). Set to **`[]`** to disable STG. |
| `rescale_scale` | Rescales the guided prediction to match the variance of the conditional prediction. Helps prevent over-saturation. Typical values: 0.5–0.7. Set to **0.0** to disable. |
| `modality_scale` | **Modality CFG** scale. Steers the model away from unsynced video and audio results, improving audio-visual coherence. Set to **1.0** to disable. |
| `skip_step` | Skip guidance every N steps. Can speed up inference with minimal quality loss. Set to **0** to disable (never skip). |

## How It Works

The multimodal guider combines three guidance signals during each denoising step:

1. **CFG (Text Guidance)**: Steers generation toward the text prompt by computing `(cond - uncond_text)`.
2. **STG (Perturbation Guidance)**: Improves structural coherence by perturbing specific transformer blocks and steering away from the perturbed prediction.
3. **Modality CFG**: For joint audio-video generation, steers the model away from unsynced video and audio results.

## Example Configuration

```python
from ltx_core.components.guiders import MultiModalGuiderParams

# Video guider: moderate CFG, STG enabled, modality isolation
video_guider_params = MultiModalGuiderParams(
    cfg_scale=3.0,
    stg_scale=1.0,
    rescale_scale=0.7,
    modality_scale=3.0,
    stg_blocks=[29],
)

# Audio guider: higher CFG for stronger prompt adherence
audio_guider_params = MultiModalGuiderParams(
    cfg_scale=7.0,
    stg_scale=1.0,
    rescale_scale=0.7,
    modality_scale=3.0,
    stg_blocks=[29],
)
```

> **Tip:** Start with the default values from [`constants.py`](../src/ltx_pipelines/utils/constants.py) and adjust based on your use case. Higher `cfg_scale` = stronger prompt adherence but potentially less natural motion; higher `stg_scale` = better temporal coherence but slower inference (requires extra forward passes).
>
> **Tip:** When generating video with audio, set `modality_scale` > 1.0 (e.g., 3.0) to improve audio-visual sync. If generating video-only, set it to 1.0 to disable.
