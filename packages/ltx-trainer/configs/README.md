# Training Configs

Start from the example closest to your task, then set model/data paths, validation inputs, and training settings.
Each file uses the built-in **flexible** strategy (`name: "flexible"`).

See [Training Modes](../docs/training-modes.md) for modality/condition behavior and dataset requirements.
For LTX-2/LTX-2.3, set the unified checkpoint and matching Gemma directory. For LTX-2.5 split packs, set the
transformer, packed LTX-specific text encoder, and VAE paths for the modalities used; see
[model configuration](../docs/configuration-reference.md#model). All examples include the optional VAE fields.

## Training Modes

| Mode | Config |
|------|--------|
| **Text-to-Video (T2V)** | [`t2v_lora.yaml`](./t2v_lora.yaml), [`t2v_lora_low_vram.yaml`](./t2v_lora_low_vram.yaml) (low VRAM) |
| **Image-to-Video (I2V)** | [`i2v_lora.yaml`](./i2v_lora.yaml) |
| **Video Extension** | [`video_extend_lora.yaml`](./video_extend_lora.yaml) (forward), [`video_suffix_lora.yaml`](./video_suffix_lora.yaml) (backward) |
| **Video-to-Video IC-LoRA (V2V)** | [`v2v_ic_lora.yaml`](./v2v_ic_lora.yaml) |
| **Audio-to-Video (A2V)** | [`a2v_lora.yaml`](./a2v_lora.yaml) |
| **Video-to-Audio / Foley (V2A)** | [`v2a_lora.yaml`](./v2a_lora.yaml) |
| **Video Inpainting** | [`video_inpainting_lora.yaml`](./video_inpainting_lora.yaml) |
| **Video Outpainting** | [`video_outpainting_lora.yaml`](./video_outpainting_lora.yaml) |
| **Text-to-Audio (T2A)** | [`t2a_lora.yaml`](./t2a_lora.yaml) |
| **Audio Extension** | [`audio_extend_lora.yaml`](./audio_extend_lora.yaml) (forward), [`audio_suffix_lora.yaml`](./audio_suffix_lora.yaml) (backward) |
| **Audio Inpainting** | [`audio_inpainting_lora.yaml`](./audio_inpainting_lora.yaml) |
| **Audio-to-Audio IC-LoRA (A2A)** | [`a2a_ic_lora.yaml`](./a2a_ic_lora.yaml) |
| **Audio-Video IC-LoRA (AV2AV)** | [`av2av_ic_lora.yaml`](./av2av_ic_lora.yaml) |

The [`accelerate/`](./accelerate) directory holds the Accelerate launch configs (FSDP, DDP) for multi-GPU training.
