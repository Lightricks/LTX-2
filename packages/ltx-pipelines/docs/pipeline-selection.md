# Pipeline Selection Guide

## Quick Decision Tree

```text
Do you have an existing video to modify?
├─ YES → Use RetakePipeline (regenerate a specific time region)
│
Do you have an audio file to drive generation?
├─ YES → Use A2VidPipelineTwoStage (audio-to-video)
│
Do you need HDR / EXR I/O?
├─ YES → Do you have EXR stills or EXR-frame folders for a standard pipeline?
│  ├─ YES → Use Distilled / TI2V / Retake / IC-LoRA with `--hdr {SRGB_LINEAR,ACESCG,ACESCCT,ACES2065_1}`
│  │         (see [HDR Support](hdr.md); EXR + BT.2020/HLG output)
│  └─ NO → Use HDRICLoraPipeline (SDR video → ACEScct HDR; HLG master + EXR)
│
Do you need to condition on existing images/videos?
├─ YES → Do you have reference videos for video-to-video?
│  ├─ YES → Use ICLoraPipeline
│  └─ NO → Do you have multiple keyframe images to interpolate?
│     ├─ YES → Use TI2VidTwoStagesPipeline with `--image … guide` bindings
│     └─ NO → Image-to-video (one still)
│        ├─ Production quality / higher fps → DFRPipeline
│        └─ Guided CFG/STG two-stage → TI2VidTwoStagesPipeline
│
└─ NO → Text-to-video only
   ├─ Do you need fastest inference?
   │  └─ YES → Use DistilledPipeline (starting point; 8 predefined sigmas)
   │
   ├─ Do you need production quality, or a higher frame rate than the base model?
   │  └─ YES → Use DFRPipeline (distilled checkpoint + detailing pass + optional temporal 2x/4x)
   │
   └─ Guided two-stage with CFG/STG
      └─ Use TI2VidTwoStagesPipeline
```

> **Note:** [`TI2VidOneStagePipeline`](../src/ltx_pipelines/ti2vid_one_stage.py) is primarily for educational purposes. For production quality, use [`DFRPipeline`](../src/ltx_pipelines/dfr_pipeline.py). For guided CFG/STG two-stage, use [`TI2VidTwoStagesPipeline`](../src/ltx_pipelines/ti2vid_two_stages.py) or [`TI2VidTwoStagesHQPipeline`](../src/ltx_pipelines/ti2vid_two_stages_hq.py). For editing existing videos, use [`RetakePipeline`](../src/ltx_pipelines/retake.py). See [Running DFR](pipelines.md#running-dfr) for DFR resolution, fps, and memory notes.

## Features Comparison

| Pipeline | Stages | [Multimodal Guidance](multimodal-guidance.md) | Upsampling | Conditioning | Best For |
| -------- | ------ | --- | ---------- | ------------- | -------- |
| [**DistilledPipeline**](pipelines.md#4-distilledpipeline) | 2 | ❌ | ✅ | Image | Fastest inference (starting point) |
| [**DFRPipeline**](pipelines.md#12-dfrpipeline) | 2 (+ optional temporal rounds) | ❌ | ✅ spatial + optional temporal 2x | Image + [generated keyframes](conditioning.md#generated-keyframe-slots) | **Production quality**; higher effective frame rate |
| [**TI2VidTwoStagesPipeline**](pipelines.md#1-ti2vidtwostagespipeline) | 2 | ✅ | ✅ | Image + keyframes | Guided two-stage with CFG/STG; animation and interpolation |
| [**TI2VidTwoStagesHQPipeline**](pipelines.md#2-ti2vidtwostageshqpipeline) | 2 | ✅ | ✅ | Image | Same guided flow, res_2s sampler |
| [**TI2VidOneStagePipeline**](pipelines.md#3-ti2vidonestagepipeline) | 1 | ✅ | ❌ | Image | Educational, prototyping |
| [**ICLoraPipeline**](pipelines.md#5-iclorapipeline) | 2 | ✅ | ✅ | Image + Video | Video-to-video transformations |
| [**A2VidPipelineTwoStage**](pipelines.md#7-a2vidpipelinetwostage) | 2 | ✅ | ✅ | Audio + Image | Audio-driven video generation |
| [**RetakePipeline**](pipelines.md#8-retakepipeline) | 1 | ✅ | ❌ | Source Video | Regenerating a time region of a video |
| [**HDRICLoraPipeline**](pipelines.md#9-hdriclorapipeline) | 1 | ❌ | ❌ | Video | SDR-to-HDR with an HDR IC-LoRA (HLG master + EXR via `--exr-colorspace`); see also native [`--hdr`](hdr.md) |
| [**DubItPipeline**](pipelines.md#10-dubitpipeline) | 2 | ✅ | ✅ | Video + Audio | Dub-It with audio ref conditioning |
| [**T2AOneStagePipeline**](pipelines.md#11-t2aonestagepipeline) | 1 | Audio only | ❌ | None (text) | Text-to-audio (audio-only output, no video) |

See [Available Pipelines](pipelines.md) for a full description of each.
