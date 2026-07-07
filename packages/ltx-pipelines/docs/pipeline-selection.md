# Pipeline Selection Guide

## Quick Decision Tree

```text
Do you have an existing video to modify?
├─ YES → Use RetakePipeline (regenerate a specific time region)
│
Do you have an audio file to drive generation?
├─ YES → Use A2VidPipelineTwoStage (audio-to-video)
│
Do you need HDR output (linear float frames for EXR / tonemapping)?
├─ YES → Use HDRICLoraPipeline (video-to-video with LogC3 inverse decode)
│
Do you need to condition on existing images/videos?
├─ YES → Do you have reference videos for video-to-video?
│  ├─ YES → Use ICLoraPipeline
│  └─ NO → Do you have multiple keyframe images to interpolate?
│     ├─ YES → Use KeyframeInterpolationPipeline
│     └─ NO → Use TI2VidTwoStagesPipeline (image conditioning only)
│
└─ NO → Text-to-video only
   ├─ Do you need best quality?
   │  └─ YES → Use TI2VidTwoStagesPipeline (recommended for production)
   │
   └─ Do you need fastest inference?
      └─ YES → Use DistilledPipeline (with 8 predefined sigmas)
```

> **Note:** [`TI2VidOneStagePipeline`](../src/ltx_pipelines/ti2vid_one_stage.py) is primarily for educational purposes. For best quality, use two-stage pipelines ([`TI2VidTwoStagesPipeline`](../src/ltx_pipelines/ti2vid_two_stages.py), [`TI2VidTwoStagesHQPipeline`](../src/ltx_pipelines/ti2vid_two_stages_hq.py), [`ICLoraPipeline`](../src/ltx_pipelines/ic_lora.py), [`KeyframeInterpolationPipeline`](../src/ltx_pipelines/keyframe_interpolation.py), [`A2VidPipelineTwoStage`](../src/ltx_pipelines/a2vid_two_stage.py), or [`DistilledPipeline`](../src/ltx_pipelines/distilled.py)). For editing existing videos, use [`RetakePipeline`](../src/ltx_pipelines/retake.py).

## Features Comparison

| Pipeline | Stages | [Multimodal Guidance](multimodal-guidance.md) | Upsampling | Conditioning | Best For |
| -------- | ------ | --- | ---------- | ------------- | -------- |
| [**TI2VidTwoStagesPipeline**](pipelines.md#1-ti2vidtwostagespipeline) | 2 | ✅ | ✅ | Image | **Production quality** (recommended) |
| [**TI2VidTwoStagesHQPipeline**](pipelines.md#2-ti2vidtwostageshqpipeline) | 2 | ✅ | ✅ | Image | Same as above, res_2s sampler (higher quality) |
| [**TI2VidOneStagePipeline**](pipelines.md#3-ti2vidonestagepipeline) | 1 | ✅ | ❌ | Image | Educational, prototyping |
| [**DistilledPipeline**](pipelines.md#4-distilledpipeline) | 2 | ❌ | ✅ | Image | Fastest inference (8 sigmas) |
| [**ICLoraPipeline**](pipelines.md#5-iclorapipeline) | 2 | ✅ | ✅ | Image + Video | Video-to-video transformations |
| [**KeyframeInterpolationPipeline**](pipelines.md#6-keyframeinterpolationpipeline) | 2 | ✅ | ✅ | Keyframes | Animation, interpolation |
| [**A2VidPipelineTwoStage**](pipelines.md#7-a2vidpipelinetwostage) | 2 | ✅ | ✅ | Audio + Image | Audio-driven video generation |
| [**RetakePipeline**](pipelines.md#8-retakepipeline) | 1 | ✅ | ❌ | Source Video | Regenerating a time region of a video |
| [**HDRICLoraPipeline**](pipelines.md#9-hdriclorapipeline) | 2 | ❌ | ✅ | Video | HDR video-to-video (linear float output for EXR) |
| [**LipDubPipeline**](pipelines.md#10-lipdubpipeline) | 2 | ✅ | ✅ | Video + Audio | Lip dubbing with audio ref conditioning |
| [**T2AOneStagePipeline**](pipelines.md#11-t2aonestagepipeline) | 1 | Audio only | ❌ | None (text) | Text-to-audio (audio-only output, no video) |

See [Available Pipelines](pipelines.md) for a full description of each.
