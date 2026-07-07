# Conditioning Types

Pipelines use different conditioning methods from [`ltx-core`](../../ltx-core/) for controlling generation. See the [ltx-core conditioning documentation](../../ltx-core/README.md#conditioning--control) for details.

## Image Conditioning

All pipelines support image conditioning, but with different methods:

- **Replacing Latents** ([`image_conditionings_by_replacing_latent`](../src/ltx_pipelines/utils/helpers.py)):
  - Replaces the latent at a specific frame with the encoded image
  - Strong control over specific frames

- **Guiding Latents** ([`image_conditionings_by_adding_guiding_latent`](../src/ltx_pipelines/utils/helpers.py)):
  - Adds the image as a guiding signal rather than replacing
  - Better for smooth interpolation between keyframes

## Video Conditioning

- **Video Conditioning** (ICLoraPipeline only):
  - Conditions on entire reference videos
  - Useful for video-to-video transformations
  - Uses `VideoConditionByKeyframeIndex` from [`ltx-core`](../../ltx-core/)
