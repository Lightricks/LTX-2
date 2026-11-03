"""Alpha-gen: one-stage IC-LoRA generation on the full model with multimodal guidance.
The diffusion pass matches ``TI2VidOneStagePipeline``: one Euler schedule, a
positive and negative prompt, and a ``FactoryGuidedDenoiser``. The extra input
is an IC-LoRA reference video, appended as ``VideoConditionByReferenceLatent``.
"""

import logging
from dataclasses import replace

import torch

from ltx_core.components.guiders import MultiModalGuiderParams, create_multimodal_guider_factory
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.conditioning import ConditioningItem
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import ModelRegistry
from ltx_core.model.video_vae import AUTO_TILING, AutoTiling, TilingConfig, VideoEncoder, get_video_chunks_number
from ltx_core.model.video_vae.transformer import DiffVAEMode
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import VideoPixelShape
from ltx_pipelines.iclora_utils import (
    append_ic_lora_reference_video_conditionings,
    read_lora_reference_downscale_factor,
    read_lora_reference_temporal_scale_factor,
)
from ltx_pipelines.utils.args import VideoConditioningAction, default_1_stage_arg_parser, resolve_cli_params
from ltx_pipelines.utils.blocks import (
    DiffusionStage,
    DurationPredictor,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    require_num_frames_source,
    resolve_num_frames,
)
from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    create_initial_video_latent,
    ensure_tiling_config,
    get_device,
    image_conditionings,
)
from ltx_pipelines.utils.media_io import EXRColorSpace, encode_video, resolve_hdr_color_space, vae_dtype_for_hdr
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.types import (
    DEFAULT_AUTO_DURATION,
    AutoDuration,
    ImageConditioningInput,
    LoadedImageConditioning,
    ModalitySpec,
    OffloadMode,
    PipelineOutput,
    VideoAudio,
)

logger = logging.getLogger(__name__)

NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"


def alpha_gen_guider_params(base: MultiModalGuiderParams, *, cfg_scale: float) -> MultiModalGuiderParams:
    """Alpha-gen guidance: ``cfg_scale``, STG off, rescale 0.7, no cross-modal guidance."""
    return replace(base, cfg_scale=cfg_scale, stg_scale=0.0, rescale_scale=0.7, modality_scale=1.0)


class AlphaGenPipeline:
    """Single-stage IC-LoRA pipeline on a full checkpoint."""

    def __init__(
        self,
        model_paths: ModelPaths,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: ModelRegistry | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
        diffvae_optimization: DiffVAEMode = DiffVAEMode.CHUNKED_EAGER,
    ) -> None:
        self.dtype = torch.bfloat16
        registry = registry or ModelRegistry()
        self.device = device or get_device()
        self.prompt_encoder = PromptEncoder(
            model_paths, dtype=self.dtype, device=self.device, registry=registry, offload_mode=offload_mode
        )
        self.image_conditioner = ImageConditioner(
            model_paths.video_vae(), dtype=self.dtype, device=self.device, registry=registry
        )
        self.stage = DiffusionStage.from_checkpoint(
            model_paths.transformer(),
            dtype=self.dtype,
            device=self.device,
            loras=tuple(loras),
            quantization=quantization,
            registry=registry,
            offload_mode=offload_mode,
        )
        self.video_decoder = VideoDecoder(
            model_paths.video_vae(),
            dtype=self.dtype,
            device=self.device,
            registry=registry,
            diffvae_optimization=diffvae_optimization,
        )
        self.duration_predictor = DurationPredictor.from_checkpoint(
            checkpoint_path=model_paths.duration_head_path, dtype=self.dtype, device=self.device
        )
        self.reference_downscale_factor, self.reference_temporal_scale_factor = _reference_scale_factors(loras)

    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams,
        images: list[ImageConditioningInput],
        video_conditioning: list[tuple[str, float]],
        num_frames: int | AutoDuration = DEFAULT_AUTO_DURATION,
        vae_dtype: torch.dtype | None = None,
        color_space: EXRColorSpace | None = None,
        tiling_config: TilingConfig | AutoTiling | None = AUTO_TILING,
    ) -> PipelineOutput:
        """Generate one clip conditioned on ``video_conditioning`` (``(path, strength)`` pairs, at least one)."""
        if not video_conditioning:
            raise ValueError("Alpha-gen requires at least one reference video")
        require_num_frames_source(num_frames, self.duration_predictor)
        images = self.image_conditioner.load_conditionings(images, color_space)
        assert_resolution(height=height, width=width, is_two_stage=False)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        ctx_p, ctx_n = self.prompt_encoder([prompt, negative_prompt])
        num_frames = resolve_num_frames(
            num_frames,
            self.duration_predictor,
            video_encoding=ctx_p.video_encoding,
            audio_encoding=None,
            frame_rate=frame_rate,
        )
        tiling_config = ensure_tiling_config(
            tiling_config,
            vae_checkpoint_path=self.video_decoder.checkpoint_path,
            video_shape=VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=frame_rate),
            diffvae_optimization=self.video_decoder.diffvae_optimization,
            device=self.device,
            vae_dtype=vae_dtype or self.dtype,
        )
        conditionings = self.image_conditioner(
            lambda enc: self._conditionings(enc, images, video_conditioning, height, width, num_frames, color_space)
        )

        video_latent = create_initial_video_latent(
            pixel_shape=VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=frame_rate),
            device=self.device,
            dtype=self.dtype,
            scale_factors=self.stage.video_scale_factors,
        )
        video_state, _ = self.stage(
            denoiser=FactoryGuidedDenoiser(
                video_guider_factory=create_multimodal_guider_factory(
                    params=video_guider_params, negative_context=ctx_n.video_encoding
                ),
            ),
            sigmas=LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=self.device),
            noiser=GaussianNoiser(generator=generator),
            modalities=VideoAudio(
                video=ModalitySpec(
                    latent=video_latent,
                    conditioning_fps=frame_rate,
                    context=ctx_p.video_encoding,
                    conditionings=conditionings,
                ),
            ),
        )

        decoded_video = self.video_decoder(
            video_state.latent, tiling_config, generator=generator, dtype=vae_dtype or self.dtype
        )
        return PipelineOutput(decoded_video, None, num_frames, tiling_config)

    def _conditionings(
        self,
        video_encoder: VideoEncoder,
        images: list[LoadedImageConditioning],
        video_conditioning: list[tuple[str, float]],
        height: int,
        width: int,
        num_frames: int,
        color_space: EXRColorSpace | None,
    ) -> list[ConditioningItem]:
        conditionings = image_conditionings(
            images=images,
            height=height,
            width=width,
            video_encoder=video_encoder,
            dtype=self.dtype,
            device=self.device,
        )
        append_ic_lora_reference_video_conditionings(
            conditionings,
            video_conditioning,
            height=height,
            width=width,
            num_frames=num_frames,
            video_encoder=video_encoder,
            dtype=self.dtype,
            device=self.device,
            reference_downscale_factor=self.reference_downscale_factor,
            reference_temporal_scale_factor=self.reference_temporal_scale_factor,
            conditioning_attention_strength=1.0,
            conditioning_attention_mask=None,
            tiling_config=None,
            color_space=color_space,
        )
        return conditionings


def _reference_scale_factors(loras: list[LoraPathStrengthAndSDOps]) -> tuple[int, int]:
    """Spatial and temporal reference scales from LoRA metadata.
    A factor of 1 means the LoRA left it unset. Any values other than 1 must match.
    """
    spatial = {read_lora_reference_downscale_factor(lora.path) for lora in loras} - {1}
    temporal = {read_lora_reference_temporal_scale_factor(lora.path) for lora in loras} - {1}
    if len(spatial) > 1:
        raise ValueError(
            f"LoRA reference_downscale_factor values disagree: {sorted(spatial)}. "
            "1 means unset and is ignored; every set value must match."
        )
    if len(temporal) > 1:
        raise ValueError(
            f"LoRA reference_temporal_scale_factor values disagree: {sorted(temporal)}. "
            "1 means unset and is ignored; every set value must match."
        )
    return next(iter(spatial), 1), next(iter(temporal), 1)


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    detected = resolve_cli_params()
    params = replace(
        detected,
        video_guider_params=alpha_gen_guider_params(detected.video_guider_params, cfg_scale=1.0),
    )
    parser = default_1_stage_arg_parser(params=params, supports_auto_duration=True)
    parser.set_defaults(negative_prompt=NEGATIVE_PROMPT)
    parser.add_argument(
        "--video-conditioning",
        action=VideoConditioningAction,
        nargs=2,
        metavar=("PATH", "STRENGTH"),
        required=True,
        help="IC-LoRA reference video and strength. Example: --video-conditioning guide.mp4 1.0",
    )
    args = parser.parse_args()

    pipeline = AlphaGenPipeline(
        model_paths=args.model_paths,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
        offload_mode=args.offload_mode,
        diffvae_optimization=args.diffvae_optimization,
    )
    hdr = resolve_hdr_color_space(
        images=args.images, video_paths=[path for path, _ in args.video_conditioning], hdr=args.hdr
    )
    result = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        video_guider_params=MultiModalGuiderParams(
            cfg_scale=args.video_cfg_guidance_scale,
            stg_scale=args.video_stg_guidance_scale,
            rescale_scale=args.video_rescale_scale,
            modality_scale=args.a2v_guidance_scale,
            skip_step=args.video_skip_step,
            stg_blocks=args.video_stg_blocks,
        ),
        images=args.images,
        video_conditioning=args.video_conditioning,
        vae_dtype=vae_dtype_for_hdr(hdr, torch.bfloat16),
        color_space=hdr,
    )
    encode_video(
        video=result.video,
        fps=args.frame_rate,
        audio=result.audio,
        output_path=args.output_path,
        video_chunks_number=get_video_chunks_number(result.num_frames, result.tiling_config),
        color_space=hdr,
    )


if __name__ == "__main__":
    main()
