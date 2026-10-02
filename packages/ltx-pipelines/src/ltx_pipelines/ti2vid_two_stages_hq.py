import logging
from collections.abc import Sequence

import torch

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.conditioning import VideoGeneratedKeyframeSlots
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae import AUTO_TILING, AutoTiling, TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae.transformer import DiffVAEMode
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import VideoLatentShape, VideoPixelShape
from ltx_pipelines.utils.args import (
    add_generated_keyframes_arg,
    add_keyframe_decode_arg,
    hq_2_stage_arg_parser,
)
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    DurationPredictor,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
    require_num_frames_source,
    resolve_num_frames,
)
from ltx_pipelines.utils.constants import (
    LTX_2_3_HQ_PARAMS,
    STAGE_2_DISTILLED_SIGMAS,
)
from ltx_pipelines.utils.denoisers import GuidedDenoiser, SimpleDenoiser
from ltx_pipelines.utils.helpers import (
    assert_generated_keyframes_request,
    assert_resolution,
    combined_image_conditionings,
    create_initial_av_latents,
    decode_keyframes_from_slots,
    ensure_tiling_config,
    generated_keyframe_conditionings,
    get_device,
    resolve_generated_keyframes,
    tiling_scale_factors_for_vae,
)
from ltx_pipelines.utils.media_io import (
    EXRColorSpace,
    encode_video,
    resolve_hdr_color_space,
    vae_dtype_for_hdr,
)
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.samplers import res2s_audio_video_denoising_loop
from ltx_pipelines.utils.types import (
    DEFAULT_AUTO_DURATION,
    AutoDuration,
    ImageConditioningInput,
    ModalitySpec,
    OffloadMode,
    PipelineOutput,
    VideoAudio,
)


class TI2VidTwoStagesHQPipeline:
    """
    Two-stage text/image-to-video generation pipeline using the res_2s sampler.
    Same structure as :class:`TI2VidTwoStagesPipeline`: stage 1 generates video at
    half of the target resolution with CFG guidance (assuming  full model is used),
    then Stage 2 upsamples by 2x and refines using a distilled LoRA for higher
    quality output.
    Uses the res_2s second-order sampler instead of Euler, allowing fewer
    steps for comparable quality. Supports optional image conditioning via
    the images parameter.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_paths: ModelPaths,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        distilled_lora_strength_stage_1: float,
        distilled_lora_strength_stage_2: float,
        spatial_upsampler_path: str,
        loras: tuple[LoraPathStrengthAndSDOps, ...],
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        compilation_config: CompilationConfig | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
        prompt_enhancer_gemma_root: str | None = None,
        diffvae_optimization: DiffVAEMode = DiffVAEMode.CHUNKED_EAGER,
    ):
        self.device = device or get_device()
        self.dtype = torch.bfloat16
        self._scheduler = LTX2Scheduler()

        distilled_lora_stage_1 = LoraPathStrengthAndSDOps(
            path=distilled_lora[0].path,
            strength=distilled_lora_strength_stage_1,
            sd_ops=distilled_lora[0].sd_ops,
        )
        distilled_lora_stage_2 = LoraPathStrengthAndSDOps(
            path=distilled_lora[0].path,
            strength=distilled_lora_strength_stage_2,
            sd_ops=distilled_lora[0].sd_ops,
        )

        self.prompt_encoder = PromptEncoder(
            model_paths,
            self.dtype,
            self.device,
            registry=registry,
            offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim_strategy,
            prompt_enhancer_gemma_root=prompt_enhancer_gemma_root,
        )
        self.image_conditioner = ImageConditioner(
            model_paths.video_vae(),
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.upsampler = VideoUpsampler(
            model_paths.video_vae(),
            spatial_upsampler_path,
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.video_decoder = VideoDecoder(
            model_paths.video_vae(),
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
            diffvae_optimization=diffvae_optimization,
        )
        self.audio_decoder = AudioDecoder(
            model_paths.audio_vae(),
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        # None on checkpoints that predate DurationHead (LTX 2.5 / gemma4 only) -- __call__ requires
        # an explicit num_frames in that case instead of crashing deep in a forward pass.
        self.duration_predictor = DurationPredictor.from_checkpoint(
            model_paths.duration_head_path,
            self.dtype,
            self.device,
        )

        self.stage_1 = DiffusionStage.from_checkpoint(
            model_paths.transformer(),
            self.dtype,
            self.device,
            loras=(*loras, distilled_lora_stage_1),
            quantization=quantization,
            registry=registry,
            compilation_config=compilation_config,
            offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.stage_2 = DiffusionStage.from_checkpoint(
            model_paths.transformer(),
            self.dtype,
            self.device,
            loras=(*loras, distilled_lora_stage_2),
            quantization=quantization,
            registry=registry,
            compilation_config=compilation_config,
            offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim_strategy,
        )

    @torch.inference_mode()
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
        audio_guider_params: MultiModalGuiderParams,
        images: list[ImageConditioningInput],
        num_frames: int | AutoDuration = DEFAULT_AUTO_DURATION,
        vae_dtype: torch.dtype | None = None,
        tiling_config: TilingConfig | AutoTiling | None = AUTO_TILING,
        enhance_prompt: bool = False,
        enhance_static_cache: bool = False,
        max_batch_size: int = 1,
        stage_1_sigmas: torch.Tensor | None = None,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
        color_space: EXRColorSpace | None = None,
        generated_keyframes: int | Sequence[int] = 0,
        decode_with_keyframes: bool = False,
    ) -> PipelineOutput:
        require_num_frames_source(num_frames, self.duration_predictor)
        images = self.image_conditioner.resolve_crf(images)
        assert_resolution(height=height, width=width, is_two_stage=True)
        assert_generated_keyframes_request(decode_with_keyframes, generated_keyframes, self.stage_1)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16
        if vae_dtype is None:
            vae_dtype = dtype

        ctx_p, ctx_n = self.prompt_encoder(
            [prompt, negative_prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_seed=seed,
            enhance_static_cache=enhance_static_cache,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
        )
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        num_frames = resolve_num_frames(
            num_frames,
            self.duration_predictor,
            video_encoding=v_context_p,
            audio_encoding=a_context_p,
            frame_rate=frame_rate,
        )

        scale_factors = tiling_scale_factors_for_vae(self.video_decoder.checkpoint_path)
        tiling_config = ensure_tiling_config(
            tiling_config,
            scale_factors=scale_factors,
            vae_checkpoint_path=self.video_decoder.checkpoint_path,
            video_shape=VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=frame_rate),
            diffvae_optimization=self.video_decoder.diffvae_optimization,
            device=self.device,
            keyframes=decode_with_keyframes,
        )

        keyframe_positions = resolve_generated_keyframes(generated_keyframes, num_frames)

        # Stage 1: Generate video at half resolution with CFG guidance using res2s sampler.
        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        stage_1_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                video_encoder=enc,
                dtype=dtype,
                device=self.device,
                color_space=color_space,
            )
        )
        stage_1_conditionings.extend(generated_keyframe_conditionings(generated_keyframes, num_frames))

        if stage_1_sigmas is None:
            empty_latent = torch.empty(
                VideoLatentShape.from_pixel_shape(
                    stage_1_output_shape,
                    scale_factors=self.stage_1.video_scale_factors,
                ).to_torch_shape()
            )
            stage_1_sigmas = self._scheduler.execute(latent=empty_latent, steps=num_inference_steps)
        sigmas = stage_1_sigmas.to(dtype=torch.float32, device=self.device)

        video_latent, audio_latent = create_initial_av_latents(
            width=stage_1_output_shape.width,
            height=stage_1_output_shape.height,
            frames=num_frames,
            fps=frame_rate,
            device=self.device,
            dtype=self.dtype,
            video_scale_factors=self.stage_1.video_scale_factors,
        )
        video_state, audio_state = self.stage_1(
            denoiser=GuidedDenoiser(
                v_context=v_context_p,
                a_context=a_context_p,
                video_guider=MultiModalGuider(
                    params=video_guider_params,
                    negative_context=v_context_n,
                ),
                audio_guider=MultiModalGuider(
                    params=audio_guider_params,
                    negative_context=a_context_n,
                ),
            ),
            sigmas=sigmas,
            noiser=noiser,
            modalities=VideoAudio(
                video=ModalitySpec(
                    latent=video_latent,
                    conditioning_fps=frame_rate,
                    context=v_context_p,
                    conditionings=stage_1_conditionings,
                ),
                audio=ModalitySpec(
                    latent=audio_latent,
                    conditioning_fps=frame_rate,
                    context=a_context_p,
                ),
            ),
            loop=res2s_audio_video_denoising_loop,
            max_batch_size=max_batch_size,
        )

        # Stage 2: Upsample and refine the video at higher resolution with distilled LoRA.
        upscaled_video_latent = self.upsampler(video_state.latent[:1])

        stage_2_sigmas = stage_2_sigmas.to(dtype=torch.float32, device=self.device)
        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=stage_2_output_shape.height,
                width=stage_2_output_shape.width,
                video_encoder=enc,
                dtype=dtype,
                device=self.device,
                color_space=color_space,
            )
        )
        if decode_with_keyframes:
            if video_state.generated_keyframes is None:
                raise RuntimeError("Stage 1 did not return generated_keyframes despite requesting slots")
            stage_2_conditionings.append(
                VideoGeneratedKeyframeSlots(
                    pixel_frame_indices=keyframe_positions,
                    initial_keyframes=self.upsampler(video_state.generated_keyframes),
                )
            )

        # Stage 2 refines video only; discard its audio. On the multi-GPU path stage-2 audio
        # runs under partial tiled/TDP video context, so the full-context stage-1 audio is kept.
        video_state, _ = self.stage_2(
            denoiser=SimpleDenoiser(v_context=v_context_p, a_context=a_context_p),
            sigmas=stage_2_sigmas,
            noiser=noiser,
            modalities=VideoAudio(
                video=ModalitySpec(
                    latent=upscaled_video_latent,
                    conditioning_fps=frame_rate,
                    context=v_context_p,
                    conditionings=stage_2_conditionings,
                    noise_scale=stage_2_sigmas[0].item(),
                ),
                audio=ModalitySpec(
                    latent=audio_state.latent,
                    conditioning_fps=frame_rate,
                    context=a_context_p,
                    noise_scale=stage_2_sigmas[0].item(),
                ),
            ),
            loop=res2s_audio_video_denoising_loop,
        )

        decode_keyframes = None
        if decode_with_keyframes:
            decode_keyframes = decode_keyframes_from_slots(
                video_state.generated_keyframes,
                keyframe_positions,
                num_frames,
            )
            if decode_keyframes is None:
                raise RuntimeError("decode_with_keyframes was set but no keyframe slots were produced")

        decoded_video = self.video_decoder(
            video_state.latent,
            tiling_config,
            generator,
            dtype=vae_dtype,
            keyframes=decode_keyframes,
        )
        decoded_audio = self.audio_decoder(audio_state.latent)
        return PipelineOutput(decoded_video, decoded_audio, num_frames, tiling_config)


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = add_keyframe_decode_arg(
        add_generated_keyframes_arg(hq_2_stage_arg_parser(params=LTX_2_3_HQ_PARAMS, supports_auto_duration=True))
    )
    args = parser.parse_args()
    pipeline = TI2VidTwoStagesHQPipeline(
        model_paths=args.model_paths,
        distilled_lora=args.distilled_lora,
        distilled_lora_strength_stage_1=args.distilled_lora_strength_stage_1,
        distilled_lora_strength_stage_2=args.distilled_lora_strength_stage_2,
        spatial_upsampler_path=args.spatial_upsampler_path,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
        compilation_config=args.compile,
        offload_mode=args.offload_mode,
        prompt_enhancer_gemma_root=args.prompt_enhancer_gemma_root,
        diffvae_optimization=args.diffvae_optimization,
    )
    hdr = resolve_hdr_color_space(images=args.images, hdr=args.hdr)
    vae_dtype = vae_dtype_for_hdr(hdr, torch.bfloat16)
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
        audio_guider_params=MultiModalGuiderParams(
            cfg_scale=args.audio_cfg_guidance_scale,
            stg_scale=args.audio_stg_guidance_scale,
            rescale_scale=args.audio_rescale_scale,
            modality_scale=args.v2a_guidance_scale,
            skip_step=args.audio_skip_step,
            stg_blocks=args.audio_stg_blocks,
        ),
        images=args.images,
        vae_dtype=vae_dtype,
        color_space=hdr,
        enhance_prompt=args.enhance_prompt,
        enhance_static_cache=args.enhance_static_cache,
        max_batch_size=args.max_batch_size,
        tiling_config=AUTO_TILING,
        generated_keyframes=args.num_generated_keyframes,
        decode_with_keyframes=args.decode_with_keyframes,
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
