import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.components.guiders import (
    MultiModalGuiderFactory,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.conditioning import ConditioningItem
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae import AUTO_TILING, AutoTiling, TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae.transformer import DiffVAEMode
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import VideoPixelShape
from ltx_pipelines.chunks import (
    Chunk,
    ChunkConfig,
    DecodedChunk,
    MakeVideoConditionings,
    decode_chunks,
    denoise_chunks,
    generate_uniform_chunks,
    pipeline_output_from_chunks,
    spatially_upsample_chunks,
    split_decoded_chunks,
)
from ltx_pipelines.chunks.conditionings import (
    assert_image_frames_in_clip,
    image_conditionings_for_chunk,
    images_for_chunk,
)
from ltx_pipelines.utils.args import (
    add_chunk_layout_args,
    add_generated_keyframes_arg,
    add_keyframe_decode_arg,
    chunk_config_from_args,
    default_2_stage_arg_parser,
    resolve_cli_params,
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
    STAGE_2_DISTILLED_SIGMAS,
)
from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser
from ltx_pipelines.utils.helpers import (
    assert_generated_keyframes_request,
    assert_resolution,
    ensure_tiling_config,
    get_device,
    snap_frames_to_grid,
    tiling_scale_factors_for_vae,
)
from ltx_pipelines.utils.media_io import EXRColorSpace, encode_video, resolve_hdr_color_space, vae_dtype_for_hdr
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.types import (
    DEFAULT_AUTO_DURATION,
    AutoDuration,
    ImageConditioningInput,
    OffloadMode,
    PipelineOutput,
    VideoAudio,
)


@dataclass(frozen=True)
class _TI2VidRunContext:
    """Setup snapshot for :meth:`TI2VidTwoStagesPipeline.stream_chunks`.
    Built by :meth:`TI2VidTwoStagesPipeline._prepare_run`.
    """

    generator: torch.Generator
    noiser: GaussianNoiser
    dtype: torch.dtype
    vae_dtype: torch.dtype
    #: ``images`` with every unset CRF filled in, so callers do not re-resolve it.
    images: list[ImageConditioningInput]
    v_context_p: torch.Tensor
    a_context_p: torch.Tensor
    v_context_n: torch.Tensor
    a_context_n: torch.Tensor
    num_frames: int
    tiling_config: TilingConfig
    stage_1_width: int
    stage_1_height: int
    chunk_config: ChunkConfig


class TI2VidTwoStagesPipeline:
    """
    Two-stage text/image-to-video generation pipeline.
    Stage 1 generates video at half of the target resolution with CFG guidance (assuming
    full model is used), then Stage 2 upsamples by 2x and refines using a distilled
    LoRA for higher quality output. Supports optional image conditioning via the
    images parameter.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_paths: ModelPaths,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        spatial_upsampler_path: str,
        loras: list[LoraPathStrengthAndSDOps],
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
            loras=tuple(loras),
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
            loras=(*tuple(loras), *distilled_lora),
            quantization=quantization,
            registry=registry,
            compilation_config=compilation_config,
            offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim_strategy,
        )

    def _prepare_run(  # noqa: PLR0913
        self,
        *,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        num_frames: int | AutoDuration,
        vae_dtype: torch.dtype | None,
        tiling_config: TilingConfig | AutoTiling | None,
        enhance_prompt: bool,
        enhance_static_cache: bool,
        generated_keyframes: int | Sequence[int],
        decode_with_keyframes: bool,
        chunk_config: ChunkConfig | None = None,
    ) -> _TI2VidRunContext:
        """Validate inputs, encode prompts, and resolve frame count / tiling before chunk planning."""
        require_num_frames_source(num_frames, self.duration_predictor)
        images = self.image_conditioner.resolve_crf(images)
        assert_resolution(height=height, width=width, is_two_stage=True)
        assert_generated_keyframes_request(decode_with_keyframes, generated_keyframes, self.stage_1)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16
        resolved_vae_dtype = dtype if vae_dtype is None else vae_dtype

        ctx_p, ctx_n = self.prompt_encoder(
            [prompt, negative_prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_seed=seed,
            enhance_static_cache=enhance_static_cache,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
        )
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        resolved_num_frames = snap_frames_to_grid(
            resolve_num_frames(
                num_frames,
                self.duration_predictor,
                video_encoding=v_context_p,
                audio_encoding=a_context_p,
                frame_rate=frame_rate,
            )
        )

        if chunk_config is None:
            chunk_config = ChunkConfig(chunk_pixel_frames=resolved_num_frames, next_video_carry_frames=0)

        scale_factors = tiling_scale_factors_for_vae(self.video_decoder.checkpoint_path)
        resolved_tiling_config = ensure_tiling_config(
            tiling_config,
            scale_factors=scale_factors,
            vae_checkpoint_path=self.video_decoder.checkpoint_path,
            video_shape=VideoPixelShape(
                batch=1,
                frames=min(resolved_num_frames, chunk_config.chunk_pixel_frames),
                height=height,
                width=width,
                fps=frame_rate,
            ),
            diffvae_optimization=self.video_decoder.diffvae_optimization,
            device=self.device,
            keyframes=decode_with_keyframes,
        )

        return _TI2VidRunContext(
            generator=generator,
            noiser=noiser,
            dtype=dtype,
            vae_dtype=resolved_vae_dtype,
            images=images,
            v_context_p=v_context_p,
            a_context_p=a_context_p,
            v_context_n=v_context_n,
            a_context_n=a_context_n,
            num_frames=resolved_num_frames,
            tiling_config=resolved_tiling_config,
            stage_1_width=width // 2,
            stage_1_height=height // 2,
            chunk_config=chunk_config,
        )

    def _video_conditionings(
        self,
        images: list[ImageConditioningInput],
        num_frames: int,
        color_space: EXRColorSpace | None,
    ) -> MakeVideoConditionings:
        assert_image_frames_in_clip(images, num_frames)

        def make(chunk: Chunk) -> list[ConditioningItem]:
            if chunk.video is None or not images_for_chunk(chunk, images):
                return []
            return self.image_conditioner(
                lambda enc: image_conditionings_for_chunk(
                    chunk,
                    images=images,
                    video_encoder=enc,
                    color_space=color_space,
                )
            )

        return make

    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        audio_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
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
        chunk_config: ChunkConfig | None = None,
    ) -> PipelineOutput:
        """Generate a video.
        ``chunk_config=None`` plans a single chunk covering the snapped clip.
        """
        chunks, num_frames, tiling_config = self.stream_chunks(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            height=height,
            width=width,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
            images=images,
            num_frames=num_frames,
            vae_dtype=vae_dtype,
            tiling_config=tiling_config,
            enhance_prompt=enhance_prompt,
            enhance_static_cache=enhance_static_cache,
            max_batch_size=max_batch_size,
            stage_1_sigmas=stage_1_sigmas,
            stage_2_sigmas=stage_2_sigmas,
            color_space=color_space,
            generated_keyframes=generated_keyframes,
            decode_with_keyframes=decode_with_keyframes,
            chunk_config=chunk_config,
        )
        return pipeline_output_from_chunks(chunks, num_frames=num_frames, tiling_config=tiling_config)

    def stream_chunks(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        audio_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
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
        chunk_config: ChunkConfig | None = None,
    ) -> tuple[Iterator[DecodedChunk], int, TilingConfig]:
        """Generate a video as a stream of decoded chunks.
        ``chunk_config=None`` plans a single chunk covering the snapped clip.
        """
        ctx = self._prepare_run(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            height=height,
            width=width,
            frame_rate=frame_rate,
            images=images,
            num_frames=num_frames,
            vae_dtype=vae_dtype,
            tiling_config=tiling_config,
            enhance_prompt=enhance_prompt,
            enhance_static_cache=enhance_static_cache,
            generated_keyframes=generated_keyframes,
            decode_with_keyframes=decode_with_keyframes,
            chunk_config=chunk_config,
        )

        sigmas = (
            stage_1_sigmas if stage_1_sigmas is not None else self._scheduler.execute(steps=num_inference_steps)
        ).to(dtype=torch.float32, device=self.device)
        stage_2_sigmas = stage_2_sigmas.to(dtype=torch.float32, device=self.device)

        def make_denoiser(chunk: Chunk) -> FactoryGuidedDenoiser:
            return FactoryGuidedDenoiser(
                v_context=chunk.video_context,
                a_context=chunk.audio_context,
                video_guider_factory=create_multimodal_guider_factory(
                    params=video_guider_params, negative_context=ctx.v_context_n
                ),
                audio_guider_factory=create_multimodal_guider_factory(
                    params=audio_guider_params, negative_context=ctx.a_context_n
                ),
            )

        target = VideoPixelShape(
            batch=1,
            frames=ctx.num_frames,
            height=ctx.stage_1_height,
            width=ctx.stage_1_width,
            fps=frame_rate,
        )
        chunks = generate_uniform_chunks(
            target=target,
            context=VideoAudio(video=ctx.v_context_p, audio=ctx.a_context_p),
            device=self.device,
            dtype=ctx.dtype,
            config=ctx.chunk_config,
            video_scale_factors=self.stage_1.video_scale_factors,
            make_video_conditionings=self._video_conditionings(ctx.images, ctx.num_frames, color_space),
            generated_keyframes=generated_keyframes,
        )
        base = denoise_chunks(
            chunks,
            self.stage_1,
            sigmas=sigmas,
            noiser=ctx.noiser,
            fps=frame_rate,
            max_batch_size=max_batch_size,
            make_denoiser=make_denoiser,
        )
        upscaled = spatially_upsample_chunks(base, self.upsampler)
        refined = denoise_chunks(
            upscaled,
            self.stage_2,
            sigmas=stage_2_sigmas,
            noiser=ctx.noiser,
            fps=frame_rate,
            freeze_audio=True,
        )
        return (
            decode_chunks(
                refined,
                self.video_decoder,
                self.audio_decoder,
                fps=frame_rate,
                tiling_config=ctx.tiling_config,
                generator=ctx.generator,
                dtype=ctx.vae_dtype,
                keyframes=decode_with_keyframes,
            ),
            ctx.num_frames,
            ctx.tiling_config,
        )


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    params = resolve_cli_params()
    parser = add_chunk_layout_args(
        add_keyframe_decode_arg(
            add_generated_keyframes_arg(default_2_stage_arg_parser(params=params, supports_auto_duration=True))
        )
    )
    args = parser.parse_args()
    pipeline = TI2VidTwoStagesPipeline(
        model_paths=args.model_paths,
        distilled_lora=args.distilled_lora,
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
    chunk_config = chunk_config_from_args(args)
    video_guider_params = MultiModalGuiderParams(
        cfg_scale=args.video_cfg_guidance_scale,
        stg_scale=args.video_stg_guidance_scale,
        rescale_scale=args.video_rescale_scale,
        modality_scale=args.a2v_guidance_scale,
        skip_step=args.video_skip_step,
        stg_blocks=args.video_stg_blocks,
    )
    audio_guider_params = MultiModalGuiderParams(
        cfg_scale=args.audio_cfg_guidance_scale,
        stg_scale=args.audio_stg_guidance_scale,
        rescale_scale=args.audio_rescale_scale,
        modality_scale=args.v2a_guidance_scale,
        skip_step=args.audio_skip_step,
        stg_blocks=args.audio_stg_blocks,
    )
    chunks, num_frames, tiling_config = pipeline.stream_chunks(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        video_guider_params=video_guider_params,
        audio_guider_params=audio_guider_params,
        images=args.images,
        vae_dtype=vae_dtype,
        color_space=hdr,
        enhance_prompt=args.enhance_prompt,
        enhance_static_cache=args.enhance_static_cache,
        max_batch_size=args.max_batch_size,
        tiling_config=AUTO_TILING,
        generated_keyframes=args.num_generated_keyframes,
        decode_with_keyframes=args.decode_with_keyframes,
        chunk_config=chunk_config,
    )
    if chunk_config is not None:
        video, audio, audio_sampling_rate = split_decoded_chunks(chunks)
        video_chunks_number = 1
    else:
        result = pipeline_output_from_chunks(
            chunks,
            num_frames=num_frames,
            tiling_config=tiling_config,
        )
        video, audio, audio_sampling_rate = result.video, result.audio, None
        video_chunks_number = get_video_chunks_number(result.num_frames, result.tiling_config)
    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
        color_space=hdr,
        audio_sampling_rate=audio_sampling_rate,
    )


if __name__ == "__main__":
    main()
