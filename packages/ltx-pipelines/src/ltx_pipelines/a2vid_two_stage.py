import argparse
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.conditioning import ConditioningItem
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae import AUTO_TILING, AutoTiling, TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae.transformer import DiffVAEMode
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
from ltx_pipelines.chunks import (
    Chunk,
    ChunkConfig,
    DecodedChunk,
    MakeVideoConditionings,
    decode_chunks,
    denoise_chunks,
    generate_uniform_chunks,
    pipeline_output_from_chunks,
    replace_chunks_audio,
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
    markers_from_args,
    resolve_cli_params,
)
from ltx_pipelines.utils.blocks import (
    AudioConditioner,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.constants import (
    STAGE_2_DISTILLED_SIGMAS,
    PipelineParams,
)
from ltx_pipelines.utils.denoisers import GuidedDenoiser
from ltx_pipelines.utils.helpers import (
    assert_generated_keyframes_request,
    assert_resolution,
    audio_duration_seconds,
    ensure_tiling_config,
    get_device,
    num_frames_from_audio_duration,
    snap_frames_to_grid,
    tiling_scale_factors_for_vae,
)
from ltx_pipelines.utils.media_io import (
    EXRColorSpace,
    decode_audio_from_file,
    encode_video,
    resolve_hdr_color_space,
    vae_dtype_for_hdr,
)
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.types import (
    ImageConditioningInput,
    LoadedImageConditioning,
    OffloadMode,
    PipelineOutput,
    VideoAudio,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _A2VidRunContext:
    """Setup snapshot for :meth:`A2VidPipelineTwoStage.stream_chunks`.
    Built by :meth:`A2VidPipelineTwoStage._prepare_run`.
    """

    generator: torch.Generator
    noiser: GaussianNoiser
    dtype: torch.dtype
    vae_dtype: torch.dtype
    images: list[LoadedImageConditioning]
    v_context_p: torch.Tensor
    a_context_p: torch.Tensor
    v_context_n: torch.Tensor
    num_frames: int
    tiling_config: TilingConfig
    stage_1_width: int
    stage_1_height: int
    encoded_audio_latent: torch.Tensor
    source_audio: Audio
    chunk_config: ChunkConfig


class A2VidPipelineTwoStage:
    """
    Two-stage audio to video generation pipeline.
    Stage 1 generates video at half the target resolution with audio conditioning
    (video-only denoising, audio frozen), then Stage 2 upsamples by 2x and refines
    both video and audio using a distilled LoRA for higher quality output.
    When ``num_frames`` is omitted, the frame count is derived from the effective
    conditioning-audio duration (``min(audio_max_duration, remaining audio after
    audio_start_time)``) at ``frame_rate``, snapped to the VAE temporal grid.
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
        self.audio_conditioner = AudioConditioner(
            model_paths.audio_vae(),
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
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
        stage_2_loras = (*tuple(loras), *tuple(distilled_lora))
        self.stage_2 = DiffusionStage.from_checkpoint(
            model_paths.transformer(),
            self.dtype,
            self.device,
            loras=stage_2_loras,
            quantization=quantization,
            registry=registry,
            compilation_config=compilation_config,
            offload_mode=offload_mode,
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

    def _prepare_run(  # noqa: PLR0913
        self,
        *,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int | None,
        frame_rate: float,
        images: list[ImageConditioningInput],
        color_space: EXRColorSpace | None,
        audio_path: str,
        audio_start_time: float,
        audio_max_duration: float | None,
        vae_dtype: torch.dtype | None,
        tiling_config: TilingConfig | AutoTiling | None,
        enhance_prompt: bool,
        enhance_static_cache: bool,
        generated_keyframes: int | Sequence[int] = 0,
        decode_with_keyframes: bool = False,
        chunk_config: ChunkConfig | None = None,
    ) -> _A2VidRunContext:
        """Validate inputs, decode/encode audio, and resolve frame count / tiling before chunk planning."""
        enhance_prompt_image = images[0].path if images else None
        images = self.image_conditioner.load_conditionings(images, color_space)
        assert_resolution(height=height, width=width, is_two_stage=True)
        assert_generated_keyframes_request(decode_with_keyframes, generated_keyframes, self.stage_1)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16
        resolved_vae_dtype = dtype if vae_dtype is None else vae_dtype

        # Decode audio first so the frame count can follow the effective clip length
        # (``audio_max_duration`` capped by remaining audio after ``audio_start_time``).
        decoded_audio = decode_audio_from_file(audio_path, self.device, audio_start_time, audio_max_duration)
        if decoded_audio is None:
            raise ValueError(f"Failed to decode audio from {audio_path}. Please check the file and try again.")
        if num_frames is None:
            num_frames = num_frames_from_audio_duration(decoded_audio, frame_rate=frame_rate)
            logger.info(
                "Derived num_frames=%d from %.2fs of audio @ %.2f fps",
                num_frames,
                audio_duration_seconds(decoded_audio),
                frame_rate,
            )
        num_frames = snap_frames_to_grid(num_frames)

        if chunk_config is None:
            chunk_config = ChunkConfig(chunk_pixel_frames=num_frames, next_video_carry_frames=0)

        ctx_p, ctx_n = self.prompt_encoder(
            [prompt, negative_prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_static_cache=enhance_static_cache,
            enhance_prompt_image=enhance_prompt_image,
        )
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, _ = ctx_n.video_encoding, ctx_n.audio_encoding

        scale_factors = tiling_scale_factors_for_vae(self.video_decoder.checkpoint_path)
        resolved_tiling_config = ensure_tiling_config(
            tiling_config,
            scale_factors=scale_factors,
            vae_checkpoint_path=self.video_decoder.checkpoint_path,
            video_shape=VideoPixelShape(
                batch=1,
                frames=min(num_frames, chunk_config.chunk_pixel_frames),
                height=height,
                width=width,
                fps=frame_rate,
            ),
            diffvae_optimization=self.video_decoder.diffvae_optimization,
            device=self.device,
            keyframes=decode_with_keyframes,
        )

        encoded_audio_latent = self.audio_conditioner(lambda enc: vae_encode_audio(decoded_audio, enc, None))
        audio_shape = AudioLatentShape.from_duration(batch=1, duration=num_frames / frame_rate, channels=8, mel_bins=16)
        encoded_audio_latent = encoded_audio_latent[:, :, : audio_shape.frames]
        source_audio = Audio(
            waveform=decoded_audio.waveform.squeeze(0),
            sampling_rate=decoded_audio.sampling_rate,
        )

        return _A2VidRunContext(
            generator=generator,
            noiser=noiser,
            dtype=dtype,
            vae_dtype=resolved_vae_dtype,
            images=images,
            v_context_p=v_context_p,
            a_context_p=a_context_p,
            v_context_n=v_context_n,
            num_frames=num_frames,
            tiling_config=resolved_tiling_config,
            stage_1_width=width // 2,
            stage_1_height=height // 2,
            encoded_audio_latent=encoded_audio_latent,
            source_audio=source_audio,
            chunk_config=chunk_config,
        )

    def _video_conditionings(
        self,
        images: list[LoadedImageConditioning],
        num_frames: int,
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
                )
            )

        return make

    @torch.inference_mode()
    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int | None,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams,
        images: list[ImageConditioningInput],
        audio_path: str,
        audio_start_time: float = 0.0,
        audio_max_duration: float | None = None,
        vae_dtype: torch.dtype | None = None,
        tiling_config: TilingConfig | AutoTiling | None = AUTO_TILING,
        enhance_prompt: bool = False,
        enhance_static_cache: bool = False,
        max_batch_size: int = 1,
        stage_1_sigmas: torch.Tensor | None = None,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
        color_space: EXRColorSpace | None = None,
        chunk_config: ChunkConfig | None = None,
        generated_keyframes: int | Sequence[int] = 0,
        decode_with_keyframes: bool = False,
    ) -> PipelineOutput:
        chunks, num_frames, tiling_config = self.stream_chunks(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            video_guider_params=video_guider_params,
            images=images,
            audio_path=audio_path,
            audio_start_time=audio_start_time,
            audio_max_duration=audio_max_duration,
            vae_dtype=vae_dtype,
            tiling_config=tiling_config,
            enhance_prompt=enhance_prompt,
            enhance_static_cache=enhance_static_cache,
            max_batch_size=max_batch_size,
            stage_1_sigmas=stage_1_sigmas,
            stage_2_sigmas=stage_2_sigmas,
            color_space=color_space,
            chunk_config=chunk_config,
            generated_keyframes=generated_keyframes,
            decode_with_keyframes=decode_with_keyframes,
        )
        return pipeline_output_from_chunks(chunks, num_frames=num_frames, tiling_config=tiling_config)

    def stream_chunks(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int | None,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams,
        images: list[ImageConditioningInput],
        audio_path: str,
        audio_start_time: float = 0.0,
        audio_max_duration: float | None = None,
        vae_dtype: torch.dtype | None = None,
        tiling_config: TilingConfig | AutoTiling | None = AUTO_TILING,
        enhance_prompt: bool = False,
        enhance_static_cache: bool = False,
        max_batch_size: int = 1,
        stage_1_sigmas: torch.Tensor | None = None,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
        color_space: EXRColorSpace | None = None,
        chunk_config: ChunkConfig | None = None,
        generated_keyframes: int | Sequence[int] = 0,
        decode_with_keyframes: bool = False,
    ) -> tuple[Iterator[DecodedChunk], int, TilingConfig]:
        """Generate a video as a stream of decoded chunks.
        ``chunk_config=None`` plans a single chunk covering the snapped clip.
        Audio is the frozen source latent both stages; decoded output is the original waveform.
        """
        ctx = self._prepare_run(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            color_space=color_space,
            audio_path=audio_path,
            audio_start_time=audio_start_time,
            audio_max_duration=audio_max_duration,
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

        target = VideoPixelShape(
            batch=1,
            frames=ctx.num_frames,
            height=ctx.stage_1_height,
            width=ctx.stage_1_width,
            fps=frame_rate,
        )

        def make_denoiser(chunk: Chunk) -> GuidedDenoiser:
            return GuidedDenoiser(
                v_context=chunk.video_context,
                a_context=chunk.audio_context,
                video_guider=MultiModalGuider(
                    params=video_guider_params,
                    negative_context=ctx.v_context_n,
                ),
                audio_guider=MultiModalGuider(params=MultiModalGuiderParams()),
            )

        chunks = generate_uniform_chunks(
            target=target,
            context=VideoAudio(video=ctx.v_context_p, audio=ctx.a_context_p),
            device=self.device,
            dtype=ctx.dtype,
            config=ctx.chunk_config,
            video_scale_factors=self.stage_1.video_scale_factors,
            audio_latent=ctx.encoded_audio_latent,
            make_video_conditionings=self._video_conditionings(ctx.images, ctx.num_frames),
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
            freeze_audio=True,
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
        decoded = decode_chunks(
            refined,
            self.video_decoder,
            audio_decoder=None,
            fps=frame_rate,
            tiling_config=ctx.tiling_config,
            generator=ctx.generator,
            dtype=ctx.vae_dtype,
            keyframes=decode_with_keyframes,
        )
        return (
            replace_chunks_audio(decoded, ctx.source_audio, frame_rate),
            ctx.num_frames,
            ctx.tiling_config,
        )


def resolve_a2vid_cli_duration(
    *,
    num_frames: int | None,
    audio_max_duration: float | None,
    frame_rate: float,
    default_num_frames: int,
) -> tuple[int | None, float]:
    """Pick the single duration driver for the A2Vid CLI.
    ``--num-frames`` and ``--audio-max-duration`` are mutually exclusive. Returns
    ``(num_frames_for_pipeline, audio_max_duration)``:
    * only ``--audio-max-duration``: ``num_frames`` is ``None`` so the pipeline derives
      frames from the decoded clip (capped by remaining audio after start time).
    * only ``--num-frames``, or neither: use that frame count (defaulting to
      ``default_num_frames``) and clip audio to ``num_frames / frame_rate``.
    """
    if num_frames is not None and audio_max_duration is not None:
        raise ValueError("argument --num-frames: not allowed with argument --audio-max-duration")
    if audio_max_duration is not None:
        return None, audio_max_duration
    frames = default_num_frames if num_frames is None else num_frames
    return frames, frames / frame_rate


def build_a2vid_arg_parser(params: PipelineParams) -> argparse.ArgumentParser:
    """Two-stage parser plus A2Vid audio flags; ``--num-frames`` default is unset.
    Leaving ``--num-frames`` as ``None`` when omitted is what lets
    ``resolve_a2vid_cli_duration`` tell "user passed --audio-max-duration" apart from
    "user passed both" (the shared parser would otherwise fill in ``params.num_frames``).
    """
    parser = default_2_stage_arg_parser(params=params)
    parser.add_argument(
        "--audio-path",
        type=str,
        required=True,
        help="Path to the audio file to condition the video generation.",
    )
    parser.add_argument(
        "--audio-start-time",
        type=float,
        default=0.0,
        help="Start time in seconds to read audio from (default: 0.0).",
    )
    parser.add_argument(
        "--audio-max-duration",
        type=float,
        default=None,
        help=(
            "Maximum audio duration in seconds, measured from --audio-start-time. "
            "The video length is derived from the effective clip "
            "(this value capped by remaining audio in the file) at --frame-rate. "
            "Mutually exclusive with --num-frames. "
            f"If neither is given, audio is clipped to the default "
            f"--num-frames / --frame-rate ({params.num_frames} frames)."
        ),
    )
    for action in parser._actions:
        if "--num-frames" in action.option_strings:
            action.default = None
            action.help = (
                "Number of frames to generate, num_frames = 8 * k + 1 "
                f"(default: {params.num_frames}). Mutually exclusive with --audio-max-duration; "
                "when set, audio is clipped to this length / --frame-rate."
            )
            break
    return parser


def resolve_a2vid_duration_or_exit(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    default_num_frames: int,
) -> tuple[int | None, float]:
    """Resolve duration knobs, or ``parser.error`` if both flags were given."""
    try:
        return resolve_a2vid_cli_duration(
            num_frames=args.num_frames,
            audio_max_duration=args.audio_max_duration,
            frame_rate=args.frame_rate,
            default_num_frames=default_num_frames,
        )
    except ValueError as e:
        parser.error(str(e))
        raise  # parser.error always exits; keep the type checker happy


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    params = resolve_cli_params()
    parser = add_chunk_layout_args(add_keyframe_decode_arg(add_generated_keyframes_arg(build_a2vid_arg_parser(params))))
    args = parser.parse_args()
    hdr = resolve_hdr_color_space(images=args.images, hdr=args.hdr)
    video_marker, _ = markers_from_args(args, device=get_device(), color_space=hdr)
    num_frames, audio_max_duration = resolve_a2vid_duration_or_exit(parser, args, default_num_frames=params.num_frames)
    pipeline = A2VidPipelineTwoStage(
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
    chunks, num_frames_out, tiling_config = pipeline.stream_chunks(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        video_guider_params=video_guider_params,
        images=args.images,
        vae_dtype=vae_dtype,
        color_space=hdr,
        tiling_config=AUTO_TILING,
        enhance_prompt=args.enhance_prompt,
        enhance_static_cache=args.enhance_static_cache,
        audio_path=args.audio_path,
        audio_start_time=args.audio_start_time,
        audio_max_duration=audio_max_duration,
        max_batch_size=args.max_batch_size,
        chunk_config=chunk_config,
        generated_keyframes=args.num_generated_keyframes,
        decode_with_keyframes=args.decode_with_keyframes,
    )
    if chunk_config is not None:
        video, audio, audio_sampling_rate = split_decoded_chunks(chunks)
        video_chunks_number = 1
    else:
        result = pipeline_output_from_chunks(
            chunks,
            num_frames=num_frames_out,
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
        video_marker=video_marker,
    )


if __name__ == "__main__":
    main()
