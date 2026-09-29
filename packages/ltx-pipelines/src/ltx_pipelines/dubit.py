"""Two-stage Dub-It pipeline with IC-LoRA and appended audio reference conditioning."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.conditioning import ConditioningItem
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae import (
    AUTO_TILING,
    AutoTiling,
    TileSizeConfig,
    TilingConfig,
    VideoEncoder,
    get_video_chunks_number,
)
from ltx_core.model.video_vae.transformer import DiffVAEMode
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import VideoPixelShape
from ltx_pipelines.chunks import (
    Chunk,
    ChunkConfig,
    DecodedChunk,
    MakeAudioConditionings,
    MakeVideoConditionings,
    SequentialVideoFrameSource,
    decode_chunks,
    denoise_chunks,
    generate_uniform_chunks,
    pipeline_output_from_chunks,
    replace_audio_conditionings,
    spatially_upsample_chunks,
    split_decoded_chunks,
)
from ltx_pipelines.chunks.conditionings import (
    assert_image_frames_in_clip,
    audio_reference_conditionings_for_chunk,
    image_conditionings_for_chunk,
    reference_video_conditionings_for_chunk,
)
from ltx_pipelines.iclora_utils import (
    patchify_audio_reference_latent,
    read_lora_reference_downscale_factor,
)
from ltx_pipelines.utils.args import (
    add_chunk_layout_args,
    add_generated_keyframes_arg,
    add_keyframe_decode_arg,
    chunk_config_from_args,
    dubit_arg_parser,
    resolve_cli_params,
)
from ltx_pipelines.utils.blocks import (
    AudioConditioner,
    AudioDecoder,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.utils.helpers import (
    assert_generated_keyframes_request,
    assert_resolution,
    ensure_tiling_config,
    get_device,
    snap_frames_to_grid,
    tiling_scale_factors_for_vae,
)
from ltx_pipelines.utils.media_io import EXRColorSpace, decode_audio_from_file, encode_video, get_videostream_metadata
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.types import ImageConditioningInput, OffloadMode, PipelineOutput, VideoAudio

__all__ = ["DubItPipeline", "main", "patchify_dubit_audio_reference_latent"]

patchify_dubit_audio_reference_latent = patchify_audio_reference_latent


@dataclass(frozen=True)
class _DubItRunContext:
    """Setup snapshot for :meth:`DubItPipeline.stream_chunks`, built by :meth:`DubItPipeline._prepare_run`."""

    generator: torch.Generator
    noiser: GaussianNoiser
    dtype: torch.dtype
    vae_dtype: torch.dtype
    images: list[ImageConditioningInput]
    video_context: torch.Tensor
    audio_context: torch.Tensor
    num_frames: int
    frame_rate: float
    tiling_config: TilingConfig
    stage_1_width: int
    stage_1_height: int
    ref_audio_latent: torch.Tensor
    reference_video_path: str
    reference_strength: float


class DubItPipeline:
    """Two-stage Dub-It with IC-LoRA video reference and appended audio reference tokens."""

    def __init__(  # noqa: PLR0913
        self,
        model_paths: ModelPaths,
        spatial_upsampler_path: str,
        ic_lora: LoraPathStrengthAndSDOps,
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        compilation_config: CompilationConfig | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
        prompt_enhancer_gemma_root: str | None = None,
        diffvae_optimization: DiffVAEMode = DiffVAEMode.CHUNKED_EAGER,
    ) -> None:
        self.device = device or get_device()
        self.dtype = torch.bfloat16
        self.ic_lora = ic_lora
        loras = (ic_lora,)

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
        self.stage = DiffusionStage.from_checkpoint(
            model_paths.transformer(),
            self.dtype,
            self.device,
            loras=loras,
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
        self.audio_decoder = AudioDecoder(
            model_paths.audio_vae(),
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.reference_downscale_factor = read_lora_reference_downscale_factor(ic_lora.path)

    def _encode_reference_audio_vae_latent(self, video_path: str) -> torch.Tensor:
        audio = decode_audio_from_file(video_path, self.device)
        if audio is None:
            msg = f"No audio stream found in {video_path}"
            raise ValueError(msg)
        return self.audio_conditioner(lambda enc: vae_encode_audio(audio, enc, None))

    def _prepare_run(  # noqa: PLR0913
        self,
        *,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        images: list[ImageConditioningInput],
        reference_video_path: str,
        reference_strength: float,
        enhance_prompt: bool,
        enhance_static_cache: bool,
        vae_dtype: torch.dtype | None,
        tiling_config: TilingConfig | AutoTiling | None,
        generated_keyframes: int | Sequence[int] = 0,
        decode_with_keyframes: bool = False,
    ) -> _DubItRunContext:
        """Validate inputs, encode prompts/reference, and resolve frame count / tiling before chunk planning."""
        images = self.image_conditioner.resolve_crf(images)
        assert_resolution(height=height, width=width, is_two_stage=True)
        assert_generated_keyframes_request(decode_with_keyframes, generated_keyframes, self.stage)

        meta = get_videostream_metadata(reference_video_path)
        num_frames = snap_frames_to_grid(meta.frames)
        frame_rate = float(meta.fps)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        resolved_vae_dtype = self.dtype if vae_dtype is None else vae_dtype

        (ctx_p,) = self.prompt_encoder(
            [prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_seed=seed,
            enhance_static_cache=enhance_static_cache,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
        )
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding

        scale_factors = tiling_scale_factors_for_vae(self.video_decoder.checkpoint_path)
        resolved_tiling_config = ensure_tiling_config(
            tiling_config,
            scale_factors=scale_factors,
            vae_checkpoint_path=self.video_decoder.checkpoint_path,
            video_shape=VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=frame_rate),
            diffvae_optimization=self.video_decoder.diffvae_optimization,
            device=self.device,
            keyframes=decode_with_keyframes,
        )
        ref_audio_latent = self._encode_reference_audio_vae_latent(reference_video_path)
        return _DubItRunContext(
            generator=generator,
            noiser=noiser,
            dtype=self.dtype,
            vae_dtype=resolved_vae_dtype,
            images=images,
            video_context=video_context,
            audio_context=audio_context,
            num_frames=num_frames,
            frame_rate=frame_rate,
            tiling_config=resolved_tiling_config,
            stage_1_width=width // 2,
            stage_1_height=height // 2,
            ref_audio_latent=ref_audio_latent,
            reference_video_path=reference_video_path,
            reference_strength=reference_strength,
        )

    def _video_conditionings(
        self,
        images: list[ImageConditioningInput],
        num_frames: int,
        reference_video_path: str,
        reference_strength: float,
        color_space: EXRColorSpace | None,
        frame_source: SequentialVideoFrameSource,
    ) -> MakeVideoConditionings:
        assert_image_frames_in_clip(images, num_frames)

        def make(chunk: Chunk) -> list[ConditioningItem]:
            if chunk.video is None:
                return []

            def encode(enc: VideoEncoder) -> list[ConditioningItem]:
                items = [
                    *image_conditionings_for_chunk(chunk, images=images, video_encoder=enc, color_space=color_space),
                    *reference_video_conditionings_for_chunk(
                        chunk,
                        video_conditioning=[(reference_video_path, reference_strength)],
                        video_encoder=enc,
                        downscale_factor=self.reference_downscale_factor,
                        encode_tiling=TileSizeConfig.default(),
                        color_space=color_space,
                        frame_sources={reference_video_path: frame_source},
                    ),
                ]
                return items

            return self.image_conditioner(encode)

        return make

    def _audio_conditionings(self, *, latent: torch.Tensor | None, fps: float) -> MakeAudioConditionings:
        def make(chunk: Chunk) -> list[ConditioningItem]:
            return audio_reference_conditionings_for_chunk(chunk, latent=latent, fps=fps)

        return make

    @torch.inference_mode()
    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        images: list[ImageConditioningInput],
        reference_video_path: str,
        reference_strength: float = 1.0,
        enhance_prompt: bool = False,
        enhance_static_cache: bool = False,
        vae_dtype: torch.dtype | None = None,
        tiling_config: TilingConfig | AutoTiling | None = AUTO_TILING,
        stage_1_sigmas: torch.Tensor = DISTILLED_SIGMAS,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
        color_space: EXRColorSpace | None = None,
        chunk_config: ChunkConfig | None = None,
        generated_keyframes: int | Sequence[int] = 0,
        decode_with_keyframes: bool = False,
    ) -> PipelineOutput:
        chunks, num_frames, tiling_config = self.stream_chunks(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            images=images,
            reference_video_path=reference_video_path,
            reference_strength=reference_strength,
            enhance_prompt=enhance_prompt,
            enhance_static_cache=enhance_static_cache,
            vae_dtype=vae_dtype,
            tiling_config=tiling_config,
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
        seed: int,
        height: int,
        width: int,
        images: list[ImageConditioningInput],
        reference_video_path: str,
        reference_strength: float = 1.0,
        enhance_prompt: bool = False,
        enhance_static_cache: bool = False,
        vae_dtype: torch.dtype | None = None,
        tiling_config: TilingConfig | AutoTiling | None = AUTO_TILING,
        stage_1_sigmas: torch.Tensor = DISTILLED_SIGMAS,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
        color_space: EXRColorSpace | None = None,
        chunk_config: ChunkConfig | None = None,
        generated_keyframes: int | Sequence[int] = 0,
        decode_with_keyframes: bool = False,
    ) -> tuple[Iterator[DecodedChunk], int, TilingConfig]:
        """Generate a dubbed clip as a stream of decoded chunks.
        ``chunk_config=None`` plans a single chunk covering the snapped clip.
        Stage 2 freezes stage-1 audio and uses it as the audio-reference tokens.
        """
        ctx = self._prepare_run(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            images=images,
            reference_video_path=reference_video_path,
            reference_strength=reference_strength,
            enhance_prompt=enhance_prompt,
            enhance_static_cache=enhance_static_cache,
            vae_dtype=vae_dtype,
            tiling_config=tiling_config,
            generated_keyframes=generated_keyframes,
            decode_with_keyframes=decode_with_keyframes,
        )
        if chunk_config is None:
            chunk_config = ChunkConfig(chunk_pixel_frames=ctx.num_frames, next_video_carry_frames=0)

        stage_1_sigmas = stage_1_sigmas.to(dtype=torch.float32, device=self.device)
        stage_2_sigmas = stage_2_sigmas.to(dtype=torch.float32, device=self.device)

        target = VideoPixelShape(
            batch=1,
            frames=ctx.num_frames,
            height=ctx.stage_1_height,
            width=ctx.stage_1_width,
            fps=ctx.frame_rate,
        )
        # Lazy stage-1 → upsample → stage-2 pulls one chunk through both resolutions before
        # the next, so lookback still holds the window for the full-res encode.
        frame_source = SequentialVideoFrameSource(ctx.reference_video_path, self.device)
        make_video = self._video_conditionings(
            ctx.images,
            ctx.num_frames,
            ctx.reference_video_path,
            ctx.reference_strength,
            color_space,
            frame_source=frame_source,
        )
        chunks = generate_uniform_chunks(
            target=target,
            context=VideoAudio(video=ctx.video_context, audio=ctx.audio_context),
            device=self.device,
            dtype=ctx.dtype,
            config=chunk_config,
            video_scale_factors=self.stage.video_scale_factors,
            make_video_conditionings=make_video,
            make_audio_conditionings=self._audio_conditionings(latent=ctx.ref_audio_latent, fps=ctx.frame_rate),
            generated_keyframes=generated_keyframes,
        )
        base = denoise_chunks(
            chunks,
            self.stage,
            sigmas=stage_1_sigmas,
            noiser=ctx.noiser,
            fps=ctx.frame_rate,
        )
        upscaled = spatially_upsample_chunks(base, self.upsampler)
        stage_2_input = replace_audio_conditionings(
            upscaled,
            self._audio_conditionings(latent=None, fps=ctx.frame_rate),
        )
        refined = denoise_chunks(
            stage_2_input,
            self.stage,
            sigmas=stage_2_sigmas,
            noiser=ctx.noiser,
            fps=ctx.frame_rate,
            freeze_audio=True,
        )

        def decoded() -> Iterator[DecodedChunk]:
            try:
                yield from decode_chunks(
                    refined,
                    self.video_decoder,
                    self.audio_decoder,
                    fps=ctx.frame_rate,
                    tiling_config=ctx.tiling_config,
                    generator=ctx.generator,
                    dtype=ctx.vae_dtype,
                    keyframes=decode_with_keyframes,
                )
            finally:
                frame_source.close()

        return (decoded(), ctx.num_frames, ctx.tiling_config)


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    params = resolve_cli_params(distilled=True)
    parser = add_chunk_layout_args(
        add_keyframe_decode_arg(add_generated_keyframes_arg(dubit_arg_parser(params=params)))
    )
    args = parser.parse_args()

    if not args.lora or len(args.lora) != 1:
        raise ValueError("Dub-It requires exactly one --lora (the Dub-It IC-LoRA).")

    pipeline = DubItPipeline(
        model_paths=args.model_paths,
        spatial_upsampler_path=args.spatial_upsampler_path,
        ic_lora=args.lora[0],
        quantization=args.quantization,
        compilation_config=args.compile,
        offload_mode=args.offload_mode,
        prompt_enhancer_gemma_root=args.prompt_enhancer_gemma_root,
        diffvae_optimization=args.diffvae_optimization,
    )
    src = get_videostream_metadata(args.reference_video)
    chunk_config = chunk_config_from_args(args)
    chunks, num_frames, tiling_config = pipeline.stream_chunks(
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        images=[],
        reference_video_path=args.reference_video,
        reference_strength=args.reference_strength,
        tiling_config=AUTO_TILING,
        enhance_prompt=args.enhance_prompt,
        enhance_static_cache=args.enhance_static_cache,
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
            num_frames=num_frames,
            tiling_config=tiling_config,
        )
        video, audio, audio_sampling_rate = result.video, result.audio, None
        video_chunks_number = get_video_chunks_number(result.num_frames, result.tiling_config)
    encode_video(
        video=video,
        fps=src.fps,
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
        color_space=None,
        audio_sampling_rate=audio_sampling_rate,
    )


if __name__ == "__main__":
    main()
