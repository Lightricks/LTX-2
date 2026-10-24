import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.conditioning import ConditioningItem
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.transformer import X0Model
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae import AUTO_TILING, AutoTiling, TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae.transformer import DiffVAEMode
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import LatentState, VideoPixelShape
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
    default_2_stage_distilled_arg_parser,
    markers_from_args,
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
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS, detect_model_version
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
from ltx_pipelines.utils.samplers import euler_ancestral_denoising_loop
from ltx_pipelines.utils.types import (
    DEFAULT_AUTO_DURATION,
    AutoDuration,
    Denoiser,
    ImageConditioningInput,
    LoadedImageConditioning,
    OffloadMode,
    PipelineOutput,
    VideoAudio,
)

# Generation from which a denoising pass uses the ancestral (SDE) Euler sampler rather than the
# deterministic one. ``DFRPipeline`` reuses this rule and the helpers below.
ANCESTRAL_SAMPLER_SINCE_VERSION = (2, 5)

# Fully ancestral noise injection: eta=0 is a plain Euler step, eta=1 injects the full
# variance-preserving amount at every step.
ANCESTRAL_ETA = 1.0
ANCESTRAL_S_NOISE = 1.0

# The loop's noise generator is seeded from the pipeline seed plus this offset. Without it the
# loop's first draw would be bit-identical to the initial latent noise: GaussianNoiser and the
# loop's ``_get_plain_noise`` both draw ``torch.randn`` at the same shape, dtype, and device from a
# freshly seeded generator. Mirrors the substep-seed offset in ``res2s_audio_video_denoising_loop``.
ANCESTRAL_NOISE_SEED_OFFSET = 10000
# Every pass draws from its own offset, so no two of them inject the same noise.
ANCESTRAL_STAGE_2_NOISE_SEED_OFFSET = 20000


def ancestral_sampler_kwargs(seed: int, dtype: torch.dtype, noise_seed_offset: int) -> dict[str, Any]:
    """``loop`` override that samples a :class:`DiffusionStage` with ancestral Euler.
    The loop increments ``noise_seed`` on each denoise call (sequential chunks) so overlapping
    windows do not reuse ancestral noise. A single ``DiffusionStage`` call uses index 0,
    matching a fixed-seed partial.
    """
    call_index = 0
    base_seed = seed + noise_seed_offset

    def ancestral_loop(
        sigmas: torch.Tensor,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        transformer: X0Model,
        denoiser: Denoiser,
    ) -> VideoAudio[LatentState]:
        nonlocal call_index
        result = euler_ancestral_denoising_loop(
            sigmas,
            video_state,
            audio_state,
            transformer,
            denoiser,
            noise_seed=base_seed + call_index,
            model_dtype=dtype,
            eta=ANCESTRAL_ETA,
            s_noise=ANCESTRAL_S_NOISE,
        )
        call_index += 1
        return result

    return {"loop": ancestral_loop}


def should_use_ancestral_sampler(transformer_path: str) -> bool:
    """Whether a checkpoint's generation calls for the ancestral sampler.
    Takes the transformer checkpoint (``ModelPaths.transformer()``) since that is the component
    carrying ``model_version`` in both the monolith and split layouts.
    This is what the pipelines resolve at construction into ``self.use_ancestral_sampler``; a named
    function rather than an inline comparison so the rule can be tested, and reused, without
    loading a checkpoint's weights.
    """
    return detect_model_version(transformer_path) >= ANCESTRAL_SAMPLER_SINCE_VERSION


@dataclass(frozen=True)
class _DistilledRunContext:
    """Setup snapshot for :meth:`DistilledPipeline.stream_chunks`, built by :meth:`DistilledPipeline._prepare_run`."""

    generator: torch.Generator
    noiser: GaussianNoiser
    dtype: torch.dtype
    vae_dtype: torch.dtype
    #: ``images`` loaded once for the run (CRF resolved, stills decoded), so callers do not reload them.
    images: list[LoadedImageConditioning]
    video_context: torch.Tensor
    audio_context: torch.Tensor
    num_frames: int
    tiling_config: TilingConfig
    stage_1_width: int
    stage_1_height: int
    chunk_config: ChunkConfig


class DistilledPipeline:
    """
    Two-stage distilled video generation pipeline.
    Stage 1 generates video at half of the target resolution, then Stage 2 upsamples
    by 2x and refines with additional denoising steps for higher quality output.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_paths: ModelPaths,
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
        self.stage = DiffusionStage.from_checkpoint(
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
        self.use_ancestral_sampler = should_use_ancestral_sampler(model_paths.transformer())

    def _prepare_run(  # noqa: PLR0913
        self,
        *,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        color_space: EXRColorSpace | None,
        num_frames: int | AutoDuration,
        vae_dtype: torch.dtype | None,
        tiling_config: TilingConfig | AutoTiling | None,
        enhance_prompt: bool,
        enhance_static_cache: bool,
        generated_keyframes: int | Sequence[int],
        decode_with_keyframes: bool,
        chunk_config: ChunkConfig | None = None,
    ) -> _DistilledRunContext:
        """Validate inputs, encode prompts, and resolve frame count / tiling before chunk planning."""
        require_num_frames_source(num_frames, self.duration_predictor)
        enhance_prompt_image = images[0].path if images else None
        images = self.image_conditioner.load_conditionings(images, color_space)
        assert_resolution(height=height, width=width, is_two_stage=True)
        assert_generated_keyframes_request(decode_with_keyframes, generated_keyframes, self.stage)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16
        if vae_dtype is None:
            vae_dtype = dtype

        (ctx_p,) = self.prompt_encoder(
            [prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_static_cache=enhance_static_cache,
            enhance_prompt_image=enhance_prompt_image,
        )
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding

        resolved_num_frames = snap_frames_to_grid(
            resolve_num_frames(
                num_frames,
                self.duration_predictor,
                video_encoding=video_context,
                audio_encoding=audio_context,
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

        return _DistilledRunContext(
            generator=generator,
            noiser=noiser,
            dtype=dtype,
            vae_dtype=vae_dtype,
            images=images,
            video_context=video_context,
            audio_context=audio_context,
            num_frames=resolved_num_frames,
            tiling_config=resolved_tiling_config,
            stage_1_width=width // 2,
            stage_1_height=height // 2,
            chunk_config=chunk_config,
        )

    def _sampler_kwargs(self, seed: int, noise_seed_offset: int) -> dict[str, Any]:
        """Ancestral overrides, or an empty dict to leave the caller on its Euler defaults.
        The overrides are consumed by :class:`DiffusionStage` and by ``denoise_chunks``; the
        latter reuses one dict across a chunk sequence, which is what advances the ancestral
        noise seed per chunk.
        """
        if not self.use_ancestral_sampler:
            return {}
        return ancestral_sampler_kwargs(seed, self.dtype, noise_seed_offset)

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
        seed: int,
        height: int,
        width: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        num_frames: int | AutoDuration = DEFAULT_AUTO_DURATION,
        vae_dtype: torch.dtype | None = None,
        tiling_config: TilingConfig | AutoTiling | None = AUTO_TILING,
        enhance_prompt: bool = False,
        enhance_static_cache: bool = False,
        stage_1_sigmas: torch.Tensor = DISTILLED_SIGMAS,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
        color_space: EXRColorSpace | None = None,
        generated_keyframes: int | Sequence[int] = 0,
        decode_with_keyframes: bool = False,
        chunk_config: ChunkConfig | None = None,
    ) -> PipelineOutput:
        """Generate a video.
        Each stage samples according to ``self.use_ancestral_sampler``: the ancestral (SDE) Euler
        sampler on an LTX-2.5+ checkpoint, the deterministic one below that.
        ``chunk_config=None`` plans a single chunk covering the snapped clip.
        """
        chunks, num_frames, tiling_config = self.stream_chunks(
            prompt=prompt,
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
        seed: int,
        height: int,
        width: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        num_frames: int | AutoDuration = DEFAULT_AUTO_DURATION,
        vae_dtype: torch.dtype | None = None,
        tiling_config: TilingConfig | AutoTiling | None = AUTO_TILING,
        enhance_prompt: bool = False,
        enhance_static_cache: bool = False,
        stage_1_sigmas: torch.Tensor = DISTILLED_SIGMAS,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
        color_space: EXRColorSpace | None = None,
        generated_keyframes: int | Sequence[int] = 0,
        decode_with_keyframes: bool = False,
        chunk_config: ChunkConfig | None = None,
    ) -> tuple[Iterator[DecodedChunk], int, TilingConfig]:
        """Generate a video as a stream of decoded chunks for long clips.
        Each stage samples according to ``self.use_ancestral_sampler``, with its own noise-seed
        offset, exactly as in :meth:`__call__`. Within a stage the ancestral seed advances per
        chunk, so overlapping windows do not reuse noise across a seam. With
        ``decode_with_keyframes=True``, stage-2 keeps generated-keyframe slots through denoise and each
        chunk decodes through the keyframe-aware DiffVAE path.
        ``chunk_config=None`` plans a single chunk covering the snapped clip, which is
        how :meth:`__call__` runs. The returned frame count is the stitched length the chunks
        actually decode to, which floors ``num_frames`` to the causal grid the same way the
        plan does.
        """
        ctx = self._prepare_run(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            frame_rate=frame_rate,
            images=images,
            color_space=color_space,
            num_frames=num_frames,
            vae_dtype=vae_dtype,
            tiling_config=tiling_config,
            enhance_prompt=enhance_prompt,
            enhance_static_cache=enhance_static_cache,
            generated_keyframes=generated_keyframes,
            decode_with_keyframes=decode_with_keyframes,
            chunk_config=chunk_config,
        )

        stage_1_sigmas = stage_1_sigmas.to(dtype=torch.float32, device=self.device)
        stage_2_sigmas = stage_2_sigmas.to(dtype=torch.float32, device=self.device)

        target = VideoPixelShape(
            batch=1,
            frames=ctx.num_frames,
            height=ctx.stage_1_height,
            width=ctx.stage_1_width,
            fps=frame_rate,
        )
        chunks = generate_uniform_chunks(
            target=target,
            context=VideoAudio(video=ctx.video_context, audio=ctx.audio_context),
            device=self.device,
            dtype=ctx.dtype,
            config=ctx.chunk_config,
            video_scale_factors=self.stage.video_scale_factors,
            make_video_conditionings=self._video_conditionings(ctx.images, ctx.num_frames),
            generated_keyframes=generated_keyframes,
        )

        base = denoise_chunks(
            chunks,
            self.stage,
            sigmas=stage_1_sigmas,
            noiser=ctx.noiser,
            fps=frame_rate,
            noise_scale=1.0,
            **self._sampler_kwargs(seed, ANCESTRAL_NOISE_SEED_OFFSET),
        )
        upscaled = spatially_upsample_chunks(base, self.upsampler)
        refined = denoise_chunks(
            upscaled,
            self.stage,
            sigmas=stage_2_sigmas,
            noiser=ctx.noiser,
            fps=frame_rate,
            **self._sampler_kwargs(seed, ANCESTRAL_STAGE_2_NOISE_SEED_OFFSET),
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
    params = resolve_cli_params(distilled=True)
    parser = add_chunk_layout_args(
        add_keyframe_decode_arg(
            add_generated_keyframes_arg(
                default_2_stage_distilled_arg_parser(params=params, supports_auto_duration=True)
            )
        )
    )
    args = parser.parse_args()
    hdr = resolve_hdr_color_space(images=args.images, hdr=args.hdr)
    video_marker, audio_marker = markers_from_args(args, device=get_device(), color_space=hdr)
    pipeline = DistilledPipeline(
        model_paths=args.model_paths,
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
    chunks, num_frames, tiling_config = pipeline.stream_chunks(
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        images=args.images,
        vae_dtype=vae_dtype,
        color_space=hdr,
        enhance_prompt=args.enhance_prompt,
        enhance_static_cache=args.enhance_static_cache,
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
        video_marker=video_marker,
        audio_marker=audio_marker,
    )


if __name__ == "__main__":
    main()
