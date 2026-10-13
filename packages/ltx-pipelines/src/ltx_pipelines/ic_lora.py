import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import NoReturn

import torch

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.conditioning import ConditioningItem
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae import AUTO_TILING, AutoTiling, TilingConfig, VideoEncoder, get_video_chunks_number
from ltx_core.model.video_vae.transformer import DiffVAEMode
from ltx_core.quantization import QuantizationPolicy
from ltx_core.tiling import TileCountConfig, TileSizeConfig
from ltx_core.types import VideoPixelShape
from ltx_pipelines.chunks import (
    ChunkConfig,
    DecodedChunk,
    SequentialVideoFrameSource,
    decode_chunks,
    denoise_chunks,
    generate_uniform_chunks,
    pipeline_output_from_chunks,
    replace_video_conditionings,
    spatially_upsample_chunks,
    split_decoded_chunks,
)
from ltx_pipelines.chunks.conditionings import (
    assert_image_frames_in_clip,
    image_conditionings_for_chunk,
    reference_video_conditionings_for_chunk,
)
from ltx_pipelines.chunks.state import Chunk, MakeVideoConditionings
from ltx_pipelines.iclora_utils import (
    read_lora_reference_downscale_factor,
    read_lora_reference_temporal_scale_factor,
)
from ltx_pipelines.utils.args import (
    VideoConditioningAction,
    VideoMaskConditioningAction,
    add_chunk_layout_args,
    add_generated_keyframes_arg,
    add_keyframe_decode_arg,
    chunk_config_from_args,
    default_2_stage_distilled_arg_parser,
    resolve_cli_params,
)
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.constants import (
    DISTILLED_SIGMA_VALUES,
    DISTILLED_SIGMAS,
    LTX_2_3_PARAMS,
    STAGE_2_DISTILLED_SIGMA_VALUES,
    STAGE_2_DISTILLED_SIGMAS,
)
from ltx_pipelines.utils.helpers import (
    assert_generated_keyframes_request,
    assert_resolution,
    ensure_tiling_config,
    get_device,
    snap_frames_to_grid,
    tiling_scale_factors_for_vae,
)
from ltx_pipelines.utils.media_io import (
    EXRColorSpace,
    decode_video_by_frame,
    encode_video,
    is_exr_dir,
    resolve_hdr_color_space,
    vae_dtype_for_hdr,
    video_preprocess,
)
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.tiled_diffusion import (
    DiffusionTilingConfig,
    FixedSizeSpatialTiling,
    diffusion_stage_with_tiling,
)
from ltx_pipelines.utils.types import (
    ImageConditioningInput,
    OffloadMode,
    PipelineOutput,
    VideoAudio,
)


@dataclass(frozen=True)
class ICLoraStageConfig:
    """One IC-LoRA diffusion stage in a target-resolution recipe.
    Attributes:
        resolution_divisor: Stage output resolution relative to the requested output.
            ``2`` means half resolution and ``1`` means the requested resolution.
            Consecutive stages must keep the same divisor or drop by exactly 2x.
            When the divisor is unchanged the pipeline reuses the previous latent;
            when it halves (for example ``2`` to ``1``, or ``4`` to ``2``) the
            built-in 2x spatial upsampler runs.
        apply_ic_lora: Whether this stage runs in context. When true the
            pipeline's LoRAs are loaded and the reference-video conditionings
            are attached; when false the stage runs the bare checkpoint, which
            is what the original stage 2 did.
        sigmas: Diffusion sigma schedule for this stage.
        tiling: Optional transformer tiling. This is independent of VAE decode
            tiling and blends tile predictions inside every diffusion step.
    """

    resolution_divisor: int
    apply_ic_lora: bool
    sigmas: tuple[float, ...]
    tiling: DiffusionTilingConfig | None = None

    def __post_init__(self) -> None:
        if self.resolution_divisor < 1:
            raise ValueError(f"resolution_divisor must be >= 1, got {self.resolution_divisor}")
        if len(self.sigmas) < 2:
            raise ValueError("A stage sigma schedule must contain at least two values")


DEFAULT_IC_LORA_STAGES = (
    ICLoraStageConfig(resolution_divisor=2, apply_ic_lora=True, sigmas=tuple(DISTILLED_SIGMA_VALUES)),
    ICLoraStageConfig(resolution_divisor=1, apply_ic_lora=False, sigmas=tuple(STAGE_2_DISTILLED_SIGMA_VALUES)),
)
DEFAULT_IC_LORA_TILE_HEIGHT = LTX_2_3_PARAMS.stage_2_height
DEFAULT_IC_LORA_TILE_WIDTH = LTX_2_3_PARAMS.stage_2_width


def spatial_tiled_ic_lora_stages(
    *,
    tile_height: int,
    tile_width: int,
    overlap_fraction: float = 0.5,
    apply_ic_lora: tuple[bool, bool] = (True, True),
) -> tuple[ICLoraStageConfig, ICLoraStageConfig]:
    """Build the existing two-stage recipe with per-step spatial tiling.
    Tiles use the requested rectangular size on both stages. An axis smaller
    than its tile remains untiled. Overlap is expressed as a fraction of each
    axis and blended with the core tiling subsystem's trapezoidal masks.
    Unlike :data:`DEFAULT_IC_LORA_STAGES`, stage 2 keeps the IC-LoRA and its
    reference conditioning. Stage 2 enters at sigma 0.909, so an untiled stage 2
    can recover structure from the upsampled full-canvas latent alone. A tiled
    stage 2 cannot: each window sees only its own crop at normalized positions,
    so without the reference every window regenerates its own content from near
    noise and the result stops following the driving video.
    """
    if tile_height < 64 or tile_width < 64:
        raise ValueError("tile_height and tile_width must be at least 64 pixels")
    if tile_height % 32 or tile_width % 32:
        raise ValueError("tile_height and tile_width must be divisible by 32")
    if not 0.0 <= overlap_fraction <= 0.5:
        raise ValueError(f"overlap_fraction must be in [0, 0.5], got {overlap_fraction}")

    height_overlap = int(tile_height * overlap_fraction) // 32 * 32
    width_overlap = int(tile_width * overlap_fraction) // 32 * 32
    tiling = FixedSizeSpatialTiling(
        height=tile_height,
        width=tile_width,
        height_overlap=height_overlap,
        width_overlap=width_overlap,
    )
    return (
        ICLoraStageConfig(
            resolution_divisor=2,
            apply_ic_lora=apply_ic_lora[0],
            sigmas=tuple(DISTILLED_SIGMA_VALUES),
            tiling=tiling,
        ),
        ICLoraStageConfig(
            resolution_divisor=1,
            apply_ic_lora=apply_ic_lora[1],
            sigmas=tuple(STAGE_2_DISTILLED_SIGMA_VALUES),
            tiling=tiling,
        ),
    )


def resolve_ic_lora_stages(
    *,
    height: int,
    width: int,
    tile: bool = False,
    tile_height: int | None = None,
    tile_width: int | None = None,
    stage_2_ic_lora: bool = False,
    skip_stage_2: bool = False,
) -> tuple[ICLoraStageConfig, ...]:
    """Two-stage CLI recipe: optional transformer tiling, optional stage-2 IC-LoRA.
    The CLI is always this two-stage ladder. Longer recipes use
    ``ICLoraPipeline(stages=...)`` in Python.
    ``tile_height`` / ``tile_width`` require ``tile``. When tiling is on they
    default to :data:`DEFAULT_IC_LORA_TILE_HEIGHT` /
    :data:`DEFAULT_IC_LORA_TILE_WIDTH`.
    """
    if not tile:
        if tile_height is not None or tile_width is not None:
            raise ValueError("--tile-height and --tile-width require --tile")
        if not stage_2_ic_lora:
            return DEFAULT_IC_LORA_STAGES
        return (
            DEFAULT_IC_LORA_STAGES[0],
            replace(DEFAULT_IC_LORA_STAGES[1], apply_ic_lora=True),
        )

    resolved_height = DEFAULT_IC_LORA_TILE_HEIGHT if tile_height is None else tile_height
    resolved_width = DEFAULT_IC_LORA_TILE_WIDTH if tile_width is None else tile_width
    stage_2_is_tiled = (not skip_stage_2) and (height > resolved_height or width > resolved_width)
    if stage_2_is_tiled and not stage_2_ic_lora:
        raise ValueError("Tiled full-resolution stages need IC-LoRA on that stage. Pass --stage-2-ic-lora.")
    return spatial_tiled_ic_lora_stages(
        tile_height=resolved_height,
        tile_width=resolved_width,
        apply_ic_lora=(True, stage_2_ic_lora),
    )


def _reference_encode_tiling(
    stage: ICLoraStageConfig,
    tiling_config: TilingConfig | AutoTiling | None,
) -> TilingConfig | AutoTiling | None:
    """Tiling for a stage's guide encode.
    Only a tiled stage encodes its guide in tiles. An untiled stage encodes the
    full canvas in one pass, so it must not inherit the pipeline's decode tiling:
    that would change the reference latent, and with it every default-recipe run.
    """
    return tiling_config if stage.tiling is not None else None


def _concrete_reference_encode_tiling(
    stage: ICLoraStageConfig,
    tiling_config: TilingConfig | AutoTiling | None,
) -> TilingConfig | None:
    """Reference VAE encode tiling for chunked IC-LoRA (concrete config only)."""
    resolved = _reference_encode_tiling(stage, tiling_config)
    if resolved is None:
        return None
    if isinstance(resolved, TileSizeConfig | TileCountConfig):
        return resolved
    return None


_IC_LORA_OOM = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)


def reraise_ic_lora_oom(exc: BaseException, *, tile: bool) -> NoReturn:
    """Re-raise GPU OOM with the CLI flags that usually recover the run."""
    if tile:
        hint = "GPU out of memory. Try smaller --tile-height / --tile-width, or --offload cpu."
    else:
        hint = "GPU out of memory. For large --height/--width pass --tile, or --offload cpu."
    raise type(exc)(f"{exc}\n\n{hint}") from exc


@dataclass(frozen=True)
class _ICLoraRunContext:
    generator: torch.Generator
    noiser: GaussianNoiser
    dtype: torch.dtype
    vae_dtype: torch.dtype
    images: list[ImageConditioningInput]
    video_context: torch.Tensor
    audio_context: torch.Tensor
    num_frames: int
    height: int
    width: int
    frame_rate: float
    tiling_config: TilingConfig
    active_stages: tuple[ICLoraStageConfig, ...]
    video_conditioning: list[tuple[str, float]]
    conditioning_attention_strength: float
    conditioning_attention_mask: torch.Tensor | None
    decode_with_keyframes: bool
    color_space: EXRColorSpace | None
    chunk_config: ChunkConfig


class ICLoraPipeline:
    """
    Two-stage video generation pipeline with In-Context (IC) LoRA support.
    Allows conditioning the generated video on control signals such as depth maps,
    human pose, or image edges via the video_conditioning parameter.
    The specific IC-LoRA model should be provided via the loras parameter.
    Stage 1 generates video at half of the target resolution, then Stage 2 upsamples
    by 2x and refines with additional denoising steps for higher quality output.
    Both stages use distilled models for efficiency.
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
        stages: Sequence[ICLoraStageConfig] | None = None,
    ) -> None:
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
        self._loras = tuple(loras)
        self.stages = tuple(stages) if stages is not None else DEFAULT_IC_LORA_STAGES
        self._validate_stages(self.stages)

        self._diffusion_stage = DiffusionStage.from_checkpoint(
            model_paths.transformer(),
            self.dtype,
            self.device,
            loras=(),
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

        # Read reference scale factors from LoRA metadata.
        # IC-LoRAs trained with scaled reference videos store these factors
        # so inference can resize/subsample reference videos to match training conditions.
        self.reference_downscale_factor = 1
        self.reference_temporal_scale_factor = 1
        for lora in self._loras:
            scale = read_lora_reference_downscale_factor(lora.path)
            if scale != 1:
                if self.reference_downscale_factor not in (1, scale):
                    raise ValueError(
                        f"Conflicting reference_downscale_factor values in LoRAs: "
                        f"already have {self.reference_downscale_factor}, but {lora.path} "
                        f"specifies {scale}. Cannot combine LoRAs with different reference scales."
                    )
                self.reference_downscale_factor = scale
            temporal = read_lora_reference_temporal_scale_factor(lora.path)
            if temporal != 1:
                if self.reference_temporal_scale_factor not in (1, temporal):
                    raise ValueError(
                        f"Conflicting reference_temporal_scale_factor values in LoRAs: "
                        f"already have {self.reference_temporal_scale_factor}, but {lora.path} "
                        f"specifies {temporal}. Cannot combine LoRAs with different temporal scales."
                    )
                self.reference_temporal_scale_factor = temporal

    def _resolve_active_stages(
        self,
        *,
        skip_stage_2: bool,
        stage_1_sigmas: torch.Tensor,
        stage_2_sigmas: torch.Tensor,
    ) -> tuple[ICLoraStageConfig, ...]:
        active_stages = self.stages
        if self.stages is DEFAULT_IC_LORA_STAGES:
            active_stages = tuple(
                ICLoraStageConfig(
                    resolution_divisor=stage.resolution_divisor,
                    apply_ic_lora=stage.apply_ic_lora,
                    sigmas=tuple(sigmas.tolist()),
                    tiling=stage.tiling,
                )
                for stage, sigmas in zip(
                    self.stages,
                    (stage_1_sigmas, stage_2_sigmas),
                    strict=True,
                )
            )
        if skip_stage_2:
            active_stages = active_stages[:1]
        return active_stages

    def _prepare_run(  # noqa: PLR0913
        self,
        *,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        video_conditioning: list[tuple[str, float]],
        enhance_prompt: bool,
        enhance_static_cache: bool,
        vae_dtype: torch.dtype | None,
        tiling_config: TilingConfig | AutoTiling | None,
        conditioning_attention_strength: float,
        skip_stage_2: bool,
        conditioning_attention_mask: torch.Tensor | None,
        stage_1_sigmas: torch.Tensor,
        stage_2_sigmas: torch.Tensor,
        color_space: EXRColorSpace | None,
        generated_keyframes: int | Sequence[int] = 0,
        decode_with_keyframes: bool = False,
        chunk_config: ChunkConfig | None = None,
    ) -> _ICLoraRunContext:
        images = self.image_conditioner.resolve_crf(images)
        assert_generated_keyframes_request(
            decode_with_keyframes,
            generated_keyframes,
            self._diffusion_stage,
        )
        active_stages = self._resolve_active_stages(
            skip_stage_2=skip_stage_2,
            stage_1_sigmas=stage_1_sigmas,
            stage_2_sigmas=stage_2_sigmas,
        )
        first_divisor = active_stages[0].resolution_divisor
        assert_resolution(
            height=height,
            width=width,
            is_two_stage=len(active_stages) > 1,
            divisor=32 * first_divisor,
        )
        if not (0.0 <= conditioning_attention_strength <= 1.0):
            raise ValueError(
                f"conditioning_attention_strength must be in [0.0, 1.0], got {conditioning_attention_strength}"
            )

        resolved_num_frames = snap_frames_to_grid(num_frames)

        if chunk_config is None:
            chunk_config = ChunkConfig(chunk_pixel_frames=resolved_num_frames, next_video_carry_frames=0)

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

        return _ICLoraRunContext(
            generator=generator,
            noiser=noiser,
            dtype=self.dtype,
            vae_dtype=resolved_vae_dtype,
            images=images,
            video_context=video_context,
            audio_context=audio_context,
            num_frames=resolved_num_frames,
            height=height,
            width=width,
            frame_rate=frame_rate,
            tiling_config=resolved_tiling_config,
            active_stages=active_stages,
            video_conditioning=video_conditioning,
            conditioning_attention_strength=conditioning_attention_strength,
            conditioning_attention_mask=conditioning_attention_mask,
            decode_with_keyframes=decode_with_keyframes,
            color_space=color_space,
            chunk_config=chunk_config,
        )

    def _video_conditionings(
        self,
        ctx: _ICLoraRunContext,
        stage_config: ICLoraStageConfig,
        frame_sources: dict[str, SequentialVideoFrameSource],
    ) -> MakeVideoConditionings:
        assert_image_frames_in_clip(ctx.images, ctx.num_frames)
        encode_tiling = _concrete_reference_encode_tiling(stage_config, ctx.tiling_config)

        def make(chunk: Chunk) -> list[ConditioningItem]:
            if chunk.video is None:
                return []

            def encode(enc: VideoEncoder) -> list[ConditioningItem]:
                items = [
                    *image_conditionings_for_chunk(
                        chunk,
                        images=ctx.images,
                        video_encoder=enc,
                        color_space=ctx.color_space,
                    ),
                    *reference_video_conditionings_for_chunk(
                        chunk,
                        video_conditioning=ctx.video_conditioning if stage_config.apply_ic_lora else (),
                        video_encoder=enc,
                        downscale_factor=self.reference_downscale_factor,
                        reference_temporal_scale_factor=self.reference_temporal_scale_factor,
                        conditioning_attention_strength=ctx.conditioning_attention_strength,
                        conditioning_attention_mask=ctx.conditioning_attention_mask,
                        encode_tiling=encode_tiling,
                        color_space=ctx.color_space,
                        frame_sources=frame_sources,
                    ),
                ]
                return items

            return self.image_conditioner(encode)

        return make

    @staticmethod
    def _validate_stages(stages: Sequence[ICLoraStageConfig]) -> None:
        if not stages:
            raise ValueError("IC-LoRA requires at least one diffusion stage")
        for current, following in pairwise(stages):
            same_res = following.resolution_divisor == current.resolution_divisor
            upsample_2x = current.resolution_divisor == following.resolution_divisor * 2
            if not (same_res or upsample_2x):
                raise ValueError(
                    "Consecutive IC-LoRA stages must keep the same resolution_divisor "
                    "or drop by exactly 2x, "
                    f"got {current.resolution_divisor} then {following.resolution_divisor}"
                )
        if stages[-1].resolution_divisor != 1:
            raise ValueError(
                "The final IC-LoRA stage must run at the requested resolution "
                f"(resolution_divisor=1), got {stages[-1].resolution_divisor}"
            )

    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        video_conditioning: list[tuple[str, float]],
        enhance_prompt: bool = False,
        enhance_static_cache: bool = False,
        vae_dtype: torch.dtype | None = None,
        tiling_config: TilingConfig | AutoTiling | None = AUTO_TILING,
        conditioning_attention_strength: float = 1.0,
        skip_stage_2: bool = False,
        conditioning_attention_mask: torch.Tensor | None = None,
        stage_1_sigmas: torch.Tensor = DISTILLED_SIGMAS,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
        generated_keyframes: int | Sequence[int] = 0,
        decode_with_keyframes: bool = False,
        color_space: EXRColorSpace | None = None,
        chunk_config: ChunkConfig | None = None,
    ) -> PipelineOutput:
        """
        Generate video with IC-LoRA conditioning.
        Args:
            prompt: Text prompt for video generation.
            seed: Random seed for reproducibility.
            height: Output video height in pixels (must be divisible by 64).
            width: Output video width in pixels (must be divisible by 64).
            num_frames: Number of frames to generate.
            frame_rate: Output video frame rate.
            images: List of (path, frame_idx, strength) tuples for image conditioning.
            video_conditioning: List of (path, strength) tuples for IC-LoRA video conditioning.
            enhance_prompt: Whether to enhance the prompt using the text encoder.
            tiling_config: Optional tiling configuration for VAE decoding.
            conditioning_attention_strength: Scale factor for IC-LoRA conditioning attention.
                Controls how strongly the conditioning video influences the output.
                0.0 = ignore conditioning, 1.0 = full conditioning influence. Default 1.0.
                When conditioning_attention_mask is provided, the mask is multiplied by
                this strength before being passed to the conditioning items.
            skip_stage_2: If True, skip Stage 2 upsampling and refinement. Output will be
                at half resolution (height//2, width//2). Default is False.
            conditioning_attention_mask: Optional pixel-space attention mask with the same
                spatial-temporal dimensions as the input reference video. Shape should be
                (B, 1, F, H, W) or (1, 1, F, H, W) where F, H, W match the reference
                video's pixel dimensions. Values in [0, 1].
                The mask is downsampled to latent space using VAE scale factors (with
                causal temporal handling for the first frame), then multiplied by
                conditioning_attention_strength.
                When None (default): scalar conditioning_attention_strength is used
                directly.
            generated_keyframes: Per-chunk slot budget (int) or explicit global pixel indices (with seam anchors).
            decode_with_keyframes: Decode with generated slots as DiffVAE anchors and carry
                overlap planes into the following chunk as fixed keyframe conditioning.
            chunk_config: Optional temporal chunk layout. ``None`` plans one window over
                the snapped clip (same output as before chunk streaming existed).
        Returns:
            PipelineOutput with decoded video and audio.
        """
        chunks, num_frames, tiling_config = self.stream_chunks(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            video_conditioning=video_conditioning,
            enhance_prompt=enhance_prompt,
            enhance_static_cache=enhance_static_cache,
            vae_dtype=vae_dtype,
            tiling_config=tiling_config,
            conditioning_attention_strength=conditioning_attention_strength,
            skip_stage_2=skip_stage_2,
            conditioning_attention_mask=conditioning_attention_mask,
            stage_1_sigmas=stage_1_sigmas,
            stage_2_sigmas=stage_2_sigmas,
            generated_keyframes=generated_keyframes,
            decode_with_keyframes=decode_with_keyframes,
            color_space=color_space,
            chunk_config=chunk_config,
        )
        return pipeline_output_from_chunks(chunks, num_frames=num_frames, tiling_config=tiling_config)

    def stream_chunks(  # noqa: PLR0913
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        video_conditioning: list[tuple[str, float]],
        enhance_prompt: bool = False,
        enhance_static_cache: bool = False,
        vae_dtype: torch.dtype | None = None,
        tiling_config: TilingConfig | AutoTiling | None = AUTO_TILING,
        conditioning_attention_strength: float = 1.0,
        skip_stage_2: bool = False,
        conditioning_attention_mask: torch.Tensor | None = None,
        stage_1_sigmas: torch.Tensor = DISTILLED_SIGMAS,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
        generated_keyframes: int | Sequence[int] = 0,
        decode_with_keyframes: bool = False,
        color_space: EXRColorSpace | None = None,
        chunk_config: ChunkConfig | None = None,
    ) -> tuple[Iterator[DecodedChunk], int, TilingConfig]:
        """Generate a video as a stream of decoded temporal chunks.
        ``chunk_config=None`` plans a single chunk covering the snapped clip, which is
        how :meth:`__call__` runs. Video conditionings are attached per diffusion stage
        immediately before each :func:`~ltx_pipelines.chunks.denoise_chunks` pass.
        Temporal chunking composes with per-stage spatial transformer tiling inside
        each denoise step.
        """
        ctx = self._prepare_run(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            video_conditioning=video_conditioning,
            enhance_prompt=enhance_prompt,
            enhance_static_cache=enhance_static_cache,
            vae_dtype=vae_dtype,
            tiling_config=tiling_config,
            conditioning_attention_strength=conditioning_attention_strength,
            skip_stage_2=skip_stage_2,
            conditioning_attention_mask=conditioning_attention_mask,
            stage_1_sigmas=stage_1_sigmas,
            stage_2_sigmas=stage_2_sigmas,
            color_space=color_space,
            generated_keyframes=generated_keyframes,
            decode_with_keyframes=decode_with_keyframes,
            chunk_config=chunk_config,
        )

        target = VideoPixelShape(
            batch=1,
            frames=ctx.num_frames,
            height=ctx.height // ctx.active_stages[0].resolution_divisor,
            width=ctx.width // ctx.active_stages[0].resolution_divisor,
            fps=ctx.frame_rate,
        )
        frame_sources: dict[str, SequentialVideoFrameSource] = {
            path: SequentialVideoFrameSource(path, self.device)
            for path, _ in ctx.video_conditioning
            if not is_exr_dir(path)
        }

        stage_1_divisor = ctx.active_stages[0].resolution_divisor
        chunks: Iterator[Chunk] = generate_uniform_chunks(
            target=target._replace(
                height=ctx.height // stage_1_divisor,
                width=ctx.width // stage_1_divisor,
            ),
            context=VideoAudio(video=ctx.video_context, audio=ctx.audio_context),
            device=self.device,
            dtype=ctx.dtype,
            config=ctx.chunk_config,
            video_scale_factors=self._diffusion_stage.video_scale_factors,
            generated_keyframes=generated_keyframes,
        )

        for stage_index, stage_config in enumerate(ctx.active_stages):
            if stage_index > 0:
                prev_config = ctx.active_stages[stage_index - 1]
                if prev_config.resolution_divisor > stage_config.resolution_divisor:
                    if prev_config.resolution_divisor != stage_config.resolution_divisor * 2:
                        raise ValueError(
                            "Unsupported resolution transition from divisor "
                            f"{prev_config.resolution_divisor} to {stage_config.resolution_divisor}: "
                            "spatial upsampling only supports 2x"
                        )
                    chunks = spatially_upsample_chunks(chunks, self.upsampler)
            chunks = replace_video_conditionings(
                chunks,
                self._video_conditionings(ctx, stage_config, frame_sources),
            )

            sigmas = torch.tensor(stage_config.sigmas, dtype=torch.float32, device=self.device)
            stage = diffusion_stage_with_tiling(
                self._diffusion_stage.with_loras(self._loras if stage_config.apply_ic_lora else ()),
                stage_config.tiling,
            )
            stage_height = ctx.height // stage_config.resolution_divisor
            stage_width = ctx.width // stage_config.resolution_divisor
            logging.info(
                "[IC-LoRA] Stage %d/%d: %dx%d, ic_lora=%s, tiled=%s",
                stage_index + 1,
                len(ctx.active_stages),
                stage_width,
                stage_height,
                stage_config.apply_ic_lora,
                stage_config.tiling is not None,
            )
            chunks = denoise_chunks(
                chunks,
                stage,
                sigmas=sigmas,
                noiser=ctx.noiser,
                fps=ctx.frame_rate,
            )

        def decoded() -> Iterator[DecodedChunk]:
            try:
                yield from decode_chunks(
                    chunks,
                    self.video_decoder,
                    self.audio_decoder,
                    fps=ctx.frame_rate,
                    tiling_config=ctx.tiling_config,
                    generator=ctx.generator,
                    dtype=ctx.vae_dtype,
                    keyframes=ctx.decode_with_keyframes,
                )
            finally:
                for source in frame_sources.values():
                    source.close()

        return (decoded(), ctx.num_frames, ctx.tiling_config)


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    params = resolve_cli_params(distilled=True)
    parser = add_chunk_layout_args(
        add_keyframe_decode_arg(add_generated_keyframes_arg(default_2_stage_distilled_arg_parser(params=params)))
    )
    parser.add_argument(
        "--video-conditioning",
        action=VideoConditioningAction,
        nargs=2,
        metavar=("PATH", "STRENGTH"),
        required=True,
        help=(
            "IC-LoRA reference: video file (SDR) or directory of scene-linear *.exr frames (HDR), "
            "plus strength. Example: --video-conditioning ref.mp4 1.0  or  --video-conditioning exr_dir/ 1.0"
        ),
    )
    parser.add_argument(
        "--conditioning-attention-mask",
        action=VideoMaskConditioningAction,
        nargs=2,
        metavar=("MASK_PATH", "STRENGTH"),
        default=None,
        help=(
            "Optional spatial attention mask: path to a grayscale mask video and "
            "attention strength. The mask video pixel values in [0,1] control "
            "per-region conditioning attention strength. The strength scalar is "
            "multiplied with the spatial mask. "
            "0.0 = ignore IC-LoRA conditioning, 1.0 = full conditioning influence. "
            "When not provided, full conditioning strength (1.0) is used. "
            "Example: --conditioning-attention-mask path/to/mask.mp4 0.5"
        ),
    )
    parser.add_argument(
        "--skip-stage-2",
        action="store_true",
        help=(
            "Skip Stage 2 upsampling and refinement. Output will be at half resolution "
            "(height//2, width//2). Useful for faster iteration or when GPU memory is limited."
        ),
    )
    parser.add_argument(
        "--stage-2-ic-lora",
        action="store_true",
        help=(
            "Keep the IC-LoRA and reference video on stage 2 as well as stage 1. "
            "Default stage 2 runs the bare checkpoint. Required when --tile makes stage 2 "
            "larger than the tile window."
        ),
    )
    parser.add_argument(
        "--tile",
        action="store_true",
        help=(
            "Tile each transformer step over a fixed spatial window so large --height/--width "
            "fits in memory. Off by default; 1080-class sizes stay the untiled two-stage recipe. "
            f"Default window is {DEFAULT_IC_LORA_TILE_HEIGHT}x{DEFAULT_IC_LORA_TILE_WIDTH} "
            "(override with --tile-height / --tile-width)."
        ),
    )
    parser.add_argument(
        "--tile-height",
        type=int,
        default=None,
        help=(
            f"Tile window height in pixels when --tile is set (default: {DEFAULT_IC_LORA_TILE_HEIGHT}). "
            "Must be at least 64 and divisible by 32. Requires --tile."
        ),
    )
    parser.add_argument(
        "--tile-width",
        type=int,
        default=None,
        help=(
            f"Tile window width in pixels when --tile is set (default: {DEFAULT_IC_LORA_TILE_WIDTH}). "
            "Must be at least 64 and divisible by 32. Requires --tile."
        ),
    )
    args = parser.parse_args()

    # Load mask video if provided via --conditioning-attention-mask
    conditioning_attention_mask = None
    conditioning_attention_strength = 1.0
    if args.conditioning_attention_mask is not None:
        mask_path, mask_strength = args.conditioning_attention_mask
        conditioning_attention_strength = mask_strength
        conditioning_attention_mask = _load_mask_video(
            mask_path=mask_path,
            height=args.height // 2,  # Stage 1 operates at half resolution
            width=args.width // 2,
            num_frames=args.num_frames,
        )

    try:
        pipeline = ICLoraPipeline(
            model_paths=args.model_paths,
            spatial_upsampler_path=args.spatial_upsampler_path,
            loras=tuple(args.lora) if args.lora else (),
            quantization=args.quantization,
            compilation_config=args.compile,
            offload_mode=args.offload_mode,
            prompt_enhancer_gemma_root=args.prompt_enhancer_gemma_root,
            diffvae_optimization=args.diffvae_optimization,
            stages=resolve_ic_lora_stages(
                height=args.height,
                width=args.width,
                tile=args.tile,
                tile_height=args.tile_height,
                tile_width=args.tile_width,
                stage_2_ic_lora=args.stage_2_ic_lora,
                skip_stage_2=args.skip_stage_2,
            ),
        )
        hdr = resolve_hdr_color_space(
            images=args.images,
            video_paths=[path for path, _ in args.video_conditioning],
            hdr=args.hdr,
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
            video_conditioning=args.video_conditioning,
            enhance_prompt=args.enhance_prompt,
            enhance_static_cache=args.enhance_static_cache,
            vae_dtype=vae_dtype,
            color_space=hdr,
            tiling_config=AUTO_TILING,
            conditioning_attention_strength=conditioning_attention_strength,
            skip_stage_2=args.skip_stage_2,
            conditioning_attention_mask=conditioning_attention_mask,
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
    except _IC_LORA_OOM as exc:
        reraise_ic_lora_oom(exc, tile=args.tile)


def _load_mask_video(
    mask_path: str,
    height: int,
    width: int,
    num_frames: int,
) -> torch.Tensor:
    """Load a mask video and return a pixel-space tensor of shape (1, 1, F, H, W).
    The mask video is loaded, resized to (height, width), converted to
    grayscale, and normalised to [0, 1].
    Args:
        mask_path: Path to the mask video file.
        height: Target height in pixels.
        width: Target width in pixels.
        num_frames: Maximum number of frames to load.
    Returns:
        Tensor of shape ``(1, 1, F, H, W)`` with values in ``[0, 1]``.
    """
    device = get_device()
    frame_gen = decode_video_by_frame(path=mask_path, frame_cap=num_frames, device=device)
    mask_video = video_preprocess(frame_gen, height, width, torch.bfloat16, device)
    # mask_video shape: (1, C, F, H, W) — take mean over channels for grayscale
    mask = mask_video.mean(dim=1, keepdim=True)  # (1, 1, F, H, W)
    # Normalise to [0, 1] — video_preprocess applies normalize_latent,
    # so undo that: values are in [-1, 1], remap to [0, 1]
    mask = (mask + 1.0) / 2.0
    return mask.clamp(0.0, 1.0)


if __name__ == "__main__":
    main()
