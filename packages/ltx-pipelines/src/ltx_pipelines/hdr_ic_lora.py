"""Single-GPU IC-LoRA from SDR video or EXR frames to HDR (HLG MP4 and EXR sequence)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import torch
from safetensors import safe_open

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.conditioning import (
    ConditioningItem,
    VideoConditionByKeyframeIndex,
    VideoConditionByReferenceLatent,
)
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.model.video_vae import AUTO_TILING, AutoTiling, TilingConfig, VideoEncoder
from ltx_core.model.video_vae.keyframes import DecodeKeyframes
from ltx_core.model.video_vae.transformer import DiffVAEMode
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import LatentState, SpatioTemporalScaleFactors, VideoPixelShape
from ltx_pipelines.dfr_helpers.layout import resolve_canvas
from ltx_pipelines.utils.args import (
    hdr_ic_lora_arg_parser,
    markers_from_args,
)
from ltx_pipelines.utils.blocks import DiffusionStage, ImageConditioner, VideoDecoder
from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import (
    assert_stage_supports_generated_keyframes,
    decode_keyframes_from_slots,
    ensure_tiling_config,
    generated_keyframe_conditionings,
    get_device,
    tiling_scale_factors_for_vae,
)
from ltx_pipelines.utils.media_io import (
    EXRVideoInput,
    ResizeMode,
    VideoInput,
    align_resolution,
    encode_video,
    get_videostream_metadata,
    load_exr_as_hdr_conditioning,
    load_video_as_hdr_conditioning,
)
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.quantization_factory import QuantizationKind
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode, VideoAudio

logger = logging.getLogger(__name__)

MIN_RESOLUTION = 32
ALIGNMENT_DIVISOR = 32

# Spatial area (H x W) above which conditioning encode uses ``tiled_encode``.
# Empirical VRAM gate (~80 GB / H100): below this, a single full-frame encode is
# assumed safe; above it, tile. Not derived from VAE scale factors or AUTO_TILING.
# On lower-VRAM GPUs pass a smaller value (e.g. ``256 * 256``) to the constructor.
TILED_VAE_ENCODE_PIXEL_THRESHOLD = 512 * 768

# Full distilled schedule (8 Euler steps, 9 sigma values including 1.0 and 0.0).
DEFAULT_DENOISE_SIGMAS = list(DISTILLED_SIGMA_VALUES)
_DEFAULT_QUANTIZATION = QuantizationKind.FP8_CAST
# Same pin as ``dfr_pipeline._ANCHOR_KEYFRAME_STRENGTH``.
DEFAULT_KEYFRAME_STRENGTH = 0.95
_SNAP_CONDITIONING_FPS_ABOVE = 30.0
_HIGH_FPS_CONDITIONING_FPS = 30.0


def _conditioning_fps(playback_fps: float) -> float:
    """RoPE fps for this pipeline."""
    if playback_fps > _SNAP_CONDITIONING_FPS_ABOVE:
        return _HIGH_FPS_CONDITIONING_FPS
    return playback_fps


def _dfr_seam_pixel_positions(num_frames: int, *, high_quality_hdr: bool) -> list[int]:
    """DFR x8-border segment seams from :func:`resolve_canvas`, clipped to the real clip.
    A 1-frame source is a valid 8k+1 clip with no interior seams, so this returns ``[]``
    rather than calling :func:`resolve_canvas`, which rejects ``num_frames < 2``.
    """
    if num_frames < 2:
        return []
    _canvas, _segment, positions = resolve_canvas(num_frames)
    positions = [int(p) for p in positions if int(p) < num_frames]
    if high_quality_hdr:
        positions = [2 * p for p in positions]
    return positions


def _seam_roles(positions: list[int]) -> tuple[list[int], list[int]]:
    """Assign canvas seams to generated HDR slots and SDR keyframe conditions.
    Every seam takes **both** roles: it gets a generated HDR slot and, at the same
    position, a 1-frame SDR guide. ``S=24``: generated ``[24, 48, 72, …]``, SDR kf
    ``[24, 48, 72, …]``.
    """
    return list(positions), list(positions)


def dfr_seam_roles(num_frames: int, *, high_quality_hdr: bool) -> tuple[list[int], list[int]]:
    """Generated HDR slot indices and SDR guide indices for an HDR DFR run."""
    return _seam_roles(_dfr_seam_pixel_positions(num_frames, high_quality_hdr=high_quality_hdr))


def _keyframe_conditionings_from_pixel_frames(
    video_encoder: VideoEncoder,
    pixel_video: torch.Tensor,
    positions: list[int],
    strength: float,
    *,
    gen_h: int,
    gen_w: int,
    tiling_config: TilingConfig | None,
    tiled_threshold: int,
    dtype: torch.dtype | None = None,
) -> list[ConditioningItem]:
    """VAE-encode each seam SDR frame as a true 1-frame (L0-style) keyframe latent."""
    if pixel_video.ndim != 5:
        raise ValueError(f"Expected pixel video (B, C, T, H, W), got {tuple(pixel_video.shape)}")
    num_frames = pixel_video.shape[2]
    use_tiled = tiling_config is not None and gen_h * gen_w > tiled_threshold
    conditionings: list[ConditioningItem] = []
    for frame_idx in positions:
        idx = int(frame_idx)
        if idx < 0 or idx >= num_frames:
            raise ValueError(f"Seam frame_idx={idx} out of range for pixel video T={num_frames}")
        frame = pixel_video[:, :, idx : idx + 1]
        encoded = video_encoder.tiled_encode(frame, tiling_config) if use_tiled else video_encoder(frame)
        if dtype is not None:
            encoded = encoded.to(dtype=dtype)
        conditionings.append(
            VideoConditionByKeyframeIndex(
                keyframes=encoded,
                frame_idx=idx,
                strength=strength,
                num_pixel_frames=1,
            )
        )
    return conditionings


def _load_video_context(path: str | Path, device: torch.device) -> torch.Tensor:
    """Load ``video_context`` (or trainer ``video_prompt_embeds``) from ``.safetensors``."""
    emb_path = Path(path)
    if not emb_path.is_file():
        raise FileNotFoundError(f"Text embeddings not found: {emb_path}")
    with safe_open(emb_path, framework="pt", device=str(device)) as f:
        keys = list(f.keys())
        for name in ("video_context", "video_prompt_embeds"):
            if name in keys:
                return f.get_tensor(name)
    raise KeyError(f"video_context/video_prompt_embeds not found in {emb_path} (keys={keys})")


class HDRICLoraPipeline:
    """Single-GPU IC-LoRA from SDR to HDR in one denoise stage.
    The model works in ACEScct; export is an HLG MP4 plus an EXR sequence.
    Pass ``keyframe_strength`` to ``__call__`` for DFR-style seam keyframes
    (generated HDR slots + 1-frame SDR guides + keyframe-aware DiffVAE decode).
    ``None`` is plain IC-LoRA.
    """

    def __init__(
        self,
        model_paths: ModelPaths,
        hdr_lora: str | Path,
        text_embeddings_path: str | Path,
        device: torch.device | None = None,
        quantization: QuantizationPolicy | QuantizationKind | None = _DEFAULT_QUANTIZATION,
        registry: Registry | None = None,
        tiled_vae_encode_pixel_threshold: int = TILED_VAE_ENCODE_PIXEL_THRESHOLD,
        offload_mode: OffloadMode = OffloadMode.NONE,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
        diffvae_optimization: DiffVAEMode = DiffVAEMode.CHUNKED_EAGER,
    ) -> None:
        self.device = device or get_device()
        self._tiled_vae_encode_threshold = tiled_vae_encode_pixel_threshold
        if isinstance(quantization, QuantizationKind):
            quantization = quantization.to_policy(checkpoint_path=model_paths.transformer())
        if offload_mode != OffloadMode.NONE and quantization is not None:
            logger.info("Offload mode enabled — disabling quantization.")
            quantization = None

        self.dtype = torch.bfloat16
        self.vae_dtype = torch.float32

        lora_path = str(Path(hdr_lora).resolve())
        loras = (LoraPathStrengthAndSDOps(lora_path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP),)

        logger.info("Loading text embeddings from %s", text_embeddings_path)
        self.video_context = _load_video_context(text_embeddings_path, self.device)

        vae_ckpt = model_paths.video_vae()
        self.image_conditioner = ImageConditioner(
            vae_ckpt,
            self.vae_dtype,
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
            offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.video_decoder = VideoDecoder(
            vae_ckpt,
            self.vae_dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
            diffvae_optimization=diffvae_optimization,
        )

        logger.info("[HDR IC-LoRA] ACEScct ready (vae_dtype=%s)", self.vae_dtype)

    @torch.inference_mode()
    def __call__(
        self,
        video: VideoInput | EXRVideoInput,
        seed: int,
        *,
        tiling_config: TilingConfig | AutoTiling | None = AUTO_TILING,
        denoise_sigmas: list[float] | None = None,
        conditioning_strength: float = 1.0,
        high_quality_hdr: bool = False,
        keyframe_strength: float | None = None,
    ) -> tuple[torch.Tensor, float]:
        """Generate ACEScct HDR from an MP4/MOV or EXR-frame folder.
        Args:
            video: :class:`VideoInput` (MP4/MOV) or :class:`EXRVideoInput` (EXR folder).
            seed: RNG seed.
            tiling_config: ``AUTO_TILING`` (default), a concrete config, or ``None`` (untiled).
            denoise_sigmas: Override default ``DEFAULT_DENOISE_SIGMAS``.
            conditioning_strength: IC-LoRA reference strength (default 1.0).
            high_quality_hdr: Duplicate each conditioning frame and generate at
                ``2N-1`` frames, then keep every other output frame. Reduces
                temporal artifacts at ~2x generation cost.
            keyframe_strength: ``None`` for plain IC-LoRA. A float puts a generated slot
                and a 1-frame SDR guide on every ``resolve_canvas`` seam at that strength.
        Returns:
            ``(acescct_hdr, fps)`` where ``acescct_hdr`` is float ``[F, H, W, C]`` in
            ``[0, 1]`` (FHWC for export) and ``fps`` is the source frame rate.
        """
        match video:
            case EXRVideoInput(dir=video_path, frame_rate=fps):
                video_path = Path(video_path)
            case VideoInput(path=video_path):
                video_path = Path(video_path)
                fps = None
        meta = get_videostream_metadata(str(video_path), fps=fps)

        # Assert 8k+1 frame count.
        video_scale = SpatioTemporalScaleFactors.default()
        if (meta.frames - 1) % video_scale.time != 0:
            snapped = ((meta.frames - 1) // video_scale.time) * video_scale.time + 1
            raise ValueError(
                f"Video frame count must satisfy 8k+1 (e.g. 97, 193). "
                f"Got {meta.frames}; use a video with {snapped} frames."
            )

        # Pad W/H up to a multiple of 32, croped back after decode.
        gen_w, gen_h, crop_w, crop_h = align_resolution(
            meta.width, meta.height, ResizeMode.REFLECT_PAD, divisor=ALIGNMENT_DIVISOR
        )
        if gen_h < MIN_RESOLUTION or gen_w < MIN_RESOLUTION:
            raise ValueError(
                f"Resolution ({meta.width}x{meta.height}) too small after alignment "
                f"(got {gen_w}x{gen_h}, need >= {MIN_RESOLUTION})."
            )

        # Generate 2N-1 frames for high-quality HDR, N frames otherwise.
        num_frames = meta.frames
        gen_frames = 2 * num_frames - 1 if high_quality_hdr else num_frames

        generated_kf: list[int] = []
        guides_kf: list[int] = []
        if keyframe_strength is not None:
            generated_kf, guides_kf = dfr_seam_roles(num_frames, high_quality_hdr=high_quality_hdr)
            assert_stage_supports_generated_keyframes(self.stage)
            if not generated_kf and not guides_kf:
                logger.warning(
                    "[HDR IC-LoRA] keyframe_strength=%.2f but this %d-frame clip has no DFR seams "
                    "after clipping the canvas; running plain IC-LoRA.",
                    keyframe_strength,
                    num_frames,
                )
            else:
                logger.info(
                    "[HDR IC-LoRA] DFR seams @ strength=%.2f → generated %s | SDR kf %s",
                    keyframe_strength,
                    generated_kf,
                    guides_kf,
                )

        scale_factors = tiling_scale_factors_for_vae(self.video_decoder.checkpoint_path)
        tiling_config = ensure_tiling_config(
            tiling_config,
            scale_factors=scale_factors,
            vae_checkpoint_path=self.video_decoder.checkpoint_path,
            video_shape=VideoPixelShape(
                batch=1,
                frames=gen_frames,
                height=gen_h,
                width=gen_w,
                fps=meta.fps,
            ),
            diffvae_optimization=self.video_decoder.diffvae_optimization,
            device=self.device,
            keyframes=bool(generated_kf),
        )

        acescct_sdr = self._load_acescct_conditioning(video, gen_h, gen_w, num_frames)
        if high_quality_hdr:
            # BCTHW: duplicate each frame, trim to gen_frames (2N-1).
            logger.info("[HDR IC-LoRA] High-quality HDR: %d -> %d internal frames", num_frames, gen_frames)
            acescct_sdr = acescct_sdr.repeat_interleave(2, dim=2)[:, :, :gen_frames]

        logger.info(
            "%s (%dx%d → gen %dx%d, %df%s, colorspace=%s)",
            video_path,
            meta.width,
            meta.height,
            gen_w,
            gen_h,
            num_frames,
            f", hq_internal={gen_frames}" if high_quality_hdr else "",
            video.color_space.value
            if isinstance(video, EXRVideoInput)
            else ("srgb_gamma" if video.gamma_encoded else "srgb"),
        )
        generator = torch.Generator(device=self.device).manual_seed(seed)
        sigmas = list(denoise_sigmas) if denoise_sigmas is not None else list(DEFAULT_DENOISE_SIGMAS)
        sigma_t = torch.tensor(sigmas, dtype=torch.float32, device=self.device)

        conditionings = self.image_conditioner(
            lambda enc: self._create_reference_conditionings(
                enc,
                video_conditioning=acescct_sdr,
                conditioning_strength=conditioning_strength,
                gen_h=gen_h,
                gen_w=gen_w,
                tiling_config=tiling_config,
                guides_kf=guides_kf,
                keyframe_strength=keyframe_strength,
            )
        )
        video_state = self._run_diffusion_stage(
            conditionings=conditionings,
            sigmas=sigma_t,
            video_context=self.video_context,
            frames=gen_frames,
            frame_rate=meta.fps,
            seed=seed,
            generated_kf=generated_kf,
        )

        acescct_hdr = self._decode_video(
            video_state,
            tiling_config,
            generator,
            crop_w=crop_w,
            crop_h=crop_h,
            generated_kf=generated_kf,
            gen_frames=gen_frames,
            high_quality_hdr=high_quality_hdr,
        )
        return acescct_hdr, meta.fps

    def _load_acescct_conditioning(
        self,
        video: VideoInput | EXRVideoInput,
        height: int,
        width: int,
        num_frames: int,
    ) -> torch.Tensor:
        """Load the IC-LoRA reference as ACEScct BCTHW on CPU."""
        match video:
            case EXRVideoInput(dir=path, color_space=color_space):
                frames = load_exr_as_hdr_conditioning(
                    path,
                    height,
                    width,
                    torch.float32,
                    torch.device("cpu"),
                    frame_cap=num_frames,
                    color_space=color_space,
                    resize_mode=ResizeMode.REFLECT_PAD,
                )
            case VideoInput(path=path, gamma_encoded=gamma_encoded):
                frames = load_video_as_hdr_conditioning(
                    path,
                    height,
                    width,
                    num_frames,
                    torch.float32,
                    torch.device("cpu"),
                    gamma_encoded=gamma_encoded,
                    resize_mode=ResizeMode.REFLECT_PAD,
                )
        chunks = list(frames)
        if not chunks:
            raise ValueError(f"No frames loaded from {path}")
        return torch.cat(chunks, dim=2)

    def _create_reference_conditionings(
        self,
        video_encoder: VideoEncoder,
        video_conditioning: torch.Tensor,
        conditioning_strength: float,
        gen_h: int,
        gen_w: int,
        tiling_config: TilingConfig | None,
        *,
        guides_kf: Sequence[int] = (),
        keyframe_strength: float | None = None,
    ) -> list[ConditioningItem]:
        """VAE-encode the IC-LoRA reference, plus optional 1-frame SDR seam guides."""
        video = video_conditioning.to(device=self.device, dtype=self.vae_dtype)
        if tiling_config is not None and gen_h * gen_w > self._tiled_vae_encode_threshold:
            encoded = video_encoder.tiled_encode(video, tiling_config)
        else:
            encoded = video_encoder(video)
        encoded = encoded.to(device=self.device, dtype=self.dtype)
        conditionings: list[ConditioningItem] = [
            VideoConditionByReferenceLatent(
                latent=encoded,
                downscale_factor=1,
                strength=conditioning_strength,
            )
        ]
        if not guides_kf:
            return conditionings
        if keyframe_strength is None:
            raise ValueError("SDR seam guides require keyframe_strength")
        guide_items = _keyframe_conditionings_from_pixel_frames(
            video_encoder,
            video,
            list(guides_kf),
            keyframe_strength,
            gen_h=gen_h,
            gen_w=gen_w,
            tiling_config=tiling_config,
            tiled_threshold=self._tiled_vae_encode_threshold,
            dtype=self.dtype,
        )
        return [*conditionings, *guide_items]

    def _run_diffusion_stage(
        self,
        conditionings: list[ConditioningItem],
        sigmas: torch.Tensor,
        video_context: torch.Tensor,
        frames: int,
        frame_rate: float,
        seed: int,
        generated_kf: Sequence[int] = (),
    ) -> LatentState:
        video_context = video_context.to(device=self.device, dtype=self.dtype)
        source_conditioning = next((c for c in conditionings if isinstance(c, VideoConditionByReferenceLatent)), None)
        if source_conditioning is None:
            raise RuntimeError("Diffusion stage requires a VideoConditionByReferenceLatent IC-LoRA source conditioning")
        phase_conditionings = list(conditionings)
        phase_conditionings.extend(generated_keyframe_conditionings(generated_kf, frames))
        video_state, _ = self.stage(
            denoiser=SimpleDenoiser(video_context, None),
            sigmas=sigmas,
            noiser=GaussianNoiser(generator=torch.Generator(device=self.device).manual_seed(seed)),
            modalities=VideoAudio(
                video=ModalitySpec(
                    latent=source_conditioning.latent,
                    conditioning_fps=_conditioning_fps(frame_rate),
                    context=video_context,
                    conditionings=phase_conditionings,
                    noise_scale=float(sigmas[0].item()),
                ),
            ),
        )
        return video_state

    def _decode_video(
        self,
        video_state: LatentState,
        tiling_config: TilingConfig | None,
        generator: torch.Generator,
        *,
        crop_w: int,
        crop_h: int,
        generated_kf: Sequence[int],
        gen_frames: int,
        high_quality_hdr: bool = False,
    ) -> torch.Tensor:
        """VAE-decode latent chunks to FHWC ``[0, 1]`` (decoder already clamps); optional pad crop.
        When ``high_quality_hdr`` is set, keep every other frame (undoes the ``2N-1``
        generation applied during high-quality mode).
        """
        decode_kf = decode_keyframes_from_slots(video_state.generated_keyframes, generated_kf, gen_frames)
        if decode_kf is not None:
            decode_kf = DecodeKeyframes(
                latents=decode_kf.latents.to(device=video_state.latent.device, dtype=self.vae_dtype),
                pixel_frame_indices=decode_kf.pixel_frame_indices,
                clip_start_frame=decode_kf.clip_start_frame,
            )
        chunks = self.video_decoder(
            video_state.latent,
            tiling_config,
            generator,
            dtype=self.vae_dtype,
            keyframes=decode_kf,
        )
        decoded = torch.cat(list(chunks), dim=0)
        decoded = decoded[:, :crop_h, :crop_w, :]
        if high_quality_hdr:
            decoded = decoded[::2]
        return decoded


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = hdr_ic_lora_arg_parser().parse_args()
    video_marker, audio_marker = markers_from_args(args, device=get_device(), color_space=args.exr_colorspace)

    logger.info("Loading HDRICLoraPipeline...")
    pipeline = HDRICLoraPipeline(
        model_paths=args.model_paths,
        hdr_lora=args.hdr_lora,
        text_embeddings_path=args.text_embeddings,
        offload_mode=args.offload_mode,
        quantization=None if args.no_quantization else _DEFAULT_QUANTIZATION,
        diffvae_optimization=args.diffvae_optimization,
    )

    output_path = Path(args.output_path)
    acescct_hdr, fps = pipeline(
        video=args.input,
        seed=args.seed,
        tiling_config=AUTO_TILING,
        high_quality_hdr=args.high_quality,
        keyframe_strength=None if args.no_keyframes else args.keyframe_strength,
    )

    encode_video(
        video=acescct_hdr,
        fps=round(fps),
        audio=None,
        output_path=str(output_path),
        video_chunks_number=1,
        color_space=args.exr_colorspace,
        video_marker=video_marker,
        audio_marker=audio_marker,
    )
    logger.info("Done → %s (+ EXR)", args.output_path)


if __name__ == "__main__":
    main()
