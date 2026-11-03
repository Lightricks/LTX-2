"""Video/audio latent encoding: VAE encode orchestration and the encode step wrappers.
Reads media via :class:`~ltx_trainer.preprocessing.media.MediaDataset`, encodes it with the
video/audio VAEs, and writes latents under the precomputed tree. The ``*Step`` classes are the
composable, primitive-parameterized wrappers used by strategy preprocessing pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from rich.console import Console
from transformers.utils.logging import disable_progress_bar

from ltx_core.model.audio_vae import AudioProcessor
from ltx_core.model.video_vae import DimensionSizeConfig, TileSizeConfig
from ltx_core.types import VIDEO_SCALE_FACTORS, Audio, SpatioTemporalScaleFactors
from ltx_trainer import logger
from ltx_trainer.media_utils import load_audio_from_file
from ltx_trainer.model_loader import load_audio_vae_encoder, load_video_vae_encoder, read_video_scale_factors
from ltx_trainer.preprocessing.common import (
    atomic_save,
    build_sharded_dataloader,
    load_source_and_naming_paths,
    output_relative,
    preprocessing_progress,
)
from ltx_trainer.preprocessing.media import MediaDataset
from ltx_trainer.preprocessing.types import PreprocessContext

disable_progress_bar()

DEFAULT_TILE_SIZE = 512  # Spatial tile size in pixels (must be ≥64 and divisible by 32)
DEFAULT_TILE_OVERLAP = 128  # Spatial tile overlap in pixels (must be divisible by 32)


def compute_latents(  # noqa: PLR0912, PLR0913, PLR0915
    dataset_file: str | Path,
    video_column: str,
    resolution_buckets: list[tuple[int, int, int]],
    output_dir: str,
    model_path: str,
    audio_model_path: str | None = None,
    main_media_column: str | None = None,
    reshape_mode: str = "center",
    batch_size: int = 1,
    device: str = "cuda",
    vae_tiling: bool = False,
    with_audio: bool = False,
    audio_output_dir: str | None = None,
    num_dataloader_workers: int = 4,
    overwrite: bool = False,
    temporal_subsample_factor: int = 1,
    save_video_latents: bool = True,
) -> None:
    """
    Process videos and save latent representations.
    Under ``accelerate launch``, each process handles an interleaved shard of
    the dataset (rank/world read from ``accelerate.PartialState``). Already-
    computed ``.pt`` outputs are skipped unless ``overwrite=True``; writes are
    atomic so an interrupted run is safe to resume.
    Args:
        dataset_file: Path to metadata file (CSV/JSON/JSONL) containing video paths
        video_column: Column name for video paths in the metadata file
        resolution_buckets: Internal (frames, height, width) tuples
        output_dir: Directory to save video latents
        model_path: Path to the video VAE checkpoint.
        audio_model_path: Optional separate audio VAE checkpoint for split packs.
        reshape_mode: How to crop videos ("center", "random")
        main_media_column: Column name for main media paths (if different from video_column)
        batch_size: Batch size for processing
        device: Device to use for computation
        vae_tiling: Whether to enable VAE tiling
        with_audio: Whether to extract and encode audio from videos
        audio_output_dir: Directory to save audio latents (required if with_audio=True)
        num_dataloader_workers: Number of DataLoader worker processes (0 for in-process loading)
        overwrite: Re-process every item even if its output exists. Use when rerunning with
            changed parameters (different model, resolution, etc.) so stale outputs are replaced.
        temporal_subsample_factor: Factor for VAE-aligned temporal subsampling of reference videos
    """
    # Video VAE compression factors, derived from the checkpoint config (default 32x32x8).
    scale_factors = read_video_scale_factors(model_path)
    _validate_latent_encoding(
        resolution_buckets=resolution_buckets,
        temporal_subsample_factor=temporal_subsample_factor,
        with_audio=with_audio,
        audio_output_dir=audio_output_dir,
        save_video_latents=save_video_latents,
        scale_factors=scale_factors,
    )

    console = Console()
    torch_device = torch.device(device)
    dataset = _build_media_dataset(
        dataset_file=dataset_file,
        main_media_column=main_media_column or video_column,
        video_column=video_column,
        resolution_buckets=resolution_buckets,
        reshape_mode=reshape_mode,
        with_audio=with_audio,
        temporal_subsample_factor=temporal_subsample_factor,
    )
    logger.info(f"Loaded {len(dataset)} valid media files")

    output_path = Path(output_dir) if save_video_latents else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    audio_output_path: Path | None = None
    if with_audio:
        audio_output_path = Path(audio_output_dir)
        audio_output_path.mkdir(parents=True, exist_ok=True)

    # Audio processing requires batch_size=1; must be applied before the dataloader is built.
    if with_audio and batch_size > 1:
        logger.warning("Audio processing requires batch_size=1. Overriding batch_size to 1.")
        batch_size = 1

    data_root = Path(dataset_file).parent

    def _is_done(idx: int) -> bool:
        rel = output_relative(dataset.main_media_paths[idx], data_root).with_suffix(".pt")
        if output_path is not None and not (output_path / rel).is_file():
            return False
        return audio_output_path is None or (audio_output_path / rel).is_file()

    dataloader = build_sharded_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_dataloader_workers,
        is_done=_is_done,
        overwrite=overwrite,
    )
    if dataloader is None:
        return

    vae = None
    if save_video_latents:
        with console.status(f"[bold]Loading video VAE encoder from [cyan]{model_path}[/]...", spinner="dots"):
            vae = load_video_vae_encoder(model_path, device=torch_device, dtype=torch.bfloat16)

    audio_vae_encoder = None
    audio_processor = None
    if with_audio:
        resolved_audio_model_path = audio_model_path or model_path
        with console.status(
            f"[bold]Loading audio VAE encoder from [cyan]{resolved_audio_model_path}[/]...", spinner="dots"
        ):
            audio_vae_encoder, audio_processor = _load_audio_encoder(resolved_audio_model_path, torch_device)

    # Track audio statistics
    audio_success_count = 0
    audio_skip_count = 0

    # Process batches
    with preprocessing_progress("Processing videos", total=len(dataloader)) as advance:
        for batch in dataloader:
            # Get video tensor - shape is [B, F, C, H, W] from DataLoader
            video = batch["video"]

            video_latent_data = None
            if vae is not None:
                with torch.inference_mode():
                    video_latent_data = _encode_video(vae=vae, video=video, use_tiling=vae_tiling)

            # Save latents for each item in batch
            for i in range(len(batch["relative_path"])):
                output_rel_path = Path(batch["main_media_relative_path"][i]).with_suffix(".pt")
                if output_path is not None and video_latent_data is not None:
                    output_file = output_path / output_rel_path
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    effective_fps = batch["video_metadata"]["fps"][i].item() / temporal_subsample_factor
                    atomic_save(
                        {
                            "latents": video_latent_data["latents"][i].cpu().contiguous(),
                            "num_frames": video_latent_data["num_frames"],
                            "height": video_latent_data["height"],
                            "width": video_latent_data["width"],
                            "fps": effective_fps,
                        },
                        output_file,
                    )

                # Process audio if enabled (audio is already extracted by the dataset)
                if with_audio:
                    audio_batch = batch.get("audio")
                    if audio_batch is not None:
                        # Extract the i-th item from batched audio data
                        # DataLoader collates [channels, samples] -> [batch, channels, samples]
                        audio_data = Audio(
                            waveform=audio_batch["waveform"][i],
                            sampling_rate=audio_batch["sample_rate"][i].item(),
                        )

                        # Encode audio
                        with torch.inference_mode():
                            audio_latents = _encode_audio(audio_vae_encoder, audio_processor, audio_data)

                        # Save audio latents
                        audio_output_file = audio_output_path / output_rel_path
                        audio_output_file.parent.mkdir(parents=True, exist_ok=True)

                        audio_save_data = {
                            "latents": audio_latents["latents"].cpu().contiguous(),
                            "num_time_steps": audio_latents["num_time_steps"],
                            "frequency_bins": audio_latents["frequency_bins"],
                            "duration": audio_latents["duration"],
                        }

                        atomic_save(audio_save_data, audio_output_file)
                        audio_success_count += 1
                    else:
                        # Video has no audio track
                        audio_skip_count += 1

            advance()

    if output_path is not None:
        logger.info(f"Processed {len(dataloader.dataset)} videos -> {output_path}")  # type: ignore[arg-type]
    if with_audio:
        logger.info(
            f"Audio processing: {audio_success_count} videos with audio, "
            f"{audio_skip_count} videos without audio (skipped)"
        )


def compute_audio_latents(  # noqa: PLR0915
    dataset_file: str | Path,
    audio_column: str,
    output_dir: str,
    model_path: str,
    main_media_column: str | None = None,
    max_duration: float | None = None,
    duration_buckets: list[float] | None = None,
    device: str = "cuda",
    overwrite: bool = False,
) -> None:
    """Encode audio files into latent representations.
    Supports standalone audio files (.wav, .mp3, etc.) and audio tracks
    extracted from video files (.mp4, etc.).
    Args:
        dataset_file: Path to metadata file (CSV/JSON/JSONL).
        audio_column: Column name containing audio file paths.
        output_dir: Directory to save audio latents.
        model_path: Path to LTX-2 checkpoint (.safetensors).
        main_media_column: Column for output file naming (defaults to audio_column).
            Ensures alignment with other latent directories.
        max_duration: Maximum audio duration in seconds. Audio is trimmed if longer.
            Mutually exclusive with duration_buckets.
        duration_buckets: List of allowed durations in seconds (e.g. [2.0, 4.0, 8.0]).
            Each audio file is matched to the largest bucket that fits its duration,
            then trimmed to exactly that length. Files shorter than the smallest
            bucket are skipped. Ensures uniform lengths for batched training.
        device: Device to use for computation.
    """
    console = Console()
    torch_device = torch.device(device)

    dataset_path = Path(dataset_file)
    data_root = dataset_path.parent
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    audio_paths, naming_paths = load_source_and_naming_paths(dataset_path, audio_column, main_media_column)
    items = [
        {"audio_path": str(audio_path), "naming_path": str(naming_path)}
        for audio_path, naming_path in zip(audio_paths, naming_paths, strict=True)
    ]

    def _is_done(idx: int) -> bool:
        naming_path = Path(items[idx]["naming_path"])
        return (output_path / output_relative(naming_path, data_root).with_suffix(".pt")).is_file()

    dataloader = build_sharded_dataloader(
        items,
        batch_size=1,
        num_workers=0,
        is_done=_is_done,
        overwrite=overwrite,
    )
    if dataloader is None:
        return

    with console.status(f"[bold]Loading audio VAE encoder from [cyan]{model_path}[/]...", spinner="dots"):
        audio_vae_encoder, audio_processor = _load_audio_encoder(model_path, torch_device)

    sorted_buckets = sorted(duration_buckets, reverse=True) if duration_buckets else None
    success_count = 0
    skip_count = 0

    with preprocessing_progress("Encoding audio", total=len(dataloader)) as advance:
        for batch in dataloader:
            audio_path = Path(batch["audio_path"][0])
            naming_path = Path(batch["naming_path"][0])
            rel_path = output_relative(naming_path, data_root)
            output_file = output_path / rel_path.with_suffix(".pt")
            output_file.parent.mkdir(parents=True, exist_ok=True)

            if not overwrite and output_file.is_file():
                success_count += 1
                advance()
                continue

            # Load audio (no trimming yet — need full duration for bucket matching)
            audio = load_audio_from_file(audio_path)
            if audio is None:
                skip_count += 1
                advance()
                continue

            file_duration = audio.waveform.shape[-1] / audio.sampling_rate

            # Determine target duration: bucket matching, max_duration cap, or full file
            target_duration = file_duration
            if sorted_buckets:
                bucket = next((b for b in sorted_buckets if b <= file_duration), None)
                if bucket is None:
                    logger.warning(
                        f"Skipping {audio_path.name} ({file_duration:.1f}s) — shorter than "
                        f"smallest bucket ({sorted_buckets[-1]:.1f}s)"
                    )
                    skip_count += 1
                    advance()
                    continue
                target_duration = bucket
            elif max_duration is not None:
                target_duration = min(file_duration, max_duration)

            # Trim to target duration
            target_samples = int(target_duration * audio.sampling_rate)
            trimmed_waveform = audio.waveform[:, :target_samples]
            audio = Audio(waveform=trimmed_waveform, sampling_rate=audio.sampling_rate)

            with torch.inference_mode():
                audio_latents = _encode_audio(audio_vae_encoder, audio_processor, audio)

            atomic_save(
                {
                    "latents": audio_latents["latents"].cpu().contiguous(),
                    "num_time_steps": audio_latents["num_time_steps"],
                    "frequency_bins": audio_latents["frequency_bins"],
                    "duration": audio_latents["duration"],
                },
                output_file,
            )
            success_count += 1
            advance()

    logger.info(f"Audio encoding complete: {success_count} encoded, {skip_count} skipped. Saved to {output_path}")


def compute_scaled_resolution_buckets(
    resolution_buckets: list[tuple[int, int, int]],
    scale_factor: int,
    scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS,
) -> list[tuple[int, int, int]]:
    """Spatially downscale internal ``(frames, height, width)`` buckets and validate the result."""
    if scale_factor == 1:
        return resolution_buckets

    scaled_buckets = []
    for frames, height, width in resolution_buckets:
        if height % scale_factor != 0:
            raise ValueError(
                f"Height {height} is not evenly divisible by scale factor {scale_factor}. "
                f"Choose a scale factor that divides {height} evenly."
            )
        if width % scale_factor != 0:
            raise ValueError(
                f"Width {width} is not evenly divisible by scale factor {scale_factor}. "
                f"Choose a scale factor that divides {width} evenly."
            )

        scaled_height = height // scale_factor
        scaled_width = width // scale_factor

        if scaled_height % scale_factors.height != 0:
            raise ValueError(
                f"Scaled height {scaled_height} (from {height} / {scale_factor}) "
                f"is not divisible by {scale_factors.height}. "
                f"Choose a different scale factor or adjust your resolution buckets."
            )
        if scaled_width % scale_factors.width != 0:
            raise ValueError(
                f"Scaled width {scaled_width} (from {width} / {scale_factor}) "
                f"is not divisible by {scale_factors.width}. "
                f"Choose a different scale factor or adjust your resolution buckets."
            )

        scaled_buckets.append((frames, scaled_height, scaled_width))

    return scaled_buckets


def _validate_latent_encoding(
    *,
    resolution_buckets: list[tuple[int, int, int]],
    temporal_subsample_factor: int,
    with_audio: bool,
    audio_output_dir: str | None,
    save_video_latents: bool,
    scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS,
) -> None:
    """Validate options that govern video/audio latent encoding."""
    if not save_video_latents and not with_audio:
        raise ValueError("At least one of save_video_latents or with_audio must be True")
    if with_audio and audio_output_dir is None:
        raise ValueError("audio_output_dir must be provided when with_audio=True")
    for frames, height, width in resolution_buckets:
        if height % scale_factors.height != 0 or width % scale_factors.width != 0:
            raise ValueError(
                f"Resolution {width}x{height} must be divisible by the VAE spatial factors "
                f"({scale_factors.width}x{scale_factors.height})."
            )
        if (frames - 1) % scale_factors.time != 0:
            raise ValueError(
                f"Frame count {frames} must satisfy (frames - 1) % {scale_factors.time} == 0 "
                "for the VAE temporal factor."
            )
    if temporal_subsample_factor == 1:
        return
    for frames, _h, _w in resolution_buckets:
        pixel_frames_minus_one = frames - 1
        if pixel_frames_minus_one % temporal_subsample_factor != 0:
            raise ValueError(
                f"Frame count {frames} is not compatible with temporal_subsample_factor={temporal_subsample_factor}. "
                f"(frames - 1) must be divisible by the factor."
            )
        subsampled = 1 + pixel_frames_minus_one // temporal_subsample_factor
        if (subsampled - 1) % scale_factors.time != 0:
            raise ValueError(
                f"After temporal subsampling {frames} → {subsampled} frames, "
                f"result does not satisfy (frames - 1) % {scale_factors.time} == 0."
            )


def _build_media_dataset(
    *,
    dataset_file: str | Path,
    main_media_column: str,
    video_column: str,
    resolution_buckets: list[tuple[int, int, int]],
    reshape_mode: str,
    with_audio: bool,
    temporal_subsample_factor: int,
) -> MediaDataset:
    """Build the media dataset used by video and video-derived audio encoding."""
    return MediaDataset(
        dataset_file=dataset_file,
        main_media_column=main_media_column,
        video_column=video_column,
        resolution_buckets=resolution_buckets,
        reshape_mode=reshape_mode,
        with_audio=with_audio,
        temporal_subsample_factor=temporal_subsample_factor,
    )


def _encode_video(vae: torch.nn.Module, video: torch.Tensor, use_tiling: bool = False) -> dict[str, torch.Tensor | int]:
    """Encode a video into a non-patchified latent, with optional spatial VAE tiling.
    Args:
        vae: Video VAE encoder model.
        video: Input tensor ``[C, F, H, W]`` or ``[B, C, F, H, W]`` in the VAE's expected range.
        use_tiling: Encode in overlapping spatial tiles (via ltx-core) to cap peak memory.
    Returns:
        ``{"latents": Tensor[B, C, F', H', W'], "num_frames", "height", "width"}``.
    """
    device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype

    if video.ndim == 4:
        video = video.unsqueeze(0)  # [C, F, H, W] -> [B, C, F, H, W]
    video = video.to(device=device, dtype=vae_dtype)

    if use_tiling:
        tiling = TileSizeConfig(
            height=DimensionSizeConfig(tile_size=DEFAULT_TILE_SIZE, overlap=DEFAULT_TILE_OVERLAP),
            width=DimensionSizeConfig(tile_size=DEFAULT_TILE_SIZE, overlap=DEFAULT_TILE_OVERLAP),
        )
        latents = vae.tiled_encode(video, tiling)
    else:
        latents = vae(video)

    _, _, num_frames, height, width = latents.shape
    return {"latents": latents, "num_frames": num_frames, "height": height, "width": width}


def _load_audio_encoder(model_path: str, device: torch.device) -> tuple[torch.nn.Module, AudioProcessor]:
    """Load the audio VAE encoder and its matching ``AudioProcessor``.
    The audio VAE runs in float32 for reconstruction quality (TODO: re-test with bfloat16).
    """
    encoder = load_audio_vae_encoder(checkpoint_path=model_path, device=device, dtype=torch.float32)
    processor = AudioProcessor(
        target_sample_rate=encoder.sample_rate,
        mel_bins=encoder.mel_bins,
        mel_hop_length=encoder.mel_hop_length,
        n_fft=encoder.n_fft,
    ).to(device)
    return encoder, processor


def _encode_audio(
    audio_vae_encoder: torch.nn.Module,
    audio_processor: torch.nn.Module,
    audio: Audio,
) -> dict[str, torch.Tensor | int | float]:
    """Encode audio waveform into latent representation.
    Args:
        audio_vae_encoder: Audio VAE encoder model from ltx-core
        audio_processor: AudioProcessor for waveform-to-spectrogram conversion
        audio: Audio container with waveform tensor and sampling rate.
    Returns:
        Dict containing audio latents and shape information:
        {
            "latents": Tensor[C, T, F],  # Non-patchified format
            "num_time_steps": int,
            "frequency_bins": int,
            "duration": float,
        }
    """
    device = next(audio_vae_encoder.parameters()).device
    dtype = next(audio_vae_encoder.parameters()).dtype

    waveform = audio.waveform.to(device=device, dtype=dtype)

    # Add batch dimension if needed: [channels, samples] -> [batch, channels, samples]
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)

    # Convert to stereo if needed (audio VAE expects 2 channels)
    # Channel order for surround: 5.1=[L,R,C,LFE,Ls,Rs], 7.1=[L,R,C,LFE,Ls,Rs,Lb,Rb]
    num_channels = waveform.shape[1]
    if num_channels == 1:
        # Mono to stereo: duplicate the channel
        waveform = waveform.repeat(1, 2, 1)
    elif num_channels == 6:
        # 5.1 downmix with normalized weights (sum to 1.0)
        # Original: L = L + 0.707*C + 0.707*Ls, weights sum = 2.414
        w_main = 1.0 / 2.414  # ~0.414
        w_other = 0.707 / 2.414  # ~0.293
        left = w_main * waveform[:, 0, :] + w_other * waveform[:, 2, :] + w_other * waveform[:, 4, :]
        right = w_main * waveform[:, 1, :] + w_other * waveform[:, 2, :] + w_other * waveform[:, 5, :]
        waveform = torch.stack([left, right], dim=1)
    elif num_channels == 8:
        # 7.1 downmix with normalized weights (sum to 1.0)
        # Original: L = L + 0.707*C + 0.707*Ls + 0.707*Lb, weights sum = 3.121
        w_main = 1.0 / 3.121  # ~0.320
        w_other = 0.707 / 3.121  # ~0.227
        center = waveform[:, 2, :]
        left = w_main * waveform[:, 0, :] + w_other * (center + waveform[:, 4, :] + waveform[:, 6, :])
        right = w_main * waveform[:, 1, :] + w_other * (center + waveform[:, 5, :] + waveform[:, 7, :])
        waveform = torch.stack([left, right], dim=1)
    elif num_channels > 2:
        # Unknown layout: average all channels to mono, then duplicate to stereo
        logger.warning(f"Unknown audio channel layout ({num_channels} channels), using mean downmix")
        mono = waveform.mean(dim=1, keepdim=True)
        waveform = mono.repeat(1, 2, 1)

    # Calculate duration
    duration = waveform.shape[-1] / audio.sampling_rate

    # Convert waveform to mel spectrogram using AudioProcessor
    mel_spectrogram = audio_processor.waveform_to_mel(Audio(waveform=waveform, sampling_rate=audio.sampling_rate))
    mel_spectrogram = mel_spectrogram.to(dtype=dtype)

    # Encode mel spectrogram to latents
    latents = audio_vae_encoder(mel_spectrogram)

    # latents shape: [batch, channels, time, freq] = [1, 8, T, 16]
    _, _channels, time_steps, freq_bins = latents.shape

    return {
        "latents": latents.squeeze(0),  # [C, T, F] - remove batch dim
        "num_time_steps": time_steps,
        "frequency_bins": freq_bins,
        "duration": duration,
    }


@dataclass(kw_only=True)
class VideoEncodeStep:
    """Encode target videos (and optionally auto-extract audio) into latent directories."""

    video_column: str
    output_subdir: str | None
    resolution_buckets: list[tuple[int, int, int]]
    model_path: str
    audio_model_path: str | None = None
    reshape_mode: str = "center"
    batch_size: int = 1
    vae_tiling: bool = False
    num_workers: int = 4
    audio_output_subdir: str | None = None
    save_video_latents: bool = True

    def run(self, ctx: PreprocessContext) -> None:
        audio_latents_dir = str(ctx.output_root / self.audio_output_subdir) if self.audio_output_subdir else None
        if self.audio_output_subdir:
            logger.info("Audio will be auto-extracted from video files (set data.skip_audio=true to disable)")

        compute_latents(
            dataset_file=ctx.dataset_file,
            video_column=self.video_column,
            resolution_buckets=self.resolution_buckets,
            output_dir=str(ctx.output_root / self.output_subdir) if self.output_subdir else "",
            model_path=self.model_path,
            audio_model_path=self.audio_model_path,
            reshape_mode=self.reshape_mode,
            batch_size=self.batch_size,
            device=ctx.device,
            vae_tiling=self.vae_tiling,
            with_audio=self.audio_output_subdir is not None,
            audio_output_dir=audio_latents_dir,
            num_dataloader_workers=self.num_workers,
            overwrite=ctx.overwrite,
            save_video_latents=self.save_video_latents,
        )


@dataclass(kw_only=True)
class ReferenceVideoEncodeStep:
    """Encode reference videos for IC-LoRA style conditioning."""

    reference_column: str
    main_media_column: str
    output_subdir: str
    resolution_buckets: list[tuple[int, int, int]]
    reference_spatial_scale_factor: int
    reference_temporal_scale_factor: int
    model_path: str
    reshape_mode: str = "center"
    batch_size: int = 1
    vae_tiling: bool = False
    num_workers: int = 4

    def run(self, ctx: PreprocessContext) -> None:
        spatial = self.reference_spatial_scale_factor
        temporal = self.reference_temporal_scale_factor
        if spatial > 1 and len(self.resolution_buckets) > 1:
            raise ValueError(
                f"reference_spatial_scale_factor > 1 requires a single resolution bucket "
                f"(got {len(self.resolution_buckets)})."
            )
        if temporal > 1 and len(self.resolution_buckets) > 1:
            raise ValueError(
                f"reference_temporal_scale_factor > 1 requires a single resolution bucket "
                f"(got {len(self.resolution_buckets)})."
            )

        scale_factors = read_video_scale_factors(self.model_path)
        reference_buckets = compute_scaled_resolution_buckets(self.resolution_buckets, spatial, scale_factors)
        if spatial > 1:
            logger.info(f"Processing reference videos at 1/{spatial} resolution...")
        if temporal > 1:
            logger.info(f"Temporally subsampling reference videos by {temporal}x (VAE-aligned pattern)...")

        compute_latents(
            dataset_file=ctx.dataset_file,
            main_media_column=self.main_media_column,
            video_column=self.reference_column,
            resolution_buckets=reference_buckets,
            output_dir=str(ctx.output_root / self.output_subdir),
            model_path=self.model_path,
            reshape_mode=self.reshape_mode,
            batch_size=self.batch_size,
            device=ctx.device,
            vae_tiling=self.vae_tiling,
            num_dataloader_workers=self.num_workers,
            overwrite=ctx.overwrite,
            temporal_subsample_factor=temporal,
        )


@dataclass(kw_only=True)
class AudioEncodeStep:
    """Encode standalone audio columns into audio latents."""

    audio_column: str
    output_subdir: str
    model_path: str
    main_media_column: str | None = None
    max_audio_duration: float | None = None
    audio_duration_buckets: list[float] | None = None

    def run(self, ctx: PreprocessContext) -> None:
        compute_audio_latents(
            dataset_file=ctx.dataset_file,
            audio_column=self.audio_column,
            output_dir=str(ctx.output_root / self.output_subdir),
            model_path=self.model_path,
            main_media_column=self.main_media_column,
            max_duration=self.max_audio_duration,
            duration_buckets=self.audio_duration_buckets,
            device=ctx.device,
            overwrite=ctx.overwrite,
        )


class ReferenceAudioEncodeStep(AudioEncodeStep):
    """Encode reference audio columns into reference audio latents."""
