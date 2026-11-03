#!/usr/bin/env python3

"""
Decode precomputed video latents back into videos using the VAE.
Loads latent files saved during preprocessing and decodes them back into
video clips using the same VAE model.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
import torchvision.utils
from einops import rearrange
from rich.console import Console
from transformers.utils.logging import disable_progress_bar

from ltx_core.model.video_vae import DimensionSizeConfig, TileSizeConfig
from ltx_trainer import logger
from ltx_trainer.media_utils import save_video
from ltx_trainer.model_loader import (
    load_audio_vae_decoder,
    load_video_vae_decoder,
    load_vocoder,
    resolve_audio_vae_path,
    resolve_video_vae_path,
)
from ltx_trainer.preprocessing.common import preprocessing_progress
from ltx_trainer.preprocessing.types import PreprocessContext

DEFAULT_TILE_SIZE_PIXELS = 512  # Spatial tile size in pixels (must be ≥64 and divisible by 32)
DEFAULT_TILE_OVERLAP_PIXELS = 128  # Spatial tile overlap in pixels (must be divisible by 32)
DEFAULT_TILE_SIZE_FRAMES = 128  # Temporal tile size in frames (must be ≥16 and divisible by 8)
DEFAULT_TILE_OVERLAP_FRAMES = 24  # Temporal tile overlap in frames (must be divisible by 8)

disable_progress_bar()
console = Console()


class LatentsDecoder:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        vae_tiling: bool = False,
        with_video: bool = True,
        with_audio: bool = False,
        video_vae_path: str | None = None,
        audio_vae_path: str | None = None,
    ):
        """Initialize the decoder with model configuration.
        Args:
            model_path: Path to LTX-2 checkpoint (.safetensors)
            device: Device to use for computation
            vae_tiling: Whether to enable VAE tiling for larger video resolutions
            with_audio: Whether to load audio VAE for audio decoding
        """
        self.device = torch.device(device)
        self.model_path = model_path
        self.vae = None
        self.audio_vae = None
        self.vocoder = None
        self.vae_tiling = vae_tiling

        resolved_video_vae_path = resolve_video_vae_path(model_path, video_vae_path) if with_video else None
        resolved_audio_vae_path = resolve_audio_vae_path(model_path, audio_vae_path) if with_audio else str(model_path)
        self._load_model(resolved_video_vae_path, resolved_audio_vae_path, with_audio)

    @torch.inference_mode()
    def decode(self, latents_dir: Path, output_dir: Path, seed: int | None = None) -> None:
        """Decode all video latent files under ``latents_dir`` recursively into videos."""
        if self._decode_directory(
            latents_dir, output_dir, lambda file, out: self._process_file(file, out, seed), "Decoding latents"
        ):
            logger.info(f"Decoding complete! Videos saved to {output_dir}")

    @torch.inference_mode()
    def decode_audio(self, latents_dir: Path, output_dir: Path) -> None:
        """Decode all audio latent files under ``latents_dir`` recursively into WAV files."""
        if self.audio_vae is None or self.vocoder is None:
            logger.warning("Audio VAE or vocoder not loaded. Skipping audio decoding.")
            return
        if self._decode_directory(latents_dir, output_dir, self._process_audio_file, "Decoding audio latents"):
            logger.info(f"Audio decoding complete! Audio files saved to {output_dir}")

    def _decode_directory(
        self,
        latents_dir: Path,
        output_dir: Path,
        process: Callable[[Path, Path], None],
        description: str,
    ) -> int:
        """Run ``process(latent_file, output_subdir)`` over every ``.pt`` under ``latents_dir``.
        Mirrors the input directory structure under ``output_dir``, shows a progress bar, and
        logs per-file errors and raises after all files have been attempted. Returns the
        number of files found when every decode succeeds.
        """
        latent_files = list(latents_dir.rglob("*.pt"))
        if not latent_files:
            logger.warning(f"No .pt files found in {latents_dir}")
            return 0

        logger.info(f"Found {len(latent_files):,} latent files to decode")
        failures: list[tuple[Path, Exception]] = []
        with preprocessing_progress(description, total=len(latent_files)) as advance:
            for latent_file in latent_files:
                output_subdir = output_dir / latent_file.relative_to(latents_dir).parent
                output_subdir.mkdir(parents=True, exist_ok=True)
                try:
                    process(latent_file, output_subdir)
                except Exception as e:
                    logger.error(f"Error processing {latent_file}: {e}")
                    failures.append((latent_file, e))
                advance()
        if failures:
            failed_files = ", ".join(str(file) for file, _error in failures)
            raise RuntimeError(f"Failed to decode {len(failures)} latent file(s): {failed_files}") from failures[0][1]
        return len(latent_files)

    def _load_model(self, video_vae_path: str | None, audio_vae_path: str, with_audio: bool = False) -> None:
        """Initialize and load the VAE model(s)."""
        if video_vae_path is not None:
            with console.status(f"[bold]Loading video VAE decoder from {video_vae_path}...", spinner="dots"):
                self.vae = load_video_vae_decoder(video_vae_path, device=self.device, dtype=torch.bfloat16)

        if with_audio:
            with console.status(f"[bold]Loading audio VAE decoder from {audio_vae_path}...", spinner="dots"):
                self.audio_vae = load_audio_vae_decoder(audio_vae_path, device=self.device, dtype=torch.bfloat16)

            with console.status(f"[bold]Loading vocoder from {audio_vae_path}...", spinner="dots"):
                self.vocoder = load_vocoder(audio_vae_path, device=self.device)

    def _process_file(self, latent_file: Path, output_dir: Path, seed: int | None) -> None:
        """Process a single latent file."""
        # Load the latent data
        data = torch.load(latent_file, map_location=self.device, weights_only=False)

        # Get latents - handle both old patchified [seq_len, C] and new [C, F, H, W] formats
        latents = data["latents"]
        num_frames = data["num_frames"]
        height = data["height"]
        width = data["width"]

        # Check if latents need reshaping (old patchified format)
        if latents.dim() == 2:
            # Old format: [seq_len, C] -> reshape to [C, F, H, W]
            latents = rearrange(latents, "(f h w) c -> c f h w", f=num_frames, h=height, w=width)

        # Add batch dimension: [C, F, H, W] -> [1, C, F, H, W]
        latents = latents.unsqueeze(0).to(device=self.device, dtype=torch.bfloat16)

        # Create generator only if seed is provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

        # Decode the video
        video = self._decode_video(latents, generator)

        # Determine output format and save
        is_image = video.shape[0] == 1
        if is_image:
            # Save as PNG for single frame
            output_path = output_dir / f"{latent_file.stem}.png"
            torchvision.utils.save_image(
                video[0],  # [C, H, W] in [0, 1]
                str(output_path),
            )
        else:
            # Save as MP4 for video using PyAV-based save_video
            output_path = output_dir / f"{latent_file.stem}.mp4"
            fps = data.get("fps", 24)  # Use stored FPS or default to 24
            save_video(
                video_tensor=video,  # [F, C, H, W] in [0, 1]
                output_path=output_path,
                fps=fps,
            )

    def _decode_video(self, latents: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """Decode latents to video frames."""
        if self.vae is None:
            raise RuntimeError("Video VAE decoder is not loaded.")
        if self.vae_tiling:
            # Use tiled decoding for reduced VRAM
            tiling_config = TileSizeConfig(
                height=DimensionSizeConfig(
                    tile_size=DEFAULT_TILE_SIZE_PIXELS,
                    overlap=DEFAULT_TILE_OVERLAP_PIXELS,
                ),
                width=DimensionSizeConfig(
                    tile_size=DEFAULT_TILE_SIZE_PIXELS,
                    overlap=DEFAULT_TILE_OVERLAP_PIXELS,
                ),
                frames=DimensionSizeConfig(
                    tile_size=DEFAULT_TILE_SIZE_FRAMES,
                    overlap=DEFAULT_TILE_OVERLAP_FRAMES,
                ),
            )
            chunks = list(
                self.vae.tiled_decode(
                    latents,
                    tiling_config=tiling_config,
                    generator=generator,
                )
            )
            # Concatenate along temporal dimension
            video = torch.cat(chunks, dim=2)  # [B, C, F, H, W]
        else:
            # Standard full decoding
            video = self.vae(latents, generator=generator)  # [B, C, F, H, W]

        # Convert to [F, C, H, W] format and normalize to [0, 1]
        video = rearrange(video, "1 c f h w -> f c h w")
        video = (video + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        video = video.clamp(0, 1)

        return video

    def _process_audio_file(self, latent_file: Path, output_dir: Path) -> None:
        """Process a single audio latent file."""
        # Load the latent data
        data = torch.load(latent_file, map_location=self.device, weights_only=False)

        latents = data["latents"].to(device=self.device, dtype=torch.float32)
        num_time_steps = data["num_time_steps"]
        freq_bins = data["frequency_bins"]

        # Handle both old patchified [seq_len, C] and new [C, T, F] formats
        if latents.dim() == 2:
            # Old format: [seq_len, channels] where seq_len = time * freq
            # Reshape to [C, T, F]
            latents = rearrange(latents, "(t f) c -> c t f", t=num_time_steps, f=freq_bins)

        # Add batch dimension: [C, T, F] -> [1, C, T, F]
        latents = latents.unsqueeze(0)

        # Set correct dtype for audio VAE
        latents = latents.to(dtype=torch.bfloat16)

        # Decode audio using audio VAE decoder (produces mel spectrogram)
        mel_spectrogram = self.audio_vae(latents)

        # Convert mel spectrogram to waveform using vocoder
        waveform = self.vocoder(mel_spectrogram)

        # Save as WAV
        output_path = output_dir / f"{latent_file.stem}.wav"
        sample_rate = self.vocoder.output_sampling_rate
        torchaudio.save(str(output_path), waveform[0].cpu(), sample_rate)


@dataclass(kw_only=True)
class DecodeLatentsStep:
    """Decode encoded latents for visual verification.
    Each ``*_subdir`` is decoded only when set (``None`` = that latent kind was not produced),
    so there are no separate presence flags to keep in sync.
    """

    model_path: str
    vae_tiling: bool
    with_audio: bool
    video_vae_path: str | None = None
    audio_vae_path: str | None = None
    video_latents_subdir: str | None = None
    reference_video_latents_subdir: str | None = None
    audio_latents_subdir: str | None = None
    reference_audio_latents_subdir: str | None = None

    def run(self, ctx: PreprocessContext) -> None:
        logger.info("Decoding latents for verification...")
        video_dirs = (self.video_latents_subdir, self.reference_video_latents_subdir)
        with_video = any(subdir and (ctx.output_root / subdir).exists() for subdir in video_dirs)
        audio_dirs = (self.audio_latents_subdir, self.reference_audio_latents_subdir)
        with_audio = self.with_audio and any(subdir and (ctx.output_root / subdir).exists() for subdir in audio_dirs)
        decoder = LatentsDecoder(
            model_path=self.model_path,
            video_vae_path=self.video_vae_path,
            audio_vae_path=self.audio_vae_path,
            device=ctx.device,
            vae_tiling=self.vae_tiling,
            with_video=with_video,
            with_audio=with_audio,
        )
        if self.video_latents_subdir:
            decoder.decode(ctx.output_root / self.video_latents_subdir, ctx.output_root / "decoded_videos")

        if self.reference_video_latents_subdir:
            ref_dir = ctx.output_root / self.reference_video_latents_subdir
            if ref_dir.exists():
                decoder.decode(ref_dir, ctx.output_root / "decoded_reference_videos")

        if with_audio:
            if self.audio_latents_subdir:
                audio_dir = ctx.output_root / self.audio_latents_subdir
                if audio_dir.exists():
                    decoder.decode_audio(audio_dir, ctx.output_root / "decoded_audio")
            if self.reference_audio_latents_subdir:
                ref_audio_dir = ctx.output_root / self.reference_audio_latents_subdir
                if ref_audio_dir.exists():
                    decoder.decode_audio(ref_audio_dir, ctx.output_root / "decoded_reference_audio")
