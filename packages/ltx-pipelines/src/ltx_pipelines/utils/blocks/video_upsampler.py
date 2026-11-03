from __future__ import annotations

import logging
from pathlib import Path

import torch

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.loader.registry import ModelRegistry
from ltx_core.loader.simple_builder import SimpleBuilder
from ltx_core.model.upsampler import LatentUpsamplerConfigurator, upsample_video
from ltx_core.model.video_vae import VideoEncoderConfigurator
from ltx_pipelines.utils.blocks.image_conditioner import VAE_ENCODER_SD_OPS
from ltx_pipelines.utils.gpu_model import gpu_model

logger = logging.getLogger(__name__)

__all__ = ["VideoUpsampler"]


class VideoUpsampler:
    """Owns video encoder + spatial upsampler lifecycle."""

    def __init__(
        self,
        checkpoint_path: str,
        upsampler_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: ModelRegistry | None = None,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
    ) -> None:
        self._upsampler_path = upsampler_path
        self._dtype = dtype
        self._device = device
        unit_registry = registry or ModelRegistry()
        self._encoder_builder = SimpleBuilder(
            VideoEncoderConfigurator,
            (Path(checkpoint_path),),
            (VAE_ENCODER_SD_OPS,),
            registry=unit_registry,
        )
        self._upsampler_builder = SimpleBuilder(
            LatentUpsamplerConfigurator,
            (Path(upsampler_path),),
            registry=unit_registry,
        )
        self._alloc_trim_strategy = alloc_trim_strategy

    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        """Upsample *latent* using video encoder + spatial upsampler, then free both."""
        logger.info("Building video encoder + spatial upsampler from %s", self._upsampler_path)
        with (
            gpu_model(
                self._encoder_builder.build(device=self._device, dtype=self._dtype).eval(),
                alloc_trim_strategy=self._alloc_trim_strategy,
            ) as encoder,
            gpu_model(
                self._upsampler_builder.build(device=self._device, dtype=self._dtype).eval(),
                alloc_trim_strategy=self._alloc_trim_strategy,
            ) as upsampler,
        ):
            return upsample_video(latent=latent, video_encoder=encoder, upsampler=upsampler)
