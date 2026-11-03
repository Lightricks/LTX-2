from __future__ import annotations

import logging
from pathlib import Path

import torch

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.loader.registry import ModelRegistry
from ltx_core.loader.simple_builder import SimpleBuilder
from ltx_core.model.audio_vae import (
    AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    VOCODER_COMFY_KEYS_FILTER,
    AudioDecoderConfigurator,
    VocoderConfigurator,
)
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.types import Audio
from ltx_pipelines.utils.gpu_model import gpu_model

logger = logging.getLogger(__name__)

__all__ = ["AudioDecoder"]


class AudioDecoder:
    """Owns audio decoder + vocoder lifecycle."""

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: ModelRegistry | None = None,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._dtype = dtype
        self._device = device
        unit_registry = registry or ModelRegistry()
        self._decoder_builder = SimpleBuilder(
            AudioDecoderConfigurator,
            (Path(checkpoint_path),),
            (AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,),
            registry=unit_registry,
        )
        self._vocoder_builder = SimpleBuilder(
            VocoderConfigurator,
            (Path(checkpoint_path),),
            (VOCODER_COMFY_KEYS_FILTER,),
            registry=unit_registry,
        )
        self._alloc_trim_strategy = alloc_trim_strategy

    def __call__(self, latent: torch.Tensor) -> Audio:
        """Decode audio *latent* through VAE decoder + vocoder, then free both."""
        logger.info("Building audio decoder + vocoder from %s", self._checkpoint_path)
        # The vocoder always runs in fp32 (bf16 accumulation degrades spectral
        # metrics). On CUDA/CPU it is stored in bf16 and autocast upcasts per-op to
        # save memory; MPS has no fp32 autocast, so store it in fp32 directly and
        # avoid the per-call cast. Negligible footprint for this small model.
        vocoder_dtype = torch.float32 if self._device.type == "mps" else self._dtype
        with (
            gpu_model(
                self._decoder_builder.build(device=self._device, dtype=self._dtype).eval(),
                alloc_trim_strategy=self._alloc_trim_strategy,
            ) as decoder,
            gpu_model(
                self._vocoder_builder.build(device=self._device, dtype=vocoder_dtype).eval(),
                alloc_trim_strategy=self._alloc_trim_strategy,
            ) as vocoder,
        ):
            return vae_decode_audio(latent, decoder, vocoder)
