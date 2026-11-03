from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import torch

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.loader.registry import ModelRegistry
from ltx_core.loader.simple_builder import SimpleBuilder
from ltx_core.model.audio_vae import AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER, AudioEncoderConfigurator
from ltx_pipelines.utils.gpu_model import gpu_model

T = TypeVar("T")

__all__ = ["AudioConditioner"]


class AudioConditioner:
    """Owns audio encoder lifecycle.
    Builds the encoder, passes it to the user-supplied callable, then frees it.
    Mirrors :class:`ImageConditioner` for the audio modality.
    """

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: ModelRegistry | None = None,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
    ) -> None:
        self._dtype = dtype
        self._device = device
        self._alloc_trim_strategy = alloc_trim_strategy
        unit_registry = registry or ModelRegistry()
        self._encoder_builder = SimpleBuilder(
            AudioEncoderConfigurator,
            (Path(checkpoint_path),),
            (AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,),
            registry=unit_registry,
        )

    def __call__(self, fn: Callable[[torch.nn.Module], T]) -> T:
        """Build audio encoder → call *fn(encoder)* → free encoder."""
        with gpu_model(
            self._encoder_builder.build(device=self._device, dtype=self._dtype).eval(),
            alloc_trim_strategy=self._alloc_trim_strategy,
        ) as encoder:
            return fn(encoder)
