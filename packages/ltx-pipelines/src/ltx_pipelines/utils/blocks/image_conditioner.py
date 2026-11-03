from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import TypeVar

import torch

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.loader import SDOps
from ltx_core.loader.registry import ModelRegistry
from ltx_core.loader.simple_builder import SimpleBuilder
from ltx_core.model.video_vae import (
    CHANNELS_LAST_3D_WEIGHTS,
    VAE_ENCODER_COMFY_KEYS_FILTER,
    VideoEncoder,
    VideoEncoderConfigurator,
)
from ltx_pipelines.utils.constants import detect_params
from ltx_pipelines.utils.gpu_model import gpu_model
from ltx_pipelines.utils.media_io.color_config import EXRColorSpace
from ltx_pipelines.utils.media_io.decode import BackgroundStill, check_conditioning_still, load_conditioning_still
from ltx_pipelines.utils.types import DecodedStill, ImageConditioningInput, LoadedImageConditioning

# Shared channels-last encoder sd_ops so the registry caches one copy across encoder builds.
VAE_ENCODER_SD_OPS = SDOps(
    name=f"sd_ops_chain_{VAE_ENCODER_COMFY_KEYS_FILTER.name}+{CHANNELS_LAST_3D_WEIGHTS.name}",
    mapping=(*VAE_ENCODER_COMFY_KEYS_FILTER.mapping, *CHANNELS_LAST_3D_WEIGHTS.mapping),
    excluded_prefixes=VAE_ENCODER_COMFY_KEYS_FILTER.excluded_prefixes,
)

T = TypeVar("T")

__all__ = ["ImageConditioner"]


class ImageConditioner:
    """Owns video encoder lifecycle, the CRF image conditionings are re-compressed at, and their stills.
    Builds the encoder, passes it to the user-supplied callable, then frees it.
    Also the single place that knows what H.264 CRF an image conditioning should be
    re-compressed at: that value is a property of the model generation, and this block
    already holds the checkpoint it belongs to. Pipelines call ``load_conditionings`` so callers may
    omit the CRF and still condition the way the model was trained, and so each still is
    decoded once per run however many stages, tiles or windows encode it.
    """

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: ModelRegistry | None = None,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._default_image_crf: int | None = None
        self._dtype = dtype
        self._device = device
        unit_registry = registry or ModelRegistry()
        self._encoder_builder = SimpleBuilder(
            VideoEncoderConfigurator,
            (Path(checkpoint_path),),
            (VAE_ENCODER_SD_OPS,),
            registry=unit_registry,
        )
        self._alloc_trim_strategy = alloc_trim_strategy
        self._decode_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="image-decode")

    @property
    def default_image_crf(self) -> int:
        """The H.264 CRF image conditionings should be re-compressed at for this checkpoint.
        Read from the checkpoint's ``model_version`` metadata (see ``detect_params``) rather
        than hardcoded, because the value the model was trained with changed between
        generations. Resolved on first use and cached, so a run without image conditionings
        never reads the header.
        """
        if self._default_image_crf is None:
            self._default_image_crf = detect_params(self._checkpoint_path).default_image_crf
        return self._default_image_crf

    def resolve_crf(self, images: Sequence[ImageConditioningInput]) -> list[ImageConditioningInput]:
        """Fill in ``default_image_crf`` for every conditioning that left its CRF unset (``crf=None``).
        Call near the top of a pipeline's ``__call__``: omitting the CRF then means "use what
        matches this model", while an explicitly passed CRF (including ``0`` for no
        re-compression) is always honoured.
        """
        return [image if image.crf is not None else replace(image, crf=self.default_image_crf) for image in images]

    def load_conditionings(
        self, images: Sequence[ImageConditioningInput], color_space: EXRColorSpace | None = None
    ) -> list[LoadedImageConditioning]:
        """Resolve every CRF (see ``resolve_crf``) and start decoding each distinct still once for the run.
        Call near the top of a pipeline's ``__call__``. Every input comes back as a
        :class:`LoadedImageConditioning` whose still is shared by every input with the same path and
        CRF and decodes on a worker thread while the run goes on (prompt encoding, model builds);
        every stage, tile, window and chunk that encodes it (or a rebased copy) waits for that still
        and cuts its resolution from it instead of decoding the file again. ``.exr`` stills need
        ``color_space`` (from ``--hdr``). A missing file, or an EXR without ``color_space``, raises
        ``ValueError`` here, before any model is built; an undecodable one raises the same
        ``ValueError`` at its first encode.
        """
        stills: dict[tuple[str, int], DecodedStill] = {}
        loaded: list[LoadedImageConditioning] = []
        for image in self.resolve_crf(images):
            key = (image.path, image.crf)
            if key not in stills:
                check_conditioning_still(image.path, color_space)
                stills[key] = BackgroundStill(
                    self._decode_pool.submit(load_conditioning_still, image.path, image.crf, color_space)
                )
            loaded.append(LoadedImageConditioning(still=stills[key], strength=image.strength, binding=image.binding))
        return loaded

    def _build_encoder(self) -> VideoEncoder:
        return self._encoder_builder.build(device=self._device, dtype=self._dtype).eval()

    def __call__(self, fn: Callable[[VideoEncoder], T]) -> T:
        """Build video encoder → call *fn(encoder)* → free encoder."""
        with gpu_model(self._build_encoder(), alloc_trim_strategy=self._alloc_trim_strategy) as encoder:
            return fn(encoder)
