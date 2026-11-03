from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import torch

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.loader.model_builder import ModelBuilder
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.registry import ModelRegistry
from ltx_core.loader.simple_builder import SimpleBuilder
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae import (
    CHANNELS_LAST_3D_WEIGHTS,
    MEMORY_EFFICIENT_DECODE,
    TilingConfig,
    VideoDecoderConfigurator,
    is_diffusion_video_vae,
    video_decoder_sd_ops_for_checkpoint,
)
from ltx_core.model.video_vae.keyframes import DecodeKeyframes
from ltx_core.model.video_vae.transformer import (
    DiffVAEMode,
    NAttentionKind,
    build_diffvae_mode_op,
    na_dsl_available,
)
from ltx_core.multigpu.vae.distributed_decoder import DistributedVideoDecoder, OwnedSingleFrames
from ltx_pipelines.utils.gpu_model import gpu_model

logger = logging.getLogger(__name__)

__all__ = ["VideoDecoder"]


class VideoDecoder:
    """Owns video decoder lifecycle.
    Returns an iterator that cleans up the decoder after all chunks are consumed.
    """

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: ModelRegistry | None = None,
        memory_efficient: bool = True,
        decoder_builder: ModelBuilder | None = None,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
        diffvae_optimization: DiffVAEMode = DiffVAEMode.CHUNKED_EAGER,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._dtype = dtype
        self._device = device
        self._diffvae_optimization = diffvae_optimization
        diffusion_vae = is_diffusion_video_vae(self._checkpoint_path)
        self._diffusion_vae = diffusion_vae
        cfg = diffvae_optimization.resolve() if diffusion_vae else None
        use_dsl = cfg is not None and cfg.attention is NAttentionKind.BLACKWELL_DSL
        if decoder_builder is not None:
            self._decoder_builder = decoder_builder
        else:
            if diffusion_vae and use_dsl and not na_dsl_available():
                raise RuntimeError(
                    "diffvae_optimization=blackwell_dsl needs a datacenter Blackwell GPU and "
                    "nvidia-cutlass-dsl (uv sync --group kernels)"
                )
            sd_ops = video_decoder_sd_ops_for_checkpoint(
                self._checkpoint_path, diffusion_vae=diffusion_vae, na_dsl_kernel=diffusion_vae and use_dsl
            )
            module_ops: tuple[ModuleOps, ...] = ()
            # channels-last + memory-efficient decode apply to ConvVideoDecoder only
            if memory_efficient and not diffusion_vae:
                sd_ops = (*sd_ops, CHANNELS_LAST_3D_WEIGHTS)
                module_ops = (MEMORY_EFFICIENT_DECODE,)
            unit_registry = registry or ModelRegistry()
            self._decoder_builder = SimpleBuilder(
                VideoDecoderConfigurator,
                (Path(self._checkpoint_path),),
                sd_ops,
                registry=unit_registry,
                module_ops=module_ops,
            )
        self._alloc_trim_strategy = alloc_trim_strategy

    @property
    def checkpoint_path(self) -> str:
        return self._checkpoint_path

    @property
    def diffvae_optimization(self) -> DiffVAEMode:
        return self._diffvae_optimization

    def with_builder(self, builder: ModelBuilder) -> "VideoDecoder":
        """Return a new ``VideoDecoder`` that builds from ``builder``.
        Functional: never mutates ``self``; shares checkpoint, dtype, device, and
        trim strategy. Mirrors :meth:`DiffusionStage.with_builder`.
        """
        new = copy.copy(self)
        new._decoder_builder = builder
        return new

    def _prepared_builder(self) -> ModelBuilder:
        """Return the builder with DiffVAE mode ops applied at build time.
        DiffVAE ``module_ops`` live on the decoder (not on the caller-held builder),
        matching :meth:`DiffusionStage._prepared_builder`. That keeps
        ``with_builder`` / Dist ``with_tiling`` swaps on the raw builder, while
        every build still installs the configured DiffVAE policy.
        """
        builder = self._decoder_builder
        if not self._diffusion_vae:
            return builder
        op = build_diffvae_mode_op(
            self._diffvae_optimization,
            CompilationConfig() if self._diffvae_optimization.resolve().compile_blocks else None,
        )
        return builder.with_module_ops((*builder.module_ops, op))

    def __call__(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
        generator: torch.Generator | None = None,
        *,
        dtype: torch.dtype | None = None,
        keyframes: DecodeKeyframes | None = None,
        device_fn: Callable[[int], str | torch.device] | None = None,
    ) -> Iterator[torch.Tensor]:
        """Decode *latent* to pixel-space video chunks. Decoder freed after exhaustion.
        ``dtype`` overrides the constructor dtype for this call only (e.g. float32 for
        HDR raw-in/raw-out). Omitted → same dtype as on construction (SDR default).
        ``keyframes`` anchors the decode on already-encoded planes. Every decoder in this repo
        takes it, so this never branches on which VAE was built: a diffusion VAE uses the planes,
        a conv VAE logs that it is dropping them, and a distributed decoder splits them across
        its ranks. The keyword is only passed when there is something to pass, so a decoder
        written against the older signature still serves plain decodes.
        """
        build_dtype = self._dtype if dtype is None else dtype
        latent = latent.to(dtype=build_dtype)
        logger.info("Building video decoder from %s", self._checkpoint_path)
        decoder = self._prepared_builder().build(device=self._device, dtype=build_dtype).eval()
        extra = {} if keyframes is None else {"keyframes": keyframes}
        if device_fn is not None:
            extra["device_fn"] = device_fn
        chunks = decoder.decode_video(latent, tiling_config, generator, **extra)
        return _cleanup_iter(
            chunks,
            decoder,
            alloc_trim_strategy=self._alloc_trim_strategy,
        )

    def decode_single_frames(
        self,
        latents: Sequence[torch.Tensor],
        generator: torch.Generator | Sequence[torch.Generator | None] | None = None,
        *,
        dtype: torch.dtype | None = None,
    ) -> Iterator[torch.Tensor]:
        """Decode each latent as its own one-frame clip on one VAE build, then free it.
        A causal VAE cannot decode stacked independent planes without bleeding neighbours.
        A Dist build splits the list across its ranks and gathers the pixels back, so every
        rank returns the whole list; it does not go through the volume-splitting
        ``decode_video`` path, whose workers yield nothing.
        """
        if not latents:
            return iter(())
        build_dtype = self._dtype if dtype is None else dtype
        moved = [latent.to(dtype=build_dtype) for latent in latents]
        logger.info("Building video decoder from %s", self._checkpoint_path)
        decoder = self._prepared_builder().build(device=self._device, dtype=build_dtype).eval()
        chunks = decoder.decode_single_frames(moved, generator)
        return _cleanup_iter(
            chunks,
            decoder,
            alloc_trim_strategy=self._alloc_trim_strategy,
        )

    def decode_owned_single_frames(
        self,
        latents: Sequence[torch.Tensor],
        generator: torch.Generator | Sequence[torch.Generator | None] | None = None,
        *,
        dtype: torch.dtype | None = None,
    ) -> OwnedSingleFrames:
        """``decode_single_frames`` without the pixel gather: only the planes this build decodes.
        A Dist build yields its round-robin share and a collective ``gather`` every rank calls
        once with what it derived from its planes; an SGPU build owns every plane and its
        ``gather`` returns its argument. The decoder is freed once the planes are exhausted.
        """
        if not latents:
            return OwnedSingleFrames([], iter(()), dict)
        build_dtype = self._dtype if dtype is None else dtype
        moved = [latent.to(dtype=build_dtype) for latent in latents]
        logger.info("Building video decoder from %s", self._checkpoint_path)
        decoder = self._prepared_builder().build(device=self._device, dtype=build_dtype).eval()
        if isinstance(decoder, DistributedVideoDecoder):
            owned = decoder.owned_single_frames(moved, generator)
        else:
            owned = OwnedSingleFrames(list(range(len(moved))), decoder.decode_single_frames(moved, generator), dict)
        return owned._replace(
            planes=_cleanup_iter(owned.planes, decoder, alloc_trim_strategy=self._alloc_trim_strategy)
        )


def _cleanup_iter(
    it: Iterator[torch.Tensor],
    model: torch.nn.Module,
    alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
) -> Iterator[torch.Tensor]:
    """Wrap an iterator to release *model* memory once exhausted or abandoned."""
    with gpu_model(model, alloc_trim_strategy=alloc_trim_strategy):
        yield from it
