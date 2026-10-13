from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, NamedTuple, Protocol, TypeVar

import torch

from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.conditioning import ConditioningItem
from ltx_core.model.transformer import X0Model
from ltx_core.model.video_vae import TilingConfig
from ltx_core.types import Audio, LatentState
from ltx_pipelines.utils.constants import VIDEO_LATENT_CHANNELS, VIDEO_SCALE_FACTORS

_T = TypeVar("_T")


class PipelineOutput(NamedTuple):
    """Return type of every video pipeline ``__call__``.
    A ``NamedTuple`` rather than a bare tuple so adding fields fails loudly at any
    call site that unpacks a fixed arity. Prefer attribute access
    (``result.video``, ``result.audio``, ...) over positional unpack.
    Keyframe-aware decoding is decided inside the pipeline via ``decode_with_keyframes``;
    the returned ``video`` iterator already reflects that choice.
    """

    video: Iterator[torch.Tensor]
    audio: Audio
    num_frames: int
    tiling_config: TilingConfig | None


class ImageConditioningInput(NamedTuple):
    """An image to condition on, with the H.264 CRF it should be re-compressed at.
    Leaving ``crf`` unset means "use whatever matches the model": the pipeline fills it in
    from the checkpoint it runs (``ImageConditioner.resolve_crf``), since the value the model
    was trained with is a property of the model generation. Pass ``crf`` explicitly to
    override it, including ``0`` to skip re-compression entirely.
    """

    path: str
    frame_idx: int
    strength: float
    crf: int | None = None


class PipelineComponents:
    """
    Container class for pipeline components used throughout the LTX pipelines.
    Attributes:
        dtype (torch.dtype): Default torch dtype for tensors in the pipeline.
        device (torch.device): Target device to place tensors and modules on.
        video_scale_factors (SpatioTemporalScaleFactors): Scale factors (T, H, W) for VAE latent space.
        video_latent_channels (int): Number of channels in the video latent representation.
        video_patchifier (VideoLatentPatchifier): Patchifier instance for video latents.
        audio_patchifier (AudioPatchifier): Patchifier instance for audio latents.
    """

    def __init__(
        self,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.dtype = dtype
        self.device = device

        self.video_scale_factors = VIDEO_SCALE_FACTORS
        self.video_latent_channels = VIDEO_LATENT_CHANNELS

        self.video_patchifier = VideoLatentPatchifier(patch_size=1)
        self.audio_patchifier = AudioPatchifier(patch_size=1)


@dataclass(frozen=True)
class DenoisedLatentResult:
    """Output of one denoiser call for a single modality.
    ``denoised`` is the final blended prediction for this modality.
    The remaining fields carry the per-pass raw outputs from ``_guided_denoise``
    (all ``None`` for ``SimpleDenoiser``).  Denoisers return a
    :class:`VideoAudio` of these results; omit the absent side.
    """

    denoised: torch.Tensor
    uncond: torch.Tensor | None = None
    cond: torch.Tensor | None = None
    ptb: torch.Tensor | None = None
    mod: torch.Tensor | None = None

    @classmethod
    def result_or_none(
        cls,
        denoised: torch.Tensor | None,
        uncond: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
        ptb: torch.Tensor | None = None,
        mod: torch.Tensor | None = None,
    ) -> DenoisedLatentResult | None:
        if denoised is None:
            return None
        return cls(denoised=denoised, uncond=uncond, cond=cond, ptb=ptb, mod=mod)


@dataclass(frozen=True)
class VideoAudio(Generic[_T]):
    """Video and audio values for one multimodal step.
    Either side may be omitted (defaults to ``None``) when that modality is
    absent. At least one side is required. Unpacks as ``(video, audio)``.
    """

    video: _T | None = None
    audio: _T | None = None

    def __post_init__(self) -> None:
        if self.video is None and self.audio is None:
            raise ValueError("VideoAudio requires at least one of video or audio")

    def __iter__(self) -> Iterator[_T | None]:
        return iter((self.video, self.audio))


class Denoiser(Protocol):
    """Protocol for a denoiser that receives the transformer at call time.
    The transformer is not stored — it is passed as the first argument so the
    caller (a denoising loop or a pipeline block) controls its lifecycle.
    Args:
        transformer: The diffusion model.
        video_state: Current video latent state, or ``None`` if absent.
        audio_state: Current audio latent state, or ``None`` if absent.
        sigmas: 1-D tensor of sigma values for each diffusion step.
        step_index: Index of the current denoising step.
    Returns:
        A :class:`VideoAudio` of :class:`DenoisedLatentResult`; omit the
        absent side.
    """

    def __call__(
        self,
        transformer: X0Model,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> VideoAudio[DenoisedLatentResult]: ...


@dataclass(frozen=True)
class ModalitySpec:
    """Specification for one modality passed to a diffusion stage.
    Carries everything needed to build the initial noised latent state
    and run the denoising loop for a single modality (video or audio).
    ``latent`` is the latent to denoise, and it is the stage's only source of shape --
    latent tools are derived from it. A first stage seeds it with zeros from
    :func:`~ltx_pipelines.utils.helpers.create_initial_video_latent` (or its audio
    counterpart); later stages pass the previous stage's output, optionally upscaled.
    ``frozen=True`` zeros the denoise mask and marks the resulting ``LatentState``
    so ``Modality.sigma`` is forced to 0 (not only per-token timesteps).
    ``conditioning_fps`` is the transformer's video time base when video is present.
    Callers pass it on every spec, including audio, so multimodal steps share one
    time base; the stage reads it from the video spec only today. Playback fps for
    decode/mux and audio latent duration are decided elsewhere
    (``create_initial_audio_latent``, export), not on this field.
    """

    latent: torch.Tensor
    conditioning_fps: float
    context: torch.Tensor | None = None
    conditionings: list[ConditioningItem] = field(default_factory=list)
    noise_scale: float = 1.0
    frozen: bool = False


@dataclass(frozen=True)
class AutoDuration:
    """Request auto-predicted duration, clamped to ``[min_seconds, max_seconds]``.
    The predicted duration is converted to a frame count snapped to the VAE's causal
    temporal grid (``8k + 1``). Defaults: 1s and 20s.
    """

    min_seconds: float = 1.0
    max_seconds: float = 20.0


DEFAULT_AUTO_DURATION = AutoDuration()


class OffloadMode(Enum):
    """Weight offloading strategy.
    Controls where model weights reside during inference:
    - ``NONE``: All weights on GPU (no streaming). Fastest inference,
      requires enough VRAM for the full model (~28 GB for the 19B LTX-2 in
      bf16; LTX-2.5 with its text encoder and VAEs resident does not fit an
      80 GB GPU in bf16, so pair it with quantization, offloading or
      multi-GPU).
    - ``CPU``: Weights pinned in CPU RAM, streamed layer-by-layer to a
      small GPU buffer. First pass reads from disk; subsequent passes
      reuse the CPU cache. Requires ~36 GB RAM + ~5 GB VRAM.
    - ``DISK``: Weights read from disk on demand through a small CPU
      buffer, then streamed to GPU. Every pass re-reads from disk.
      Lowest memory: ~5 GB RAM + ~5 GB VRAM.
    """

    NONE = "none"
    CPU = "cpu"
    DISK = "disk"
