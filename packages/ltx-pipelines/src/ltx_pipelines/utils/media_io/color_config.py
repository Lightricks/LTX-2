"""EXR colour-space policy for pipelines."""

from __future__ import annotations

import enum
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import Tensor

from ltx_core.color.primaries import Primaries
from ltx_core.hdr import srgb_eotf_to_linear, to_acescct_working_space, to_hdr_linear


class EXRColorSpace(enum.Enum):
    """Colour space of an EXR plate."""

    SRGB_LINEAR = "srgb_linear"
    ACESCG = "acescg"
    ACESCCT = "acescct"
    ACES2065_1 = "aces2065-1"

    @property
    def is_log_working(self) -> bool:
        """True when the VAE signal is already ACEScct log (no load-time transfer)."""
        return self is EXRColorSpace.ACESCCT

    @property
    def is_aces_container(self) -> bool:
        """True when frames should be written as a SMPTE ST 2065-4 ACES container."""
        return self is EXRColorSpace.ACES2065_1

    @property
    def source_primaries(self) -> Primaries:
        match self:
            case EXRColorSpace.ACESCG | EXRColorSpace.ACESCCT:
                return Primaries.AP1
            case EXRColorSpace.ACES2065_1:
                return Primaries.AP0
            case EXRColorSpace.SRGB_LINEAR:
                return Primaries.REC709


def to_working_space(video: Tensor, current_colorspace: EXRColorSpace) -> Tensor:
    """Map HDR float RGB of ``current_colorspace`` into ACEScct ``[0, 1]`` (channel-first)."""
    if current_colorspace.is_log_working:
        return video.float().clamp(0.0, 1.0)
    return to_acescct_working_space(video.float(), source_primaries=current_colorspace.source_primaries)


def srgb_to_acescct(video: Tensor, *, gamma_encoded: bool) -> Tensor:
    """Rec.709 float RGB → ACEScct ``[0, 1]``. ``gamma_encoded`` applies the sRGB EOTF first."""
    if not video.is_floating_point():
        raise ValueError(
            f"srgb_to_acescct expects float RGB; got dtype={video.dtype}. Normalize integer pixels before calling."
        )
    video = video.float()
    if gamma_encoded:
        video = srgb_eotf_to_linear(video)
    return to_acescct_working_space(video, source_primaries=Primaries.REC709)


def vae_dtype_for_hdr(hdr: EXRColorSpace | None, default: torch.dtype) -> torch.dtype:
    """VAE decode dtype: float32 for HDR, else ``default`` (pipelines use bf16 today)."""
    return torch.float32 if hdr is not None else default


def _has_exr_input(
    images: Iterable[object] = (),
    video_paths: Iterable[str | Path] = (),
) -> bool:
    from ltx_pipelines.utils.media_io.exr import is_exr_dir  # noqa: PLC0415

    for img in images:
        path = getattr(img, "path", img)
        if str(path).lower().endswith(".exr"):
            return True
    return any(is_exr_dir(p) for p in video_paths)


def resolve_hdr_color_space(
    images: Iterable[object] = (),
    video_paths: Iterable[str | Path] = (),
    hdr: EXRColorSpace | None = None,
) -> EXRColorSpace | None:
    """Return the explicit ``--hdr`` value, or raise if EXR inputs lack it.
    ``None`` means SDR.
    """
    if _has_exr_input(images, video_paths) and hdr is None:
        raise ValueError(
            "EXR input requires --hdr {SRGB_LINEAR,ACESCG,ACESCCT,ACES2065_1} to declare the source colour space."
        )
    return hdr


def decode_hdr_video(
    decoded_video: torch.Tensor,
    out_primaries: Primaries = Primaries.REC709,
) -> torch.Tensor:
    """VAE decode ``[F,H,W,C]`` in ``[0,1]`` ACEScct → scene-linear HDR float32."""
    video = decoded_video.float().permute(3, 0, 1, 2).unsqueeze(0)
    hdr = to_hdr_linear(video, out_primaries=out_primaries)
    return hdr[0].permute(1, 2, 3, 0).contiguous()
