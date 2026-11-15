"""Map between display/working [0,1] and VAE [-1,1]."""

from __future__ import annotations

import torch


def normalize_images(images: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return (images / 127.5 - 1.0).to(device=device, dtype=dtype)


def to_vae_range(x: torch.Tensor) -> torch.Tensor:
    """Map [0, 1] working-space codes to [-1, 1] (VAE input convention).
    Raises:
        ValueError: If a host ``x`` is non-finite or outside ``[0, 1]`` (callers must compress
            or already hold valid working-space codes — this does not clamp). A CUDA ``x`` is
            checked asynchronously: a violation fails the stream with a device-side assert.
    """
    if x.is_cuda:
        # Checked on the device: reading the result back would wait for every queued kernel.
        # NaN makes both comparisons false, so it needs its own check.
        torch._assert_async(
            (torch.isfinite(x) & (x >= 0) & (x <= 1)).all(),
            "to_vae_range expects finite values in [0, 1]",
        )
    elif not bool((torch.isfinite(x) & (x >= 0) & (x <= 1)).all()):
        raise ValueError(
            f"to_vae_range expects finite values in [0, 1]; "
            f"got min={float(x.min())}, max={float(x.max())}"
        )
    return x * 2.0 - 1.0


def from_vae_range(z: torch.Tensor) -> torch.Tensor:
    """Map [-1, 1] (VAE output convention) to [0, 1]."""
    return torch.clamp((z + 1.0) / 2.0, 0.0, 1.0)
