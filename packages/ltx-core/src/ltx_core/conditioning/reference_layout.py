"""RoPE layouts and source tags for reference conditioning.

The helpers in this module are shared by training and inference so reference
tokens occupy the same coordinate frame in both paths.
"""

from __future__ import annotations

import torch

VALID_REFERENCE_LAYOUTS = ("overlap", "st_drc", "sidecar", "virtual_sidecar", "strata")
STRATA_F_LIM = 128
STRATA_P_STM = 4
STRATA_P_LTM = 2


def strata_temporal_start(
    slot: str,
    *,
    f_lim: int = STRATA_F_LIM,
    p_stm: int = STRATA_P_STM,
    p_ltm: int = STRATA_P_LTM,
) -> float:
    """Return the absolute temporal start for an LTM or STM Strata-RoPE band."""
    normalized_slot = str(slot).lower()
    if normalized_slot == "stm":
        return float(f_lim - p_stm)
    if normalized_slot == "ltm":
        return float(f_lim - p_stm - p_ltm)
    raise ValueError(f"Unknown strata slot {slot!r}; expected 'ltm' or 'stm'.")


def apply_reference_layout(
    reference_positions: torch.Tensor,
    target_positions: torch.Tensor,
    *,
    layout: str,
    sidecar_margin_pixels: float = 0.0,
    strata_start: float | None = None,
) -> torch.Tensor:
    """Place reference patch bounds in the selected RoPE coordinate layout.

    Inputs use the standard ``[B, axes, tokens, bounds]`` representation. The
    ``virtual_sidecar`` spelling is accepted as an alias for ``sidecar``.
    """
    normalized_layout = str(layout or "overlap").lower()
    if normalized_layout == "virtual_sidecar":
        normalized_layout = "sidecar"
    if normalized_layout not in VALID_REFERENCE_LAYOUTS:
        raise ValueError(f"Unsupported reference layout {layout!r}. Expected one of {VALID_REFERENCE_LAYOUTS}.")
    if normalized_layout == "overlap":
        return reference_positions
    if reference_positions.ndim != 4 or target_positions.ndim != 4:
        raise ValueError("Reference layouts expect patch bounds shaped [batch, axis, token, bound]")
    if reference_positions.shape[:2] != target_positions.shape[:2]:
        raise ValueError(
            "Reference/target coordinate axes do not match: "
            f"{tuple(reference_positions.shape)} vs {tuple(target_positions.shape)}"
        )

    shifted = reference_positions.clone()
    target_min = target_positions.amin(dim=(2, 3), keepdim=True)
    target_extent = target_positions.amax(dim=(2, 3), keepdim=True)
    reference_origin = shifted.amin(dim=(2, 3), keepdim=True)

    if normalized_layout == "st_drc":
        shifted += target_extent - reference_origin
        return shifted

    if normalized_layout == "strata":
        if strata_start is None:
            raise ValueError("layout='strata' requires strata_start")
        start = torch.as_tensor(float(strata_start), device=shifted.device, dtype=shifted.dtype)
        shifted[:, 0:1, ...] += start - reference_origin[:, 0:1, ...]
        return shifted

    if shifted.shape[1] < 3:
        raise ValueError(f"sidecar layout expects at least 3 coordinate axes, got {shifted.shape[1]}")
    reference_extent = shifted.amax(dim=(2, 3), keepdim=True)
    target_center_h = (target_min[:, 1:2, ...] + target_extent[:, 1:2, ...]) * 0.5
    reference_center_h = (reference_origin[:, 1:2, ...] + reference_extent[:, 1:2, ...]) * 0.5
    shifted[:, 1:2, ...] += target_center_h - reference_center_h
    shifted[:, 2:3, ...] += (
        target_extent[:, 2:3, ...]
        + torch.as_tensor(float(sidecar_margin_pixels), device=shifted.device, dtype=shifted.dtype)
        - reference_origin[:, 2:3, ...]
    )
    shifted[:, 0, :, 0] = target_min[:, 0, 0, 0].unsqueeze(-1)
    shifted[:, 0, :, 1] = target_extent[:, 0, 0, 0].unsqueeze(-1)
    return shifted


def extend_segment_ids(
    segment_ids: torch.Tensor | None,
    denoise_mask: torch.Tensor,
    num_new_tokens: int,
    *,
    value: float = 0.0,
) -> torch.Tensor | None:
    """Append source ids while keeping the per-token field aligned.

    A zero-only field stays ``None`` to preserve the exact upstream fast path.
    """
    if segment_ids is None and value == 0.0:
        return None
    if segment_ids is None:
        segment_ids = torch.zeros(
            denoise_mask.shape[0],
            denoise_mask.shape[1],
            device=denoise_mask.device,
            dtype=torch.float32,
        )
    pad = torch.full(
        (segment_ids.shape[0], num_new_tokens),
        float(value),
        device=segment_ids.device,
        dtype=segment_ids.dtype,
    )
    return torch.cat([segment_ids, pad], dim=1)
