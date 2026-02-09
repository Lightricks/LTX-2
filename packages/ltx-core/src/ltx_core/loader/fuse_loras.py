import torch

from ltx_core.loader.primitives import LoraStateDictWithStrength, StateDict
from ltx_core.quantization.fp8_cast import calculate_weight_float8
from ltx_core.quantization.fp8_scaled_mm import quantize_weight_to_fp8_per_tensor


def apply_loras(
    model_sd: StateDict,
    lora_sd_and_strengths: list[LoraStateDictWithStrength],
    dtype: torch.dtype | None = None,
    destination_sd: StateDict | None = None,
) -> StateDict:
    sd = {}
    if destination_sd is not None:
        sd = destination_sd.sd
    size = 0
    device = torch.device("meta")
    inner_dtypes = set()
    for key, weight in model_sd.sd.items():
        if weight is None:
            continue
        # Skip scale keys - they are handled together with their weight keys
        if key.endswith(".weight_scale"):
            continue
        device = weight.device
        target_dtype = dtype if dtype is not None else weight.dtype
        deltas_dtype = target_dtype if target_dtype not in [torch.float8_e4m3fn, torch.float8_e5m2] else torch.bfloat16

        scale_key = key.replace(".weight", ".weight_scale") if key.endswith(".weight") else None
        is_scaled_fp8 = scale_key is not None and scale_key in model_sd.sd

        deltas = _prepare_deltas(lora_sd_and_strengths, key, deltas_dtype, device)
        fused = _fuse_deltas(deltas, weight, key, sd, target_dtype, device, is_scaled_fp8, scale_key, model_sd)

        sd.update(fused)
        for tensor in fused.values():
            inner_dtypes.add(tensor.dtype)
            size += tensor.nbytes

    if destination_sd is not None:
        return destination_sd
    return StateDict(sd, device, size, inner_dtypes)


def _prepare_deltas(
    lora_sd_and_strengths: list[LoraStateDictWithStrength], key: str, dtype: torch.dtype, device: torch.device
) -> torch.Tensor | None:
    deltas = []
    prefix = key[: -len(".weight")]
    key_a = f"{prefix}.lora_A.weight"
    key_b = f"{prefix}.lora_B.weight"
    for lsd, coef in lora_sd_and_strengths:
        if key_a not in lsd.sd or key_b not in lsd.sd:
            continue
        product = torch.matmul(lsd.sd[key_b] * coef, lsd.sd[key_a])
        deltas.append(product.to(dtype=dtype, device=device))
    if len(deltas) == 0:
        return None
    elif len(deltas) == 1:
        return deltas[0]
    return torch.sum(torch.stack(deltas, dim=0), dim=0)


def _fuse_deltas(
    deltas: torch.Tensor | None,
    weight: torch.Tensor,
    key: str,
    sd: dict[str, torch.Tensor],
    target_dtype: torch.dtype,
    device: torch.device,
    is_scaled_fp8: bool,
    scale_key: str | None,
    model_sd: StateDict,
) -> dict[str, torch.Tensor]:
    if deltas is None:
        if key in sd:
            return {}
        fused = _copy_weight_without_lora(weight, key, target_dtype, device, is_scaled_fp8, scale_key, model_sd)
    elif weight.dtype == torch.float8_e4m3fn:
        if is_scaled_fp8:
            fused = _fuse_delta_with_scaled_fp8(deltas, weight, key, scale_key, model_sd)
        else:
            fused = _fuse_delta_with_cast_fp8(deltas, weight, key, target_dtype, device)
    elif weight.dtype == torch.bfloat16:
        fused = _fuse_delta_with_bfloat16(deltas, weight, key, target_dtype)
    else:
        raise ValueError(f"Unsupported dtype: {weight.dtype}")

    return fused


def _copy_weight_without_lora(
    weight: torch.Tensor,
    key: str,
    target_dtype: torch.dtype,
    device: torch.device,
    is_scaled_fp8: bool,
    scale_key: str | None,
    model_sd: StateDict,
) -> dict[str, torch.Tensor]:
    """Copy original weight (and scale if applicable) when no LoRA affects this key."""
    result = {key: weight.clone().to(dtype=target_dtype, device=device)}
    if is_scaled_fp8:
        result[scale_key] = model_sd.sd[scale_key].clone()
    return result


def _fuse_delta_with_scaled_fp8(
    deltas: torch.Tensor,
    weight: torch.Tensor,
    key: str,
    scale_key: str,
    model_sd: StateDict,
) -> dict[str, torch.Tensor]:
    """Dequantize scaled FP8 weight, add LoRA delta, and re-quantize."""
    weight_scale = model_sd.sd[scale_key]

    original_weight = weight.t().to(torch.float32) * weight_scale

    new_weight = original_weight + deltas.to(torch.float32)

    new_fp8_weight, new_weight_scale = quantize_weight_to_fp8_per_tensor(new_weight)
    return {key: new_fp8_weight, scale_key: new_weight_scale}


def _fuse_delta_with_cast_fp8(
    deltas: torch.Tensor,
    weight: torch.Tensor,
    key: str,
    target_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Fuse LoRA delta with cast-only FP8 weight (no scale factor)."""
    if str(device).startswith("cuda"):
        deltas = calculate_weight_float8(deltas, weight)
    else:
        deltas.add_(weight.to(dtype=deltas.dtype, device=device))
    return {key: deltas.to(dtype=target_dtype)}


def _fuse_delta_with_bfloat16(
    deltas: torch.Tensor,
    weight: torch.Tensor,
    key: str,
    target_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Fuse LoRA delta with bfloat16 weight."""
    deltas.add_(weight)
    return {key: deltas.to(dtype=target_dtype)}
