import logging

import torch
import triton

from ltx_core.loader.kernels import fused_add_round_kernel
from ltx_core.loader.primitives import LoraStateDictWithStrength, StateDict

logger = logging.getLogger(__name__)

BLOCK_SIZE = 1024

# FP8 E4M3 (fp8e4nv) Triton kernels require Ada Lovelace (sm_89) or newer.
# Ampere (sm_86, e.g. A40/A100) can store FP8 tensors but cannot run Triton
# FP8 arithmetic. Detect once at import time and fall back to pure PyTorch.
_GPU_SUPPORTS_TRITON_FP8: bool | None = None


def _gpu_supports_triton_fp8() -> bool:
    global _GPU_SUPPORTS_TRITON_FP8
    if _GPU_SUPPORTS_TRITON_FP8 is None:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            _GPU_SUPPORTS_TRITON_FP8 = (major, minor) >= (8, 9)
            if not _GPU_SUPPORTS_TRITON_FP8:
                logger.info(
                    f"GPU compute capability {major}.{minor} < 8.9 — "
                    f"using PyTorch fallback for FP8 LoRA fusion (no Triton fp8e4nv)"
                )
        else:
            _GPU_SUPPORTS_TRITON_FP8 = False
    return _GPU_SUPPORTS_TRITON_FP8


def fused_add_round_launch(target_weight: torch.Tensor, original_weight: torch.Tensor, seed: int) -> torch.Tensor:
    if original_weight.dtype == torch.float8_e4m3fn:
        exponent_bits, mantissa_bits, exponent_bias = 4, 3, 7
    elif original_weight.dtype == torch.float8_e5m2:
        exponent_bits, mantissa_bits, exponent_bias = 5, 2, 15  # noqa: F841
    else:
        raise ValueError("Unsupported dtype")

    if target_weight.dtype != torch.bfloat16:
        raise ValueError("target_weight dtype must be bfloat16")

    # Calculate grid and block sizes
    n_elements = original_weight.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch kernel
    fused_add_round_kernel[grid](
        original_weight,
        target_weight,
        seed,
        n_elements,
        exponent_bias,
        mantissa_bits,
        BLOCK_SIZE,
    )
    return target_weight


def calculate_weight_float8_(target_weights: torch.Tensor, original_weights: torch.Tensor) -> torch.Tensor:
    if _gpu_supports_triton_fp8():
        result = fused_add_round_launch(target_weights, original_weights, seed=0).to(target_weights.dtype)
        target_weights.copy_(result, non_blocking=True)
    else:
        # Pure PyTorch fallback: add in bfloat16, then deterministic cast to FP8.
        # Slightly less accurate than stochastic rounding but equivalent for inference.
        result = (target_weights + original_weights.to(dtype=torch.bfloat16)).to(original_weights.dtype)
        target_weights.copy_(result.to(target_weights.dtype), non_blocking=True)
    return target_weights


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


def apply_loras(
    model_sd: StateDict,
    lora_sd_and_strengths: list[LoraStateDictWithStrength],
    dtype: torch.dtype,
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
        device = weight.device
        target_dtype = dtype if dtype is not None else weight.dtype
        deltas_dtype = target_dtype if target_dtype not in [torch.float8_e4m3fn, torch.float8_e5m2] else torch.bfloat16
        deltas = _prepare_deltas(lora_sd_and_strengths, key, deltas_dtype, device)
        if deltas is None:
            if key in sd:
                continue
            deltas = weight.clone().to(dtype=target_dtype, device=device)
        elif weight.dtype == torch.float8_e4m3fn:
            if str(device).startswith("cuda"):
                deltas = calculate_weight_float8_(deltas, weight)
            else:
                deltas.add_(weight.to(dtype=deltas.dtype, device=device))
        elif weight.dtype == torch.bfloat16:
            deltas.add_(weight)
        else:
            raise ValueError(f"Unsupported dtype: {weight.dtype}")
        sd[key] = deltas.to(dtype=target_dtype)
        inner_dtypes.add(target_dtype)
        size += deltas.nbytes
    if destination_sd is not None:
        return destination_sd
    return StateDict(sd, device, size, inner_dtypes)
