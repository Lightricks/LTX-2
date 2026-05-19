from collections.abc import Iterable, Iterator
from typing import NamedTuple

import torch

from ltx_core.loader.kernels import TRITON_AVAILABLE
from ltx_core.loader.primitives import LoraStateDictWithStrength, StateDict
from ltx_core.quantization.fp8_cast import fused_add_round_launch
from ltx_core.quantization.fp8_scaled_mm import quantize_weight_to_fp8_per_tensor


class LoraProduct(NamedTuple):
    """A LoRA's ``A``, ``B`` factors and its strength scalar."""

    a: torch.Tensor
    b: torch.Tensor
    strength: float


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def aggregate_lora_products(
    products: Iterable[LoraProduct],
    dtype: torch.dtype | None = None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Accumulate ``sum((B * strength) @ A)`` across :class:`LoraProduct` items.
    If ``out`` is provided, ``addmm_`` accumulates directly into it — caller
    ensures A/B dtypes and devices match ``out``. Otherwise the first product
    materializes the ``(out, in)``-shape aggregator at ``dtype``; subsequent
    products use ``addmm_`` to avoid allocating the full intermediate delta.
    Returns ``out`` (or the new aggregator), or ``None`` if ``products`` was empty
    and ``out`` was not given.
    """
    aggregated = out
    for product in products:
        if aggregated is None:
            aggregated = torch.matmul(product.b * product.strength, product.a).to(dtype=dtype)
        else:
            aggregated.addmm_(product.b, product.a, alpha=product.strength)
    return aggregated


def fuse_cast_fp8_weight(
    delta_bf16: torch.Tensor,
    weight_fp8: torch.Tensor,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Return ``(delta_bf16 + dequantize(weight_fp8)).to(target_dtype)``.
    CUDA with Triton uses stochastic rounding; otherwise uses a deterministic bf16 add.
    ``delta_bf16`` is the bf16 accumulator and is mutated in place.
    """
    if delta_bf16.dtype != torch.bfloat16:
        raise ValueError(f"delta_bf16 must be bfloat16, got {delta_bf16.dtype}")
    if str(weight_fp8.device).startswith("cuda") and TRITON_AVAILABLE:
        fused_add_round_launch(delta_bf16, weight_fp8, seed=0)
    else:
        delta_bf16.add_(weight_fp8.to(dtype=torch.bfloat16))
    return delta_bf16.to(dtype=target_dtype)


def fuse_lora_weights(
    model_sd: StateDict,
    lora_sd_and_strengths: list[LoraStateDictWithStrength],
    dtype: torch.dtype | None = None,
    preserve_input_device: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(key, fused_tensor)`` for each weight modified by at least one LoRA.
    For scaled-FP8 weights, this includes both the updated ``.weight`` tensor
    and its corresponding ``.weight_scale`` tensor.
    When ``preserve_input_device`` is False, fused tensors are yielded on the device
    used for fusion; caller is responsible for moving them to their final
    destination.
    """
    for key, original_weight in model_sd.sd.items():
        if original_weight is None or key.endswith(".weight_scale"):
            continue
        original_device = original_weight.device
        weight = original_weight.to(device=_get_device())
        target_dtype = dtype if dtype is not None else weight.dtype
        deltas_dtype = target_dtype if target_dtype not in [torch.float8_e4m3fn, torch.float8_e5m2] else torch.bfloat16

        deltas = _aggregate_deltas(lora_sd_and_strengths, key, deltas_dtype, weight.device)
        if deltas is None:
            continue

        scale_key = key.replace(".weight", ".weight_scale") if key.endswith(".weight") else None
        is_scaled_fp8 = scale_key is not None and scale_key in model_sd.sd

        if weight.dtype == torch.float8_e4m3fn:
            if is_scaled_fp8:
                fused = _fuse_delta_with_scaled_fp8(deltas, weight, key, scale_key, model_sd)
            else:
                fused = {key: fuse_cast_fp8_weight(deltas, weight, target_dtype)}
        elif weight.dtype == torch.bfloat16:
            deltas.add_(weight)
            fused = {key: deltas.to(dtype=target_dtype)}
        else:
            raise ValueError(f"Unsupported dtype: {weight.dtype}")

        for k, v in fused.items():
            yield k, v.to(device=original_device) if preserve_input_device else v


def apply_loras(
    model_sd: StateDict,
    lora_sd_and_strengths: list[LoraStateDictWithStrength],
    dtype: torch.dtype | None = None,
    destination_sd: StateDict | None = None,
) -> StateDict:
    """Fuse LoRAs into ``model_sd`` and place the results in ``destination_sd``.
    When ``destination_sd`` is provided, the fused tensors are placed directly into it.
    """
    if destination_sd is not None:
        for key, fused in fuse_lora_weights(model_sd, lora_sd_and_strengths, dtype):
            destination_sd.sd[key] = fused
        return destination_sd

    fused = dict(fuse_lora_weights(model_sd, lora_sd_and_strengths, dtype))
    sd = {k: (fused[k] if k in fused else v.clone()) for k, v in model_sd.sd.items()}
    return StateDict(sd, model_sd.device, model_sd.size, model_sd.dtype)


def _aggregate_deltas(
    lora_sd_and_strengths: list[LoraStateDictWithStrength], key: str, dtype: torch.dtype, device: torch.device
) -> torch.Tensor | None:
    prefix = key[: -len(".weight")]
    key_a = f"{prefix}.lora_A.weight"
    key_b = f"{prefix}.lora_B.weight"

    def _ab_products() -> Iterator[LoraProduct]:
        for lsd, coef in lora_sd_and_strengths:
            if key_a not in lsd.sd or key_b not in lsd.sd:
                continue
            a = lsd.sd[key_a].to(device=device, dtype=dtype, non_blocking=True)
            b = lsd.sd[key_b].to(device=device, dtype=dtype, non_blocking=True)
            yield LoraProduct(a, b, coef)

    return aggregate_lora_products(_ab_products(), dtype)


def _fuse_delta_with_scaled_fp8(
    deltas: torch.Tensor,
    weight: torch.Tensor,
    key: str,
    scale_key: str,
    model_sd: StateDict,
) -> dict[str, torch.Tensor]:
    """Dequantize scaled FP8 weight, add LoRA delta, and re-quantize."""
    weight_scale = model_sd.sd[scale_key]

    original_weight = weight.to(torch.float32) * weight_scale

    new_weight = original_weight + deltas.to(torch.float32)

    new_fp8_weight, new_weight_scale = quantize_weight_to_fp8_per_tensor(new_weight)
    return {key: new_fp8_weight, scale_key: new_weight_scale}
