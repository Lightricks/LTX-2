import torch

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.sd_ops import KeyValueOperationResult, SDOps
from ltx_core.model.transformer.model import LTXModel

BLOCK_SIZE = 1024


def calculate_weight_float8(target_weights: torch.Tensor, original_weights: torch.Tensor) -> torch.Tensor:
    result = _fused_add_round_launch(target_weights, original_weights, seed=0).to(target_weights.dtype)
    target_weights.copy_(result, non_blocking=True)
    return target_weights


def _fused_add_round_launch(target_weight: torch.Tensor, original_weight: torch.Tensor, seed: int) -> torch.Tensor:
    # Lazy import triton - only available on CUDA platforms
    import triton  # noqa: PLC0415

    from ltx_core.loader.kernels import fused_add_round_kernel  # noqa: PLC0415

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


def _naive_weight_or_bias_downcast(key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
    """
    Downcast the weight or bias to the float8_e4m3fn dtype.
    """
    return [KeyValueOperationResult(key, value.to(dtype=torch.float8_e4m3fn))]


def _upcast_and_round(
    weight: torch.Tensor, dtype: torch.dtype, with_stochastic_rounding: bool = False, seed: int = 0
) -> torch.Tensor:
    """
    Upcast the weight to the given dtype and optionally apply stochastic rounding.
    Input weight needs to have float8_e4m3fn or float8_e5m2 dtype.
    """
    if not with_stochastic_rounding:
        return weight.to(dtype)
    return _fused_add_round_launch(torch.zeros_like(weight, dtype=dtype), weight, seed)


def _replace_fwd_with_upcast(
    layer: torch.nn.Linear,
    with_stochastic_rounding: bool = False,
    seed: int = 0,
    weight_scale: float | None = None,
) -> None:
    """
    Replace linear.forward with a version that:
      - upcasts weight and bias to input's dtype
      - applies weight_scale if the checkpoint was quantized with per-tensor scaling
      - returns F.linear calculated in that dtype

    Args:
        layer: The Linear layer to patch.
        with_stochastic_rounding: Whether to use stochastic rounding during upcast.
        seed: Seed for stochastic rounding.
        weight_scale: Per-tensor scale factor from the FP8 checkpoint. When provided,
            the dequantized weight is multiplied by this value. This is required for
            FP8 checkpoints that were quantized with per-tensor scaling (e.g.
            ``ltx-2.3-22b-dev-fp8.safetensors``) where each weight tensor has an
            associated ``weight_scale`` stored alongside it.
    """

    layer.original_forward = layer.forward

    def new_linear_forward(*args, **_kwargs) -> torch.Tensor:
        # assume first arg is the input tensor
        x = args[0]
        w_up = _upcast_and_round(layer.weight, x.dtype, with_stochastic_rounding, seed)

        # Apply per-tensor weight scale from FP8 checkpoint if available.
        # Without this, pre-quantized FP8 checkpoints produce incorrect outputs
        # because the raw FP8 values are not in the correct magnitude range.
        if weight_scale is not None:
            w_up = w_up * weight_scale

        b_up = None

        if layer.bias is not None:
            b_up = _upcast_and_round(layer.bias, x.dtype, with_stochastic_rounding, seed)

        return torch.nn.functional.linear(x, w_up, b_up)

    layer.forward = new_linear_forward


def _amend_forward_with_upcast(
    model: torch.nn.Module, with_stochastic_rounding: bool = False, seed: int = 0
) -> torch.nn.Module:
    """
    Replace the forward method of the model's Linear layers to forward
    with upcast and optional stochastic rounding.

    If the model was loaded from a pre-quantized FP8 checkpoint that includes
    per-tensor ``weight_scale`` values (stashed on the model by the builder as
    ``_fp8_weight_scales``), those scales are automatically applied during the
    upcast to produce correctly-scaled outputs.

    This is necessary because pre-quantized FP8 checkpoints (e.g.
    ``ltx-2.3-22b-dev-fp8.safetensors``) store weights in a scaled FP8 format
    where the raw FP8 values must be multiplied by their associated
    ``weight_scale`` to recover the correct magnitude.  Without this, a naive
    ``.to(bfloat16)`` produces values that are orders of magnitude too large,
    resulting in noise output.
    """
    # Retrieve per-tensor weight scales stashed by the model builder
    weight_scales: dict[str, float] = getattr(model, "_fp8_weight_scales", {})

    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            scale = weight_scales.get(name, None)
            _replace_fwd_with_upcast(m, with_stochastic_rounding, seed, weight_scale=scale)
    return model


TRANSFORMER_LINEAR_DOWNCAST_MAP = (
    SDOps("TRANSFORMER_LINEAR_DOWNCAST_MAP")
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_q.weight", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_q.bias", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_k.weight", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_k.bias", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_v.weight", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_v.bias", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_out.0.weight", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_out.0.bias", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix="ff.net.0.proj.weight", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix="ff.net.0.proj.bias", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix="ff.net.2.weight", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix="ff.net.2.bias", operation=_naive_weight_or_bias_downcast
    )
)

UPCAST_DURING_INFERENCE = ModuleOps(
    name="upcast_fp8_during_linear_forward",
    matcher=lambda model: isinstance(model, LTXModel),
    mutator=lambda model: _amend_forward_with_upcast(model, False),
)


class UpcastWithStochasticRounding(ModuleOps):
    """
    ModuleOps for upcasting the model's float8_e4m3fn weights and biases to the bfloat16 dtype
    and applying stochastic rounding during linear forward.
    """

    def __new__(cls, seed: int = 0):
        return super().__new__(
            cls,
            name="upcast_fp8_during_linear_forward_with_stochastic_rounding",
            matcher=lambda model: isinstance(model, LTXModel),
            mutator=lambda model: _amend_forward_with_upcast(model, True, seed),
        )
