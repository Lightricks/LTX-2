"""BF16 linear projection with a fused approximate GELU epilogue."""

import torch


@torch.library.custom_op("ltx_kernels::linear_gelu", mutates_args=(), device_types="cuda")
def linear_gelu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """Apply a BF16 linear projection and tanh-approximate GELU.
    The cuBLASLt GELU epilogue behind ``_addmm_activation`` takes a bias vector, so a
    bias-free projection passes zeros. The zeros are allocated per call rather than
    cached: a first call inside CUDA-graph capture would otherwise cache a tensor
    owned by that graph's private pool.
    """
    batch_shape = x.shape[:-1]
    if bias is None:
        bias = x.new_zeros(weight.shape[0])
    out = torch.ops.aten._addmm_activation.default(
        bias,
        x.flatten(0, -2),
        weight.t(),
        use_gelu=True,
    )
    return out.unflatten(0, batch_shape)


@linear_gelu.register_fake
def _linear_gelu_fake(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:  # noqa: ARG001
    return x.new_empty((*x.shape[:-1], weight.shape[0]))
