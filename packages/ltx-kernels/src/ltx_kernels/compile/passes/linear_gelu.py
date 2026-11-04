"""Pass that fuses bf16 linear projections with approximate GELU."""

from typing import ClassVar

import torch
from torch.fx.passes.utils.matcher_utils import InternalMatch

from ltx_kernels.compile.subgraph_rewrite_pass import SubgraphRewritePass, is_cuda_bf16


class LinearGELUPass(SubgraphRewritePass):
    """Rewrite bf16 ``linear`` + tanh GELU to ``ltx_kernels::linear_gelu``."""

    ignore_literals: ClassVar[bool] = False

    @staticmethod
    def pattern_fn(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(torch.nn.functional.linear(x, weight, bias), approximate="tanh")

    @staticmethod
    def replacement_fn(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return torch.ops.ltx_kernels.linear_gelu.default(x, weight, bias)

    def _is_match_eligible(self, match: InternalMatch) -> bool:
        inputs = self._matched_inputs(match)
        if not super()._is_match_eligible(match) or not inputs:
            return False
        x, weight, bias = inputs
        # ``_addmm_activation`` takes a 2D mat1, which the op flattens x into.
        return is_cuda_bf16(x, weight, bias) and x.ndim >= 2 and weight.ndim == 2 and bias.ndim == 1

    def example_inputs(self) -> tuple:
        return (
            torch.empty(2, 64, 4096, dtype=torch.bfloat16, device="cuda"),
            torch.empty(16384, 4096, dtype=torch.bfloat16, device="cuda"),
            torch.empty(16384, dtype=torch.bfloat16, device="cuda"),
        )

    def _register_ops(self) -> None:
        from ltx_kernels.ops import linear_gelu as _linear_gelu_ops  # noqa: F401, PLC0415


class LinearGELUNoBiasPass(SubgraphRewritePass):
    """Rewrite bias-free bf16 ``linear`` + tanh GELU to ``ltx_kernels::linear_gelu``.
    Matches ``linear(x, weight, None)``, the call ``nn.Linear(bias=False)`` makes, as in
    the LTX-2.5 video FFN (``ff_bias=false``). :class:`LinearGELUPass` binds the bias to a
    placeholder and so never matches it; neither pattern matches the other's graphs.
    """

    ignore_literals: ClassVar[bool] = False

    @staticmethod
    def pattern_fn(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(torch.nn.functional.linear(x, weight, None), approximate="tanh")

    @staticmethod
    def replacement_fn(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.ops.ltx_kernels.linear_gelu.default(x, weight, None)

    def _is_match_eligible(self, match: InternalMatch) -> bool:
        inputs = self._matched_inputs(match)
        if not super()._is_match_eligible(match) or not inputs:
            return False
        x, weight = inputs
        # ``_addmm_activation`` takes a 2D mat1, which the op flattens x into.
        return is_cuda_bf16(x, weight) and x.ndim >= 2 and weight.ndim == 2

    def example_inputs(self) -> tuple:
        return (
            torch.empty(2, 64, 4096, dtype=torch.bfloat16, device="cuda"),
            torch.empty(16384, 4096, dtype=torch.bfloat16, device="cuda"),
        )

    def _register_ops(self) -> None:
        from ltx_kernels.ops import linear_gelu as _linear_gelu_ops  # noqa: F401, PLC0415
