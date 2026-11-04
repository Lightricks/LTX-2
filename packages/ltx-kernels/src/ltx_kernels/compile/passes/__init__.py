"""Concrete FX passes applied by the ltx-kernels backend."""

from torch.fx.passes.infra.pass_base import PassBase

from ltx_kernels.compile.passes.ada_zero import AdaZeroPass
from ltx_kernels.compile.passes.blockwise_quantize_cse import BlockwiseQuantizeCSE
from ltx_kernels.compile.passes.fuse_trailing_quantize import (
    AdaZeroTrailingQuantizePass,
    GatedAttentionTrailingQuantizePass,
)
from ltx_kernels.compile.passes.fused_send_recv import CoalescedExchangeBarrierPass, FusedSendRecvPass
from ltx_kernels.compile.passes.gated_attention import GatedAttentionPass
from ltx_kernels.compile.passes.grouped_upcast import GroupedUpcastPass
from ltx_kernels.compile.passes.linear_gelu import LinearGELUNoBiasPass, LinearGELUPass
from ltx_kernels.compile.passes.pre_attention import PreAttentionPass
from ltx_kernels.compile.passes.rms_fma import RMSFmaAdaLNPass, RMSFmaPass
from ltx_kernels.compile.passes.sdpa_select import SDPASelectPass


def inductor_passes() -> list[PassBase]:
    """The passes worth running when inductor compiles what is left.
    :class:`PreAttentionPass` + :class:`GatedAttentionPass` + :class:`LinearGELUPass`
    is the fastest combination measured end to end (:class:`LinearGELUNoBiasPass` is the
    same fusion for bias-free FFNs, which LinearGELUPass cannot match); the passes :func:`eager_passes`
    adds cost more there than they save, alone and together, by cutting inductor's own
    fusion across the block. The quantize passes are inert unless
    ``ltx_core.quantization.blockwise`` installs the linears that emit
    ``ltx_kernels::blockwise_quantize``. :class:`FusedSendRecvPass` is inert outside
    sequence parallelism and binds the op :class:`PreAttentionPass` leaves behind;
    :class:`CoalescedExchangeBarrierPass` then gives the exchanges it fuses one barrier per attention.
    :class:`GroupedUpcastPass` is inert outside the fp8-cast policy.
    One instance per stream shape (LTX-2.3: video 4096, audio 2048), since a pass pins
    the ``hidden_dim`` and ``heads`` literals its pattern carries.
    """
    return [
        PreAttentionPass(hidden_dim=4096, heads=32, eps=1e-6),
        PreAttentionPass(hidden_dim=2048, heads=32, eps=1e-6),
        GatedAttentionPass(),
        LinearGELUPass(),
        LinearGELUNoBiasPass(),
        FusedSendRecvPass(),
        CoalescedExchangeBarrierPass(),
        BlockwiseQuantizeCSE(),
        AdaZeroTrailingQuantizePass(hidden_dim=4096, eps=1e-6),
        AdaZeroTrailingQuantizePass(hidden_dim=2048, eps=1e-6),
        GatedAttentionTrailingQuantizePass(),
        GroupedUpcastPass(),
        SDPASelectPass(),
    ]


def eager_passes() -> list[PassBase]:
    """Every pass, for an inner backend that fuses nothing itself.
    The passes :func:`inductor_passes` leaves out lose to inductor's fusion, not to
    the ops they replace, so they are kept here -- reasoned from the inductor
    measurements rather than measured on eager.
    The tail is in claim order: the quantize passes precede the RMSFma passes, which
    have no quantized twin and would otherwise take those sites, and
    :class:`AdaZeroPass` is last because RMSFma claims its subgraph.
    """
    return [
        *inductor_passes(),
        RMSFmaAdaLNPass(hidden_dim=4096, eps=1e-6),
        RMSFmaAdaLNPass(hidden_dim=2048, eps=1e-6),
        RMSFmaPass(hidden_dim=4096, eps=1e-6),
        RMSFmaPass(hidden_dim=2048, eps=1e-6),
        AdaZeroPass(hidden_dim=4096, eps=1e-6),
        AdaZeroPass(hidden_dim=2048, eps=1e-6),
    ]


__all__ = [
    "AdaZeroPass",
    "AdaZeroTrailingQuantizePass",
    "BlockwiseQuantizeCSE",
    "CoalescedExchangeBarrierPass",
    "FusedSendRecvPass",
    "GatedAttentionPass",
    "GatedAttentionTrailingQuantizePass",
    "GroupedUpcastPass",
    "LinearGELUNoBiasPass",
    "LinearGELUPass",
    "PreAttentionPass",
    "RMSFmaAdaLNPass",
    "RMSFmaPass",
    "SDPASelectPass",
    "eager_passes",
    "inductor_passes",
]
