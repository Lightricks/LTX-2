from dataclasses import dataclass
from enum import Enum

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.sd_ops import SDOps
from ltx_core.quantization.fp8_cast import TRANSFORMER_LINEAR_DOWNCAST_MAP, UPCAST_DURING_INFERENCE
from ltx_core.quantization.fp8_scaled_mm import get_fp8_swap_module_ops


@dataclass(frozen=True)
class QuantizationPolicy:
    """Configuration for model quantization during loading.
    Attributes:
        kind: Discriminator for the policy variant.
        sd_ops: State-dict operations applied to each tensor during load.
        module_ops: Post-load module transformations applied to the meta model.
    """

    class Kind(str, Enum):
        FP8_CAST = "fp8_cast"
        FP8_SCALED_MM = "fp8_scaled_mm"

    kind: Kind
    sd_ops: SDOps | None = None
    module_ops: tuple[ModuleOps, ...] = ()

    @classmethod
    def fp8_cast(cls) -> "QuantizationPolicy":
        """FP8 casting with upcasting during inference."""
        return cls(
            kind=cls.Kind.FP8_CAST,
            sd_ops=TRANSFORMER_LINEAR_DOWNCAST_MAP,
            module_ops=(UPCAST_DURING_INFERENCE,),
        )

    @classmethod
    def fp8_scaled_mm(cls, checkpoint_path: str) -> "QuantizationPolicy":
        """FP8 scaled matmul for checkpoints pre-quantized with per-tensor scales.
        The set of layers to swap to ``FP8Linear`` is discovered from the
        checkpoint's ``.weight_scale`` tensors via suffix-matching against the
        model's named modules. Requires a pre-quantized checkpoint; for BF16
        checkpoints, use :meth:`fp8_cast` instead.
        """
        return cls(
            kind=cls.Kind.FP8_SCALED_MM,
            sd_ops=None,
            module_ops=get_fp8_swap_module_ops(checkpoint_path),
        )
