"""Runtime detection of TensorRT-LLM FP8 scaled-matmul availability.
When the TRT-LLM ops are usable on the current host (Linux + Hopper-class CUDA
+ tensorrt_llm wheel installed) we use them since they outperform the PyTorch-native
``torch._scaled_mm`` path. Otherwise we fall back to the native implementation,
which is portable across platforms (Windows, macOS, AMD GPUs).
The check runs once and is cached.
"""

from __future__ import annotations

import platform
from functools import cache

import torch


@cache
def trtllm_scaled_mm_usable() -> bool:
    if platform.system() != "Linux":
        return False

    if not torch.cuda.is_available():
        return False

    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor

    if sm < 90 or sm >= 120:
        return False

    # The import is load-bearing — registers the trtllm torch ops as a side effect.
    try:
        import tensorrt_llm  # noqa: F401, PLC0415
    except Exception:
        return False

    return True
