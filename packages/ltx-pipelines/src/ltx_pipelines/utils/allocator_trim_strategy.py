from enum import Enum


class AllocatorTrimStrategy(Enum):
    """How a block releases its model's memory when its scope exits."""

    TRIM = "trim"  # sync, release storage (to meta), and empty_cache() back to the OS
    DEFER = "defer"  # skip teardown; let GC reclaim it and keep the CUDA cache warm
