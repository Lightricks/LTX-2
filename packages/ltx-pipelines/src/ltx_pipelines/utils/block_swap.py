"""
Block-swapping for LTX transformer to run 19B model on limited VRAM.

Keeps only a subset of transformer blocks in GPU memory at a time,
swapping them in/out from CPU as needed. This can reduce VRAM usage
by ~40% when using 6 blocks in memory (out of 48 total).

Based on the block swapping approach from Kandinsky5.
"""

import torch
from torch import nn
from typing import Callable

from ltx_core.model.transformer.model import LTXModel, X0Model


class BlockSwapManager:
    """
    Manages block swapping for an LTXModel's transformer blocks.

    This is a runtime manager that can be attached to any existing LTXModel
    to enable block swapping without requiring model re-instantiation.

    Memory estimates for 19B model (48 blocks):
    - Full model: ~48GB VRAM
    - 12 blocks in memory: ~19GB VRAM
    - 6 blocks in memory: ~14GB VRAM (default)
    - 4 blocks in memory: ~13GB VRAM
    """

    def __init__(
        self,
        model: LTXModel,
        blocks_in_memory: int = 6,
        device: torch.device | str = "cuda",
    ):
        """
        Initialize block swap manager for an LTXModel.

        Args:
            model: The LTXModel to manage block swapping for.
            blocks_in_memory: Number of transformer blocks to keep in GPU memory.
            device: Target GPU device.
        """
        self.model = model
        self.blocks_in_memory = blocks_in_memory
        self.device = torch.device(device) if isinstance(device, str) else device
        self.num_blocks = len(model.transformer_blocks)
        self._blocks_on_gpu: set[int] = set()
        self._enabled = False

    def enable(self) -> "BlockSwapManager":
        """
        Enable block swapping by moving all blocks to CPU except the first N.

        Returns self for chaining.
        """
        if self._enabled:
            return self

        # Move all blocks to CPU first
        for i in range(self.num_blocks):
            self.model.transformer_blocks[i].to("cpu", non_blocking=True)

        self._blocks_on_gpu.clear()

        # Prefetch first blocks to GPU
        for i in range(min(self.blocks_in_memory, self.num_blocks)):
            self.model.transformer_blocks[i].to(self.device, non_blocking=True)
            self._blocks_on_gpu.add(i)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self._enabled = True
        print(f"[BlockSwap] Enabled: {self.blocks_in_memory}/{self.num_blocks} blocks in GPU")
        return self

    def disable(self) -> "BlockSwapManager":
        """
        Disable block swapping by moving all blocks back to GPU.

        Returns self for chaining.
        """
        if not self._enabled:
            return self

        for i in range(self.num_blocks):
            self.model.transformer_blocks[i].to(self.device, non_blocking=True)
            self._blocks_on_gpu.add(i)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self._enabled = False
        print(f"[BlockSwap] Disabled: all {self.num_blocks} blocks in GPU")
        return self

    def ensure_block_on_gpu(self, block_idx: int) -> None:
        """
        Ensure a specific transformer block is on GPU.

        Uses FIFO strategy: when at capacity, offloads the lowest-indexed
        block currently on GPU.
        """
        if not self._enabled:
            return

        if block_idx in self._blocks_on_gpu:
            return

        # If at capacity, offload oldest block
        if len(self._blocks_on_gpu) >= self.blocks_in_memory:
            oldest_idx = min(self._blocks_on_gpu)
            self.model.transformer_blocks[oldest_idx].to("cpu", non_blocking=True)
            self._blocks_on_gpu.remove(oldest_idx)

        # Load requested block
        self.model.transformer_blocks[block_idx].to(self.device, non_blocking=True)
        self._blocks_on_gpu.add(block_idx)

    def prefetch_next(self, current_idx: int) -> None:
        """Prefetch the next block while current block is processing."""
        next_idx = current_idx + 1
        if next_idx < self.num_blocks:
            self.ensure_block_on_gpu(next_idx)

    def offload_all(self) -> None:
        """Offload all transformer blocks to CPU."""
        if not self._enabled:
            return

        for idx in list(self._blocks_on_gpu):
            self.model.transformer_blocks[idx].to("cpu", non_blocking=True)
        self._blocks_on_gpu.clear()

        if torch.cuda.is_available():
            torch.cuda.synchronize()


def _create_block_swap_forward(
    original_process_blocks: Callable,
    manager: BlockSwapManager,
) -> Callable:
    """
    Create a wrapper for _process_transformer_blocks that handles block swapping.
    """
    def block_swap_process_transformer_blocks(self, video, audio, perturbations):
        if not manager._enabled:
            return original_process_blocks(video, audio, perturbations)

        # Determine device from input tensors
        if video is not None and video.x is not None:
            device = video.x.device
        elif audio is not None and audio.x is not None:
            device = audio.x.device
        else:
            device = manager.device

        # Process each block with swapping
        for i, block in enumerate(self.transformer_blocks):
            # Prefetch next block while processing current
            manager.prefetch_next(i)

            # Ensure current block is on GPU
            manager.ensure_block_on_gpu(i)

            # Synchronize to ensure block is fully loaded
            if i > 0 and torch.cuda.is_available():
                torch.cuda.synchronize()

            # Process block
            video, audio = block(
                video=video,
                audio=audio,
                perturbations=perturbations,
            )

        # Synchronize to ensure all async transfers complete before returning
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        return video, audio

    return block_swap_process_transformer_blocks


def enable_block_swap(
    model: X0Model | LTXModel,
    blocks_in_memory: int = 6,
    device: torch.device | str = "cuda",
) -> BlockSwapManager:
    """
    Enable block swapping on an existing X0Model or LTXModel.

    This function:
    1. Creates a BlockSwapManager for the model
    2. Monkey-patches _process_transformer_blocks to use block swapping
    3. Moves blocks to CPU except the first N

    Args:
        model: X0Model (wraps LTXModel) or LTXModel directly.
        blocks_in_memory: Number of transformer blocks to keep in GPU (default: 6).
        device: Target GPU device.

    Returns:
        BlockSwapManager instance for controlling the swapping behavior.

    Example:
        transformer = model_ledger.transformer()
        manager = enable_block_swap(transformer, blocks_in_memory=6)
        # ... run inference ...
        manager.offload_all()  # Free GPU memory
    """
    # Get the underlying LTXModel
    if isinstance(model, X0Model):
        ltx_model = model.velocity_model
    else:
        ltx_model = model

    # Create manager
    manager = BlockSwapManager(ltx_model, blocks_in_memory, device)

    # Store original method
    original_process_blocks = ltx_model._process_transformer_blocks

    # Create and bind the new method
    import types
    new_method = _create_block_swap_forward(original_process_blocks, manager)
    ltx_model._process_transformer_blocks = types.MethodType(new_method, ltx_model)

    # Store manager reference on model for later access
    ltx_model._block_swap_manager = manager
    if isinstance(model, X0Model):
        model._block_swap_manager = manager

    # Enable block swapping (moves blocks to CPU except first N)
    manager.enable()

    return manager


def disable_block_swap(model: X0Model | LTXModel) -> None:
    """
    Disable block swapping and restore original behavior.

    Args:
        model: Model that had block swapping enabled.
    """
    if isinstance(model, X0Model):
        ltx_model = model.velocity_model
    else:
        ltx_model = model

    if hasattr(ltx_model, "_block_swap_manager"):
        ltx_model._block_swap_manager.disable()


def get_block_swap_manager(model: X0Model | LTXModel) -> BlockSwapManager | None:
    """
    Get the BlockSwapManager for a model if block swapping is enabled.

    Args:
        model: Model to check.

    Returns:
        BlockSwapManager instance or None if not enabled.
    """
    if isinstance(model, X0Model):
        ltx_model = model.velocity_model
    else:
        ltx_model = model

    return getattr(ltx_model, "_block_swap_manager", None)
