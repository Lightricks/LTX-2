"""
Block-swapping for LTX transformer using ModelOffloader pattern.

Keeps only a subset of transformer blocks in GPU memory at a time,
swapping them in/out from CPU as needed using ThreadPoolExecutor
and CUDA streams for efficient async transfers.

Based on the working block swapping implementation from h1111/modules/custom_offloading_utils.py.
"""

import types

import torch
from torch import nn

from ltx_core.model.transformer.model import LTXModel, X0Model

from .custom_offloading_utils import ModelOffloader, clean_memory_on_device, weighs_to_device


def enable_block_swap(
    model: X0Model | LTXModel,
    blocks_in_memory: int = 6,
    device: torch.device | str = "cuda",
) -> ModelOffloader:
    """
    Enable block swapping on an existing X0Model or LTXModel using ModelOffloader.

    This function:
    1. Creates a ModelOffloader for async block transfers
    2. Prepares initial block positions (first N on GPU, rest on CPU)
    3. Monkey-patches _process_transformer_blocks to use wait/submit pattern

    Args:
        model: X0Model (wraps LTXModel) or LTXModel directly.
        blocks_in_memory: Number of transformer blocks to keep in GPU (default: 6).
        device: Target GPU device.

    Returns:
        ModelOffloader instance for controlling the swapping behavior.

    Example:
        transformer = model_ledger.transformer()
        offloader = enable_block_swap(transformer, blocks_in_memory=6)
        # ... run inference ...
        # Cleanup handled automatically
    """
    # Get the underlying LTXModel
    if isinstance(model, X0Model):
        ltx_model = model.velocity_model
    else:
        ltx_model = model

    device = torch.device(device) if isinstance(device, str) else device
    num_blocks = len(ltx_model.transformer_blocks)
    blocks_to_swap = num_blocks - blocks_in_memory

    if blocks_to_swap <= 0:
        print(f"[BlockSwap] blocks_in_memory ({blocks_in_memory}) >= num_blocks ({num_blocks}), no swapping needed")
        return None

    # Get reference to the actual blocks (not a copy!)
    blocks = ltx_model.transformer_blocks

    # Create offloader with ThreadPoolExecutor for async transfers
    offloader = ModelOffloader(
        block_type="ltx_transformer_block",
        blocks=blocks,
        num_blocks=num_blocks,
        blocks_to_swap=blocks_to_swap,
        supports_backward=False,
        device=device,
    )

    # Store on model for access in forward pass
    ltx_model._block_swap_offloader = offloader
    ltx_model._blocks_to_swap = blocks_to_swap
    ltx_model._blocks_ref = blocks  # Store reference for forward pass
    if isinstance(model, X0Model):
        model._block_swap_offloader = offloader
        model._blocks_to_swap = blocks_to_swap
        model._blocks_ref = blocks

    # Prepare block positions: first (num_blocks - blocks_to_swap) on GPU, rest on CPU
    offloader.prepare_block_devices_before_forward(blocks)

    # Store original method for potential restoration
    ltx_model._original_process_transformer_blocks = ltx_model._process_transformer_blocks

    # Create replacement method using wait/submit pattern
    def block_swap_process_transformer_blocks(self, video, audio, perturbations):
        """Process transformer blocks with block swapping using wait/submit pattern."""
        offloader = self._block_swap_offloader
        blocks = self._blocks_ref  # Use stored reference, not a copy

        for block_idx, block in enumerate(self.transformer_blocks):
            # Wait for this block to be ready BEFORE using it
            offloader.wait_for_block(block_idx)

            # Process the block
            video, audio = block(
                video=video,
                audio=audio,
                perturbations=perturbations,
            )

            # Submit swap for next iteration AFTER using block
            offloader.submit_move_blocks_forward(blocks, block_idx)

        return video, audio

    # Monkey-patch the method
    ltx_model._process_transformer_blocks = types.MethodType(block_swap_process_transformer_blocks, ltx_model)

    print(f"[BlockSwap] Enabled: {blocks_in_memory}/{num_blocks} blocks in GPU, {blocks_to_swap} swapping")
    return offloader


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

    if hasattr(ltx_model, "_block_swap_offloader"):
        offloader = ltx_model._block_swap_offloader
        device = offloader.device

        # Move all blocks back to GPU
        for block in ltx_model.transformer_blocks:
            block.to(device)
            weighs_to_device(block, device)

        # Restore original method if saved
        if hasattr(ltx_model, "_original_process_transformer_blocks"):
            ltx_model._process_transformer_blocks = ltx_model._original_process_transformer_blocks
            del ltx_model._original_process_transformer_blocks

        del ltx_model._block_swap_offloader
        del ltx_model._blocks_to_swap

        if isinstance(model, X0Model):
            if hasattr(model, "_block_swap_offloader"):
                del model._block_swap_offloader
            if hasattr(model, "_blocks_to_swap"):
                del model._blocks_to_swap

        clean_memory_on_device(device)
        print(f"[BlockSwap] Disabled: all blocks moved to GPU")


def get_block_swap_offloader(model: X0Model | LTXModel) -> ModelOffloader | None:
    """
    Get the ModelOffloader for a model if block swapping is enabled.

    Args:
        model: Model to check.

    Returns:
        ModelOffloader instance or None if not enabled.
    """
    if isinstance(model, X0Model):
        ltx_model = model.velocity_model
    else:
        ltx_model = model

    return getattr(ltx_model, "_block_swap_offloader", None)


def offload_all_blocks(model: X0Model | LTXModel) -> None:
    """
    Offload all transformer blocks to CPU and cleanup the offloader.

    Used to free GPU memory after inference is complete.

    Args:
        model: Model with block swapping enabled.
    """
    if isinstance(model, X0Model):
        ltx_model = model.velocity_model
    else:
        ltx_model = model

    offloader = getattr(ltx_model, "_block_swap_offloader", None)
    if offloader is None:
        return

    device = offloader.device

    # Wait for any pending async operations
    for idx in range(len(ltx_model.transformer_blocks)):
        if idx in offloader.futures:
            offloader._wait_blocks_move(idx)

    # Shutdown the ThreadPoolExecutor
    offloader.thread_pool.shutdown(wait=True)
    offloader.futures.clear()

    # Synchronize CUDA before moving blocks
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Move all blocks to CPU
    for block in ltx_model.transformer_blocks:
        weighs_to_device(block, "cpu")

    # Synchronize again after moves complete
    if device.type == "cuda":
        torch.cuda.synchronize()

    clean_memory_on_device(device)
    print("[BlockSwap] All blocks offloaded to CPU")
