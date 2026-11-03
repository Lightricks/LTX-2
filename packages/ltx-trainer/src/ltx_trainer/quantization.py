"""LTX staging and exclusion policy over Optimum Quanto's public quantize/freeze API.
Quantization kernels and module replacement are provided by ``optimum-quanto``:
https://github.com/huggingface/optimum-quanto#quantization-workflow-for-vanilla-pytorch-models-low-level-api
"""

from collections.abc import Iterator
from typing import Literal

import torch
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from ltx_trainer import logger

QuantizationOptions = Literal["int8-quanto", "int4-quanto", "int2-quanto", "fp8-quanto", "fp8uz-quanto"]

# Quanto matches these globs against names relative to each conversion group.
EXCLUDE_PATTERNS = [
    "patchify_proj",
    "audio_patchify_proj",
    "proj_out",
    "audio_proj_out",
    "*adaln*",
    "time_proj",
    "timestep_embedder*",
    "caption_projection*",
    "audio_caption_projection*",
    "*norm*",
]
SKIP_ROOT_MODULES = {"patchify_proj", "audio_patchify_proj", "proj_out", "audio_proj_out", "audio_caption_projection"}


def _quantization_groups(model: torch.nn.Module) -> Iterator[torch.nn.Module]:
    """Yield conversion groups, showing progress while transformer blocks are processed."""
    if not hasattr(model, "transformer_blocks"):
        yield model
        return

    blocks = list(model.transformer_blocks)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Quantizing transformer blocks", total=len(blocks))
        for block in blocks:
            yield block
            progress.advance(task)

    logger.debug("Quantizing remaining model components")
    for name, child in model.named_children():
        if name == "transformer_blocks":
            continue
        if name in SKIP_ROOT_MODULES:
            logger.debug(f"Skipping quantization for module: {name}")
            continue
        yield child


def quantize_model(
    model: torch.nn.Module,
    precision: QuantizationOptions,
    quantize_activations: bool = False,
    device: torch.device | str | None = None,
) -> torch.nn.Module:
    """Quantize in place and restore the model's original device.
    Models with ``transformer_blocks`` stage each block, then eligible root children,
    on the conversion device in the first parameter's dtype, returning each to CPU.
    Other models convert as one group without a dtype cast. Exclusions are relative
    to each group; root children listed in ``SKIP_ROOT_MODULES`` are skipped entirely.
    Activation quantization is opt-in and requires Quanto calibration before use.
    """
    from optimum import quanto  # noqa: PLC0415

    conversion_device = torch.device(device if device is not None else "cuda" if torch.cuda.is_available() else "cpu")
    formats = {
        "int8-quanto": quanto.qint8,
        "int4-quanto": quanto.qint4,
        "int2-quanto": quanto.qint2,
        "fp8-quanto": quanto.qfloat8,
        "fp8uz-quanto": quanto.qfloat8_e4m3fnuz,
    }
    if precision not in formats:
        raise ValueError(f"Invalid quantization precision: {precision}")
    if precision in ("fp8-quanto", "fp8uz-quanto") and torch.backends.mps.is_available():
        raise ValueError("FP8 quantization is not supported on MPS devices. Use int2, int4, or int8 instead.")

    if quantize_activations:
        logger.debug("Quantizing model weights and activations")

    source_device = next(model.parameters()).device
    source_dtype = next(model.parameters()).dtype
    blockwise = hasattr(model, "transformer_blocks")
    if blockwise:
        logger.debug("Quantizing model using block-by-block approach for memory efficiency")
    for group in _quantization_groups(model):
        if blockwise:
            group.to(conversion_device, dtype=source_dtype, non_blocking=True)
        else:
            group.to(conversion_device)
        quanto.quantize(
            group,
            weights=formats[precision],
            activations=formats[precision] if quantize_activations else None,
            exclude=EXCLUDE_PATTERNS,
        )
        quanto.freeze(group)
        if blockwise:
            # Quanto repacks INT4 weights on CPU inside this transfer, so host data must be ready.
            group.to("cpu")
    model.to(source_device)
    return model
