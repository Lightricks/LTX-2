"""Load GemmaTextEncoder with Accelerate ``device_map="auto"``.
The Gemma LLM backbone is spread across available CUDA devices using
HuggingFace Accelerate's automatic device placement.
Mirrors the ``PromptEncoder`` text-encoder loading in
``ltx_pipelines.utils.blocks`` but uses ``device_map="auto"`` instead of
placing the entire model on a single GPU.
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoImageProcessor, Gemma3ForConditionalGeneration, Gemma3Processor

from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoder
from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
from ltx_core.utils import find_matching_file

logger = logging.getLogger(__name__)


def load_gemma_with_device_map(
    gemma_root_path: str,
    dtype: torch.dtype = torch.bfloat16,
) -> GemmaTextEncoder:
    """Load GemmaTextEncoder with the LLM backbone spread across GPUs.
    Uses ``Gemma3ForConditionalGeneration.from_pretrained(device_map="auto")``
    to distribute layers across available CUDA devices.
    Args:
        gemma_root_path: Path to Gemma model directory.
        dtype: Data type for model weights.
    """
    model_folder = str(find_matching_file(gemma_root_path, "model*.safetensors").parent)
    tokenizer_path = str(find_matching_file(gemma_root_path, "tokenizer.model").parent)
    processor_path = str(find_matching_file(gemma_root_path, "preprocessor_config.json").parent)

    logger.info("Loading Gemma LLM with device_map='auto'...")
    gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
        model_folder,
        dtype=dtype,
        device_map="auto",
        local_files_only=True,
    )

    tokenizer = LTXVGemmaTokenizer(tokenizer_path, 1024)
    image_processor = AutoImageProcessor.from_pretrained(processor_path, local_files_only=True, use_fast=False)
    processor = Gemma3Processor(image_processor=image_processor, tokenizer=tokenizer.tokenizer)

    return GemmaTextEncoder(
        model=gemma_model,
        tokenizer=tokenizer,
        processor=processor,
        dtype=dtype,
    )
