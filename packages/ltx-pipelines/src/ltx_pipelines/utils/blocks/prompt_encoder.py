from __future__ import annotations

import logging
from contextlib import AbstractContextManager

import torch

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.block_streaming import DISK_CPU_SLOTS, StreamingModelBuilder
from ltx_core.loader.primitives import BuilderProtocol
from ltx_core.loader.registry import ModelRegistry, Registry
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.text_encoders.gemma import (
    EMBEDDINGS_PROCESSOR_KEY_OPS,
    EmbeddingsProcessorConfigurator,
    GemmaTextEncoderConfigurator,
    gemma_model_type,
    get_gemma_ops,
    resolve_gemma_weight_paths,
)
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor, EmbeddingsProcessorOutput
from ltx_pipelines.utils.gpu_model import gpu_model
from ltx_pipelines.utils.helpers import generate_enhanced_prompt
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.streaming_model import streaming_model
from ltx_pipelines.utils.types import OffloadMode

logger = logging.getLogger(__name__)

__all__ = ["PromptEncoder"]


class PromptEncoder:
    """Owns text encoder + embeddings processor lifecycle.
    Loads Gemma, optionally enhances then encodes prompts, frees Gemma, then loads the
    embeddings processor. ``_enhancer_text_encoder_builder`` is always set: when enhance
    and encode share a checkpoint it aliases ``_text_encoder_builder`` (same object /
    residency); a distinct builder means sequential enhance-then-encode contexts.
    """

    def __init__(
        self,
        model_paths: ModelPaths,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
        text_encoder_builder: BuilderProtocol | None = None,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
        prompt_enhancer_gemma_root: str | None = None,
        enhancer_text_encoder_builder: BuilderProtocol | None = None,
    ) -> None:
        text_encoder_path = model_paths.text_encoder()
        embeddings_paths = model_paths.embeddings_weight_paths
        if not embeddings_paths:
            raise ValueError(
                "PromptEncoder requires ModelPaths.embeddings_weight_paths "
                "(transformer checkpoint and, in split mode, the text-encoder file)."
            )
        self._text_encoder_path = text_encoder_path
        self._embeddings_paths = embeddings_paths
        self._dtype = dtype
        self._device = device
        self._offload_mode = offload_mode
        self._alloc_trim_strategy = alloc_trim_strategy
        self._prompt_enhancer_gemma_root = prompt_enhancer_gemma_root

        registry = registry or ModelRegistry(cache_models=True, cache_weights=False)
        self._registry = registry

        if text_encoder_builder is not None:
            encode_type = text_encoder_builder.model_config().get("model_type") or gemma_model_type(text_encoder_path)
        else:
            encode_type = gemma_model_type(text_encoder_path)
        if encode_type not in _ENCODE_MODEL_TYPES:
            raise ValueError(
                f"Encode root model_type={encode_type!r} is not supported for encoding; "
                f"expected one of {sorted(_ENCODE_MODEL_TYPES)}."
            )
        self._encode_model_type = encode_type

        if text_encoder_builder is not None:
            if offload_mode != OffloadMode.NONE:
                raise ValueError(
                    "text_encoder_builder cannot be used with offload_mode != OffloadMode.NONE "
                    "because no streaming text encoder builder is available."
                )
            self._text_encoder_builder = text_encoder_builder
            self._streaming_text_encoder_builder = None
        else:
            gemma_sd_ops, gemma_module_ops = get_gemma_ops(text_encoder_path)
            weight_paths = resolve_gemma_weight_paths(text_encoder_path)
            self._text_encoder_builder = Builder(
                model_path=weight_paths,
                model_class_configurator=GemmaTextEncoderConfigurator.with_gemma_model_path(text_encoder_path),
                model_sd_ops=gemma_sd_ops,
                module_ops=gemma_module_ops,
                registry=registry,
            )
            self._streaming_text_encoder_builder = StreamingModelBuilder(
                model_path=weight_paths,
                model_class_configurator=GemmaTextEncoderConfigurator.with_gemma_model_path(text_encoder_path),
                model_sd_ops=gemma_sd_ops,
                module_ops=gemma_module_ops,
                registry=registry,
                blocks_attr="model.model.language_model.layers",
                blocks_prefix="model.model.language_model.layers",
                cpu_slots_count=DISK_CPU_SLOTS if offload_mode == OffloadMode.DISK else None,
            )

        # Always set: shared enhance aliases the encode builder (same object / residency).
        if enhancer_text_encoder_builder is not None:
            self._enhancer_text_encoder_builder = enhancer_text_encoder_builder
        elif prompt_enhancer_gemma_root is not None and prompt_enhancer_gemma_root != text_encoder_path:
            enhancer_sd_ops, enhancer_module_ops = get_gemma_ops(prompt_enhancer_gemma_root)
            enhancer_paths = resolve_gemma_weight_paths(prompt_enhancer_gemma_root)
            self._enhancer_text_encoder_builder = Builder(
                model_path=enhancer_paths,
                model_class_configurator=GemmaTextEncoderConfigurator.with_gemma_model_path(prompt_enhancer_gemma_root),
                model_sd_ops=enhancer_sd_ops,
                module_ops=enhancer_module_ops,
                registry=registry,
            )
        else:
            self._enhancer_text_encoder_builder = self._text_encoder_builder

        self._embeddings_processor_builder = Builder(
            model_path=embeddings_paths,
            model_class_configurator=EmbeddingsProcessorConfigurator.with_gemma_model_path(text_encoder_path),
            model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
            registry=registry,
        )

    def _build_text_encoder(self) -> torch.nn.Module:
        """Build the Gemma text encoder (non-streaming path)."""
        return self._text_encoder_builder.build(device=self._device, dtype=self._dtype).eval()

    def _build_enhancer_text_encoder(self) -> torch.nn.Module:
        return self._enhancer_text_encoder_builder.build(device=self._device, dtype=self._dtype).eval()

    def _build_embeddings_processor(self) -> EmbeddingsProcessor:
        """Build the embeddings processor on the target device."""
        return self._embeddings_processor_builder.build(device=self._device, dtype=self._dtype).eval()

    def _text_encoder_ctx(self) -> AbstractContextManager:
        if self._offload_mode != OffloadMode.NONE:
            return streaming_model(
                self._streaming_text_encoder_builder, self._device, self._dtype, self._alloc_trim_strategy
            )
        return gpu_model(self._build_text_encoder(), alloc_trim_strategy=self._alloc_trim_strategy)

    @torch.inference_mode()
    def __call__(
        self,
        prompts: list[str],
        *,
        enhance_first_prompt: bool = False,
        enhance_prompt_image: str | None = None,
        enhance_prompt_seed: int = 42,
        enhance_static_cache: bool = False,
    ) -> list[EmbeddingsProcessorOutput]:
        """Encode *prompts* through Gemma -> embeddings processor, freeing each model after use."""
        prompts = list(prompts)
        enhance_kwargs = {
            "image_path": enhance_prompt_image,
            "seed": enhance_prompt_seed,
            "static_cache": enhance_static_cache,
        }
        separate_enhancer = self._enhancer_text_encoder_builder is not self._text_encoder_builder

        if enhance_first_prompt and separate_enhancer:
            logger.info(
                "Enhancing with separate Gemma%s",
                f" from {self._prompt_enhancer_gemma_root}" if self._prompt_enhancer_gemma_root else "",
            )
            with gpu_model(
                self._build_enhancer_text_encoder(),
                alloc_trim_strategy=self._alloc_trim_strategy,
            ) as enhancer:
                prompts[0] = generate_enhanced_prompt(enhancer, prompts[0], **enhance_kwargs)
        elif enhance_first_prompt and self._encode_model_type != "gemma3":
            raise ValueError(
                f"Prompt enhancement with encode root model_type={self._encode_model_type!r} "
                "requires --prompt-enhancer-gemma-root pointing at a generative instruct checkpoint "
                "(e.g. gemma3 or gemma4 E2B-it)."
            )

        logger.info("Building text encoder from %s", self._text_encoder_path)
        with self._text_encoder_ctx() as text_encoder:
            if enhance_first_prompt and not separate_enhancer:
                prompts[0] = generate_enhanced_prompt(text_encoder, prompts[0], **enhance_kwargs)
            raw_outputs = text_encoder.encode(prompts)
        logger.info("Text encoder done, building embeddings processor from %s", self._embeddings_paths)

        with gpu_model(
            self._build_embeddings_processor(), alloc_trim_strategy=self._alloc_trim_strategy
        ) as embeddings_processor:
            result = [embeddings_processor.process_hidden_states(hs, mask) for hs, mask in raw_outputs]
        logger.info("Prompt encoding complete")
        return result


_ENCODE_MODEL_TYPES = frozenset({"gemma3", "gemma4", "gemma4_unified"})
