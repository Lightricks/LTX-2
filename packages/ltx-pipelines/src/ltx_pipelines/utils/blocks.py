"""Pipeline blocks — each block owns its model lifecycle.
Blocks build a model on each ``__call__``, use it, then free GPU memory.
This eliminates manual ``del model; cleanup_memory()`` in pipelines: each
block is self-contained, so no central model-coordinator object is needed.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import replace
from typing import Callable, TypeVar

import torch

from ltx_core.batch_split import BatchSplitAdapter
from ltx_core.block_streaming import DISK_CPU_SLOTS, StreamingModelBuilder
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import Noiser
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.loader import SDOps
from ltx_core.loader.attention_ops import set_attention_module_op
from ltx_core.loader.fuse_loras import bf16_fuse_rule
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import BuilderProtocol, LoraPathStrengthAndSDOps, ModelBuilderProtocol
from ltx_core.loader.registry import DummyRegistry, Registry
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.model.audio_vae import (
    AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    VOCODER_COMFY_KEYS_FILTER,
    AudioDecoderConfigurator,
    AudioEncoderConfigurator,
    VocoderConfigurator,
)
from ltx_core.model.audio_vae import (
    decode_audio as vae_decode_audio,
)
from ltx_core.model.model_protocol import LTXModelProtocol, ModelConfigurator
from ltx_core.model.transformer import (
    LTXV_MODEL_COMFY_RENAMING_MAP,
    LTXModelConfigurator,
    X0Model,
)
from ltx_core.model.transformer.attention import (
    AttentionCallable,
    AttentionFunction,
)
from ltx_core.model.transformer.compiling import (
    CompilationConfig,
    build_compile_transformer_op,
    modify_sd_ops_for_compilation,
)
from ltx_core.model.upsampler import LatentUpsamplerConfigurator, upsample_video
from ltx_core.model.video_vae import (
    MEMORY_EFFICIENT_DECODE,
    VAE_DECODER_COMFY_KEYS_FILTER,
    VAE_ENCODER_COMFY_KEYS_FILTER,
    TilingConfig,
    VideoDecoderConfigurator,
    VideoEncoder,
    VideoEncoderConfigurator,
)
from ltx_core.quantization import QuantizationPolicy, fp8_cast_fuse_rule
from ltx_core.text_encoders.gemma import (
    EMBEDDINGS_PROCESSOR_KEY_OPS,
    GEMMA_LLM_KEY_OPS,
    GEMMA_MODEL_OPS,
    EmbeddingsProcessorConfigurator,
    GemmaTextEncoderConfigurator,
    module_ops_from_gemma_root,
)
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor, EmbeddingsProcessorOutput
from ltx_core.tools import AudioLatentTools, LatentTools, VideoLatentTools
from ltx_core.types import Audio, AudioLatentShape, LatentState, VideoLatentShape, VideoPixelShape
from ltx_core.utils import find_matching_file
from ltx_pipelines.utils.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_pipelines.utils.gpu_model import gpu_model
from ltx_pipelines.utils.helpers import (
    cleanup_memory,
    create_noised_state,
    generate_enhanced_prompt,
)
from ltx_pipelines.utils.samplers import euler_denoising_loop
from ltx_pipelines.utils.types import Denoiser, ModalitySpec, OffloadMode

logger = logging.getLogger(__name__)

T = TypeVar("T")
_M = TypeVar("_M", bound=torch.nn.Module)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _chain_quantization(
    sd_ops: SDOps,
    module_ops: tuple[ModuleOps, ...],
    quantization: QuantizationPolicy,
) -> tuple[SDOps, tuple[ModuleOps, ...]]:
    chained_sd_ops = sd_ops
    if quantization.sd_ops is not None:
        chained_sd_ops = SDOps(
            name=f"sd_ops_chain_{sd_ops.name}+{quantization.sd_ops.name}",
            mapping=(*sd_ops.mapping, *quantization.sd_ops.mapping),
        )
    return chained_sd_ops, (*module_ops, *quantization.module_ops)


def _apply_compile_ops(
    sd_ops: SDOps,
    module_ops: tuple[ModuleOps, ...],
    loras: tuple[LoraPathStrengthAndSDOps, ...],
    number_of_layers: int,
    compilation_config: CompilationConfig,
) -> tuple[SDOps, tuple[ModuleOps, ...], tuple[LoraPathStrengthAndSDOps, ...]]:
    """Rewrite sd_ops/module_ops/LoRAs for compiled blocks (params land under ``_orig_mod``)."""
    sd_ops = modify_sd_ops_for_compilation(sd_ops, number_of_layers)
    compile_op = build_compile_transformer_op(compilation_config)
    module_ops = (*module_ops, compile_op)
    loras = tuple(
        LoraPathStrengthAndSDOps(
            lora.path,
            lora.strength,
            modify_sd_ops_for_compilation(lora.sd_ops, number_of_layers),
        )
        for lora in loras
    )
    return sd_ops, module_ops, loras


@contextmanager
def _streaming_model(
    builder: StreamingModelBuilder,
    target_device: torch.device,
    dtype: torch.dtype,
    alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
) -> Iterator:
    """Build a streaming wrapper, yield it, then tear down and free memory.
    The builder's own ``cpu_slots_count`` selects RAM vs disk streaming.
    ``teardown()`` always runs -- it releases non-memory resources (forward
    hooks, the disk I/O worker thread, open file handles) that GC would not
    reclaim promptly. ``alloc_trim_strategy=DEFER`` only skips the eager allocator
    reclaim (``to("meta")`` + ``cleanup_memory()``), leaving param storage for GC.
    """
    wrapped = builder.build(device=target_device, dtype=dtype)
    try:
        yield wrapped
    finally:
        wrapped.teardown()
        if alloc_trim_strategy == AllocatorTrimStrategy.TRIM:
            wrapped.to("meta")
            cleanup_memory()


def _build_state(
    spec: ModalitySpec,
    tools: LatentTools,
    noiser: Noiser,
    dtype: torch.dtype,
    device: torch.device,
) -> LatentState:
    """Create a noised latent state from a modality spec and tools."""
    state = create_noised_state(
        tools=tools,
        conditionings=spec.conditionings,
        noiser=noiser,
        dtype=dtype,
        device=device,
        noise_scale=spec.noise_scale,
        initial_latent=spec.initial_latent,
    )
    if spec.frozen:
        state = replace(state, denoise_mask=torch.zeros_like(state.denoise_mask))
    return state


def _cleanup_iter(
    it: Iterator[torch.Tensor],
    model: torch.nn.Module,
    alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
) -> Iterator[torch.Tensor]:
    """Wrap an iterator to release *model* memory (per ``alloc_trim_strategy``) once exhausted or abandoned."""
    with gpu_model(model, alloc_trim_strategy=alloc_trim_strategy):
        yield from it


# ---------------------------------------------------------------------------
# DiffusionStage
# ---------------------------------------------------------------------------


class DiffusionStage:
    """Owns transformer lifecycle. Builds on each call, frees on exit.
    Replaces the manual build-transformer / ``del transformer`` pattern that
    every pipeline previously repeated.
    """

    def __init__(
        self,
        transformer_builder: ModelBuilderProtocol[LTXModelProtocol],
        dtype: torch.dtype,
        device: torch.device,
        *,
        quantization: QuantizationPolicy | None = None,
        compilation_config: CompilationConfig | None = None,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
    ) -> None:
        """Construct a stage from a single pre-built transformer ``builder``.
        Holds only that builder plus build-time configuration (dtype, device,
        quantization, compilation). Turning a checkpoint path + LoRA set into a
        builder -- and choosing a :class:`StreamingModelBuilder` when offloading --
        lives in :meth:`from_checkpoint`, which is how pipelines normally create a
        stage. A :class:`StreamingModelBuilder` selects the block-streaming build
        path; any other builder uses the standard (all-on-GPU) path.
        ``quantization`` and ``compilation_config`` are applied lazily on the
        standard path; on the streaming path they are already baked into the
        streaming builder by :meth:`from_checkpoint` and these fields are unused.
        """
        self._transformer_builder = transformer_builder
        self._dtype = dtype
        self._device = device
        self._quantization = quantization
        self._compilation_config = compilation_config
        self._alloc_trim_strategy = alloc_trim_strategy

    @classmethod
    def from_checkpoint(  # noqa: PLR0913
        cls,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        loras: tuple[LoraPathStrengthAndSDOps, ...] = (),
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        compilation_config: CompilationConfig | None = None,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
        offload_mode: OffloadMode = OffloadMode.NONE,
        model_configurator: type[ModelConfigurator] = LTXModelConfigurator,
        model_sd_ops: SDOps = LTXV_MODEL_COMFY_RENAMING_MAP,
    ) -> "DiffusionStage":
        """Build a stage from a checkpoint path and LoRA set.
        Constructs a single transformer builder from ``checkpoint_path`` +
        ``loras`` + ``quantization`` and delegates to ``__init__``. When
        ``offload_mode != OffloadMode.NONE`` that builder is a
        :class:`StreamingModelBuilder` (with quantization/compilation baked in and
        its ``cpu_slots_count`` set for the requested mode); otherwise it is the
        standard single-GPU builder. This is the high-level entry point used by
        pipelines; ``__init__`` itself takes an already-built builder.
        ``model_configurator`` / ``model_sd_ops`` let callers (e.g. the audio-only
        T2A pipeline) override the model class configurator and the state-dict key
        mapping. A quantization policy that pins its own configurator takes
        precedence over ``model_configurator``.
        """
        # A quantization policy may pin its own configurator; otherwise use the one
        # provided by the caller (defaults to the audio-video LTXModelConfigurator).
        configurator = (
            quantization.model_configurator
            if quantization is not None and quantization.model_configurator is not None
            else model_configurator
        )

        transformer_builder: ModelBuilderProtocol[LTXModelProtocol]
        if offload_mode == OffloadMode.NONE:
            transformer_builder = Builder(
                model_path=checkpoint_path,
                model_class_configurator=configurator,
                model_sd_ops=model_sd_ops,
                loras=tuple(loras),
                registry=registry or DummyRegistry(),
            )
        else:
            transformer_builder = cls._build_streaming_builder(
                checkpoint_path=checkpoint_path,
                configurator=configurator,
                model_sd_ops=model_sd_ops,
                loras=tuple(loras),
                quantization=quantization,
                registry=registry or DummyRegistry(),
                offload_mode=offload_mode,
            )

        return cls(
            transformer_builder,
            dtype,
            device,
            quantization=quantization,
            compilation_config=compilation_config,
            alloc_trim_strategy=alloc_trim_strategy,
        )

    @staticmethod
    def _build_streaming_builder(
        *,
        checkpoint_path: str,
        configurator: type[ModelConfigurator],
        model_sd_ops: SDOps,
        loras: tuple[LoraPathStrengthAndSDOps, ...],
        quantization: QuantizationPolicy | None,
        registry: Registry,
        offload_mode: OffloadMode,
    ) -> StreamingModelBuilder:
        """Construct the streaming transformer builder for an offloading stage.
        Holds only raw config (``model_sd_ops`` / ``loras``); compilation and
        quantization are applied at build time by :meth:`_prepared_builder`, exactly
        as on the standard path -- so the builder's LoRA set stays raw and
        :meth:`with_loras` swaps it consistently. ``cpu_slots_count`` is pinned for
        the requested ``offload_mode`` (disk streaming uses a small slot count;
        CPU/RAM streaming pins every block).
        """
        # WeightsProvider currently only supports plain bf16 + fp8_cast LoRA fusion
        # (no companion-key emission). Quantization policies that emit
        # companion keys (e.g. ``.weight_scale``) cannot be streamed yet.
        if quantization is not None and quantization.fuse_rule is not fp8_cast_fuse_rule:
            raise ValueError(
                "Block streaming is not supported with this quantization policy "
                "(only bf16 and fp8_cast are currently supported)."
            )
        return StreamingModelBuilder(
            model_class_configurator=configurator,
            model_path=checkpoint_path,
            model_sd_ops=model_sd_ops,
            loras=loras,
            registry=registry,
            fuse_rule=quantization.fuse_rule if quantization is not None else bf16_fuse_rule,
            blocks_attr="transformer_blocks",
            blocks_prefix="transformer_blocks",
            cpu_slots_count=DISK_CPU_SLOTS if offload_mode == OffloadMode.DISK else None,
        )

    def with_attention(self, attention: AttentionFunction | AttentionCallable | None) -> "DiffusionStage":
        """Return a new ``DiffusionStage`` that pins the transformer build to ``attention``.
        Functional: never mutates ``self``. The returned stage shares all other
        configuration with the original; only the underlying builders' ``module_ops``
        gain a ``set_attention_module_op(attention)`` entry so subsequent transformer
        builds use that kernel. ``attention=None`` is a no-op (returns ``self``).
        """
        if attention is None:
            return self
        op = set_attention_module_op(attention)
        new = copy.copy(self)
        new._transformer_builder = self._transformer_builder.with_module_ops(
            (*self._transformer_builder.module_ops, op),
        )
        return new

    def with_builder(self, builder: ModelBuilderProtocol[LTXModelProtocol]) -> "DiffusionStage":
        """Return a new ``DiffusionStage`` that builds its transformer from ``builder``.
        Functional: never mutates ``self``; shares all other configuration (dtype, device,
        quantization, compilation). Affects the standard (non-offload) build path.
        """
        new = copy.copy(self)
        new._transformer_builder = builder
        return new

    def with_loras(self, loras: tuple[LoraPathStrengthAndSDOps, ...]) -> "DiffusionStage":
        """Return a new ``DiffusionStage`` built with exactly ``loras`` (replacing the current set)."""
        return self.with_builder(self._transformer_builder.with_loras(loras))

    def _prepared_builder(self) -> ModelBuilderProtocol[LTXModelProtocol]:
        """Return the configured builder with the stage's build-time ops applied.
        Compilation and quantization live on the stage (not on the builder) and are
        applied here, lazily, for both the standard and streaming paths. This keeps
        the builder holding only raw sd_ops/module_ops/LoRAs, so ``with_loras`` /
        ``with_builder`` swap them consistently regardless of the build path. The
        returned copy preserves the builder's concrete type (e.g. a
        ``StreamingModelBuilder`` stays one).
        """
        builder = self._transformer_builder
        sd_ops = builder.model_sd_ops
        module_ops = builder.module_ops
        loras = builder.loras
        if self._compilation_config is not None:
            number_of_layers = builder.model_config()["transformer"]["num_layers"]
            sd_ops, module_ops, loras = _apply_compile_ops(
                sd_ops, module_ops, loras, number_of_layers, self._compilation_config
            )
        if self._quantization is not None:
            sd_ops, module_ops = _chain_quantization(sd_ops, module_ops, self._quantization)
            builder = builder.with_fuse_rule(self._quantization.fuse_rule)
        return builder.with_module_ops(module_ops).with_sd_ops(sd_ops).with_loras(loras)

    def _build_transformer(self, *, device: torch.device | None = None, **kwargs: object) -> X0Model:
        target = device or self._device
        return X0Model(self._prepared_builder().build(device=target, **kwargs)).to(target).eval()

    @property
    def _is_streaming(self) -> bool:
        """Whether the configured builder uses the block-streaming build path."""
        return isinstance(self._transformer_builder, StreamingModelBuilder)

    @contextmanager
    def _streaming_transformer_ctx(self) -> Iterator[X0Model]:
        builder = self._prepared_builder()
        assert isinstance(builder, StreamingModelBuilder)
        with _streaming_model(builder, self._device, self._dtype, self._alloc_trim_strategy) as streaming_wrapper:
            yield X0Model(streaming_wrapper).eval()

    def _transformer_ctx(self, **kwargs: object) -> AbstractContextManager:
        if self._is_streaming:
            return self._streaming_transformer_ctx()
        return gpu_model(self._build_transformer(**kwargs), alloc_trim_strategy=self._alloc_trim_strategy)

    def model_context(self, **kwargs: object) -> AbstractContextManager:
        """Build the transformer, yield it, then free its memory on exit.
        Keyword arguments are forwarded to the underlying builder (e.g.
        ``video_tools`` required by ``TiledDataParallelBuilder``).
        """
        return self._transformer_ctx(**kwargs)

    def run(  # noqa: PLR0913
        self,
        transformer: object,
        denoiser: Denoiser,
        sigmas: torch.Tensor,
        noiser: Noiser,
        width: int,
        height: int,
        frames: int,
        fps: float,
        video: ModalitySpec | None = None,
        audio: ModalitySpec | None = None,
        stepper: DiffusionStepProtocol | None = None,
        loop: Callable[..., tuple[LatentState | None, LatentState | None]] | None = None,
        max_batch_size: int = 1,
    ) -> tuple[LatentState | None, LatentState | None]:
        """Run denoising with a pre-built transformer.
        Same semantics as ``__call__`` but accepts a pre-built transformer so
        the model can be shared across multiple calls (e.g. tiled inference
        inside a single ``model_context()`` block). Audio supports
        ``ModalitySpec(frozen=True)`` to keep the latent unchanged throughout
        denoising while still providing cross-modal context to the transformer.
        Returns ``(video_state | None, audio_state | None)`` with cleared
        conditionings and unpatchified latents for present modalities.
        """
        if video is None and audio is None:
            raise ValueError("At least one of `video` or `audio` must be provided")

        if loop is None:
            loop = euler_denoising_loop
        if stepper is None:
            stepper = EulerDiffusionStep()

        pixel_shape = VideoPixelShape(batch=1, frames=frames, height=height, width=width, fps=fps)

        video_state: LatentState | None = None
        video_tools: LatentTools | None = None
        if video is not None:
            v_shape = VideoLatentShape.from_pixel_shape(pixel_shape)
            video_tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), v_shape, fps)
            video_state = _build_state(video, video_tools, noiser, self._dtype, self._device)

        audio_state: LatentState | None = None
        audio_tools: LatentTools | None = None
        if audio is not None:
            a_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
            audio_tools = AudioLatentTools(AudioPatchifier(patch_size=1), a_shape)
            audio_state = _build_state(audio, audio_tools, noiser, self._dtype, self._device)

        wrapped = BatchSplitAdapter(transformer, max_batch_size=max_batch_size)  # type: ignore[arg-type]
        video_state, audio_state = loop(
            sigmas=sigmas,
            video_state=video_state,
            audio_state=audio_state,
            stepper=stepper,
            transformer=wrapped,
            denoiser=denoiser,
        )

        if video_state is not None and video_tools is not None:
            video_state = video_tools.clear_conditioning(video_state)
            video_state = video_tools.unpatchify(video_state)
        if audio_state is not None and audio_tools is not None:
            audio_state = audio_tools.clear_conditioning(audio_state)
            audio_state = audio_tools.unpatchify(audio_state)

        return video_state, audio_state

    def __call__(  # noqa: PLR0913
        self,
        denoiser: Denoiser,
        sigmas: torch.Tensor,
        noiser: Noiser,
        width: int,
        height: int,
        frames: int,
        fps: float,
        video: ModalitySpec | None = None,
        audio: ModalitySpec | None = None,
        stepper: DiffusionStepProtocol | None = None,
        loop: Callable[..., tuple[LatentState | None, LatentState | None]] | None = None,
        max_batch_size: int = 1,
    ) -> tuple[LatentState | None, LatentState | None]:
        """Build transformer -> run denoising loop -> free transformer.
        Returns ``(video_state | None, audio_state | None)`` with cleared
        conditionings and unpatchified latents for present modalities.
        """
        # Build video_tools up front so it can be forwarded to the transformer
        # context (required by TiledDataParallelBuilder in multi-GPU mode).
        # `run()` rebuilds its own tools internally; the duplication is cheap.
        video_tools: LatentTools | None = None
        if video is not None:
            pixel_shape = VideoPixelShape(batch=1, frames=frames, height=height, width=width, fps=fps)
            v_shape = VideoLatentShape.from_pixel_shape(pixel_shape)
            video_tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), v_shape, fps)

        mode = "streaming" if self._is_streaming else "standard"
        logger.info("Building transformer (%s) from %s", mode, self._transformer_builder.checkpoint)
        with self._transformer_ctx(video_tools=video_tools) as transformer:
            logger.info(
                "Running denoising loop (%d steps, %dx%d %d frames @ %.1f fps)",
                len(sigmas) - 1,
                width,
                height,
                frames,
                fps,
            )
            return self.run(
                transformer,
                denoiser,
                sigmas,
                noiser,
                width,
                height,
                frames,
                fps,
                video,
                audio,
                stepper,
                loop,
                max_batch_size,
            )


# ---------------------------------------------------------------------------
# PromptEncoder
# ---------------------------------------------------------------------------


class PromptEncoder:
    """Owns text encoder + embeddings processor lifecycle.
    Loads Gemma, encodes prompts, frees Gemma, then loads the embeddings
    processor to produce final outputs.
    """

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
        text_encoder_builder: BuilderProtocol | None = None,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
    ) -> None:
        self._gemma_root = gemma_root
        self._checkpoint_path = checkpoint_path
        self._dtype = dtype
        self._device = device
        self._offload_mode = offload_mode
        self._alloc_trim_strategy = alloc_trim_strategy

        if text_encoder_builder is not None:
            if offload_mode != OffloadMode.NONE:
                raise ValueError(
                    "text_encoder_builder cannot be used with offload_mode != OffloadMode.NONE "
                    "because no streaming text encoder builder is available."
                )
            self._text_encoder_builder = text_encoder_builder
            self._streaming_text_encoder_builder = None
        else:
            module_ops = module_ops_from_gemma_root(gemma_root)
            model_folder = find_matching_file(gemma_root, "model*.safetensors").parent
            weight_paths = [str(p) for p in model_folder.rglob("*.safetensors")]
            self._text_encoder_builder = Builder(
                model_path=tuple(weight_paths),
                model_class_configurator=GemmaTextEncoderConfigurator,
                model_sd_ops=GEMMA_LLM_KEY_OPS,
                module_ops=(GEMMA_MODEL_OPS, *module_ops),
                registry=registry or DummyRegistry(),
            )
            self._streaming_text_encoder_builder = StreamingModelBuilder(
                model_path=tuple(weight_paths),
                model_class_configurator=GemmaTextEncoderConfigurator,
                model_sd_ops=GEMMA_LLM_KEY_OPS,
                module_ops=(GEMMA_MODEL_OPS, *module_ops),
                registry=registry or DummyRegistry(),
                blocks_attr="model.model.language_model.layers",
                blocks_prefix="model.model.language_model.layers",
                cpu_slots_count=DISK_CPU_SLOTS if offload_mode == OffloadMode.DISK else None,
            )
        self._embeddings_processor_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=EmbeddingsProcessorConfigurator,
            model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
            registry=registry or DummyRegistry(),
        )

    def _build_text_encoder(self) -> torch.nn.Module:
        """Build the Gemma text encoder (non-streaming path)."""
        return self._text_encoder_builder.build(device=self._device, dtype=self._dtype).eval()

    def _build_embeddings_processor(self) -> EmbeddingsProcessor:
        """Build the embeddings processor on the target device."""
        return self._embeddings_processor_builder.build(device=self._device, dtype=self._dtype).eval()

    def _text_encoder_ctx(self) -> AbstractContextManager:
        if self._offload_mode != OffloadMode.NONE:
            return _streaming_model(
                self._streaming_text_encoder_builder, self._device, self._dtype, self._alloc_trim_strategy
            )
        return gpu_model(self._build_text_encoder(), alloc_trim_strategy=self._alloc_trim_strategy)

    def __call__(
        self,
        prompts: list[str],
        *,
        enhance_first_prompt: bool = False,
        enhance_prompt_image: str | None = None,
        enhance_prompt_seed: int = 42,
    ) -> list[EmbeddingsProcessorOutput]:
        """Encode *prompts* through Gemma -> embeddings processor, freeing each model after use."""
        logger.info("Building text encoder from %s", self._gemma_root)
        with self._text_encoder_ctx() as text_encoder:
            if enhance_first_prompt:
                prompts = list(prompts)
                prompts[0] = generate_enhanced_prompt(
                    text_encoder, prompts[0], enhance_prompt_image, seed=enhance_prompt_seed
                )
            raw_outputs = text_encoder.encode(prompts)
        logger.info("Text encoder done, building embeddings processor from %s", self._checkpoint_path)

        with gpu_model(
            self._build_embeddings_processor(), alloc_trim_strategy=self._alloc_trim_strategy
        ) as embeddings_processor:
            result = [embeddings_processor.process_hidden_states(hs, mask) for hs, mask in raw_outputs]
        logger.info("Prompt encoding complete")
        return result


# ---------------------------------------------------------------------------
# ImageConditioner
# ---------------------------------------------------------------------------


class ImageConditioner:
    """Owns video encoder lifecycle.
    Builds the encoder, passes it to the user-supplied callable, then frees it.
    """

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
    ) -> None:
        self._dtype = dtype
        self._device = device
        self._encoder_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=VideoEncoderConfigurator,
            model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
            registry=registry or DummyRegistry(),
        )
        self._alloc_trim_strategy = alloc_trim_strategy

    def _build_encoder(self) -> VideoEncoder:
        return self._encoder_builder.build(device=self._device, dtype=self._dtype).eval()

    def __call__(self, fn: Callable[[VideoEncoder], T]) -> T:
        """Build video encoder → call *fn(encoder)* → free encoder."""
        with gpu_model(self._build_encoder(), alloc_trim_strategy=self._alloc_trim_strategy) as encoder:
            return fn(encoder)


# ---------------------------------------------------------------------------
# VideoUpsampler
# ---------------------------------------------------------------------------


class VideoUpsampler:
    """Owns video encoder + spatial upsampler lifecycle."""

    def __init__(
        self,
        checkpoint_path: str,
        upsampler_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
    ) -> None:
        self._upsampler_path = upsampler_path
        self._dtype = dtype
        self._device = device
        self._encoder_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=VideoEncoderConfigurator,
            model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
            registry=registry or DummyRegistry(),
        )
        self._upsampler_builder = Builder(
            model_path=upsampler_path,
            model_class_configurator=LatentUpsamplerConfigurator,
            registry=registry or DummyRegistry(),
        )
        self._alloc_trim_strategy = alloc_trim_strategy

    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        """Upsample *latent* using video encoder + spatial upsampler, then free both."""
        logger.info("Building video encoder + spatial upsampler from %s", self._upsampler_path)
        with (
            gpu_model(
                self._encoder_builder.build(device=self._device, dtype=self._dtype).eval(),
                alloc_trim_strategy=self._alloc_trim_strategy,
            ) as encoder,
            gpu_model(
                self._upsampler_builder.build(device=self._device, dtype=self._dtype).eval(),
                alloc_trim_strategy=self._alloc_trim_strategy,
            ) as upsampler,
        ):
            return upsample_video(latent=latent, video_encoder=encoder, upsampler=upsampler)


# ---------------------------------------------------------------------------
# VideoDecoder
# ---------------------------------------------------------------------------


class VideoDecoder:
    """Owns video decoder lifecycle.
    Returns an iterator that cleans up the decoder after all chunks are consumed.
    """

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
        memory_efficient: bool = True,
        decoder_builder: BuilderProtocol | None = None,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._dtype = dtype
        self._device = device
        if decoder_builder is not None:
            self._decoder_builder = decoder_builder
        else:
            self._decoder_builder = Builder(
                model_path=checkpoint_path,
                model_class_configurator=VideoDecoderConfigurator,
                model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
                registry=registry or DummyRegistry(),
                module_ops=(MEMORY_EFFICIENT_DECODE,) if memory_efficient else (),
            )
        self._alloc_trim_strategy = alloc_trim_strategy

    def __call__(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
        generator: torch.Generator | None = None,
    ) -> Iterator[torch.Tensor]:
        """Decode *latent* to pixel-space video chunks. Decoder freed after exhaustion."""
        logger.info("Building video decoder from %s", self._checkpoint_path)
        decoder = self._decoder_builder.build(device=self._device, dtype=self._dtype).eval()
        return _cleanup_iter(
            decoder.decode_video(latent, tiling_config, generator),
            decoder,
            alloc_trim_strategy=self._alloc_trim_strategy,
        )


# ---------------------------------------------------------------------------
# AudioDecoder
# ---------------------------------------------------------------------------


class AudioDecoder:
    """Owns audio decoder + vocoder lifecycle."""

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._dtype = dtype
        self._device = device
        self._decoder_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=AudioDecoderConfigurator,
            model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
            registry=registry or DummyRegistry(),
        )
        self._vocoder_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=VocoderConfigurator,
            model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
            registry=registry or DummyRegistry(),
        )
        self._alloc_trim_strategy = alloc_trim_strategy

    def __call__(self, latent: torch.Tensor) -> Audio:
        """Decode audio *latent* through VAE decoder + vocoder, then free both."""
        logger.info("Building audio decoder + vocoder from %s", self._checkpoint_path)
        # The vocoder always runs in fp32 (bf16 accumulation degrades spectral
        # metrics). On CUDA/CPU it is stored in bf16 and autocast upcasts per-op to
        # save memory; MPS has no fp32 autocast, so store it in fp32 directly and
        # avoid the per-call cast. Negligible footprint for this small model.
        vocoder_dtype = torch.float32 if self._device.type == "mps" else self._dtype
        with (
            gpu_model(
                self._decoder_builder.build(device=self._device, dtype=self._dtype).eval(),
                alloc_trim_strategy=self._alloc_trim_strategy,
            ) as decoder,
            gpu_model(
                self._vocoder_builder.build(device=self._device, dtype=vocoder_dtype).eval(),
                alloc_trim_strategy=self._alloc_trim_strategy,
            ) as vocoder,
        ):
            return vae_decode_audio(latent, decoder, vocoder)


# ---------------------------------------------------------------------------
# AudioEncoder
# ---------------------------------------------------------------------------


class AudioConditioner:
    """Owns audio encoder lifecycle.
    Builds the encoder, passes it to the user-supplied callable, then frees it.
    Mirrors :class:`ImageConditioner` for the audio modality.
    """

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
    ) -> None:
        self._dtype = dtype
        self._device = device
        self._alloc_trim_strategy = alloc_trim_strategy
        self._encoder_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=AudioEncoderConfigurator,
            model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
            registry=registry or DummyRegistry(),
        )

    def __call__(self, fn: Callable[[torch.nn.Module], T]) -> T:
        """Build audio encoder → call *fn(encoder)* → free encoder."""
        with gpu_model(
            self._encoder_builder.build(device=self._device, dtype=self._dtype).eval(),
            alloc_trim_strategy=self._alloc_trim_strategy,
        ) as encoder:
            return fn(encoder)
