from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path

import torch

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.batch_split import BatchSplitAdapter
from ltx_core.components.noisers import Noiser
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.conditioning import ConditioningItem, VideoGeneratedKeyframeSlots
from ltx_core.devices import synchronize_device
from ltx_core.loader import SDOps
from ltx_core.loader.attention_ops import set_attention_module_op
from ltx_core.loader.distributed_builders import SequenceParallelBuilder, TiledDataParallelBuilder
from ltx_core.loader.helpers import to_device
from ltx_core.loader.lora_request import LoraPathStrengthAndSDOps
from ltx_core.loader.model_builder import ModelBuilder
from ltx_core.loader.registry import ModelRegistry
from ltx_core.loader.simple_builder import LoRARequest, SimpleBuilder
from ltx_core.loader.streaming_builder import StreamingBuilder, StreamingConfig
from ltx_core.model.model_protocol import ModelConfigurator
from ltx_core.model.transformer import (
    LTXV_MODEL_COMFY_RENAMING_MAP,
    LTXModelConfigurator,
    X0Model,
)
from ltx_core.model.transformer.attention import AttentionCallable, AttentionFunction
from ltx_core.model.transformer.compiling import (
    CompilationConfig,
    compose_compilation,
)
from ltx_core.model.transformer.model_configurator import transformer_checkpoint_selection_ops
from ltx_core.quantization import QuantizationPolicy
from ltx_core.quantization.policy import compose_quantization
from ltx_core.tools import AudioLatentTools, LatentTools, VideoLatentTools
from ltx_core.types import (
    VIDEO_SCALE_FACTORS,
    AudioLatentShape,
    DenoiseMask,
    LatentState,
    SpatioTemporalScaleFactors,
    VideoLatentShape,
)
from ltx_pipelines.utils.gpu_model import gpu_model
from ltx_pipelines.utils.helpers import (
    assert_stage_supports_generated_keyframes,
    cleanup_memory,
    create_noised_state,
)
from ltx_pipelines.utils.samplers import euler_denoising_loop
from ltx_pipelines.utils.streaming_model import streaming_model
from ltx_pipelines.utils.types import Denoiser, ModalitySpec, OffloadMode, VideoAudio

logger = logging.getLogger(__name__)

ModelWrapper = Callable[[torch.nn.Module, LatentTools | None], torch.nn.Module]
"""Wraps a built transformer for one :meth:`DiffusionStage.__call__` (see
:meth:`DiffusionStage.with_model_wrapper`). Must keep the
``(video, audio, perturbations) -> (video, audio)`` forward signature."""

DenoisingLoop = Callable[..., VideoAudio[LatentState]]

__all__ = ["DenoisingLoop", "DiffusionStage", "ModelWrapper", "RecordingDiffusionStage"]


def _unit_lora_request(lora: LoraPathStrengthAndSDOps) -> LoRARequest:
    return LoRARequest((Path(lora.path),), lora.strength, (lora.sd_ops,))


class DiffusionStage:
    """Owns transformer lifecycle. Builds on each call, frees on exit.
    Under ``AllocatorTrimStrategy.DEFER`` a builder whose weights the registry keeps GPU-resident
    (``keeps_gpu_resident_weights``) leaves its build assigned instead: every build reassigns those
    weights, so a dispose would release no memory.
    Stages given one ``active_weights`` dict share one transformer allocation: the first build fills it, and
    later builds, also of copies made with ``with_loras`` and similar, reuse the registry's bound model or rewrite
    only the weights an adapter switch touches. Those builds stay assigned until :meth:`release_transformer`.
    Replaces the manual build-transformer / ``del transformer`` pattern that
    every pipeline previously repeated.
    """

    def __init__(
        self,
        transformer_builder: ModelBuilder,
        dtype: torch.dtype,
        device: torch.device,
        *,
        quantization: QuantizationPolicy | None = None,
        compilation_config: CompilationConfig | None = None,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
        scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS,
        active_weights: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Construct a stage from a single pre-built transformer ``builder``.
        Holds only that builder plus build-time configuration (dtype, device,
        quantization, compilation). Turning a checkpoint path + LoRA set into a
        builder -- and choosing a :class:`StreamingBuilder` when offloading --
        lives in :meth:`from_checkpoint`, which is how pipelines normally create a
        stage. A :class:`StreamingBuilder` streams units; a :class:`SimpleBuilder`
        loads them for an ordinary full-model build.
        ``quantization`` and ``compilation_config`` are applied lazily by
        :meth:`_prepared_builder` on both the standard and streaming paths.
        ``scale_factors`` are the video VAE's spatiotemporal downscaling factors,
        defaulting to the 32x32x8 layout.
        """
        self._transformer_builder = transformer_builder
        self._dtype = dtype
        self._device = device
        self._quantization = quantization
        self._compilation_config = compilation_config
        self._alloc_trim_strategy = alloc_trim_strategy
        self.video_scale_factors = scale_factors
        self._model_wrapper: ModelWrapper | None = None
        self._active_weights = active_weights

    @classmethod
    def from_checkpoint(  # noqa: PLR0913
        cls,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        loras: tuple[LoraPathStrengthAndSDOps, ...] = (),
        quantization: QuantizationPolicy | None = None,
        registry: ModelRegistry | None = None,
        compilation_config: CompilationConfig | None = None,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
        offload_mode: OffloadMode = OffloadMode.NONE,
        model_configurator: type[ModelConfigurator] = LTXModelConfigurator,
        model_sd_ops: SDOps = LTXV_MODEL_COMFY_RENAMING_MAP,
        scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS,
        active_weights: dict[str, torch.Tensor] | None = None,
    ) -> "DiffusionStage":
        """Build a stage from a checkpoint path and LoRA set.
        Constructs a single transformer builder from ``checkpoint_path`` +
        ``loras`` and delegates to ``__init__``. When ``offload_mode != OffloadMode.NONE``
        that builder is a :class:`StreamingBuilder` (with a backing mode);
        otherwise it is a :class:`SimpleBuilder`. Quantization and compilation
        stay on the stage and are applied lazily at build time. This is the
        high-level entry point used by pipelines. ``__init__`` itself takes
        an already-built builder.
        ``model_configurator`` / ``model_sd_ops`` let callers (e.g. the audio-only
        T2A pipeline) override the model class configurator and the state-dict key
        mapping.
        """
        unit_registry = registry or ModelRegistry()
        transformer_builder: ModelBuilder
        if offload_mode == OffloadMode.NONE:
            transformer_builder = SimpleBuilder(
                model_configurator,
                (Path(checkpoint_path),),
                (model_sd_ops,),
                loras=tuple(_unit_lora_request(lora) for lora in loras),
                registry=unit_registry,
            )
        else:
            transformer_builder = cls._build_streaming_builder(
                checkpoint_path=checkpoint_path,
                configurator=model_configurator,
                model_sd_ops=model_sd_ops,
                loras=tuple(loras),
                registry=unit_registry,
                offload_mode=offload_mode,
            )

        return cls(
            transformer_builder,
            dtype,
            device,
            quantization=quantization,
            compilation_config=compilation_config,
            alloc_trim_strategy=alloc_trim_strategy,
            scale_factors=scale_factors,
            active_weights=active_weights,
        )

    @staticmethod
    def _build_streaming_builder(
        *,
        checkpoint_path: str,
        configurator: type[ModelConfigurator],
        model_sd_ops: SDOps,
        loras: tuple[LoraPathStrengthAndSDOps, ...],
        registry: ModelRegistry,
        offload_mode: OffloadMode,
    ) -> StreamingBuilder:
        """Construct the streaming transformer builder for an offloading stage.
        Returns a :class:`StreamingBuilder` with the checkpoint path, LoRAs,
        and backing mode selected by ``offload_mode``.
        Compilation / quantization sd_ops and module_ops are not applied here --
        the stage applies them lazily via :meth:`_prepared_builder`.
        """
        base = SimpleBuilder(
            configurator,
            (Path(checkpoint_path),),
            (model_sd_ops,),
            loras=tuple(_unit_lora_request(lora) for lora in loras),
            registry=registry,
        )
        return StreamingBuilder(
            base,
            StreamingConfig("transformer_blocks", "disk" if offload_mode == OffloadMode.DISK else "cpu"),
        )

    @property
    def supports_generated_keyframes(self) -> bool:
        """Whether this stage's checkpoint was trained with the keyframe absolute-position embedding.
        Read from the checkpoint's transformer config, so it is answered before any weights are
        built. Generated keyframe slots on a checkpoint without it would be denoised as unmarked
        tokens while still costing a full latent frame of tokens each, so pipelines refuse rather
        than silently degrade.
        """
        transformer_config = self._model_config().get("transformer", {})
        return bool(transformer_config.get("use_keyframes_abs_pos_embedding", False))

    def _model_config(self) -> dict:
        builder = self._transformer_builder
        if isinstance(builder, (SequenceParallelBuilder, TiledDataParallelBuilder)):
            return builder.base.model_config()
        return builder.model_config()

    def _assert_supports_conditionings(self, conditionings: Sequence[ConditioningItem] | None) -> None:
        if conditionings is None or self.supports_generated_keyframes:
            return
        if any(isinstance(item, VideoGeneratedKeyframeSlots) for item in conditionings):
            assert_stage_supports_generated_keyframes(self)

    @property
    def dtype(self) -> torch.dtype:
        """Dtype this stage builds its transformer and sizes its latents in."""
        return self._dtype

    @property
    def device(self) -> torch.device:
        """Device this stage builds its transformer and sizes its latents on."""
        return self._device

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
        builder = self._transformer_builder
        new._transformer_builder = (
            replace(builder, base=builder.base.with_module_ops((*builder.base.module_ops, op)))
            if isinstance(builder, (StreamingBuilder, SequenceParallelBuilder, TiledDataParallelBuilder))
            else builder.with_module_ops((*builder.module_ops, op))
        )
        return new

    def with_builder(self, builder: ModelBuilder) -> "DiffusionStage":
        """Return a new ``DiffusionStage`` that builds its transformer from ``builder``.
        Functional: never mutates ``self``; shares all other configuration
        (dtype, device, quantization, compilation). The given builder determines
        the build path on subsequent calls.
        """
        new = copy.copy(self)
        new._transformer_builder = builder
        return new

    def with_model_wrapper(self, wrapper: "ModelWrapper | None") -> "DiffusionStage":
        """Return a new stage that passes each built transformer through ``wrapper``.
        Functional: never mutates ``self``. The wrapper is applied inside the transformer
        context -- after the model is built, before the denoising loop -- on both the standard
        and block-streaming paths, and it receives the call's video tools. Wrapping *here*
        rather than around the builder is what lets a wrapper compose with offloading: a
        wrapping builder would hide a :class:`StreamingBuilder` from :attr:`_is_streaming`,
        so the stage would take the plain ``gpu_model`` path and never tear the stream down.
        ``wrapper=None`` clears any previously set wrapper.
        """
        new = copy.copy(self)
        new._model_wrapper = wrapper
        return new

    @property
    def quantization(self) -> QuantizationPolicy | None:
        """The quantization policy applied to every transformer build."""
        return self._quantization

    def with_quantization(self, quantization: QuantizationPolicy | None) -> "DiffusionStage":
        """Return a new ``DiffusionStage`` whose builds apply ``quantization`` instead.
        Functional: never mutates ``self``. A policy with the same ``sd_ops`` loads the same
        weights, so the two stages share the registry's tensors. Like a copy with another builder,
        the copy is outside any :meth:`resident` block of ``self``: its builds apply its own policy.
        """
        new = copy.copy(self)
        new._quantization = quantization
        return new

    def with_loras(self, loras: tuple[LoraPathStrengthAndSDOps, ...]) -> "DiffusionStage":
        """Return a new ``DiffusionStage`` built with exactly ``loras`` (replacing the current set)."""
        builder = self._transformer_builder
        if isinstance(builder, (StreamingBuilder, SequenceParallelBuilder, TiledDataParallelBuilder)):
            requests = tuple(_unit_lora_request(lora) for lora in loras)
            return self.with_builder(replace(builder, base=builder.base.with_loras(requests)))
        return self.with_builder(builder.with_loras(tuple(_unit_lora_request(lora) for lora in loras)))

    def _prepared_builder(self) -> ModelBuilder:
        """Return the configured builder with the stage's build-time ops applied.
        Compilation and quantization live on the stage (not on the builder) and are
        applied here, lazily, for both the standard and streaming paths. This keeps
        the builder holding only raw sd_ops/module_ops/LoRAs, so ``with_loras`` /
        ``with_builder`` swap them consistently regardless of the build path. The
        returned copy preserves the builder's concrete type (e.g. a
        ``StreamingBuilder`` stays one).
        """
        builder = self._transformer_builder
        _ensure_cudagraph_compatible_builder(builder, self._compilation_config)
        prepared = builder if isinstance(builder, SimpleBuilder) else builder.base
        if self._quantization is not None:
            prepared = compose_quantization(prepared, self._quantization)
        if self._compilation_config is not None:
            number_of_layers = self._model_config()["transformer"]["num_layers"]
            prepared = compose_compilation(prepared, self._compilation_config, number_of_layers)
        return prepared if isinstance(builder, SimpleBuilder) else replace(builder, base=prepared)

    def _build_transformer(self, *, device: torch.device | None = None, **kwargs: object) -> X0Model:
        target = device or self._device
        builder = self._prepared_builder()
        if isinstance(builder, (SimpleBuilder, SequenceParallelBuilder, TiledDataParallelBuilder)):
            selection = transformer_checkpoint_selection_ops(self._model_config())
            if selection is not None:
                if isinstance(builder, SimpleBuilder):
                    builder = builder.with_sd_ops((*builder.sd_ops, selection))
                else:
                    builder = replace(builder, base=builder.base.with_sd_ops((*builder.base.sd_ops, selection)))
        if isinstance(builder, SimpleBuilder):
            kwargs.pop("video_tools", None)
            if self._shares_weights:
                kwargs["dst"] = self._active_weights
        return to_device(X0Model(builder.build(device=target, **kwargs)), target).eval()

    @property
    def _shares_weights(self) -> bool:
        """Whether builds fill and reuse ``active_weights``: only a plain builder loads into it."""
        return self._active_weights is not None and isinstance(self._transformer_builder, SimpleBuilder)

    @property
    def _is_streaming(self) -> bool:
        """Whether the configured builder uses unit streaming."""
        return isinstance(self._transformer_builder, StreamingBuilder)

    @contextmanager
    def _streaming_transformer_ctx(self) -> Iterator[X0Model]:
        builder = self._prepared_builder()
        assert isinstance(builder, StreamingBuilder)
        selection = transformer_checkpoint_selection_ops(builder.model_config())
        if selection is not None:
            builder = replace(builder, base=builder.base.with_sd_ops((*builder.base.sd_ops, selection)))
        with streaming_model(builder, self._device, self._dtype, self._alloc_trim_strategy) as streaming_wrapper:
            yield X0Model(streaming_wrapper).eval()

    def _transformer_ctx(self, **kwargs: object) -> AbstractContextManager:
        if self._is_streaming:
            return self._streaming_transformer_ctx()
        return self._transformer_lifetime(self._build_transformer(**kwargs))

    def _transformer_lifetime(self, transformer: X0Model) -> AbstractContextManager:
        if self._shares_weights:
            # The shared allocation outlives the call; release_transformer frees it.
            return nullcontext(transformer)
        if self._transformer_builder.keeps_gpu_resident_weights and (
            self._alloc_trim_strategy == AllocatorTrimStrategy.DEFER
        ):
            # The registry keeps these weights resident and every build reassigns them, so a
            # dispose would release no memory.
            return nullcontext(transformer)
        return gpu_model(transformer, alloc_trim_strategy=self._alloc_trim_strategy)

    def release_transformer(self) -> None:
        """Free the transformer weights shared through ``active_weights``, and every model bound to them."""
        if not self._shares_weights:
            return
        assert self._active_weights is not None
        if self._alloc_trim_strategy == AllocatorTrimStrategy.TRIM:
            synchronize_device()
        builder = self._transformer_builder
        assert isinstance(builder, SimpleBuilder)
        builder.registry.release_active_models(self._active_weights)
        if self._alloc_trim_strategy == AllocatorTrimStrategy.TRIM:
            cleanup_memory()

    def __call__(
        self,
        denoiser: Denoiser,
        sigmas: torch.Tensor,
        noiser: Noiser,
        modalities: VideoAudio[ModalitySpec],
        loop: DenoisingLoop | None = None,
        max_batch_size: int = 1,
    ) -> VideoAudio[LatentState]:
        """Build transformer -> run denoising loop -> free transformer (see the class docstring for DEFER).
        Returns a :class:`VideoAudio` of unpatchified :class:`LatentState` with cleared
        conditionings for present modalities.
        """
        video, audio = modalities
        if video is None and audio is None:
            raise ValueError("At least one of `modalities.video` or `modalities.audio` must be provided")
        if video is not None:
            self._assert_supports_conditionings(video.conditionings)

        if loop is None:
            loop = euler_denoising_loop

        video_latent = video.latent if video is not None else None
        audio_latent = audio.latent if audio is not None else None
        conditioning_fps = video.conditioning_fps if video is not None else None

        # Build video_tools up front so it can be forwarded to the transformer
        # context (required by TiledDataParallelBuilder in multi-GPU mode).
        video_tools = (
            VideoLatentTools(
                VideoLatentPatchifier(patch_size=1),
                VideoLatentShape.from_torch_shape(video_latent.shape),
                conditioning_fps,
                scale_factors=self.video_scale_factors,
            )
            if video_latent is not None
            else None
        )

        mode = "streaming" if self._is_streaming else "standard"
        builder = self._transformer_builder
        checkpoint = builder.paths if isinstance(builder, SimpleBuilder) else builder.base.paths
        logger.info("Building transformer (%s) from %s", mode, checkpoint)
        with self._transformer_ctx(video_tools=video_tools) as built:
            transformer = built if self._model_wrapper is None else self._model_wrapper(built, video_tools)
            logger.info(
                "Running denoising loop (%d steps, video latent %s, audio latent %s @ %.1f fps)",
                len(sigmas) - 1,
                tuple(video_latent.shape) if video_latent is not None else None,
                tuple(audio_latent.shape) if audio_latent is not None else None,
                conditioning_fps,
            )
            video_state: LatentState | None = None
            if video is not None and video_tools is not None:
                video_state = _build_state(video, video_tools, noiser, self._dtype, self._device)

            audio_tools = (
                AudioLatentTools(
                    AudioPatchifier(patch_size=1),
                    AudioLatentShape.from_torch_shape(audio_latent.shape),
                )
                if audio_latent is not None
                else None
            )
            audio_state: LatentState | None = None
            if audio is not None and audio_tools is not None:
                audio_state = _build_state(audio, audio_tools, noiser, self._dtype, self._device)

            wrapped = BatchSplitAdapter(transformer, max_batch_size=max_batch_size)  # type: ignore[arg-type]
            video_state, audio_state = loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                transformer=wrapped,
                denoiser=denoiser,
                v_context=video.context if video is not None else None,
                a_context=audio.context if audio is not None else None,
            )

            if video_state is not None and video_tools is not None:
                video_state = video_tools.clear_conditioning(video_state)
                video_state = video_tools.unpatchify(video_state)
            if audio_state is not None and audio_tools is not None:
                audio_state = audio_tools.clear_conditioning(audio_state)
                audio_state = audio_tools.unpatchify(audio_state)

            return VideoAudio(video_state, audio_state)


class RecordingDiffusionStage:
    """Wraps a :class:`DiffusionStage` and records the states each call produced.
    Diagnostics wrapper for a stage's intermediate ``LatentState``s. Use this to inspect
    per-stage states (including half-res stage-1 slots) without a lifecycle callback. Pipelines
    hold their stage in a plain attribute, so a caller substitutes this in::
        pipeline.stage = RecordingDiffusionStage(pipeline.stage)
        result = pipeline(...)
        keyframes = pipeline.stage.generated_keyframes
    Delegates every other attribute to the wrapped stage, so a substituted stage stays usable
    wherever the original was. Recording holds references to the returned states, which keeps
    their latents alive -- fine for diagnostics on one run, not something to leave enabled in a
    long-lived serving process.
    """

    def __init__(self, stage: DiffusionStage) -> None:
        self._stage = stage
        self.video_states: list[LatentState] = []
        self.audio_states: list[LatentState | None] = []

    def __call__(self, *args: object, **kwargs: object) -> VideoAudio[LatentState]:
        video_state, audio_state = self._stage(*args, **kwargs)  # type: ignore[arg-type]
        if video_state is not None:
            self.video_states.append(video_state)
            self.audio_states.append(audio_state)
        return VideoAudio(video_state, audio_state)

    @property
    def generated_keyframes(self) -> torch.Tensor | None:
        """Generated keyframe latents from the first recorded call, if it produced any."""
        for state in self.video_states:
            if state.generated_keyframes is not None:
                return state.generated_keyframes
        return None

    def __getattr__(self, name: str) -> object:
        return getattr(self._stage, name)


def _ensure_cudagraph_compatible_builder(
    builder: ModelBuilder,
    compilation_config: CompilationConfig | None,
) -> None:
    """CUDA-graph compile paths need GPU-resident weights across dispose/rebuild."""
    if compilation_config is None:
        return
    cudagraph_modes = frozenset({"reduce-overhead", "max-autotune"})
    needs_resident = compilation_config.capture or compilation_config.mode in cudagraph_modes
    if not needs_resident or isinstance(builder, StreamingBuilder) or builder.keeps_gpu_resident_weights:
        return
    raise ValueError(
        "CompilationConfig capture / mode=reduce-overhead|max-autotune requires a builder "
        "with keeps_gpu_resident_weights=True (SequenceParallelBuilder / "
        "TiledDataParallelBuilder, or StreamingBuilder). "
        "The default SimpleBuilder does not promise stable GPU weight storage across stages, "
        "which invalidates CUDA graphs (or repeatedly recaptures until Dynamo gives up)."
    )


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
        initial_latent=spec.latent,
    )
    if spec.frozen:
        state = replace(
            state,
            denoise_mask=DenoiseMask(torch.zeros(state.denoise_mask.tensor.shape)).compact(device),
            frozen=True,
        )
    return state
