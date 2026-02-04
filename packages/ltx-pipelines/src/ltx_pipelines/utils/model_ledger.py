from dataclasses import replace

import torch

from ltx_core.loader.fuse_loras import apply_loras
from ltx_core.loader.primitives import LoraPathStrengthAndSDOps, LoraStateDictWithStrength
from ltx_core.loader.registry import DummyRegistry, Registry
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.model.audio_vae import (
    AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    VOCODER_COMFY_KEYS_FILTER,
    AudioDecoder,
    AudioDecoderConfigurator,
    AudioEncoder,
    AudioEncoderConfigurator,
    Vocoder,
    VocoderConfigurator,
)
from ltx_core.model.transformer import (
    LTXV_MODEL_COMFY_RENAMING_MAP,
    LTXV_MODEL_COMFY_RENAMING_WITH_TRANSFORMER_LINEAR_DOWNCAST_MAP,
    UPCAST_DURING_INFERENCE,
    LTXModelConfigurator,
    X0Model,
)
from ltx_core.model.upsampler import LatentUpsampler, LatentUpsamplerConfigurator
from ltx_core.model.video_vae import (
    VAE_DECODER_COMFY_KEYS_FILTER,
    VAE_ENCODER_COMFY_KEYS_FILTER,
    VideoDecoder,
    VideoDecoderConfigurator,
    VideoEncoder,
    VideoEncoderConfigurator,
)
from ltx_core.text_encoders.gemma import (
    AV_GEMMA_TEXT_ENCODER_KEY_OPS,
    AVGemmaTextEncoderModel,
    AVGemmaTextEncoderModelConfigurator,
    module_ops_from_gemma_root,
)
from ltx_core.text_encoders.gemma.encoders.av_encoder import GEMMA_MODEL_OPS
from ltx_core.utils import find_matching_file


class ModelLedger:
    """
    Central coordinator for loading and building models used in an LTX pipeline.
    The ledger wires together multiple model builders (transformer, video VAE encoder/decoder,
    audio VAE decoder, vocoder, text encoder, and optional latent upsampler) and exposes
    factory methods for constructing model instances.
    ### Model Building
    Each model method (e.g. :meth:`transformer`, :meth:`video_decoder`, :meth:`text_encoder`)
    constructs a new model instance on each call. The builder uses the
    :class:`~ltx_core.loader.registry.Registry` to load weights from the checkpoint,
    instantiates the model with the configured ``dtype``, and moves it to ``self.device``.
    .. note::
        Models are **not cached** by default. Each call to a model method creates a new instance.
        Callers are responsible for storing references to models they wish to reuse
        and for freeing GPU memory (e.g. by deleting references and calling
        ``torch.cuda.empty_cache()``).
        Use :meth:`cache_components` to preload and cache all models for faster subsequent calls.
    ### Constructor parameters
    dtype:
        Torch dtype used when constructing all models (e.g. ``torch.bfloat16``).
    device:
        Target device to which models are moved after construction (e.g. ``torch.device("cuda")``).
    checkpoint_path:
        Path to a checkpoint directory or file containing the core model weights
        (transformer, video VAE, audio VAE, text encoder, vocoder). If ``None``, the
        corresponding builders are not created and calling those methods will raise
        a :class:`ValueError`.
    gemma_root_path:
        Base path to Gemma-compatible CLIP/text encoder weights. Required to
        initialize the text encoder builder; if omitted, :meth:`text_encoder` cannot be used.
    spatial_upsampler_path:
        Optional path to a latent upsampler checkpoint. If provided, the
        :meth:`spatial_upsampler` method becomes available; otherwise calling it raises
        a :class:`ValueError`.
    loras:
        Optional collection of LoRA configurations (paths, strengths, and key operations)
        that are applied on top of the base transformer weights when building the model.
    registry:
        Optional :class:`Registry` instance for weight caching across builders.
        Defaults to :class:`DummyRegistry` which performs no cross-builder caching.
    fp8transformer:
        If ``True``, builds the transformer with FP8 quantization and upcasting during inference.
    tokenizer_max_length:
        Optional maximum token length for the text encoder tokenizer. If provided,
        overrides the default tokenizer max_length (useful for longer prompts).
    ### Creating Variants
    Use :meth:`with_loras` to create a new ``ModelLedger`` instance that includes
    additional LoRA configurations while sharing the same registry for weight caching.
    """

    def __init__(
        self,
        dtype: torch.dtype,
        device: torch.device,
        checkpoint_path: str | None = None,
        gemma_root_path: str | None = None,
        spatial_upsampler_path: str | None = None,
        loras: LoraPathStrengthAndSDOps | None = None,
        registry: Registry | None = None,
        fp8transformer: bool = False,
        tokenizer_max_length: int | None = None,
    ):
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.gemma_root_path = gemma_root_path
        self.spatial_upsampler_path = spatial_upsampler_path
        self.loras = loras or ()
        self.registry = registry or DummyRegistry()
        self.fp8transformer = fp8transformer
        self.tokenizer_max_length = tokenizer_max_length
        self._cached_components: dict[str, object] = {}
        self.build_model_builders()

    def build_model_builders(self) -> None:
        if self.checkpoint_path is not None:
            self.transformer_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=LTXModelConfigurator,
                model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
                loras=tuple(self.loras),
                registry=self.registry,
            )

            self.vae_decoder_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=VideoDecoderConfigurator,
                model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
                registry=self.registry,
            )

            self.vae_encoder_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=VideoEncoderConfigurator,
                model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
                registry=self.registry,
            )

            self.audio_decoder_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=AudioDecoderConfigurator,
                model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
                registry=self.registry,
            )

            self.audio_encoder_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=AudioEncoderConfigurator,
                model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
                registry=self.registry,
            )

            self.vocoder_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=VocoderConfigurator,
                model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
                registry=self.registry,
            )

            if self.gemma_root_path is not None:
                module_ops = module_ops_from_gemma_root(self.gemma_root_path)
                model_folder = find_matching_file(self.gemma_root_path, "model*.safetensors").parent
                weight_paths = [str(p) for p in model_folder.rglob("*.safetensors")]

                self.text_encoder_builder = Builder(
                    model_path=(str(self.checkpoint_path), *weight_paths),
                    model_class_configurator=AVGemmaTextEncoderModelConfigurator,
                    model_sd_ops=AV_GEMMA_TEXT_ENCODER_KEY_OPS,
                    registry=self.registry,
                    module_ops=(GEMMA_MODEL_OPS, *module_ops),
                )

        if self.spatial_upsampler_path is not None:
            self.upsampler_builder = Builder(
                model_path=self.spatial_upsampler_path,
                model_class_configurator=LatentUpsamplerConfigurator,
                registry=self.registry,
            )

    def _target_device(self) -> torch.device:
        if isinstance(self.registry, DummyRegistry) or self.registry is None:
            return self.device
        else:
            return torch.device("cpu")

    def with_loras(self, loras: LoraPathStrengthAndSDOps) -> "ModelLedger":
        return ModelLedger(
            dtype=self.dtype,
            device=self.device,
            checkpoint_path=self.checkpoint_path,
            gemma_root_path=self.gemma_root_path,
            spatial_upsampler_path=self.spatial_upsampler_path,
            loras=(*self.loras, *loras),
            registry=self.registry,
            fp8transformer=self.fp8transformer,
            tokenizer_max_length=self.tokenizer_max_length,
        )

    def cache_components(
        self,
        text_encoder: bool = True,
        video_encoder: bool = True,
        transformer: bool = True,
        video_decoder: bool = True,
        audio_decoder: bool = True,
        audio_encoder: bool = True,
        vocoder: bool = True,
        spatial_upsampler: bool = True,
    ) -> None:
        """
        Preload and cache model components for faster subsequent calls.

        After calling this method, the corresponding model methods will return
        cached instances instead of building new ones. This is useful for
        serverless deployments where container startup time matters.

        Args:
            text_encoder: Cache the text encoder if available.
            video_encoder: Cache the video encoder if available.
            transformer: Cache the transformer if available.
            video_decoder: Cache the video decoder if available.
            audio_decoder: Cache the audio decoder if available.
            audio_encoder: Cache the audio encoder if available.
            vocoder: Cache the vocoder if available.
            spatial_upsampler: Cache the spatial upsampler if available.
        """
        if text_encoder and hasattr(self, "text_encoder_builder"):
            self._cached_components["text_encoder"] = self._build_text_encoder()

        if video_encoder and hasattr(self, "vae_encoder_builder"):
            self._cached_components["video_encoder"] = self._build_video_encoder()

        if transformer and hasattr(self, "transformer_builder"):
            self._cached_components["transformer"] = self._build_transformer()

        if video_decoder and hasattr(self, "vae_decoder_builder"):
            self._cached_components["video_decoder"] = self._build_video_decoder()

        if audio_decoder and hasattr(self, "audio_decoder_builder"):
            self._cached_components["audio_decoder"] = self._build_audio_decoder()

        if audio_encoder and hasattr(self, "audio_encoder_builder"):
            self._cached_components["audio_encoder"] = self._build_audio_encoder()

        if vocoder and hasattr(self, "vocoder_builder"):
            self._cached_components["vocoder"] = self._build_vocoder()

        if spatial_upsampler and hasattr(self, "upsampler_builder"):
            self._cached_components["spatial_upsampler"] = self._build_spatial_upsampler()

    def clear_cache(self, component: str | None = None) -> None:
        """
        Clear cached components to free GPU memory.

        Args:
            component: Name of the component to clear. If None, clears all.
        """
        if component is None:
            self._cached_components.clear()
        elif component in self._cached_components:
            del self._cached_components[component]

    def apply_loras_to_cached_transformer(
        self,
        additional_loras: tuple[LoraPathStrengthAndSDOps, ...] = (),
    ) -> None:
        """
        Apply LoRAs to the cached transformer in-place.

        This method allows dynamic LoRA switching without rebuilding the entire
        pipeline. It combines the ledger's built-in loras with additional_loras,
        fuses them with the base model weights, and updates the cached transformer.

        The base model state dict is loaded from the registry (cached), making
        subsequent calls fast. LoRA state dicts are also cached in the registry.

        Args:
            additional_loras: Additional LoRAs to apply on top of built-in ones.
                Pass empty tuple to restore to just the built-in loras.

        Raises:
            ValueError: If transformer is not cached or checkpoint_path not set.

        Example:
            # Setup: cache transformer with built-in loras (e.g., distilled_lora)
            ledger.cache_components(transformer=True)

            # Per-request: apply camera LoRA
            camera_lora = LoraPathStrengthAndSDOps(path, strength, sd_ops)
            ledger.apply_loras_to_cached_transformer((camera_lora,))

            # Later: remove camera LoRA, restore to built-in only
            ledger.apply_loras_to_cached_transformer(())
        """
        if "transformer" not in self._cached_components:
            raise ValueError(
                "Transformer must be cached first via cache_components(transformer=True)"
            )
        if not hasattr(self, "transformer_builder"):
            raise ValueError(
                "Transformer builder not initialized. Provide checkpoint_path to constructor."
            )

        transformer = self._cached_components["transformer"]
        all_loras = (*self.loras, *additional_loras)

        # Get base model state dict from registry (cached after first load)
        model_paths = (
            [self.checkpoint_path]
            if isinstance(self.checkpoint_path, str)
            else list(self.checkpoint_path)
        )
        base_sd = self.transformer_builder.load_sd(
            model_paths,
            sd_ops=self.transformer_builder.model_sd_ops,
            registry=self.registry,
            device=self._target_device(),
        )

        if not all_loras:
            # No LoRAs - restore to base weights
            # X0Model wraps LTXModel via .velocity_model attribute
            transformer.velocity_model.load_state_dict(base_sd.sd, strict=False, assign=True)
            return

        # Load LoRA state dicts (cached in registry)
        lora_state_dicts = [
            self.transformer_builder.load_sd(
                [lora.path],
                sd_ops=lora.sd_ops,
                registry=self.registry,
                device=self._target_device(),
            )
            for lora in all_loras
        ]
        lora_sd_and_strengths = [
            LoraStateDictWithStrength(sd, lora.strength)
            for sd, lora in zip(lora_state_dicts, all_loras, strict=True)
        ]

        # Fuse base weights with LoRAs
        # Note: apply_loras() automatically handles FP8 weights via Triton kernel
        fused_sd = apply_loras(
            model_sd=base_sd,
            lora_sd_and_strengths=lora_sd_and_strengths,
            dtype=None,  # Keep original dtype from base_sd
        )

        # Update transformer weights in-place
        # X0Model wraps LTXModel via .velocity_model attribute
        transformer.velocity_model.load_state_dict(fused_sd.sd, strict=False, assign=True)

    def _build_transformer(self) -> X0Model:
        """Internal method to build a new transformer instance."""
        if self.fp8transformer:
            fp8_builder = replace(
                self.transformer_builder,
                module_ops=(UPCAST_DURING_INFERENCE,),
                model_sd_ops=LTXV_MODEL_COMFY_RENAMING_WITH_TRANSFORMER_LINEAR_DOWNCAST_MAP,
            )
            model = X0Model(fp8_builder.build(device=self._target_device())).to(self.device)
        else:
            model = X0Model(self.transformer_builder.build(device=self._target_device(), dtype=self.dtype)).to(
                self.device
            )
        model.eval()
        return model

    def _build_video_decoder(self) -> VideoDecoder:
        """Internal method to build a new video decoder instance."""
        model = self.vae_decoder_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device)
        model.eval()
        return model

    def _build_video_encoder(self) -> VideoEncoder:
        """Internal method to build a new video encoder instance."""
        model = self.vae_encoder_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device)
        model.eval()
        return model

    def _build_text_encoder(self) -> AVGemmaTextEncoderModel:
        """Internal method to build a new text encoder instance."""
        encoder = self.text_encoder_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device)
        encoder.eval()
        if self.tokenizer_max_length is not None:
            encoder.tokenizer.max_length = self.tokenizer_max_length
        return encoder

    def _build_audio_decoder(self) -> AudioDecoder:
        """Internal method to build a new audio decoder instance."""
        model = self.audio_decoder_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device)
        model.eval()
        return model

    def _build_audio_encoder(self) -> AudioEncoder:
        """Internal method to build a new audio encoder instance."""
        model = self.audio_encoder_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device)
        model.eval()
        return model

    def _build_vocoder(self) -> Vocoder:
        """Internal method to build a new vocoder instance."""
        model = self.vocoder_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device)
        model.eval()
        return model

    def _build_spatial_upsampler(self) -> LatentUpsampler:
        """Internal method to build a new spatial upsampler instance."""
        model = self.upsampler_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device)
        model.eval()
        return model

    def transformer(self) -> X0Model:
        if not hasattr(self, "transformer_builder"):
            raise ValueError(
                "Transformer not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )
        if "transformer" in self._cached_components:
            return self._cached_components["transformer"]
        return self._build_transformer()

    def video_decoder(self) -> VideoDecoder:
        if not hasattr(self, "vae_decoder_builder"):
            raise ValueError(
                "Video decoder not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )
        if "video_decoder" in self._cached_components:
            return self._cached_components["video_decoder"]
        return self._build_video_decoder()

    def video_encoder(self) -> VideoEncoder:
        if not hasattr(self, "vae_encoder_builder"):
            raise ValueError(
                "Video encoder not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )
        if "video_encoder" in self._cached_components:
            return self._cached_components["video_encoder"]
        return self._build_video_encoder()

    def text_encoder(self) -> AVGemmaTextEncoderModel:
        if not hasattr(self, "text_encoder_builder"):
            raise ValueError(
                "Text encoder not initialized. Please provide a checkpoint path and gemma root path to the "
                "ModelLedger constructor."
            )
        if "text_encoder" in self._cached_components:
            return self._cached_components["text_encoder"]
        return self._build_text_encoder()

    def audio_decoder(self) -> AudioDecoder:
        if not hasattr(self, "audio_decoder_builder"):
            raise ValueError(
                "Audio decoder not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )
        if "audio_decoder" in self._cached_components:
            return self._cached_components["audio_decoder"]
        return self._build_audio_decoder()

    def audio_encoder(self) -> AudioEncoder:
        if not hasattr(self, "audio_encoder_builder"):
            raise ValueError(
                "Audio encoder not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )
        if "audio_encoder" in self._cached_components:
            return self._cached_components["audio_encoder"]
        return self._build_audio_encoder()

    def vocoder(self) -> Vocoder:
        if not hasattr(self, "vocoder_builder"):
            raise ValueError(
                "Vocoder not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )
        if "vocoder" in self._cached_components:
            return self._cached_components["vocoder"]
        return self._build_vocoder()

    def spatial_upsampler(self) -> LatentUpsampler:
        if not hasattr(self, "upsampler_builder"):
            raise ValueError("Upsampler not initialized. Please provide upsampler path to the ModelLedger constructor.")
        if "spatial_upsampler" in self._cached_components:
            return self._cached_components["spatial_upsampler"]
        return self._build_spatial_upsampler()
