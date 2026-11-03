"""Flexible training strategy for a unified conditioning framework.
This strategy implements the unified conditioning framework that supports:
- Simple fine-tuning with text conditioning (text-to-video/audio)
- Intrinsic conditioning (first_frame, prefix, suffix, spatial_crop, mask)
- Extrinsic conditioning (concatenation-based, IC-LoRA style)
"""

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from accelerate import Accelerator, DistributedType
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from torch import Tensor
from torch.utils.data import DataLoader

from ltx_core.model.transformer.modality import Modality
from ltx_core.text_encoders.gemma import convert_to_additive_mask
from ltx_core.types import SpatioTemporalScaleFactors
from ltx_trainer import logger
from ltx_trainer.datasets import PrecomputedDataset
from ltx_trainer.gpu_utils import free_gpu_memory
from ltx_trainer.model_loader import (
    embedding_weight_paths,
    load_embeddings_processor,
    load_transformer,
    read_video_scale_factors,
    resolve_video_vae_path,
)
from ltx_trainer.preprocessing.types import PreprocessingFingerprint, PreprocessingPipeline
from ltx_trainer.quantization import quantize_model
from ltx_trainer.sigma_tracker import SigmaBucketTracker
from ltx_trainer.timestep_samplers import SAMPLERS, TimestepSampler
from ltx_trainer.training_strategies.base_strategy import TrainingStrategy
from ltx_trainer.training_strategies.flexible.conditioning import (
    ReferenceScaleFactors,
    compute_modality_loss,
    infer_scale_factor,
    infer_temporal_scale_factor,
    process_modality,
)
from ltx_trainer.training_strategies.flexible.config import (
    FlexibleStrategyConfig,
    ReferenceConditionConfig,
)
from ltx_trainer.training_strategies.flexible.preprocessing import build_flexible_preprocessing_pipeline
from ltx_trainer.training_strategies.optimization import create_optimizer, create_scheduler
from ltx_trainer.training_strategies.registry import register_strategy
from ltx_trainer.training_strategies.types import (
    CheckpointPayload,
    MetricValue,
    ModelBundle,
    OptimizationBundle,
    ResumeFingerprint,
    ValidationContext,
    ValidationResult,
)

if TYPE_CHECKING:
    from ltx_trainer.validation.runner import ValidationRunner


@dataclass
class ModelInputs:
    """Container for model inputs using the Modality-based interface."""

    video: Modality | None
    audio: Modality | None

    # Training targets (for loss computation)
    video_targets: Tensor | None
    audio_targets: Tensor | None

    # Masks for loss computation (True = compute loss for this token)
    video_loss_mask: Tensor | None
    audio_loss_mask: Tensor | None


# =============================================================================
# FlexibleStrategy Implementation
# =============================================================================


@register_strategy("flexible", FlexibleStrategyConfig)
class FlexibleStrategy(TrainingStrategy):
    """Unified training strategy supporting all conditioning scenarios.
    This strategy implements the Unified Conditioning Framework, allowing
    any training scenario to be expressed through configuration.
    """

    config: FlexibleStrategyConfig

    def __init__(self, config: FlexibleStrategyConfig) -> None:
        """Initialize strategy from its typed config (which owns model/LoRA/data/algorithm settings)."""
        super().__init__(config)
        self.config = config
        self._sigma_tracker = SigmaBucketTracker()
        sampling = config.timestep_sampling
        self._timestep_sampler: TimestepSampler = SAMPLERS[sampling.timestep_sampling_mode](
            **sampling.timestep_sampling_params
        )

        # Built lazily on first validate() and reused across validation intervals (caching prompt
        # embeddings / conditioning media / decoders is expensive).
        self._validation_runner: "ValidationRunner | None" = None

        # Reference (IC-LoRA) scale factors come from config and are checked against every batch.
        self._reference_scale_factors: ReferenceScaleFactors | None = None

        # Video VAE compression factors. The VAE isn't loaded during training (latents are
        # precomputed), so these are read from the checkpoint's embedded config in build_models to
        # keep RoPE positions consistent with the VAE that produced the latents (and with
        # inference), instead of assuming the default 32x32x8 layout. Defaults until then.
        self._video_scale_factors: SpatioTemporalScaleFactors = SpatioTemporalScaleFactors.default()

    def build_models(self) -> ModelBundle:
        """Load the transformer + embeddings processor and make the transformer trainable.
        Returns the training-ready model graph: the base is loaded, optionally quantized, then made
        trainable -- LoRA adapters attached (lora) or the base unfrozen (full) -- with gradient
        checkpointing configured.
        """

        # Build the validation runner up front (before any training weights are on-device) so the
        # text encoder's transient GPU use during prompt-embedding caching never coexists with the
        # training transformer -- preserving the pre-refactor peak-memory profile and text-encoder
        # lifecycle. Skipped when there is no validation work.
        if self.config.validation.samples:
            self._ensure_validation_runner()

        model_path = self.config.model.model_path
        # Source the video compression factors from the checkpoint's embedded VAE config so the
        # RoPE positions match the VAE that produced the precomputed latents (and inference).
        if self.config.video is not None:
            video_vae_path = resolve_video_vae_path(model_path, self.config.model.video_vae_path)
            self._video_scale_factors = read_video_scale_factors(video_vae_path)
        transformer = load_transformer(
            checkpoint_path=model_path,
            device="cpu",
            dtype=torch.bfloat16,
        )

        init_device = self._initial_module_device()
        embeddings_processor = load_embeddings_processor(
            checkpoint_path=embedding_weight_paths(model_path, self.config.model.text_encoder_path),
            gemma_model_path=self.config.model.text_encoder_path,
            device=init_device,
            dtype=torch.bfloat16,
        )
        embeddings_processor.feature_extractor = None
        # The connectors are an inference-only helper applied during loss computation; they are
        # never trained. Freezing them keeps the trainer's accelerator.prepare() from wrapping the
        # module in DDP (Accelerate only wraps modules with grad-requiring params), which would
        # otherwise break the `.create_embeddings(...)` call on the wrapped object under multi-GPU.
        embeddings_processor.requires_grad_(False)

        training_mode = self.config.training_mode
        transformer_dtype = torch.bfloat16 if training_mode == "lora" else torch.float32
        transformer = transformer.to(dtype=transformer_dtype)

        if self.config.quantization is not None:
            if training_mode == "full":
                raise ValueError("Quantization is not supported in full training mode.")
            transformer = quantize_model(transformer, precision=self.config.quantization)

        transformer.requires_grad_(False)

        # Make the model trainable: attach LoRA adapters (lora) or unfreeze the base (full). This
        # defines the trainable parameter set, so it must run before build_optimizers and before
        # accelerator.prepare(); for QLoRA the adapters attach onto the already-quantized base above.
        if training_mode == "lora":
            transformer = self._setup_lora(transformer)
        elif training_mode == "full":
            transformer.requires_grad_(True)
        else:
            raise ValueError(f"Unknown training mode: {training_mode}")

        # Gradient checkpointing is also pre-prepare model surgery; for PEFT-wrapped models we
        # toggle it on the underlying base model.
        base_model = transformer.get_base_model() if hasattr(transformer, "get_base_model") else transformer
        base_model.set_gradient_checkpointing(self.config.optimizer.gradient_checkpointing)

        # Keep a handle to the (unprepared) bundle so checkpoint model-state loading, which
        # runs before accelerator.prepare(), can reach the transformer. attach_prepared_components
        # later replaces this with the prepared bundle.
        self.models = ModelBundle(
            models={
                "transformer": transformer,
                "embeddings_processor": embeddings_processor,
            }
        )
        return self.models

    def build_optimizers(self, models: ModelBundle, total_steps: int) -> OptimizationBundle:
        """Build the optimizer/scheduler pair over the model's trainable parameters."""
        transformer = models.models["transformer"]
        opt_cfg = self.config.optimizer
        trainable_params = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
        optimizer = create_optimizer(
            trainable_params,
            optimizer_type=opt_cfg.optimizer_type,
            learning_rate=opt_cfg.learning_rate,
        )
        scheduler = create_scheduler(
            optimizer,
            scheduler_type=opt_cfg.scheduler_type,
            total_steps=total_steps,
            scheduler_params=opt_cfg.scheduler_params,
        )
        schedulers = {"main": scheduler} if scheduler is not None else {}
        return OptimizationBundle(optimizers={"main": optimizer}, schedulers=schedulers)

    def resume_fingerprint(self) -> ResumeFingerprint:
        """Resume-critical, diffable config values (learning rate is intentionally excluded)."""
        return ResumeFingerprint(
            fields={
                "optimizer_type": self.config.optimizer.optimizer_type,
                "scheduler_type": self.config.optimizer.scheduler_type,
                "training_mode": self.config.training_mode,
                "lora_rank": self.config.lora.rank if self.config.lora is not None else None,
            }
        )

    def preprocessing_fingerprint(self) -> PreprocessingFingerprint:
        """Describe config values that affect the strategy's precomputed artifacts."""
        data = self.config.data
        model = self.config.model
        return PreprocessingFingerprint(
            artifact_root=data.resolved_output_root().expanduser().resolve(),
            fields={
                "strategy": self.config.name,
                "model_path": str(Path(model.model_path).expanduser().resolve()),
                "text_encoder_path": (
                    str(Path(model.text_encoder_path).expanduser().resolve()) if model.text_encoder_path else None
                ),
                **{
                    name: str(Path(component_path).expanduser().resolve())
                    for name, component_path in (
                        ("video_vae_path", model.video_vae_path),
                        ("audio_vae_path", model.audio_vae_path),
                    )
                    if component_path is not None
                },
                "load_text_encoder_in_8bit": model.load_text_encoder_in_8bit,
                "lora_trigger": data.lora_trigger,
                "resolution_buckets": (
                    [list(bucket) for bucket in data.resolution_buckets] if data.resolution_buckets else None
                ),
                "reference_spatial_scale_factor": data.reference_spatial_scale_factor,
                "reference_temporal_scale_factor": data.reference_temporal_scale_factor,
                "skip_audio": data.skip_audio,
                "audio_durations": data.audio_durations,
                "reshape_mode": data.reshape_mode,
                "vae_tiling": data.vae_tiling,
                "data_sources": dict(sorted(self.config.get_data_sources().items())),
            },
        )

    def load_model_state(self, weights: dict[str, Tensor]) -> None:
        """Apply adapter (or full) weights to the unprepared transformer (before prepare)."""
        transformer = self.models.models["transformer"]
        if self.config.training_mode == "lora":
            state_dict = {k.replace("diffusion_model.", "", 1): v for k, v in weights.items()}
            set_peft_model_state_dict(transformer.get_base_model(), state_dict)
        else:
            transformer.load_state_dict(weights, strict=True)

    def on_before_prepare(self, accelerator: Accelerator) -> None:
        """Distributed-aware model surgery just before ``accelerator.prepare()``.
        FSDP requires a uniform dtype across a wrapped module; in LoRA mode PEFT adds fp32
        adapters onto the bf16 base, so the base is cast to fp32 to match. Also warns about
        the experimental FSDP + quantization combination.
        """
        if accelerator.distributed_type != DistributedType.FSDP:
            return
        if self.config.training_mode == "lora":
            logger.debug("FSDP: casting transformer to FP32 for uniform dtype")
            transformer = self.models.models["transformer"]
            self.models.models["transformer"] = transformer.to(dtype=torch.float32)
        if self.config.quantization is not None:
            logger.warning(
                f"FSDP with quantization ({self.config.quantization}) may have compatibility issues. "
                "Monitor training stability and consider disabling quantization if issues arise."
            )

    def load_training_state(self, training_state: dict[str, Any]) -> None:
        """Restore optimizer/scheduler state into the prepared optimization bundle (after prepare)."""
        if "optimizer" in training_state:
            self.optimization.optimizers["main"].load_state_dict(training_state["optimizer"])
        scheduler = self.optimization.schedulers.get("main")
        if scheduler is not None and "scheduler" in training_state:
            scheduler.load_state_dict(training_state["scheduler"])

    def build_preprocessing_pipeline(self) -> PreprocessingPipeline:
        """Return the preprocessing pipeline declared by this strategy's config."""
        return build_flexible_preprocessing_pipeline(self.config)

    def build_dataloader(self, batch_size: int) -> Iterable:
        """Build the unprepared precomputed-data loader for this strategy."""
        if self.config.data.preprocessed_data_root is None:
            raise ValueError("data.preprocessed_data_root is required for training")
        data_root = Path(self.config.data.preprocessed_data_root)
        for dir_name in self.config.get_data_sources():
            if not (data_root / dir_name).is_dir():
                raise ValueError(
                    f"Required data directory '{dir_name}' does not exist under preprocessed_data_root: "
                    f"{data_root / dir_name}"
                )

        dataset = PrecomputedDataset(
            self.config.data.preprocessed_data_root,
            data_sources=self.config.get_data_sources(),
        )
        num_workers = self.config.data.num_dataloader_workers
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=num_workers > 0,
            persistent_workers=num_workers > 0,
        )

    def on_train_start(self) -> None:
        """Put the trainable transformer in train mode (frozen inference modules stay as-is)."""
        self.models.models["transformer"].train()

    def validate(self, ctx: ValidationContext) -> ValidationResult:
        """Run distributed validation and return the media and scalars to log.
        The shared ``ValidationRunner`` (prompt-embedding cache, conditioning media, decoders) is
        built once -- normally up front in ``build_models`` -- and reused across validation
        intervals. The trainer owns cadence, optimizer offload, and logging; this method owns the
        content + drives the shared distributed runner over the strategy's prepared transformer.
        """

        if not self.config.validation.samples:
            return ValidationResult()

        runner = self._ensure_validation_runner()

        # Validation samples under no_grad; drop grads and free VRAM before generating.
        for optimizer in self.optimization.optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        free_gpu_memory()

        return runner.run(self.models.models["transformer"], ctx)

    def collect_checkpoint(self, include_optimizer: bool) -> CheckpointPayload:
        """Gather the comfy-format adapter (or full) weights + optimizer/scheduler state.
        Collective: ``get_state_dict`` gathers FSDP shards, so this runs on every rank. The
        trainer persists the returned data (and casts the weights to the configured precision).
        """
        accelerator = self.accelerator
        transformer = self.models.models["transformer"]
        is_lora = self.config.training_mode == "lora"
        is_fsdp = accelerator.distributed_type == DistributedType.FSDP

        accelerator.wait_for_everyone()
        full_state_dict = accelerator.get_state_dict(transformer)

        if is_lora:
            unwrapped = accelerator.unwrap_model(transformer, keep_torch_compile=False)
            weights = get_peft_model_state_dict(unwrapped, state_dict=full_state_dict if is_fsdp else None)
            weights = {k.replace("base_model.model.", "", 1): v for k, v in weights.items()}
            weights = {f"diffusion_model.{k}": v for k, v in weights.items()}
        else:
            weights = full_state_dict

        training_state: dict[str, Any] = {}
        if include_optimizer:
            training_state["optimizer"] = self.optimization.optimizers["main"].state_dict()
        scheduler = self.optimization.schedulers.get("main")
        if scheduler is not None:
            training_state["scheduler"] = scheduler.state_dict()

        return CheckpointPayload(
            weights=weights,
            weights_metadata=self._checkpoint_metadata(),
            training_state=training_state,
        )

    def _ensure_validation_runner(self) -> "ValidationRunner":
        """Build (once) and return the cached shared validation runner."""
        # Heavy ltx-core inference deps -- import lazily so plain training does not pay for them.
        from ltx_trainer.validation.runner import ValidationRunner  # noqa: PLC0415

        if self._validation_runner is None:
            data = self.config.data
            validation = self.config.validation.model_copy(
                update={
                    "reference_spatial_scale_factor": data.reference_spatial_scale_factor,
                    "reference_temporal_scale_factor": data.reference_temporal_scale_factor,
                }
            )
            self._validation_runner = ValidationRunner(
                config=validation,
                model_path=self.config.model.model_path,
                text_encoder_path=self.config.model.text_encoder_path,
                video_vae_path=self.config.model.video_vae_path,
                audio_vae_path=self.config.model.audio_vae_path,
                load_text_encoder_in_8bit=self.config.model.load_text_encoder_in_8bit,
            )
        return self._validation_runner

    def _prepare_training_inputs(self, batch: dict[str, Any]) -> ModelInputs:
        """Process the video and audio modalities into model inputs (using the strategy's sampler)."""
        video_result = process_modality(
            self.config.video,
            batch,
            "video",
            self._timestep_sampler,
            self._reference_scale_factors,
            self._video_scale_factors,
        )
        # Only a generated video can provide a shared noise level. A frozen
        # video stays clean while generated audio samples its own timestep.
        shared_sigmas = (
            video_result.modality.sigma
            if self.config.share_sigmas and video_result is not None and video_result.targets is not None
            else None
        )
        audio_result = process_modality(
            self.config.audio,
            batch,
            "audio",
            self._timestep_sampler,
            self._reference_scale_factors,
            self._video_scale_factors,
            sigmas=shared_sigmas,
        )
        return ModelInputs(
            video=video_result.modality if video_result else None,
            audio=audio_result.modality if audio_result else None,
            video_targets=video_result.targets if video_result else None,
            audio_targets=audio_result.targets if audio_result else None,
            video_loss_mask=video_result.loss_mask if video_result else None,
            audio_loss_mask=audio_result.loss_mask if audio_result else None,
        )

    def _compute_loss(
        self,
        batch: object,
        models: ModelBundle,
    ) -> tuple[Tensor, dict[str, float | MetricValue]]:
        """Run the standard flexible flow-matching forward/loss path."""
        if not isinstance(batch, dict):
            raise TypeError(f"FlexibleStrategy expected dict batch, got {type(batch).__name__}")

        self._ensure_reference_scale_factors(batch)
        self._apply_context_connectors(batch["conditions"], models)

        model_inputs = self._prepare_training_inputs(batch)
        transformer = models.models["transformer"]
        video_pred, audio_pred = transformer(
            video=model_inputs.video,
            audio=model_inputs.audio,
            perturbations=None,
        )
        loss = self._combine_modality_losses(video_pred, audio_pred, model_inputs)

        if self.config.video is not None and self.config.video.is_generated and model_inputs.video is not None:
            sigma = model_inputs.video.sigma.detach()
        elif self.config.audio is not None and self.config.audio.is_generated and model_inputs.audio is not None:
            sigma = model_inputs.audio.sigma.detach()
        else:
            raise ValueError("No generated modality available for sigma metrics")

        detached_loss = loss.detach()
        self._sigma_tracker.update(sigma.cpu().tolist(), detached_loss.cpu().tolist())
        # Weighted-mean reduction across ranks: weight by this rank's element count so the logged
        # loss is the true global mean regardless of per-rank batch size. Sigma-bucket metrics stay
        # plain floats (main-rank-only diagnostics).
        metrics: dict[str, float | MetricValue] = {
            "train/loss": MetricValue(
                value=detached_loss.mean().item(),
                weight=float(detached_loss.numel()),
                reduction="mean",
            ),
        }
        metrics.update(self._sigma_tracker.get_metrics())
        return loss, metrics

    def _apply_context_connectors(self, conditions: dict[str, Any], models: ModelBundle) -> None:
        """Run the embeddings-processor connectors over the precomputed text features, in place.
        The new precompute format carries separate video/audio prompt embeds; the legacy format
        only stores a single ``prompt_embeds`` reused for both. Afterwards ``conditions`` holds the
        connector outputs under ``video_prompt_embeds``/``audio_prompt_embeds`` and the processed
        ``prompt_attention_mask``.
        """
        if "video_prompt_embeds" in conditions:
            video_features = conditions["video_prompt_embeds"]
            audio_features = conditions.get("audio_prompt_embeds")
        else:
            video_features = conditions["prompt_embeds"]
            audio_features = conditions["prompt_embeds"]

        additive_mask = convert_to_additive_mask(conditions["prompt_attention_mask"], video_features.dtype)
        embeddings_processor = models.models["embeddings_processor"]
        video_embeds, audio_embeds, attention_mask = embeddings_processor.create_embeddings(
            video_features,
            audio_features,
            additive_mask,
        )
        conditions["video_prompt_embeds"] = video_embeds
        conditions["audio_prompt_embeds"] = audio_embeds
        conditions["prompt_attention_mask"] = attention_mask

    def _combine_modality_losses(
        self,
        video_pred: Tensor | None,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Sum the masked per-modality MSE losses into the scalar training loss. Returns [B,]."""
        total_loss = None

        if video_pred is not None and inputs.video_targets is not None:
            video_loss = compute_modality_loss(
                pred=video_pred,
                targets=inputs.video_targets,
                loss_mask=inputs.video_loss_mask,
            )
            total_loss = video_loss

        if audio_pred is not None and inputs.audio_targets is not None:
            audio_loss = compute_modality_loss(
                pred=audio_pred,
                targets=inputs.audio_targets,
                loss_mask=inputs.audio_loss_mask,
            )
            total_loss = audio_loss if total_loss is None else total_loss + audio_loss

        if total_loss is None:
            raise ValueError("No valid predictions and targets provided for loss computation")

        return total_loss

    def _checkpoint_metadata(self) -> dict[str, str]:
        """Reference scale factors stored as safetensors metadata for inference pipelines."""
        metadata: dict[str, str] = {}
        factors = self._reference_scale_factors
        if factors is None:
            return metadata
        if factors.spatial != 1:
            metadata["reference_spatial_scale_factor"] = str(factors.spatial)
            metadata["reference_downscale_factor"] = str(factors.spatial)  # backward compat
        if factors.temporal != 1:
            metadata["reference_temporal_scale_factor"] = str(factors.temporal)
        return metadata

    @staticmethod
    def _initial_module_device() -> torch.device:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    def _setup_lora(self, transformer: torch.nn.Module) -> torch.nn.Module:
        """Setup the LoRA adapter for the transformer."""
        lora = self.config.lora
        if lora is None:
            raise ValueError("LoRA configuration must be provided when training_mode is 'lora'")
        lora_config = PeftLoraConfig(
            r=lora.rank,
            lora_alpha=lora.alpha,
            target_modules=lora.target_modules,
            lora_dropout=lora.dropout,
            init_lora_weights=True,
        )
        return get_peft_model(transformer, lora_config)

    def _ensure_reference_scale_factors(self, batch: dict[str, Any]) -> None:
        """Resolve reference (IC-LoRA) scale factors from config, verifying every reference batch.
        Reference conditioning concatenates a (possibly lower-resolution) reference clip to the
        target; its patch positions must be rescaled into the target's frame using the
        spatial/temporal ratios between reference and target. Those ratios are declared on
        ``data.reference_*_scale_factor`` and are the single source of truth — preprocessing and
        validation use the same numbers, so training reads them from config rather than
        re-deriving. On every batch we cross-check the declared factors against the actual
        target/reference latent shapes and fail loudly on a mismatch: a disagreement means the
        config no longer matches the ``.precomputed`` tree, which would otherwise silently diverge
        between training and validation. No-op unless the video modality declares a reference
        condition. Verify all references, including those after the first configured condition.
        """
        if self.config.video is None:
            return

        for cond in self.config.video.conditions:
            if not isinstance(cond, ReferenceConditionConfig):
                continue
            data = self.config.data
            declared = ReferenceScaleFactors(
                spatial=data.reference_spatial_scale_factor,
                temporal=data.reference_temporal_scale_factor,
            )
            target, ref = batch["video_latents"], batch[cond.latents_dir]
            target_frames, target_height, target_width = target["latents"].shape[2:]
            ref_frames, ref_height, ref_width = ref["latents"].shape[2:]
            observed_spatial = infer_scale_factor(ref_height, ref_width, target_height, target_width)
            # A single-frame pair has no temporal groups from which to infer the rate.
            # Preprocessing still records the effective reference FPS, so trust the
            # declared factor when positioning image references in a mixed dataset.
            observed_temporal = (
                declared.temporal
                if target_frames == ref_frames == 1
                else infer_temporal_scale_factor(ref_frames, target_frames)
            )
            if (observed_spatial, observed_temporal) != (declared.spatial, declared.temporal):
                raise ValueError(
                    f"Reference scale factors in config (spatial={declared.spatial}, "
                    f"temporal={declared.temporal}) do not match the preprocessed reference latents in "
                    f"'{cond.latents_dir}' (observed spatial={observed_spatial}, temporal={observed_temporal}). "
                    f"Set data.reference_spatial_scale_factor / data.reference_temporal_scale_factor to match "
                    f"the .precomputed tree."
                )
            self._reference_scale_factors = declared
