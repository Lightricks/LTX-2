import contextlib
import os
import time
import warnings
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Callable

import torch
import yaml
from accelerate import Accelerator, DistributedType
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from pydantic import BaseModel
from torch import Tensor

from ltx_trainer import logger
from ltx_trainer.checkpoint_manager import CheckpointManager, ResumePlan
from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.gpu_utils import get_gpu_memory_gb
from ltx_trainer.preprocessing.manifest import assert_preprocess_manifest_matches
from ltx_trainer.progress import TrainingProgress
from ltx_trainer.training_strategies import get_training_strategy
from ltx_trainer.training_strategies.metrics import reduce_step_metrics
from ltx_trainer.training_strategies.types import (
    MetricValue,
    ModelBundle,
    OptimizationBundle,
    StepContext,
    StepResult,
    ValidationContext,
    ValidationResult,
)
from ltx_trainer.wandb_logger import WandbRun

# Disable irrelevant warnings from transformers
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Silence bitsandbytes warnings about casting
warnings.filterwarnings(
    "ignore", message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization"
)

# Disable progress bars if not main process
IS_MAIN_PROCESS = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0
if not IS_MAIN_PROCESS:
    from transformers.utils.logging import disable_progress_bar

    disable_progress_bar()

StepCallback = Callable[[int, int, "ValidationResult | None"], None]  # (step, total, last_validation) -> None

MEMORY_CHECK_INTERVAL = 200

# Runaway guard: cap micro-steps at N times the expected count in case a strategy never reports completed_step.
_MICRO_STEP_SAFETY_FACTOR = 8


class TrainingStats(BaseModel):
    """Statistics collected during training"""

    total_time_seconds: float
    steps_per_second: float
    samples_per_second: float
    peak_gpu_memory_gb: float
    global_batch_size: int
    num_processes: int


class LtxvTrainer:
    def __init__(self, trainer_config: LtxTrainerConfig) -> None:
        self._config = trainer_config
        self._training_strategy = get_training_strategy(self._config.training_strategy)
        self._checkpoints = CheckpointManager(
            self._config.output_dir,
            precision=self._config.checkpoints.precision,
            keep_last_n=self._config.checkpoints.keep_last_n,
        )
        self._wandb = WandbRun(
            self._config.wandb,
            run_name=Path(self._config.output_dir).name,
            config_dump=self._config.model_dump(),
            is_main_process=IS_MAIN_PROCESS,
        )
        self._started = False

        # Built in _build_and_prepare(); None until then.
        self._accelerator: Accelerator
        self._models: ModelBundle
        self._optimization: OptimizationBundle
        self._resume: ResumePlan

        # Loop state.
        self._global_step = -1
        self._last_checkpoint_step: int | None = None
        self._fsdp_optimizer_warned = False

        # Run-scoped state (set when train() starts; see train()/_setup_for_training).
        self._initial_step = 0
        self._remaining_steps = 0
        self._train_start_time = 0.0
        self._start_mem = 0.0
        self._disable_progress_bars = False
        self._step_callback: StepCallback | None = None

    # --- Training entry point ---
    def train(
        self,
        disable_progress_bars: bool = False,
        step_callback: StepCallback | None = None,
    ) -> tuple[Path, TrainingStats]:
        """Run training to completion and return ``(saved_checkpoint_path, stats)``.
        Args:
            disable_progress_bars: Disable Rich progress bars (useful for multi-process runs).
            step_callback: Optional callback invoked after each optimization step.
        """
        if self._started:
            raise RuntimeError("LtxvTrainer.train() is single-use; construct a new LtxvTrainer to run again.")
        self._started = True
        self._disable_progress_bars = disable_progress_bars
        self._step_callback = step_callback

        if IS_MAIN_PROCESS:
            self._config.print_config()

        # Check preprocessed data is compatible with the current config.
        fingerprint = self._training_strategy.preprocessing_fingerprint()
        if fingerprint is not None:
            assert_preprocess_manifest_matches(
                fingerprint.artifact_root,
                fingerprint.fields,
                allow_stale=self._config.allow_stale_preprocessed,
            )

        # Build/prepare before the try: a failure here propagates without _teardown() running on a
        # half-built accelerator (matches construction-failure semantics).
        self._build_and_prepare()

        self._start_mem = get_gpu_memory_gb(self._accelerator.device)
        self._train_start_time = time.time()

        try:
            progress, dataloader = self._setup_for_training()
            with progress:
                peak_mem = self._run_training_loop(progress, dataloader)
            return self._finalize_training(peak_mem)
        finally:
            self._teardown()

    # --- Setup and initialization ---
    def _build_and_prepare(self) -> None:
        """Build models/optimizers, start Accelerate, resolve resume + load weights, and wrap for
        distributed training. Deferred out of __init__ so constructing a trainer stays cheap and
        free of GPU, distributed, and disk side effects.
        """
        self._models = self._training_strategy.build_models()
        self._setup_accelerator()
        self._optimization = self._training_strategy.build_optimizers(
            self._models, total_steps=self._config.optimization.steps
        )

        # Resolve resume intent and load model weights BEFORE accelerator.prepare(): adapter
        # weights can only be applied once LoRA modules exist (build_models) and before
        # the model is FSDP/DDP-wrapped. Optimizer/scheduler/RNG resume happens after prepare.
        self._resume = self._checkpoints.resolve(
            self._config.checkpoints.load_checkpoint,
            no_resume=self._config.checkpoints.no_resume,
            expected_fingerprint=self._training_strategy.resume_fingerprint().fields,
        )
        if self._resume.weights_source is not None:
            self._training_strategy.load_model_state(self._checkpoints.load_weights(self._resume.weights_source))

        self._prepare_models_for_training()

    def _setup_accelerator(self) -> None:
        """Initialize the Accelerator with the appropriate settings."""

        # find_unused_parameters=True keeps DDP happy when LoRA targets a branch the forward
        # pass skips (e.g. audio LoRA with `with_audio: false`, or short module patterns like
        # "to_k" that match the audio branch unintentionally). It's a no-op for FSDP and
        # single-GPU runs. The probing cost is paid only on the first step.
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        # All distributed setup (DDP/FSDP, number of processes, etc.) is controlled by
        # the user's Accelerate configuration (accelerate config / accelerate launch).
        self._accelerator = Accelerator(
            mixed_precision=self._config.optimization.mixed_precision_mode,
            gradient_accumulation_steps=self._config.optimization.gradient_accumulation_steps,
            kwargs_handlers=[ddp_kwargs],
        )

        if self._accelerator.num_processes > 1:
            logger.info(
                f"{self._accelerator.distributed_type.value} distributed training enabled "
                f"with {self._accelerator.num_processes} processes"
            )

            local_batch = self._config.optimization.batch_size
            accumulation_steps = self._config.optimization.gradient_accumulation_steps
            effective_global_batch = local_batch * accumulation_steps * self._accelerator.num_processes
            logger.info(
                f"Per-rank micro-batch size: {local_batch}, gradient accumulation: {accumulation_steps}, "
                f"effective global batch size: {effective_global_batch}"
            )

        # Log torch.compile status from Accelerate's dynamo plugin
        is_compile_enabled = (
            hasattr(self._accelerator.state, "dynamo_plugin") and self._accelerator.state.dynamo_plugin.backend != "NO"
        )
        if is_compile_enabled:
            plugin = self._accelerator.state.dynamo_plugin
            logger.info(f"🔥 torch.compile enabled via Accelerate: backend={plugin.backend}, mode={plugin.mode}")

            if self._accelerator.distributed_type == DistributedType.FSDP:
                logger.warning(
                    "⚠️ FSDP + torch.compile is experimental and may hang on the first training iteration. "
                    "If this occurs, disable torch.compile by removing dynamo_config from your Accelerate config."
                )

    def _prepare_models_for_training(self) -> None:
        """Prepare models for training with Accelerate."""

        # Let the strategy do any distributed-aware model surgery (e.g. FSDP dtype cast) first.
        self._training_strategy.on_before_prepare(self._accelerator)

        model_names = list(self._models.models)
        prepared_models = self._prepare_tuple([self._models.models[name] for name in model_names])
        self._models = ModelBundle(models=dict(zip(model_names, prepared_models, strict=True)))

        # Log GPU memory usage after model preparation
        vram_usage_gb = torch.cuda.memory_allocated() / 1024**3
        logger.debug(f"GPU memory usage after models preparation: {vram_usage_gb:.2f} GB")

    def _prepare_tuple(self, modules: list[Any]) -> tuple:
        """Run ``accelerator.prepare`` and always return a tuple (it returns a bare value for one arg)."""
        prepared = self._accelerator.prepare(*modules)
        return (prepared,) if len(modules) == 1 else prepared

    def _setup_for_training(self) -> tuple[TrainingProgress, Iterable]:
        """Arm the run and return the two live objects the loop drives: progress bar + dataloader.
        Prepares optimizers, restores resume state, initializes W&B, builds the dataloader, saves
        the config, and sets the run-scoped step counters (``_initial_step``/``_remaining_steps``/
        ``_global_step``).
        """
        cfg = self._config
        set_seed(cfg.seed)
        logger.debug(f"Process {self._accelerator.process_index} using seed: {cfg.seed}")

        self._prepare_optimization()
        # Prepared components must be attached before resume restore (the strategy restores
        # optimizer/scheduler state into the prepared bundle).
        self._training_strategy.attach_prepared_components(
            models=self._models,
            optimization=self._optimization,
            accelerator=self._accelerator,
        )

        self._initial_step, resuming = self._restore_for_resume()

        # Init W&B after restore so we only resume the run when state restore succeeded.
        self._wandb.start(resume_run_id=self._resume.wandb_run_id if resuming else None)

        dataloader = self._accelerator.prepare(
            self._training_strategy.build_dataloader(batch_size=cfg.optimization.batch_size)
        )
        self._accelerator.wait_for_everyone()

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        self._save_config()

        self._remaining_steps = cfg.optimization.steps - self._initial_step
        if self._remaining_steps <= 0:
            raise ValueError(
                f"No remaining training steps: initial_step={self._initial_step} >= "
                f"target_steps={cfg.optimization.steps}. Nothing to train."
            )

        if resuming:
            logger.info(f"🚀 Resuming training from step {self._initial_step} → {cfg.optimization.steps}")
        else:
            logger.info("🚀 Starting training...")

        if IS_MAIN_PROCESS and self._disable_progress_bars:
            logger.warning("Progress bars disabled. Intermediate status messages will be logged instead.")
        progress = TrainingProgress(
            enabled=IS_MAIN_PROCESS and not self._disable_progress_bars, total_steps=self._remaining_steps
        )

        self._training_strategy.on_train_start()
        self._global_step = self._initial_step
        return progress, dataloader

    def _restore_for_resume(self) -> tuple[int, bool]:
        """Restore optimizer/scheduler/RNG when resuming; return ``(initial_step, resuming)``.
        Returns ``(0, False)`` when there is nothing to resume or restore fails.
        """
        if self._resume.resume_dir is None:
            return 0, False
        if self._restore_resume_state(self._resume.resume_dir):
            return self._resume.initial_step, True
        return 0, False

    def _prepare_optimization(self) -> None:
        """Prepare named optimizers and schedulers with Accelerator, preserving their names."""
        optimizer_names = list(self._optimization.optimizers)
        scheduler_names = list(self._optimization.schedulers)
        components = [
            *[self._optimization.optimizers[name] for name in optimizer_names],
            *[self._optimization.schedulers[name] for name in scheduler_names],
        ]
        if not components:
            raise RuntimeError("Strategy returned no optimizers to prepare")

        prepared = self._prepare_tuple(components)
        prepared_optimizers = dict(zip(optimizer_names, prepared[: len(optimizer_names)], strict=True))
        prepared_schedulers = dict(zip(scheduler_names, prepared[len(optimizer_names) :], strict=True))
        self._optimization = OptimizationBundle(
            optimizers=prepared_optimizers,
            schedulers=prepared_schedulers,
        )

    def _restore_resume_state(self, resume_dir: Path) -> bool:
        """Restore optimizer/scheduler (via the strategy) and RNG (trainer-owned) after prepare.
        Returns True on success; False signals the caller to fall back to step 0.
        """
        try:
            blob = self._checkpoints.read_resume_state(resume_dir)
            self._training_strategy.load_training_state(blob.get("strategy_state", {}))
            self._restore_rng(blob.get("rng"))
            return True
        except Exception as e:
            logger.warning(f"⚠️ Failed to restore resume state from {resume_dir}: {e}. Starting from step 0.")
            return False

    def _restore_rng(self, rng: dict[str, Tensor | None] | None) -> None:
        """Restore RNG state. Multi-process runs never broadcast rank-0 RNG to all ranks."""
        if rng is None:
            return
        if self._accelerator.num_processes > 1:
            logger.debug("Skipping RNG restore in multi-process mode (only main process state was saved)")
            return
        if rng.get("torch") is not None:
            torch.random.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng["cuda"])
        logger.debug("Restored RNG states")

    def _run_training_loop(self, progress: TrainingProgress, dataloader: Iterable) -> float:
        """Run optional step-0 validation and the main training loop; return peak GPU memory (GB)."""
        cfg = self._config
        device = self._accelerator.device
        peak_mem = get_gpu_memory_gb(device)
        last_validation: ValidationResult | None = None
        max_micro_steps = (
            self._remaining_steps * cfg.optimization.gradient_accumulation_steps * _MICRO_STEP_SAFETY_FACTOR
        )

        if cfg.validation.interval and not cfg.validation.skip_initial_validation:
            with self._offloaded_optimizer_state():
                last_validation = self._run_validation(progress)
        self._accelerator.wait_for_everyone()

        batches = self._iter_batches(dataloader)
        micro_step = 0
        while self._global_step < cfg.optimization.steps:
            if micro_step >= max_micro_steps:
                raise RuntimeError(
                    f"Training exceeded {max_micro_steps} micro-steps without reaching the target "
                    f"{cfg.optimization.steps} optimization steps; the strategy may never be reporting "
                    "completed_step. Aborting to avoid an infinite loop."
                )

            step_start_time = time.time()
            result = self._training_strategy.train_step(
                next(batches),
                StepContext(
                    global_step=self._global_step,
                    accelerator=self._accelerator,
                    device=device,
                    output_dir=Path(cfg.output_dir),
                    max_grad_norm=cfg.optimization.max_grad_norm,
                ),
            )
            is_optimization_step = self._completed_step_consensus(result.completed_step)
            if is_optimization_step:
                self._global_step += 1

            if self._on_cadence(cfg.validation.interval, is_optimization_step=is_optimization_step):
                with self._offloaded_optimizer_state():
                    last_validation = self._run_validation(progress)
            if self._on_cadence(cfg.checkpoints.interval, is_optimization_step=is_optimization_step):
                self._save_checkpoint()
            if self._step_callback and is_optimization_step:
                self._accelerator.wait_for_everyone()
                self._step_callback(self._global_step, cfg.optimization.steps, last_validation)
                self._accelerator.wait_for_everyone()

            self._log_step(result, step_start_time, is_optimization_step=is_optimization_step, progress=progress)

            if micro_step % MEMORY_CHECK_INTERVAL == 0:
                peak_mem = max(peak_mem, get_gpu_memory_gb(device))
            micro_step += 1
        return peak_mem

    @staticmethod
    def _iter_batches(dataloader: Iterable) -> Iterator:
        """Yield batches forever, restarting the dataloader each time it is exhausted."""
        while True:
            yield from dataloader

    def _completed_step_consensus(self, completed_step: bool) -> bool:
        """Check that all ranks agree on logical step completion."""
        if self._accelerator.num_processes == 1:
            return completed_step
        local = torch.tensor([int(completed_step)], device=self._accelerator.device)
        gathered = self._accelerator.gather(local)
        completed_steps = [bool(value.item()) for value in gathered]

        if not completed_steps:
            return False
        first = completed_steps[0]
        if any(completed != first for completed in completed_steps[1:]):
            raise RuntimeError(f"Rank disagreement for completed_step: {completed_steps}")
        return first

    def _on_cadence(self, interval: int | None, *, is_optimization_step: bool) -> bool:
        """Whether a periodic action (validation/checkpoint) is due at the current step."""
        return bool(interval and is_optimization_step and self._global_step > 0 and self._global_step % interval == 0)

    def _log_step(
        self,
        result: StepResult,
        step_start_time: float,
        *,
        is_optimization_step: bool,
        progress: TrainingProgress,
    ) -> None:
        """Update the progress bar and log metrics (W&B + console fallback) for one step.
        Only optimization steps are logged: they advance the logical clock, and reducing across
        ranks on every micro-step would add a needless collective per accumulation sub-step. The
        trainer makes no assumptions about which metrics a strategy reports: it merges the
        strategy's metrics with the run-shell learning rate(s) (added only if the strategy did not
        already report them), reduces the whole set across ranks by each metric's policy, and logs
        everything. Any ``StepResult.media`` is logged through the shared W&B sink.
        """
        if not is_optimization_step:
            return

        cfg = self._config
        step_time = (time.time() - step_start_time) * cfg.optimization.gradient_accumulation_steps

        metrics: dict[str, float | MetricValue] = dict(result.metrics)
        for name, optimizer in self._optimization.optimizers.items():
            # LR is identical across ranks; reduce it anyway (mean) to keep the collective symmetric
            # and to include it in the logged set. Skip if the strategy already reported this key.
            metrics.setdefault(
                f"train/learning_rate/{name}",
                MetricValue(value=float(optimizer.param_groups[0]["lr"]), reduction="mean"),
            )

        reduced = reduce_step_metrics(metrics, self._accelerator)
        progress.update_training(metrics=reduced, step_time=step_time, advance=True)

        self._wandb.log(
            {
                **reduced,
                "train/step_time": step_time,
                "train/global_step": self._global_step,
            }
        )
        if result.media:
            self._wandb.log_media(result.media, step=self._global_step)

        if self._disable_progress_bars and IS_MAIN_PROCESS and self._global_step % 20 == 0:
            elapsed = time.time() - self._train_start_time
            steps_done = self._global_step - self._initial_step
            if steps_done > 0:
                total_estimated = elapsed / steps_done * self._remaining_steps
                total_time = f"{total_estimated // 3600:.0f}h {(total_estimated % 3600) // 60:.0f}m"
            else:
                total_time = "calculating..."
            metrics_str = ", ".join(f"{key}={value:.4g}" for key, value in reduced.items())
            logger.info(
                f"Step {self._global_step}/{cfg.optimization.steps} - "
                f"{metrics_str}, Time/Step: {step_time:.2f}s, Total Time: {total_time}",
            )

    def _finalize_training(self, peak_mem_during_training: float) -> tuple[Path, TrainingStats]:
        """Compute final stats, write the final checkpoint, and log (teardown happens in train()'s finally)."""
        # Symmetric with on_train_start() (fired in _setup_for_training); only on a successful run.
        self._training_strategy.on_train_end()

        cfg = self._config
        total_time_seconds = time.time() - self._train_start_time
        peak_mem = max(self._start_mem, get_gpu_memory_gb(self._accelerator.device), peak_mem_during_training)
        steps_per_second = self._remaining_steps / total_time_seconds
        stats = TrainingStats(
            total_time_seconds=total_time_seconds,
            steps_per_second=steps_per_second,
            samples_per_second=(
                steps_per_second
                * self._accelerator.num_processes
                * cfg.optimization.batch_size
                * cfg.optimization.gradient_accumulation_steps
            ),
            peak_gpu_memory_gb=peak_mem,
            num_processes=self._accelerator.num_processes,
            global_batch_size=(
                cfg.optimization.batch_size
                * cfg.optimization.gradient_accumulation_steps
                * self._accelerator.num_processes
            ),
        )

        # Skip the final checkpoint if the last training step already wrote one for this step.
        if self._last_checkpoint_step == self._global_step:
            saved_path = self._checkpoints.step_dir(self._global_step)
        else:
            saved_path = self._save_checkpoint()

        if IS_MAIN_PROCESS:
            self._log_training_stats(stats)
        self._wandb.log(
            {
                "stats/total_time_minutes": stats.total_time_seconds / 60,
                "stats/steps_per_second": stats.steps_per_second,
                "stats/samples_per_second": stats.samples_per_second,
                "stats/peak_gpu_memory_gb": stats.peak_gpu_memory_gb,
            }
        )

        self._accelerator.wait_for_everyone()
        return saved_path, stats

    def _teardown(self) -> None:
        """Always-run cleanup (called from train()'s finally): finish W&B and end the Accelerate run.
        Deliberately performs NO collective (no wait_for_everyone): on an exception ranks may have
        diverged, and a barrier here would hang instead of letting the original error surface.
        """
        self._wandb.finish()
        self._accelerator.end_training()

    # --- Validation ---
    def _run_validation(self, progress: TrainingProgress) -> ValidationResult:
        """Delegate validation to the strategy and log the returned result.
        The trainer owns cadence (caller), optimizer CPU offload (caller), and logging; the
        strategy owns validation content and drives the shared distributed runner (sharding,
        FSDP padding, gather, deterministic sort). The trainer logs media through the shared W&B
        sink and reduces the returned scalars across ranks before logging them. Returns the
        ``ValidationResult`` for the step callback.
        Note: Multi-node training requires a shared filesystem so rank 0 can read media
        written by other ranks.
        """
        ctx = ValidationContext(
            accelerator=self._accelerator,
            device=self._accelerator.device,
            step=self._global_step,
            output_dir=Path(self._config.output_dir),
            progress=progress,
        )
        result = self._training_strategy.validate(ctx)

        if self._config.wandb.log_validation_videos:
            self._wandb.log_media(result.media, step=self._global_step)

        # Collective on every rank (symmetric on the validation cadence); empty on ranks with no
        # scalars. Reduce first, then log the main-process result.
        scalars = reduce_step_metrics(result.scalars, self._accelerator)
        if scalars:
            self._wandb.log(scalars, step=self._global_step)

        # Non-main ranks must not reach checkpoint collectives while main is still logging to W&B.
        self._accelerator.wait_for_everyone()

        return result

    @contextlib.contextmanager
    def _offloaded_optimizer_state(self) -> Iterator[None]:
        """Context manager that offloads optimizer state to CPU during validation.
        Opt-in via `optimization.offload_optimizer_during_validation`. Frees VRAM for
        validation video generation when optimizer state is large (e.g. full fine-tune
        AdamW, high-rank LoRA). No-op for FSDP (sharded state -- manual `.cpu()` breaks
        metadata).
        """
        enabled = (
            self._config.optimization.offload_optimizer_during_validation
            and self._accelerator.distributed_type != DistributedType.FSDP
        )

        # Track exactly which tensors we move so we don't promote ones that were
        # intentionally on CPU (e.g. AdamW's `step` scalar on recent PyTorch). All of the
        # strategy's optimizers are offloaded (a strategy may expose more than one).
        offloaded: list[tuple[dict, str]] = []
        if enabled:
            offloaded_bytes = 0
            for optimizer in self._optimization.optimizers.values():
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor) and v.is_cuda:
                            offloaded.append((state, k))
                            offloaded_bytes += v.nbytes
            if offloaded:
                logger.info(f"Offloading optimizer state to CPU ({offloaded_bytes / 1e9:.1f} GB)")
                for state, k in offloaded:
                    state[k] = state[k].cpu()

        try:
            yield
        finally:
            device = self._accelerator.device
            for state, k in offloaded:
                state[k] = state[k].to(device)

    # --- Checkpointing ---
    def _save_checkpoint(self) -> Path:
        """Save a checkpoint for the current step and return its (eventual) directory on every rank.
        The trainer owns the distributed dance: it picks the save mode, runs the collective
        ``collect_checkpoint`` on every rank, captures RNG, and barriers. The actual byte
        serialization, atomic write, metadata, and retention live in ``CheckpointManager``.
        """
        mode = self._config.checkpoints.save_training_state
        is_fsdp = self._accelerator.distributed_type == DistributedType.FSDP
        include_optimizer = mode == "full" and not is_fsdp
        if mode == "full" and is_fsdp and not self._fsdp_optimizer_warned:
            self._fsdp_optimizer_warned = True
            logger.warning(
                "⚠️ save_training_state='full' is not supported with FSDP. "
                "Saving scheduler + RNG only (no optimizer state)."
            )

        # Collective: must run on every rank (FSDP gathers shards inside collect_checkpoint).
        payload = self._training_strategy.collect_checkpoint(include_optimizer=include_optimizer)

        step_dir = self._checkpoints.step_dir(self._global_step)
        if IS_MAIN_PROCESS:
            rng = {
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            }
            self._checkpoints.write(
                step=self._global_step,
                payload=payload,
                rng=rng,
                wandb_run_id=self._wandb.run_id,
                fingerprint=self._training_strategy.resume_fingerprint().fields,
                write_resume_state=mode != "off",
            )

        # Recorded on every rank so the final-save skip decision stays rank-consistent.
        self._last_checkpoint_step = self._global_step
        self._accelerator.wait_for_everyone()
        return step_dir

    # --- Logging & config ---
    @staticmethod
    def _log_training_stats(stats: TrainingStats) -> None:
        """Log training statistics."""
        stats_str = (
            "📊 Training Statistics:\n"
            f" - Total time: {stats.total_time_seconds / 60:.1f} minutes\n"
            f" - Training speed: {stats.steps_per_second:.2f} steps/second\n"
            f" - Samples/second: {stats.samples_per_second:.2f}\n"
            f" - Peak GPU memory: {stats.peak_gpu_memory_gb:.2f} GB"
        )
        if stats.num_processes > 1:
            stats_str += f"\n - Number of processes: {stats.num_processes}\n"
            stats_str += f" - Global batch size: {stats.global_batch_size}"
        logger.info(stats_str)

    def _save_config(self) -> None:
        """Save the training configuration as a YAML file in the output directory."""
        if not IS_MAIN_PROCESS:
            return

        config_path = Path(self._config.output_dir) / "training_config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(self._config.model_dump(mode="json"), f, default_flow_style=False, indent=2)

        logger.info(f"💾 Training configuration saved to: {config_path.relative_to(self._config.output_dir)}")
