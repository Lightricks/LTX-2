# Implementing Custom Training Strategies

Add a strategy for algorithms that the built-in `flexible` config cannot express: custom losses/noising,
new conditioning, multi-model training, or multi-phase updates such as GAN/RL cycles. For first-frame, extension,
inpainting, outpainting, references, and frozen cross-modal conditioning, use [Training Modes](training-modes.md).

## 📋 Overview

The trainer owns Accelerate setup, the loop, validation/checkpoint cadence, checkpoint files, W&B, and distributed
coordination. A strategy owns:

- **Its models** — what to build and which parameters are trainable.
- **Its optimizers/schedulers**.
- **Its data** — how batches are produced and, optionally, how the dataset is preprocessed.
- **Its training step** — forward pass, loss, and what counts as one logical step.
- **Its validation and checkpoint payload**.

Strategies self-register; adding one does not require trainer-core branches or a static config union.

## 🏗️ Architecture Overview

### The Strategy Contract

Implement [`TrainingStrategy`](../src/ltx_trainer/training_strategies/base_strategy.py), directly or by reusing an
existing implementation. A *hook* is a method the trainer calls when it needs your strategy's part of the work.
The walkthrough below follows a run from setup to its final checkpoint. The six hooks marked **required** are abstract;
implement them even if your first use case does not involve resuming. Other hooks have inherited defaults.
In distributed runs, each GPU process is a *rank*; collective operations require every rank to participate.

#### 1. Check the data and build the models

Before loading models, the trainer checks cached-data compatibility using `preprocessing_fingerprint()`.
Its default, `None`, opts out of manifest checks and writes; return a fingerprint to use that protocol.

**Required: `build_models()` → `ModelBundle`.** Construct the models and give them stable names, such as `"student"`.
Add adapters and decide which parameters are trainable here. The next stage builds optimizers from these parameters,
so changes to the trainable graph must be finished before then.

#### 2. Set up optimization and prepare for distributed training

**Required: `build_optimizers(models, total_steps)` → `OptimizationBundle`.** Create the named optimizers and optional
schedulers for your models. `total_steps` is the configured training budget; use it to set the schedule's duration.

The optional `resume_fingerprint()` declares architecture/shape constraints the trainer checks for a safe resume;
its default is empty. Keep freely adjustable settings, such as learning rate, out of this fingerprint.
If a checkpoint is being loaded, the trainer calls **required `load_model_state(weights)`** to restore its weights.
This happens after adapters exist but **before Accelerate wraps the models**, so you can load into ordinary modules.

The trainer then prepares models, optimizers, and schedulers with Accelerate for device placement and distributed
execution. `on_before_prepare(accelerator)` is an optional place for distributed-aware model adjustments, such as an
FSDP dtype cast. After preparation, `attach_prepared_components(...)` stores the wrapped components as
`self.models`, `self.optimization`, and `self.accelerator`. Usually, keep this inherited implementation.

#### 3. Restore training state and start supplying batches

When resuming, the trainer calls **required `load_training_state(state)`** to restore strategy-owned state, including
saved optimizer/scheduler state. This happens **after preparation**, so it restores into the components training will
actually use. The trainer restores RNG state separately; see [resume limitations](training-guide.md#resume-training).

**Required: `build_dataloader(batch_size)` → an unprepared iterable of batches.** Use the requested per-process batch
size; the trainer prepares the dataloader with Accelerate. It then calls `on_train_start()`, an optional setup hook
where you can put models in training mode. The run is now ready for its first batch.

#### 4. Turn batches into training updates

For each batch, the trainer calls `train_step(batch, ctx)` and expects a `StepResult`. `ctx` supplies the Accelerator,
device, current global step, output directory, and gradient-clipping limit.

For a standard one-model, one-optimizer update, implement `_compute_loss(batch, models)` returning a loss tensor of shape
`[B]` (one value per sample) and metrics. The inherited `train_step` handles accumulation, backward, clipping using
`ctx.max_grad_norm`, optimizer/scheduler steps, and completion reporting. Override `train_step` for a different
update cycle, such as alternating generator and discriminator updates.

> [!IMPORTANT]
> **A batch is not necessarily a completed training step.** `StepResult.completed_step` tells the trainer when an
> update is finished. Only then does it advance `global_step`, check whether validation/checkpoints are due, and count
> toward the step budget. The inherited step uses `accelerator.sync_gradients`, so accumulation batches do not each
> count as a step. For a multi-phase algorithm, report completion once per full cycle. **All ranks must agree** on
> this flag; the trainer raises an error if they disagree.

#### 5. Validate and checkpoint at the configured intervals

The trainer calls optional `validate(ctx)` when validation is due, including before the first batch if initial
validation is enabled. Return a `ValidationResult` with media and scalar metrics for the trainer to log and reduce;
the default is empty. After each completed step, validation runs before checkpointing if both are due.

**Required: `collect_checkpoint(include_optimizer)` → `CheckpointPayload`.** Gather weights and training state,
respecting whether optimizer state was requested. The trainer writes the files; your strategy returns the data.
**Every rank must participate**, including any FSDP state gathering, even though only the main rank writes files.
The two restore hooks above are the reverse path: they load this payload into a later run.

#### 6. Finish the run

After the step budget is reached, the trainer calls optional `on_train_end()` and saves a final checkpoint if that step
has not already been saved. `on_train_start()` and `on_train_end()` do nothing by default. `on_train_end()` runs after
a successful loop, so do not rely on it for exception cleanup. The trainer handles W&B and Accelerate teardown.

**Preprocessing is a separate command.** To support `scripts/preprocess.py`, also implement
`build_preprocessing_pipeline()`; its default raises `NotImplementedError`. The preprocessing command runs that pipeline
to create cached data before training. See [Preprocessing Pipeline](#preprocessing-pipeline).

### The Building Blocks (types)

The contract uses dataclasses in [`training_strategies/types.py`](../src/ltx_trainer/training_strategies/types.py):

| Type | Contents / use |
|------|----------------|
| `ModelBundle` | Named modules, e.g. `models={"transformer": module}`. Names stay stable across preparation. |
| `OptimizationBundle` | Named optimizers; optional schedulers use the corresponding optimizer names. The trainer reattaches components by name. |
| `StepContext` | Accelerator, device, output directory, global step, and max gradient norm. |
| `ValidationContext` | Accelerator, device, output directory, step, and nested progress reporter. |
| `StepResult` | `completed_step`, metrics, and optional media. |
| `MetricValue` | Scalar plus reduction policy: `mean`, `sum`, `min`, `max`, `main_rank`, or `already_reduced`; weighted means are supported. Plain floats use an unweighted mean across ranks. |
| `MediaArtifact` | Media file with `kind` (`image`, `video`, `audio`) and logging key. |
| `ValidationResult` | Media list and scalar metrics reduced by the trainer. |
| `CheckpointPayload` | Weights, optional weight metadata, and optional training state; the trainer writes the files. |
| `ResumeFingerprint` | Diffable fields that must match on resume. |

## 📝 Minimal Example: Offline Guidance Distillation

Suppose you want a model to produce guided predictions without an extra unconditional pass at inference.
Use an existing model (the **teacher**) to produce training targets, then train another model (the **student**) to match them.

Classifier-free guidance (**CFG**) combines a text-conditioned prediction with an unconditional prediction to strengthen
prompt guidance. Here, you compute that combination ahead of time and save it to disk: this is **offline guidance
distillation**. Training then needs only the student, one optimizer, and one forward pass per batch.
The target is the teacher's guided velocity, rather than the usual noise-minus-clean flow target. At inference,
evaluate the student with CFG scale 1, since guidance is already included in its targets.

The strategy below implements `TrainingStrategy` directly. It trains full student weights and uses the base
`train_step` for accumulation, backward, clipping, and optimizer updates. It distills guidance; it does not change
the sampling schedule. Use [full-training resources](training-modes.md#-full-transformer-fine-tuning) for real checkpoints.

### 1. Cache Teacher Predictions

For each text-conditioned video example, run the teacher's conditional and unconditional predictions on the **same
noisy tokens and timestep**, then cache `target_velocity = uncond + guidance_scale * (cond - uncond)`.
Use varied prompts, noise samples, and timesteps. Save each record as a dictionary of detached CPU tensors in
`/path/to/cache/distillation/<sample>.pt` with these **unbatched** fields:

| Field | Shape / contents |
|-------|------------------|
| `latent` | `[T, D]` noisy video tokens given to the teacher. |
| `sigma` | Scalar noise level. |
| `timesteps` | `[T]` dense per-token timesteps, taken from `video.timesteps.per_token()[0, :, 0]`. |
| `positions` | Teacher video positions: `[3, T, 2]` patch bounds, or `[3, T]` legacy indices. Preserve its time/FPS convention. |
| `context` | `[L, C]` conditional **transformer-ready** text context, after text projection/connectors. |
| `target_velocity` | `[T, D]` guided teacher prediction. |
| `context_mask`, `keyframes_mask` | Optional teacher-input masks, with the batch dimension removed. Omit keys whose value is `None`. |

`T` is video token count, `D` is latent feature width, `L` is text token count, and `C` is text feature width.
Use the same checkpoint architecture and input conventions for teacher and student. This example uses text-only
conditioning and video-only forward passes. Ordinary flexible
preprocessing artifacts are not these records: they do not contain noisy model inputs or guided teacher targets.
See [`Modality`](../../ltx-core/src/ltx_core/model/transformer/modality.py) for the complete input shapes.

### 2. Implement and Register the Strategy

Create `src/ltx_trainer/training_strategies/guidance_distillation/__init__.py`. Keeping config and strategy together
makes the complete example one file; split them when the implementation grows.

```python
from typing import Any, Literal

import torch
from pydantic import DirectoryPath, Field, FilePath
from torch import Tensor
from torch.utils.data import DataLoader

from ltx_core.model.transformer.modality import Modality, TokenTimesteps
from ltx_trainer.datasets import PrecomputedDataset
from ltx_trainer.model_loader import load_transformer
from ltx_trainer.training_strategies.base_strategy import TrainingStrategy, TrainingStrategyConfigBase
from ltx_trainer.training_strategies.registry import register_strategy
from ltx_trainer.training_strategies.types import (
    CheckpointPayload,
    ModelBundle,
    OptimizationBundle,
    ResumeFingerprint,
)


class GuidanceDistillationConfig(TrainingStrategyConfigBase):
    name: Literal["guidance_distillation"] = "guidance_distillation"
    model_path: FilePath
    data_root: DirectoryPath
    learning_rate: float = Field(default=1e-5, gt=0)


@register_strategy("guidance_distillation", GuidanceDistillationConfig)
class GuidanceDistillationStrategy(TrainingStrategy):
    config: GuidanceDistillationConfig

    def build_models(self) -> ModelBundle:
        student = load_transformer(self.config.model_path, device="cpu", dtype=torch.float32)
        student.requires_grad_(True).train()
        student.set_gradient_checkpointing(True)
        self.models = ModelBundle(models={"student": student})
        return self.models

    def build_optimizers(self, models: ModelBundle, total_steps: int) -> OptimizationBundle:  # noqa: ARG002
        optimizer = torch.optim.AdamW(models.models["student"].parameters(), lr=self.config.learning_rate)
        return OptimizationBundle(optimizers={"student": optimizer})

    def build_dataloader(self, batch_size: int) -> DataLoader:
        dataset = PrecomputedDataset(str(self.config.data_root), data_sources={"distillation": "record"})
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    def _compute_loss(self, batch: dict[str, Any], models: ModelBundle) -> tuple[Tensor, dict[str, float]]:
        record = batch["record"]
        video = Modality(
            latent=record["latent"],
            sigma=record["sigma"],
            timesteps=TokenTimesteps(record["timesteps"]),
            positions=record["positions"],
            context=record["context"],
            context_mask=record.get("context_mask"),
            keyframes_mask=record.get("keyframes_mask"),
        )
        prediction, _ = models.models["student"](video=video, audio=None, perturbations=None)
        loss = (prediction.float() - record["target_velocity"].float()).square().mean(dim=(1, 2))
        return loss, {}

    def collect_checkpoint(self, include_optimizer: bool) -> CheckpointPayload:
        weights = self.accelerator.get_state_dict(self.models.models["student"])
        state = {"optimizer": self.optimization.optimizers["student"].state_dict()} if include_optimizer else {}
        return CheckpointPayload(weights=weights, training_state=state)

    def load_model_state(self, weights: dict[str, Tensor]) -> None:
        self.models.models["student"].load_state_dict(weights, strict=True)

    def load_training_state(self, state: dict[str, Any]) -> None:
        if "optimizer" in state:
            self.optimization.optimizers["student"].load_state_dict(state["optimizer"])

    def resume_fingerprint(self) -> ResumeFingerprint:
        return ResumeFingerprint(
            fields={
                "model_path": str(self.config.model_path.resolve()),
                "objective": "video_guidance_distillation_v1",
            }
        )
```

The example implements all six required hooks. The ordering matters:

- `build_models()` stores `self.models` so saved weights can be restored before Accelerate wraps the student.
  The trainer later replaces this with the prepared bundle.
- Checkpoint collection runs on every rank and uses Accelerate to gather weights. The trainer writes the files;
  these weights are student state dictionaries for this strategy's restore hook.
- Optimizer state is restored after preparation, when present. The trainer restores the step and handles
  random-number-generator (RNG) state; see [resume limitations](training-guide.md#resume-training).

The empty metrics dictionary keeps the example focused on the objective; add scalar `MetricValue` entries for loss
logging. Optional validation and preprocessing hooks retain their base defaults. Discovery imports the package and
runs its decorator: the folder, registered name, and config Literal must agree. **No central import, static config
union, or trainer-core branch is needed.**
Start a fresh Python process after adding a strategy so discovery and the config union are rebuilt.

### 3. Create a Run Config

Save as `configs/guidance_distillation.yaml`, using an existing model checkpoint and the parent of `distillation/`:

```yaml
training_strategy:
  name: "guidance_distillation"
  model_path: "/path/to/model.safetensors"
  data_root: "/path/to/cache"
  learning_rate: 1e-5
optimization:
  steps: 100
  batch_size: 1
validation:
  interval: null
checkpoints:
  interval: 50
  save_training_state: "full"
output_dir: "outputs/guidance_distillation"
```

This strategy owns its own schema; do not copy flexible-only fields such as `video`, `lora`, or `data` into it.
Batch size 1 allows records with different token counts. Larger batches require matching shapes and optional keys.
Cache generation is a separate step here; implement a preprocessing pipeline if you want it integrated with the CLI.

### Extending the Example

The student can now train and resume. Next, make its progress visible and check whether it matches teacher targets
on data that it does not train on. Both additions fit in the same strategy package.

#### Log the distillation loss

Add `MetricValue` to the imports from `training_strategies.types`. In `_compute_loss`, change the return annotation to
`tuple[Tensor, dict[str, float | MetricValue]]` and replace the final `return loss, {}` with:

```python
return loss, {
    "train/distillation_mse": MetricValue(value=loss.detach().mean().item(), weight=loss.numel()),
}
```

The returned `loss` still drives backward. The metric is a detached scalar for logging; `weight` is the number of
samples, so the trainer can compute a sample-weighted mean across ranks. Keep the inherited `train_step`.

#### Check a held-out example during validation

Prepare a separate cache with the same record format and teacher settings, excluding those samples from training.
Add this optional field to `GuidanceDistillationConfig`:

```python
validation_data_root: DirectoryPath | None = None
```

Add `from accelerate.utils import send_to_device` and import `ValidationContext` and `ValidationResult` from
`training_strategies.types`. Then add this method to `GuidanceDistillationStrategy`:

```python
@torch.no_grad()
def validate(self, ctx: ValidationContext) -> ValidationResult:
    if self.config.validation_data_root is None:
        return ValidationResult()
    dataset = PrecomputedDataset(
        str(self.config.validation_data_root), data_sources={"distillation": "record"}
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    batch = send_to_device(batch, ctx.device)
    student = self.models.models["student"]
    was_training = student.training
    try:
        student.eval()
        loss, _ = self._compute_loss(batch, self.models)
    finally:
        student.train(was_training)
    return ValidationResult(
        scalars={"validation/distillation_mse": MetricValue(value=loss.mean().item(), weight=loss.numel())}
    )
```

This deliberately checks the first held-out record each time: a small, repeatable check rather than a full evaluation.
Every rank runs the same forward pass, keeping calls aligned for FSDP. The hook restores the student's previous
training mode; the trainer handles metric reduction and logging.

> [!IMPORTANT]
> Use `@torch.no_grad()` for strategy validation. `@torch.inference_mode()` can break FSDP validation.

Merge these settings into the run config from step 3 to enable the check before training and every 25 updates:

```yaml
training_strategy:
  validation_data_root: "/path/to/held_out_cache"
validation:
  interval: 25
  skip_initial_validation: false
```

For larger extensions, reuse `ModelSourceConfig`, `LoraConfig`, `DataConfig`, and `OptimizerConfig` from the
[shared config blocks](../src/ltx_trainer/training_strategies/shared_config.py) when their schemas fit.
Keep model/optimizer/data logic in your package. `PrecomputedDataset` maps **directory names to batch keys**;
`FlexibleStrategy.get_data_sources()` is a helper, not a base-contract requirement. See the
[built-in strategy](../src/ltx_trainer/training_strategies/flexible/strategy.py) for LoRA and media-validation examples.

### Preprocessing Pipeline

Now automate the teacher-prediction caching described in step 1. A
[`PreprocessingPipeline`](../src/ltx_trainer/preprocessing/types.py) is an ordered list of steps; each step has a
`run(ctx)` method. The CLI supplies the dataset file, output root, device, and overwrite setting through `ctx`.

This example starts from **prepared teacher inputs**, after video/text encoding and noise sampling. It generates the
guided targets rather than encoding raw videos. Ordinary flexible caches need additional preparation: the input must
include noisy tokens, timesteps, positions, and transformer-ready conditional **and unconditional** text contexts.

#### Describe the input records

Save one unbatched tensor dictionary per example. Use the fields in [step 1](#1-cache-teacher-predictions), omitting
`target_velocity` and adding `uncond_context` with shape `[L_uncond, C]`. If the unconditional pass uses a context mask,
also include `uncond_context_mask`. The conditional and unconditional passes share the same noisy video and timestep;
only their text context and its mask differ. Omit optional keys whose value is `None`.

List these files in `teacher_inputs.jsonl`. Paths are relative to that JSONL file (absolute paths also work):

```jsonl
{"record_path": "teacher_inputs/sample_001.pt"}
{"record_path": "teacher_inputs/sample_002.pt"}
```

#### Implement the caching step

Create `guidance_distillation/preprocessing.py` beside the strategy's `__init__.py`:

```python
import json
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from accelerate import PartialState
from accelerate.utils import send_to_device

from ltx_core.model.transformer.modality import Modality, TokenTimesteps
from ltx_trainer.model_loader import load_transformer
from ltx_trainer.preprocessing.types import PreprocessContext


@dataclass(frozen=True)
class CacheTeacherPredictions:
    model_path: str
    guidance_scale: float

    @torch.inference_mode()
    def run(self, ctx: PreprocessContext) -> None:
        manifest = Path(ctx.dataset_file)
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        output = ctx.output_root / "distillation"
        output.mkdir(parents=True, exist_ok=True)
        state = PartialState()
        teacher = load_transformer(self.model_path, device=ctx.device, dtype=torch.float32)
        teacher.requires_grad_(False).eval()
        for index in range(state.process_index, len(rows), state.num_processes):
            destination = output / f"{index:08d}.pt"
            if destination.exists() and not ctx.overwrite:
                continue
            source = manifest.parent / rows[index]["record_path"]
            record = torch.load(source, map_location="cpu", weights_only=True)
            record = {key: value.float() if value.is_floating_point() else value for key, value in record.items()}
            batch = send_to_device({key: value.unsqueeze(0) for key, value in record.items()}, ctx.device)
            video = Modality(
                latent=batch["latent"],
                sigma=batch["sigma"],
                timesteps=TokenTimesteps(batch["timesteps"]),
                positions=batch["positions"],
                context=batch["context"],
                context_mask=batch.get("context_mask"),
                keyframes_mask=batch.get("keyframes_mask"),
            )
            unconditional = replace(
                video, context=batch["uncond_context"], context_mask=batch.get("uncond_context_mask")
            )
            cond, _ = teacher(video=video, audio=None, perturbations=None)
            uncond, _ = teacher(video=unconditional, audio=None, perturbations=None)
            target = uncond + self.guidance_scale * (cond - uncond)
            record = {key: value for key, value in record.items() if not key.startswith("uncond_")}
            record["target_velocity"] = target.squeeze(0).cpu()
            temporary = destination.with_suffix(".tmp")
            torch.save(record, temporary)
            temporary.replace(destination)
```

The output is exactly what the existing dataloader expects: `data_root/distillation/<sample>.pt`, with CPU tensors and
no batch dimension. The loop gives each rank different rows when launched with Accelerate. Existing files are skipped
unless `--overwrite` is passed; the temporary file avoids leaving a partial `.pt` record after an interrupted write.
This step uses an unwrapped teacher, so `inference_mode()` is appropriate here.

#### Connect the pipeline and compatibility checks

Add these fields to `GuidanceDistillationConfig`:

```python
dataset_file: FilePath
guidance_scale: float = Field(default=3.0, gt=0)
```

In `__init__.py`, add `from pathlib import Path`, import `CacheTeacherPredictions` from `.preprocessing`, and import
`PreprocessingFingerprint` and `PreprocessingPipeline` from `ltx_trainer.preprocessing.types`.
Add both methods to `GuidanceDistillationStrategy`:

```python
def build_preprocessing_pipeline(self) -> PreprocessingPipeline:
    return PreprocessingPipeline(
        steps=[CacheTeacherPredictions(str(self.config.model_path), self.config.guidance_scale)],
        dataset_file=str(self.config.dataset_file),
        output_root=Path(self.config.data_root).resolve(),
    )

def preprocessing_fingerprint(self) -> PreprocessingFingerprint:
    return PreprocessingFingerprint(
        artifact_root=Path(self.config.data_root).resolve(),
        fields={
            "record_format": "video_guidance_distillation_v1",
            "dataset_file": str(self.config.dataset_file.resolve()),
            "teacher_model": str(self.config.model_path.resolve()),
            "guidance_scale": self.config.guidance_scale,
        },
    )
```

The output root and fingerprint root must agree. These fields describe what affects cached targets; learning rate,
batch size, and device do not belong here. The CLI writes the compatibility manifest after the pipeline succeeds,
and training checks it before loading models. Your step writes records, not the compatibility manifest.
The base fingerprint's `None` disables **all** manifest checks/writes, even with a preprocessing pipeline.

> [!WARNING]
> Fingerprints compare declared settings, not file contents. If input records or weights change at the same paths,
> rebuild with `--overwrite` or use a fresh cache root, including after reordering JSONL rows. Use a fresh root when
> removing samples: overwrite replaces current outputs but leaves orphan records. See
> [cache rebuild semantics](dataset-preparation.md#preprocessing-manifest-and-staleness-guard).

Merge the new fields into the same training YAML, keeping `data_root` as the parent of `distillation/`:

```yaml
training_strategy:
  dataset_file: "/path/to/teacher_inputs.jsonl"
  guidance_scale: 3.0
```

From `packages/ltx-trainer`, create the cache root (required by the existing `DirectoryPath` field), then preprocess,
check compatibility without regenerating records, and train:

```bash
mkdir -p /path/to/cache
uv run python scripts/preprocess.py configs/guidance_distillation.yaml
uv run python scripts/preprocess.py configs/guidance_distillation.yaml --check
uv run python scripts/train.py configs/guidance_distillation.yaml
```

To add raw video/text encoding ahead of teacher prediction, prepend steps to `steps=[...]`; they run in list order
and share `ctx.output_root`. See [flexible preprocessing](../src/ltx_trainer/training_strategies/flexible/preprocessing.py)
for encoder composition. Keep `resume_fingerprint()` separate: it describes checkpoint-resume compatibility.

## ✅ Testing Your Strategy

After creating the package and config, check discovery and parsing from `packages/ltx-trainer`:

```bash
uv run python - <<'PY'
from pathlib import Path

import yaml

from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.training_strategies import get_training_strategy

raw = yaml.safe_load(Path("configs/guidance_distillation.yaml").read_text())
config = LtxTrainerConfig.from_dict(raw)
strategy = get_training_strategy(config.training_strategy)
assert type(strategy).__name__ == "GuidanceDistillationStrategy"
PY
```

Test with a small transformer and synthetic cached records: the loss must have shape `[B]`, vanish when predictions
match targets, and produce student gradients otherwise. Exercise the inherited `train_step`, then checkpoint and
restore both weights and optimizer state. Check that minimal saves omit optimizer state and resume fingerprints
reject incompatible model sources.

If you added the extensions, check that validation leaves the student's training mode and weights unchanged.
For preprocessing, compare a cached target against `uncond + guidance_scale * (cond - uncond)` on a known input;
verify that reruns skip existing records, `--overwrite` replaces them, and a changed guidance scale fails compatibility
checks until the cache is rebuilt.

Then use a small real cache for a short run, choosing an appropriate [training launch](training-guide.md):

```bash
uv run python scripts/train.py configs/guidance_distillation.yaml
```

The example consumes cached teacher records directly; `scripts/preprocess.py` is available only after you implement
its preprocessing hook. Verify save/resume and every distributed mode you intend to support; see
[resume limitations](training-guide.md#resume-training).

## 📚 Reference Implementation

[`FlexibleStrategy`](../src/ltx_trainer/training_strategies/flexible/strategy.py) implements the complete contract,
including masked flow-matching loss, validation, Comfy-format adapter checkpoints, resume fingerprints, and FSDP
preparation. See [Configuration Reference](configuration-reference.md) for its schema and
[ltx-core](../../ltx-core/README.md) for model components.
