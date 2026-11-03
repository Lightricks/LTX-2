# Training Guide

Launch, distribute, monitor, and resume training. Commands assume `packages/ltx-trainer` as the working directory.
Prepare your [dataset](dataset-preparation.md) and [config](configuration-reference.md) first.

## ⚡ Basic Training (Single GPU)

```bash
uv run python scripts/train.py configs/t2v_lora.yaml
```

The trainer checks the config and, when available, the saved preprocessing settings before loading models.
During training it generates **validation samples** at configured intervals so you can see what the model is learning
and catch overfitting. It also saves **checkpoints** that you can use for inference or to continue training.
Try a small dataset and a few hundred steps before a long run. Use `--disable-progress-bars` to hide terminal progress bars.

For the built-in strategy, a training **step** means one completed parameter update. With gradient accumulation,
several batches contribute to that update before the step counter advances.

### Agent-Assisted Training

In environments with repository skills, [`/train`](../../../.claude/skills/train/SKILL.md) guides mode selection,
hardware/data checks, preprocessing, launch, and monitoring. It follows these docs and requests approval before
captioning, preprocessing, or expensive training work.

### Output Files

The trainer writes under `output_dir`:

```text
output_dir/
├── training_config.yaml          # Resolved training configuration
├── samples/                      # Validation media, if enabled
│   └── step_000100_1.mp4          # step_<step>_<sample index>.{mp4,png,wav}
└── checkpoints/
    └── step_00100/               # One directory per saved step
        ├── weights.safetensors   # LoRA adapters or full transformer weights
        ├── state.pt              # Optional resume state
        └── metadata.json         # Step, W&B run id, resume fingerprint, format version
```

The trainer first writes a temporary checkpoint directory, then renames it when complete. This prevents an interrupted
save from appearing as a usable checkpoint. `state.pt` and `metadata.json` are present together when resume state is enabled.

> [!IMPORTANT]
> **Save interval:** `checkpoints.interval: null` disables intermediate saves; a final checkpoint is still saved on
> normal completion. Set a positive interval to save checkpoints during the run for recovery after interruptions.

### Resume Training

Weights contain what the model has learned. Training state also tracks the step, the learning-rate schedule, and
optionally the optimizer's running statistics. Restoring that state lets you continue the run rather than start a new
one from the saved weights. RNG means random-number-generator state.

Set `checkpoints.load_checkpoint` to a run's `output_dir`, its `checkpoints/` directory (selects the latest checkpoint),
or a specific `step_NNNNN/` directory. The saved `checkpoints.save_training_state` setting determines what resumes:

| Saved state | Restored behavior |
|-------------|-------------------|
| `"full"` | Weights, step, scheduler, RNG, and optimizer on single GPU/DDP. |
| `"minimal"` | Weights, step, scheduler, and RNG; optimizer starts fresh. |
| `"off"` | Weights only; no `state.pt` or `metadata.json`, so the training state and step counter cannot resume. |

> [!NOTE]
> **Resume limits:** FSDP full saves omit optimizer state and warn; the optimizer starts fresh on resume.
> RNG state is restored only in single-process runs; multi-process runs do not restore each process's RNG state.

For example, merge this into the run config to continue a checkpoint at step 1,000 for another 500 steps:

```yaml
checkpoints:
  load_checkpoint: "outputs/my_run/checkpoints/step_01000"
optimization:
  steps: 1500  # Total target step, not additional steps
```

> [!WARNING]
> **Confirm resume in the logs:** Look for **“Resuming training from step …”** to verify that the step counter was restored.
>
> - Pointing directly at `weights.safetensors`, or setting `checkpoints.no_resume: true`, loads weights and starts at step 0.
> - Missing training state or incompatible resume metadata also falls back to saved weights at step 0.
> - A missing checkpoint path or empty directory warns and starts from the base model.

Use the **directory** when you intend to continue a run. Older flat checkpoint layouts are rejected. See
[checkpoint fields](configuration-reference.md#checkpointsconfig-checkpoints) for save cadence, retention, and precision.

## 🖥️ Distributed / Multi-GPU Training

The trainer uses [Accelerate](https://huggingface.co/docs/accelerate/index) to run one process per GPU. Choose how the
model is distributed:

- **DDP (Distributed Data Parallel)** keeps a full model copy on each GPU and splits batches across them.
- **FSDP (Fully Sharded Data Parallel)** splits model state across GPUs to reduce memory per GPU.
  Each GPU still needs enough memory for its local computation.

Start with one of the supplied profiles; the compile variants also enable `torch.compile`:

| Profile | Config |
|---------|--------|
| DDP | [ddp.yaml](../configs/accelerate/ddp.yaml) |
| DDP + `torch.compile` | [ddp_compile.yaml](../configs/accelerate/ddp_compile.yaml) |
| FSDP | [fsdp.yaml](../configs/accelerate/fsdp.yaml) |
| FSDP + `torch.compile` | [fsdp_compile.yaml](../configs/accelerate/fsdp_compile.yaml) |

Launch with a profile, visible GPUs, and a matching process count:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
uv run accelerate launch --config_file configs/accelerate/ddp.yaml --num_processes 2 \
  scripts/train.py configs/t2v_lora.yaml
```

Use another config from the table for FSDP or compilation. Without `--num_processes`, the profile's process count
applies; `CUDA_VISIBLE_DEVICES` selects available devices and should agree with that count. Compile profiles use
Inductor via Accelerate's `dynamo_config`; FSDP profiles auto-wrap `BasicAVTransformerBlock`.

Effective global batch size is `optimization.batch_size × gradient_accumulation_steps × num_processes`.
For example, batch size 1 with 4 accumulation batches on 2 GPUs gives an effective batch of **8 samples per update**.
Accumulation processes those batches in sequence, allowing a larger effective batch without holding them all in memory.

### Use a Default Accelerate Profile

Alternatively, run the interactive wizard once:

```bash
uv run accelerate config
```

It writes `~/.cache/huggingface/accelerate/default_config.yaml`. Launch without `--config_file` to use it:

```bash
uv run accelerate launch scripts/train.py configs/t2v_lora.yaml
```

The same `--num_processes` override and `CUDA_VISIBLE_DEVICES` selection apply.

## 📊 Weights & Biases Logging

Enable W&B to track loss, learning rate, validation media, configuration, and progress in one run dashboard:

```yaml
wandb:
  enabled: true
  project: "ltx-2-trainer"
  entity: null  # Your W&B username or team; null uses the default account
  tags: ["ltx2", "lora"]
  log_validation_videos: true
```

Use consistent validation prompts to compare checkpoints.

## 🚀 Next Steps

After training completes:

- Review your samples and refine [validation prompts and settings](configuration-reference.md#validation-content)
  to compare checkpoints and catch overfitting.
- Run your trained weights with a suitable [inference pipeline](training-modes.md#-using-trained-models-for-inference).
- Use [Troubleshooting](troubleshooting.md) to address failures, memory issues, or poor validation outputs.
- Share results or discuss your run on the [community Discord](https://discord.gg/ltxplatform).
