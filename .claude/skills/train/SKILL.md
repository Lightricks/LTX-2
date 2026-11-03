---
name: train
description: Prepare datasets or configs, launch, monitor, resume, or validate LTX training runs. Use for training, fine-tuning, LoRA creation, or individual stages of that workflow.
argument-hint: [optional source path or run name]
user-invocable: true
allowed-tools: Bash, Read, Grep, Glob, Edit, Write, Agent, AskUserQuestion, TodoWrite
---

# Train

Complete the requested stages: dataset preparation, configuration, launch, monitoring, resume, validation, or a full run.
Skip unrelated stages and stop when the requested outcome is delivered.
The trainer docs and example configs own the technical details; this skill owns the workflow.

## Read as needed

Read the relevant sections and selected example config before using them. Do not load every guide up front.

| Need | Source |
|------|--------|
| Prerequisites, installation, model compatibility | [Quick start](../../../packages/ltx-trainer/docs/quick-start.md) |
| Intent → modalities, conditioning, example config | [Training modes](../../../packages/ltx-trainer/docs/training-modes.md) |
| Metadata, paths, buckets, triggers, preprocessing and reuse | [Dataset preparation](../../../packages/ltx-trainer/docs/dataset-preparation.md) |
| Config fields, validation samples and conditions | [Configuration reference](../../../packages/ltx-trainer/docs/configuration-reference.md) |
| Scene splitting, captioner setup and CLI options | [Utility scripts](../../../packages/ltx-trainer/docs/utility-scripts.md) |
| Launch, distributed training, checkpoints and tracking | [Training guide](../../../packages/ltx-trainer/docs/training-guide.md) |
| A failed step | [Troubleshooting](../../../packages/ltx-trainer/docs/troubleshooting.md) |

Resolve repo paths from the repository root. Run trainer scripts from `packages/ltx-trainer/` with absolute config paths.

## Boundaries

- Keep generated files in one run workspace. Read source data in place or stage links/copies; preserve the originals.
  Get explicit approval before overwriting existing datasets, captions, preprocessing artifacts or trained outputs,
  or modifying files outside the run workspace. Disposable scratch created for this invocation can be regenerated.
- Present a concrete plan and get approval before captioning, encoding, autotuning or training. Honor authorization
  already given; ask again when the work would exceed it. Include missing installations and model downloads in the plan.
- Use supported configurations. If the requested behavior requires trainer code changes, explain the gap and get
  consent for that separate work.
- Report observed counts, timings, memory and losses. Label uncertain recommendations; do not promise training quality
  or infer dataset sufficiency from sample counts. Successful setup checks establish only what they exercised,
  not learning quality.

## 1. Establish the goal and inspect the environment

Use what the user already said. Clarify only unresolved choices needed for the requested stages: what should be
generated or conditioned on, LoRA versus full fine-tuning, target dimensions/duration, steps, and an optional trigger.
For a plain concept/style LoRA, clarify text-only versus starting-image use; propose `i2v_lora.yaml` for both/unsure.

Inspect the relevant source metadata/media, existing artifacts, dependencies, available disk, model components,
and training environment. Probe clip lengths for small datasets before proposing a frame bucket.
For missing captions, check viable backends, credentials and cached weights using
the utility guide; skip captioner setup when captions already exist.

Check prerequisites only for the requested stages. Writing a config or inspecting metadata does not require CUDA,
model downloads, or captioner setup. Before encoding or training, verify CUDA and required data/model files are
available; explain any missing prerequisites.

When configuring or launching a tracked run, check W&B credentials without printing secrets:
`uv run python -c "import wandb; print(bool(wandb.Api().api_key))"`.
Use its credential resolution rather than `wandb status`; an error is not evidence that tracking is unavailable.
Record whether tracking will be enabled in the plan.

If a run already exists, determine whether it is live, completed or interrupted before creating another process.
For monitoring requests, report its status without relaunching. For resume, inspect the checkpoint/state and use the
training guide's explicit `checkpoints.load_checkpoint` flow within the user's authorization.
Reuse the existing plan, config and data; perform only the work needed for the requested monitoring or resume action.

## 2. Prepare a reviewable plan and config

Use a user-selected workspace, otherwise `./projects/<run-name>/`.

When a config is needed, reuse the supplied config or copy the selected mode's example to the run workspace.
Adapt memory settings from the shipped low-VRAM example when needed, preserving the chosen modalities, conditions
and LoRA targets. Keep defaults unless the user's goal or measured hardware behavior warrants a change.

Before expensive work, run cheap checks relevant to the request: config/schema validation without model loading,
paths, dependencies, model compatibility, metadata/media readability, bucket constraints, and existing-artifact
compatibility. Reuse equivalent successful checks from an existing run where applicable.

For a new run, write a short `plan.md` and present it in chat: goal/mode, data and captioning work, model paths,
hardware, resolution/duration, steps and key config choices, validation, outputs, tracking, optional tuning,
and consequential assumptions. For a partial request, cover only the relevant work.
Give cost/time estimates only where grounded; leave unmeasured durations unknown until timings are available.
Obtain approval for this scope before expensive work.

## 3. Prepare the data

Follow the dataset and utility guides for the selected mode. Keep media paths resolvable relative to the metadata
file when staging or filtering it; verify paths before launching an encoder.

- Reuse supplied captions when complete. Otherwise use the captioner's default instruction. Unless the user explicitly
  waived preview review, show captions for up to three samples in full and wait for approval before captioning the rest.
  Apply requested changes and repeat the preview.
- Required references and masks express the user's task: use supplied inputs or a generation method the user
  requested. Ask for missing inputs rather than inventing their content or substituting a default control method.
- Set a concept trigger through `training_strategy.data.lora_trigger`; do not also insert it into raw captions.

When validation is in scope, use a small set of **training-set prompts** and **task-relevant examples**, reusing user
examples where available. Match conditioning to the mode, include the trigger when used, and describe generated audio
consistently with the captions. Keep periodic validation lightweight; plan any broader final evaluation separately.
Arrange validation of the final weights while respecting the user's chosen cadence.

## 4. Check execution as needed, then preprocess

Only where uncertainty remains, run the smallest useful execution check for the requested work: for example,
encode a sample, complete an optimizer step, or generate a validation sample. Choose checks for the actual risk,
without a fixed training-step count. For memory checks, exercise the intended batch size, dimensions/duration,
and relevant conditioning paths; one easy sample does not establish that every configured workload fits.

Keep scratch configs, latents and outputs separate, preserving relative media paths when filtering metadata.
Reset scratch checkpoint loading so checks cannot resume or overwrite the main run. Require the selected checks
to succeed before the expensive work they are meant to verify. Report what was checked and what remains untested;
do not require loss to decrease in a setup check.

Autotune only if included in the approved plan. Bound the trial count and change one runtime setting at a time,
recording measured step time and peak memory. Keep the last working configuration; leave task/quality choices fixed.

Reuse compatible artifacts or preprocess with the same config used for training. Follow the dataset guide's
manifest/staleness policy before reuse or incremental preprocessing. Use `scripts/preprocess.py <config> --check`
to check compatibility; legacy or mismatched reuse requires explicit acceptance under that policy.
Prefer a new output root for rebuilding; never silently overwrite or bypass the guard.

Reconcile artifact identities and counts against the intended training samples for all configured data sources;
a manifest match alone does not verify dataset contents. Stop if a required modality has no artifacts, including audio.
If samples or modalities are missing, diagnose and surface the discrepancy before training; let the user choose
whether to change buckets, repair data or accept dropped samples.

## 5. Launch, monitor and hand off

Launch using the training guide's command and distributed config for this run. Use the environment's managed
background execution so work and logs remain trackable. Add `--disable-progress-bars` when redirecting
trainer output or running multi-GPU; preserve logs and checkpoints across sessions.

Explain phase transitions briefly. During training, report actual step/total, recent loss, measured throughput,
memory, latest checkpoint and validation samples, and the W&B link when enabled. Measure steady-state step time
separately from setup, validation and checkpointing. For completion estimates, include remaining overhead where
measured; disclose unknown or excluded costs instead of presenting a steps-only estimate as completion time.

On failure, diagnose using the troubleshooting guide and preserve artifacts. Fix routine issues within the agreed
scope; stop when the same failure repeats without new evidence. Describe the proposed recovery and seek approval
if it changes the agreed scope or risks existing work.

For a training run, verify completion and the final checkpoint. Surface validation samples as training-set prompts
or task-relevant examples, identifying whether they use the final weights. Report any planned but missing validation
as incomplete. For separate inference, use the documented matching
[pipeline](../../../packages/ltx-pipelines/README.md) within the approved scope.

For a training run, write a concise `outputs/run-summary.md` with the goal, data, final config and commit,
measured statistics, checkpoint/sample/log paths, tracking link, exact launch command and resume instructions
(including available state).
For partial requests, hand off the requested artifacts or status without starting further stages.
Present results without inventing a quality verdict or demanding one.
