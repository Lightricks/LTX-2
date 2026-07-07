# MGPU Controller

**Source**: [`multigpu/controller.py`](../../src/ltx_pipelines/multigpu/controller.py), [`multigpu/runner.py`](../../src/ltx_pipelines/multigpu/runner.py)

The controller is a persistent, one-job-at-a-time GPU fleet. It spawns one worker
process per GPU, runs a user-defined **runner** in SPMD lockstep, and streams
results back.

## Public classes

### `MGPUController`

```python
MGPUController(
    runner_cls: type[MGPURunner],
    *,
    num_gpus: int | None = None,      # GPUs 0..num_gpus-1 (default: all visible)
    devices: Sequence[int] | None = None,  # place on specific physical GPUs, e.g. [2, 3]
    logs_specs: LogsSpecs | None = None,
)
```

`num_gpus` and `devices` are mutually exclusive. `devices=[2, 3]` puts rank r on
`cuda:devices[r]`, so two controllers can share one machine on disjoint GPU sets.

Lifecycle:

| Method | What it does |
| ------ | ------------ |
| `start(*, timeout=30min, **setup_kwargs)` | Spawn the fleet, run `setup(**setup_kwargs)` on every rank, block until all report ready. `timeout` bounds NCCL init + CUDA init + `setup()`; it must exceed the slowest model load. |
| `stream(*, timeout=None, **kwargs) -> Stream` | Dispatch one job and return **immediately**. Iterate the returned `Stream` to collect. |
| `shutdown(*, graceful_timeout=60.0)` | Tear down the fleet; also force-terminates it — safe to call from another thread to recover a job that cannot be drained. |
| `is_alive` (property) | True while the fleet is up and unpoisoned. |

### `MGPURunner`

`MGPURunner` is the abstract base class implemented per pipeline. The controller
ships the subclass to every worker by value (a runner defined in `__main__` or a test
module is supported), builds one instance per worker, injects the NCCL groups, calls
`setup()` once, then invokes the instance per job.

```python
class MyRunner(MGPURunner):
    @torch.inference_mode()
    def setup(self, *, checkpoint_path: str, ...) -> None:
        # build the pipeline + swap in MGPU builders (see pipeline-setup.md)
        ...

    @torch.inference_mode()
    def __call__(self, *, prompt: str, ...) -> Iterator[...]:
        video, audio = self._pipeline(...)
        yield output_path   # __call__ MUST be a generator (use `yield`, even once)
```

- `setup()` and `__call__()` run on **every** rank. `self.groups` gives the
  per-component `NCCLGroups` (`gemma_group`, `transformer_group`, `vae_group`).
- The framework does **not** apply inference mode — decorate `setup`/`__call__` explicitly.

## Usage

```python
from ltx_pipelines.multigpu import MGPUController

controller = MGPUController(MyRunner, num_gpus=8)
controller.start(checkpoint_path="...", gemma_root="...")   # setup kwargs
stream = controller.stream(prompt="a cat", seed=42)
try:
    for item in stream:      # one element per yield, as it arrives (NOT gathered across ranks)
        show(item)
finally:
    stream.drain()           # free the controller even on early exit
controller.shutdown()
```

### Passing tensors

Tensors are transparent:

- **Inputs.** Pass them as **top-level** kwargs (`stream(latent=t, steps=30)`) and
  the relay (rank 0) broadcasts them to every rank over NCCL — `__call__` receives
  them already on the local GPU. An input tensor nested inside a list/dict kwarg is
  **not** broadcast; it falls back to the (slower) pickle path.
- **Outputs.** Yield tensors back (including nested inside a dict) and they return
  via the result queue by shared memory / CUDA IPC — no pickling, regardless of
  nesting.

Everything else must be picklable and small.

## Contract and limitations

- **Single machine only.** `MASTER_ADDR=localhost`, `RANK == LOCAL_RANK`, one rank per GPU.
- **One job at a time.** No job queue, no pipelining. Consume the `Stream` to the
  end before the next `stream()`. Abandoning it is **not** cleaned up: the next
  `stream()` raises `ControllerBusyError` until `stream.drain()` or `shutdown()` is
  called. The recommended pattern is `try: ... finally: stream.drain()`.
- **SPMD lockstep.** Yields are forwarded individually (in result-queue order), not
  gathered. Only per-rank terminals are collected to end the stream.
- **Thread ownership (baton-lock).** Any thread may call `stream()`. Each job
  belongs to its **dispatching** thread — only that
  thread may iterate or `drain()` its `Stream` (enforced in `Stream.__next__`). A
  single lock guards only the in-flight check-and-set: among concurrent `stream()`
  callers one proceeds and the rest raise `ControllerBusyError`.

## Error handling

| Situation | Outcome |
| --------- | ------- |
| Runner raises an **unexpected** exception | **Fatal** — a desynced NCCL collective cannot be unwound. The controller is poisoned and a new one must be constructed. |
| Runner raises `RunnerError` (or `ValueError`, auto-converted) **identically on every rank** | Recoverable. Iterating the `Stream` re-raises `SymmetricRunnerError`; the fleet survives — fix the input and retry. Raise it outside any collective (e.g. validating broadcast kwargs before the first one). |
| Some ranks raise `RunnerError`, others finish clean | `AsymmetricRunnerError` — surfaced prominently (a latent hang risk), but does not terminate the fleet. |
| Worker death / exceeded per-job `timeout` | Surfaced when the `Stream` is next iterated; the controller is poisoned. |

The public API (`from ltx_pipelines.multigpu import ...`) exports `MGPUController`,
`MGPURunner`, `Stream`, `RunnerError`, `SymmetricRunnerError`,
`AsymmetricRunnerError`, `ControllerBusyError`, and `NCCLGroups`.
