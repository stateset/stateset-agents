# Remote Fine-Tune Executor — Design

**Date:** 2026-07-30
**Status:** Approved, pending implementation plan

## Problem

The improvement loop (`ingest` → `improve` → fine-tune) stops at the last
step for anyone without a GPU. `improve` is CPU-only and cheap; the SFT that
consumes `curated.jsonl` is not. Today a user who has just produced a curated
dataset has no supported path to a trained adapter unless they own hardware.

Hosted training services (River, and the compute-rental tier of
Modal/RunPod/Baseten) exist precisely because this gap is common. Closing it
with compute rental — rather than a hosted training API — keeps the
framework's trainers, rewards, and multi-turn credit assignment in play
instead of delegating them to a vendor's loss functions.

## Goal

Let a user run the existing SFT job on rented GPUs with one command, without
changing how the job itself works.

```bash
stateset-agents improve run --transcripts transcripts/ --output improved/
stateset-agents train remote \
    --provider modal \
    --dataset improved/curated.jsonl \
    --base-model Qwen/Qwen3.5-0.8B \
    --output-dir outputs/sft_v1
```

## Non-goals

Explicitly out of scope for v1, to be revisited only with evidence of demand:

- **RL rollouts.** Generation-heavy multi-turn RL needs persistent vLLM
  workers, rollout/train interleaving, and checkpoint streaming. That is a
  different, stateful design.
- **RunPod.** v2. Note that `deployment/runpod_deployment.py` already exists
  and is a *different, heavier* GRPO-job abstraction; it is left untouched
  rather than partially merged into this one.
- **Baseten.** Serving, not training. Belongs next to `serving_artifacts.py`
  and the existing Helm/GKE paths.
- **River.** A hosted training *API*, not compute rental. Integrating it would
  bypass the framework's own trainers. If ever pursued it belongs as a loss
  backend under a trainer, not as a peer of these executors.
- Multi-node, multi-GPU, and hyperparameter sweeps.

## Architecture

A new top-level package. This is job orchestration, not an algorithm, so it
does **not** go under `training/` — that directory is already 45 files and
adding to it worsens the discoverability problem it has.

```
stateset_agents/remote/
  __init__.py     public exports
  job.py          RemoteJobSpec, JobHandle, JobStatus, RemoteJobResult
  executor.py     RemoteExecutor ABC, RemoteExecutionError
  local.py        LocalExecutor
  modal.py        ModalExecutor
  registry.py     provider name -> executor class
stateset_agents/cli_remote.py
```

`cli_remote.py` follows the established sub-app pattern: define
`@app.command(...)` against the shared Typer app and import the module at the
bottom of `cli.py`, exactly as `cli_improve` / `cli_ingest` do.

### Data flow

```
LOCAL                                   REMOTE
-----                                   ------
improve/ -> curated.jsonl
                |
        RemoteJobSpec (validated)
                |
        executor.submit()  ------------> provision GPU
                                         pip install stateset-agents[training]==<pin>
                                         upload curated.jsonl
                                         run scripts/sft_from_curated.py
                                                |
        executor.status()/logs() <------- poll
        executor.fetch(dest)   <-------- adapter directory
                |
        outputs/sft_v1/  (drop-in for AgentConfig.peft_path)
```

## Components

### `RemoteJobSpec` (`job.py`)

The provider-agnostic contract, derived **exactly** from
`scripts/sft_from_curated.py`'s existing argparse surface:

| Field | Default | Source |
|---|---|---|
| `dataset` | required | `--dataset` |
| `base_model` | required | `--base-model` |
| `output_dir` | `outputs/sft_v1` | `--output-dir` |
| `num_epochs` | 3 | `--num-epochs` |
| `lora_r` | 16 | `--lora-r` |
| `lora_alpha` | 32 | `--lora-alpha` |
| `learning_rate` | 2e-5 | `--learning-rate` |
| `max_length` | 1024 | `--max-length` |
| `per_device_batch_size` | 2 | `--per-device-batch-size` |
| `gradient_accumulation_steps` | 4 | `--gradient-accumulation-steps` |
| `dry_run` | False | `--dry-run` |

Plus provider-agnostic resource fields: `gpu` (e.g. `"A10G"`), `timeout_s`,
`package_version` (the pinned `stateset-agents` version installed remotely).

**No new training knobs.** If a user needs a capability the script lacks, the
fix is a change to the script, not to this layer. The spec serializes to JSON
for submission and round-trips losslessly.

Secrets are never spec fields. `HF_TOKEN` and provider credentials come from
the environment at submit time and are never serialized or logged.

### `RemoteExecutor` (`executor.py`)

Five methods, stateless and poll-based, matching the "one job, retry = rerun"
model:

```python
submit(spec) -> JobHandle
status(handle) -> JobStatus      # PENDING | RUNNING | SUCCEEDED | FAILED | CANCELLED
logs(handle) -> Iterator[str]
fetch(handle, dest) -> Path
cancel(handle) -> None
```

`JobHandle` is an opaque, serializable identifier (provider name + provider
job id) so a job can be polled from a later process. `JobStatus` is the enum
above. `RemoteJobResult` bundles the terminal status, the fetched adapter
path, and the captured logs, and is what the CLI renders.

`RemoteExecutionError` subclasses `StateSetError`, consistent with the unified
hierarchy. Provider SDK exceptions are wrapped via the existing
`wrap_exception` helper so the `.cause` chain is preserved.

### `LocalExecutor` (`local.py`)

Runs `sft_from_curated.py` in a subprocess on the current machine.

This ships **first**, before Modal. It is the reference implementation that
keeps the interface honest — an abstraction with exactly one implementation
tends to become that implementation with extra indirection. It is also fully
testable in CI at zero cloud cost, and independently useful to anyone who does
own a GPU but wants the uniform command.

### `ModalExecutor` (`modal.py`)

Provisions a Modal function with the requested GPU, installs
`stateset-agents[training]==<package_version>` from PyPI, mounts the dataset,
and runs the same entrypoint.

**Artifacts ship, not code.** No working-tree sync: pinning to a published
version is what keeps the remote environment reproducible and is the single
biggest reason executors of this kind rot.

The `modal` SDK is imported lazily behind a `MODAL_AVAILABLE` flag, mirroring
the existing `RUNPOD_AVAILABLE` pattern in `deployment/runpod_deployment.py`.

### Extras

```toml
remote = ["stateset-agents[training]"]
modal  = ["modal>=0.63"]
```

## Dry-run parity

`sft_from_curated.py` already prints its training plan and exits 0 when
`--dry-run` is passed or no GPU is detected. That path is reused verbatim, so
a local dry-run and a remote dry-run produce identical plans. This makes the
end-to-end path exercisable on CPU-only CI.

## Error handling

| Failure | Behavior |
|---|---|
| Missing/unreadable dataset | Validated at spec construction, before submit |
| `modal` not installed | `RemoteExecutionError` naming the `[modal]` extra |
| Missing provider credentials | `RemoteExecutionError`, checked before provisioning |
| Job fails remotely | Status `FAILED`; `logs()` still retrievable; non-zero exit |
| Timeout | Job cancelled, status `FAILED`, cause recorded |
| `fetch` before completion | `RemoteExecutionError` |

No automatic retries. A failed job is rerun by the user; this keeps the
executor stateless and avoids silently burning GPU budget.

## Testing

TDD order:

1. `RemoteJobSpec` construction, validation, and JSON round-trip.
2. `LocalExecutor` end-to-end against a tiny stub dataset in `--dry-run`,
   asserting the printed plan — real integration coverage on CPU-only CI.
3. `registry` resolution and unknown-provider errors.
4. `ModalExecutor` against a mocked `modal` SDK, mirroring the mocked
   peft/transformers approach already used in `test_serving_artifacts.py`.
5. CLI wiring: argument parsing, provider selection, exit codes.

Modal's real network path is **not** covered by CI. It is manually verified
and marked as such; the test suite will not imply otherwise.

## Risks

- **Executor rot.** Provider SDKs and the torch/transformers/vLLM pinning
  matrix drift. Mitigated by shipping artifacts against a pinned published
  package rather than syncing the working tree, and by keeping the job surface
  frozen to one job type.
- **Untestable without spend.** Accepted and bounded: `LocalExecutor` carries
  the interface contract in CI; Modal's transport layer is the only manually
  verified part.
- **Scope creep toward the full loop.** The non-goals above are the guardrail.
  Requests for RL rollouts should produce a new spec, not an extension here.
