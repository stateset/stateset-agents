# Lifecycles

Several long-running things in this package move through a small set of
statuses. Four of them are declared as data in
[`stateset_agents/lifecycles.py`](../stateset_agents/lifecycles.py), and the
modules that write those statuses take their constants from the declaration.

```bash
stateset-agents lifecycles                 # every declared lifecycle
stateset-agents lifecycles fanout-run      # one of them
stateset-agents lifecycles --json          # the same data for a program or an agent
```

For each status the command shows what it means, whether it is **active** (keep
polling), **terminal** (stop polling) or **settled** (terminal and will never
change), and which statuses can follow it.

| Lifecycle | What it is | Statuses | Terminal |
| --- | --- | --- | --- |
| `remote-job` | A job submitted with `train-remote` | pending, running, succeeded, failed, cancelled | succeeded, failed, cancelled |
| `fanout-run` | One run of a fan-out | pending, running, succeeded, failed, cancelled | succeeded, failed, cancelled |
| `finetune-run` | A local run in a fine-tune workspace | prepared, queued, starting, running, interrupted, succeeded, failed, cancelled | interrupted, succeeded, failed, cancelled |
| `gateway-training-job` | A job submitted to `POST /training` | queued, running, cancelling, completed, failed, cancelled | completed, failed, cancelled |

Two details differ between them and are easy to get wrong:

- The gateway says `completed`; everything else says `succeeded`.
- A `finetune-run` that is `interrupted` is terminal (its worker is gone) but
  not settled: it can be resumed. A `succeeded` run can still become `failed`
  or `cancelled` until its artifacts confirm the success.

## From Python

```python
from stateset_agents.lifecycles import FANOUT_RUN, get_lifecycle

FANOUT_RUN.is_terminal("running")            # False
FANOUT_RUN.next_states("running")            # pending, succeeded, failed, cancelled
get_lifecycle("remote-job").require("cancelled", "failed")   # ValueError
```

`require` raises a `ValueError` that names the statuses the current one can
become. Declaring a lifecycle does not add enforcement to existing code: the
fine-tune workspace already refused illegal moves and still does, by the same
rules; the other writers are unchanged.

## Adding or changing a status

Change it in `stateset_agents/lifecycles.py`: add the status with its meaning
and its transitions. `tests/unit/test_lifecycles.py` fails if a module writes a
status its lifecycle does not declare, if a lifecycle has an unreachable status
or an active status with no way out, or if the fine-tune declaration stops
matching `finetune_workspace.transition_state`.

## State machines that are not declared yet

These also carry state through string literals, enums or flags. They are listed
so the next one to cause a bug has a known home; none was changed.

| Where | States | Note |
| --- | --- | --- |
| `fanout/journal.py` attempts, `fanout/backend.py` process state | per-attempt outcomes; `running`, `exited`, `timed_out`, `lost` | Separate from the run status above |
| `finetune_validation.py` comparisons | queued, running, submitted, submission_uncertain, stale, unreadable, succeeded, failed, interrupted | Has its own `ACTIVE_COMPARISON_STATUSES` |
| `finetune_evaluation_worker.py` evaluations | queued, running, succeeded, cancelled | |
| `api/routers/v1.py` training jobs | starting, running, cancelled | In-memory only; reports `starting` while storing `running` |
| `api/routers/training_lab.py` `ExperimentStatus` | enum | Simulation only |
| `api/models.py` `TrainingStatus` | enum | Deprecated module |
| `training/advanced_training_models.py` `TrainingStatus` | enum | `PAUSED` is never set |
| `training/experiment_tracking.py` | running plus a private terminal set | |
| `training/auto_research` experiments, `training/hpo` trials | keep, discard, crash; success, failed | A crashed HPO trial is recorded as success |
| `remote/serve_session.py`, `remote/chat_session.py` | implied by `pod_id` and `_closed` | No status field at all |
| `remote/river.py` jobs | maps provider states onto `JobStatus` | |
| `core/async_pool.py` `PoolState`, `ResourceState` | enums | A failed resource is reset to idle on release |
| `core/error_handling.py` `CircuitBreakerState`, `api/resilience.py` `CircuitState` | closed, open, half-open | Two implementations |
| `api/shutdown.py` `ShutdownPhase` | enum | |
| `core/environment_base.py` `EpisodeStatus`; refund and industry environments | episode and case statuses | |
| `experimental/long_term_planning.py` `PlanStatus` | enum | Step numbers shown differ from the indices used to update |
| `experimental/weight_space` | export verification: unverified, passed, rejected, failed; recipe: accepted, rejected, baseline_unmet, export_rejected | |
| `tui.py` and `finetune_evaluation_tui.py` | overlay and mode strings | Selection mode sets are repeated per screen |
