# Initial formal verification models

Run `bash formal/check.sh /path/to/tla2tools.jar` from any directory with Lean 4
and Java installed. The script compiles both Lean files and checks
five bounded TLA+ specifications in seven configurations. The configuration files
bound workers, rollout IDs, policy versions, scores, and
experiments; passing TLC means all states *within those bounds* were checked.
To run the Python refinement and regression tests in the same command, use
`PYTHON=.venv/bin/python bash formal/check.sh /path/to/tla2tools.jar --with-python`.
The Python mode requires the project's test dependencies; optional PyTorch and
Hypothesis checks skip when those packages are unavailable.
The [formal verification workflow](../.github/workflows/formal-verification.yml)
runs the seven TLC configurations and both Lean files on relevant pull requests. It pins
Lean 4.15.0 and checks the SHA-256 of the TLA+ v1.7.4 tool JAR; the existing
Python CI job runs the regression and refinement tests.
The root Lake manifest declares an empty dependency set for the Lean setup
action; the proof files are checked directly by `formal/check.sh`.

| Priority | Artifact | Python implementation | Checked contract |
| --- | --- | --- | --- |
| 1 | `tla/RolloutControl.tla` | `training/distributed_rollouts.py`, `training/async_rollouts.py` | Queue capacity, unique admitted IDs, accepted-counter equality, bounded lag, seen-ID admission, artifact availability, lease fencing at admission, and rejection of bad hashes. |
| 2 | `tla/AsyncRuntime.tla` | `training/async_runtime.py` | Initial publication precedes worker start; each later version is published before visibility; updates are bounded. A separate fair, failure-free configuration checks eventual completion. A Python regression also checks that failed publication does not advance the visible version. |
| 3 | `lean/Objective.lean` | `training/objectives.py`, `training/rl_losses.py` | Log-ratio clipping bounds and idempotence, clipped-surrogate behavior including zero advantage, zero-mask behavior, and exact cross-multiplied group centering. |
| 4 | `tla/AutoResearch.tla`, `tla/CheckpointSwap.tla`, `tla/ResearchCommit.tla` | `training/auto_research/experiment_loop.py`, `experiment_tracker.py`, `checkpoint_manager.py` | Only evaluated improvements become best in either maximize or minimize mode, crashed records cannot become best, interrupted directory swaps recover, and provisional checkpoints reconcile with the experiment log after restart. |
| 5 | `lean/Reward.lean` | `rewards/multi_objective_reward.py`, `core/trajectory.py` | Final score clamp and idempotence, failed/zero-weight component behavior, and a weighted-sum bound for normalized, bounded components. |

The research model ranges over bounded integer scores, including negative
values. The Python record now
rejects NaN and infinite objective values, the loop records such evaluations
as crashes before checkpoint promotion, and replay skips legacy invalid
records. This makes the model's finite-score assumption explicit at the
implementation boundary.

## Rollout counterexamples resolved

The first model revision found `Register; BeginSubmit; Register; FinishSubmit`:
the old generation's submission could complete after replacement. The
coordinator now invokes a synchronous admission check immediately before its
queue append. `FencingSafety` is included in the passing TLC configuration,
and a Python regression test holds a submission behind queue backpressure
while the worker is replaced.

The first model revision also found `Register; Publish; BeginSubmit;
FinishSubmit` with artifact capacity one. The old artifact was pruned and the
hash comparison was skipped. Strict artifact submissions now reject an
unavailable descriptor. `ArtifactSafety` is included in the passing TLC
configuration and has a Python regression test.

`tests/unit/test_formal_rollout_refinement.py` runs every four-operation trace
over registration, publication, heartbeat, submission, and consumption in a
small Python instance. It checks the queue, policy-lag, artifact, and counter
invariants after every step. This found a timeout boundary where the queue
admitted a rollout but the control plane reported a timeout and omitted its
accepted counter. The coordinator now resolves the child operation after a
timeout cancellation before returning either its committed result or a
timeout. The trace suite is a refinement check for these bounded operations,
not a proof that every Python execution refines the TLA+ model.

The model also checks that the accepted counter equals the number of admitted
rollout IDs. A Python regression simulates cancellation immediately after
admission; the control-plane counter now updates in the coordinator's
synchronous admission turn, keeping queue and counter state aligned.

The TLA+ models abstract payloads, network delivery, clock values, and
optimizer state. `CheckpointSwap.tla` checks recovery when the old best
directory was moved but the new one was not installed. `ResearchCommit.tla`
checks reconciliation after a crash between checkpoint installation and
experiment log append. The JSONL append now fsyncs the file and, on first
creation on POSIX, its parent directory before the checkpoint is finalized.
These models do not establish power-loss durability of model-weight files or
every supported filesystem. `Objective.lean` and `Reward.lean`
use exact integers. Their theorems do not establish IEEE floating-point, PyTorch
autograd, or GPU-kernel behavior. `tests/unit/test_formal_numeric_refinement.py`
exhaustively compares small score, weight, failure, clipping, and group domains
against the Python implementations. Larger tensors, GPU kernels, and optimizer
updates remain outside these checks.

`AsyncRuntimeProgress.cfg` checks that a run reaches `done` when no operation
fails and every enabled phase eventually executes. Its `Learn` step assumes a
batch becomes available and the learner finishes; `PublishUpdate` assumes the
publisher finishes. It does not claim progress if a callback hangs, a worker
fails, or the coordinator cannot supply a batch.

For reward composition, Python now rejects negative or non-finite component
weights and treats a non-finite component score as a failed component with zero
contribution. The Lean weighted-sum theorem still assumes each successful
component score lies in `[0, 1]`; the Python composer clamps the final score
but does not impose that range on every component.
