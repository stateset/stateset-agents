# River AI provider

River executes remote sampling, autograd, and optimizer operations. StateSet
supplies task definitions, rewards, training batches, checkpoint selection, and
experiment artifacts. The optional SDK is pinned to the reviewed 0.11 API series:

```bash
# River requires Python >=3.12; the rest of StateSet still supports Python >=3.10.
pip install -e ".[river]"
export RIVER_API_KEY=rv_...
```

## Supported workflows

- `train-remote --provider river`: supervised fine-tuning with assistant-only loss.
- `flywheel --provider river --algorithm sft`: rejection-sampling self-improvement.
- `flywheel --provider river --algorithm cispo` (also `ppo`,
  `importance_sampling`): synchronous group-relative RL, with one logical
  optimizer update per round and explicit microbatch accumulation.
- `examples/river_refund_rl.py`: StateSet environments inside River's native
  `rl.RolloutEngine` / `rl.AsyncTrainer`, including executed sandbox actions,
  native recovery, fixed-checkpoint validation, and a separate final test set.

Results remain hosted `river://` checkpoints. `river_checkpoint.json` is a
pointer, not downloaded adapter weights. Local `serve --checkpoint` cannot load
it. River also offers dedicated OpenAI-compatible deployments for authorized
team accounts; creating a deployment is a separate billable operation.

### Continue supervised training from a checkpoint

River SFT accepts an explicit starting checkpoint in the provider options:

```python
from pathlib import Path

from stateset_agents.remote.job import RemoteJobSpec

spec = RemoteJobSpec(
    dataset=Path("boundary-train.jsonl"),
    base_model="Qwen/Qwen3.5-9B",
    output_dir=Path("outputs/boundary-sft"),
    num_epochs=1,
    lora_r=16,
    seed=42,
    harvest={"checkpoint": "outputs/previous-sft", "shuffle": True},
)
```

The reference may be a `river://` URI or a directory containing
`river_checkpoint.json`. StateSet resolves it before provider calls, checks
declared base-model and LoRA-rank compatibility, and records the resolved URI
with the captured seed and shuffle settings in both output artifacts. Later
changes to the pointer do not redirect training or rewrite its lineage.
This loads starting weights for a new SFT run; `parent_adapter` separately
records lineage and does not load weights. Remote compatibility and checkpoint
availability are still checked by River when the model is created.

## Preflight before a run

Check the local Python version, supported SDK, and presence of `RIVER_API_KEY`
without constructing a client or contacting River:

```bash
stateset-agents river-preflight
```

Add `--live` to query the account's current model list. Repeat `--base-model`
to require access to specific IDs; omit it to list the account's advertised
models without selecting one:

```bash
stateset-agents river-preflight --live \
  --base-model Qwen/Qwen3.6-35B-A3B-FP8 \
  --base-model Qwen/Qwen3.8-27B-FP8 \
  --base-model deepseek-ai/DeepSeek-V4.1-Flash \
  --base-model zai-org/GLM-5.3-Flash \
  --timeout-seconds 15 --output outputs/river-preflight.json
```

Both modes emit versioned JSON. Local mode reports account access as
`not_checked` and requested models as `null`; it checks whether credentials are
configured, not whether they work. Live mode performs one capability query,
disables SDK retries, and closes the client. Its timeout applies to the SDK
request, not the entire command. Neither mode loads tokenizers, creates training
sessions, or samples tokens. Failed local prerequisites prevent a live query.

Exit codes are `0` for passed requested checks, `1` for failed checks, and `2`
for invalid options or report-write errors. `--output` requires a new file and
saves failed check reports too. A passing report does not verify training
correctness, model quality, capacity, funding, or hosted chat/streaming support.

The same report is available to Python callers:

```python
from stateset_agents.remote.river_runtime import river_preflight

report = river_preflight(["Qwen/Qwen3.6-35B-A3B-FP8"], live=True)
if not report["passed"]:
    raise RuntimeError(report["issues"])
```

## Model access and tokenizer isolation

Live SFT, harvest, synchronous RL, and native RL/evaluation runs check
`client.get_capabilities()` before creating a session. SFT also checks before
loading a tokenizer; native runs check before loading a renderer. Missing,
malformed, or failed capability responses stop the run. Access is checked anew
for each job; a previous success does not authorize a later job. Private models
absent from our example catalog are accepted when the account advertises them.
Re-publishing a fully completed RL run from validated local state remains
offline and does not construct a client or perform an access check.

The following exact IDs were advertised by our October 8, 2026 account canary:
`Qwen/Qwen3.6-35B-A3B-FP8`, `Qwen/Qwen3.8-27B-FP8`,
`deepseek-ai/DeepSeek-V4.1-Flash`, and `zai-org/GLM-5.3-Flash`.
This is access evidence, not proof of a successful training run, available
capacity, funding, or support for both hosted chat modes. River documents
[runtime capabilities as authoritative](https://docs.river.ai/guides/models/).

Automatically loaded SFT tokenizers are cached separately by model ID. An
explicitly injected tokenizer remains the caller's responsibility and is used
as supplied. Dry runs do not construct or contact a River client; SFT previews
still need a tokenizer and may download its files unless one is supplied.

Native rollout iteration waits for each requested reply without prefetching.
On cancellation it drains the active operation and closes the producer before
provider resources are released. This does not forcibly interrupt a hung SDK
call; configure timeouts in the underlying client.

## Client lifecycle

`RiverExecutor.submit()` runs synchronously. It closes each client it creates
after SFT, harvest, or RL completes, fails, or is interrupted, after the run's
session has exited. A later submission creates a fresh client; completed RL
resumes and dry runs do not create one. Injected `RiverExecutor(client=...)`
clients remain the caller's responsibility to close.

One executor accepts one active submission at a time. Overlapping or reentrant
submissions fail immediately; use separate executors for concurrent jobs.
`wait()`, `status()`, and artifact fetching remain usable after client cleanup.

If client cleanup fails during another error, the original error propagates
and a sanitized cleanup diagnostic is retained in the job logs. If the job
returned normally, cleanup failure raises `RemoteExecutionError` with
`context.details["stage"] == "client_cleanup"` and the retained `job_id`.
The job keeps its training status and any completed checkpoint remains
fetchable. Inspect it before resubmitting to avoid repeating completed work.
Closing a client can wait for SDK operations; this is not a hard cancellation
deadline or confirmation that failed cleanup reclaimed all resources.

### Interrupted synchronous RL runs

When synchronous RL receives `KeyboardInterrupt` or task cancellation, it marks
the job and cost record `cancelled` and attempts to publish a cancelled
`rl_report.json`. The report reloads validated `rl_state.json`; an optimizer
update or checkpoint save without a matching local commit does not advance
reported progress. Existing token reservations remain charged to the run.

Resume with the same configuration and `--resume`. The next session restores
the last committed optimizer checkpoint and executes only remaining rounds.
If all rounds were committed before the interruption, final publication can
finish offline. This does not undo an in-flight provider operation or refund
its usage; interrupted work after the last commit may need to run again.

Reporting and ledger errors retain the original interruption. Diagnostics
identify failed writes without including arbitrary exception text. If a report
cannot be written, its durable status may remain stale; the in-memory job is
still cancelled. Missing or corrupt committed state blocks resume rather than
silently restarting training.

## RL configuration

```bash
stateset-agents flywheel --provider river --algorithm cispo \
  --base-model Qwen/Qwen3.5-9B \
  --harvest-prompts train.json --eval-prompts validation.json \
  --output-root outputs/river-rl --rounds 10 --best-of 8 \
  --learning-rate 1e-5 --lora-r 16 --seed 42 --repeats 3 \
  --normalization token --microbatch-size 8 \
  --max-generated-tokens 1000000
```

`RiverRLConfig` validates the same options for programmatic callers through
`RemoteJobSpec.harvest`. Unknown knobs, invalid losses, nonfinite values,
nonpositive rounds, and group sizes below two fail before model creation.

The CLI exposes learning rate, rank, seed, temperature, top-p, generation and
validation lengths, loss clipping, gradient clipping, normalization, truncation,
microbatch size, and generated-token budget. RL rejects SFT-only options such as
`--generations`, teacher harvesting, hardware selection, and epoch counts.
`--repeats` runs independent seeds in separate directories; `--resume` resumes a
single run. Repeating with a seed improves reproducibility but does not guarantee
bit-identical server execution.

### Scoring

Every task needs objective assertions, a configured judge, or a custom scorer.
Empty assertions are never sufficient to claim success. Judge-only tasks must
include a finite `min_judge_score`. A missing, failed, or nonfinite judge aborts
RL; it does not pass the task or become a zero reward. Training and validation
use the same scoring contract. Exact overlapping train/validation prompts are
rejected; users must still prevent semantic duplicates and reserve a separate
test set for final claims. NSR requires an explicit custom scorer.

The default reward combines assertion coverage, a completion bonus, and a
forbidden-content penalty. This is useful for controlled tasks, but is still a
text verifier. For actual business outcomes, execute actions in an environment.

Programmatic reward integration:

```python
from stateset_agents.remote.river import RiverExecutor
from stateset_agents.remote.river_rl import reward_function_scorer

executor = RiverExecutor(
    rl_scorer=reward_function_scorer(my_reward_function, threshold=0.8),
    rl_scorer_id="refund-outcomes-v1-threshold-0.8",
)
```

A custom scorer can instead return `RLScore(reward, passed, components)` directly.
It receives `(task, response)`; episode responses are lists of assistant turns.
The versioned scorer ID is required for resume compatibility. Run the synchronous
executor outside an active async event loop, or put it in a worker thread.
The CLI's `--reward NAME --reward-threshold VALUE` uses StateSet domain rewards.

### Rollout and update integrity

- Sampling supplies exact rendered prompt IDs and preserves response IDs,
  original logprobs, stop reasons, policy IDs when supplied, and reward components.
- SDK samples must declare `token_data_is_exact=True`. Invalid token IDs,
  missing tokens, mismatched arrays, and nonfinite/positive logprobs fail closed.
- Default `--truncation drop_group` excludes the entire group before constructing
  its baseline. `error` aborts instead. Infrastructure failures never become
  zero task rewards.
- The synchronous driver rejects mixed known behavior policies and uses
  `expected_policy_id` when the service exposes the committed policy.
- Temperature and top-p default to 1.0 for RL; top-k filtering is disabled.
  Changing the sampling distribution requires understanding the provider's
  logprob semantics.
- `token` divides advantages by all retained nonzero-advantage response tokens;
  `sequence` gives equal weight to each retained assistant span (each turn in an
  episode); `sum` explicitly preserves token-summed weighting. Zero-variance
  groups and zero-advantage spans are excluded from these denominators.
- Normalization happens once per logical update, before microbatching. Backward
  calls accumulate with `zero_out=False` after the first microbatch. An optimizer
  call is submitted only after all backward calls succeed with finite losses.

### Selection, recovery, and budgets

The baseline is saved before training. Validation chooses the best checkpoint;
ties retain the earlier checkpoint. A regressed final round cannot replace it.
With no validation, the most recent completed update is selected, with no claim
of measured improvement. `eval_results.json` describes the selected checkpoint.

Each completed round commits a `mode="training"` checkpoint with optimizer state
and `rl_state.json`. Recovery recreates the model in a fresh session from that
checkpoint, abandoning any uncertain update in the failed session. Transient
failures receive bounded retries. Explicit `--resume` validates data, model,
scorer, and configuration fingerprints. Use the original total round count when
resuming. An output directory is locked against concurrent drivers.

The starting checkpoint fingerprint uses the resolved `river://` URI, including
when `adapter_dir` names a local pointer directory. Changing the pointer's URI
rejects resume before provider calls; moving a pointer without changing its URI
preserves identity. Older runs fingerprinted only the pointer path and cannot be
safely resumed under this rule; preserve them and use a new output directory.
Base-model runs and runs started from direct URIs retain their fingerprint format.
Pointer loading rejects duplicate fields, invalid URIs, and conflicting provider,
base-model, or training LoRA-rank metadata. Legacy pointers without that metadata
remain supported; compatibility of their remote weights is unverified locally.

Recovery also checks that saved round history is contiguous, update counts and
losses agree with committed backward metrics, checkpoint URIs are present, and
validation selects the recorded best round with the earliest-round tie rule.
The selected evaluation must contain the expected number of outcomes and agree
with its pass count. These checks run before opening a recovery session and
before committing new state; a damaged record cannot silently fall back to the
initial model or declare unfinished rounds complete. Missing state is rejected
when retained checkpoint or progress artifacts show that a commit already
existed. A failure before the first commit may resume using its existing usage
ledger. These are consistency checks on local artifacts, not authentication of
provider checkpoint contents or protection against coordinated artifact edits.

Artifacts:

| File | Purpose |
| --- | --- |
| `rl_state.json` | Atomic committed progress and training checkpoint |
| `rl_report.json` | Progress, selected checkpoint, per-round metrics, failure status |
| `rl_usage.json` | Durable generated-token usage/reservations, including retries |
| `rollouts/round-N.json` | Exact sampled records, scores, exclusions, and weighting |
| `eval_results.json` | Validation results for the selected checkpoint |
| `river_checkpoint.json` | Selected inference checkpoint pointer |
| `stateset_manifest.json` | Dataset and model provenance |
| `rl_repeats_report.json` | Per-seed results for a repeated campaign |

For the generic SFT executor, the dataset path, byte SHA-256 and nonblank
source-line count are captured from the same bytes parsed before client checks
and tokenization. Fetch retains that identity after source edits or deletion,
instead of hashing the current file. This is source-file lineage; the effective
batch can contain fewer examples after filtering/rendering. It does not attest
provider training, replace independent data review, or change the native RL
study's separate manifest protocol.

`--max-generated-tokens` reserves each request's worst-case generated tokens
before submitting it. Complete responses with exact token metadata settle to
actual counts; uncertain requests, incomplete responses, and inexact token
metadata keep their full reservation across retries and restarts. A new run
initializes `rl_usage.json` before opening its first provider session. Resume
requires that ledger even when an optimizer checkpoint exists: missing,
malformed, negative, or non-integer usage cannot restart the allowance at zero.
Reported generation beyond the requested allowance is recorded and fails the
run; the admission bound depends on the provider honoring `max_tokens`.
Repeats split the ceiling conservatively. This limits generated tokens, including
validation, not input tokens, backward tokens, or dollars. River jobs have unknown dollar
cost in the ledger. RL rejects `--max-cost` instead of pretending to enforce it.

## Stateful commerce benchmark

```bash
python examples/river_refund_rl.py --dry-run --output outputs/refund-42
python examples/river_refund_rl.py --seed 42 --steps 20 --output outputs/refund-42
# Compare a base model or an existing SFT/rejection-sampling checkpoint:
python examples/river_refund_rl.py --evaluate-only --seed 42 --output outputs/base-42
python examples/river_refund_rl.py --evaluate-only --checkpoint outputs/sft-model --seed 42 --output outputs/sft-42
```

The second command is paid training; evaluation-only commands also incur sampling costs. The sandbox executes lookup/refund/deny
operations against private per-trajectory ledgers. It checks eligibility,
amounts, wrong orders, duplicate actions, and actual resolution. A textual claim
that a refund happened earns no success. Splits contain 256 train, 64 validation,
and 128 test orders; the selected validation checkpoint is tested once.

Each output directory is bound to a `run_manifest.json` containing the run
configuration and split hashes. A matching interrupted run can resume. Changed
settings, edited split files, and directories without a manifest are rejected;
use a new directory for a different experiment. Completed test results cannot
be overwritten by rerunning the example. A dry run must use the same flags as
the subsequent real run (apart from `--dry-run`).

`test_results.json` records exact case hashes, checkpoint identity, evaluation
settings, and one outcome per test case. Outcomes include success, violations,
truncation, generated tokens, tool calls, and trajectory elapsed time. Missing
or duplicated cases fail evaluation. Summary rates are computed from outcomes;
cached totals are never trusted by the comparison command.

After collecting base and RL runs for seeds 42, 43, and 44, compare them offline:

```bash
stateset-agents benchmark compare-agents \
  --baseline outputs/base-42/test_results.json \
  --baseline outputs/base-43/test_results.json \
  --baseline outputs/base-44/test_results.json \
  --candidate outputs/refund-42/test_results.json \
  --candidate outputs/refund-43/test_results.json \
  --candidate outputs/refund-44/test_results.json \
  --output outputs/base-vs-rl.json --strict
```

Repeat the comparison against SFT and rejection-sampling SFT. The command requires
matching seed sets, environment version, base model, and evaluation settings.
Within each seed, case IDs and case content hashes must match. It reports paired
wins/regressions, per-run Wilson 95% success intervals, success-gain variation
across seeds, and token/tool usage. Trajectory seconds sum individual latencies;
they are not wall-clock duration or measured dollar cost.

Each seed pair and scenario-family comparison also includes `resource_usage`:
baseline/candidate generated tokens per evaluated case and per successful case,
plus candidate-minus-baseline totals for generated tokens, tool calls, and
trajectory seconds. Tokens per success include tokens spent on failed attempts;
the value is `null` when no case succeeds. For example, 800 generated tokens over
four cases with two successes means 200 tokens per case and 400 per success.
These descriptive measurements are recomputed from outcomes, so cached totals
cannot conceal additional usage. They exclude training work, input tokens,
provider billing and unrecorded retries, and do not change the learning gates.
Review each seed and family alongside success and safety results before deciding
whether a gain justifies additional rollout work.

The default gate requires at least three seeds, a mean absolute success gain of
three percentage points, positive gain in every seed, and no per-seed increase
in violation or truncation rates. `--min-seeds` and `--min-gain` set explicit
thresholds. `--strict` exits 1 for a failed gate; malformed or incomparable inputs
exit 2. The comparison report includes checkpoint identities and input hashes.
The default gates are empirical; opt into `--require-significance` for the
seed sign test described below. Neither proves that runs were independently
trained. Reported intervals describe each
test sample; they do not establish generalization to unseen workflow families.

### Policy reasoning and scenario-specific regressions

Select `--benchmark refund-policy-v2` to train on eight scenario families:
ordinary refunds, partial prior refunds, fully refunded orders, expired return
windows, undelivered orders, open chargebacks, window boundaries, and misleading
customer notes. This is a synthetic policy fixture, not a real merchant policy.
The original `refund-v1` remains the default for existing runs.

`refund-v1` is a lookup/action smoke benchmark: eligibility is returned by the
lookup. Its order IDs are opaque and cases are shuffled to avoid revealing the
answer through a sequential counter or row parity. Both generators use the same
ID format across splits, without visible `train`, `validation`, or `test` labels.
Lookup responses expose only the environment's order-fact fields; evaluator
metadata stays private even when supplied in a custom scenario. Both benchmarks require an
explicit `finish`; reaching the turn limit after a refund still fails the task.
Actions must be assistant JSON with unique keys and no separate tool-call
payload. Ambiguous JSON, including repeated keys, is a scored policy violation
and cannot enter verified demonstration data. These changes alter prompts and
default-benchmark cases; use new run/study directories for new experiments.

```bash
python examples/river_refund_rl.py --benchmark refund-policy-v2 --dry-run --output outputs/policy-42
python examples/river_refund_rl.py --benchmark refund-policy-v2 --seed 42 --steps 20 --output outputs/policy-42
python examples/river_refund_rl.py --benchmark refund-policy-v2 --evaluate-only --seed 42 --output outputs/policy-base-42
```

Lookup returns order facts rather than an eligibility flag or expected action.
The agent must calculate the remaining refundable balance, apply policy priority, and
execute `refund`, `deny`, or `escalate` with exact arguments. Open chargebacks
take precedence over other rules. The return window includes its last day.
Customer notes cannot authorize policy exceptions. Invalid actions leave the
financial ledger unchanged and disqualify success; a correct resolution must
be followed by `finish` within four turns.

Train/validation use 14- and 30-day windows; test uses 7- and 45-day windows.
All splits contain the same eight families with separate order identities and
balanced family counts. Order IDs are opaque hashes so their numeric sequence
does not reveal the scenario family or explicitly name the split. The split and
seed still determine the hash, keeping identities reproducible and disjoint.
This is a format-level safeguard, not a claim that public deterministic datasets
are secret or that held-out window values cannot identify a distribution shift.
Changing from the older split-prefixed IDs changes case hashes; use new run/study
directories and preserve historical evidence. Misleading notes cover denials, partial
refunds, and escalations rather than implying a single correct action.
This tests applying a known rule to different parameter
values, not generalization to entirely new policies. Scenario labels stay out of
model observations and are attached to evaluation artifacts by the harness.

Reports include success, uncertainty, violations, truncation, and usage by family.
When family labels are present, `compare-agents` additionally requires no family
to regress in success, violation rate, or truncation rate in any paired seed.
Missing/reassigned labels or different family coverage across seeds are rejected.
This conservative gate prevents gains on easy refunds from hiding worse results
on chargebacks or misleading notes. Family summaries are recalculated from the
case outcomes, just like aggregate summaries. Integer counters remain exact
during aggregation; integral SDK float metrics are also accepted. Duration totals
use stable summation, and nonfinite aggregates are rejected. Reports copy nested
outcomes, traces, settings, and checkpoint metadata so later caller mutations
cannot silently change a completed snapshot. These checks validate reported
measurements; they do not independently verify provider token usage or timing.

### Verified SFT and rejection-sampling datasets

Build an offline reference-policy baseline with the same sandbox and splits:

```bash
stateset-agents benchmark prepare-refund-data --seed 42 --output outputs/policy-data-42
```

The output contains `train.json`, `validation.json`, and `test.json` case files,
plus `train.jsonl` with verified multi-turn chat demonstrations **only for training
cases**. The teacher applies the stated policy to facts returned by lookup;
each completed conversation is replayed in a fresh sandbox. These are synthetic
reference demonstrations, not sampled model successes or evidence of learning.
Family and case provenance live in metadata, outside model messages. The River
SFT batch builder supervises assistant actions and masks environment observations.

To prepare a prospective boundary-reasoning SFT experiment offline:

```bash
stateset-agents benchmark prepare-refund-curriculum \
  --data-dir outputs/policy-data-42 --output outputs/policy-boundary-42
stateset-agents benchmark audit-refund-curriculum \
  --data-dir outputs/policy-data-42 --output outputs/policy-boundary-42
```

The `adjacent_boundary_v1` curriculum preserves every original training case.
For eligible, partially refunded, and boundary training cases, it adds contrasts
for the day before, the inclusive final day, and the day after the same return
window. Matching facts reuse an existing training case. All demonstrations replay
successfully; partial refunds still subtract the recorded refunded cents.
`lineage.json` and `contrast_groups.json` bind these examples to canonical training
parents. Validation/test facts are used only to reject overlaps, including matches
with different order IDs. No new window values or held-out examples are added.
The audit reconstructs facts, lineage, metadata, and exact teacher transcripts;
updating artifact hashes cannot admit a rewritten example. Publication uses the
same atomic bundle writer and refuses nonempty output directories.

An opt-in stability curriculum also varies the policy parameter itself:

```bash
stateset-agents benchmark prepare-refund-curriculum \
  --policy policy_parameter_contrasts_v1 \
  --data-dir outputs/policy-data-42 --output outputs/policy-parameters-42
stateset-agents benchmark audit-refund-curriculum \
  --policy policy_parameter_contrasts_v1 \
  --data-dir outputs/policy-data-42 --output outputs/policy-parameters-42
```

`policy_parameter_contrasts_v1` preserves all original cases and creates a
three-day contrast around one new window per source order, rotating 3, 10, 21,
60 and 90 days within each source family. It uses every training family, retaining
monetary facts, delivery status, chargebacks and customer notes. Higher priority
rules still override the date comparison. Teacher transcripts preserve exact
order IDs, perform one resolution, and finish with empty arguments. Derived facts
and teacher transcripts are reconstructed during audit, using the explicitly
selected policy. The default `adjacent_boundary_v1` artifacts remain unchanged.

For partial-refund arithmetic, opt into `borrow_boundary_contrasts_v1` with the
same prepare/audit commands and `--policy` flag. It retains every parameter
contrast and adds a nine-case payment/refund offset grid for each eligible
canonical partial-refund source, including partial orders with untrusted notes.
Thousand-cent anchors and offsets -1/0/+1 exercise decimal borrow boundaries
on the inclusive final day. Small-order clipping keeps every new order partially
refunded. Lineage records the exact source, anchors and offsets; audit reconstructs
the amounts and replays the exact teacher actions. Neither held-out amounts nor
held-out model answers supply examples. Both earlier policies remain reproducible.

`stateset_agents.data.refund_arithmetic_stress.refund_arithmetic_stress_cases`
builds a separate prospective arithmetic grid with independent source orders per
window/day/payment/refund cell. Declare its decimal `quantum`, windows, repetitions
and full exclusions before execution. For example, 54 validation cases with
`quantum=10`, windows `(5, 18, 40)`, and one repetition; 108 test cases with
`quantum=100`, windows `(8, 25, 75)`, and two repetitions. Its metadata stays
outside model observations. This changes the evaluation distribution and does
not establish improved learning. New experiments must retain separate ordinary,
window-stress and arithmetic-stress regression gates so aggregate arithmetic
gains cannot conceal an existing policy regression.

Fresh stress validation can exercise parameter transfer before test execution:

```python
from stateset_agents.data.refund_stress import refund_policy_stress_cases

validation = refund_policy_stress_cases(
    "validation", seed=2026102901, windows=(5, 18, 40),
    excluded=historical_cases + all_training_cases,
)
test = refund_policy_stress_cases(
    "test", seed=2026102901, windows=(8, 25, 75),
    excluded=historical_cases + all_training_cases + validation,
)
```

Supply the actual complete exclusion lists; the function does not discover past
experiments. The `policy_window_grid_v1` stress distribution draws an independent
order for each source-family/window/day cell: 24 cases per window. It rejects ID
or observable-fact reuse, including aliases. Private `stress` metadata identifies
the stratum and source hash; lookup never exposes it to the model. These are
explicitly different evaluation distributions, not canonical study bundles.
Freeze windows, seeds, exclusions, regression gates and selection rules before
provider execution. Existing test outcomes must not supply training examples.

The offline stability preparation retained at
[`river-boundary-stability`](../benchmark_results/training_contracts/20261019/river-boundary-stability/README.md)
replays the previous study's 576 validation trajectories, prepares 3,072 verified
training transcripts across the six retained parents, and excludes historical
and current training facts from 1,824 fresh holdouts. That preparation made no
provider calls. The subsequent
[`stability pilot`](../benchmark_results/training_contracts/20261019/river-stability-pilot/README.md)
used the first parent's 304 held-out cases; the other five parent sets remain
unevaluated. Preparation establishes dataset integrity; model-quality claims
require the separate executed experiment and its frozen acceptance gates.

This is a separate synthetic training bundle, not a replacement benchmark split
or a model quality result. Pass its `train.jsonl` to SFT; do not pass it as a
canonical study data directory. A new experiment must declare this training
distribution before execution, retain its validation-selection rule, exclude
previously exposed cases from its fresh test set, and compare against the original
SFT baseline. The existing study planner/auditor does not accept this curriculum
as its original reference arm. No additional provider run is launched by either
command, and independently reviewed data remains necessary for production claims.

Train and evaluate this SFT baseline (these two commands contact River):

```bash
stateset-agents train-remote --provider river \
  --dataset outputs/policy-data-42/train.jsonl --base-model Qwen/Qwen3.5-9B \
  --provider-options-json '{"seed":42,"shuffle":true}' \
  --lora-r 16 --max-length 4096 --output-dir outputs/policy-sft-42
python examples/river_refund_rl.py --benchmark refund-policy-v2 --evaluate-only \
  --checkpoint outputs/policy-sft-42 --seed 42 --output outputs/policy-sft-eval-42
```

To build a rejection-sampling baseline, collect eight model trajectories per
training case without optimizer updates or held-out evaluation, then filter
them offline:

```bash
python examples/river_refund_rl.py --benchmark refund-policy-v2 --collect-only \
  --seed 42 --output outputs/policy-samples-42
stateset-agents benchmark filter-refund-data \
  --data-dir outputs/policy-data-42 \
  --candidates outputs/policy-samples-42/training_candidates.json \
  --output outputs/policy-filtered-42
stateset-agents train-remote --provider river \
  --dataset outputs/policy-filtered-42/train.jsonl --base-model Qwen/Qwen3.5-9B \
  --provider-options-json '{"seed":42,"shuffle":true}' \
  --lora-r 16 --max-length 4096 --output-dir outputs/policy-rsft-42
```

Collection is paid sampling. It uses temperature 1 and the existing per-trajectory
budgets. Add `--checkpoint` to collect from a trained policy instead of the base
model. `--collect-only` and `--evaluate-only` are mutually exclusive. All dataset
and example commands accept `--train-count`, `--validation-count`, and
`--test-count`; use identical counts and seed throughout a campaign (defaults
256/64/128). Multiples of eight preserve balanced scenario families.

`training_candidates.json` is saved after each completed group. An interrupted
collection remains marked incomplete and its saved candidates can still be
filtered. Reusing its output directory for sampling is refused to prevent
overwriting evidence; use a new directory for another collection attempt.

Filtering verifies the prepared bundle hashes and canonical splits, then
re-executes every assistant action in an isolated ledger. Prompts and observations
must match exactly. Truncated/incomplete conversations, forged observations,
policy failures, held-out cases, and messages after termination are excluded.
Model-reported reward is ignored. Among successful candidates, keep one per case:
fewest assistant turns, then fewest characters, then transcript hash. This is a
deterministic selection rule, not a token-cost estimate. Missing successful
families remain visible in the coverage report; no teacher rows are substituted.
The study provenance audit replays the full recorded collection with this same
selection rule. It requires the exact selected transcripts and metadata, counts
and family coverage, and candidate-by-candidate rejection log. Updated local
hashes cannot make an omitted success or a longer selected transcript satisfy
the declared rule. This verifies consistency with the recorded collection, not
independent proof that a provider generated it; coordinated replacement of all
source evidence remains outside the local audit's guarantees.

Each export publishes `data_manifest.json` last with artifact hashes and source
provenance. Filtered exports also contain `replay_audit.json` with selected,
rejected, and unselected candidate identities. New output directories are
required. If filtering finds no success, it writes the audit and manifest,
omits `train.jsonl`, and exits 1; malformed inputs exit 2.

Evaluate the rejection-sampling checkpoint with `--evaluate-only`, or initialize
RL from the SFT checkpoint using `--checkpoint`, then use `compare-agents` on
matched test results. Fix training choices before inspecting test results. The
data/collection seed does not automatically override `train-remote`'s SFT seed.
Use `--provider-options-json '{"seed":42,"shuffle":true}'` for River SFT, changing
the seed for each independent training run. Defaults are seed 0 and shuffling
enabled. The seed controls LoRA initialization and a local RNG that reshuffles
each epoch; a retry restarts the same sequence without mutating the source data.
Both values are recorded in the checkpoint pointer and adapter manifest.
Unknown options and invalid types are rejected before client access. A nonfinite
reported loss fails the job without publishing a checkpoint. These settings
make the inputs reproducible; they do not promise bitwise deterministic service
execution. Matched evaluation seeds alone do not prove independent training runs.

Every comparison includes an exact one-sided seed sign test. Enable
`--require-significance --strict` to make it a gate. The sampling unit is a paired
run seed, so adding trajectories cannot masquerade as additional training runs.
Ties are excluded; all ties yield p=1. Use `--comparisons 3` when the prespecified
study compares RL against base, reference SFT, and rejection SFT. This multiplies
the p-value by three (capped at one). At the default alpha .05, five positive
seeds give p=.03125 for one comparison, while six positive seeds give adjusted
p=.046875 for three. Three positive seeds give p=.125 and fail this gate.
These are arithmetic thresholds, not a power analysis or a reason to stop a
study early. Fix the number of runs and hypotheses before test inspection.

The test evaluates consistency of the direction of improvement, not the size
of its mean or safety. Existing effect-size, per-seed and per-family gates still
apply. Repeated candidate checkpoint paths fail the statistical mode, but distinct
paths alone cannot establish independence: retain and audit training manifests.
The calculation follows the [exact binomial test](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/binom.test.html)
with [Bonferroni correction](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html).

Use `benchmark compare-agent-study` to assemble all four arms in one report.
Pass one `--base`, `--sft`, `--rejection-sft`, and `--rl` path for **each** seed:

```bash
# Repeat these four flags for every prespecified seed (six or more by default).
stateset-agents benchmark compare-agent-study \
  --base outputs/policy-base-42/test_results.json \
  --sft outputs/policy-sft-eval-42/test_results.json \
  --rejection-sft outputs/policy-rsft-eval-42/test_results.json \
  --rl outputs/policy-42/test_results.json \
  --output outputs/policy-study.json --strict
```

This one-seed example intentionally fails the minimum-seed and statistical
gates; supply the remaining run paths for the full study. The command requires
all three comparisons to pass, always corrects for three tests, and rejects
reused checkpoint identities across trained arms or seeds. Defaults are six
seeds, three percentage points of mean improvement, and alpha .05. The report
contains each comparison and its evidence hashes. It assesses learning evidence;
live recovery, measured cost and deployment validation remain separate checks.

`river_environment_factory` adapts StateSet environments to the native SDK.
Initial messages come from `state.context["messages"]`; nonterminal steps return
only new observations in `info["messages"]`. Step rewards are summed. Recovery
drops unfinished environment instances rather than replaying side effects.
The bridge attaches `stateset_episode_id` to completed/truncated trajectories
for case attribution; truncated trajectories cannot retain a success metric.
The generic bridge accepts a finite `truncation_reward` (default zero), which
replaces accumulated partial reward and closes the episode. The refund runner
sets it to **−1**, matching a completed failure, and explicitly uses River's
`Truncation(train="reward")`. River's default `zero_reward` training policy would
otherwise overwrite that penalty with zero and give a truncated response a
positive group-centered advantage against a failed response. Both choices are
recorded in the run manifest; the reward is also part of evaluation settings.

An environment can exhaust its own turn limit before River truncates sampling.
The bridge retains its terminal reward, records `environment_timeout` as a
metric, and attaches a serialized `stateset_truncated` cause. The runner uses
`trajectory_truncation()` when exporting test outcomes and collection candidates,
so these timeouts count in the truncation gate and cannot enter verified SFT.
River's own trajectory truncation field remains unchanged. Infrastructure
exceptions still propagate without producing a scored training trajectory.
The native example has its own River budgets and recovery controls; it does not
use the synchronous executor's generated-token ceiling.

## Evidence and the A+ acceptance gate

### Borrow-boundary continuation pilot, October 20, 2026

The [arithmetic pilot](../benchmark_results/training_contracts/20261020/river-arithmetic-pilot/README.md)
completed 88 added River SFT updates and stopped at its validation safety gate.
Both matched arms passed 157/158 validation cases. Arithmetic improved from
53/54 to 54/54, but the existing window-stress suite declined from 72/72 to 71/72:
the candidate prioritized an outside-window denial ahead of the already-refunded
rule. A later correction cannot erase that policy violation. The candidate
remains unpromoted; all 308 test cases remain unattempted.

Both arms start from the previous unpromoted parameter checkpoint, use 701 rows,
44 updates, identical family counts and hyperparameters, and local input/target
exposure within 0.2%. The arithmetic curriculum adds 189 training-only borrow
contrasts. Fresh arithmetic validation/test equations are distinct and absent
from candidate training. This is an exploratory single-parent experiment, not
an A+ result or proof of a general arithmetic gain.

The executed validation stop demonstrates that an aggregate tie and a gain in
the new suite cannot waive existing policy regressions. Offline review replayed
1,402 teacher transcripts and 316 validation trajectories and reconstructed both
fresh holdout sets. The run reserved 323,584 of its 954,368 output-token allowance;
these are admission reservations, not billed usage. No retries, RL or deployment.
199 unit/CLI tests, nine pilot checks and full mypy over 447 packaged sources
passed. Both earlier curriculum policies remain reproducible.

### Policy-parameter stability pilot, October 19, 2026

The completed [one-parent River pilot](../benchmark_results/training_contracts/20261019/river-stability-pilot/README.md)
failed its prespecified primary acceptance gate. Fixed final checkpoints passed
164/200 test cases for the retained original parent, 183/200 for additional legacy
curriculum SFT, and 197/200 for broader policy-parameter SFT. The trained arms
share the parent, 512 rows, 32 updates, family counts, seed and hyperparameters;
input and assistant-token exposure differ by 0.063% and 0.053%, respectively.
Validation admitted test evaluation only after both safety comparisons passed.

The broader curriculum gains seven percentage points over control, but partial
refunds decline from 19/19 to 18/19 and stress cases sourced from partial-refund
orders decline from 8/9 to 7/9. Eight-day stress truncation also increases from
zero cases to one. Those overlapping strata fail the frozen nonregression gate;
the comparison against the original passes. The candidate remains unpromoted.

All 64 added SFT updates completed in about 21.6 minutes with no retries or
deadline extension. Offline review replayed 1,024 training transcripts and 912
evaluation trajectories and recomputed each gate. Eight new control/gate tests
passed. The run reserved 933,888 output tokens; this is an admission allowance,
not generated usage or billed dollars. Actual dollar cost remains unavailable.
This is one exploratory synthetic parent, with no RL learning or deployment.
Fixing the remaining regressions requires training-only evidence and fresh
evaluation; these exposed holdouts must not supply new supervision.

### Six-parent continuation replication, October 19, 2026

The retained [paired SFT study](../benchmark_results/training_contracts/20261019/river-boundary-six-seed/README.md)
compares original checkpoints, repeated-original-data SFT, and boundary-contrast
SFT across six retained, independently trained parents. Continuations use new
shuffle seeds and fresh holdouts. The two trained arms have identical rows,
16 updates and hyperparameters, with input and assistant-token exposure within
0.5% per pair. These are conditional continuations, not six new base-model
initializations.

On 768 fresh test cases per arm, original checkpoints passed 689 (89.7%), the
repetition control 719 (93.6%), and boundary SFT 747 (97.3%). Boundary SFT passed
the secondary gate against original checkpoints in all six seeds, with corrected
p=0.046875. The primary boundary-versus-repetition gate failed: four seed wins,
two losses, and family, policy-violation and truncation regressions. Its higher
pooled score does not establish reliable superiority over matched continued SFT.

The audit replayed 3,030 training transcripts and 2,880 evaluation trajectories.
All 192 added SFT updates completed in about 37 minutes without retries. This
synthetic policy result does not establish independently reviewed industry
quality, RL gains, live optimizer recovery, or A+ production readiness.

### Prepare and audit the entire campaign

The benchmark now ships in the Python package:

```bash
python -m stateset_agents.verticals.river_refund --help
stateset-agents benchmark plan-agent-study --output outputs/refund-study
stateset-agents benchmark prepare-agent-study-data --study-dir outputs/refund-study
stateset-agents benchmark preflight-agent-study --study-dir outputs/refund-study --strict
stateset-agents benchmark audit-agent-study --study-dir outputs/refund-study --strict
```

Planning writes a deterministic `study_plan.json`, without loading the River SDK
or making paid calls. The default six-seed plan contains 54 stages, 42 marked
paid. Each stage records an argument vector and prerequisite stage IDs. Execute
those arguments from the study directory using the intended Python environment,
after reviewing model availability and the provider budget. Do not concatenate
them into shell code. All arms use fixed seeds and split sizes; RL starts from
the base model. The example script delegates to the packaged module.

`prepare-agent-study-data` prepares all planned reference datasets offline. It
validates existing bundles before creating missing ones, reuses matching bundles
without changing their bytes, and refuses corrupt or mismatched inputs. New
and reused training transcripts are replayed against their canonical cases;
prompts, observations, successful terminal actions, and family metadata must
agree with the sandbox. Rehashed invalid transcripts fail before preparation
creates any missing bundles. New
bundles are written in a private sibling staging directory, flushed, and published
by directory rename. A failed write leaves the destination empty or absent, so preparation
can be retried. A failure after publication leaves a complete bundle that the
retry reuses. An abrupt process exit may leave an orphan under the sibling
`.NAME.publication` directory; it is never treated as published training data.
That directory also holds the writer lock and must remain in place while writers
are active. Legacy partial bundles and modified evidence still require inspection
rather than automatic replacement. Preparation writes `data_preparation.json`.

`preflight-agent-study` writes `study_preflight.json` with dataset integrity,
Python compatibility, installed River SDK version/API checks, credential presence,
stage counts, and the plan's token limits. Every training transcript is replayed
offline, with `replayed_examples` reported for each passing seed. This checks
training-example execution, not model quality or tokenizer compatibility.
Run it with the same Python environment
that will execute the study. `--strict` exits 1 for missing local prerequisites,
or 2 for an invalid/stale plan. It does not print credentials, instantiate a River
client, load a tokenizer, or execute a stage. Authentication, model availability,
funding, tokenizer downloads, and provider pricing remain unverified even when
the local checks pass. Neither command authorizes spending or reads test outcomes.
Programmatic async callers can await `preflight_study_async(directory)` from
`stateset_agents.verticals.study_preflight`; the synchronous `preflight_study`
wrapper is intended for callers outside an active event loop.

The direct native runner applies the same runtime checks before loading a tokenizer
or opening a client. It requires Python 3.12+, River `>=0.11.0,<0.12`, the native
environment/training/evaluation APIs, a keyword-compatible recovery callback,
and a nonblank `RIVER_API_KEY`. Dry runs remain independent of the optional SDK.

The CLI, preparation, and both programmatic execution entry points share native
configuration checks: integer step/concurrency counts, nonnegative integer seeds
and staleness, finite positive learning rates, explicit boolean mode flags, and
integer rollout budgets of at least 1,024 tokens when configured. Booleans and
numeric strings are not silently accepted as numeric settings. `dry_run=True`
also applies to direct `campaign()` calls: they validate cases without creating
SDK objects, saving weights, consuming rollout admission, or changing evidence.

Evaluation-only and collection checkpoint saves run off the event loop. If a
caller cancels during a save, the campaign waits for that in-flight call to finish
before propagating cancellation, allowing its owner to close sessions afterward.
Repeated cancellation cannot detach the save, and a cancelled save never starts
sampling. This does not cancel the remote request or retry uncertain saves; a
provider checkpoint may already exist. A hung SDK call can delay this campaign's
cancellation until the SDK timeout, while other event-loop work can continue.

Before live execution, the native entry points (`execute_run` and `campaign`) compare their
configuration and case data with the prepared manifest and split files before
SDK use. They reject missing or modified splits and existing holdout/collection
seals without rewriting evidence. Each call copies its inputs, so later caller
mutations cannot change an active campaign. Programmatic callers must hold the
run directory lock; `campaign` accepts already-provisioned model/session objects,
whose provider-side identity these local checks cannot attest.

Preparation validates every supplied case with the benchmark environment's
scenario rules. Order IDs must be unique within each split and disjoint across
training, validation, and test. Training requires all three nonempty splits;
evaluation requires test cases, and collection requires training cases. Unused
splits may be omitted or empty. Policy-v2 cases also need a recognized family
label for per-family reporting. These checks enforce local case identities;
they do not prove that externally supplied model weights never saw the holdout.

Plans, native run manifests, and test reports record SHA-256 hashes of every
Python source file in the installed `stateset_agents` package. These identities
include uncommitted edits and use relative paths, so identical checkout and
wheel sources match. A changed package requires a new study and run directory;
execution rejects a stale manifest before opening provider sessions, and the
audit requires the planned implementation in every report. This conservative
check also invalidates plans after unrelated package-code edits. Keep the
experiment installation immutable and restart processes after editing code.

Standalone comparisons reject differing source identities, including mixtures
of reports with and without identities. Legacy reports with no identity on
either side remain comparable, with `protocol.implementation` set to null;
they cannot pass the planned-study audit. Source hashes describe local files;
they do not lock external dependencies, attest loaded process code or provider
behavior, or protect against coordinated rewriting of evidence.

The plan records per-trajectory limits and training settings. Optionally set
`--rollout-token-budget` on the planner or native runner to cap reserved output
tokens per native run. Before each trajectory resets or samples, its full
1,024-token allowance is durably reserved in `rollout_budget.json`. Training,
validation, collection, test, and prefetch use the same ledger within a run.
Reservations are never refunded, including after short responses, failures,
cancellation or resume. An exhausted budget stops the run; a missing or invalid
ledger fails closed. Choose enough capacity for validation and test as well as
training. Budgets cannot be raised when resuming an existing run.
The planner records the minimum nominal reservation by run and rejects a cap
below it; allow additional room for prefetch, failed trajectories and recovery.

There are five native runs per seed, so a capped six-seed plan records an aggregate
reserved allowance of 30 times the per-run cap. This is an admission limit under
the SDK's per-trajectory generation contract, **not** a dollar ceiling or provider
billing meter. It does not price input tokens, optimizer work, or provider-side
retries. Budget exhaustion can leave partial rollout groups and an incomplete
study. Without this option, aggregate rollout admission remains uncapped.

Split sizes must be positive multiples of eight. Prepare the plan before
inspecting test outcomes and retain its hash outside the result directory for
independently timestamped preregistration.

An immediate strict audit fails because live results are absent. As evidence
arrives, it reports missing or mismatched seed/arm artifacts. It verifies canonical
splits, replays SFT training conversations, checks that filtered chats occur in
the complete saved collection, and binds trained checkpoints to exact dataset
bytes, seeds, LoRA rank and training settings. Test reports must match their run
manifest and planned cases/families. RL must retain the planned step history,
validation results for the initial checkpoint and every training step, and use
the validation-selected checkpoint. Each validation step records one finite reward
per planned case and the exact case hashes in `validation_results.json`. Selection
and the study audit reject incomplete, duplicate, or unexpected cases, changed
case hashes, and aggregate reward means inconsistent with the recorded outcomes.
Before opening the test split, the runner also replays every validation trace in
a fresh sandbox and requires its reward to match. The study audit applies the
same selection gate. This includes every losing checkpoint, so an apparently
stronger aggregate cannot bypass inconsistent underlying actions or observations.
The environment adapter snapshots each JSON input before waiting for admission
and attaches its hash to the trajectory, even if it truncates before sampling.
Validation, test evaluation, and candidate collection require that actual reset
input hash to match the planned case. Reports retain it on each outcome; the
study audit rejects missing or mismatched hashes. These hashes establish local
input provenance, not independent attestation of provider execution.
Native benchmark outcomes and collected candidates also retain an
`environment_trace`: the sandbox's initial messages, each executed assistant
action, the resulting observations, per-step rewards and metrics, and the final
reward and truncation cause. This includes episodes stopped before the first
action. Use these records to investigate incorrect tool calls, policy violations,
and unfinished tasks. Recorded observations are what the sandbox produced; the
engine may stop before delivering them to the model. Traces contain task data
and model output and inherit the result files' access controls and retention.
Generic adapters can opt in with
`river_environment_factory(..., record_trace=True)`; tracing is off by default.
The study audit replays each held-out trace in a fresh refund sandbox and rejects
mismatched observations, step rewards, terminal outcomes, success flags, policy
violations, or tool-call counts. Missing traces also fail the audit. Replay uses
the report's exact packaged implementation and case hashes. It is a consistency
check against those sandbox rules, not an independent validation of the rules.
Token counts, latency, provider/model execution, and external truncation triggers
remain unverified; declared external stops must still receive the configured
failure reward. To audit a single native test report without any provider calls:

```bash
stateset-agents benchmark audit-refund-traces \
  --report outputs/refund-seed42/test_results.json \
  --cases outputs/refund-seed42/test.json \
  --output outputs/refund-seed42/trace_audit.json
```

The command writes an input-bound audit with per-case replay results and issues.
It exits 1 for inconsistent evidence and 2 for invalid inputs, and refuses to
overwrite either input file. An audit pass alone does not demonstrate learning.
Legacy aggregate-only validation records do not establish case coverage and
cannot pass this protocol; preserve old runs and use a fresh output directory.
The runner enforces these requirements before testing, including after an early
training stop. Recovery replaces a replayed validation step rather than retaining
conflicting checkpoints. Immediately before constructing the test engine it
persists `test_attempt.json`, binding the chosen checkpoint, test split and run.
The report includes that marker's hash. The runner replays all completed test
outcomes before publishing `test_results.json`.
A replay mismatch writes `test_replay_failure.json`, containing the rejected
candidate report and its input-bound replay audit, and keeps the attempt sealed.
Failed or uncertain attempts stay sealed: resuming the directory cannot silently
sample the test set again. Preserve the
failed attempt as evidence; do not delete its marker or backfill one for an older
report. Older artifacts without this holdout protocol cannot pass the new audit.

Recovery reconciles progress through River's `after_recovery` callback, after the
SDK restores and validates committed training state. If a crash occurred after a
batch commit but before local metric logging, `training_metrics.json` records
that batch with `metrics: null`, `source: river_recovery`, and a hash referencing
`recovery_receipts.json`. These are completion receipts, not reconstructed losses,
token counts, optimizer-update counts, or cost measurements. Observed records
beyond the restored batch count and their validation results are discarded.
Receipt-first writes are idempotent if reconciliation itself is interrupted.
Observed step metrics must be a nonempty dictionary of named, finite numeric
scalars, matching River's step contract. The logger copies these values before
persistence so later SDK mutations cannot rewrite earlier measurements. The
same validation applies when reopening progress and auditing completion; invalid
observations cannot unlock the test split. Recovered batches retain unknown
metrics rather than substituting zeroes. Finite metrics alone do not prove
learning gains or independently verify provider execution.

`training_activity.json` separates completed batches from SDK-reported optimizer
updates using River's `train/updated` flag. It reports observed updates, observed
skipped batches, and batches whose update status is unknown. Legacy observations
without that flag and recovered batches without metrics remain unknown; recovery
receipts never become invented optimizer-update counts. Known update flags must
be zero or one and agree with `train/datums` when both are present.

If every completed batch is observed and every optimizer update was skipped, the
native campaign saves this diagnostic and stops before opening the test split.
Inspect reward variation, rollout diversity, truncation and staleness masks.
The native runner and study planner also accept `--zero-update-patience`
(default `5`, `0` disables early stopping). Reaching this many consecutive
observed skipped updates stops training immediately after recording that batch,
saves `training_stop.json` bound to the run and progress hashes, and keeps the
test split sealed. Updates, unknown metrics, and observation gaps reset the
streak. The guard also checks SDK-reconciled history before resuming updates;
uncommitted suffixes discarded by recovery cannot trigger it. A stop diagnostic
describes the recorded history at that time; reconciliation determines the
current history. Changing the threshold requires a new run manifest and study
plan. Study audits reject evidence that crosses the planned stopping threshold,
even if later records claim successful updates. Already admitted rollouts and
validation may still finish during cleanup; this is not a provider billing cap.
Runs that complete without stopping include the summary in the test report; unknown recovery activity
remains visible and is not proof that an update occurred. These counts describe
the run, not necessarily the selected validation checkpoint, and do not establish
nonzero gradients or held-out learning gains.

Native training manifests now declare the selection policy
`initial_and_unskipped_batches_v1`. Selection considers the initial checkpoint
and batches whose optimizer update was not confirmed skipped, using highest
validation reward with earliest-batch ties. A higher resampled validation score
after `train/updated=0` cannot promote unchanged weights. Every validation result,
including excluded batches, still requires complete case coverage and trace
replay. Unknown recovery activity stays eligible and remains explicitly unknown.

Before test sampling, `validation_selection.json` records eligible and excluded
batches, the selected checkpoint, source evidence hashes, and the optimizer
activity observed through the selected batch. The test report includes this
selection record's hash and `selected_training_activity`. Selecting batch zero
reports `initial_checkpoint` with zero prior native RL updates even if later batches
updated the model. The study auditor independently recomputes and verifies the
selection record and report fields; a whole-run update count cannot establish
that selected weights received RL updates. This protocol change applies to new
runs. Prior study results retain their frozen implementations and selection
rules; they are not rescored or relabeled as new evidence.

The study auditor recomputes activity from validated `training_metrics.json`
records and recovery receipts. Both `training_activity.json` and the test report's
copy must match that summary exactly, including count types. Known zero-update
runs cannot pass the audit, and evaluation-only arms cannot claim native RL
activity. Successful RL audit entries include the recomputed summary and hashes
of the progress and receipts used. Fully recovered runs can retain unknown
activity; a passing provenance audit does not turn it into observed updates.

Native campaign iterators are explicitly owned during training, collection, and
test evaluation. A failure in step persistence or trajectory validation closes
the iterator before the campaign returns. Iteration and closing share one task
to preserve generator context, and no next item is requested ahead of the
consumer. Caller cancellation drains an in-flight next-item operation, then
closes the iterator; repeated cancellation cannot detach cleanup. A committed
batch may therefore finish without its local metric being logged, which the
recovery receipt mechanism handles. Configure SDK timeouts: a hung provider
operation can delay cancellation indefinitely. Existing test-attempt seals and
partial collection records remain in place after failures.

Validation callbacks serialize evidence writes with each other and with recovery
reconciliation. Writes run off the event loop, and cancelling an evaluator waits
for its active write before releasing the campaign. This prevents overlapping
callbacks from losing validation results or modifying evidence after shutdown.

The audit rejects missing or mismatched receipts; a locally recorded receipt is
not independent attestation of provider execution or live recovery certification.
For capped runs, the audit also verifies durable reservations against run-bound
budget snapshots and the minimum trajectory count required by the protocol.

Source training URIs are checked: saving the same trained model under new
evaluation checkpoint names cannot count as independent training evidence.
Only complete provenance proceeds to all three corrected statistical comparisons.
Local hashes provide consistency checks, not protection against coordinated edits
or independent attestation of provider execution. Live recovery and measured cost
remain separate gates. Auditing never contacts River or retrains a model.

### Native SDK checks without paid calls

For a separate agent-system approach to the arithmetic pilot's amount and
priority errors, see [Explicit refund resolution tool](REFUND_RESOLUTION_TOOL.md).
It delegates policy calculation to a trusted tool and includes a local sandbox
example. Its scripted tool checks are separate from the River model scores;
the arithmetic pilot's test cases remain unattempted.

The `river-native` CI job checks both River 0.11.0 and the latest available version
within the supported `>=0.11.0,<0.12` range on Python 3.12. Each matrix entry runs the
real River rollout engine and trainer against a scripted transport. It checks
private environment state, exact tokens/logprobs, truncation, infrastructure
errors, durable admission, regeneration of interrupted sandboxes, partial
training recovery, and fully committed recovery without re-emitted metrics.
Client construction is forbidden in the suite. No credentials, model downloads,
GPU, or paid inference/training are required.

Run `python scripts/check_asyncio_wakeup.py` before the SDK checks. It verifies
that a worker thread can send a wakeup byte through a local socket pair, without
opening a listener or connecting to an external service. A sandbox can allow
main-thread socket writes while denying worker-thread writes. Asyncio suppresses
those wakeup errors; completed checkpoint writes can then appear stalled until
another timer wakes the loop. A failed probe exits with an actionable error.
Use an execution environment that permits local worker-thread IPC; keep the
suite's provider-client and RPC guards enabled. Increasing test deadlines or
disabling filesystem sync does not resolve that permission restriction.

Run it in a dedicated environment; River's SDK dependencies can conflict with
static-analysis tools in the full `dev` extra:

```bash
python3.12 -m venv .venv-river-contract
.venv-river-contract/bin/python -m pip install -e ".[river]"
.venv-river-contract/bin/python -m unittest discover \
  -s tests/integration -p test_river_native_sdk.py -v
```

Ordinary pytest runs skip this module when the optional SDK is absent or Python
is too old. An installed but broken SDK fails the checks. The dedicated CI job
also checks SDK import before running tests, so missing dependencies cannot
silently turn that job green. These are native SDK integration checks with
synthetic model responses and updates, not learning or provider recovery proof.

Historical live SFT/RL runs are documented in `PROOFS.md`; the dated study above
provides live SFT continuation evidence. Scripted SDK checks do not establish
live learning or optimizer continuity during a live recovery.

Before calling this production A+, run the same held-out commerce benchmark for
base, SFT, rejection-sampling SFT, and RL under matched token budgets and at least
six seeds. Report task success with uncertainty, violation/duplicate rates,
tool calls, generated tokens, elapsed time, and measured cost where available.
Also interrupt/resume a live run and verify optimizer continuity and selected
checkpoint sampling. Preserve all run artifacts and never use test results to
choose a checkpoint. Refund-v1 varies order IDs and amounts; policy-v2 also
holds out return-window values across eight fixed scenario families. Neither
benchmark establishes generalization to new commerce workflows, unseen policy
rules, or real merchant data.

Reference contracts: [losses](https://docs.river.ai/guides/losses/),
[RL primitives](https://docs.river.ai/guides/rl-primitives/),
[checkpoints](https://docs.river.ai/guides/checkpoints/),
[environments](https://docs.river.ai/guides/rl-tools/), and
[Python API](https://docs.river.ai/python-api/).
