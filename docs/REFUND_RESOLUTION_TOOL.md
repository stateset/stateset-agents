# Explicit refund resolution tool

The River arithmetic pilot fixed its subtraction validation cases but still
failed the policy priority gate. `SyntheticRefundResolver` addresses those
operations in the tool layer: the model delegates a resolution intent, and
trusted code selects the action and exact integer amount from a fresh lookup.
This is a new agent-system variant. It does not improve or replace the recorded
raw-model scores, and it implements only the fictional `refund-policy-v2` policy.

```python
from stateset_agents.core.refund_tools import SyntheticRefundResolver

resolver = SyntheticRefundResolver(
    order_id=authorized_order_id,          # supplied by the trusted application
    idempotency_key=persisted_request_key, # stable across process restarts
    lookup=trusted_order_lookup,           # async (order_id) -> facts mapping
    submit=atomic_resolution_backend,      # async (request) -> receipt dict
)
agent.register_tool(resolver.as_function())
```

Use an agent with `FunctionCallingMixin` initialized. The model-visible argument
object is exactly `{"intent":"resolve_refund"}`. Order IDs, amounts, reasons,
policy facts, and idempotency keys are supplied by the application, not the
model. Notes and evaluator annotations never determine the decision. The tool
prioritizes chargeback, already refunded, not delivered, outside window, then
the remaining refundable cents. The last return-window day is inclusive.

`submit` receives an immutable `RefundResolutionRequest`, including its trusted
fact snapshot and stable idempotency key. Its `action()` method returns the
derived action. Before applying it, the backend must atomically recheck
authorization and the snapshot against current order facts, enforce one
resolution per order, and persist idempotency across requests and processes.
It acknowledges a committed resolution with a receipt containing `applied: True`.
Lookup alone cannot protect against a chargeback or refund posted before commit.

Concurrent invocations on one resolver share one submission. Returned receipts
are copied. A lookup failure can be retried before submission begins. Any
submission exception, missing acknowledgment, or task cancellation makes that
resolver refuse further submissions: the application must reconcile the
persisted request key with its backend. The in-memory resolver does not provide
durable exactly-once payments, restart recovery, or merchant authorization.
Create a separate resolver per authorized order on one event loop.

The function dispatcher rejects malformed/non-object JSON, duplicate keys,
nonfinite numbers, and excessive nesting before invoking handlers. Previously
malformed or non-object arguments could become `{}` and execute a handler with
default arguments. Direct callers of `ToolCall.parsed_arguments()` must now
handle `ValueError`; dispatched calls return a tool error.

Run the local integration example:

```bash
python -m examples.refund_resolution_tool
pytest -q -o addopts= tests/unit/test_refund_tools.py
```

The example executes scripted intents on 128 synthetic training fixtures in the
unchanged policy sandbox. It records intent separately from executed actions and
observations, makes no provider calls, and produces no model-quality evidence.
The sandbox still scores a raw model's incorrect action as a policy violation.
Its existing River trace schema and the failed pilot artifacts remain unchanged.

A River comparison of the delegated variant needs a separately declared protocol
that attributes correctness to the whole agent system, captures original model
intents and executed actions separately, and uses fresh evaluation cases. The
arithmetic pilot's sealed test set remains unattempted.

## River adapter and comparison

`river_refund_delegation_factory()` in
`stateset_agents.remote.river_refund_delegation` connects this tool to River's
rollout engine. The model requests a `resolve_refund` JSON intent and then
`finish`. The adapter captures schema-2 `refund-delegation-v1` traces: each step
contains the original model `action`, its observations and receipt, and a separate
`execution` array of actual lookup, resolution, and finish actions. Legacy
schema-1 raw-model traces keep their existing meaning and reader.

Use `trajectory_delegation_trace()` to extract attributed evidence and
`replay_delegation_trace()` from
`stateset_agents.evaluation.refund_delegation_trace` to independently regenerate
and compare both evidence channels. Wrong order IDs, amounts, receipts, metrics,
prompts, or terminal outcomes cause replay failure. Invalid or repeated model
intents apply no additional resolution and permanently fail the episode's
success criterion. Backend failures propagate as infrastructure errors.

The comparison runner is `examples/river_refund_delegation_study.py`:

```bash
python -m examples.river_refund_delegation_study prepare \
  --output outputs/river-delegation-pilot-20261020
python outputs/river-delegation-pilot-20261020/runtime/experiment.py validate \
  --output outputs/river-delegation-pilot-20261020
# Paid inference; prompts for a credential if RIVER_API_KEY is unset:
python outputs/river-delegation-pilot-20261020/runtime/experiment.py launch \
  --output outputs/river-delegation-pilot-20261020
# Offline audit after completion:
python outputs/river-delegation-pilot-20261020/runtime/experiment.py audit \
  --output outputs/river-delegation-pilot-20261020
```

Preparation currently uses the retained October 20 arithmetic candidate and
historical dataset index from this checkout. It freezes the packaged sources,
runner, checkpoint pointer, and input hashes before sampling. Both arms use the
same fixed weights, greedy sampling, 1,024 output tokens per trajectory, and a
four-turn limit. Their prompts and tools differ intentionally; compare whole
agent-system behavior rather than claiming a model-weight improvement.

The protocol defines 74 fresh validation cases and 170 fresh test cases per arm,
covering ordinary, window, and subtraction stress suites. IDs and fact
fingerprints are checked against historical training and evaluation inputs,
including the earlier 308-case unattempted test set. Input seeds can change only
prospectively for overlap rejection, never after observing model outcomes.

Each split's two attempts are sealed before either arm samples. Validation must
pass the declared overall, family, suite, and suite-family success and safety
gates before test attempts are sealed. There are no retries, training updates,
or automatic model promotions. The whole comparison is capped at 488 admitted
trajectories, 499,712 reserved output tokens, and 20 minutes of worker time.
The token reservation bounds output admission; it is not a dollar cap and does
not measure provider billing. This single-checkpoint synthetic pilot does not
establish production A+ quality or generalization to merchant policies.

The [October 20 comparison evidence](../benchmark_results/training_contracts/20261020/river-delegation-pilot/README.md)
records 74/74 validation and 170/170 test successes in both arms, with zero
violations or truncations. Delegation used 5,270 test output tokens versus
16,922 for the raw protocol (68.9% fewer), and two model tool turns versus three.
All 488 trajectories passed independent replay. This run shows an observed
efficiency gain at equal tested accuracy, with no training updates or measured
accuracy gain. Provider billed cost was unavailable.

## Recovery and authorization workload

`RefundRecoveryEnvironment` adds the separate `refund-recovery-v1` task. Its ten
families cover normal resolution, denied or revoked authorization, a transient
lookup failure, stale chargeback or refund facts, lost acknowledgments with and
without a commit, and unavailable or pending reconciliation. These faults are
explicit deterministic fixtures. Unexpected infrastructure exceptions propagate;
they are never converted into training rewards or safe-to-retry observations.

The agent must act on the trusted backend's current knowledge:

| Observed status | Required next action |
| --- | --- |
| `applied` | Finish with outcome `resolved`. |
| `unauthorized` | Finish with outcome `unauthorized`. |
| `retryable`, `retry_safe: true` | Retry the bound resolution intent. |
| `outcome_unknown` | Reconcile before retrying or finishing. |
| `not_applied`, `retry_safe: true`, `fenced: true` | Retry after the confirmed fence. |
| `pending` or `unavailable` reconciliation | Finish with outcome `handoff`. |

Finishing requires `{"tool":"finish","args":{"outcome":"resolved"}}`,
with the outcome changed to `unauthorized` or `handoff` when indicated. An
ambiguous submission cannot authorize a blind retry. A pending outcome cannot
be relabelled as a confirmed failure. Invalid actions permanently disqualify
success even when a later action corrects the mistake, and cannot create an
additional resolution effect.

The backend and its fences remain in-memory simulation. A production backend
must persist the request, enforce authorization and atomic fact checks, and
prevent an earlier attempt from committing after returning a retry-safe fence.
The fixture establishes none of those properties for a real payment provider.

Prepare and audit a training bundle without model calls:

```bash
python -m examples.refund_recovery --output /tmp/refund-recovery --seed 42
python -m examples.refund_recovery --output /tmp/refund-recovery --audit
```

The default bundle contains 160 train, 40 validation, and 80 test cases, balanced
over the ten families. Only train cases generate teacher transcripts. The
reference follows public tool observations and independently replays each
transcript. The audit regenerates canonical seed/split/count inputs, verifies
train-only provenance, rejects rehashed held-out substitutions, checks case IDs
and fact-fingerprint exclusions, and requires the original packaged source
identity. Existing bundles are never overwritten. The manifest is published
last; its absence indicates an incomplete bundle. All data is fictional and
has not been independently reviewed by humans.

`river_refund_recovery_factory()` in
`stateset_agents.remote.river_refund_recovery` records schema-3 recovery traces.
Model intents remain separate from backend attempts, commit evidence, and
reconciliation responses. `trajectory_recovery_trace()` checks attribution;
`replay_recovery_trace()` in
`stateset_agents.evaluation.refund_recovery_trace` verifies both evidence
channels and every knowledge transition. Schema-1 and schema-2 readers reject
schema-3 evidence, preserving the earlier benchmark meanings.

The October 20 recovery pilot freezes a train-only variance probe before model
sampling: one predetermined training case per family, four sampled rollouts
per case at temperature 1. It reports actual within-group reward variance.
Greedy validation and test use separate fresh cases, with a perfect validation
gate before test sampling. This probe performs zero optimizer updates: positive
reward variance is a prerequisite to investigate RL, not evidence that learning
has already occurred. The run permits at most 160 trajectories, 163,840 reserved
output tokens, no provider retries, and 15 minutes of worker time. Provider
billing remains separate from the output-token admission bound.

The [recovery pilot evidence](../benchmark_results/training_contracts/20261020/river-recovery-pilot/README.md)
records 37/40 sampled training successes and nonzero reward variance in 3/10
groups. Those sampled failures began with malformed JSON. Greedy validation
scored 35/40: five cases incorrectly reported `unauthorized` after an applied
denial or escalation. A confirmed policy denial/escalation is a resolved
workflow, distinct from an authorization failure. The strict validation gate
kept all 80 new test cases unattempted. All 80 sampled/evaluated outcomes and
160 teachers passed independent replay. The run performed zero optimizer
updates; it identifies training work rather than establishing an RL improvement.

## Recovery SFT continuation protocol

The follow-up recovery warmup study uses the same 160 audited training
transcripts without adding validation-derived examples. It starts from the
retained arithmetic candidate's weights, uses a fresh optimizer, and fixes one
epoch at batch size 16, rank 16, and learning rate `2e-5`: ten acknowledged SFT
updates. Its final checkpoint is fixed before sampling; no intermediate
checkpoint is selected by held-out outcomes.

The study freezes 80 fresh validation and 160 fresh test cases, excluding all
earlier training, prepared, and evaluated case IDs and order fact fingerprints.
The earlier recovery pilot's 80 test cases and arithmetic pilot's 308 test cases
are excluded and remain unattempted. Parent and warmup arms share prompts,
policy tools, token limits, and greedy evaluation settings.

Replay-based diagnostics count malformed action JSON separately from invalid
tool contracts, incorrect completion outcomes, unsafe retries, reconciliation
misuse, and premature finish. Safety gates require a perfect warmup validation
result before sealing either test arm. Test acceptance also requires strict
success improvement over the parent, no family regression, and no new error
category. A tie at perfect accuracy does not establish an accuracy improvement.

The post-warmup stochastic probe covers one training case per recovery family
and public terminal receipt type, four rollouts per case at temperature 1. This
selection covers applied refunds, denials, and escalations where present in the
training data, alongside unauthorized and handoff outcomes. It measures residual
reward variance and diagnoses formatting versus semantic errors; it performs
zero RL optimizer updates. Any later RL study needs fresh on-policy sampling.

The frozen October 20 plan permits ten SFT updates, 76 post-warmup training
rollouts, and at most 480 paired validation/test trajectories. It disables
provider retries, reserves at most 569,344 output tokens, and bounds worker time
to 25 minutes. Token reservations do not cap provider billing. This remains a
single-parent synthetic study, without automatic production promotion.

The [recovery SFT warmup evidence](../benchmark_results/training_contracts/20261020/river-recovery-warmup/README.md)
records ten acknowledged SFT updates on the unchanged 160 training transcripts.
On fresh paired cases, validation scored 79/80 for the parent and 80/80 for the
warmup; test scored 147/160 and 160/160 respectively. The warmup passed the
declared pilot gate with no policy violations, truncations, family regressions,
or new error categories. Thirteen test cases improved and none regressed; all
thirteen parent failures were incorrect completion outcomes. These results
describe one fixed checkpoint comparison on synthetic cases, not production
quality certification or evidence of RL learning.

The broader temperature-1 training probe scored 67/76, with reward variance in
6/19 groups. All nine failed rollouts contained malformed JSON, leaving a
specific target for a later RL experiment using fresh on-policy token/logprob
evidence. No RL updates occurred in this study. Independent review replayed
all 556 model outcomes and 160 teachers, regenerated the fresh evaluation
cases, and verified 670 earlier retained artifacts. The earlier blocked test
sets remain unattempted. The evidence archive records the completed run,
192 passing local contract tests, exact frozen sources, and its limits;
provider billing was unavailable.

## Recovery RL formatting pilot

The next frozen pilot starts from the accepted recovery SFT checkpoint and
permits one native CISPO optimizer update, with a fresh optimizer and no SFT.
It uses 19 unchanged training cases selected by family and public terminal
receipt type, four fresh on-policy rollouts per case at temperature 1, and
group-centered advantages. The actual generated token IDs, sampled log
probabilities, committed policy identity, and logical training datums are
retained. Any truncated training trajectory or invalid policy/token evidence
rejects the batch before forward/backward. A zero-variance batch consumes the
attempt without an optimizer update or replacement batch.

Evaluation cases exclude all earlier training, prepared, and evaluated inputs.
The fixed final RL checkpoint first faces 80 paired greedy safety cases, then
20 paired validation cases with four stochastic rollouts each. A failed gate
keeps the separate 40-case, four-rollout stochastic test unattempted. Greedy
validation requires perfect RL safety. Stochastic validation permits a tie but
requires no overall, family, or diagnostic-category regression, no semantic
errors, and no truncations. Test additionally requires strictly more successful
rollouts and fewer formatting-error rollouts than the SFT parent. Replicates
are grouped by case; their outcome counts are descriptive, not independent
merchant or training-seed replications.

The plan bounds the experiment to one RL update, at most 716 sampled
trajectories, 733,184 reserved output tokens, no provider retries, and 25 minutes.
These token reservations do not establish a dollar billing cap. Neither the
earlier blocked test sets nor the completed SFT evaluation cases are reused.

The [recovery RL pilot evidence](../benchmark_results/training_contracts/20261020/river-recovery-rl-pilot/README.md)
records one acknowledged native optimizer update and a direct committed-policy
successor checkpoint. The fresh pre-update training batch scored 75/76, with
one formatting failure and reward variance in 1/19 groups. Independent token
alignment and group advantages reproduced all four usable logical training
datums exactly; the provider reported a finite gradient norm.

Both checkpoints scored 80/80 on fresh greedy validation. Stochastic validation
scored 78/80 for the SFT parent and 80/80 for RL across 20 cases with four
rollouts each. The separate stochastic test scored 155/160 and 160/160 across
40 cases with four rollouts each. Five test cases had one more successful RL
rollout, and none regressed. All five parent test failures were malformed JSON;
the RL arm had no policy violations, semantic errors, or truncations. Every
declared gate passed without selecting or retrying a checkpoint on test scores.

Independent review replayed all 716 outcomes, verified 691 earlier retained
artifacts, and reconstructed the fresh evaluation inputs. Fourteen offline
study tests passed with provider RPCs blocked. The run finished in about
16 minutes within its fixed limits; billed cost was unavailable. This single
synthetic comparison supports further replication, without establishing
repeatable RL gains, production A+ quality, or real payment recovery guarantees.

## Six-seed recovery RL replication

The initial consecutive-seed protocol was [stopped and archived](../benchmark_results/training_contracts/20261020/river-recovery-six-seed-aborted/README.md)
after detecting overlapping sampler streams. It admitted 24 training
trajectories but submitted zero optimizer updates and sampled no evaluation
cases. Its original plan and frozen sources remain unchanged, and its prepared
evaluation cases are excluded from the corrected protocol.

The corrected protocol repeats the one-update pilot from the same accepted
SFT parent for six fixed seeds separated by 350,000,000. It preserves the 19
training cases, four fresh on-policy rollouts per case, optimizer, learning
rate, loss, and token limits. Each seed gets one training batch and at most
one acknowledged RL update with a fresh optimizer. A skipped, failed, or
uncertain attempt is never replaced or omitted from the cohort.

`validate_river_replication_seeds()` audits River's inspected group/member/sample
seed schedule across all permitted indices, including modulo wraparound. The
bound covers 80 groups, four members, and up to 1,024 sample calls per member.
The corrected plan pins the inspected SDK engine source and reruns the audit
before provider work. Separate seed identities avoid deterministic stream
reuse; statistical independence still depends on the provider's generator.

All six training/validation attempts run, with three seed workers at a time.
Each qualifying seed must acknowledge an update and pass the original 80-case
greedy and 20-case, four-rollout stochastic validation gates. If any seed
does not qualify, the entire test cohort remains unattempted. Otherwise, all
six fixed checkpoint pointers and all twelve test arms are sealed before
sampling any test case. The common fresh test set contains 40 cases with four
rollouts per case per arm and seed. Every evaluation case excludes all earlier
training, prepared, and evaluated identities and order fact fingerprints.

`stateset_agents.evaluation.replication.paired_seed_case_summary()` aggregates
rollouts within case. Its exact two-sided sign-flip calculation uses all six
training-seed paired differences, retains ties, and rejects missing or extra
seeds/cases. A seeded 10,000-replicate bootstrap resamples training seeds and
case clusters within each recovery family, sharing the resampled cases across
seeds. Replication acceptance requires every seed to pass the unchanged pilot
test gate, an exact p-value at most 0.05, and a positive lower endpoint of the
95% bootstrap interval. These conditional estimates depend on sign
exchangeability and resampling assumptions; they do not replace reviewed
production data or independent merchant evidence.

The frozen plan permits at most six RL updates, 4,296 trajectories, and
4,399,104 reserved output tokens. Each phase worker has 25 minutes; the
coordinator has a 70-minute total deadline. Provider retries and sealed-phase
retries are disabled. Token reservations do not enforce a dollar cost cap.
The protocol does not automatically promote a checkpoint or choose a best
seed using test results. The older blocked test sets remain excluded.

The [completed six-seed evidence](../benchmark_results/training_contracts/20261020/river-recovery-replication/README.md)
records six acknowledged native CISPO optimizer updates and six committed
successor checkpoints. Every seed passed both validation gates. On the shared
40-case stochastic test, with four rollouts per case per seed, the SFT parent
passed 954/960 and RL passed 960/960. Four seeds improved and two tied; none
regressed. All six parent failures were formatting errors. RL had no observed
semantic errors, policy violations, or truncations.

The paired gain was 0.625 percentage points, with an exact two-sided seed
sign-flip p-value of 0.125 and a seed/case bootstrap 95% interval of
[0, 1.458] percentage points. The two tied seeds also failed the strict
per-seed improvement requirement. Replication acceptance therefore failed,
and no checkpoint was promoted. These results demonstrate working RL updates
and a promising formatting gain on synthetic cases, without establishing a
repeatable quality improvement. The 960 rollouts are clustered within 40
shared cases and six training seeds, rather than 960 independent cases.

Offline review replayed all 4,296 outcomes, reconstructed all 72 usable native
training datums, and verified 718 earlier retained artifacts unchanged.
Verification passed 136 regression tests, seven cohort-control tests, three
additional supervisor tests, and type checking for all 456 packaged sources.
The separate industry inventory verified 24 dataset artifacts; ecommerce,
logistics, and SaaS datasets remain synthetic and not independently reviewed.
Provider billing was unavailable, and this study does not certify production
A+ quality.
