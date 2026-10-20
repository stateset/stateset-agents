# River refund recovery pilot — October 20, 2026

Inference-only evaluation of the retained Qwen/Qwen3.5-9B arithmetic
candidate on the separate synthetic `refund-recovery-v1` protocol.
This pilot performed **zero training updates** and promoted no model.

- Train-only stochastic probe: **37/40** successful rollouts.
- Nonzero within-group reward variance: **3/10 groups** (four rollouts per group).
- Greedy validation: **35/40**, with a **12.5%** policy-violation rate.
- Test attempted: **False**; all 80 new test cases remain unattempted.

## What failed

All three sampled training failures began with malformed JSON. Their
positive within-group variance is an output-reliability learning signal;
it does not establish learning about uncertain commits or authorization.

All five greedy validation failures incorrectly finished with outcome
`unauthorized` after the tool confirmed an applied denial or escalation.
An applied denial/escalation resolves the workflow; it does not indicate
that the caller lacked authorization. The model must finish `resolved`.
No duplicate resolution or unauthorized payment effect was applied.

## Training data and provenance

The ten fault families cover clean resolution, denied/revoked
authorization, transient lookup failure, stale chargeback/refund facts,
lost acknowledgments with and without commits, and unavailable/pending
reconciliation. Labels and commit knowledge remain private; model
messages expose only the tool's current knowledge and receipts.

The frozen bundle contains **160 train, 40 validation, and 80 test cases**.
Only training cases produce teacher transcripts. The reference follows
public observations and each transcript is independently replayed.
Canonical split/seed/count reconstruction and artifact hashes reject
held-out substitution or transcript fabrication, including rehashed edits.

All **160** teachers are usable by the River SFT batch builder; maximum
sequence length was **423** tokens, with no truncation.
Assistant loss masks exclude backend observations. The batch builder
intentionally omits trailing prompt-only terminal observations.

The train probe selected one case per family before sampling, four
rollouts at temperature 1. Validation used temperature 0. A native SDK
transport check verifies explicit zero reaches the sampling session;
provider-side sampling behavior is not independently attested.

## Limits and verification

One attempt, no provider retries, at most 160 trajectories and 163,840
reserved output tokens, with a 15-minute worker deadline. Only 80
trajectories were admitted: 40 training-probe and 40 validation cases.
Validation failure stopped the run before the test-attempt seal.
Actual billed cost was unavailable; the output budget is not a dollar cap.

Independent review replayed **80 model outcomes** plus all
160 teachers. All **652 earlier retained artifacts** matched their hashes.
The prior arithmetic pilot's 308 test cases also remain unattempted.

Local checks: **1,181 regression tests passed**, one skipped; two added
outcome-contract checks passed; **22 native SDK checks passed** with
scripted transport and provider clients forbidden. Type checking passed
for **455 packaged source files**. The archive includes the full frozen
runtime, controller, inputs, traces, test sources, and offline review scripts.

Archive byte verification checked **535 files**. Local hashes are
consistency checks, not independent provider attestation.

All policies and faults are fictional, in-memory fixtures. This pilot
does not prove real payment persistence, production A+ quality, live
optimizer recovery, or an RL learning improvement.

The next training study should use train-only examples to distinguish
authorization from successfully applied policy denials/escalations,
measure format versus semantic errors separately, and reassess rollout
reward variance before selecting an RL budget.
