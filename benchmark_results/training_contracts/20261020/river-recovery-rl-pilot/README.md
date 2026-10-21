# River recovery RL formatting pilot — October 20, 2026

One fixed native CISPO batch from the [accepted recovery SFT checkpoint](../river-recovery-warmup/README.md).
Acknowledged RL optimizer updates: **1**. New SFT updates: **0**.
Declared pilot acceptance: **True**. Test attempted: **True**.
No automatic production promotion occurred.

| Stage | SFT parent | RL | Parent format-error rollouts | RL format-error rollouts |
| --- | ---: | ---: | ---: | ---: |
| greedy-validation | 80/80 | 80/80 | 0 | 0 |
| stochastic-validation | 78/80 | 80/80 | 2 | 0 |
| test | 155/160 | 160/160 | 5 | 0 |

| Paired stage | Cases with more RL successes | Cases with fewer RL successes | Cases tied |
| --- | ---: | ---: | ---: |
| greedy-validation | 0 | 0 | 80 |
| stochastic-validation | 1 | 0 | 19 |
| test | 5 | 0 | 35 |

Four stochastic rollouts share a case. Counts are descriptive and must
not be treated as independent case, merchant, or training-seed replications.
A pilot pass alone cannot establish repeatable learning improvement.

## Native training evidence

The fresh on-policy batch passed **75/76 rollouts**,
with nonzero within-group reward variance in **1/19 groups**.
Training case selection was carried forward unchanged from the SFT plan:
first training case per family and public terminal receipt type, four
samples per case at temperature 1, top-p 1, top-k -1, and seed 2026109601.
Earlier sampled traces were not replayed into training.

The native trainer uses group-centered advantages without standardization,
token normalization, CISPO eps_max 6, Adam learning rate 1e-5, LoRA rank 16,
one logical batch, and zero policy staleness. Starting weights came from
the accepted recovery SFT checkpoint with a fresh optimizer. This does
not establish optimizer continuity from SFT.

Every training outcome was exactly replayed before forward/backward.
Any training truncation, missing committed policy identity, invalid native
token/logprob data, or mixed policy rejects the batch. A zero-variance batch
consumes the attempt without an update or replacement. Generated token IDs
and sampled logprobs remain in the native trajectory states; assistant text
is used for tool execution and reward inspection, never retokenized as
replacement training actions.

Independent token alignment and group advantages reconstructed **4 logical datums**
without calling River's batch builder. Their full bytes matched the
persisted datums and native reconstruction. The optimizer acknowledgement
and subsequent server read bind the child policy to the initial committed
policy. These are client-retained protocol receipts, not weight-byte or
independent provider attestation.

Provider-reported gradient norm: `0.02118874341249466`;
provider-reported loss: `-4.4481439545052126e-05`. These are
training diagnostics, not held-out quality evidence.

Training error counts: `{"completion_outcome": 0, "format": 1, "premature_finish": 0, "reconciliation": 0, "tool_contract": 0, "unsafe_retry": 0}`.

## Frozen evaluation and gates

The final checkpoint is fixed after one batch. There is no validation- or
test-based checkpoint selection. Fresh cases exclude every historical
training, prepared, and evaluated case ID and order fact fingerprint.
Both arms share prompts, tools, seed, six model turns, 1,024 generated
tokens, 256 tokens per turn, and an 8,192-token context limit.
Report stage-level temperature/group_size fields override the shared
base settings block: greedy uses temperature 0 and one rollout; stochastic
uses temperature 1 and four rollouts. The review summary records the
effective settings explicitly; frozen sampler construction enforces them.

First, the candidate must pass 80/80 paired greedy validation cases
without violations or truncation. Next, 20 validation cases receive four
stochastic rollouts per arm. No overall, family, or diagnostic-category
regression, semantic error, or truncation is allowed. A validation tie
is permitted. Only then may the 40-case, four-rollout stochastic test
be sealed. Test additionally requires strictly more successful rollouts
and fewer formatting-error rollouts than parent. Both arms are sealed
before sampling each stage; uncertain or failed attempts are not retried.

| Stage / semantic category | Parent affected rollouts | RL affected rollouts |
| --- | ---: | ---: |
| greedy-validation / completion_outcome | 0 | 0 |
| greedy-validation / tool_contract | 0 | 0 |
| greedy-validation / unsafe_retry | 0 | 0 |
| greedy-validation / reconciliation | 0 | 0 |
| greedy-validation / premature_finish | 0 | 0 |
| stochastic-validation / completion_outcome | 0 | 0 |
| stochastic-validation / tool_contract | 0 | 0 |
| stochastic-validation / unsafe_retry | 0 | 0 |
| stochastic-validation / reconciliation | 0 | 0 |
| stochastic-validation / premature_finish | 0 | 0 |
| test / completion_outcome | 0 | 0 |
| test / tool_contract | 0 | 0 |
| test / unsafe_retry | 0 | 0 |
| test / reconciliation | 0 | 0 |
| test / premature_finish | 0 | 0 |

## Audit and limits

Independent review replayed **716 outcomes** and regenerated
all fresh cases. It checked training source bytes, exact native spans,
group advantages, masks, logical datums, committed policy progression,
attempt seals, error classifications, family cells, and final gates.
All **691 prior retained artifacts** matched their hashes.
The earlier recovery pilot's 80 and arithmetic pilot's 308 blocked test
cases remain unattempted. Completed SFT evaluation cases were also excluded.

Prelaunch verification: **14 offline tests passed**, including the real
River engine/trainer with scripted transport, one acknowledged optimizer
step, exact submitted-batch reconstruction, zero-update behavior, rejection
of inexact samples/truncations/missing policy support, and test sealing.
All provider RPCs were blocked during those tests. The packaged 455-source
implementation is unchanged from the preceding validated recovery study.

Bounds: one RL optimizer update, at most 716 trajectories, 733,184 reserved
output tokens, no transport retries, one attempt per stage, and 25 minutes
of worker time. Provider billed cost was unavailable. Token reservations
are not a dollar cap. Optimizer acknowledgement alone does not establish
nonzero gradients or a useful learned policy.

Archive byte verification checked **527 files**. The archive contains
frozen package/runner sources, exact inputs, native states/datums, committed
policy receipts, checkpoint pointers, traces, review scripts, and the
tested SDK source snapshots. Local hashes establish consistency only.

This remains a single-parent synthetic pilot. It does not certify A+
production quality, independent merchant generalization, real payment
recovery, durable idempotency, or repeatable RL improvement.
