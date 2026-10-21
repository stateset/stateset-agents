# Six-seed River recovery RL replication — October 20, 2026

Six fixed one-batch attempts from the same [accepted recovery SFT parent](../river-recovery-warmup/README.md).
Status: **completed**. Qualifying validation seeds: **6/6**.
Acknowledged RL updates: **6**. New SFT updates: **0**.
Replication accepted: **False**. Test attempted: **True**.
No automatic promotion or best-seed nomination occurred.

| Seed | Train passed | Variable groups | Native datums | RL updates | Greedy validation: parent → RL | Stochastic validation: parent → RL | Test: parent → RL |
| ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 20261101 | 72/76 | 4/19 | 16 | 1 | 80/80 → 80/80 | 80/80 → 80/80 | 159/160 → 160/160 |
| 370261101 | 75/76 | 1/19 | 4 | 1 | 80/80 → 80/80 | 80/80 → 80/80 | 160/160 → 160/160 |
| 720261101 | 71/76 | 5/19 | 20 | 1 | 80/80 → 80/80 | 77/80 → 80/80 | 160/160 → 160/160 |
| 1070261101 | 73/76 | 3/19 | 12 | 1 | 80/80 → 80/80 | 78/80 → 80/80 | 158/160 → 160/160 |
| 1420261101 | 73/76 | 3/19 | 12 | 1 | 80/80 → 80/80 | 80/80 → 80/80 | 158/160 → 160/160 |
| 1770261101 | 74/76 | 2/19 | 8 | 1 | 80/80 → 80/80 | 80/80 → 80/80 | 159/160 → 160/160 |

## Prespecified test inference

Parent passed **954/960**; RL passed **960/960**.
Paired success-rate difference: **0.625%**.
Exact two-sided seed sign-flip p-value: **0.125**.
Seed/case bootstrap 95% interval: **[0.000%, 1.458%]**.
These estimates condition on 40 shared synthetic case clusters and six
training seeds. Four rollouts within a case are aggregated before
inference. The exact calculation retains ties and enumerates all 64
seed sign patterns; the 10,000-replicate bootstrap resamples seeds and
case clusters within families, sharing resampled cases across seeds.
Sign exchangeability and resampling assumptions apply. There are no
independent merchant replications or production quality claims.

## Replication protocol

Every seed starts from the same SFT checkpoint with a fresh optimizer.
The 19 training cases and public-receipt coverage are unchanged from the
pilot. Each receives four fresh on-policy samples at temperature 1.
One logical CISPO batch permits at most one optimizer update: group-centered
advantages without standardization, token normalization, eps_max 6, Adam
learning rate 1e-5, LoRA rank 16, and zero policy staleness. Any invalid
native token/logprob evidence, committed-policy mismatch, or training
truncation rejects the batch before forward/backward. Zero signal consumes
the attempt without a replacement batch or optimizer update.

All six training/validation attempts run, even when another seed fails.
Each seed needs an acknowledged update, 80/80 safe greedy validation
cases, and no overall/family/error-category regression on 20 stochastic
validation cases with four rollouts each. Semantic errors or truncation
block validation. Every seed must qualify before all six fixed final
checkpoints and all twelve test arms are sealed. Test then uses 40 cases
with four rollouts each; every seed must strictly improve successes and
formatting without family/category regression, semantic error, or truncation.
Acceptance additionally requires exact two-sided p≤0.05 and a positive
lower bootstrap endpoint. No seed can be dropped, substituted, or selected
using test outcomes. There is no checkpoint search after evaluation.

All seeds share fresh cases. Parent/RL arms share the same per-seed
sampling configuration. Inputs exclude every historical training,
prepared, and evaluated case ID and order fact fingerprint, including
the prior pilot, completed SFT comparisons, and the stopped protocol.

## Sampler stream audit and stopped protocol

The [initial consecutive-seed protocol](../river-recovery-six-seed-aborted/README.md)
was stopped after finding deterministic member-offset stream reuse.
It admitted 24 training trajectories (24,576 reserved output tokens),
submitted zero optimizer updates, and sampled no evaluation cases. Its
plan, partial state, and original frozen sources are preserved unchanged.
Those admissions and unknown billed costs remain separate from this study.

This corrected prospective protocol uses fixed run seeds spaced by
350,000,000. The packaged audit checks all possible differences for 80
groups, four members, and up to 1,024 token-bearing sample calls per
member, including modulo-2³¹ wraparound. It pins the inspected SDK engine
source and reruns the audit before provider work. Distinct sampler seed
identities prevent direct stream reuse; they do not prove PRNG independence.

## Verification and limits

Independent review replayed **4296 outcomes** and reconstructed
native logical datums from generated token IDs, sampled logprobs, group
rewards, and masks without invoking River's batch builder. It checked
server-committed parent/child policy receipts, fresh input regeneration,
attempt seals, diagnostics, per-family gates, and whole-cohort decisions.
Admitted trajectories: **4296**; admitted without a retained/replayed final outcome: **0**.
Uncertain activity is not relabelled as zero updates or successful learning.
All **718 prior retained artifacts** matched their hashes.
The earlier recovery pilot's 80 and arithmetic pilot's 308 blocked test
cases remain unattempted. Local hashes and client receipts provide
consistency evidence, not independent provider or weight-byte attestation.

Verification: **136 regression tests**, including **34 analysis/stream-audit tests**,
and **7 cohort-control tests** passed. Type checking passed all **456**
packaged sources. Only the new replication analysis module changes the
preceding 455-source package; recovery environment/reward/trainer behavior
is unchanged. Test transport makes no paid provider calls.
Three additional offline supervisor checks verified deadline rejection,
including a late zero exit and a forced kill, without retrying the worker.

The separate industry inventory verified 24 existing dataset artifacts.
Ecommerce, logistics, and SaaS each declare 512 training, 64 validation,
and 128 test rows, with status not_independently_reviewed. Production
validation still requires reviewed source conversations, source-group
and temporal isolation, and retained domain review/outcome evidence.

Study bounds: at most six optimizer updates, 4,296 trajectories,
4,399,104 reserved output tokens, three seed workers at a time, 25 minutes
per phase worker, and 70 minutes overall. Provider and sealed-phase retries
are disabled. Billed dollar cost is unavailable; token reservations are
not a dollar cap. This study plus the stopped attempt reserve at most
4,423,680 output tokens across 4,320 admissions; the overall acknowledged
RL update bound remains six because the stopped attempt submitted none.

Archive byte verification checked **808 files**. Frozen package/runner
sources, inspected SDK seed source, inputs, partial or complete native
trainer states, datums, checkpoint receipts, attempt seals, full available
reports, diagnostics, and executable review scripts are retained.

This conditional synthetic replication does not certify production A+
quality, reviewed industry generalization, real payment recovery, durable
idempotency, optimizer continuity, or independent merchant performance.
