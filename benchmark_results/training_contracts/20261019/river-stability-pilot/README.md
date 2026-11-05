# One-parent River stability pilot

**Prespecified pilot acceptance: FAILED.** The candidate remains unpromoted.
Primary regression strata: family:partial_refund, stress_source_family:partial_refund, stress_window:8.

The parameter arm gains 14 successes (7 percentage points) over control, but the
partial-refund family declines from 19/19 to 18/19, stress cases sourced from
partial-refund orders decline from 8/9 to 7/9, and eight-day stress truncation
increases from zero cases to one. These overlapping strata fail the frozen
nonregression gates. The secondary comparison against the original passes.

Qwen/Qwen3.5-9B, rank 16, training seed 2026104001, fresh evaluation
case seed 2026103001. The first retained original SFT parent
(2026101901) was fixed before evaluation. It has already participated in earlier
pilots: this is a conditional, exploratory experiment with one parent, not an
independent multi-seed replication or population-significance result.

The primary comparison is broader policy-parameter contrasts versus additional
legacy-curriculum SFT. Both train from the same original SFT checkpoint with a
fresh optimizer, one epoch, 512 rows, 32 updates, batch 16, learning rate 2e-5,
rank 16, shuffle seed and family counts. The control preserves all 255 legacy
curriculum rows and repeats only those existing facts to meet the new arm's
family quotas. Near-matched local input exposure differs by 0.063%; supervised
assistant tokens differ by 0.053%, below the frozen 0.5% tolerance. These limits
do not establish equal provider compute or billed cost.

The parameter arm uses the training-only `policy_parameter_contrasts_v1` policy
from the preceding offline preparation. Teacher transcripts were re-executed in
fresh sandboxes. All 304 evaluation cases are disjoint from historical inputs,
all new training facts, unused earlier preparations and the other five parents'
fresh holdouts. Validation contains 32 ordinary plus 72 stress cases, test 128
ordinary plus 72 stress cases. Stress windows 5/18/40 and 8/25/75 remain distinct
from the new 3/10/21/60/90 training windows. Original 14/30 training cases remain.
Private stress metadata is never included in model observations.

Both trained checkpoints are fixed final-epoch checkpoints. The parameter arm
must show no validation success, policy-violation or truncation regression versus
both control and original across all, family, ordinary/stress, stress source
family and stress-window marginal strata. Failure stops without test. Otherwise
all three test attempts are durably sealed before sampling. Primary test
acceptance requires at least 3 percentage points gain versus control and the
same safety strata, plus nonregression versus original. No tuning or retry.

| Split | Arm | Successes | Ordinary | Stress | Violation / truncated cases |
|---|---|---:|---:|---:|---:|
| validation | original | 85/104 | 29/32 | 56/72 | 19 / 16 |
| validation | control | 95/104 | 32/32 | 63/72 | 9 / 6 |
| validation | parameter | 103/104 | 32/32 | 71/72 | 1 / 0 |
| test | original | 164/200 | 110/128 | 54/72 | 36 / 29 |
| test | control | 183/200 | 122/128 | 61/72 | 17 / 12 |
| test | parameter | 197/200 | 127/128 | 70/72 | 3 / 3 |

Execution exit code: 0, timed out:
False; elapsed
1293.90 seconds. Pilot result:
`{'pilot_accepted': False, 'plan_hash': '505bbd0b8a9a5d13ee99d097aef6fa5c5a6d5231ba63cd403768437d9e9d99d8', 'quality_certification': False, 'status': 'completed', 'test_attempted': True}`. Reconstructed decisions: `{'validation': {'passed': True, 'parameter_vs_control_gain': 0.07692307692307693}, 'test': {'passed': False, 'parameter_vs_control_gain': 0.07}}`.

Acknowledged added SFT updates: 64.
Replayed 1024 teacher transcripts and
912 completed evaluation trajectories.
Output reservations: 933,888 /
933,888 tokens. Reservations are admission limits, not
actual generated tokens or billed dollars. Actual dollar cost is unavailable.
The original workload caps were 64 SFT updates, 933,888 reserved output tokens,
30 minutes, SDK requests bounded to 180 seconds and no operational or transport
retries. No RL, deployment, or additional provider calls during review.

Eight new control/gate tests passed, including masked stratum regressions,
corrupted aggregate rejection and exact teacher replay. The packaged sources
are unchanged from the prior 182-test, 446-source mypy validation. The worker
socketpair wakeup probe passed outside the sandbox. Independent offline review
recomputed each saved gate and replayed the executed environment actions.
605 older retained evidence artifacts verified unchanged.

The archive retains exact package/runner sources, matched datasets, untouched or
consumed heldouts, durable attempts, provider pointer lineage, logs, token ledger,
and replay evidence. Every member was byte-verified. No credential is retained.
Hashes establish local identity, not provider attestation. This synthetic pilot
does not establish A+ performance or independently reviewed production readiness.
