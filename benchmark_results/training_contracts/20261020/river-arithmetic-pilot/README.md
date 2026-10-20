# Borrow-boundary River SFT continuation pilot

**Pilot accepted: False.** Result: `{'pilot_accepted': False, 'plan_hash': '5bdb9e6569bf9c32a7ec9048bf15042f371100d55c003dcfa81219dd0cda21db', 'status': 'validation_blocked_test_not_attempted', 'test_attempted': False}`.
Regression strata: `{'validation': ['family:already_refunded', 'stress_source_family:already_refunded', 'stress_window:40', 'suite:window', 'suite_family:window:already_refunded', 'window_source_family:already_refunded', 'window_window:40']}`. No deployment or production certification.

Validation stopped the pilot before any test attempt. Both arms passed 157/158.
Arithmetic improved from 53/54 to 54/54, while the existing window suite declined
from 72/72 to 71/72. The candidate denied an already-refunded order with reason
outside_window before correcting to already_refunded. That priority violation
fails the strict success rule even though no financial ledger error is applied.
The fully-refunded family declines from 13/13 to 12/13. The 308 test cases remain
unattempted, with no test seal or sampling. The candidate remains unpromoted.

The previous policy-parameter pilot passed 197/200 test cases but failed its
partial-refund safety gates. Its three failures involved subtraction; a separate
validation failure involved order-ID copying. This exploratory new hypothesis
uses only canonical training facts, never those exposed amounts, IDs or answers.
Both new arms load that retained, unpromoted candidate with fresh optimizers.
This is one conditional parent, not a new base-model initialization or multi-seed
population result. Qwen/Qwen3.5-9B, rank 16, seed 2026106001.

The packaged, opt-in `borrow_boundary_contrasts_v1` curriculum retains all 512
parameter-curriculum rows and adds 189 training-only amount contrasts: 21 eligible
partial-refund sources, each with a nine-case payment/refund grid at offsets
-1/0/+1 around thousand-cent anchors on the inclusive final day. Small-order
clipping keeps all new cases partially refunded. Other order facts and untrusted
notes remain intact. Teacher transcripts perform the exact refund and finish
explicitly. Audit reconstructs every amount, group, lineage and teacher action.
Both prior curriculum policies remain byte-reproducible: 12 old bundles checked.

Control repeats only earlier parameter-curriculum facts. Both arms use 701 rows,
44 added SFT updates, identical family counts, one epoch, batch 16, lr 2e-5,
rank, initialization and shuffle seed. Local input-token exposure differs by
0.194%; supervised assistant tokens differ by 0.070%, within the frozen 0.5%
tolerance. Amount/window mixtures, provider compute and billed dollars are not
exactly matched; this does not isolate a single mechanistic cause of improvement.

Fresh case seed: 2026107001. Each arm has 158 validation cases
(32 ordinary, 72 window stress, 54 arithmetic stress) and 308 test cases
(128 ordinary, 72 window stress, 108 arithmetic stress). Arithmetic training uses
thousand-cent anchors; validation uses tens and test uses hundreds. Every
arithmetic cell draws its own source order. Day offsets -1/0 and payment/refund
offsets -1/0/+1 are represented. All heldouts are ID/fact-disjoint from historical
cases, all current training facts, unused earlier preparations and each other.
Private stress metadata is never exposed in model observations.

Fixed final-epoch checkpoints. Validation must show no success, policy-violation
or truncation regression overall and across the frozen families, suites,
suite-families, stress source families/windows, and arithmetic amount/day offsets.
Existing window-stress source strata are preserved separately so new arithmetic
gains cannot mask their regressions. If validation fails, test stays unattempted.
Otherwise both test attempts are sealed before sampling. Test must additionally
gain at least one arithmetic success versus control. No tuning or retries.

| Split | Arm | Successes | Ordinary | Window stress | Arithmetic stress | Violation / truncated cases |
|---|---|---:|---:|---:|---:|---:|
| validation | control | 157/158 | 32/32 | 72/72 | 53/54 | 1 / 1 |
| validation | arithmetic | 157/158 | 32/32 | 71/72 | 54/54 | 1 / 0 |
| test | control | not attempted | — | — | — | — |
| test | arithmetic | not attempted | — | — | — | — |

Acknowledged added SFT updates: 88.
Replayed 1402 teacher transcripts and
316 completed evaluation trajectories.
Execution exit code 0, timed out
False, elapsed 908.14s.
Output reservations 323,584 /
954,368. Reservations are admission allowances, not actual
generated usage or billed dollars. Actual cost remains unavailable. Original
caps: 88 SFT updates, 954,368 reserved output tokens, 40 minutes, request timeout
180s, no transport or operational retries. No RL or deployment ran.

199 unit/CLI tests and nine pilot checks passed. Full mypy passed 447 packaged
sources; Ruff, Black, isort and repository hygiene checks passed. The SDK worker
socketpair probe passed outside the sandbox. Independent offline review
reconstructed the source data, fresh holdouts and control, replayed all published
actions and recomputed every saved gate. 620 older evidence artifacts
verified unchanged. The archive retains exact package and runner sources,
datasets, provenance, attempts, pointer lineage, logs, token ledger and audits;
all 652 archive members were verified byte-for-byte. No credential is
retained. Local hashes do not establish provider attestation, independently
reviewed industry quality, or A+ production readiness.
