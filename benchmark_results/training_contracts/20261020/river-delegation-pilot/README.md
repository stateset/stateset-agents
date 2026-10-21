# River refund delegation comparison — October 20, 2026

Exploratory inference-only comparison of the same fixed Qwen/Qwen3.5-9B
arithmetic-candidate checkpoint under raw-model and delegated-tool protocols.
This measures agent-system behavior; tool-selected policy and arithmetic
do not establish model learning. No training or model promotion occurred.

| Split | Raw | Delegated | Raw violation rate | Delegated violation rate |
| --- | ---: | ---: | ---: | ---: |
| validation | 74/74 | 74/74 | 0.00% | 0.00% |
| test | 170/170 | 170/170 | 0.00% | 0.00% |

Declared safety/nonregression gates passed: **True**.
Independent replay checked **488 outcomes** and reconstructed
all 244 fresh input cases. Both intent and actual tool execution evidence
were checked, including receipts, observations, metrics, and terminal states.

Both arms used greedy sampling, four model turns, and 1,024 output tokens
per trajectory. Prompts and tools differ intentionally. Validation gates
cover overall, family, suite, and suite-family cells. Both arm attempts
were sealed before each split's sampling; test required validation success.

The run allowed one attempt, disabled provider retries, bounded worker time
to 20 minutes, and reserved at most 499,712 output tokens. This is an output
admission bound, not a dollar cap. Actual billed cost was unavailable.

All **637 earlier retained artifacts** matched their
published hashes. The prior arithmetic pilot's 308 test cases were excluded
from this study and remain unattempted.

Local verification: 1,024 unit/regression checks passed, one skipped;
Eight additional protocol gate checks passed. Also, 20 native River SDK
checks passed with scripted transport and no paid
clients; type checking passed for 451 packaged source files. The archive
includes the full frozen runtime, exact inputs, attempt seals, outcome
traces, offline audit scripts, and the unit-test XML report.

Archive byte verification checked **524 files**. Local hashes
provide consistency checks, not independent provider attestation.

This single-checkpoint study uses fictional synthetic policies. It does
not establish production A+ quality, independently reviewed merchant
generalization, live payment idempotency, or optimizer recovery.

## Observed test efficiency

Output tokens fell from **16,922** to
**5,270** (68.9% fewer).
Median trajectory time was **8.94 s** for raw
and **5.85 s** for delegated.
Timing reflects this concurrent provider run, not a controlled latency benchmark.

Model-requested tool turns fell from three to two per case; both
protocols still executed three backend actions per case (lookup,
resolution, finish). Both arms passed every case, so this run shows
an observed output-efficiency gain and no measured accuracy gain.
