# River selection hardening — 2026-10-19

Confirmed skipped optimizer batches can no longer win checkpoint selection by rescoring unchanged weights. The native manifest declares `initial_and_unskipped_batches_v1`. Selection keeps the initial checkpoint eligible, excludes batches with observed `train/updated=0`, and preserves earliest-batch ties. Unknown recovery activity stays eligible and explicit. Every validation trace, including excluded batches, must still replay successfully.

Before held-out sampling, the campaign writes `validation_selection.json` with eligible/excluded batches, the selected checkpoint, source hashes, and optimizer activity observed through the selected batch. Test reports bind that record and expose selected activity separately from whole-run activity. The study auditor independently recomputes these fields and rejects missing or altered records, including mutually consistent fabricated summaries. Evaluation-only arms cannot claim native RL activity.

## Verification

- The first focused run had 183 passing tests and one 60-second timeout inside filesystem `fsync`. That case passed with normal sync calls on a separate run with a 180-second deadline (49.23 seconds). Five additional selection-record corruption tests passed. Together, all 189 unique focused cases have passing results; the initial failed attempt remains disclosed.
- Type checks passed across 439 packaged Python files. Ruff, Black, isort, repository hygiene, and whitespace checks passed for the affected scope.
- A validation-only retrospective replay of all six archived study histories selected the same checkpoints under the new policy. This is post-study diagnostic evidence, not a new preregistered experiment or held-out quality result. No original report was changed and no provider calls were made.
- **The native SDK regression suite did not complete successfully.** The ordinary run stalled in cancellation/cleanup. Bounded diagnostics observed filesystem sync waits and event-loop waits. Runs with sync calls elided, cooperative scripted sampling, and batching delay disabled also failed to establish a complete passing result. The specific selection-flow test likewise remained incomplete. These experimental harness settings were not applied to packaged runtime. The SDK stall remains an open integration blocker; this change is not fully verified end to end.

The two XML files retain the five corruption tests and the recovered-commit retry. `focused-checks.json` records the initial failure and verification limits. Partial `sdk-*-checks.txt` files are incomplete logs, not successful test receipts. Diagnostic runners are retained for reproduction. The source archive contains the canonical package, relevant tests and documentation; `source-implementation.json` binds all 439 package files. Only four package files changed from the frozen six-seed study; the other 435 match it.

## Model limitations retained

The archived study remains base 325/384, SFT 338/384, and SFT+RL selection 339/384, with no accepted comparison gate. SFT reduced misleading-note failures from 28 to 6, while eligible-refund failures rose from 1 to 7 and partial-refund failures rose from 2 to 6. `archived-safety-diagnostics.json` binds every source report and counts rejected sandbox actions separately from failed cases. These exposed held-out cases are diagnostic evidence, not new training data.

This protects future checkpoint selection against a known source of sampling noise. It does not establish deterministic provider inference, explain the observed repeatability differences, fix model safety regressions, establish live crash durability, or certify production quality. A+ readiness is not claimed. Resolve the SDK cleanup stall and safety regressions, then validate under a newly frozen protocol on fresh independently reviewed data.

[Archived six-seed study](../river-six-seed-recovery/README.md). No new paid jobs were submitted.
