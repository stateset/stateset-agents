# River boundary-curriculum RL continuation

Status: completed. User-authorized, separate exploratory experiment; prior studies preserved.

Qwen/Qwen3.5-9B, rank-16 existing first-seed SFT checkpoint, four planned CISPO RL batches, 255 synthetic training cases with adjacent boundary contrasts, 32 fresh validation cases and 64 fresh paired test cases. Validation alone selects the checkpoint; confirmed skipped batches cannot promote unchanged weights. The fixed SFT baseline is sampled only after RL test evaluation.

Observed optimizer updates: 4; skipped batches: 0; selected validation step: 1.

Paired baseline/RL successes: 56/64 and 58/64. This single seed does not establish a reliable RL quality gain, independently reviewed data quality, or A+ certification.

The paired results have two wins and no case-level regressions. Model policy-violation incidence fell from 8/64 to 6/64; truncations fell from 8/64 to 5/64. Return-window boundary accuracy stayed at 3/8 for both checkpoints: the targeted weakness remains unresolved. The selected checkpoint contains one observed RL update; four updates occurred across the complete run.

Reserved generated-output tokens: 425,984, ceiling 458,752; these are admission reservations, not actual usage or verified billed dollars. The supervisor completed successfully in 687.5 seconds without a timeout or operational retry.

131 focused tests passed. All completed validation and test reports were replayed offline; fresh cases were checked against historical identities and policy facts. 471 previously retained artifacts verified unchanged.

`frozen-run.tar.gz` retains package source, input cases, manifests, selection and training activity, exact traces, token ledgers and supervisor logs. No River credential is retained. Hashes establish local file identity, not provider attestation.
The archive also includes `selected/river_checkpoint.json`, a reusable StateSet `--checkpoint` pointer directory.
