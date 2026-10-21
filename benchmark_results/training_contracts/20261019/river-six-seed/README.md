# River six-seed synthetic replication — 2026-10-19

This frozen study compares base, expert SFT, and SFT followed by two native RL batches on six independent training seeds. Every seed uses 128 synthetic training cases, 64 validation cases, and 64 fresh held-out cases. The ordinary data generator, policy, rewards, and held-out 7/45-day windows are unchanged. Training coverage doubles the earlier warmup, including 16 boundary and 16 misleading-note examples per seed.

| Seed | Base test /64 | SFT test /64 | SFT+RL test /64 | RL selected batch |
| --- | --- | --- | --- | --- |
| 2026101901 | 53 | failed | 58 | 2 |
| 2026101902 | failed | failed | 54 | 2 |
| 2026101903 | failed | failed | failed | failed |
| 2026101904 | failed | failed | failed | failed |
| 2026101905 | failed | failed | failed | failed |
| 2026101906 | failed | failed | failed | failed |

Pooled descriptive counts: base 53/64; SFT 0/0; SFT+RL 112/128. Statistical comparisons use independent seeds as the units; pooled case counts do not create additional training seeds.

All six seeds complete across every arm: **False**. All prespecified comparison gates passed: **False**.

- sft_vs_base: passed=False; mean gain=unavailable
- sft_rl_vs_sft: passed=False; mean gain=unavailable
- sft_rl_vs_base: passed=False; mean gain=0.078125

The comparison rules require six seeds, at least 3 percentage points mean success gain, improvement in every seed, no violation/truncation regression per seed, and an exact seed-sign test with Bonferroni correction across three planned comparisons. Thresholds were fixed before provider calls. The JSON review includes per-family results and family regressions separately; successful transport does not certify model quality.

Validation score changes after a reported skipped update at the same optimizer step: 1. Their cause is unconfirmed; they are retained as a repeatability limitation and are not evidence of learning.

The 768 rule-based expert chats were replayed in fresh sandbox instances and checked with the real tokenizer. All 224 historical canary cases were excluded by ID and factual fingerprint. All study splits and seeds were mutually disjoint. Base and SFT checkpoints are fixed by the plan; the native RL arm chooses only on validation, with earliest-step ties. Each model's held-out test attempt is sealed before sampling, including uncertain attempts. Any failed or zero-update RL arm remains failed; no seed is resubmitted.

The bound is two concurrent seeds, 20 minutes per seed, 60 minutes for the study, 96 requested SFT updates, 12 requested RL batches, and at most 3,690,000 reserved rollout output tokens across the separate durable ledgers. These limits do not establish a dollar cap or verified billed cost. SFT automatic restarts are disabled; RL recovery remains limited by its durable admission ledger and execution deadline.

[Review summary](review-summary.json), [frozen plan](plan.json), and per-seed reports contain the evidence. The frozen-runtime.tar.gz archive includes the exact package files and runners needed for source-matched replay. JSONL exports and cost ledgers have an added .txt suffix; their bytes are unchanged. Dataset manifests retain original names and hashes. No credential is retained.

This remains a synthetic experiment rather than independent human-reviewed production-quality evidence. It does not establish live crash recovery, backend parity, provider inventory cleanup, or billing correctness. A+ readiness is not claimed. Preserve all exposed held-out cases.
