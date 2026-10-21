# River six-seed synthetic replication — 2026-10-19

This frozen study compares base, expert SFT, and SFT followed by two native RL batches on six independent training seeds. Every seed uses 128 synthetic training cases, 64 validation cases, and 64 fresh held-out cases. The ordinary data generator, policy, rewards, and held-out 7/45-day windows are unchanged. Training coverage doubles the earlier warmup, including 16 boundary and 16 misleading-note examples per seed.

| Seed | Base test /64 | SFT test /64 | SFT+RL test /64 | RL selected batch |
| --- | --- | --- | --- | --- |
| 2026101901 | 52 | 55 | 54 | 2 |
| 2026101902 | 53 | 52 | 53 | 2 |
| 2026101903 | 54 | 55 | 54 | 1 |
| 2026101904 | 57 | 61 | 61 | 0 |
| 2026101905 | 53 | 55 | 56 | 0 |
| 2026101906 | 56 | 60 | 61 | 2 |

Pooled descriptive counts: base 325/384; SFT 338/384; SFT+RL 339/384. Statistical comparisons use independent seeds as the units; pooled case counts do not create additional training seeds.

All six seeds complete across every arm: **True**. All prespecified comparison gates passed: **False**.

- sft_vs_base: passed=False; mean gain=0.033854166666666664
- sft_rl_vs_sft: passed=False; mean gain=0.0026041666666666665
- sft_rl_vs_base: passed=False; mean gain=0.036458333333333336

The comparison rules require six seeds, at least 3 percentage points mean success gain, improvement in every seed, no violation/truncation regression per seed, no success or safety regression within any seed's scenario families, and an exact seed-sign test with Bonferroni correction across three planned comparisons. Thresholds were fixed before provider calls. The JSON review includes per-seed family gates and pooled family regressions; successful transport does not certify model quality.

Validation score changes after a reported skipped update at the same optimizer step: 1. Their cause is unconfirmed; they are retained as a repeatability limitation and are not evidence of learning.

The SFT+RL procedure may select its initial SFT checkpoint rather than updated weights. Per-seed records distinguish the selected validation batch from the optimizer step and state whether an actual optimizer update was observed before the selected checkpoint. Missing or failed RL arms remain explicit in the summary.

For selections with no observed RL update, the review also compares fresh test outcomes against the separately sampled SFT arm. Any differences in these outcomes are repeatability limitations with an unconfirmed cause, not evidence of RL learning. The raw model scores remain unchanged.

Reported SFT updates across six model sets: 96. Completed RL batches: 12; observed RL optimizer updates: 10; skipped batches: 2. These counts include the two inherited model sets once. Selected RL checkpoints with at least one observed prior update: 4.

The original attempt and its partial results are preserved in ../river-six-seed/. The recovery amendment changes operational deadlines and removes redundant fixed-arm validation rescoring for four seeds; model hyperparameters, package code, validation selection, scoring, and acceptance thresholds remain unchanged. New primary holdouts were frozen before recovery sampling. The 768 rule-based expert chats were replayed in fresh sandbox instances and checked with the real tokenizer. The training data excludes all 224 historical canary cases. Fresh recovery primary holdouts additionally exclude every original reserved study case by ID and factual fingerprint. The original dataset bundle is retained intact; primary-test.json, rather than data/test.json, defines the fresh paired test cases. All study splits and seeds were mutually disjoint. Base and SFT checkpoints are fixed by the plan; the native RL arm chooses only on validation, with earliest-step ties. Each model's held-out test attempt is sealed before sampling, including uncertain attempts. Original timeouts and stops remain archived in the first-attempt evidence. Two completed model sets are reused without training; only four previously untrained seeds perform new SFT/RL. This is one operational recovery with fresh primary holdouts, and no remaining failure is retried.

The operational recovery allows at most two concurrent seeds, 30 minutes per seed, and a 90-minute recovery supervisor deadline. Across the original attempt and recovery, the limits remain 96 requested SFT updates, 12 requested RL batches, and the original 3,690,000-token rollout reservation ceiling; the conservative combined allocated bound is 3,478,720 tokens. These limits do not establish a dollar cap or verified billed cost. SFT automatic restarts are disabled; RL recovery remains limited by its durable admission ledger and execution deadline.

The final durable admission ledgers reserve 3,389,440 generated-output tokens across both attempts. Reservations are upper bounds for admitted trajectories, not actual token usage or billed cost. The audit verifies frozen runner hashes, dataset artifact hashes, all 384 primary cases, disjointness from prior cases, and every ledger ceiling.

[Review summary](review-summary.json), [frozen plan](plan.json), and per-seed reports contain the evidence. The frozen-runtime.tar.gz archive includes the exact package files and runners needed for source-matched replay. JSONL exports and cost ledgers have an added .txt suffix; their bytes are unchanged. Dataset manifests retain original names and hashes. No credential is retained.

This remains a synthetic experiment rather than independent human-reviewed production-quality evidence. It does not establish live crash recovery, backend parity, provider inventory cleanup, or billing correctness. A+ readiness is not claimed. Preserve all exposed held-out cases.
