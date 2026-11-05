# Six-parent boundary SFT continuation replication

Qwen/Qwen3.5-9B, rank 16. All six independently trained retained original SFT parents were fixed before the new study; one was used in the earlier pilot. This is conditional continuation replication with new shuffle seeds and fresh holdouts, not six wholly new base initializations. Each parent has unchanged, original-repetition SFT and boundary-curriculum SFT arms. The two trained arms have identical rows, 16 updates, seed and hyperparameters. Input and assistant-target token exposure are within 0.5% per pair; exact provider compute or dollar costs are not matched.

| Arm | Test successes | Boundary successes | Policy-violation cases | Truncated cases |
|---|---:|---:|---:|---:|
| original | 689/768 | 48/96 | 79 | 71 |
| repeat | 719/768 | 65/96 | 49 | 30 |
| boundary | 747/768 | 76/96 | 21 | 19 |

Status: completed. Prespecified primary boundary-versus-repetition acceptance: False. The gates require six pairs, at least three percentage points mean gain, positive gain in every seed, no family/policy/truncation regression, and a seed sign test at alpha 0.05 with three-comparison Bonferroni correction. Secondary original-versus-continued-SFT comparisons use the same gates and correction. Validation is descriptive; final one-epoch checkpoints were fixed without test-driven selection.

Primary comparison: 4 positive and 2 negative seed pairs; corrected p=1.0. The boundary arm has a higher pooled success rate, but the prespecified matched-control gate fails on consistency, family/policy/truncation regressions and significance. The secondary boundary-versus-original gate passes: True; corrected p=0.046875.

| Continuation seed | Original /128 | Repetition /128 | Boundary /128 |
|---|---:|---:|---:|
| 2026101931 | 110 | 127 | 124 |
| 2026101932 | 113 | 112 | 125 |
| 2026101933 | 116 | 123 | 128 |
| 2026101934 | 117 | 125 | 119 |
| 2026101935 | 121 | 122 | 125 |
| 2026101936 | 112 | 110 | 126 |

Supervised return-window values: [14, 30]; held-out test values: [7, 45].

Replayed 3030 training transcripts and 2880 completed evaluation trajectories. 960 new validation/test cases are disjoint from historical and current training identities and policy facts, and other new holdouts. Acknowledged added SFT steps: 192; reserved output tokens: 2,949,120/2,949,120. These are admission reservations, not actual generated tokens or billed dollars. No RL or deployment was run. No automatic retry or deadline extension.

Four new control-data tests passed. Packaged sources match the prior validated implementation (370 unit tests, 19 scripted SDK tests, 445 type-checked sources). 493 older evidence artifacts verified unchanged. This synthetic policy study is not independently reviewed industry data or A+ production certification. Pooled counts are descriptive; six parent/run pairs are the inference units.

`frozen-run.tar.gz` retains exact package sources, training data, heldouts, per-arm attempts and traces, pointer lineage, token ledgers, supervisor logs, and historical input snapshots. No River credential is retained. Hashes establish local identity, not provider attestation.
