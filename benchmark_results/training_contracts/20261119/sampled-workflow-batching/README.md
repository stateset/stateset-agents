# Balanced batches on the synthetic refund workflow

Across eight held-out training seeds, alternating batches resolved **561/1,024**
sampled refunds and balanced batches resolved **836/1,024**. The paired gain
was **26.9 percentage points** (95% bootstrap interval **18.4 to 37.1**,
two-sided exact seed sign-flip **p = 0.0078125**). Every seed improved. The
comparison met the success rule published before the confirmation runs.
[Preregistration at the training commit](https://github.com/stateset/stateset-agents/blob/a5c4bfbf6ca6d1d9d013b25321d0284ccb0f7982/benchmark_results/training_contracts/20261119/sampled-workflow-batching/protocol.json)

| Held-out seed | Alternating | Balanced |
|---:|---:|---:|
| 101 | 66 | 85 |
| 102 | 69 | 86 |
| 103 | 87 | 109 |
| 104 | 81 | 111 |
| 105 | 85 | 113 |
| 106 | 45 | 112 |
| 107 | 80 | 113 |
| 108 | 48 | 107 |

Each cell counts successes in 128 stochastic evaluations of two policy states,
64 draws per eligibility class. Both arms started from the same syntax-warmed
policy for each seed, completed 20 optimizer updates and 480 training rollouts,
and replayed their final checkpoints. Across the eight seeds, the alternating
arm generated 9,598 training action tokens in 61.7 summed training seconds.
The balanced arm generated 9,969 in 63.7 seconds. Dollar cost was unmeasured.
The intervals describe these two states. Generalization to customer cases and
comparisons with other frameworks require separate trials.
[Paired analysis](comparison.json) · [Alternating reports](alternating/) · [Balanced reports](balanced/)

## Reproduce the comparison

Use training commit `a5c4bfbf` with Python 3.12.12, PyTorch 2.11.0 and
Transformers 5.14.1. Both original arms ran concurrently on a Core i7-1195G7
CPU, with one Torch thread per arm. Use new output directories for each run.

```bash
python -m stateset_agents.training.sampled_workflow --device cpu --updates 20 --seeds 101 102 103 104 105 106 107 108 --output /tmp/refund-alternating
python -m stateset_agents.training.sampled_workflow --device cpu --updates 20 --seeds 101 102 103 104 105 106 107 108 --balanced-batches --output /tmp/refund-balanced
```

For each seed and arm, sum final `candidate` successes in `result.json`
separately for even-numbered test IDs (ineligible) and odd-numbered IDs
(eligible). Pass the paired counts to `paired_seed_case_summary` with one
case per family, group size 64, analysis seed 20261119 and 10,000 bootstrap
replicates. Each arm's own `summary.json` compares with its syntax-warmed
starting policy. The between-arm comparison is in [comparison.json](comparison.json).
