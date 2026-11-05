# A stronger learned control narrows the handwritten-weight advantage

We reran the weighted-cost task with a graph-aware recurrent neural baseline
and intermediate message supervision. Two seeds reach over 99% exact whole-graph
accuracy on the random test families; a third reaches 36.91%. All three fail the
maximum-cost chains. The analytic network remains 100% exact on every family.

This qualifies the earlier comparison against a shallow parameter-matched MLP:
its roughly 14.5% random-graph result is not a ceiling on what training can learn.
Architecture and supervision explain much of the gap. The analytic construction
still provides exactness throughout its supported domain without fitting.

## Results

Each graph is correct only when all seventeen node costs are correct. Learned
outputs are rounded to the nearest integer for scoring; their unrounded errors
are retained. The handwritten network has zero raw error without rounding.

| Family | Cases | Handwritten | Learned seed 0 | Learned seed 1 | Learned seed 2 |
|---|---:|---:|---:|---:|---:|
| Four random densities, combined | 2,048 | 100% | 99.66% | 99.32% | 36.91% |
| Maximum-cost chains | 512 | 100% | 0% | 0% | 0% |
| Broken chains | 512 | 100% | 31.05% | 31.05% | 21.88% |
| Cheaper detours | 512 | 100% | 100% | 100% | 0% |
| Disconnected cycles | 512 | 100% | 55.86% | 54.49% | 14.45% |

These are three separately validation-selected runs, not confidence intervals.
We retain the weak third seed rather than selecting a seed using test accuracy.
All selected learning rates were 0.003. Raw maximum error on maximum-cost chains
is 38.63, 47.85, and 36.52 cost units respectively. Small random-family errors do
not establish reliable extrapolation to long, expensive routes.

## What was supplied to the learned model

The shared message MLP is `2 -> 32 -> 32 -> 1`, with **1,185 learned parameters**.
Its inputs are an incoming distance and an edge cost, normalized by 113. The
architecture supplies sixteen recurrent rounds, minimum aggregation over actual
edges, retention of the previous distance, and clamping to the valid range.
It learns the message arithmetic; it is not asked to discover graph propagation.

For training, exact Bellman–Ford intermediate states are generated on the
training graphs. Each update samples 64 graphs and one of sixteen intermediate
rounds per graph, and supervises every candidate message d[i]+cost[i,j]. This is
privileged algorithmic supervision: substantially more information than the
final-distance labels given to the original shallow classifier. Messages on
absent/diagonal edges are included in that loss, although inference masks them.

The train/validation/test adjacency matrices are **identical** to the original
weighted-cost benchmark: 16,384 training, 1,024 validation, and 4,096 test graphs.
Every split hash matches. Training and validation contain only four random
edge-density families. Structured graph families remain absent from training.
This is a follow-up on the existing public test set, not a newly sealed test set.

Three seeds each run two Adam learning rates (0.001, 0.003), 5,000 updates,
batch 64, cosine decay, and gradient clipping at 1. Validation every 500 updates
selects checkpoints by exact graphs, then lower raw mean absolute error. The
same rule selects the learning rate separately for each seed. All six training
trials finish before this job computes test labels or metrics.

Each trial draws 320,000 training graphs and receives 92,480,000 scalar message
targets. This matches the earlier update and graph-draw counts, **not** its
label count, architecture, parameter count, or compute. The six local CPU trials
took about 132 seconds of measured training/validation time in total; setup,
dataset preparation, final testing, and design work are additional costs.

## Verification and reproduction

All three exported selected checkpoints were reloaded and all per-family metrics
reproduced across 12,288 graph predictions. Intermediate teacher states end at
the independent heap-Dijkstra labels. Unit tests separately check a full
maximum-cost chain and a cheaper detour using exact message arithmetic.

```bash
PYTHONPATH=. python benchmarks/weight_space/learned_weighted_graph_job.py --out /tmp/learned-graph-replay
```

`protocol.json`, `splits.json`, `trials.json`, `summary.json`, and `verification.json`
retain the complete results. `source.py` and `dependencies/` preserve the exact
trainer, sampler, and encoder. All six `.safetensors` checkpoints and the saved
evaluation tensors are retained locally and ignored by Git. The source recipe
regenerates them. The [original handwritten-cost report](../20261104_handwritten_weighted_graph/README.md)
contains the analytic proof and the earlier shallow-MLP experiment. The separate
[route experiment](../20261104_handwritten_routes/README.md) returns full paths;
this learned baseline predicts costs only and is not a route-quality comparison.
