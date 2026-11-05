# Replacing a learned graph message with 69 weight assignments

Writing **69 coefficients (5.82%)** in each of three previously trained models
produces correct rounded costs on **all 6,283 cases per model**: 4,096 fresh
graphs and 2,187 exhaustive regression cases. More training on the original
distribution still fails every maximum-cost chain. Training with full distance
coverage also reaches 100% on every tested case. This supports precise operator
replacement and shows that changing message supervision can close the observed
training gap; it does not establish that writing weights generally outperforms
training.

This is a controlled repair of a known arithmetic operator. The existing solver
already supplies graph masking, minimum aggregation, distance clamping, and
sixteen recurrent rounds. Its learned part is a 1,185-parameter
`2 -> 32 -> 32 -> 1` ReLU MLP that approximates an incoming distance plus an edge
cost. The edit knows that arithmetic in advance. It does not infer an unknown
algorithm from a pretrained model.

## Edit and guarantees

The edit sets the first unit of the first hidden layer to `ReLU(d+c)`, makes the
first unit of the second hidden layer copy it, and makes the output read only
that second unit. Inputs are nonnegative. The assigned coefficients are:

| Parameter slice | Assigned entries |
|---|---:|
| First-layer row and bias | 3 |
| Second-layer row and bias | 33 |
| Output row and bias | 33 |
| Total | 69 of 1,185 (5.82%) |

`edit_ledger.json` records every before/after value. The architecture, input
encoding, and graph forward pass remain unchanged. Only four connections on the
written path determine the message. The remaining learned branches are
computed but disconnected from the output; unchanged coefficients therefore do
not demonstrate preservation of the learned function. The recipe rejects
non-finite weights and bounds unused-branch activations to avoid overflow.

In real arithmetic the message is exactly d+c throughout the nonnegative
input domain. For the measured float32 CPU execution, the final costs can have
small rounding errors. With round-to-nearest binary32 and u=2^-24, input cost
normalization contributes at most u absolute error. Each addition of two
values bounded by one contributes at most 2u. Masking, minimum, and clamping
are nonexpansive, so sixteen rounds accumulate at most 48u normalized error.
Rescaling by 113 contributes at most another 113u, giving a conservative
absolute cost-error bound of 113*49u < 0.000331, well below the 0.5 threshold
for correct integer rounding. Initial distances zero and one are exact. The
presence mask is exact because absent edges normalize to one, diagonal entries
to zero, and real costs 1–7 lie strictly between those values. The other layers
on the selected path multiply by one and add zero. This argument
assumes the supported float32 CPU arithmetic and finite unused activations;
it is not a claim about arbitrary reduced-precision backends.

Positive integer edge costs 1–7 and seventeen nodes imply every reachable
shortest path has cost at most 112; 113 denotes unreachable. Thus rounding
recovers the correct integer costs over this supported domain. This differs
from the earlier all-analytic cost network, which reports zero raw error.

## Comparisons and fresh graph results

The starting models are the three previously validation-selected checkpoints
from the [stronger learned baseline](../20261104_learned_weighted_graph/README.md).
All three seeds are retained. Each is compared with:

- The 69-coordinate analytic edit, with zero optimizer updates.
- One random control on the same coordinates, with the same total parameter
  delta norm and seed 65000 plus the model seed.
- Further Adam training on the original intermediate graph states.
- Further Adam training on distances drawn uniformly from [0,1] and each real
  edge cost 1–7, with the same addition targets. This deliberately covers the
  whole valid distance interval instead of sampling graph-derived states.

The coverage regime also supervises only real edge costs; the original regime
includes absent-edge and diagonal messages even though inference masks them.
It changes distance coverage, cost weighting, and their correlations together.
The comparison demonstrates a benefit from the changed message distribution;
it does not isolate long-distance coverage as the sole cause.

Both training regimes run two fixed learning rates (0.001 and 0.003), each for
5,000 updates with 18,496 scalar message targets per update: 92,480,000 targets
per trial. They start from identical saved checkpoints. Validation every 500
updates, including step zero, selects exact validation graphs first and lower
raw error second. The same original random-graph validation set selects both
checkpoints and learning rates. No seed is selected by test performance.
This is equal update and target-count budgeting between the two training
regimes; it does not equate their information content with hand-specified
arithmetic or count the author's design effort as free compute.

All twelve trials finish before fresh test graphs are generated, using seed
66000. The 4,096 new adjacency matrices exclude all 21,504 matrices from the
previous training, validation, and test splits, regardless of source vertex.
They cover the same eight known families, including 2,048 random graphs and
512 graphs per structured family. This is a follow-up on known failure
families, not a previously unseen task. Isomorphic graphs may occur across
splits; permuted chains do not represent independent new graph structures.

Each case is correct only when all seventeen rounded node costs match the
independent shortest-path oracle. Raw errors are also retained.

| Method | Seed | Random (2,048) | Max-cost chains (512) | Broken chains (512) | Cheap detours (512) | Two cycles (512) |
|---|---:|---:|---:|---:|---:|---:|
| Original | 0 | 99.27% | 0.00% | 30.08% | 100.00% | 57.23% |
| Original | 1 | 99.37% | 0.00% | 30.08% | 100.00% | 55.47% |
| Original | 2 | 35.11% | 0.00% | 18.16% | 0.00% | 18.16% |
| 69-coordinate edit | 0 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| 69-coordinate edit | 1 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| 69-coordinate edit | 2 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| Random control | 0 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Random control | 1 | 14.89% | 0.00% | 6.45% | 0.00% | 0.00% |
| Random control | 2 | 14.89% | 0.00% | 6.45% | 0.00% | 0.00% |
| More original-data training | 0 | 99.71% | 0.00% | 36.72% | 100.00% | 66.41% |
| More original-data training | 1 | 99.90% | 0.00% | 36.72% | 100.00% | 70.90% |
| More original-data training | 2 | 93.51% | 0.00% | 23.83% | 0.00% | 36.91% |
| Full-range training | 0 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| Full-range training | 1 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| Full-range training | 2 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

A separate known regression set exhausts 729 directed three-node graphs with
edge values 0,1,2, embedded in seventeen vertices, and all three source choices:
2,187 cases. It is not counted among the fresh graphs.

| Method | Seed 0 exact | Seed 1 exact | Seed 2 exact |
|---|---:|---:|---:|
| Original | 2187/2,187 | 2187/2,187 | 1539/2,187 |
| 69-coordinate edit | 2187/2,187 | 2187/2,187 | 2187/2,187 |
| Random control | 27/2,187 | 243/2,187 | 243/2,187 |
| More original-data training | 2187/2,187 | 2187/2,187 | 2187/2,187 |
| Full-range training | 2187/2,187 | 2187/2,187 | 2187/2,187 |

The edited models have a maximum raw cost error of **2.28881836e-05**,
with exact integer outputs after rounding. Full-range training has maximum
raw errors of 0.089424, 0.109276, and 0.007479
for seeds 0, 1, and 2. All remain below the half-unit scoring threshold on these
cases. The twelve training/validation trials took 449.4 measured CPU
seconds in total; preparation, test evaluation, verification, and design are
additional work. No cloud allocation was used.

## Continuous component analysis

After model selection and testing, `certify_graph_repair.py` analyzes each
selected message MLP using exact rational arithmetic on its stored float32
coefficients. For each of the seven float32-encoded costs, it enumerates all
first-layer breakpoints and all second-layer breakpoints within those
intervals. The resulting message and its addition error are affine between
breakpoints, so the maximum absolute error occurs at an endpoint. This covers
all real distances in [0,1], not just the sampled component grid.

| Method | Seed 0 maximum message error | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Original | 0.0368083559 | 0.0574263446 | 0.0365758085 |
| 69-coordinate edit | 0 | 0 | 0 |
| Random control | 0.233666834 | 0.371960311 | 0.959339227 |
| More original-data training | 0.0291370641 | 0.0434157399 | 0.0141708721 |
| Full-range training | 0.000813662004 | 0.000645820276 | 0.000314304233 |

The edited networks have exactly zero real-arithmetic component error, expressed
as the rational number `0`. Learned full-range models have nonzero component
error despite scoring perfectly after rounding on the graph tests.
`certificates.json` preserves exact rational bounds, interval counts, and
witness inputs, including every model and seed.

These certificates describe real-arithmetic evaluation of the component with
stored coefficients and encoded costs. They do not bound the rounding behavior
of arbitrary learned networks. The explicit float32 argument above applies to
the simple analytic path. No certificate was used to choose a checkpoint or
learning rate. Tests cover exact addition, known first- and second-layer kinks,
and agreement of witnesses with independent double-precision forward passes.

## Verification and reproduction

NetworkX independently reproduced all 6,283 shortest-path label vectors. The
verifier reloaded all fifteen selected model variants and reproduced **94,245
graph predictions**, all test metrics, and selected validation metrics. It also
checked the source and checkpoint hashes, all 69 edited coordinates per seed,
random-control coordinates and delta norms, and zero overlap with prior graph
matrices. All 18 experiment tests and 10 documentation checks passed. Source
formatting, lint, type-hook, and repository-hygiene checks passed. The full repository suite was not rerun.

The original protocol, sources, input-checkpoint hashes, every trial history,
selected weights, scalar edit ledger, input tensors, oracle labels, component
certificates, and replay report are retained. Binary tensor files are local and
ignored by Git; the source recipe regenerates them. `artifact_manifest.json`
binds both tracked and local artifacts. The original benchmark reports and
checkpoints remain unchanged.

```bash
PYTHONPATH=. python benchmarks/weight_space/repair_weighted_graph_job.py \
  --initial benchmark_results/weight_space/20261104_learned_weighted_graph \
  --out /tmp/graph-repair-replay
PYTHONPATH=. python benchmarks/weight_space/verify_graph_repair.py \
  /tmp/graph-repair-replay \
  --initial benchmark_results/weight_space/20261104_learned_weighted_graph
PYTHONPATH=. python benchmarks/weight_space/certify_graph_repair.py /tmp/graph-repair-replay
```

If the starting binary checkpoints are absent, first regenerate the earlier
learned baseline with `learned_weighted_graph_job.py`. The repair job verifies
starting hashes and reproduces the original split hashes before proceeding.
