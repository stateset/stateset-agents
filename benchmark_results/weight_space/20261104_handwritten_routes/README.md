# Handwritten neural weights produce complete optimal routes

The network accepts a seventeen-node directed graph with integer edge costs
1–7, a source, and a target. Zero input entries mean absent edges. It outputs
the minimum cost and the full route as seventeen one-hot vertex rows followed
by zero padding. Unreachable pairs have cost 113 and an empty route; equal
endpoints have cost zero and a one-vertex route. Ties choose the lexicographically
smallest vertex sequence under the input vertex labels.

All weights are analytic. There are no training examples, optimization updates,
or pretrained parameters. Inference consists entirely of standard PyTorch
`Sequential`, `Linear`, and `ReLU` modules. The encoder transposes the adjacency
matrix and packs endpoint flags; the display code reads one-hot output rows.
Neither computes distances, selects edges, or searches for a route.

## Construction and correctness

1. Run the existing exact Bellman–Ford neural block sixteen times on the reversed
   graph, starting at the target. The resulting d[i] is the cost from i to the
   target, with M=113 representing unreachable vertices. Carry the source flags
   unchanged alongside the distance state.
2. A real edge i→j is tight exactly when c[i,j]+d[j]−d[i]=0. For finite integer
   costs, ReLU(1−gap−ReLU(c−7)) is its indicator, with diagonal indicators forced
   to zero. The missing-edge penalty excludes a spurious equality between an
   unreachable vertex and the target. The distance invariant makes every gap
   nonnegative. On present edges the penalty is zero.
3. For each row of tight-edge bits b, ReLU(b[j]−sum(b[:j])) selects the first
   tight neighbor. Every reachable non-target vertex has at least one. Positive
   edge costs imply d[j]<d[i], so following selected edges cannot cycle and reaches
   the target in at most sixteen hops. Choosing the smallest feasible next vertex
   at each step produces the lexicographically smallest optimal route.
4. ReLU(policy[i,j]+current[i]−1) activates exactly the selected transition out
   of the current vertex. Sum over i to obtain the next one-hot vertex. A shared
   Linear/ReLU block performs this transition and shifts route history sixteen
   times. The target has no tight outgoing edge, producing zero padding.
5. The reachability bit ReLU(M−d)−ReLU(M−1−d) suppresses the initial current
   vertex when the source is unreachable. ReLU(d+M*source_flag−M) selects its
   distance. These computations are also in neural layers.

The proof applies to every valid graph in the configured finite domain, not
only the samples below. Other node counts and integer cost bounds can be
constructed subject to the existing float32 exactness bound. Changing the node
count requires a new network; the exported checkpoint has fixed input dimensions.

## Parameter accounting

The route extension has **3,920,670 unique stored parameters**, including
**13,792 nonzero coefficients**. Repeated calls to a shared block count once.
It carries source flags through the natural distance circuit, then adds neural
edge selection and traversal. No arbitrary parameter budget or neuron splitting
is imposed here. The earlier exact 2,000,000-parameter cost-only model is unchanged.

## Measured results

| Evaluation | Cases | Exact cost and entire route | Maximum raw error |
|---|---:|---:|---:|
| Four random-graph densities | 2,048 | 100% | 0 |
| Maximum-cost chains | 512 | 100% | 0 |
| Broken chains | 512 | 100% | 0 |
| Cheaper detours | 512 | 100% | 0 |
| Disconnected cycles | 512 | 100% | 0 |
| Exhaustive embedded three-node graphs/endpoints | 6,561 | 100% | 0 |

The exhaustive set contains all 3^6 directed cost matrices with edge values
0,1,2, embedded at vertices 0,8,16, and all nine source/target pairs. Other
vertices are isolated. This is exhaustive over that subset, not all seventeen-node
graphs. Expected routes come from an independent heap-based Dijkstra ordered
by cost and complete path. A separate verification job reloads the exported
checkpoint and compares every one of the 10,657 cases with NetworkX's enumeration
of shortest paths. It also verifies artifact hashes and exact equality between
the exported tensors and the delivered analytic constructor.

Reloading the exported checkpoint reproduces 512 cases (the first 64 of each
family). Float16 and bfloat16 both achieve exact results on those same 512 cases.
Those precision probes are empirical checks, not guarantees for all configurations.
The full float32 evaluation contains **10,657** cases.

This is a correctness and composition result. The dense neural realization was
slower than the Python oracle in this CPU run. Timings in `summary.json` are
single-run observations, include output packing on the oracle side, and are
not a controlled performance benchmark. No trained route baseline was run;
the earlier cost-only MLP scores must not be presented as route comparisons.

## Reproduction and retained artifacts

```bash
PYTHONPATH=. python examples/handwritten_weighted_routes.py
PYTHONPATH=. python benchmarks/weight_space/handwritten_routes_job.py --out /tmp/neural-route-replay
PYTHONPATH=. python benchmarks/weight_space/verify_routes_artifacts.py /tmp/neural-route-replay
```

`protocol.json` records the seed, case counts, tie rule, and source hashes before
evaluation. `sources/` contains the exact recipe, benchmark, and graph sampler.
`summary.json` retains all counts and precision checks. `artifact_manifest.json`
hashes the generated artifacts. `routes.safetensors` and `evaluation.safetensors`
are retained locally and ignored by Git; the checked-in recipe regenerates them.
The pilot GPU experiment is separate and does not affect these weights or cases.
