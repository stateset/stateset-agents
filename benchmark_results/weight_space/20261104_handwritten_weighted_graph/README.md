# Two million handwritten parameters solve weighted shortest paths

This network takes a seventeen-node directed graph with integer edge costs 1–7
and a starting node. It returns the minimum cost to every node, including the
source at cost zero; output **113 means unreachable**. It handles competing
routes, ties, cycles, and the case where a cheaper path uses more edges. It
outputs costs, not an explicit sequence of route vertices.

Every weight is constructed analytically. There are **zero training updates**,
no pretrained checkpoint, and no graph lookup table. Forward inference uses only
standard PyTorch `Sequential`, `Linear`, and `ReLU` modules. Input encoding merely
packs the edge costs and initializes source distances; it does not solve paths.

## Parameter count, including the exact-budget adjustment

The network stores **exactly 2,000,000 unique dense parameters**, of which
**5,679 are nonzero**. A shared block is applied sixteen times; its repeated uses
are counted once. Most coefficients are zero because the compiled algorithm is
a sparse circuit represented as ordinary dense neural layers.

The natural circuit has **1,865,053 parameters**. To meet the requested exact
budget, 137 active hidden neurons are split into identical pairs, halving each
pair's outgoing coefficients. Both copies contribute to the output. This adds
134,947 stored coefficients without changing the function; it is representation
redundancy, not additional algorithmic capability. No unused padding tensors
are added. This does not claim two million independently invented nonzero rules.
The constructor option `exact_budget=False` returns the natural circuit.

## Results

A graph counts as correct only if **all seventeen output costs are correct**.
The trained ranges cover three initialization seeds, each selected using only
validation data; these are min–max ranges, not confidence intervals.

| Evaluation family | Cases | Handwritten | Trained MLP |
|---|---:|---:|---:|
| Random graphs | 2,048 | 100% | 14.26–14.55% |
| Maximum-cost chains | 512 | 100% | 0.00–0.00% |
| Broken chains | 512 | 100% | 7.42–7.42% |
| Cheaper detours | 512 | 100% | 0.00–0.00% |
| Disconnected cycles | 512 | 100% | 0.00–0.00% |

The handwritten network also has **zero raw floating-point error** before any
rounding on every family. The exhaustive check covers **2,187 cases**: all 729
directed three-node graphs with each off-diagonal cost in {0,1,2}, and all three
sources, embedded at vertices 0, 8, and 16 of the seventeen-node model. Remaining
vertices are isolated. This correctness check is separate from the held-out split.

## Why the weights are correct

Let M=(n−1)W+1, with n=17 and W=7, giving M=113. Any reachable node has a simple
shortest path costing at most (n−1)W=112. Encode absent edges as M, diagonal edges
as zero, and initialize d[source]=0 and d[other]=M. One shared block computes

```
d_next[j] = min(d[j], min_i(d[i] + cost[i,j]))
```

The old d[j] candidate automatically caps unreachable costs at M. After k rounds,
d is the minimum cost of a path with at most k edges, capped at M. Induction on k
proves this invariant. Positive costs mean every reachable shortest path has a
simple representative using at most n−1 edges. Sixteen rounds therefore return
all exact shortest costs, with M only for unreachable nodes.

Each round first expands the 306-element state (289 costs plus 17 distances)
into 595 channels: the unchanged costs and eighteen candidates for each node.
Five stages reduce the candidates by pairwise minima. For nonnegative a and b,

```
min(a,b) = a - ReLU(a-b)
```

A first Linear/ReLU layer retains a and max(a−b,0); a second Linear/ReLU layer
subtracts them. Nonnegative minima survive its ReLU unchanged. Costs are copied
through every stage. The natural stage widths are 595→442→374→340→323→306.
Each stage's first layer preserves its input width; its second layer reduces it.
An output Linear layer selects the seventeen distances.

The first three comparison layers split respectively 106, 27, and 4 active
neurons, giving hidden widths 701, 469, and 378. Their parameter increments are
`106*1038 + 27*817 + 4*715 = 134,947`. The split pairs use coefficients ±0.5;
all other nonzero coefficients are ±1. All inputs/intermediates remain small
integers or exactly representable half-integers in float32. The proof covers
**every valid seventeen-node cost matrix and source**, beyond the measured cases.

The code also constructs other n or cost bounds, subject to its explicit float32
exactness bound. These require a newly constructed configuration; the saved
seventeen-node checkpoint is not a variable-size graph model.

## Comparison protocol and limitations

- Train: 16,384 distinct weighted adjacency matrices, equally divided among four
  Bernoulli edge-density families (.05, .12, .25, .5). Every present edge has an
  independently sampled integer cost 1–7, with no self-loops.
- Validation: 1,024 graphs from those same families. Test: 512 per random family
  plus 512 each of maximum-cost chains, broken chains, cheaper-detour graphs,
  and disconnected pairs of cycles: 4,096 total.
- Cost matrices are unique across train, validation and test, independently of
  the source. Duplicate rejection conditions the sampling on uniqueness.
  All structured graph node labels are randomly permuted. Structured families
  are distribution shifts and never used for training or model selection.
- Maximum-cost chains require costs up to 112. Broken chains test unreachable
  nodes. Cheaper-detour graphs have unit-cost chain edges plus direct edges
  costing 7, including equal-cost alternatives. Disconnected cycles have random
  positive costs. The source begins at the first node of the permuted structure.
- Baseline: a conventional `306 -> 890 -> 1938` ReLU MLP, **1,999,988 parameters**,
  reshaped into 114 cost classes per node. It receives the same input information,
  divided by 113, and is trained using cross entropy. This allows a conventional
  shallow classifier to learn without backpropagating through sixteen long
  recurrent rounds; it is not the same architecture as the compiled network.
- Adam, 5,000 updates, batch 64: 320,000 sampled examples per trial. Three seeds
  and two learning rates (.001, .003), cosine decay to one tenth of the initial
  rate, gradient norm clipping at 1. Validation every 500 updates selects the
  checkpoint/rate per seed, prioritizing exact graphs then lower cost MAE.
- All six trial histories are retained. Test labels/metrics are computed only
  after every training trial finishes. Budgets/families were fixed before the
  full run. A separate 20-step smoke run checked execution on smaller splits.
- Label oracle: independent heap-based Dijkstra. NetworkX independently verifies
  all 4,096 saved test labels. Reloading all four checkpoints reproduces reported
  metrics. The natural and split circuits agree on every saved test graph.

This is a known algorithm compiled into weights, with a finite-budget training
comparison. It does not show that training cannot learn weighted shortest paths,
that two million parameters are necessary, or that this dense NN is faster than
ordinary Dijkstra/Bellman–Ford. Graph-specific neural architectures, intermediate
supervision, and larger budgets are untested comparisons. Human/LLM design work
is not counted as optimizer training or benchmark runtime.

## Release validation

Candidate v0.71.3 passes the 42 final release/model guard tests, all 18 dedicated
weighted-network tests, and type checking across 551 package files. The initial
full run recorded 16,808 passes, 133 skips, and 13 failures. Twelve failing cases
pass after selecting the compatible patch version, updating the example index
and typing count, and using an absolute checkout path for subprocess tests.
One pre-existing online-DPO learning test still fails with the same values on
the unchanged v0.71.2 worktree; neither its test nor training code is changed.
The full suite is therefore **not all green**. See `release_checks.json` for the
exact counts, failed test, and baseline-control result. Default suite markers
exclude performance, slow, and GPU tests.

## Reproduce

From the repository root, with torch and safetensors installed:

```bash
PYTHONPATH=. python examples/handwritten_weighted_shortest_paths.py
PYTHONPATH=. python examples/handwritten_weighted_shortest_paths.py --weights benchmark_results/weight_space/20261104_handwritten_weighted_graph/handwritten.safetensors
PYTHONPATH=. python benchmarks/weight_space/handwritten_weighted_graph_job.py --out /tmp/weighted-2m-reproduction
```

The default benchmark arguments reproduce this configuration. The output path
must be new. Recorded environment: Python 3.12.12, torch 2.11.0+cu130, CPU with two
threads and deterministic algorithms. Bitwise training reproducibility across
hardware/library versions is not guaranteed.

`summary.json` contains configuration, data hashes, trial histories, and metrics.
`sources/` preserves the exact model and benchmark used for the run. After the
run began, public input validation was tightened to reject fractional float64
costs before they could round to an integer in float32. The benchmark uses integer
costs, so its data and tensors are unchanged. `model_delivered.py` preserves that
final source separately. The audit verifies identical constructed weights and
identical encodings for every saved graph, and records both source hashes in
`verification.json`. Checkpoint
and evaluation binaries are retained locally under the repository's existing
ignored-artifact policy; running the command regenerates them. They are not part
of the Git tag. `artifact_manifest.json` binds all evidence files by SHA-256.
Use `safetensors.torch.load_model` to restore shared checkpoint tensors.
