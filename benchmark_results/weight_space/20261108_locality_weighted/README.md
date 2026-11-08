# Locality-weighted editing on the same dense trained models

Adding a locality penalty reduces mean final confirmation-set squared output
drift by **79.5%**, from `0.170958` to `0.0349785`, relative to the original
minimum-weight-norm constrained editor. Improvement occurs in all five seeds.
Current and earlier targets, plus the protected references, remain within
`4.7684e-7` of their required values.

The independent verifier checks all 30 trials, including full-covariance KKT
solutions for the weighted objective, preservation spans, fresh split seeds,
and byte equality of both control arms against the earlier experiment.

## Results and limits

| Arm | Worst current/earlier-target error | Worst reference drift | Worst confirmation drift | Final mean confirmation drift MSE | Final mean original-task MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unedited base | — | 0 | 0 | 0 | 0.00352787 |
| Minimum weight norm, with protection | 3.5763e-7 | 3.5763e-7 | 3.68847 | 0.170958 | 0.172437 |
| Locality-weighted, with protection | 3.8743e-7 | 4.7684e-7 | 1.36985 | 0.0349785 | 0.0374834 |
| Unconstrained minimum norm | 0.331949 | 0.385105 | 0.446206 | 0.0201656 | 0.0234179 |

Drift MSE averages squared changes from base predictions across inputs and output
coordinates. The primary comparison averages the final-round value across the
five models. Worst-case columns include both sequential rounds. Original-task
MSE measures error against the original smooth regression functions.

| Seed | Minimum-norm final drift MSE | Locality-weighted final drift MSE |
| --- | ---: | ---: |
| 0 | 0.108982 | 0.0175819 |
| 1 | 0.213876 | 0.0306238 |
| 2 | 0.140541 | 0.0263684 |
| 3 | 0.265976 | 0.0413714 |
| 4 | 0.125415 | 0.0589470 |

This improves locality under the same hard constraints, but does not eliminate
damage. Original-task error remains about ten times the unedited baseline. The
unconstrained arm changes fewer held-out outputs but damages protected references
and earlier edits. No language-model efficacy or universal preservation claim
follows from these small dense-readout experiments.

## Frozen design

The five trained `4 → 32 → 3` ReLU networks, eight protected reference inputs,
eight edit inputs, two four-edit rounds, and target shifts are unchanged from
the [earlier experiment](../20261108_constrained_dense/README.md). Its full
auditable bundle is copied under `base/`, with hashes bound to this protocol.
There is no new model training or optimizer-based edit fitting.

The weighted arm minimizes
`||D||_F² + 100 * ||H D.T||_F² / 256`, subject to the same target and preservation
equalities. Its 256 locality inputs use seed 92181. The 4,096 confirmation inputs
use seed 92183. Both sets are disjoint from each other and from every earlier
training, reference, edit and held-out input. The confirmation set is never
passed to the solver. The strength 100 is fixed before the run; no checkpoint
or hyperparameter selection uses these results.

All arms start from their own previous-round weights. The constrained arms
protect earlier edit keys in round two. Target and per-update span tolerances
remain `1e-5`; replay allows `2e-5` for accumulated target/reference error.
The unweighted and unconstrained controls reproduce the previous stored weights
exactly, so the comparison does not replace the baseline with a different run.

## Independent replay

The verifier imports neither the editor nor experiment jobs. It checks the
original dependency bundle, hashes, condition coverage, frozen input seeds,
split disjointness, model forwards, sequential updates, reference spans, reported
metrics and summaries. For optimality it solves a full-covariance KKT system,
independently of the editor's projected SVD implementation. Tests also substitute
a feasible unweighted edit and recompute its metrics; the optimality check rejects
it even though its targets and protection still pass.

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m \
  benchmarks.weight_space.constrained_edit_job --out /tmp/locality-base
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m \
  benchmarks.weight_space.locality_edit_job \
  --base /tmp/locality-base --out /tmp/locality-reproduction
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m \
  benchmarks.weight_space.verify_locality_edit /tmp/locality-reproduction
```

JSON evidence and frozen source copies are committed. Safetensors inputs and
states are retained locally as ignored files; the commands regenerate a complete
bundle. `verification.json` records the audit of the original run. The results
were measured in float32 on CPU and do not establish equivalence on other
devices, precision modes or inference kernels.
