# Dense constrained edits: local success and held-out damage

All ten constrained edits of trained dense readouts preserve their reference
activation spans and satisfy their new targets. Earlier target edits survive
the second round. The largest target/reference discrepancy is `3.5763e-7`.
Independent replay checks the saved weights, all reported metrics, and the
minimum-norm solution using an augmented least-squares system.

**This does not establish good general locality.** Unprotected held-out outputs
change substantially, and original-task error increases. These negative results
are retained alongside the successful constraints.

| Arm | Worst current-target error | Worst current/earlier-target error | Worst reference drift | Worst held-out drift | Final mean original-task MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | — | — | 0 | 0 | 0.00354547 |
| Constrained | 2.3842e-7 | 3.5763e-7 | 3.5763e-7 | 3.38740 | 0.165115 |
| Unconstrained minimum norm | 2.3842e-7 | 0.331949 | 0.385105 | 0.415933 | 0.0230777 |
| Norm-matched random | 3.61317 | 3.61317 | 4.28595 | 4.83810 | 1.79162 |

## Fixed design

Five ordinary `Linear(4,32) → ReLU → Linear(32,3)` models train for 500 full-batch
Adam steps at 0.01 on the same 512 uniform inputs in `[-2,2]^4`. All seeds
`0..4` are retained, with no checkpoint selection or pruning. Every hidden and
readout weight is structurally nonzero. The original outputs approximate
`sin(x0)+x1/2`, `cos(x2)-x3/4`, and `x0*x3/2-x2`.

Eight fresh reference inputs and eight edit inputs are shared across models.
Each edit round shifts four input/output mappings from the base prediction by
the fixed vectors `(0.25,-0.125,0.0625)` and `(-0.125,0.25,-0.0625)` respectively.
Only readout weights change; the hidden map and output bias stay fixed. No
optimizer steps occur during editing.

The constrained arm protects the eight references and, in round two, the first
four edit keys. The unconstrained arm solves for each round's new targets
without preservation constraints. A fixed-seed random rank-at-most-four update
matches each constrained update's Frobenius norm. Every arm runs both rounds
from its own previous weights. The 1,024 held-out inputs are disjoint from all
training, reference and edit inputs and are never used to choose an update.
The protocol and source copies were written before training.

Target tolerance is `atol=rtol=1e-5`; protected span tolerance is `1e-5` per
update. Replay allows `2e-5` for accumulated reference/target error across two
rounds. All five controls with the entire 32-dimensional input space protected
are rejected. Rejections are consistent with the incompatible equality
constraints; unit tests separately exercise the rejection behavior.

## Audit and reproduction

The independent verifier imports neither the editor nor the training job. It
reconstructs forward calculations from stored tensors, checks complete trial
coverage, source/artifact hashes, disjoint inputs, sequential update arithmetic,
training/held-out scores, and random-control norms. It compares each constrained
delta with a separate augmented-system least-squares solution and measures span
drift using QR. This is a numerical audit, not a formal certificate.

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m \
  benchmarks.weight_space.constrained_edit_job --out /tmp/constrained-reproduction
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m \
  benchmarks.weight_space.verify_constrained_edit /tmp/constrained-reproduction
```

JSON evidence and source snapshots are retained here. Binary inputs and model
states are retained locally as ignored safetensors files; the command above
regenerates a complete auditable bundle in a fresh directory. `verification.json`
records the audit of the original bundle. The tests also regenerate the experiment
and reject corrupted evidence even after its artifact hashes are recomputed.

## Implication

The constrained solver supplies a useful local equality contract, including
sequential preservation. Minimizing weight norm alone can force a large response
outside the reference span. A useful next comparison is a locality-weighted
objective on separate design activations, followed by untouched locality tests.
This experiment does not demonstrate language-model reasoning improvements,
knowledge-editing efficacy, or superiority to retraining.
