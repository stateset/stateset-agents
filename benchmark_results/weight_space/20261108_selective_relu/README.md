# Selective arithmetic edits with preserved learned outputs

The compiler replaces one output of nine trained sparse ReLU networks with
three supplied arithmetic functions. **All 27 edits** pass independent target
identity and preservation checks. All **139,995 target predictions** satisfy
the fixed absolute tolerance of `2e-6`, with maximum observed float32 error
**4.76837158203125e-7**. All **279,990 protected output values** remain exactly
equal under the recorded CPU execution. Nine controls with connected output
paths are rejected.

This extends the [scalar arithmetic compiler](../20261104_relu_compiler/README.md)
to selective replacement. The earlier compiler disconnected all other branches
from its scalar readout. Here, two other trained output functions retain their
complete dependency paths. The sparse connectivity was deliberately supplied
before training; this does not find spare neurons in an arbitrary dense model.

## Fixed protocol

Three hidden-layer shapes (`8`, `12→8`, `16→12→8`) and three seeds produce nine
ordinary two-input, three-output `Linear/ReLU` networks. Even-indexed hidden
rows can serve the edited output. Odd-indexed rows read only earlier odd rows,
and protected output rows 0 and 2 read only the last hidden layer's odd rows.
This connectivity is masked at initialization and throughout training.

Each network trains all three outputs for 400 full-batch Adam updates at 0.01
on the same 512 uniform inputs from `[-4,4]²`, using fixed formulas:

- Output 0: `sin(x0) + 0.5*x1`.
- Output 1: `x0 - x1`.
- Output 2: `cos(x1) - 0.25*x0`.

Every final checkpoint is retained. There is no checkpoint, seed, learning-rate,
or architecture selection. Training histories show both protected outputs
improving for every seed. They are learned approximations, not exact sine or
cosine functions. The edits preserve those approximations, including their
remaining errors.

| Hidden widths | Parameters | Output 0 evaluation MSE across seeds | Output 2 evaluation MSE across seeds |
| --- | ---: | ---: | ---: |
| 8 | 51 | 0.29458–0.39073 | 0.05788–0.07353 |
| 12→8 | 167 | 0.07560–0.22884 | 0.05956–0.07535 |
| 16→12→8 | 383 | 0.02242–0.08476 | 0.05418–0.08586 |

These values are identical before and after each edit. They describe the three
fixed seeds, not confidence intervals or a comparison with retraining.

Each edit starts from the same saved trained checkpoint and replaces output 1
with one of the following formulas. The other outputs are protected.

| Supplied target | Assignments: 8 / 12→8 / 16→12→8 | Maximum absolute target error |
| --- | ---: | ---: |
| `x0 + x1` | 15 / 41 / 75 | 2.384185791015625e-7 |
| `max(x0, x1)` | 18 / 57 / 108 | 2.384185791015625e-7 |
| `clip(x0-x1, -1, 1)` | 15 / 41 / 75 | 4.76837158203125e-7 |

Counts include no-op assignments; every before/after scalar is retained in
`edit_ledgers.json`. No optimizer steps, new layers, or custom forward methods
are used during editing. The original target function is intentionally replaced;
it is not included in the preservation claim.

All edits evaluate the same 1,089-point dyadic grid and 4,096 uniform float32
inputs, disjoint from the training inputs. All 29,403 grid target predictions
are exact. The larger prediction totals repeat these 5,185 inputs across
conditions; they are not independent random problems. Uniform evaluation inputs
are generated in float64 and then rounded to float32. Target references use
float64 arithmetic on those stored float32 values.

## Independent checks

The replay verifier does not call the compiler or training job. It reloads all
saved weights, reconstructs ordinary Linear/ReLU models, checks the edit ledgers,
recomputes the numerical outcomes and final training errors, verifies complete
condition coverage, and checks source and artifact hashes.

Two separate symbolic checks operate on the stored tensors:

1. **Arithmetic identity:** trace the selected readout through its identity
   wires, extract the first-layer hinges, and compare exact rational normal
   forms with the supplied function over the declared domain.
2. **Preservation:** trace every nonzero edge backward from outputs 0 and 2.
   Every reachable affine row, including its zero edges and bias, must be
   unchanged. Induction through the ordinary affine/ReLU layers establishes
   equality of these real-arithmetic functions.

The compiler also checks conservative bounds on both the initial and edited
dense networks, including unused branches. The empirical float32 equalities
above are separate from the real-arithmetic proof; they are not a general
floating-point error certificate for different programs, hardware, or precision.

For each base network, a control makes protected output 0 read every last-hidden
row with coefficient 0.25. This removes all available last-layer capacity. Every
control is rejected before producing an edited state. Unit tests separately
cover indirect shared paths, noncontiguous channels, subnormal nonzero edges,
constant targets, named layers, overflow, and corrupted artifacts.

Review found that enabling CPU subnormal flushing can make a stored nonzero
edge compare equal to zero. The compiler and symbolic verifier now reject that
mode explicitly. The original successful numerical run was retained locally
at `/tmp/stateset-selective-relu-before-denormal-guard`; the recorded run here
repeats the same frozen conditions with the guard and final source snapshots.
No formula, seed, training budget, or tolerance was adjusted based on results.

## Reproduction and limits

From the repository root with torch and safetensors installed:

```bash
PYTHONPATH=. python benchmarks/weight_space/selective_relu_job.py \
  --out /tmp/selective-relu-new-run
PYTHONPATH=. python benchmarks/weight_space/verify_selective_relu.py \
  /tmp/selective-relu-new-run
```

`protocol.json` records the specifications before training or evaluation.
`training.json`, `summary.json`, `edit_ledgers.json`, and `verification.json`
retain all results. `sources/` contains the executed experiment, compiler, and
verifiers. `artifact_manifest.json` binds both tracked metadata and the locally
retained `inputs.safetensors` and `checkpoints.safetensors` files. Tensor files
are ignored by Git, consistent with earlier experiments; rerunning the recipe
regenerates a complete auditable bundle when they are absent. The verifier's
report is regenerated from the bound inputs rather than included in its own
manifest.

The nine CPU training runs took about 2.92 seconds in total; this excludes
setup, compilation, and verification. There was no cloud allocation or spend.
The method requires disconnected capacity and a supplied target formula. It
does not preserve a selected output's old function, handle arbitrary dense
models, discover formulas, or establish improved language-model reasoning.
The unsuccessful [2B conclude-neuron pilot](../20261104_conclude_2b_pilot/README.md)
remains a separate result.

Validation: 256 targeted weight-space, compiler, replay, and documentation tests
pass; full type checking passes on 555 packaged source files. Ruff, Black, isort,
and repository hygiene pass. A broader regression run also surfaced six
unchanged layering allowlist entries whose removal deadlines are v0.72.0;
those deadline checks fail on the release version and remain separate follow-up
work. The full repository test suite was not rerun.
