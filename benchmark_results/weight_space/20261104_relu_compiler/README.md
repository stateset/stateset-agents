# From one handwritten repair to a reusable arithmetic compiler

The same compiler writes eight supplied arithmetic programs into **96 existing
ReLU networks**, without optimizer updates or changing their architectures.
All **492,192 predictions** pass the fixed absolute tolerance of `2e-6`; the
largest observed float32 error is **8.940696716308594e-7**. All **98,976 dyadic
grid predictions** are exact. A separate rational checker verifies the stored
networks' real-arithmetic identities over their entire declared input boxes.

The compiler also produces tensors identical to the earlier handwritten
69-coordinate repair for **all three trained graph checkpoints**. This is a
checkpoint compatibility check against the [graph-repair experiment](../20261104_graph_repair/README.md),
not a new set of graph predictions or a new training comparison.

## What is reusable

[`relu_program.py`](../../../stateset_agents/experimental/weight_space/relu_program.py)
accepts a scalar function of the form

`f(x) = linear @ x + bias + sum(scale[j] * ReLU(weights[j] @ x + offset[j]))`.

It compiles this specification into an existing float32 CPU `nn.Sequential`
with one or more `Linear/ReLU` hidden pairs and a scalar `Linear` output. Hidden
widths may differ. Each needs space for the nonnegative channels: one or two for
the linear term, plus one per nonzero hinge. Coefficients must already be exactly
representable in float32; the compiler rejects silent quantization.

The result contains a cloned state dictionary and a ledger of every assigned
scalar, including no-op assignments. It neither mutates the supplied model nor
consumes RNG state. It validates the architecture, dimensions, biases, finite
weights, parameter aliasing, unsupported forward hooks, and channel capacity.
A conservative interval calculation guards against overflow in all dense
branches, including those disconnected from the readout.

```python
import torch
from torch import nn
from stateset_agents.experimental.weight_space.relu_program import (
    Hinge, ReLUProgram, compile_relu_program,
)

net = nn.Sequential(
    nn.Linear(2, 32), nn.ReLU(),
    nn.Linear(32, 32), nn.ReLU(),
    nn.Linear(32, 1),
)
# max(a, b) = b + ReLU(a - b)
program = ReLUProgram((0, 1), hinges=(Hinge((1, -1)),))
edit = compile_relu_program(net, program, domain=((-4, 4), (-4, 4)))
net.load_state_dict(edit.state_dict)
print(net(torch.tensor([[-3.0, 2.0], [1.0, -4.0]])))  # [[2.0], [1.0]]
```

## Fixed experiment

The protocol and source snapshots were written before evaluation. Every
combination of eight programs, four hidden-layer shapes (`4`, `8→8`, `32→32`,
`24→16→8`), and three deterministic random initializations is reported. There
are no optimizer steps, parameter searches, checkpoint selections, or omitted
failing conditions. The three trained graph checkpoints are separate from
these 96 initially random networks.

All inputs are float32. Seven two-input tasks use a 33×33 dyadic grid and 4,096
uniform samples generated in float64 before conversion to float32. The
four-input Manhattan-distance task uses a 5⁴ grid and 4,096 such random samples.
Inputs are shared across architectures and seeds for each task; these are
repeated evaluations, not 492,192 independent random problems. References use
float64 arithmetic on the exact float32 input values.

| Program | Domain | Channels | Assigned weights in a 32→32 MLP | Maximum absolute error |
| --- | --- | ---: | ---: | ---: |
| Nonnegative addition | `[0,1]²` | 1 | 69 / 1,185 | 5.96e-8 |
| Signed addition | `[-4,4]²` | 2 | 105 / 1,185 | 2.38e-7 |
| Minimum | `[-4,4]²` | 3 | 141 / 1,185 | 2.38e-7 |
| Maximum | `[-4,4]²` | 3 | 141 / 1,185 | 2.38e-7 |
| Absolute difference | `[-4,4]²` | 2 | 105 / 1,185 | 2.38e-7 |
| Difference clipped to `[-1,1]` | `[-4,4]²` | 2 | 105 / 1,185 | 4.77e-7 |
| Triangle pulse `max(0,1−abs(x₀))` | `[-4,4]²` | 3 | 141 / 1,185 | 2.38e-7 |
| Manhattan distance between two 2D points | `[-4,4]⁴` | 4 | 185 / 1,249 | 8.94e-7 |

Counts are scalar assignments; all assigned values actually changed for these
initializations. Error maxima cover every architecture and seed. The second
input of the triangle task is intentionally irrelevant, testing whether the
compiled network ignores it.

## Why the construction works

For a signed affine-free linear term `z = linear @ x`,
`ReLU(z) − ReLU(−z) = z`. If exact rational interval bounds establish `z ≥ 0`
or `z ≤ 0` over the whole input box, only one channel is necessary. Each hinge
already produces a nonnegative channel.

The compiler writes these channels into selected rows of the first hidden
layer. At every later hidden layer, those rows are replaced with identity
connections and zero biases. ReLU preserves their nonnegative values. The final
readout combines them with the specified coefficients and constant bias. Thus
the real-arithmetic network implements the supplied program throughout the box.

For input dimension `d`, hidden widths `w₁,…,wₗ`, and `k` channels, the assignment
count is `k(d+1) + Σ[j=2…l] k(wⱼ₋₁+1) + wₗ+1`. This recovers 69 assignments
for the original nonnegative two-input addition path with widths 32→32.

The independent verifier reads the saved tensors without calling the compiler.
It follows every nonzero readout connection backward, checks the reachable
identity rows, and extracts the first-layer hinges. It then normalizes both
this function and the supplied specification using exact rational coefficients,
domain bounds, and `ReLU(−z) = ReLU(z) − z`. All 96 identities pass. Tests also
confirm that this checker rejects corrupted first-layer, intermediate, and
readout weights, and a domain wider than the construction permits.

These are exact **real-arithmetic** identities. The float32 result has rounding
error, measured separately above. The empirical tolerance is not a universal
floating-point error certificate for arbitrary programs, coefficients, hardware,
or depths. Inference outside the declared domain is not covered.

## Interpretation

This extends a single hand-coded graph repair into a small reusable compiler:
the same implementation supports several functions, signed inputs, differing
dimensions, and differing network widths and depths. It shows how to translate
a known local specification into existing weights and audit the result.

The entire scalar output function is replaced. Unselected learned parameters
remain stored, but their branches are disconnected from the result; the fraction
of unchanged parameters does not measure preserved learned behavior. No claim
is made that these 96 random starting networks learned useful behavior first.

The supplied formulas remain essential human knowledge. This experiment neither
discovers a missing operator automatically nor demonstrates a general advantage
over training. The earlier coverage-supervised graph models also succeeded.
It does not extend the unsuccessful 2B conclude-neuron pilot or establish an
improvement to language-model reasoning. Programs outside the affine-plus-hinge
family require a different construction or approximation.

## Reproduction and retained evidence

From the repository root, with torch and safetensors installed:

```bash
PYTHONPATH=. python benchmarks/weight_space/relu_compiler_job.py \
  --out /tmp/relu-compiler-new-run \
  --graph-evidence benchmark_results/weight_space/20261104_graph_repair

PYTHONPATH=. python benchmarks/weight_space/verify_relu_compiler.py \
  benchmark_results/weight_space/20261104_relu_compiler \
  --graph-evidence benchmark_results/weight_space/20261104_graph_repair
```

- `protocol.json`: fixed programs, input domains, architectures, seeds, tolerance,
  runtime, source hashes, and graph-checkpoint dependency hashes.
- `summary.json`: all 96 outcomes, assignment and parameter counts, and three
  exact graph-repair compatibility checks.
- `verification.json`: replay of every prediction with a separate Python
  arithmetic oracle, edit-isolation checks, and rational circuit certificates.
- `sources/`: frozen compiler, experiment, and independent verifier.
- `checkpoints.safetensors`: all initial and compiled networks, plus the three
  graph compatibility outputs. `inputs.safetensors`: every evaluation input.
- `artifact_manifest.json`: SHA-256 bindings for the retained evidence.

Tensor artifacts are retained locally in this directory and ignored by Git,
consistent with earlier experiments. Source and unit tests are available in
Git; replaying this recorded run needs the retained tensors. The compatibility
check additionally needs the six original/edited graph checkpoints from the
preceding experiment.

Validation: 137 distinct targeted tests pass across the new compiler, symbolic
checker, earlier graph repair/certificate, layering, and type-surface ratchet.
The repository pre-commit checks pass for the changed source and documentation;
the full type check passes on 553 packaged source files. This local CPU
experiment incurs no cloud compute. Changes remain unreleased.
