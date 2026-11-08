# Weight editing

The experimental package offers two different preservation contracts. Dense
linear editing protects a supplied activation span and checks the deployed
numeric update. Selective ReLU compilation protects other output functions
structurally, but requires disconnected hidden capacity.

## Dense linear edits

By default, `constrained_edit(weight, keys, targets, protect=references, bias=bias)` computes
a minimum-Frobenius-norm update satisfying multiple desired input/output pairs
while leaving the supplied reference activations unchanged within tolerance.
It uses float64 SVDs on CPU and returns float32 `LowRankDelta` factors on the
weight's device. It needs no disconnected neurons and takes no optimizer steps.
The solve uses thin row bases instead of allocating a square input-width
projection. Memory and solve time still grow with the number of reference rows.

```python
import torch
from stateset_agents.experimental.weight_space import constrained_edit

layer = torch.nn.Linear(4, 3)
keys = torch.tensor([[1., 0., 0., 0.]])
references = torch.tensor([[0., 1., 0., 0.]])
targets = torch.tensor([[2., 3., 4.]])  # Full layer outputs, including bias.
edit = constrained_edit(
    layer.weight, keys, targets, protect=references, bias=layer.bias,
    max_update_norm=10,
)
# Inspect edit.diagnostics, then apply to this same unmodified layer.
with torch.no_grad():
    layer.weight.add_(edit.delta.dense().to(layer.weight.dtype))
```

The constraints are `K (W + D).T + b = targets` and `P D.T = 0`. Rows are
normalized before numerical rank selection. Projection roundoff cannot invent
capacity when the protected rows span the whole input space. Conflicting
targets, insufficient numerical capacity, nonfinite updates, a requested norm
cap, and failed rounding/execution checks produce a typed `ConfigurationError`.
No input tensors or random state are modified by solving.

The returned diagnostics report ranks, conditioning, update norm, target error,
reference drift and drift on an orthonormal basis of the retained protected
span. The span error bounds the realized update's operator norm on that span.
Directions discarded by `rcond` are outside that bound. This is numerical
validation, not a formal real-arithmetic certificate. fp16, bfloat16 and float32
weights are supported, but low precision often cannot meet strict tolerances.
Validation disables autocast; rerun model checks under the actual serving
precision, kernel and device. Current retained experiments use float32 CPU.

This approach is related to the nullspace preservation idea in
[AlphaEdit](https://arxiv.org/abs/2410.02355) and equality-constrained editing in
[EMMET](https://aclanthology.org/2024.findings-emnlp.903/). This implementation
solves the stated local minimum-weight-change problem; it is not a reproduction
of either full language-model editing pipeline.

### Reducing changes outside the protected span

Supplying `locality=calibration_inputs` adds a soft penalty on changes to these
input activations. The constraints on target edits and protected references stay
the same. The objective becomes:

```text
minimize ||D||_F² + locality_weight * ||H D.T||_F² / number_of_locality_rows
subject to K (W + D).T + b = targets and P D.T = 0
```

`locality_weight` defaults to 1.0 when locality rows are provided. Zero retains
the original minimum-weight-change solve. Locality rows keep their magnitudes;
repeating all rows leaves the mean-based objective unchanged. Representative
activation coverage matters more than making the penalty arbitrarily large.

The solver uses a thin SVD to apply the inverse square root of the locality
metric inside the allowed edit subspace. It does not construct a dense covariance
matrix. It rejects numerically singular metrics and still checks rounding,
target accuracy and protection in the original coordinates. Diagnostics include
the realized objective, locality RMS drift per output coordinate, and maximum
locality drift. Neither the penalty nor those measurements guarantees locality
on inputs outside the calibration set.

`max_locality_rms=...` optionally rejects an edit whose actual rounded linear
execution exceeds a caller-specified RMS drift on the supplied locality rows.
This is an additional acceptance gate; it does not relax the hard constraints
or search for a different solution. An impossible cap produces an error. Session
application rejects before mutating model weights or undo history.

Use separate design/calibration, validation, and held-out sets when selecting
penalty strengths or acceptance caps. Do not choose settings from final held-out
results. The retained follow-up below uses one strength fixed before evaluation.

### Sessions and sequential edits

`WeightSpaceSession.capture_many([module], prompts, read="input")` captures
module input activations in their original dtype. It spends one measurement
budget unit, possibly using several microbatches. The default `position="last"`
captures the last nonpadding token; `"mean"` averages tokens. Neither mode
captures every token's individual constraints.

`session.apply_constrained(module, keys, targets, protect=references)` solves
and applies the update, returning `(edit_id, result)`. It automatically includes
the layer's bias and supports the solver's tolerances and norm cap. Pass
`locality=calibration_inputs`, `locality_weight=...`, and optionally
`max_locality_rms=...` through the same method for the locality objective. Collect
fresh activations after changes to upstream layers. To retain previous edits
to this same layer, include their input keys among the next edit's references.
This consumes capacity and may eventually make a new request infeasible.

Sessions own detached copies of factors and undo snapshots, reject nonfinite
updates and shared parameter storage, and detect external changes to edited
weights before applying, reverting or exporting. `active_edit_ids` reads
metadata without cloning snapshots. LIFO rollback restores exact prior bytes.

LoRA export concatenates the active factors. Real arithmetic equivalence does
not imply identical floating-point accumulation. The manifest now reports
`merge_weight_max_errors` from a CPU comparison against the session's weights;
check the reloaded adapter's outputs before serving it. In particular, two
small bfloat16 in-place updates can round differently from a combined merge.

### What the dense experiment shows

The [retained five-seed experiment](../benchmark_results/weight_space/20261108_constrained_dense/README.md)
trains dense networks and makes two sequential batches of four edits. All ten
constrained edits pass independent minimum-norm and span checks. The largest
error across current/earlier targets and protected references is `3.58e-7`.

That local success has a material cost: the largest held-out output change is
`3.39`, compared with `0.416` for unconstrained edits. Mean original-task MSE
increases from `0.00355` to `0.165`. Unconstrained edits damage earlier targets
and protected references, but cause less drift outside the protected span in
this experiment. Minimum weight change does not ensure minimum behavior change.

The [locality-weighted follow-up](../benchmark_results/weight_space/20261108_locality_weighted/README.md)
reuses those same trained models and edit/reference inputs. It adds 256 fresh
calibration inputs and evaluates on 4,096 fresh confirmation inputs, using a
single predeclared penalty strength of 100. Mean final squared output drift on
confirmation inputs falls from `0.17096` to `0.03498`, a **79.5% reduction**,
with improvement in every seed. Current/earlier target and reference errors
remain below `4.8e-7`. Independent replay checks the weighted optimum using a
separate full-covariance constrained solve and confirms both control arms' weights
are byte-identical to the original experiment.

Damage remains: maximum confirmation drift is `1.37`, and original-task MSE is
`0.03748`, versus `0.00353` for the unedited model. These are dense readout
experiments, not language-model efficacy results. The next validation priorities
are full token coverage for locality activations, model-level acceptance checks,
and realistic knowledge-editing evaluations with independent held-out data.

## Selective arithmetic compilation

The experimental ReLU compiler can replace one output of an existing network
while preserving the functions computed by all other outputs. The caller
supplies an affine-plus-hinge formula and an input domain. The compiler uses
hidden neurons that have no nonzero connection path to the protected outputs.
It rejects networks without enough disconnected capacity.

```python
from stateset_agents.experimental.weight_space.relu_program import (
    Hinge, ReLUProgram, compile_relu_program,
)

# net is an existing vanilla float32 CPU Linear/ReLU Sequential model.
# Example: replace output 1 with max(x0, x1), protecting every other output.
program = ReLUProgram((0, 1), hinges=(Hinge((1, -1)),))
edit = compile_relu_program(
    net, program, domain=((-4, 4), (-4, 4)), output_index=1,
)
# Inspect edit.channel_rows, edit.protected_outputs and edit.assignments first.
net.load_state_dict(edit.state_dict)
```

The returned state dictionary is detached and cloned. Compilation changes
neither the supplied model nor the RNG state. Omitting `output_index` retains
the original scalar-only API, which replaces the complete scalar function.
The new argument must be an integer identifying an existing readout row.

## Preservation contract

The compiler traces all exact nonzero edges backward from every protected
output. It leaves every reachable affine row and bias unchanged. Rewritten
rows therefore cannot influence protected outputs before or after the edit.
Channels can occupy different, noncontiguous rows in each hidden layer.

This establishes preservation of the protected **real-arithmetic functions**.
The target output implements the supplied formula on its declared domain.
Floating-point error is measured separately. Both the initial and edited dense
networks must pass a conservative overflow check on that domain, including
branches disconnected from the readout. CPU modes that flush subnormal values
to zero are rejected, because they can hide stored nonzero edges.

Supported models have one or more ordinary `Linear/ReLU` hidden pairs and a
`Linear` readout. Layers need biases and separate finite, contiguous float32 CPU
parameters. Named Sequential layers are supported by the compiler. Hooks,
parameter aliasing, custom layer subclasses, and silent coefficient
quantization are rejected.

Fully connected trained networks generally have no available channels. Small
weights, inactive neurons on sampled inputs, and cancellation between paths do
not qualify as disconnection. The compiler does not prune weights to make room.
For new models, sparse connectivity can be designed before training and kept
fixed during training, as the experiment below demonstrates. Zeroing connections
in an already trained model is a separate edit with its own potential damage.
A constant target uses no hidden channels and can replace one readout even in a
dense model, subject to the same validation checks.

## Independent audit and experiment

The repository's verifier reads saved tensors independently of the compiler:

- `certify(..., output_index=1)` checks the target arithmetic using exact rational
  identities over the supplied domain.
- `certify_preservation(before, after, output_index=1)` walks protected paths
  and checks every reachable parameter, without trusting the compiler's ledger.
- Replay checks floating-point outputs and every recorded scalar assignment.

These benchmark verifiers assume ordinary Sequential numeric layer names
(`0`, `2`, ...); they are not general proofs for arbitrary model forward methods.
Preservation and target correctness are separate checks: either can pass while
the other fails.

The [retained CPU experiment](../benchmark_results/weight_space/20261108_selective_relu/README.md)
uses nine trained sparse networks and three supplied formulas per network.
All 27 target identities and preservation certificates pass. All 139,995 target
predictions meet the fixed `2e-6` tolerance; all 279,990 protected output values
are exactly equal under the recorded float32 CPU execution.

This establishes selective component replacement under an explicit connectivity
contract. It does not establish a language-model reasoning improvement or an
advantage over retraining. The earlier
[Qwen3.5-2B conclude-neuron pilot](../benchmark_results/weight_space/20261104_conclude_2b_pilot/README.md)
failed its promotion gate and remains a separate result.
