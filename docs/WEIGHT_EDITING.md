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
captures the last nonpadding token; `"mean"` averages tokens. Use
`position="all"` to return every nonpadding token, flattened in prompt/token order.
Input dtype is preserved. `max_rows` defaults to 8,192 rows per module and rejects
oversized requests before running the model. Repeated or missing module calls
are rejected rather than silently duplicating or dropping activation rows.
Capture also accepts non-Linear modules whose selected input/output is a tensor
with full batch and sequence dimensions. For example, `read="input"` on a final
RMSNorm reads the residual stream before normalization. Tuple/dictionary outputs
and modules invoked more than once per microbatch are rejected.

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

### Choosing a write that changes the predicted token

`minimum_margin_target(readout, residual, token_id, output_scale=scale, margin=1)`
finds a minimum-Euclidean-change residual that puts the requested token ahead of
every competing vocabulary row. It scans the complete vocabulary in chunks,
solves a nonnegative dual problem on the constraints it encounters, then checks
global feasibility and the primal/dual objective gap. It returns owned CPU
float64 target/delta tensors, the active token IDs, dual multipliers, and numeric
diagnostics. It makes no model forwards and does not modify the readout.

For a **bias-free head immediately after RMSNorm**, pass the final norm's learned
weight as `output_scale`. The common positive RMS denominator preserves ranking,
so the constraints are linear before normalization:

```text
minimize ||delta||² / 2
subject to ((head[target] - head[j]) * norm_scale) @ (residual + delta) >= margin
for every j != target
```

`margin` is in **pre-normalization readout units**, not final logit units. This
equivalence does not apply to a biased head, LayerNorm, or an edit site followed
by additional attention/MLP blocks. The function accepts tensors and cannot infer
those architectural assumptions for the caller.

For a final MLP output projection in a Qwen-like architecture:

```python
from stateset_agents.experimental.weight_space import minimum_margin_target

name = "model.layers.27.mlp.down_proj"  # Use your actual final layer.
rows = session.capture_many(
    [name, "model.norm"], [prompt], read="input",
)
key, residual = rows[name], rows["model.norm"][0]
value = minimum_margin_target(
    model.get_output_embeddings().weight,
    residual,
    target_token_id,
    output_scale=model.model.norm.weight,
    margin=1.0,
)
layer = model.get_submodule(name)
desired_output = (key.to(layer.weight.device) @ layer.weight.detach().T).double()
desired_output += value.delta.to(desired_output.device)
# Use desired_output as the constrained edit's target, then run trial_edit
# and generation checks before retaining or exporting the candidate.
```

The solver bounds working-set size and total coordinate sweeps. Exhausting them
raises `ConfigurationError` and does **not** prove infeasibility. Duplicate
readout rows make a positive margin impossible and are rejected explicitly.
`feasibility_atol` applies to unit-normalized contrast constraints;
`duality_atol` applies to the absolute objective gap. The returned target must
also have a strictly positive minimum raw margin. These are numerical checks,
not formal real-arithmetic proofs or guarantees after float32/bfloat16 rounding.

This target calculation only addresses the supplied residual. It does not make
paraphrases succeed or bound unrelated model behavior. To reduce repeated writes
after the answer, include captured **last-token** keys from completed-answer
contexts in `constrained_edit(..., protect=...)`. Do not indiscriminately protect
all tokens of those contexts: they include the edit prompt itself, creating a
conflicting constraint. Continue to evaluate unseen continuation contexts and
whole-model locality; an exact constraint at supplied keys is a narrow guarantee.

The retained [Qwen target-design experiment](../benchmark_results/weight_space/20261108_qwen_margin/README.md)
compares this calculation with the earlier embedding write and a post-answer
protection ablation. The margin-1 write uses a residual change of 50.33 instead
of 128 and generates the requested answer without repeating it on the edit
prompt. Its validation mean KL falls from 0.0318 to 0.00551, still above the
fixed 0.001 limit, and it fails all six validation and six confirmation
paraphrases. Additional hard post-answer protection raises teacher-forced KL
despite preserving the six baseline locality completions exactly. No condition
passes the combined gate. Better local targets are useful, but this experiment
does not establish generalization or general knowledge-editing efficacy.

LoRA export concatenates the active factors. Real arithmetic equivalence does
not imply identical floating-point accumulation. The manifest now reports
`merge_weight_max_errors` from a CPU comparison against the session's weights;
check the reloaded adapter's outputs before serving it. `verify_export` below
automates this comparison. In particular, two small bfloat16 in-place updates
can round differently from a combined merge.

### Whole-model acceptance and rollback

A local linear-layer guarantee does not establish that the complete model still
behaves acceptably. `session.trial_edit` tests the proposed low-rank delta against
predeclared model-level limits before retaining it:

```python
from stateset_agents.experimental.weight_space import BehaviorLimits

trial = session.trial_edit(
    module, edit.delta,
    locality_prompts=validation_prompts,
    target_pairs=[("The secret word is", " amber")],
    limits=BehaviorLimits(
        max_mean_kl=0.001, max_token_kl=0.01,
        max_nll_increase=0.005, min_top1_agreement=0.95,
        min_target_gain=0.01, min_each_target_gain=0.01,
    ),
)
# Accepted: the candidate remains active. Rejected: it is already reverted.
# Inspect trial.accepted, trial.violations and trial.metrics.
```

Use a model fully in evaluation mode. The trial snapshots the proposed factors,
runs baseline and candidate measurements on identical token IDs, and compares:

- Full-vocabulary `KL(base || edited)` at every valid locality token, including
  each sequence's final token. A maximum-token cap catches isolated spikes.
- Mean observed-next-token NLL, excluding padding and positions without a next
  token. These loss values and target gains are in nats.
- Next-token top-1 agreement, and mean/per-case continuation log-probability gains
  for optional target pairs. Prompt and continuation are tokenized separately;
  the continuation is teacher-forced, not sampled.

Without registered references, the check consumes two measurement units, each
potentially comprising multiple microbatches. The complete required budget is
checked before it starts. A default cap of
32 million logit elements is checked before forwards when vocabulary size is
available and checked again at runtime. Logits stay float32 on CPU; normalization
and KL use float64 in chunks. The cap bounds individual snapshots and padded
batches, not total model or backend memory.

A failed acceptance gate or post-edit forward exception restores the previous
weight bytes and active edit stack. Failed measurements still spend their budget.
Session reads, edits, forwards and exports use a reentrant lock, preventing other
session operations from interleaving with the trial. Direct external model
mutation is unsupported. Accepted/rejected trial metrics, limits and a hash of
the exact case token IDs appear in the adapter manifest's `behavior_trials`.

These are validation gates for an in-place edit, not guarantees about unseen
prompts, free-running generation, or a subsequently merged adapter. Use untouched
confirmation data under a separate fixed protocol, and recheck exported artifacts.
The trial's own baseline is the current session, including earlier active edits:
its limits bound incremental drift. Register fixed references below to also bound
accumulated drift from the original checkpoint or regression of earlier targets.

The [pinned Qwen3-0.6B pilot](../benchmark_results/weight_space/20261108_qwen_behavior/README.md)
tests one synthetic next-token preference with six validation and six confirmation
sentences. The weighted small edit preserves the unweighted edit's roughly
`0.122`-nat target gain and reduces mean confirmation KL by **82.4%**. A larger
edit also passes the fixed validation limits, but changes the top prediction at
one of 48 confirmation positions. The zero update is rejected for lacking target
gain. Independent replay reloads the pinned pretrained model and recomputes all
metrics and decisions. This is a small mechanism/acceptance experiment, not a
knowledge-editing, generation-quality or reasoning benchmark.

### Cumulative drift and preservation of earlier edits

Small accepted edits can accumulate large changes even when each incremental
check passes. Python sessions can register fixed behavior references that every
later `trial_edit` automatically enforces in addition to its incremental gates:

```python
# Run before the first edit to anchor locality to the original model.
original_reference = session.add_behavior_reference(
    locality_prompts=validation_prompts,
    limits=BehaviorLimits(
        max_mean_kl=0.001, max_token_kl=0.01,
        max_nll_increase=0.005, min_top1_agreement=0.95,
    ),
)

# After accepting an edit, retain its target's current continuation probability.
target_reference = session.add_behavior_reference(
    locality_prompts=validation_prompts,
    target_pairs=[("The secret word is", " amber")],
    limits=BehaviorLimits(
        max_mean_kl=0.001, max_token_kl=0.01,
        max_nll_increase=0.005, min_each_target_gain=-0.01,
    ),
)
# Every subsequent trial must pass both references. The negative target-gain
# threshold allows at most a 0.01-nat mean log-probability drop for this target.
```

Each reference captures the current state once, spending one measurement unit.
It retains exact token IDs, full-vocabulary locality logits, target scores and
fixed limits. Its handle records which edits were active at capture, the token
hash, retained logit count and budget position. Capturing after some edits anchors
to that edited state, not to the original checkpoint. Changed tokenizer callbacks
cannot silently redefine an existing reference's cases.

Trials report both incremental `metrics` and `reference_checks`, with each failed
reference gate identified in `violations`. Any failed gate rejects the candidate
and restores the prior edit stack exactly. A trial costs two measurement units
plus one for each distinct reference case set absent from the trial itself.
Identical ordered token plans share candidate forwards, even when their reference
states differ. Each comparison still uses its own fixed baseline and limits.

The session constructor's `reference_logit_limit` defaults to 32 million retained
float32 elements across all references (about 128 MB). This bounds persistent
reference logits; temporary snapshots, model memory and kernel buffers are
additional. Requests exceeding the limit fail before a forward when the model
advertises vocabulary size, with another check after measurement otherwise.
Every trial also rechecks batching against each reference's `max_logit_elements`.

`session.behavior_references` lists active handles. To explicitly stop enforcing
one and free its logits, call `session.release_behavior_reference(handle)`.
Handles belong to their creating session. Export retains reference metadata,
active/released status and trial checks, including references later released;
raw logits are not exported and these handles cannot restore a session.

References constrain only the supplied cases. They do not establish generation
quality, semantic correctness or preservation on unseen prompts. Direct low-level
`apply` and `apply_constrained` remain ungated operations; use `trial_edit` for
behavioral acceptance. Re-evaluate the final exported artifact independently.

### Measure generated answers

Teacher-forced probability gains do not establish that an edited model generates
the requested answer. Use bounded greedy decoding to measure actual continuations:

```python
from stateset_agents.experimental.weight_space import matches_answer_prefix

outputs = session.generate(
    ["The secret word is", "Our secret word is"], max_new_tokens=8,
)
for output in outputs:
    text = tokenizer.decode(output.token_ids, skip_special_tokens=True)
    correct = matches_answer_prefix(text, ["amber"])
    # Keep the exact token IDs and finish_reason alongside any reported score.
```

`generate` processes prompts separately without sampling or padding. Each prompt
gets its own cache; models that return no cache are measured using full prefixes.
The model must be in evaluation mode. EOS IDs default to the model's generation
configuration, then model configuration; pass `eos_token_ids=()` to disable EOS
stopping. Returned IDs include a stopping EOS token. `finish_reason` distinguishes
EOS from the token limit.

One call spends one measurement unit but can execute up to
`len(prompts) * max_new_tokens` model forwards. Results and the session's
`generations` audit record actual counts. Defaults bound each completed sequence
to 4,096 tokens and the sum of input lengths plus maximum requested output lengths
to 16,384 tokens; these limits reject before a forward. A two-million-element
logit cap is checked against known vocabulary size and returned logits. Failed
calls retain attempted forward counts and completed-prompt counts. Exported audit
records contain token hashes, policy and counts rather than decoded text.

`matches_answer_prefix` checks only a declared answer at the start of a completion,
normalizing case, whitespace, Unicode and opening quotation marks. It rejects
negation, reasoning before the answer and partial words such as `ambergris` for
`amber`. This is an explicit answer-prefix metric, not a semantic judge. Keep
generation validation separate from untouched confirmation cases, and declare
selection rules before inspecting confirmation results. Calling `generate` does
not promote or revert edits automatically.

The [retained generation follow-up](../benchmark_results/weight_space/20261108_qwen_generation/README.md)
exposes a gap in the earlier probability-only pilot: all three nonzero edits that
pass its teacher-forced gate generate zero `amber` answers across 13 target cases.
At scale 128, answer-prefix success reaches 6/6 validation and 5/6 confirmation
paraphrases, but the edit fails fixed drift limits and repeatedly emits `amber`.
No condition passes the combined selection rule. These are plain completion
prompts on one synthetic preference, not evidence of factual editing or semantic
quality. Future changes need fresh confirmation cases.

### Verify the reloaded adapter or merged model

`session.verify_export` reloads actual exported safetensors through PEFT and
compares model outputs with the live edited session. It also enforces every
active behavior reference, so an artifact cannot pass merely by reproducing an
edit that violates the registered cumulative limits.

Supply an independently loaded copy of the same pinned base model in the same
dtype. The verifier checks every named parameter and buffer against the session's
reconstructed base tensor bytes, including tensors outside edited layers. It
rejects shared modules/storage before PEFT can modify the model. Loading the base
and keeping two complete models in memory are the caller's responsibility.

```python
adapter_dir = session.export("./edited-adapter")
reloaded, report = session.verify_export(
    adapter_dir,
    fresh_base_model,
    locality_prompts=validation_prompts,
    target_pairs=[("The secret word is", " amber")],
    mode="merged",  # Use "adapter" to test inference with unmerged PEFT layers.
    limits=BehaviorLimits(
        max_mean_kl=1e-6, max_token_kl=1e-5,
        max_nll_increase=1e-4, min_each_target_gain=-1e-4,
    ),
)
# Success: reloaded is the tested model. Gate rejection: reloaded is None.
# Inspect report.accepted, report.violations and report.reference_checks.
```

The numerical limits in this example are illustrative; declare tolerances for
your model and precision before evaluating it. `report.metrics` measures
`KL(live edited || reloaded)`, observed-token loss increase and target gains
relative to the live edited session. The negative target-gain threshold allows a
small probability loss during export. Registered references retain their own
original baselines and limits. Merged mode uses PEFT's finite-weight merge check;
adapter mode measures its actual low-rank inference path.

Verification requires an export created by the current session with the same
active edit IDs. It checks SHA-256 hashes of the configuration, adapter tensors
and manifest, loads verified private file copies, and rechecks source hashes
after evaluation. The output directory contains `weight_space_verification.json`
with status `unverified`, `passed`, `rejected` or `failed`. Passed/rejected reports
bind artifact hashes, base tensor fingerprint, case hash, edit IDs, limits,
metrics, reference checks, package versions, dtype, device and loading mode.
Re-exporting invalidates an earlier report before overwriting artifacts, including
when the re-export subsequently fails. Prior attempts remain in the session's
`export_verifications` history.

The verifier **takes ownership of `fresh_base_model`**: PEFT can modify it even on
failure. Discard that argument after rejection or error. The live editing model
and its undo stack remain unchanged. Verification preflights its complete budget:
two measurement units plus distinct additional reference case sets. Hashing,
loading and merging do not consume forward-measurement units; they still require
memory and compute. Forward failures spend the attempted measurement and raise
with a failed report.

Evidence covers the chosen runtime and exact token IDs supplied by the session's
encoder. It does not verify a separate serving tokenizer, free-running generation,
other precision modes, different kernels or a later independently saved checkpoint.
Use the returned model for the tested loading path, and revalidate artifacts after
any further transformation. A persisted report applies only to files whose hashes
still match it. Independent held-out evaluation remains necessary for quality claims.

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
are realistic multi-edit and knowledge-editing evaluations, generated-response
checks, and verification of exported models on independent held-out data.

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
