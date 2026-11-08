# Full-model keyed editing on Qwen3-0.6B

**Neither learned edit passes acceptance.** Optimizing a small output vector
through the complete model produces complete counterfactual answers on unseen
prompts, but the two candidates trade off transfer, drift, and preservation.
This experiment deliberately replaces France's capital with Lyon; that is the
requested counterfactual, not a claim about the real capital.

The new library method freezes the base model, constructs a fixed input selector
with hard activation protection and a soft locality penalty, and optimizes only
the output vector of a rank-one update. Its objective includes the full target
continuation and EOS. No base parameter or existing parameter gradient changes
during design. The returned proposal still requires behavioral validation.

## Results

A correct complete answer must consist of the declared name, allowing quotes or
a final period, and terminate with EOS. Confirmation never selects a candidate.

| Condition | Complete design answers | Validation | Confirmation | Retained baseline-correct complete locality answers | Mean teacher-forced KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sham | 0/2 | 0/3 | 0/3 | 2/2 | 0 |
| Subject key, layer 14 | 2/2 | 1/3 | 2/3 | 2/2 | 0.001248819 |
| Final-token key, layer 27 | 2/2 | 2/3 | 3/3 | 1/2 | 0.000033533 |

The early edit fails the mean-KL limit (0.001), target accuracy, and baseline
locality-coverage requirement. The final-layer edit passes every teacher-forced
check but fails target accuracy, baseline coverage, and answer retention.
Both restore exact prior weights. No model or adapter is promoted or exported.

The final-layer candidate changes the language answer from `French` to
`Linguist`, and the currency answer from `France uses the euro.` to `L'argent`.
Its missed target validation answer begins `L'Établissement central de la`
and reaches the token limit. These outputs suggest a tendency toward the initial
target token `L` beyond the intended relation; that is an interpretation of the
retained outputs, not a proven internal mechanism. The early candidate preserves
`French` and produces `French currency is the euro.` on the currency question.

| Design measurement | Subject key, layer 14 | Final-token key, layer 27 |
| --- | ---: | ---: |
| Selected optimization step | 20 | 13 |
| Initial target NLL | 2.618176 | 2.618176 |
| Selected target NLL | 0.00125874 | 0.13401929 |
| Selected calibration KL | 0.0000212694 | 0.0000000000544 |
| Selected penalized objective | 0.184126 | 0.135503 |
| Output-vector norm | 38.5971 | 40.0000 |
| Rounded weight-update norm | 4.27605 | 0.385139 |

The final layer's near-zero calibration KL does not protect the held-out French
language answer. Generic calibration questions omit other properties of the
edited subject. The next experiment should include subject-related preservation
data and evaluate new paraphrases of those properties. The present locality and
confirmation outputs are now consumed evidence.

## Scope and baseline limitations

Chat formatting improves answer termination compared with the earlier raw
completion pilot, but the base model still produces the exact original name on
only 3/5 design/validation prompts. It answers `France` to `Name the capital city
of France.`, and gives a longer response to the seat-of-government question.
All three baseline confirmation answers are `Paris`.

Only two locality outputs satisfy the declared complete-name format: `Berlin`
and `French`. The Italy and currency answers are sensible full sentences but do
not meet that format; the Spain answer is incorrectly `Spain`. Consequently the
fixed minimum of three baseline-correct locality answers is unmet from the
start. No questions, aliases or thresholds were changed after observation.
This limits the acceptance evidence even before considering edit-induced harm.

The experiment uses one fact, one small model, and a lexical full-answer rule.
It does not establish general knowledge-editing efficacy or semantic evaluation
of arbitrary prose. Both the edit layer and key position differ between learned
conditions, so this is not an isolated causal test of layer choice. The change
from prior pilots also includes chat formatting and a different objective/data
set; cross-pilot score changes cannot be attributed solely to the new optimizer.

## Fixed protocol

- Checkpoint: `Qwen/Qwen3-0.6B`, revision
  `c1899de289a04d12100db370d81485cdf75e47ca`, float32 CPU, eager attention,
  four threads, local files only.
- The system prompt requests only the requested name and no explanation.
  The tokenizer chat template uses `enable_thinking=False`.
- Two design questions, three validation questions, three confirmation
  questions, and five locality questions are declared before forwards.
  Subject indices come from tokenizer offsets over the complete rendered prompt.
- Both learned conditions use a fixed selector with unit response at both keys.
  Thirteen token rows from two sentences provide hard protection; 143 token
  rows from four separate chat questions provide soft locality, weight 100.
- Twenty Adam steps, learning rate 0.5, target NLL plus calibration KL with
  weight 1 and squared update norm with weight 0.01. Output-vector norm is capped
  at 40; dense update norm at 5. The best evaluated design objective selects
  the vector, independently of validation and confirmation.
- The target is `Lyon<|im_end|>`, separately encoded as three tokens. Learning
  covers the whole continuation, not just its first token.
- Teacher-forced limits remain mean KL ≤ 0.001, maximum token KL ≤ 0.01,
  NLL increase ≤ 0.005, top-1 agreement ≥ 0.95, and mean/each target gain ≥ 0.01.
- The library answer gate requires every design/validation prefix, all
  baseline-correct locality prefixes, at least three such baseline cases,
  and at most two target-name repetitions. An additional complete-name/EOS rule
  checks every target and retention of baseline-correct complete locality names.
- Greedy decoding has an eight-token cap and model-configured EOS. Of 82 retained
  completions, 75 finish with EOS and seven reach the length limit.

All three candidates are retained, including the sham and rejected conditions.
Rejected proposals are reapplied only for diagnostic confirmation, then reverted.
Source snapshots, protocol, checkpoint hashes, rendered prompts, token IDs,
optimization history, factors and all generation outputs are retained.

The run uses **61 measurement units, 598 actual model forwards, and 240 model
backward calls**. Forward accounting is 8 activation captures, 252 design
forwards, 48 teacher-forced validation forwards, and 290 generated-token forwards.
Each learned condition uses 126 forwards and 120 model backwards. Scalar
regularizer derivatives are additional local tensor work.

## Independent checks

Independent replay passed for all three conditions. It regenerated 182 tokens;
the remaining 108 retained tokens are repeated baseline records checked against
the independently regenerated baseline. All acceptance decisions, complete-name
scores, rollback checks, and accounting matched. All 37 original artifact hashes
were verified.

The verifier imports no optimizer, session, constrained solver, or experiment
runner. It checks each fixed selector using a Woodbury inverse and a separate
equality solve. It applies the saved factors directly to dense weights and
recomputes each selected target NLL, calibration KL and penalized objective.
This checks the selected candidate, not every optimizer iterate or a global
optimum. It separately checks protected-span drift, unit-key responses and
generated tokens through Transformers generation.

The focused regression suite passed 540 tests with one expected skip. The new
optimizer tests include full gradient comparisons against finite differences
of dense-weight NLL, KL and regularization, independently measured locality
drift, exact state/gradient preservation, bounded work and failure cleanup.
All 563 packaged source files pass type checking; Ruff, Black, isort and
repository hygiene checks pass.

```bash
PYTHONPATH=. HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m \
  benchmarks.weight_space.keyed_model_trial_job \
  --model /path/to/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --out /tmp/qwen-keyed-reproduction
PYTHONPATH=. HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m \
  benchmarks.weight_space.verify_keyed_model_trial \
  --root /tmp/qwen-keyed-reproduction \
  --model /path/to/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
```

JSON evidence and frozen source copies are committed. The ignored
`edits.safetensors` remains local and can be regenerated with the runner.
Checkpoint weights are not copied into this bundle.
