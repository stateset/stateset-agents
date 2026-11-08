# Joint multi-context editing and preservation coverage on Qwen3-0.6B

**No condition passes the combined acceptance gate.** Joint optimization reduces
the declared weight/locality cost compared with independently chosen residual
targets. Four-context edits achieve limited paraphrase transfer. Broader
preservation lowers model drift but removes that observed transfer. These are
useful distinctions between local optimization, preservation and generalization;
they do not establish a deployable knowledge edit.

## Measured outcomes

| Condition | Design answers | Validation | Confirmation | Locality answers | Mean validation KL | Maximum validation answer repetitions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero update | 0/4 | 0/6 | 0/6 | 4/6 | 0 | 0 |
| Joint, one context | 1/4 | 0/6 | 0/6 | 4/6 | 0.00565490 | 0 |
| Separate residual targets, four contexts | 4/4 | 1/6 | 2/6 | 5/6 | 0.00738974 | 4 |
| Joint, four contexts | 4/4 | 1/6 | 2/6 | 5/6 | 0.00709430 | 4 |
| Joint, broader soft preservation | 4/4 | 0/6 | 0/6 | 4/6 | 0.00248395 | 0 |
| Joint, broader hard protection | 4/4 | 0/6 | 0/6 | 4/6 | 0.00249647 | 0 |

Both narrow four-context methods answer these same three unseen prompts:

- Validation: `The chosen word, kept secret, is` →
  ` amber. The amber is amber, amber`
- Confirmation: `I can reveal that the secret word is` →
  ` amber. But I don't know what`
- Confirmation: `The hidden secret word is` →
  ` amber. The amber is amber. The`

The first and third completions demonstrate why prefix accuracy does not imply
good output. Every condition preserves the four baseline-correct locality
answers. Exact locality completion agreement is 6/6 for the zero update, 4/6 for
both narrow four-context methods, and 5/6 for the other nonzero methods.

On identical narrow calibration data, the rounded weight/locality objective
(`(||D||² + 100 * mean(||H D.T||²)) / 2`) is **0.3211184754** for separate
targets and **0.2636360115** for the joint edit, about **17.9% lower**. The joint
ideal objective is 0.2636359825. Its lower cost does not improve the observed
paraphrase scores. It lowers mean validation KL slightly but also crosses the
top-1 agreement limit, which the separate-target method does not.

Broad soft preservation reduces mean validation KL by about **65.0%** relative
to the narrow joint edit, while increasing the rounded weight norm from 0.7261
to 1.6649. This is still 2.48 times the fixed mean-KL cap of 0.001; the maximum
token KL is 0.04225 versus a cap of 0.01. Both broad variants also exceed the
observed-token loss-increase limit. The hard-protected version has similar
behavior and a weight norm of 1.6694. No threshold is relaxed after measurement.

## Declared experiment

The checkpoint remains `Qwen/Qwen3-0.6B`, revision
`c1899de289a04d12100db370d81485cdf75e47ca`, float32 CPU with eager attention and
four threads. Hashes bind the checkpoint, tokenizer, protocol, factors and source
snapshots. No model downloads or gradient-based model training occur.

All conditions are frozen before model forwards. The final-layer edit site is
`model.layers.27.mlp.down_proj`. Four previously measured prompts are explicitly
reused as **development** contexts: `The secret word is`, `Our secret word is`,
`In this game, the secret word is`, and `The word kept secret is`. Each requests
the next token ` amber`. Six new validation and six new confirmation prompts
are disjoint from both preceding generation pilots, including at token level.

The separate-target baseline minimizes each residual change independently and
uses the existing equality-constrained layer solver. Joint conditions optimize
one shared weight update directly against all vocabulary inequalities. The
requested margin is 8 before the final RMSNorm's common denominator. Soft
locality weight 100 and the original two hard-protection sentences are shared.

Narrow locality contains 42 token rows from the original six calibration
sentences. Broader locality adds 251 rows from 32 declared sentences and twelve
last-token rows after answers to the design prompts. Broad hard protection also
protects these locality rows, producing 318 reference rows with numerical rank
291. The input width is 3,072. No validation or confirmation activation supplies
a key or calibration row.

Selection requires all four design-prefix answers, at least 50% validation
accuracy, retention of every baseline-correct locality answer, and the unchanged
teacher-forced limits. A declared repetition rule permits at most two whole-word
`amber` occurrences in every design/validation completion. Teacher-forced target
gain uses the original edit prompt for comparability; generation evaluates all
four design prompts. Confirmation never selects a condition. All six conditions
are retained, including the failed gates. There are no omitted solver failures,
promoted adapters or model exports.

All 132 completions reach the eight-token limit; none emits EOS. The completed
experiment uses 21 measurement units and 1,188 actual model forwards: 48 capture,
84 teacher-forced, and 1,056 generation forwards. Replay work is additional.

## Independent replay

The verifier imports neither the edit solver, whitening implementation, session
nor experiment runner. It recaptures all design/calibration activations and
checks joint stationarity using a **Woodbury inverse metric**, rather than the
solver's inverse-square-root SVD. It checks nonnegative dual witnesses, objective
gaps, every vocabulary constraint, rounded margins and protected activations.
The separate-target baseline receives layer-mapping and margin checks; the joint
optimality certificate applies only to the four joint conditions.

It also reloads the checkpoint, recomputes teacher-forced metrics, calls
Transformers generation to replay exact tokens, and independently recomputes
prefix/repetition scores, selection, rollback and accounting.
`verification.json` records the completed replay outcome.

The replay **passed**: all 1,056 generated tokens, all four joint certificates,
and every acceptance/selection decision matched. No candidate was selected.
The focused regression suite passes 638 tests with one expected skip; all 561
source files pass type checking, alongside lint, format and hygiene checks.

```bash
PYTHONPATH=. HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m \
  benchmarks.weight_space.joint_margin_trial_job \
  --model /path/to/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --base benchmark_results/weight_space/20261108_qwen_behavior \
  --out /tmp/qwen-joint-reproduction
PYTHONPATH=. HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m \
  benchmarks.weight_space.verify_generation_trial /tmp/qwen-joint-reproduction \
  --model /path/to/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
```

The earlier bundle's README explains regenerating its ignored tensor file.
Factors remain locally available as ignored `edits.safetensors`; JSON evidence
and frozen source copies are committed. Model weights are not copied here.

## Implications

The joint solver improves the declared local optimization problem, and additional
context keys create some transfer in this small synthetic preference task.
Preserving more supplied activations does not by itself produce the desired
balance of transfer, locality and continuation quality. The next work needs
model-level objectives or different edit sites, alongside broader tasks and
fresh confirmation data. Current confirmation cases are now consumed evidence.
