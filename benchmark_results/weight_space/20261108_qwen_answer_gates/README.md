# Atomic generated-answer gates on two counterfactual factual edits

**No candidate is accepted.** Three nonzero candidates pass the teacher-forced
limits but fail generated-answer validation. The library now makes these checks
part of `trial_edit`, so failed answers, locality retention, repetition, or
candidate decoding restore the prior weights and edit stack automatically.

This pilot deliberately requests false replacements: France's capital → Lyon,
and Hamlet's author → Dickens. It tests an editing mechanism, not the truth of
those statements. Two facts on one small model are not a knowledge-editing
benchmark or evidence of general efficacy.

## Measured outcomes

| Task / condition | Design prefixes | Validation | Confirmation | Retained baseline-correct locality | Mean teacher-forced KL | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| France / sham | 0/2 | 0/3 | 0/3 | 5/5 | 0 | Reject |
| France / narrow | 2/2 | 0/3 | 0/3 | 5/5 | 0.000039761 | Reject: target accuracy |
| France / broad | 2/2 | 0/3 | 0/3 | 5/5 | 0.000006634 | Reject: target accuracy |
| Hamlet / sham | 0/2 | 0/3 | 0/3 | 2/2 | 0 | Reject |
| Hamlet / narrow | 2/2 | 0/3 | 2/3 | 1/2 | 0.001248631 | Reject: drift, accuracy, baseline coverage, retention |
| Hamlet / broad | 2/2 | 0/3 | 1/3 | 2/2 | 0.000252417 | Reject: accuracy, baseline coverage |

Both France variants and the broad Hamlet variant pass every teacher-forced
limit. Their mean target log-probability gains are 3.369, 3.352, and 8.291 nats,
respectively. Better probability and exact design-prompt outputs do not imply
success on the declared validation paraphrases. Broader preservation reduces
France's measured mean KL by 83.3% and Hamlet's by 79.8%, with no validation
answer-prefix successes. Confirmation never selects a candidate.

The narrow Hamlet edit changes `Macbeth was written by` from `Shakespeare` to
`Dickens`; the broad variant preserves that baseline-correct answer. France
locality prefixes remain correct, although three completions change their later
sentence from `The capital of France is Paris` to `... Lyon`. Retention here
measures the declared answer prefix, not identical text throughout a completion.

The base model answers the original fact on only 3/5 France and 1/5 Hamlet
design/validation prompts. It meets the required locality baseline on France
(5 correct), but not Hamlet (2 versus the declared minimum of 3). Both facts and
all conditions were kept; prompts, aliases, and thresholds were not repaired
after observing outputs. The Hamlet results cannot establish adequate baseline
knowledge or locality coverage under this prompt format.

Prefix scoring is deliberately narrow. For example, the narrow Hamlet edit
produces `Dickens, but Dickens is not the author` on a design prompt, which passes
the prefix and two-occurrence repetition checks despite contradicting the
requested edit. A base completion `famous play written by Shakespeare. It is`
does not match the predeclared `play` or `tragedy` prefixes. These examples limit
the interpretation of both target accuracy and baseline coverage. There is no
semantic correctness claim.

## Frozen protocol

- Cached `Qwen/Qwen3-0.6B`, revision
  `c1899de289a04d12100db370d81485cdf75e47ca`; float32 CPU, eager attention,
  four threads, no downloads or gradient updates.
- Two design, three validation, three confirmation, and five locality prompts
  per task. The first France design prompt appeared as locality data in earlier
  pilots and is explicitly development data here. Confirmation is disjoint from
  design and validation and never supplies solver activations.
- Each nonzero candidate jointly solves full-vocabulary margin constraints for
  two contexts at `model.layers.27.mlp.down_proj`, with margin 8 before RMSNorm
  and locality weight 100. Hard protection uses all tokens from two fixed sentences. Narrow
  soft preservation uses six sentences; broad preservation adds 32 sentences
  and four last-token contexts after the task's design answers.
- Teacher-forced locality uses six separate sentences. Target gains cover both
  design continuations. Limits remain mean KL ≤ 0.001, maximum token KL ≤ 0.01,
  NLL increase ≤ 0.005, top-1 agreement ≥ 0.95, and mean/each target gain ≥ 0.01.
- Generated-answer acceptance requires all five design/validation prefixes,
  retention of every baseline-correct locality prefix, at least three such
  baseline cases, and at most two target-answer occurrences in each target
  completion. These checks run inside the library transaction.
- Greedy decoding uses eight new tokens, model-configured EOS, and a per-prompt
  cache. All 144 retained completions reach the length limit, with no EOS.

`protocol.json` and source snapshots were written before model forwards.
`tasks.json` retains cases and token IDs; `trials.json` retains baseline/candidate
tokens, text, scores, checks and exact rollback observations. The decoder was
instrumented to capture the exact completions scored by the library, avoiding a
second scoring-time generation. Rejected candidates were reapplied only for
diagnostic confirmation, then reverted exactly. No adapter was exported.

The run spends 39 measurement units and 1,300 actual model forwards: 52 capture,
96 teacher-forced, and 1,152 generated-token forwards. Independent replay is
additional work.

## Independent replay and reproduction

`verify_answer_gate_trial.py` imports neither the session, answer gate, solver,
nor experiment runner. It uses Transformers generation for exact token replay,
an independent prefix scorer and lexical span counter, the scalar-position
teacher-forced oracle, and the existing Woodbury joint-edit certificate checker.
It checks hashes, captures, all vocabulary constraints, rounded protection and
margins, decisions, rollback, and accounting. Repeated baseline records are
checked against one independently generated baseline per task.

**Independent replay passed**: all six combined decisions, all four joint-edit
certificates, and 832 regenerated tokens matched. Repeated baseline records
account for the remaining 320 of the 1,152 retained tokens. All 35 original
artifact hashes, 1,300 model forwards and 39 measurement units reconciled.
`independent_verification.json` records the result and executing verifier hashes.

The focused regression run passed 512 tests with one expected skip. The final
answer-gate suite passed all 37 tests, including additional decoder-factory and
multiple-EOS cases. All 562 source files passed type checking; Ruff, Black,
isort and repository hygiene checks passed.

The first replay stopped on EOS metadata order: the model lists `[151645, 151643]`
while the session records the same stop-token set sorted as `[151643, 151645]`.
The verifier now canonicalizes that set. `verification_revision.json` retains
the failed attempt and old/new verifier hashes; `verification_sources/` retains
the revised source. The original protocol, source snapshots, generations,
thresholds and artifact manifest are unchanged. Replaying this original bundle
with the revised verifier requires `--allow-verifier-revision`; fresh runs freeze
the revised source and do not need that flag.

```bash
PYTHONPATH=. HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m \
  benchmarks.weight_space.answer_gate_trial_job \
  --model /path/to/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --out /tmp/qwen-answer-reproduction
PYTHONPATH=. HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m \
  benchmarks.weight_space.verify_answer_gate_trial \
  --root /tmp/qwen-answer-reproduction \
  --model /path/to/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
```

The ignored `edits.safetensors` file remains local; reproduce the run to regenerate
it on another checkout. JSON evidence and frozen source copies are committed;
checkpoint weights are not copied. The next experiment needs reliable baseline
answers, semantic checks of complete answers, fresh confirmation prompts, and
an editing objective that improves transfer beyond the supplied keys. Current
confirmation cases are now consumed evidence.
