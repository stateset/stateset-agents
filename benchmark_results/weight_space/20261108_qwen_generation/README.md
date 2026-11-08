# Generated-answer follow-up on pinned Qwen3-0.6B

**No condition satisfies the combined selection rule.** The earlier small edits
raise the target's log-probability but never generate the requested answer in this
pilot. A much larger edit generates `amber` on the edit prompt, all six validation
paraphrases and five of six confirmation paraphrases, while exceeding the fixed
model-drift limits and often repeating the answer.

This is a controlled continuation-preference experiment, not a factual knowledge
editing or semantic-quality benchmark. These results qualify the probability-only
claims in the [earlier pilot](../20261108_qwen_behavior/README.md).

## Outcomes

| Condition | Edit prompt | Validation prefixes | Confirmation prefixes | Locality answers | Exact locality generation agreement | Teacher-forced gate | Mean validation KL |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Zero update | 0/1 | 0/6 | 0/6 | 4/6 | 6/6 | Rejected: no target gain | 0 |
| Unweighted, scale 0.5 | 0/1 | 0/6 | 0/6 | 4/6 | 6/6 | Accepted | 5.60242e-6 |
| Weighted, scale 0.5 | 0/1 | 0/6 | 0/6 | 4/6 | 6/6 | Accepted | 4.40073e-7 |
| Weighted, scale 8 | 0/1 | 0/6 | 0/6 | 4/6 | 6/6 | Accepted | 1.13196e-4 |
| Weighted, scale 32 | 0/1 | 0/6 | 0/6 | 4/6 | 5/6 | Rejected: KL | 0.00184373 |
| Weighted, scale 128 | 1/1 | 6/6 | 5/6 | 5/6 | 3/6 | Rejected: KL and agreement | 0.0318310 |

The mean target log-probability gains for the nonzero conditions are approximately
`0.122`, `0.122`, `1.930`, `7.414` and `14.710` nats. Even a `7.414`-nat gain does
not make the target the generated answer in this run. At scale 128 the maximum
validation token KL is `0.368335`, compared with the declared cap `0.01`.

The zero update produces ` a 4-digit number formed by the` after
`The secret word is`. At scale 128 the completion is
` amber. The amber is amber amber amber`. Another validation completion is
` amber, amber, amber amber amber amber`. All 114 completions stop at the fixed
eight-token limit; none ends at EOS. Exact generated IDs, text decoded with and
without special tokens, stops and scores are retained in `trials.json`.

As a descriptive check after the run, 10 of the 13 target-related completions at
scale 128 contain at least three occurrences of the word `amber` (maximum six).
This repetition count was not used for selection and is not a preregistered
quality metric. It demonstrates why answer-prefix success alone is insufficient.
The four locality answers correct in the baseline remain correct in every arm;
the scale-128 increase to five correct answers does not establish general quality
improvement on this tiny set.

## Frozen protocol

The model, checkpoint hashes, float32 CPU execution and eager attention are
unchanged from the earlier pinned experiment:
`Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`.
The first four conditions reuse its exact saved factors. New weighted conditions
use scales 32 and 128 with the same layer, key, protected inputs, locality inputs,
direction, locality weight 100 and solver tolerances. The earlier evidence bundle
is retained under `base/`. There are no optimizer steps or model downloads.

All conditions, prompts, answers, limits and the selection rule are written to
`protocol.json` before model forwards. There is one edit prompt, six validation
paraphrases, six separate confirmation paraphrases, and six unrelated factual
completion prompts. The original six sentence-locality prompts are separately
reused for teacher-forced gates. None of the new paraphrases supplies solver keys
or locality activations.

Generation uses greedy argmax, cache enabled, eight new tokens maximum, and the
pinned model's EOS IDs. Prompts are decoded individually without padding or
sampling. Answer-prefix scoring normalizes Unicode, case, whitespace and opening
quotes, then checks a complete answer boundary. It does not count a later answer,
negation or a longer word containing the answer. It is not a semantic judge.

Eligibility requires the unchanged teacher-forced gate, a correct edit-prompt
answer, at least 50% validation-prefix accuracy, and retention of every baseline-
correct locality answer. Eligible conditions would be ordered by validation
accuracy, then validation mean KL, then declared condition order. Confirmation
never selects a condition. All conditions are measured and reported, including
rejected ones; no thresholds are loosened after seeing them. No adapter is exported
or promoted, and every condition restores the original weight bytes exactly.

The session spends 18 measurement units: 12 teacher-forced snapshots and six
generation measurements. Actual work is 996 model forwards: 84 teacher-forced
microbatch forwards plus 912 generation forwards. Generation audit records make
this distinction explicit.

## Replay and reproduction

The independent verifier imports neither the session, its generator/scorer, the
solver nor the experiment job. It reloads the pinned model, applies saved factors
directly, recomputes teacher-forced metrics with the earlier scalar-position
oracle, and calls Transformers' `generate` to replay every generated token. It
independently checks answer prefixes, stopping conditions, selection, rollback and
forward accounting against retained records.

The completed replay matched all **912 generated tokens** and all six conditions;
`verification.json` records the successful audit and the unchanged selection of
no candidate. There were no solver-design failures to exclude from replay.

```bash
PYTHONPATH=. HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m \
  benchmarks.weight_space.generation_trial_job \
  --model /path/to/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --base benchmark_results/weight_space/20261108_qwen_behavior \
  --out /tmp/qwen-generation-reproduction
PYTHONPATH=. HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m \
  benchmarks.weight_space.verify_generation_trial /tmp/qwen-generation-reproduction \
  --model /path/to/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
```

The first command needs the earlier bundle's ignored `edits.safetensors`; its
README explains how to regenerate it. JSON evidence and executed source copies
are committed; tensor factors are retained locally as ignored files. Type
annotations were tightened after measurement without changing numerical behavior.
The frozen source copies preserve the executed implementation. Model weights are
not copied into this repository.

## What this changes next

The existing linear solver achieves its declared layer constraints, but this
pilot's final-layer, embedding-derived write direction either underperforms on
generated answers or causes substantial drift and repetition. Further tuning
needs better write targets and preservation on post-answer contexts, followed by
fresh confirmation cases. These confirmation prompts are now consumed evidence.

As research context, [ROME](https://rome.baulab.info/) investigates subject-related
MLP computations and evaluates specificity and generalization. Its mechanism
differs from this pilot's final-layer write. That motivates comparing write sites
and targets; it does not establish that ROME or a proposed replacement will work
on this model. This experiment claims neither that comparison nor general
knowledge-editing efficacy.
