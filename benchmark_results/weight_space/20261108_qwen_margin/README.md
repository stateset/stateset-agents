# Minimum-change readout targets and post-answer protection

**No condition passes the combined acceptance gate.** A contrastive,
minimum-change residual target fixes the exact edit prompt with a much smaller
write and without repeating the answer, but does not transfer to any of the
twelve paraphrases. It reduces model drift relative to the earlier strong
embedding write, while still exceeding the fixed drift limits.

This is one synthetic continuation preference, not factual knowledge editing or
a semantic-quality benchmark. The earlier [generation pilot](../20261108_qwen_generation/README.md)
motivated the target-direction and post-answer-protection comparisons.

## Results

| Condition | Residual change norm | Edit answer | Validation | Confirmation | Locality answers | Exact locality generation agreement | Mean validation KL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero update | 0 | 0/1 | 0/6 | 0/6 | 4/6 | 6/6 | 0 |
| Earlier embedding write, scale 128 | 128 | 1/1 | 6/6 | 5/6 | 5/6 | 3/6 | 0.0318310 |
| Minimum change, margin 1 | 50.3266 | 1/1 | 0/6 | 0/6 | 4/6 | 5/6 | 0.00550983 |
| Margin 1, post-answer protection | 50.3266 | 1/1 | 0/6 | 0/6 | 4/6 | 6/6 | 0.00935888 |
| Margin 8, post-answer protection | 51.8589 | 1/1 | 0/6 | 0/6 | 4/6 | 6/6 | 0.00960694 |

The margin-1 change is about 61% smaller than the scale-128 write and its mean
validation KL is about 83% lower. This is a comparison on six fixed sentences;
the result is still 5.51 times the mean-KL cap of 0.001. Its maximum token KL is
0.048714, above the 0.01 cap. The post-answer conditions have *higher*
teacher-forced KL despite preserving all six baseline locality completions
exactly. Only four of those baseline completions are correct answers, so exact
preservation should not be mistaken for correctness.

The edit-prompt completions are:

- Baseline: ` a 4-digit number formed by the`
- Embedding write: ` amber. The amber is amber amber amber`
- Margin 1: ` amber. The game is to guess the`
- Both protected margin targets: ` amber. The first letter is A,`

The embedding arm repeats `amber` up to eight times in one validation completion
and seven times in one confirmation completion. The margin arms mention it once
on the edit prompt and zero times on all paraphrases. None of the 95 completions
ends at EOS; all reach the eight-token limit. Prefix success and the narrow
repetition count are not semantic judgments or proof of fluent longer outputs.

## Fixed design and selection

The model is the local float32 CPU `Qwen/Qwen3-0.6B` checkpoint at revision
`c1899de289a04d12100db370d81485cdf75e47ca`, with eager attention and four threads.
Checkpoint hashes, all prompts, conditions, thresholds and sources are retained
in `protocol.json`. There are no model downloads, gradients or model-training
optimizer steps. The readout target uses a numerical dual coordinate solver.

The edit prompt/answer remains `The secret word is` / ` amber`, at
`model.layers.27.mlp.down_proj`. Keys, two hard-protection sentences, six soft
calibration sentences and locality weight 100 are unchanged from the original
behavior pilot. Its bundle is retained under `base/`. The zero and scale-128
factors were checked byte-for-byte against the preceding generation pilot.

The new target minimizes the Euclidean residual change subject to making the
answer token outrank **every** competing vocabulary row. The final RMSNorm's
learned scale is included. Its positive common denominator cancels for ranking;
the declared margins 1 and 8 are pre-normalization readout units, not final-logit
margins. This argument is specific to a bias-free head after RMSNorm at the final
MLP site. It does not extend unchanged to LayerNorm, biased heads or earlier
nonlinear edit sites.

Both target calculations use seven active constraints and eight vocabulary
scans. Margin 1 takes 222 coordinate sweeps; margin 8 takes 225. The absolute
primal/dual gaps are approximately 3.39e-8 and 1.85e-8, below the declared 1e-6
limit. Full-vocabulary unit-normalized constraint violations are below 1e-8.
These are float64 numerical certificates; actual rounded layer mappings and
whole-model behavior are evaluated separately.

Post-answer protection adds the **last-token** keys of twelve fixed continuations
such as `The secret word is amber` and `The secret word is amber. The`. It does
not protect every token, which would also protect the edit key and conflict with
the desired write. None of the new validation or confirmation paraphrases
supplies target-design or calibration activations.

The six validation and six confirmation paraphrases were declared before the
first attempt and differ from the preceding pilot. Eligibility requires the
unchanged teacher-forced gate, correct edit-prefix generation, at least 50%
validation-prefix accuracy, and preservation of every baseline-correct locality
answer. A new declared repetition gate permits at most two whole-word `amber`
occurrences in any edit/validation completion. Confirmation accuracy and
repetition never select an arm. Ties use validation accuracy, mean KL, then
declared condition order. All five conditions are reported; no solver failures
were excluded and no adapter was promoted or exported.

The completed attempt uses 17 measurement units and 843 actual model forwards:
13 capture forwards, 70 teacher-forced forwards and 760 generation forwards.
These counts exclude the interrupted attempt and the independent replay.

## Serialization repair and replay

The first attempt completed four conditions, then safetensors rejected shared
storage in the cached margin-1 target tensors. The repaired writer stores owned
copies and atomically replaces its factor file. A regression test covers shared
and noncontiguous tensors. The run was repeated with identical experimental
fields; `prior_protocol.json`, `prior_trials.json` and `retry.json` retain the
original declaration and completed results. The replay checks their equality
with the completed run, rather than treating the retry as fresh evidence.

The independent verifier imports no session, target designer, linear edit solver,
or experiment runner. It reloads the model, recaptures keys/residuals, checks each
full-vocabulary target and its dual witness, checks deployed layer mappings and
protection, recomputes teacher-forced metrics, and replays generation through
Transformers. It also checks prefix/repetition scores, gates, rollback, accounting
and the unchanged retry outcomes. `verification.json` records its outcome.

The completed replay **passed**: all 760 generated tokens, the full-vocabulary
target certificates, all five gate decisions and the four completed conditions
from the interrupted attempt matched. It selected no candidate. The focused
regression suite passes 588 tests with one expected skip; type checks pass across
560 source files, alongside lint, formatting and repository hygiene checks.

```bash
PYTHONPATH=. HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m \
  benchmarks.weight_space.margin_trial_job \
  --model /path/to/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --base benchmark_results/weight_space/20261108_qwen_behavior \
  --out /tmp/qwen-margin-reproduction
PYTHONPATH=. HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m \
  benchmarks.weight_space.verify_generation_trial /tmp/qwen-margin-reproduction \
  --model /path/to/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
```

The first command needs the earlier bundle's local ignored `edits.safetensors`;
its README explains regeneration. JSON evidence and frozen source copies are
committed; tensor factors are retained locally as ignored files. A numerical
underflow rejection and an explicit `zip(strict=False)` spelling were added
after measurement without affecting the measured calculations. Frozen source
copies retain the actual experiment and replay implementation.

## Remaining problem

The single-residual optimum solves a local readout problem. It does not supply
a useful model edit across wording changes. Additional hard protection can also
increase drift elsewhere. Further work needs multiple edit contexts, stronger
calibration coverage, and output directions or edit sites that account for
whole-model preservation. These results do not establish which approach will
succeed. The current confirmation cases are consumed evidence; future selection
needs fresh confirmation data.
