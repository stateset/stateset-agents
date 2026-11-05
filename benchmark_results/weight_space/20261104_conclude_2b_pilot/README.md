# A controlled conclude-neuron pilot on Qwen3.5-2B

One candidate was fixed before practice evaluation: two `Wait` token rows scaled
by 0.5, plus one rewritten layer-23 MLP neuron. The detector is ridge-fitted from
labeled training-split activations. This is targeted editing without gradient
descent, not construction without data or fitting.

The candidate achieved **46.25% strict accuracy with thinking on**, versus
**74.00% for unedited thinking off**. It **failed the prespecified gate**:
`beats_base_nothinking`. No final-test generation was run.

## Practice results

Every condition answers the same 400 problems (200 GSM8K-train and 200 MATH-train),
with an 8,192-token maximum. These problems are disjoint from the 256 GSM8K design
problems. They are a selection set, not a final generalization test.

| Checkpoint | Thinking-on strict | Thinking-off strict | Thinking-on cap hits | Mean thinking-on tokens | Thinking-off stray close |
|---|---:|---:|---:|---:|---:|
| Base | 20.75% | 74.00% | 79.25% | 7,200.1 | 0.25% |
| Token-only | 35.75% | 73.75% | 63.75% | 6,525.4 | 0.25% |
| Candidate | 46.25% | 74.00% | 52.00% | 5,793.5 | 0.50% |
| Random-write control | 32.00% | 73.75% | 67.25% | 6,671.0 | 0.50% |

Strict correctness requires a correct extracted answer, a natural stop, and
exactly one generated `</think>` with thinking on or none with thinking off.
Lenient results, per-subset results, and completion-at-budget curves are retained
in `pilot_summary.json`. Curves at smaller budgets are derived from the recorded
8,192-token responses, not separate runs at each cap.

Against unedited thinking off, candidate thinking on changes strict accuracy by -27.75 percentage points (paired bootstrap 95% interval [-32.75, -22.75]); 13 problems improve and 124 regress, with McNemar p=6.84e-24.

Against the matched control with thinking on, candidate thinking on changes strict accuracy by +14.25 percentage points (paired bootstrap 95% interval [+9.25, +19.25]); 85 problems improve and 28 regress, with McNemar p=7.344e-08.

## Frozen design and decision rule

The design uses 256 baseline thinking traces capped at 3,072 tokens and the same
256 problems with thinking off capped at 2,048. It collects 412 answer-newline
positives, 416 working-newline negatives, 93 post-thinking negatives, 206
answer-token negatives, 300 thinking-off newline negatives, and 200 prose
newline negatives. The latter two groups use training sources. A ridge penalty
of 100, read gain 4, and write norm 3 were chosen before the run. The automatic
slot selection chose layer 23, neuron 4831, and up-unit 2876. The `heldout_*`
fields in `neuron_2b_pilot_summary.json` describe withheld design contexts;
they are not held-out task accuracy.

The four checkpoints are unedited base, token-only ablation, candidate, and
matched control. The control has the same edited slot, read gate, up vector,
write norm, and token-row scaling, but a random write direction. Each checkpoint
runs with thinking on and off. Checkpoints are verified against base plus their
edit specs before serving. There is no tuning on these eight practice results.

The prespecified gate requires candidate thinking-on strict accuracy to exceed
base thinking-on, base thinking-off, token-only thinking-on, and control
thinking-on. Candidate thinking-off accuracy may fall at most one percentage
point and its stray-close rate may rise at most two points relative to base.
Passing this screen would only justify a later independent test; a positive
practice difference alone would not establish a general improvement.
`practice_summary.json` also retains the older thinking-mode selector's
`kept_name: candidate`. That selector does not compare against base thinking off;
the stronger `pilot_summary.json` gate is the final decision and rejects promotion.

## Protocol, verification, and limits

The model revision is `15852e8c16360a2fea060d615a32b45270f8a8fc`.
Sampling uses temperature 0.6, top-p 0.95, top-k 20, seed 1234, and the vLLM default
zero presence penalty. Only one generation per problem and condition is run.
Dataset revisions and samplers are preserved in `sources/`. The loader reads
held-out questions for its existing MATH overlap exclusion, but generates no
final-test answers. The earlier 1,000-problem held-out 2B scores use a different
set and are not paired comparisons with this pilot.

All 3,200 practice replies were regraded locally with matching math-parser
versions. **Every strict outcome reproduces.** Three capped MATH replies have
different lenient grades: candidate thinking-off `mathtrain-3646`, and control
thinking-off `mathtrain-3444` and `mathtrain-4419`. Local parsing of their repeated
boxed answers exceeds the default five-second timeout. Separate diagnostics
with a thirty-second allowance recover all three saved lenient grades in
5.985–6.442 seconds. These diagnostics do not replace the original scores.
All three replies reached the token cap, so both versions of their strict
grade are false. The initial failed audit log, complete replay discrepancies,
and diagnostics are retained in `audit_initial_failure.log.gz`,
`verification.json`, and `parser_diagnostics.json`.

The auditor also verifies executed-source hashes, edit hashes, matched control
vectors/norms, design/practice separation, identical prompts within each thinking
mode, aggregates and paired statistics from the recorded grades, and the
promotion decision. No strict score, budget curve, or gate decision changes
under local regrading. Lenient grading can vary with parser timeouts even with
matching package versions; automatic mathematical grading is not infallible.
The bootstrap intervals and McNemar comparisons are exploratory descriptions
of this fixed selection set.

The A40 allocation lasted 95.9 minutes at the quoted $0.59/hour, an estimated
**$0.943** allocation cost. This is measured runtime multiplied by the quote,
not a billing invoice. The launcher terminated the pod after downloading the
artifacts; a subsequent provider query confirms no allocation remains for this
run (`cleanup_verification.json`).

`per_problem/` retains compact outcomes for every condition. Full generations,
activation captures, and compressed design contexts/logs are retained locally
and ignored by Git. `artifact_manifest.json` binds both tracked and local
artifacts. `sources/` holds the executed jobs and the replay verifier;
`provenance.json` records the uploaded wheel hash and cloud cost. No model
checkpoint or final-test result is released by this pilot.

To repeat the experiment, use the same uploaded jobs listed in
`pilot_protocol.json`, the wheel identified in `provenance.json`, and the package
versions in `runtime_versions.json`; invoke `conclude_pilot_job.py --out
/workspace/out` on a GPU. For a downloaded result directory, install the recorded
math-verify, latex2sympy2_extended, ANTLR, SymPy, and mpmath versions, then run:

```bash
PYTHONPATH=. python benchmarks/weight_space/verify_conclude_pilot.py /path/to/downloaded/results
```

The separate [analytic route network](../20261104_handwritten_routes/README.md)
constructs its weights without fitting; it should not be conflated with this
pretrained-model repair experiment.
