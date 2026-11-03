# Weight-space programming on Qwen3.5-0.8B — 2026-11-02

Direct, hand-designed weight edits (no gradients, no training data) to a small
reasoning model, judged on held-out math with matched controls and paired
statistics. Inspired by "Sparks of Weight-Space Programming" (Tsuchida et al.,
<https://rikiyat.github.io/blog/weight-space-programming/>), which edits weights
by reading them; this study reproduces the *practice* (read weights and
activations, write a targeted edit, verify, test held-out) on a different
model and failure mode. It does **not** reproduce the article's specific
direction-ablation edit, which was null here (see Negative results).

## Headline (held-out, strict scoring, 1,000 problems)

| Edit | Weights changed | Accuracy | vs its control |
|---|---|---|---|
| none (base) | 0 | 1.0–1.3% | — |
| W5T13: "Wait" rows x0.5, `</think>` row x1.3 | 3,072 | 33.6% | +334 / -0 |
| W5T15: "Wait" rows x0.5, `</think>` row x1.5 | 3,072 | 39.1% (37.9% on a second pod) | +378 / -0 |
| One designed neuron (v2) + "Wait" rows x0.5 | 3 x 1,024 + 2,048 | 42.1% (41.2% from its spec) | +373 / -11 |
| Designed neuron v3 (thinking-gated; negatives touched held-out questions, see caveat) + "Wait" rows x0.5 | 3 x 1,024 + 2,048 | **45.1%** | +418 / -9 (45.2% on a second pod) |

The designed neuron is a single MLP unit rewritten by hand into an "answer
computed -> conclude" circuit that fires at the line break after the model has
computed its answer. v2 beat both global edits (vs W5T13 +8.5 points,
p = 7e-9; vs W5T15 +3.0, p = 0.07) but was not confined to thinking: it put a
stray `</think>` into 22% of thinking-off replies and raised general-text
perplexity 5.9%. v3 adds thinking-off and prose line breaks as negatives: it
beats v2 on the same pod (+4.8, p = 5e-4), and outside thinking its effect is
small (stray `</think>` in 3.4% of thinking-off replies, thinking-off GSM8K
-2.0 vs base with p = 0.27, perplexity +0.3%). W5T13 remains the cleanest edit
outside thinking (see Side effects).

## The failure being edited

With thinking on and an 8,192-token budget, Qwen3.5-0.8B
(`2fc06364715b967f1860aea9cf38778875588b17`) almost never finishes: 97–98% of
traces loop to the cap ("Wait, ...? Yes, that's what I did. Wait, ..."),
scoring 1.0–1.2% on held-out GSM8K-test + MATH-500. In 63% of practice
traces it has already computed the gold answer, at a median 6% of the way in.
The edits remove the obstacle to concluding; they do not add knowledge.

## Protocol

- Model: `Qwen/Qwen3.5-0.8B` at the revision above; vLLM 0.33, bf16, L40S
  (practice baseline also checked on H100: identical 1.5%).
- Decoding: thinking on, temperature 0.6, top_p 0.95, top_k 20, seed 1234,
  8,192 new tokens. Base outputs reproduce byte-for-byte across pods (runs 7
  and 9: 8,120,172 identical generated tokens).
- Practice: 200 GSM8K-train problems (`openai/gsm8k` at
  `740312add88f781978c0658806c59bc2815b9866`). Edits were chosen on practice
  only; held-out was run once per kept edit.
- Held-out: GSM8K-test[:500] + MATH-500
  (`HuggingFaceH4/MATH-500` at `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`),
  1,000 problems.
- Every edited checkpoint is verified tensor by tensor against base + spec
  before serving (`stateset-agents weight-edit verify`; `verify_spec.py`).
- Controls: same edit kinds, scales and sizes on other targets (token rows
  This/If for Wait/`</think>`; random output directions of equal norm for the
  designed neuron), same pod, same problems.
- Statistics: paired by problem id — exact McNemar on correctness flips,
  10,000-sample bootstrap 95% CI on the per-problem difference
  (`stateset_agents.experimental.weight_space.protocol`).

### Scoring: strict is primary

The harness grade reads the text after the **first** `</think>`. An edit that
raises the `</think>` logit can emit it inline ("J = 12</think>") while the
model keeps reasoning; the lenient grade then credits whatever number the
continued reasoning ends on. **Strict** accuracy additionally requires that the
generation finished on its own and that `</think>` occurs exactly once. All
claims below are strict; lenient numbers are shown for transparency. Edits at
`</think>` x1.3 or below agree within 1 point; at x1.5 lenient overstates by
8–12 points; at x1.7+ it is mostly artifact (`weight-edit compare --strict`).

## Edits

- **Token rows** (tied embedding `model.language_model.embed_tokens.weight`,
  which is also the unembedding): multiplying a token's row by `c` scales its
  logit by `c` everywhere. Wait = ids 13784 (`Wait`) and 13428 (` Wait`);
  `</think>` = 248069.
  - **W5T13**: Wait x0.5 + `</think>` x1.3 — 3,072 weights.
  - **W5T15**: Wait x0.5 + `</think>` x1.5 — 3,072 weights.
- **Designed neuron** (newline-anchored "answer computed -> conclude" unit):
  MLP neuron 1998 of layer 23 (the least-used unit) is rewritten — gate row =
  4 x a ridge detector over layer-23 gate pre-activations for "line break
  after the answer has been computed", up row = a scaled copy of a
  near-constant up neuron (~1.0 everywhere), down column = 3.0 x the unit
  `</think>` logit direction through the final norm. Plus Wait x0.5.
  3 x 1,024 neuron weights + 2,048 token-row weights.

## Held-out results (1,000 problems)

| Edit | Strict | Lenient | Multi-`</think>` | Matched control (strict) | Run |
|---|---|---|---|---|---|
| base | **1.0–1.2%** | 1.0–1.2% | 0% | — | 6, 7, 8, 9 |
| W5T13 | **33.6%** | 34.7% | 1.3% | **0.2%** | 6 |
| W5T15 | **39.1%** | 50.6% | 36.1% | **0.1%** | 8 |
| W5T15 (replication, new pod) | **37.9%** | 50.9% | 37.2% | — | 9 |
| `</think>` x1.5 alone | **34.9%** | 48.1% | 38.0% | **0.1%** | 7 |
| Designed neuron + Wait x0.5 | **42.1%** | 53.6% | 27.6% | **5.9%** (random output directions, same norm) | kernel |

### Paired comparisons (strict)

| Comparison | Accuracy | Delta [95% CI] | Gained / lost | McNemar p |
|---|---|---|---|---|
| **W5T13 vs matched control** | 0.2% -> 33.6% | **+33.4 [+30.5, +36.3]** | **+334 / -0** | **6e-101** |
| W5T13 vs base | 1.2% -> 33.6% | +32.4 [+29.6, +35.3] | +325 / -1 | 5e-96 |
| control vs base | 1.2% -> 0.2% | -1.0 [-1.7, -0.4] | +0 / -10 | 0.002 |
| W5T15 vs base (run 8) | 1.2% -> 39.1% | +37.9 [+34.9, +41.0] | +381 / -2 | 8e-111 |
| — GSM8K-test 500 | 1.8% -> 49.8% | +48.0 [+43.6, +52.4] | +242 / -2 | 2e-69 |
| — MATH-500 | 0.6% -> 28.4% | +27.8 [+24.0, +31.8] | +139 / -0 | 3e-42 |
| W5T15 vs base (run 9) | 1.0% -> 37.9% | +36.9 [+34.0, +40.0] | +371 / -2 | 7e-108 |
| W5T15 vs W5T13 | 33.6% -> 39.1% | +5.5 [+2.7, +8.4] | +138 / -83 | 3e-4 |
| **W5T15 vs matched control** (run 9) | 0.1% -> 37.9% | **+37.8 [+34.8, +40.8]** | **+378 / -0** | **3e-114** |
| **`</think>` x1.5 alone vs matched control** (run 7) | 0.1% -> 34.9% | **+34.8 [+31.9, +37.8]** | **+348 / -0** | **4e-105** |
| **Designed neuron + Wait x0.5 vs base** (kernel pod) | 1.3% -> 42.1% | **+40.8 [+37.7, +43.9]** | **+409 / -1** | **3e-121** |
| — GSM8K-test 500 | 1.8% -> 52.8% | +51.0 [+46.6, +55.4] | +256 / -1 | 2e-75 |
| — MATH-500 | 0.8% -> 31.4% | +30.6 [+26.6, +34.8] | +153 / -0 | 2e-46 |
| Designed neuron vs W5T13 | 33.6% -> 42.1% | +8.5 [+5.7, +11.4] | +150 / -65 | 7e-9 |
| Designed neuron vs W5T15 | 39.1% -> 42.1% | +3.0 [-0.1, +6.1] | +141 / -111 | 0.07 |
| **Designed neuron vs its random-direction control** | 5.9% -> 42.1% | **+36.2 [+33.1, +39.3]** | **+373 / -11** | **3e-95** |
| control (random-direction neuron + Wait x0.5) vs base | 1.3% -> 5.9% | +4.6 [+3.1, +6.2] | +55 / -9 | 4e-9 |

## Portable spec of the designed neuron (v0.71.1)

`specs/conclude_neuron_spec.json` is the designed neuron plus the Wait rows as
one `EditSpec` (`token_rows` x0.5 on ids 13784/13428 and a `neuron_rewrite` of
layer 23 neuron 1998), regenerated with the packaged recipe
(`stateset_agents.experimental.weight_space.design_conditional_neuron`, via
`benchmarks/weight_space/neuron_job.py`) from the same contrast prefixes the
interactive kernel used. `specs/conclude_neuron_control_spec.json` is its
control: the same Wait rows and read vectors with a random write direction of
the same norm. Apply and audit either with:

```bash
stateset-agents weight-edit apply specs/conclude_neuron_spec.json \
  --base Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 --out ./edited
stateset-agents weight-edit verify specs/conclude_neuron_spec.json \
  --base Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 --edited ./edited
```

The spec's content hash is `749a5d716e564da6` on both pods it was built on and
locally after `apply`; `verify` passes on the pinned checkpoint (5,118 weights
changed: 2,048 token-row weights and 3,070 of the neuron's 3,072).

Held-out with the spec applied (fresh pod, 1,000 problems, strict scoring):

| | Spec run (2026-11-03) | Kernel run (2026-11-02) |
|---|---|---|
| base | 0.4% | 1.3% |
| designed neuron + Wait x0.5 | **41.2%** (55.6% lenient) | 42.1% |
| random-write control + Wait x0.5 | 5.2% | 5.9% |
| neuron vs control | **+36.0 [+32.9, +39.1], +367 / -7, p = 1e-98** | +36.2, +373 / -11, p = 3e-95 |
| neuron vs base | +40.8 [+37.7, +43.9], +410 / -2, p = 2e-119 | +40.8, +409 / -1, p = 3e-121 |

The packaged recipe reproduces the kernel's neuron and its effect. Detector
AUC on held-out contexts: 0.90. Base held-out accuracy varies by pod between
0.4% and 1.3% under identical settings. Summary: `summaries/neuron_spec_summary.json`;
per-problem results: `per_problem/spec_*`.

## v3: a conclude neuron that stays out of normal replies (2026-11-03)

v2 fired at line breaks outside thinking too (stray `</think>` in 22% of
thinking-off replies, WikiText perplexity +5.9%). v3 is the same construction
(layer 23 neuron 1998, up unit 44, write 3.0, ridge lam 100) with two more
negative groups: 300 line breaks from thinking-off GSM8K replies and 150 line
breaks from WikiText-2 *train* prose. Spec `conclude_neuron_v3_spec.json`, hash
`7bac7db23edf6b2c`. One L40S pod ($0.83).

Held-out firing rate by contrast group (activation > 0.5, odd-indexed half):

| answer line break (thinking, target) | working line break | after `</think>` | answer token | thinking-off line break | prose line break |
|---|---|---|---|---|---|
| 19.3% (v2 22.1%) | 5.4% | 0% | 0% | **0.7%** | **0%** |

Detector AUC 0.923 (v2 0.900).

| | v3 | v2 (same pod) |
|---|---|---|
| practice, strict | **57.0%** | 49.5-53.0% (earlier pods) |
| held-out thinking on, strict | **45.1%** | 40.3% |
| — lenient / multi-`</think>` / hit limit | 52.7% / 12.1% / 33.5% | 52.8% / 31.6% / 25.4% |
| **v3 vs v2, paired (strict)** | **+4.8 [+2.1, +7.4], +116 / -68, p = 5e-4** | |

Side effects of v3, paired against base on the same pod:

| | Base | v3 | (v2, earlier pod) |
|---|---|---|---|
| GSM8K-test 500, thinking off, strict | 54.0% | 52.0% (-2.0 [-5.2, +1.2], p = 0.27) | 49.2% (-5.6, p = 0.004) |
| — stray `</think>` in replies | 0.0% | **3.4%** | 21.8% |
| MMLU 1,000 | 45.0% | 46.1% (+1.1 [+0.2, +2.0], p = 0.03; uncorrected) | 44.0% |
| WikiText-2 perplexity | 17.378 | 17.430 (**+0.3%**) | 18.400 (+5.9%) |
| p(`</think>`) per WikiText token | 3e-6 | 4.8e-4 | 8.5e-3 |

v3 is better inside thinking (fewer inline `</think>`, more clean finishes) and
mostly confined to it. Residual: 3.4% of thinking-off replies still carry a
stray `</think>`.

### v3 with its own control (second pod, 2026-11-03, $1.40)

Rebuilt from scratch on a new pod: identical spec hash `7bac7db23edf6b2c` (and
control `9f6a5c5a17754e4d`). Held-out, strict, same pod:

| | Accuracy | vs v3, paired |
|---|---|---|
| base | 1.2% | v3 +44.0 [+40.9, +47.1], **+440 / -0**, p = 7e-133 |
| v3 random-write control (+ Wait x0.5) | 4.3% | v3 +40.9 [+37.8, +44.1], **+418 / -9**, p = 7e-111 |
| **v3** | **45.2%** (first pod: 45.1%) | |

Practice strict on this pod: 61.5% (first pod 57.0%; identical weights).

**Caveat.** v3's thinking-off negatives were line breaks from base replies to
GSM8K-test[:500] with thinking off — the same questions as the GSM8K half of
the held-out set and as the thinking-off side-effect check. No labels were
used, but held-out questions shaped the edit, and the 3.4% stray-`</think>`
figure was measured on the replies v3 was trained to stay quiet on. Treat v3's
numbers as optimistic; v3t (below) rebuilds it with negatives from GSM8K-train
only.

## Mechanism

- Two levers: conclusion pressure (`</think>` up) raises how often the model
  concludes; damping "Wait" raises how often a conclusion is correct at equal
  pressure (practice, concluded traces: `</think>` x1.3 66% -> W5T13 83%).
- The model knows when it has the answer: 16-neuron detectors separate
  "answer just computed" from "still working" and "after `</think>`" at AUC
  0.985–0.999 in every layer 16–23 (held-out prefixes).
- Where a conclude signal is written matters more than how strongly the
  answer is detected: a neuron firing on the answer token (AUC 0.99) emits
  `</think>` mid-sentence, which the model ignores (strict 1.5–9.5%); the
  same construction firing at the following line break (AUC 0.85) concludes
  cleanly (practice strict 53.0%, Wait x0.5 alone 6.0%; held-out 42.1%).
- A conditional circuit beats a global boost: the neuron raises `</think>` only
  where the answer is in hand, while W5T13/W5T15 raise it everywhere. It does
  not stack with a global boost (neuron on W5T15: 8.0% strict on practice).

## Negative results

- **Article-style direction ablation**: removing a contrastive "loop"
  direction (layers 15/19, k = 64, alpha 1–4, ~130k weights) left practice at
  1.0–2.0% (base 1.5%).
- **Weight-only atlas neurons** (neurons whose output columns align with
  Wait / Therefore / Yes / check / answer): every neuron-column edit was null
  or harmful alone; only the "answer" neurons helped, +3.5 strict on W5T13 and
  nothing on W5T15.
- **Linear rank-1 circuit** (detector -> `</think>`): cannot threshold; strong
  writes force stops at any digit (0.0% strict), weak ones suppress
  `</think>` while working (0.5%).
- **Number-anchored designed neuron**: 24–36% lenient, 1.5–9.5% strict.
- **Stacking**: the designed neuron on top of W5T15 collapsed (8.0% strict).
- **Stronger global pressure**: `</think>` x1.7 / x2.0 score 9.5% / 0.0%
  strict (inline and post-answer `</think>` spam).

## Side effects

The loop-direction edit on a healthy model (Qwen3-0.6B,
`c1899de289a04d12100db370d81485cdf75e47ca`, GSM8K-test 500): 79.4% -> 79.8%
and, repeated, 79.2% -> 79.8%; loop rate unchanged.

Outside thinking-mode math (2026-11-03, `summaries/side_effects_summary.json`,
paired against base):

| | Base | W5T13 | W5T15 | Designed neuron |
|---|---|---|---|---|
| GSM8K-test 500, thinking **off**, strict | 54.8% | 51.2% (p = 0.06) | 50.0% (p = 0.013) | 49.2% (p = 0.004) |
| — replies with a stray `</think>` | 0.0% | 0.4% | 6.4% | **21.8%** |
| MMLU 1,000, zero-shot | 45.0% | 45.6% | 45.4% | 44.0% (p = 0.15) |
| WikiText-2 perplexity | 17.378 | +0.01% | +0.05% | **+5.9%** |

No edit measurably changes knowledge. The token-row edits leave general text
untouched; x1.5 leaks a stray `</think>` into 6% of thinking-off replies. The
designed neuron is not confined to thinking: it also fires at line breaks in
thinking-off answers and prose (22% of replies; perplexity +5.9%). With
thinking off the unedited model already scores 54.8% on GSM8K-test (1.8% with
thinking on), so the edits repair thinking mode rather than exceed the model's
non-thinking accuracy.

## What this does not show

- Not a general capability gain: the edits fix one failure (non-termination
  under thinking at an 8,192-token cap) on one 0.8B model. A larger budget
  (e.g. 32k) or a model that already terminates may gain nothing.
- No comparison with fine-tuning or with decoding-time fixes (a logit bias
  on the same tokens is likely equivalent for the token-row edits).
- Single decoding seed per configuration; launch-to-launch noise for an
  identical checkpoint is about ±1 point.
- Edits were designed by Claude Opus 5.5 working interactively with full
  context; the blind-programmer arm (an independent agent given only the
  harness) was not run.

## Runs and costs

| Run | What | GPU | Cost (USD) |
|---|---|---|---|
| 2–5b | diagnosis, loop-direction edits, Wait sweep, first held-out | L40S / H100 | 10.38 |
| 6 | token-edit sweep, W5T13 held-out + control, Qwen3-0.6B side effects | L40S | 2.28 |
| 7 | neuron-column edits, `</think>` x1.5 held-out + control | L40S | 2.56 |
| 8 | atlas neurons on W5T13/W5T15, W5T15 held-out + control | L40S | 2.55 |
| 9 | conclusion sweep, W5T15 replication | L40S | 2.05 |
| kernel | interactive circuit/neuron design, neuron held-out + control | L40S | 2.53 |
| spec | portable-spec regeneration + held-out (one failed start, $0.17) | L40S | 1.55 |
| side effects | thinking-off GSM8K, MMLU, WikiText for base and three edits | L40S | 0.40 |
| v3 | thinking-gated neuron: design, held-out vs v2, side effects | L40S | 0.83 |
| v3 control | v3 rebuilt, held-out vs base and its random-write control | L40S | 1.40 |
| **total** | all pods terminated | | **26.53** |

Incidents: run 5b's per-problem files were lost when the launcher crashed
decoding a byte-cut UTF-8 log tail (aggregates survive in its event log;
fixed with ASCII-only polling and a 5-minute mirror). On runs 7–9 the vLLM
EngineCore process hung after a 1,000-prompt generation, idling each pod
~30 minutes until it was terminated by hand. A first fix (hard exit once
results are written) orphaned EngineCore with its GPU memory, which made the
next vLLM start fail: it cost runs 7 and 9 their optional Qwen3-0.6B
side-effect repeat and the kernel's first held-out attempt, which was then
completed from the pod shell (`finish_heldout.py`; see
`summaries/kernel_summary.json`). Final fix: kill EngineCore, then exit.

## Files

- `EDIT_LEDGER.md` — every edit tried, practice and held-out, lenient and strict.
- `per_problem/` — compact per-problem held-out results (`id`, `kind`,
  `correct` (lenient), `correct_strict`, `finish`, `think_close`, `tokens`; no
  generated text) and `SOURCES.json` (SHA-256 of each raw generation file).
  `stateset-agents weight-edit compare BASE.jsonl OTHER.jsonl` reproduces the
  lenient paired statistics directly; for strict, compare copies with
  `correct` replaced by `correct_strict`, e.g.
  `jq -c '.correct = .correct_strict' run6_base_heldout.jsonl > base_strict.jsonl`.
- `summaries/` — each run's `summary.json` (configuration, practice gate,
  kept edit, held-out aggregates) and `kernel_summary.json` (the interactive
  session: submitted neuron edits, all seven practice evaluations lenient and
  strict, held-out base / neuron / control, and how the held-out run was
  completed after the evaluator's thread failed).
- `specs/conclude_neuron_v3_spec.json` / `conclude_neuron_v3_control_spec.json`
  — **v3**, the recommended designed neuron (thinking-gated), and its control.
- `specs/conclude_neuron_spec.json` / `conclude_neuron_control_spec.json` —
  the v2 designed neuron and its control as portable `EditSpec`s (the kernel
  session's own weights were not retained; these were regenerated with the
  packaged recipe and re-validated, see "Portable spec" above).
- `specs/token_spec*.json` — the token-row / neuron-column edit specs each run
  evaluated.
- Code: `benchmarks/weight_space/` (pod job, launcher, interactive kernel and
  the kernel programs under `kernel/`), `stateset_agents.experimental.weight_space`.
