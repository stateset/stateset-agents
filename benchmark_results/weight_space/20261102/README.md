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
| Designed neuron v3 (thinking-gated; negatives touched held-out questions, see caveat) + "Wait" rows x0.5 | 3 x 1,024 + 2,048 | 45.1% | +418 / -9 (45.2% on a second pod) |
| **Designed neuron v3t** (thinking-gated, clean negatives; recommended) + "Wait" rows x0.5 | 3 x 1,024 + 2,048 | **45.9%** | +398 / -9 |

The designed neuron is a single MLP unit rewritten by hand into an "answer
computed -> conclude" circuit that fires at the line break after the model has
computed its answer. v2 beat both global edits (vs W5T13 +8.5 points,
p = 7e-9; vs W5T15 +3.0, p = 0.07) but was not confined to thinking: it put a
stray `</think>` into 22% of thinking-off replies and raised general-text
perplexity 5.9%. v3t adds thinking-off line breaks (from GSM8K-train replies)
and prose line breaks as negatives: 45.9% held-out (+450 / -1 vs base,
+398 / -9 vs its control), and outside thinking a stray `</think>` in 6.8% of
thinking-off GSM8K replies (but 28.8% of thinking-off MATH-500 replies at an
8,192-token cap; see "v5") (thinking-off GSM8K -2.6 vs base, p = 0.17), perplexity
+0.3%, MMLU unchanged or slightly up. (v3, the same idea with negatives drawn
from held-out questions, reported an optimistic 3.4%.) W5T13 remains the
cleanest edit outside thinking (0.4%; see Side effects).

The bar these edits do **not** clear: the unedited model with thinking
switched off scores 47.9% strict at the same 8,192-token cap (GSM8K 55.6%,
MATH 40.2%). v3t with thinking on ties it on GSM8K (57.0%, p = 0.59), trails
on MATH (33.8%, p = 0.006), and leads only under lenient scoring (+5.5 overall,
p = 0.0009): it reasons better but a quarter of its replies are malformed. See
"Thinking on vs thinking off at an equal budget". Redesigns of the neuron
(v4, v5, on-policy rounds) and an explicit firing threshold all failed to beat
v3t on held-out; see their sections. A designed attention head that silences
the neuron once `</think>` is in context removes its thinking-off side effect
(-1.1 and -1.4 vs base on two pods, p = 0.53 and 0.41; was -3.9, p = 0.001)
and, in v2, edges thinking on past v3t (+2.3, p = 0.14), still short of the
base with thinking off (-2.8, p = 0.093); see "A `</think>` already seen
guard".

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

## v3t: the thinking-gated neuron with clean negatives (recommended; 2026-11-03)

Same construction as v3, but the thinking-off negatives are line breaks from
base replies to **GSM8K-train** (the trace split, generated on the pod), so no
held-out question shapes the edit. Prose negatives: WikiText-2 train. Spec
`conclude_neuron_v3t_spec.json`, hash `f221470a088d0860`. One L40S pod ($1.42).

Held-out firing by group: target (answer line break in thinking) 20.0%,
working 4.1%, after `</think>` 0%, answer token 0%, thinking-off line break
(train replies) 0.7%, prose 0%. Detector AUC 0.882.

| Held-out (1,000, strict), same pod | Accuracy | v3t vs it, paired |
|---|---|---|
| base | 1.0% | **+44.9 [+41.8, +48.0], +450 / -1, p = 2e-133** |
| v3t random-write control (+ Wait x0.5) | 7.0% | **+38.9 [+35.8, +42.0], +398 / -9, p = 5e-105** |
| **v3t** | **45.9%** (lenient 53.7%, multi-`</think>` 16.1%) | |

Side effects (paired against base, same pod; thinking-off check on held-out
questions v3t never saw):

| | Base | v3t |
|---|---|---|
| GSM8K-test 500, thinking off, strict | 54.0% | 51.4% (-2.6 [-6.0, +1.0], p = 0.17) |
| — stray `</think>` in replies | 0.0% | **6.8%** |
| MMLU 1,000 | 45.0% | 46.3% (+1.3 [+0.5, +2.2], p = 0.004)* |
| WikiText-2 perplexity | 17.378 | 17.431 (+0.3%) |

\* v3 showed the same small MMLU rise (+1.1, p = 0.03) on another pod. Reported
as observed; this study was not designed to test it.

The clean rebuild confirms the thinking-mode result (v3t 45.9% vs v3 45.1-45.2%)
and shows v3's stray-`</think>` figure was optimistic: the true thinking-off
leak is 6.8% (v2 21.8%, W5T15 6.4%, W5T13 0.4%).

## Thinking on vs thinking off at an equal budget (2026-11-03)

The fair bar for a thinking-mode repair is the same model with thinking
switched off. One A100 pod ($1.17,
`summaries/thinking_budget_summary_v3t_v4.json`,
`per_problem/tb_tb_*.jsonl`) generated the held-out 1,000 under every
condition with the same sampling, seed and 8,192-token cap, and scored each
strictly (thinking on: one `</think>` and a clean stop; thinking off: none).
v3t with thinking on reproduces across GPUs (45.4% here, 45.9% on L40S).

| Strict (lenient) | Base, thinking off | v3t, thinking on | v3t vs base-off, paired strict |
|---|---|---|---|
| GSM8K-test 500 | 55.6% (57.8%) | 57.0% (65.2%) | +1.4 [-3.2, +6.0], +67 / -60, p = 0.59 |
| MATH-500 | 40.2% (42.0%) | 33.8% (45.6%) | **-6.4 [-10.8, -2.0], +48 / -80, p = 0.006** |
| all 1,000 | 47.9% (49.9%) | 45.4% (55.4%) | -2.5 [-5.6, +0.6], +115 / -140, p = 0.13 |

Lenient (grade the text after the first `</think>`), v3t with thinking on is
ahead: +5.5 [+2.3, +8.7] overall (p = 0.0009), +7.4 on GSM8K (p = 0.002).
So the edited model reasons better with thinking on than the base does with
it off, but a quarter of its replies are malformed (early close, then more
reasoning and a second `</think>`, or the cap), and strict scoring counts
those as wrong. **Strictly, the edits repair thinking mode to parity with
thinking off on GSM8K and short of it on MATH; they do not exceed it.** An
earlier comparison against a 2,048-token thinking-off run (54.0%) overstated
v3t's GSM8K lead; at equal caps thinking-off reaches 55.6%. With one budget
curve per run (`protocol.budget_curve`, valid because each reply is a fixed
function of its prompt under a per-request seed), thinking-off leads at 512
and 1,024 tokens on both subsets; from 2,048 tokens v3t leads on GSM8K
(56.4% vs 54.8% at 2,048) and trails on MATH at every budget.

Failure profile of v3t with thinking on (same pod): GSM8K 285 clean right,
78 clean wrong, 67 two or more `</think>`, 39 closed then ran to the cap,
31 never closed; MATH 169 / 25 / 105 / 75 / 126. In 34.5% of GSM8K replies
that closed, the gold answer had not yet appeared in the thinking.

## v4: hard negatives after intermediate results (negative result, 2026-11-03)

Diagnosis: v3t's neuron fires at 12.7% of line breaks right after an
intermediate result ("Total = 126 x 6 = 756"), three times its rate at other
working line breaks (4.1%), measured by scoring the stored v3t rewrite on
captured base activations (`circuits.rewrite_activation`). v4 adds 300 such
line breaks (base GSM8K-train traces, before the gold answer is first
written) as negatives. Offline (design-only pod, $1.22,
`summaries/neuron_v4_design_summary.json`) it cut that rate to 5.3% at a
cost of target recall (20.0% -> 14.5%).

In generation it was worse on every axis (same pod as above):

| | v3t | v4 | v4 vs v3t, paired strict |
|---|---|---|---|
| thinking on, strict | 45.4% | 40.7% | -4.7 [-7.4, -2.0], +69 / -116, p = 0.0007 |
| — two or more `</think>` (GSM8K / MATH) | 67 / 105 | 98 / 179 | |
| — GSM8K closes before the gold appeared | 34.5% | 34.7% | |
| thinking off, strict | 45.2% | 42.0% | |

It did not reduce early closes; it doubled second closes. The neuron fires
again inside answer sections where the model keeps reasoning after an early
close, contexts the `post` negatives (clean, formatted answers) never covered.
Offline fire rates on curated contexts did not predict behaviour in
generation. Spec kept as evidence: `specs/conclude_neuron_v4_spec.json`
(`7664695121410597`).

## v5: MATH positives and post-close negatives (negative result, 2026-11-03)

v3t never closes thinking on a quarter of MATH-500 (its positives came from
GSM8K only), and it closes a second time inside run-on answers. v5 adds
(design-only H100 pod, $1.21, `summaries/neuron_v5_design_summary.json`):

- **MATH positives**: base thinking-on traces on 600 MATH-*train* problems
  (`DigitalLearningGmbH/MATH-lighteval` @ `0530c786`; 0 of 6,798 usable
  problems match a MATH-500 question; single-character answers skipped).
  177 traces write the gold as a result in thinking (`answer_positions.py`
  matches `\dfrac` / `\frac` / `a/b`, `\boxed{}`, `= ...`, `is ...`): 354
  positive and 354 working line breaks.
- **Post-close negatives**: 151 line breaks after the first `</think>` in v3t's
  run-on replies to the GSM8K-*train* practice problems, starting with the line
  break before each extra `</think>`.
- the v4 intermediate-result negatives for GSM8K and MATH (300 each).

Offline it looked better than v3t on every group. One generation pod (RTX
4090, $1.06, `summaries/thinking_budget_summary_v5.json`,
`per_problem/tb5_tb_*`) tested v5 and v5b (the same contexts, fit on all of
them, see the next section) against v3t and base thinking off:

| Same pod, strict (lenient) | GSM8K 500 | MATH-500 | all | two+ `</think>` | never closed | median line breaks before first close |
|---|---|---|---|---|---|---|
| base, thinking off | 55.0% (55.8%) | 40.8% (42.8%) | 47.9% | | | |
| v3t, thinking on | 57.0% (62.8%) | 35.6% (48.2%) | **46.3%** | 186 | 155 | 41 |
| v5, thinking on | 43.2% (60.4%) | 10.6% (42.4%) | 26.9% | **622** | 21 | 23 |
| v5b, thinking on | 41.0% (53.2%) | 15.0% (36.6%) | 28.0% | 535 | 13 | **0** |

v5 vs v3t, paired strict: -19.4, +37 / -231, p = 2e-35 (v5b: -18.3, p = 6e-29).
The MATH positives did what they were for (never-closed 155 -> 21), but the
broadened detector fires at most line breaks of the edited model's own
generations: it closes early, closes again, and (v5b) closes at the first line
break. With thinking off, v5 leaves a stray `</think>` in 379 of 500 MATH
replies. Base and v3t reproduce across pods (base 47.9% on both; v3t 45.9%,
45.4%, 46.3% on L40S, A100 and RTX 4090).

**Correction to v3t's side effects.** On this pod, with thinking off and an
8,192-token cap, v3t leaves a stray `</think>` in 46 of 500 GSM8K replies
(9.2%) and **144 of 500 MATH-500 replies (28.8%)**. The 6.8% on record was
GSM8K only, at a 2,048-token cap.

## Practice-gated, on-policy redesign (negative result, 2026-11-03)

`onpolicy_job.py` (A40, $1.10, `summaries/onpolicy_summary.json`) judged
every design by generation on training problems and mined contexts from the
edited model's own replies (`misfires.py`: line breaks before extra closes,
before early closes in replies that then closed again or ran to the cap, and
before stray closes with thinking off as negatives; before clean correct
closes and after unclosed answers as positives). Each round refit on v3t's
contexts plus everything mined so far. Practice: MINE and SELECT, each 100
GSM8K-train + 100 MATH-train problems, cap 4,096; held-out never used.

| Round | SELECT thinking on, strict (lenient) | two+ `</think>` | never closed | thinking off, strict | stray `</think>` | gate cosine to v3t |
|---|---|---|---|---|---|---|
| 0 (v3t) | **44.5%** (52.0%) | 15.0% | 20.0% | 44.0% | 15.0% | 1.00 |
| 1 | 35.5% (45.0%) | 17.5% | 30.0% | 48.5% | 10.0% | 0.64 |
| 2 | 31.5% (45.5%) | 25.0% | 27.0% | 42.5% | 21.5% | 0.59 |
| 3 | 30.0% (44.5%) | 28.0% | 22.5% | 38.0% | 31.0% | 0.52 |

The selection rule kept round 0 (v3t), so no held-out run was spent. The
first refit, with 134 mined contexts added to about 1,300, already turned the
detector to cosine 0.64 from v3t's and enlarged it (gate norm 1.94 -> 2.43).

**Why designs swing.** The detector is a ridge fit of 3,584 features on about
800 contexts at `lam` = 100. Fit on random halves of v3t's own contexts, the
two detectors' scores correlate 0.909 (0.971 at `lam` = 1,000, 0.988 at
10,000, held-out AUC 0.98 throughout). The neuron has no bias, so where it
starts firing is set by how much the fit shrinks the score, which moves
whenever the contexts change; in generation that threshold sits where small
shifts matter. v3t is one draw from this procedure, and v4, v5 and these
rounds were partly variance, not only their extra data. A stable redesign
needs a stronger prior (larger `lam`) and an explicit threshold (an offset
read through a near-constant unit, the way `up_proj` reads unit 44),
calibrated by practice generation with the detector held fixed. The explicit
threshold was tested next; it did not help on held-out (below).

## Explicit threshold on v3t (negative result, 2026-11-03)

`circuits.shift_threshold` gives the neuron an explicit threshold with v3t's
detector held fixed: gate row minus `bias` times its own up row. The up row
reads about 1.0 in every context (mean 1.006, std 0.108 over 2,790 captured
contexts), so the pre-activation drops by `bias`. The down column is rescaled
so the mean write when firing is `write_norm`; bias 0 at write norm 3 rebuilds
v3t's hash exactly. `threshold_design.py` built the grid offline;
`threshold_job.py` (RTX 4090, $1.35, `summaries/th_threshold_summary.json`)
scored each candidate by generation on 400 practice problems (200 GSM8K-train
+ 200 MATH-train, cap 4,096), kept the best strict score whose thinking-off
stray rate rose at most 2 points, then ran held-out.

| bias, write norm | practice, thinking on strict | two+ `</think>` | never closed | thinking off strict | stray `</think>` |
|---|---|---|---|---|---|
| 0, 3 (v3t) | 48.5% | 14.0% | 17.0% | 43.0% | 18.3% |
| -1, 3 | 48.3% | 21.8% | 12.8% | 43.8% | 22.0% |
| **+1, 3 (kept)** | **51.3%** | 9.3% | 18.8% | 45.5% | 13.8% |
| +2, 3 | 45.0% | 9.8% | 22.5% | 45.0% | 12.3% |
| 0, 2 | 47.0% | 4.0% | 31.0% | 45.8% | 5.3% |
| 0, 4 | 45.8% | 24.3% | 9.8% | 44.0% | 24.3% |

The threshold behaves as designed: lower fires more (fewer unclosed replies,
more double closes and leaks), higher the reverse, and a weaker write nearly
removes the leaks (5.3%) but leaves 31% unclosed. On held-out the kept
candidate did not beat v3t:

| Held-out 1,000, cap 8,192, one pod | strict | GSM8K | MATH | hit the cap |
|---|---|---|---|---|
| base, thinking off | 48.5% | 56.6% | 40.4% | 20.5% |
| v3t, thinking on | 46.3% | 57.0% | 35.6% | 30.0% |
| bias +1, thinking on | 45.6% | 56.0% | 35.2% | 35.7% |
| bias +1, thinking off | 44.6% | 53.0% | 36.2% | 8.9% |

Bias +1 vs v3t: -0.7 [-3.0, +1.7], +70 / -77, p = 0.62. Its practice gain
(+2.8 points, best of seven on 400 problems) was selection noise, and at the
held-out 8,192-token cap the higher threshold left more replies running to
the cap. Bias +1 with thinking off is 3.9 points below the base with thinking
off (+51 / -90, p = 0.001): the neuron's leak into thinking-off mode costs
real accuracy. One knob trades unclosed replies against double closes and
leaks; v3t sits near its best setting.

## A "</think> already seen" guard (2026-11-03)

The neuron also fires after reasoning has closed: a second `</think>` fails a
thinking-on reply under strict scoring, and with thinking off, whose template
already ends `<think>\n\n</think>\n\n`, it writes stray closes (the base
model leaks none). One rule covers both: once `</think>` is in context, the
neuron must stay silent. No existing head looks back at an earlier
`</think>` (at most 7% of any full-attention head's weight from later line
breaks), so one was built, with no fitting (`guard_design.py`): every
quantity is a mean over positions labelled exactly by the token sequence.

- **Head.** Layer 23, head 0 (`HeadRewrite`). At layer 23 the keys at
  `</think>` score at least 17.2 along their mean direction (position-free
  dims) against at most 12.7 at 7,430 other positions, so a constant query
  along it (`circuits.lookback_query`) puts at least 99.3% of the head's
  attention on an earlier `</think>`. Heads 1 and 3 of the same KV group hold
  `</think>` back at line breaks (removing them releases closes) and head 2
  promotes it; head 0 is the most neutral.
- **Flag.** The head writes along a residual direction that the unembedding
  barely sees and the model's residual barely uses (`atlas.quiet_direction`;
  logit visibility 0.07 of a `</think>` write). The neuron reads it
  (`circuits.silence_on_flag`): its near-constant multiplier drops by 1 and its
  gate by 6.3 once the flag is up.
- **Why not push the logit.** A first version wrote `-</think>` from the head
  (with `atlas.protected_write`, which keeps `<|im_end|>` and other overlapping
  tokens' logits exactly unchanged). The final residual is small (rms 0.3), so
  strengths that blocked second closes also blocked every reply's end and
  raised ordinary-text loss tenfold on CPU.

Practice (400 problems, cap 4,096; RTX 4090, $0.95 with held-out,
`summaries/gd_practice_summary.json`):

| candidate | thinking on strict | two+ `</think>` | never closed | thinking off strict | stray `</think>` |
|---|---|---|---|---|---|
| v3t | 49.3% | 15.3% | 16.5% | 43.0% | 18.3% |
| v3t + guard | 47.3% | **0.8%** | 24.3% | **51.0%** | **0.0%** |
| **bias -1 + guard** (kept) | **50.3%** | 1.5% | 18.3% | **51.8%** | 0.3% |
| write norm 4 + guard | 49.0% | 2.0% | 19.3% | 50.3% | 0.0% |

Held-out 1,000, cap 8,192, same pod:

| | strict | GSM8K | MATH | hit the cap |
|---|---|---|---|---|
| base, thinking off | 48.5% | 56.6% | 40.4% | 20.5% |
| v3t, thinking on | 46.3% | 57.0% | 35.6% | 30.0% |
| bias -1 + guard, thinking on | 46.1% | 58.4% | 33.8% | 40.2% |
| bias -1 + guard, thinking off | 47.4% | 52.4% | 42.4% | 12.8% |

- Thinking off, guard vs base: **-1.1 [-4.3, +2.1], +124 / -135, p = 0.53**.
  Unguarded, the neuron cost thinking-off accuracy 3.9 points (p = 0.001);
  the guard removes that side effect.
- Thinking on, guard vs v3t: -0.2 [-3.1, +2.7], p = 0.95. Double closes are
  gone, but more replies run to the cap without closing (40.2% vs 30.0%;
  lenient MATH -7.4, p = 0.0005), so thinking on still trails base thinking
  off on MATH (-6.6, p = 0.004) while matching it on GSM8K (+1.8, p = 0.53).

Why more replies stay open: before any close the head attends diffusely
(15-24% on `<think>`, the rest on prompt words), so its value read leaked a
flag of 0.03-0.09 at the neuron's decision points, which sit just above
threshold; on CPU the guard alone cut `</think>` probability at v3t's own
first closes from 0.93 to 0.03 (bias -1 restored it to 0.67). Neither half
alone does it (head 0 removed: 0.92; neuron rewired without the flag: 0.91;
flag written but unread: 0.93). Guard v2 reads orthogonally to the top 32
directions of the before outputs and probes the edited model: the leak's p99
falls from 0.039 to 0.007.

**Guard v2** (A40, $1.04, practice at the full 8,192 cap,
`summaries/gd2_practice_summary.json`). Strict outcome of every thinking-on
practice reply:

| candidate | strict correct | never closed | two+ `</think>` | closed once, ran to the cap | clean but wrong | thinking off strict (stray) |
|---|---|---|---|---|---|---|
| v3t | 46.2% | 17.0% | 18.8% | 9.5% | 8.5% | 45.3% (20.5%) |
| **v2 bias -1 + guard** (kept) | **47.0%** | 16.5% | 4.2% | 23.0% | 9.2% | **53.0% (0.0%)** |
| v2 bias -1.5 + guard | 46.5% | 10.8% | 7.2% | 27.5% | 8.0% | 52.0% (0.3%) |
| v2 bias -2 + guard | 44.5% | 7.5% | 12.0% | 28.5% | 7.5% | 53.5% (0.5%) |

The double closes the guard removes reappear as replies that close once and
run to the cap. Read, they are premature first closes: the neuron closed
after an intermediate result ("22 + 8 = 30." for an answer of 13), sometimes
mid-sentence, and the model re-reasoned in the answer section. Under v3t the
neuron then closed again; with the guard the reply runs on. Both fail
strictly, so on thinking on the guard neither loses nor gains these; what
bounds thinking on now is where the detector first fires, not what happens
after.

Held-out, same A40 (absolute rates differ a little by GPU type; all
comparisons are paired on one pod):

| | strict | GSM8K | MATH | hit the cap |
|---|---|---|---|---|
| base, thinking off | 49.8% | 57.4% | 42.2% | 19.7% |
| v3t, thinking on | 44.7% | 55.0% | 34.4% | 31.9% |
| v2 guard, thinking on | 47.0% | 58.2% | 35.8% | 39.9% |
| v2 guard, thinking off | 48.4% | 53.8% | 43.0% | 14.0% |

- Thinking on, v2 guard vs v3t: **+2.3 [-0.7, +5.2], +122 / -99, p = 0.14**
  (v1: -0.2).
- Thinking on vs base thinking off: -2.8, p = 0.093 (GSM8K +0.8, p = 0.79;
  MATH -6.4, p = 0.006).
- Thinking off vs base thinking off: -1.4 [-4.4, +1.7], p = 0.41. Two
  independent pods agree the guard removes the neuron's thinking-off cost
  (v1 -1.1, p = 0.53; unguarded -3.9, p = 0.001).

**Gating the first close (offline negatives, no GPU).** Over the 1,467
thinking-on practice replies above, no phrase in the thinking marks a good
first close well enough to gate on: "Final Answer" precedes 49% of good first
closes and 21% of premature ones, `\boxed` 22% and 11%. The correct value
after "=" precedes 68% of good and 18% of premature closes, so premature
closes mostly come before the model has the answer at all. The model's own
continuation belief (logit lens at layer 23 for "Wait", "But", "Let" and
similar) is *higher* at final-answer line breaks than at intermediate ones
(AUC 0.76 the wrong way round for a stop signal: this model re-verifies after
answering), and adding it to v3t's detector moves the detector's AUC from
0.822 to 0.830 (GSM8K) and 0.743 to 0.760 (MATH) at a weight chosen on the
same contexts. Neither was worth a generation run.

## Qwen3.5-2B baseline (2026-11-04)

Does the larger model share the failure? `thinking_budget_job.py --model
Qwen/Qwen3.5-2B --base-think` (revision `15852e8c`, A40, $0.51,
`summaries/q2b_thinking_budget_summary_2b.json`, `per_problem/q2b_*`). Same
family and tokenizer as 0.8B (24 layers, full attention at 3/7/11/15/19/23,
8 query heads over 2 KV heads of 256, tied embedding), hidden 2,048, MLP
6,144.

| Qwen3.5-2B, held-out 1,000, cap 8,192 | strict | lenient | clean finish | hit the cap |
|---|---|---|---|---|
| thinking off | **77.1%** | 79.6% | 87.8% | 12.0% |
| thinking on | 25.0% | 25.1% | 25.2% | 74.8% |

Thinking on vs off: -52.1 [-55.2, -48.9], +5 / -526, p = 1e-148. The 2B
model has the 0.8B's broken thinking, in a cleaner form: when it finishes it
is right (25.0 of 25.2), and of the replies that run to the cap without
closing, the correct value is already written in the thinking in 87% on GSM8K
(312 / 359) and 67% on MATH (258 / 383), at a median 7% and 11% of the way in.
Closing there and answering would give about 90% and 74%, a ceiling near 82%
against 77.1% with thinking off (approximate: a gold match can be
coincidental, and the first mention is not always the final answer). Unlike
the 0.8B, whose closes were often premature, the 2B finds the answer early
and does not stop.

## Design diagnostics: what the held-out fire rates measured

`design_conditional_neuron` fits on even-indexed contexts and reports on odd
ones. Every contrast builder adds a trace's two positives (the first and
second line break after the answer) at consecutive indices, so the fit only
ever saw the *first* line break and the "held-out target fire rate" measured
only the *second*. That is why v3t's target rate reads 20%: refit on
trace-disjoint halves (`offline_design.py --split halves`), detectors built
from v3t's groups fire at 87% of held-out answer line breaks. The rates
reported for v3t, v4 and v5 are internally consistent (an offline refit
reproduces each pod's numbers to four decimals) but measure that one context.
More important, none of the offline rates predicted generation: v4 and v5
improved them and lost 5 and 19 points. Contexts curated from base traces do
not cover the edited model's own trajectories, which is where the neuron
misfires. Designs should be judged by generation on practice problems, and
negatives mined from the edited model's own replies to them.

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
- **v4, intermediate-result negatives**: 40.7% strict vs v3t 45.4% (paired
  -4.7, p = 0.0007); twice the second-`</think>` replies (see "v4").
- **v5 / v5b, MATH positives + post-close negatives**: 26.9% / 28.0% strict vs
  v3t 46.3% (-19.4, p = 2e-35); fires at most line breaks in generation
  (see "v5").
- **On-policy, practice-gated refits**: three rounds scored 35.5%, 31.5%,
  30.0% on practice vs v3t 44.5%; v3t kept (see "Practice-gated").

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
| v3t | clean-negative rebuild: held-out vs base and control, side effects | L40S | 1.42 |
| thinking budget | base thinking off vs v3t / v4 thinking on and off | A100 | 1.17 |
| v4 design | intermediate-result negatives, design only (one start lost its SSH, $0.41) | H100 | 1.63 |
| v5 design | MATH positives + post-close negatives, captures saved (one superseded start, $0.10) | H100 / A100 | 1.31 |
| v5 test | base thinking off vs v3t / v5 / v5b thinking on and off | RTX 4090 | 1.06 |
| on-policy | practice-gated redesign, 3 rounds (kept v3t; no held-out run) | A40 | 1.10 |
| threshold | explicit-threshold grid on v3t (7 on practice), bias +1 held-out | RTX 4090 | 1.35 |
| guard v1 | `</think>`-seen guard: 4 on practice, bias -1 + guard held-out | RTX 4090 | 0.95 |
| guard v2 | v2 guard at bias -1 / -1.5 / -2 on practice (cap 8,192), bias -1 held-out | A40 | 1.04 |
| Qwen3.5-2B baseline | base thinking off vs on, held-out | A40 | 0.51 |
| failed starts | dependency installs hung on one H100 host (26 min) and one RTX 4090 host (15 min) | H100 / RTX 4090 | 2.32 |
| **total** | all pods terminated | | **40.39** |

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
On 2026-11-03 one H100 host never finished `pip install` (26 idle, billed
minutes; installs now time out after 15 minutes), and RunPod reported "no
instances currently available" for L40S, L40, A40 and RTX 4090 at times as a
bare HTTP 500 (the client now says "no capacity").

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
  — v3 (its thinking-off negatives touched held-out questions; superseded).
- `specs/conclude_neuron_v3t_spec.json` / `conclude_neuron_v3t_control_spec.json`
  — **v3t**, the recommended designed neuron (thinking-gated, clean negatives),
  and its control.
- `specs/conclude_neuron_v4_spec.json` / `conclude_neuron_v4_control_spec.json`
  — v4 (intermediate-result negatives; negative result, kept as evidence).
- `specs/threshold_grid/` — the explicit-threshold grid on v3t
  (`threshold_grid.json` plus one spec per bias and write norm), and
  `summaries/th_threshold_summary.json` with `per_problem/th_th_*` (practice
  generations on training problems and the held-out runs).
- `specs/guard_v1/` — the `</think>`-seen guard on v3t, bias -1 and write norm
  4 (each with a control whose flag is written in a random direction) and
  `guard_design.json` (attention and read diagnostics);
  `summaries/gd_practice_summary.json` with `per_problem/gd_pr_*`.
- `specs/guard_v2/` — guard v2 on v3t and bias -1 / -1.5 / -2 (with controls)
  and its `guard_design.json`; `summaries/gd2_practice_summary.json` with
  `per_problem/gd2_pr_*`.
- `summaries/thinking_budget_summary_v3t_v4.json` and `per_problem/tb_tb_*` —
  base thinking off vs v3t / v4 thinking on and off, one pod, equal caps,
  strict and lenient, per subset and per token budget, paired.
- `summaries/neuron_v4_design_summary.json` — v4's design report, with v3t's
  stored neuron scored on the same contexts.
- `specs/conclude_neuron_v5_spec.json` (pod design, `76a7e4686dce6348`) and
  `specs/conclude_neuron_v5b_spec.json` (offline, fit on all contexts,
  `a9929a6009bafbcd`), each with its control; `summaries/neuron_v5_design_summary.json`,
  `summaries/neuron_v5b_offline_design.json` (in-sample rates);
  `summaries/thinking_budget_summary_v5.json` and `per_problem/tb5_tb_*` —
  v5 / v5b / v3t / base thinking off on one pod. Negative results.
- `summaries/onpolicy_summary.json`, `specs/onpolicy_r{1,2,3}_spec.json` — the
  practice-gated, on-policy rounds (round 0 is v3t). Negative result.
- `specs/conclude_neuron_spec.json` / `conclude_neuron_control_spec.json` —
  the v2 designed neuron and its control as portable `EditSpec`s (the kernel
  session's own weights were not retained; these were regenerated with the
  packaged recipe and re-validated, see "Portable spec" above).
- `specs/token_spec*.json` — the token-row / neuron-column edit specs each run
  evaluated.
- Code: `benchmarks/weight_space/` (pod job, launcher, interactive kernel and
  the kernel programs under `kernel/`), `stateset_agents.experimental.weight_space`.
