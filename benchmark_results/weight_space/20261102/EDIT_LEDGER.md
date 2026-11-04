# Weight-edit ledger — Qwen3.5-0.8B, thinking on (2026-11-02)

Every edit is a direct write to the checkpoint (no training, no gradients),
verified by reading the changed tensors back (`verify_spec.py`), served with
vLLM 0.33. Decoding: temperature 0.6, top_p 0.95, top_k 20, seed 1234, 8,192
new tokens. Practice = 200 GSM8K-train problems; held-out = GSM8K-test[:500] +
MATH-500. Same-hardware practice baseline (L40S): **1.5%** (H100: 1.5%).
Launch-to-launch noise for an identical checkpoint: about ±1 point (6% vs 8%
observed); identical outputs often reproduce byte-for-byte across pods (base
held-out on runs 7 and 9: identical 8,120,172 generated tokens).

## Scoring: lenient vs strict

The harness grade takes the text after the **first** `</think>` as the answer.
Edits that raise the `</think>` logit can emit it inline ("J = 12</think>")
while the model keeps reasoning; the lenient grade then credits whatever number
the continued reasoning ends on. **Strict** = lenient AND the generation
finished (`stop`) AND `</think>` occurs exactly once. Strict is the primary
measure below; every claim is stated under strict scoring. Up to x1.3 the two
agree within 1 point; from x1.5 they diverge, and from x1.7 lenient is mostly
artifact.

## Practice results (200 GSM8K-train problems)

| Edit | Weights touched | Lenient | **Strict** | Multi-`</think>` | Hit limit | Run |
|---|---|---|---|---|---|---|
| base | 0 | 1.5% | **1.5%** | 0% | 98% | 3, 6, 7 |
| loop direction L15/19 k64 a=1/2/4 | ~130k | 1.0–2.0% | — | — | 97.5–99% | 3 |
| Wait x0.3 / x0.5 / x0.7 (saturate) | 2,048 | 6–8% | **6–8%** | 0% | 92–93% | 6 |
| Wait+Yes x0.5 | 4,096 | 6.0% | 6.0% | 0% | 92.5% | 6 |
| </think> x1.3 | 1,024 | 23.5% | **23.5%** | 0% | 73.5% | 6, 7 |
| **W5T13** (Wait x0.5 + </think> x1.3) | 3,072 | 38.0% | **37.5%** | 1.0% | 57.0% | 6, 8 (reproduced) |
| W0T13 (Wait x0 + </think> x1.3) | 3,072 | 38.0% | 37.5% | 1.0% | 57.0% | 8 (= W5T13) |
| </think> x1.5 alone | 1,024 | 57.5% | 49.0% | 29.0% | 20.0% | 7 |
| **W5T15** (Wait x0.5 + </think> x1.5) | 3,072 | 63.0% | **55.5%** | 20.5% | 16.5% | 8 |
| W5T17 (</think> x1.7) | 3,072 | 47.5% | 9.5% | 86.5% | 31.5% | 9 |
| W5T20 (</think> x2.0) | 3,072 | 7.0% | 0.0% | 100% | 96.0% | 9 |
| W5T13 + answer neurons x1.3 | 3,072 + 2 cols | 41.5% | 41.0% | 0% | 54.5% | 8 (+3.5 strict) |
| W5T13 + check/verify neurons x0.5 | 3,072 + 6 cols | 36.5% | 36.5% | 0% | 59.5% | 8 (no gain) |
| W5T13 + Wait neurons x0.5 | 3,072 + 3 cols | 38.0% | 38.0% | 0% | 56.5% | 8 (no gain) |
| W5T15 + answer neurons x1.3 | 3,072 + 2 cols | 62.0% | 56.0% | 20.0% | 14.5% | 9 (no gain) |
| W5T15 + check/verify neurons x0.5 | 3,072 + 6 cols | 58.0% | 51.0% | 23.0% | 17.0% | 9 (-4.5: verification helps) |
| W5T15 + Therefore x1.3 | 3,072 + 1,024 | 61.5% | 51.0% | 24.5% | 15.5% | 9 (-4.5) |
| Wait neurons x0.5 alone | 3 cols | 1.0% | 1.0% | 0% | 99.0% | 7 (null) |
| Therefore neurons x1.3 alone | 3 cols | 1.5% | 1.5% | 0% | 98.5% | 7 (null) |
| </think> x1.3 + Wait neurons x0.5 | 1,024 + 3 cols | 10.5% | 10.5% | 0% | 88.0% | 7 (harmful) |
| </think> x1.3 + Yes neuron x0.5 | 1,024 + 1 col | 25.0% | 24.5% | 1.0% | 73.0% | 7 (no change) |

Conclusion-strength sweep with Wait x0.5, strict: x1.3 37.5% -> **x1.5 55.5%** ->
x1.7 9.5% -> x2.0 0%. Past x1.5 a global </think> boost fires inline and after
thinking ends ("</think>\n\n</think>..." until the cap).

Weight-only atlas neurons are candidates, not causes: every neuron-column edit
was null or harmful alone; only the answer neurons helped, and only at W5T13.

## Two levers (practice, lenient correct | concluded)

| Edit | Concludes | Correct when concluded | Overall |
|---|---|---|---|
| base | 3/200 | 3/3 | 1.5% |
| </think> x1.3 | 71 | 66% | 23.5% |
| </think> x1.5 | 193 | 60% | 57.5% |
| W5T13 | 92 | 83% | 38.0% |
| W5T15 | 185 | 68% | 63.0% |

Conclusion pressure buys completion at the cost of premature stops; damping
"Wait" raises the quality of conclusions at equal pressure. ("Concludes" here
counts any `</think>`, so at x1.5 it includes inline emissions.)

## Held-out results (1,000 problems: GSM8K-test[:500] + MATH-500)

| Edit | Lenient | **Strict** | Multi-`</think>` | Hit limit | Matched control (strict) | Run |
|---|---|---|---|---|---|---|
| base | 1.2% | **1.2%** | 0% | 98.8% | — | 5b, 6, 8 (1.2%); 7, 9 (1.0%) |
| Wait x0.5 | 6.9% | (files lost) | — | 92.8% | This x0.5: 0.9% (lenient) | 5b |
| **W5T13** | 34.7% | **33.6%** | 1.3% | 63.3% | This x0.5 + If x1.3: **0.2%** | 6 |
| **W5T15** | 50.6% / 50.9% | **39.1% / 37.9%** | 36–37% | 30–31% | C_W5T15 (This x0.5 + If x1.5): **0.1%** | 8 / 9 (replication) |
| </think> x1.5 alone | 48.1% | **34.9%** | 38.0% | 36.5% | If x1.5: **0.1%** | 7 |
| **Designed neuron L23#1998 (newline v2) + Wait x0.5** | 53.6% | **42.1%** | 27.6% | 25.2% | random output directions, same norm (+ Wait x0.5): **5.9%** | kernel |

Run 5b's per-problem files were lost (launcher crashed decoding a byte-cut
UTF-8 log tail and terminated the pod before download); its aggregate numbers
come from the launcher's event log. Fixed: ASCII-only polling + a 5-min mirror.

### Paired comparisons, strict (same problems; exact McNemar; bootstrap 95% CI)

| Comparison | Accuracy | Delta [95% CI] | Gained / lost | McNemar p |
|---|---|---|---|---|
| W5T13 vs base (run 6) | 1.2% -> 33.6% | +32.4 [+29.6, +35.3] | +325 / -1 | 5e-96 |
| control vs base (run 6) | 1.2% -> 0.2% | -1.0 [-1.7, -0.4] | +0 / -10 | 0.002 |
| **W5T13 vs matched control** (run 6) | 0.2% -> 33.6% | **+33.4 [+30.5, +36.3]** | **+334 / -0** | **6e-101** |
| W5T15 vs base (run 8) | 1.2% -> 39.1% | +37.9 [+34.9, +41.0] | +381 / -2 | 8e-111 |
| — GSM8K-test 500 | 1.8% -> 49.8% | +48.0 [+43.6, +52.4] | +242 / -2 | 2e-69 |
| — MATH-500 | 0.6% -> 28.4% | +27.8 [+24.0, +31.8] | +139 / -0 | 3e-42 |
| W5T15 vs base (run 9, replication) | 1.0% -> 37.9% | +36.9 [+34.0, +40.0] | +371 / -2 | 7e-108 |
| W5T15 vs W5T13 (runs 8 vs 6) | 33.6% -> 39.1% | +5.5 [+2.7, +8.4] | +138 / -83 | 3e-4 |
| **W5T15 vs matched control** (run 9) | 0.1% -> 37.9% | **+37.8 [+34.8, +40.8]** | **+378 / -0** | **3e-114** |
| </think> x1.5 alone vs base (run 7) | 1.0% -> 34.9% | +33.9 [+31.0, +36.9] | +342 / -3 | 2e-97 |
| **</think> x1.5 vs matched control** (run 7) | 0.1% -> 34.9% | **+34.8 [+31.9, +37.8]** | **+348 / -0** | **4e-105** |
| **Designed neuron vs base** (kernel pod) | 1.3% -> 42.1% | **+40.8 [+37.7, +43.9]** | **+409 / -1** | **3e-121** |
| — GSM8K-test 500 | 1.8% -> 52.8% | +51.0 [+46.6, +55.4] | +256 / -1 | 2e-75 |
| — MATH-500 | 0.8% -> 31.4% | +30.6 [+26.6, +34.8] | +153 / -0 | 2e-46 |
| **Designed neuron vs its random-direction control** | 5.9% -> 42.1% | **+36.2 [+33.1, +39.3]** | **+373 / -11** | **3e-95** |
| control (random neuron + Wait x0.5) vs base | 1.3% -> 5.9% | +4.6 [+3.1, +6.2] | +55 / -9 | 4e-9 |
| Designed neuron vs W5T13 | 33.6% -> 42.1% | +8.5 [+5.7, +11.4] | +150 / -65 | 7e-9 |
| Designed neuron vs W5T15 | 39.1% -> 42.1% | +3.0 [-0.1, +6.1] | +141 / -111 | 0.07 |

The three controls (This x0.5 + If x1.3, This x0.5 + If x1.5, If x1.5) all land
at or below base (0.1-0.2%); at x1.5 the two controls produce byte-identical
generations (8,184,595 tokens), so This x0.5 never changes sampling there.

Lenient (for reference): W5T13 vs control +34.5 [+31.6, +37.4], +345 / -0,
p = 3e-104.

## "Answer computed" detector (interactive kernel, layers 16-23)

Base traces compute the gold answer as a result (= / is / answer / boxed) in
63% of problems (149/237), at a median 6% of the way into the trace, and then
loop to the cap. 16-neuron detectors chosen on half the prefixes separate
"answer just computed" from "still working" and from "after </think>" on the
other half at AUC 0.985-0.999 in every layer 16-23 (layer 23: projection
0.99 vs -0.27 / -0.29). The model internally registers that it has the answer
— at the answer token. At the line break after the answer (where the model's
own conclusions put `</think>`), a ridge detector over layer-23 gate
pre-activations reaches only AUC 0.85 vs working line breaks (0.89 vs post).

## Writing a circuit: "answer computed -> conclude" (interactive kernel, practice)

All on top of Wait x0.5 (alone: 6.0% strict). B = unit </think> logit
direction through the final norm, (1+gamma) * E[</think>].

| Write | Weights | Write at answer / work / post | Lenient | **Strict** | Multi-`</think>` |
|---|---|---|---|---|---|
| Linear rank-1, L23 down_proj += 5.6 B (x) A | rank 1 | +5.4 / -1.1 / -1.3 | 2.5% | **0.0%** | 99% |
| Linear rank-1, beta 1.0 | rank 1 | +0.97 / -0.19 / -0.23 | 0.5% | **0.5%** | 0% |
| Designed neuron L23#1998, write 3.0 | 3 x 1,024 | +3.0 / -0.04 / -0.05 | 24.0% | **9.5%** | 21.5% |
| Designed neuron, write 5.0 | 3 x 1,024 | +5.0 / -0.07 / -0.08 | 35.5% | **1.5%** | 83% |
| Designed neuron (3.0) on W5T15 (55.5% strict alone) | 3 x 1,024 + 3 rows | — | 52.0% | **8.0%** | 85.5% |
| **Newline-anchored neuron v1** (fires at the line break after the answer) | 3 x 1,024 | — | 64.0% | **52.5%** | 30% |
| **Newline-anchored neuron v2** (+ answer-token negatives) | 3 x 1,024 | — | 66.0% | **53.0%** | 25% |

**What worked: fire at the line break, not the number.** The model's own
conclusions read "...answer\n</think>". A neuron whose gate is a ridge
detector for "line break after the answer is computed" (positives: first two
line breaks after the gold answer; negatives: working line breaks, post-
`</think>` prefixes and — in v2 — the answer token itself) lifts Wait x0.5
from 6.0% to **53.0% strict** on practice, within 2.5 points of the best
global edit (W5T15, 55.5%) without any global `</think>` boost. The detector is
weaker at line breaks (AUC 0.85; fires on 21% of answer line breaks and 4% of
working line breaks) but the positions are right, so a modest write (3.0)
suffices. Held-out (1,000 problems, strict): **42.1%** vs its random-direction
control 5.9% (+373 / -11, p = 3e-95), vs W5T13 +8.5 (p = 7e-9), vs W5T15 +3.0
(p = 0.07) — the best edit in this study.

**The number-anchored neuron was a negative result.** The model knows when it has the answer, and a single
repurposed neuron can be made to fire on exactly those positions (95.8% of
answer-computed prefixes vs 2.7-4% of working/post prefixes), but pushing
`</think>` there emits it immediately after the number, mid-sentence; the model
does not treat an inline `</think>` as the end of thinking and keeps reasoning.
The lenient grade rewarded this (24-36%); the strict grade shows at most +3.5
points at write 3.0 (n = 200, not significant) and harm at larger writes or on
top of W5T15. The linear rank-1 write fails for a different reason: it cannot
threshold (the detector is negative on working contexts), so a strong write
forces stops wherever digits appear and a weak one suppresses `</think>` while
working.

Neuron construction (kept for reuse): gate row = 4 x (ridge detector over
layer-23 gate pre-activations, so the new pre-activation is computable exactly
from captured activations), up row = scaled copy of a near-constant up neuron
(#44, ~1.0 in every context), down column = write x B. Victim #1998 was the
least-used unit in layer 23 (mean |act| x column norm = 0.007 vs median 0.020).
Read-back error after writing: <= 1e-3 (bf16).

## Mechanism

- Loops are "Wait, ...? Yes, that's what I did. Wait, ..." re-verification
  cycles: "Wait" opens 37.7% of in-loop lines (3.25x pre-onset).
- When the edited model concludes (reaches </think>), it is right 67–92% of the
  time; base concludes 2% of the time. Edits remove the obstacle (not
  stopping); they do not add knowledge.
- Suppressing "Wait" removes it entirely (0.0% of lines) with no substitution
  (But/Yes/Hmm flat) and no CJK drift (0.00/10k chars).
- Weight atlas (no forward passes): dedicated neurons for wait (L22#573,
  L17#1915, L18#881), therefore (L22#2001, L14#2590, L17#3064), yes (L23#3308),
  answer (L23#742, L18#458), check/verify (L21#474, L18#1824/#1126/#2486/#1118,
  L22#865), Chinese particle (L20#80).

## Side effects

- Phase-0 loop-direction edit on healthy Qwen3-0.6B (GSM8K-test 500): base
  79.4% → edited 79.8%; loops 5.6% → 5.6%. No damage. Repeated in runs 6 and 8
  (byte-identical): 79.2% → 79.8%, loops 5.2% → 5.6%. Base alone on runs 7 / 9:
  79.4% / 79.0%.

## Portable spec of the designed neuron (2026-11-03)

Regenerated with the packaged recipe (`design_conditional_neuron`, pinned to
neuron 1998 and up unit 44) from the same contrast prefixes, saved as an
`EditSpec` (`conclude_neuron_spec.json`, hash `749a5d716e564da6` — identical on
two pods), applied with `apply_spec_to_checkpoint` and verified tensor by
tensor. Detector AUC on held-out contexts 0.90; fires on 22% of answer line
breaks and 2% of negatives.

| Held-out (1,000, strict) | Spec run | Kernel run |
|---|---|---|
| base | 0.4% | 1.3% |
| designed neuron + Wait x0.5 | **41.2%** (55.6% lenient) | 42.1% |
| random-write control (+ Wait x0.5) | 5.2% | 5.9% |
| neuron vs control | **+36.0 [+32.9, +39.1], +367 / -7, p = 1e-98** | +36.2, +373 / -11 |
| neuron vs base | +40.8 [+37.7, +43.9], +410 / -2, p = 2e-119 | +40.8, +409 / -1 |

Practice (200, strict): 49.5% (kernel v2: 53.0%). Base held-out varies by pod
between 0.4% and 1.3% with identical settings.

## Side effects outside thinking-mode math (2026-11-03)

`benchmarks/weight_space/side_effect_job.py`, one L40S pod ($0.40). Each edit
applied from its spec and verified. Paired against base on the same items.

| | Base | W5T13 | W5T15 | Designed neuron |
|---|---|---|---|---|
| GSM8K-test 500, thinking **off**, lenient | 54.8% | 51.6% | 53.0% | 57.6% |
| — strict (finished, no `</think>` in the reply) | 54.8% | 51.2% | 50.0% | 49.2% |
| — replies with a stray `</think>` | 0.0% | 0.4% | 6.4% | **21.8%** |
| — vs base, strict (paired) | — | -3.6 [-7.0, 0.0], p = 0.06 | **-4.8 [-8.4, -1.2], p = 0.013** | **-5.6 [-9.2, -2.0], p = 0.004** |
| MMLU 1,000 (zero-shot letter, thinking off) | 45.0% | 45.6% | 45.4% | 44.0% |
| — vs base (paired) | — | +0.6, p = 0.34 | +0.4, p = 0.64 | -1.0, p = 0.15 |
| WikiText-2 perplexity (65k tokens) | 17.378 | 17.379 (+0.01%) | 17.386 (+0.05%) | **18.400 (+5.9%)** |
| p(`</think>`) per WikiText token | 3e-6 | 7e-5 | 4e-4 | **8.5e-3** |
| p(`</think>`) at the MMLU answer position | 1e-5 | 9e-4 | 1.5e-2 | 1.6e-2 |

- No edit measurably changes knowledge (MMLU).
- The token-row edits leave general text untouched; the stronger global boost
  (x1.5) leaks a stray `</think>` into 6% of thinking-off replies.
- The designed neuron is **not** confined to thinking: trained only on thinking
  traces, it also fires at line breaks in thinking-off answers and in prose,
  writing `</think>` there (22% of thinking-off replies; perplexity +5.9%).
- Context: thinking **off**, the unedited model already scores 54.8% on
  GSM8K-test, against 1.8% with thinking on (it loops). The edits repair
  thinking mode (the neuron: 52.8% on GSM8K-test with thinking on); they do not
  exceed the model's non-thinking accuracy.

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
stray `</think>`. v3's own control and base were run on a second pod (below).

### v3 with its own control (second pod, 2026-11-03, $1.40)

Rebuilt from scratch on a new pod: identical spec hash `7bac7db23edf6b2c` (and
control `9f6a5c5a17754e4d`). Held-out, strict, same pod:

| | Accuracy | vs v3, paired |
|---|---|---|
| base | 1.2% | v3 +44.0 [+40.9, +47.1], **+440 / -0**, p = 7e-133 |
| v3 random-write control (+ Wait x0.5) | 4.3% | v3 +40.9 [+37.8, +44.1], **+418 / -9**, p = 7e-111 |
| **v3** | **45.2%** (first pod: 45.1%) | |

Practice strict on this pod: 61.5% (first pod 57.0%; identical weights).

**Caveat (found 2026-11-03, after the runs above).** v3's thinking-off
negatives were line breaks from base replies to GSM8K-test[:500] with thinking
off — the same questions as the GSM8K half of the held-out set and as the
thinking-off side-effect check. No labels were used, but held-out questions
shaped the edit, and the 3.4% stray-`</think>` figure was measured on the same
replies v3 was trained to stay quiet on. v3's numbers above are therefore not
clean held-out results; v3t (below) rebuilds it with negatives from GSM8K-train
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

## Thinking on vs off at an equal budget (2026-11-03)

One A100 pod ($1.17): held-out 1,000, same sampling / seed / 8,192-token cap
for every condition. `summaries/thinking_budget_summary_v3t_v4.json`.

| Strict (lenient) | GSM8K 500 | MATH-500 | all |
|---|---|---|---|
| base, thinking off | 55.6% (57.8%) | 40.2% (42.0%) | 47.9% (49.9%) |
| v3t, thinking on | 57.0% (65.2%) | 33.8% (45.6%) | 45.4% (55.4%) |
| v3t, thinking off | 53.2% (56.4%) | 37.2% (42.8%) | 45.2% (49.6%) |
| v4, thinking on | 53.4% (61.8%) | 28.0% (43.8%) | 40.7% (52.8%) |
| v4, thinking off | 51.8% (55.6%) | 32.2% (41.0%) | 42.0% (48.3%) |

v3t thinking on vs base thinking off, paired strict: GSM8K +1.4 (p = 0.59),
MATH -6.4 (p = 0.006), all -2.5 (p = 0.13); lenient all +5.5 (p = 0.0009).
The edits reach parity with thinking off on GSM8K, not beyond; v3t's lead
over a 2,048-token thinking-off run (54.0%) came from the shorter cap.

## v4: intermediate-result negatives (2026-11-03) — negative

v3t's stored neuron, scored on captured base activations, fires at 12.7% of
line breaks after an intermediate "= number" result (4.1% at other working
line breaks). v4 adds 300 of those (base GSM8K-train traces, before the gold
answer) as negatives: offline 5.3% on them, target 14.5% (v3t 20.0%). Spec
`conclude_neuron_v4_spec.json`, hash `7664695121410597`; design pod $1.22
(plus $0.41 for a start whose SSH dropped).

Held-out, thinking on, strict: 40.7% vs v3t 45.4% on the same pod (-4.7
[-7.4, -2.0], +69 / -116, p = 0.0007). Early closes unchanged (GSM8K closes
before the gold: 34.7% vs 34.5%); replies with two or more `</think>` rose
GSM8K 67 -> 98, MATH 105 -> 179. The neuron fires again in answer sections
where the model keeps reasoning after an early close, which the `post`
negatives (clean answers) never covered.

## v5 / v5b: MATH positives + post-close negatives (2026-11-03) — negative

Design-only H100 pod ($1.21; captures saved): adds 354 MATH-*train* positive
line breaks (177 of 600 base traces write the gold as a result in thinking;
0 MATH-500 overlaps), 354 MATH working negatives, 151 post-close negatives
(v3t's run-on replies to GSM8K-train practice problems) and the v4
intermediate negatives. v5 = pod design (`76a7e4686dce6348`); v5b = the same
contexts fit on all of them offline (`a9929a6009bafbcd`).

Generation, one RTX 4090 pod ($1.06), held-out 1,000, 8,192-token cap:

| strict | GSM8K | MATH | all | two+ `</think>` | never closed |
|---|---|---|---|---|---|
| base, thinking off | 55.0% | 40.8% | 47.9% | | |
| v3t, thinking on | 57.0% | 35.6% | 46.3% | 186 | 155 |
| v5, thinking on | 43.2% | 10.6% | 26.9% | 622 | 21 |
| v5b, thinking on | 41.0% | 15.0% | 28.0% | 535 | 13 |

v5 vs v3t -19.4 (+37 / -231, p = 2e-35). The MATH positives fixed
non-closing but the neuron now fires at most line breaks of the edited
model's own generations (v5b at the first). v5 thinking off: stray
`</think>` in 379 / 500 MATH replies. v3t thinking off on this pod: stray
`</think>` in 9.2% of GSM8K and 28.8% of MATH-500 replies (the 6.8% on record
was GSM8K at a 2,048 cap).

Diagnostics note: the even/odd split trained every design on the first line
break after the answer and reported the second; offline fire rates (any
split) did not predict generation for v4 or v5.

## On-policy, practice-gated redesign (2026-11-03) — negative

`onpolicy_job.py`, one A40 pod ($1.10). Rounds refit v3t's contexts plus
contexts mined from the edited model's own replies to MINE practice problems
(`misfires.py`); scored on SELECT (100 GSM8K-train + 100 MATH-train, cap
4,096). Held-out untouched.

| Round | SELECT thinking on strict | two+ close | never closed | thinking off stray | gate cos to v3t |
|---|---|---|---|---|---|
| 0 (v3t) | 44.5% | 15.0% | 20.0% | 15.0% | 1.00 |
| 1 | 35.5% | 17.5% | 30.0% | 10.0% | 0.64 |
| 2 | 31.5% | 25.0% | 27.0% | 21.5% | 0.59 |
| 3 | 30.0% | 28.0% | 22.5% | 31.0% | 0.52 |

Kept v3t; no held-out run. Cause: the ridge detector (3,584 features, ~800
contexts, lam 100) is unstable (half-data fits correlate 0.909; 0.971 at lam
1,000) and the neuron's firing threshold is set implicitly by shrinkage.

## Explicit threshold on v3t (2026-11-03) — negative on held-out

`circuits.shift_threshold`: gate row minus `bias` x up row (the up row reads
~1.0 everywhere: mean 1.006, std 0.108 on 2,790 captured contexts), so the
pre-activation drops by `bias`; the down column is rescaled so the mean write
when firing is `write_norm`. v3t's detector is fixed (bias 0, write norm 3
rebuilds v3t's hash `f221470a088d0860`). `threshold_design.py` (offline),
`threshold_job.py`, one RTX 4090 pod ($1.35). Practice = MINE + SELECT (200
GSM8K-train + 200 MATH-train, cap 4,096), strict.

| bias, write norm | spec | think strict | two+ close | never closed | nothink strict | stray |
|---|---|---|---|---|---|---|
| 0, 3 (v3t) | f221470a088d0860 | 48.5% | 14.0% | 17.0% | 43.0% | 18.3% |
| -1, 3 | 60308af0b3654402 | 48.3% | 21.8% | 12.8% | 43.8% | 22.0% |
| +1, 3 (kept) | efae812819cec461 | 51.3% | 9.3% | 18.8% | 45.5% | 13.8% |
| +2, 3 | 80dd4c89309a3547 | 45.0% | 9.8% | 22.5% | 45.0% | 12.3% |
| 0, 2 | a2dd8469d9430bfe | 47.0% | 4.0% | 31.0% | 45.8% | 5.3% |
| 0, 4 | 72c1d8d8c40de507 | 45.8% | 24.3% | 9.8% | 44.0% | 24.3% |

Held-out 1,000 at 8,192 on the same pod (strict): base thinking off 48.5%
(GSM8K 56.6, MATH 40.4); v3t thinking on 46.3% (57.0 / 35.6, as on the earlier
4090); bias +1 thinking on 45.6% (56.0 / 35.2), hit the cap 35.7% vs v3t 30.0%;
bias +1 thinking off 44.6%.

- bias +1 vs v3t, thinking on: -0.7 [-3.0, +1.7], +70 / -77, p = 0.62.
- bias +1 thinking on vs base thinking off: -2.9, p = 0.066 (MATH -5.2, p = 0.013).
- bias +1 thinking off vs base thinking off: -3.9 [-6.2, -1.6], +51 / -90,
  p = 0.001: the leak into thinking-off mode costs accuracy.

The threshold trades unclosed replies against double closes and leaks; the
practice gain (+2.8 on 400, best of 7) did not survive held-out, where the
higher threshold left more replies running to the cap.

## "</think> already seen" guard (2026-11-03) — fixes thinking-off, ties v3t thinking on

AND gate on the conclude neuron, designed without fitting (`guard_design.py`,
v1 = commit 60ac1955): layer 23 head 0 (KV head 0; heads 1 and 3 hold
`</think>` back at line breaks, so not them) gets a constant query along the
`</think>` key direction (position-free dims) and puts >= 99.3% of its
attention on an earlier `</think>`; it writes a flag along a residual
direction the unembedding barely sees (`atlas.quiet_direction`, visibility
0.07) and the neuron reads it (`circuits.silence_on_flag`: multiplier -1,
gate -1.5 x 4.2). An earlier version pushed the `</think>` logit down from
the head instead; on CPU it blocked reply ends too and raised text loss
tenfold at strengths that worked.

Practice (400, cap 4,096, RTX 4090, $0.95 with held-out):

| candidate | spec | think strict | two+ close | never closed | nothink strict | stray |
|---|---|---|---|---|---|---|
| v3t | f221470a088d0860 | 49.3% | 15.3% | 16.5% | 43.0% | 18.3% |
| v3t + guard | f3fdbd1cde4cb89f | 47.3% | 0.8% | 24.3% | 51.0% | 0.0% |
| bias -1 + guard (kept) | 1068017e94a3d344 | 50.3% | 1.5% | 18.3% | 51.8% | 0.3% |
| write norm 4 + guard | 7ed84d47cfaabf54 | 49.0% | 2.0% | 19.3% | 50.3% | 0.0% |

Held-out 1,000 at 8,192 (strict): base thinking off 48.5%; v3t 46.3%;
bias -1 + guard thinking on 46.1% (GSM8K 58.4, MATH 33.8; hit the cap 40.2%
vs 30.0%); thinking off 47.4% (GSM8K 52.4, MATH 42.4).

- guard thinking on vs v3t: -0.2 [-3.1, +2.7], +109 / -111, p = 0.95
  (lenient -3.7, p = 0.017; MATH lenient -7.4, p = 0.0005: more replies run
  to the cap).
- guard thinking on vs base thinking off: -2.4, p = 0.17 (GSM8K +1.8,
  p = 0.53; MATH -6.6, p = 0.004).
- guard thinking off vs base thinking off: -1.1 [-4.3, +2.1], +124 / -135,
  p = 0.53 (the unguarded edit: -3.9, p = 0.001).

Cause of the extra unclosed replies: before any close the head attends
diffusely and leaks a flag of 0.03-0.09 at the neuron's decision points,
which sit just above threshold (CPU: first-close P(</think>) 0.93 -> 0.03
for v3t + guard). Guard v2 (commit 3c0b1ef8) reads orthogonally to the top
32 directions of the before outputs and probes the edited model: leak p99
0.039 -> 0.007.
