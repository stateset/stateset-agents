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
