# Flywheel Experiment — does generation 2 beat generation 1?

**Date:** 2026-08-13 · **Status:** complete — gen-2 job SUCCEEDED, all eval assertions passed

First empirical, end-to-end test of the self-improvement flywheel:

> conversation transcripts → `ingest` → `improve` (grade/curate) → `train-remote` → a new adapter

The question: if a generation-1 fine-tune's *own conversations* are fed back
through the real curation pipeline and used to train a generation-2 adapter,
does gen-2 hold (or improve) persona fidelity and task behavior?

## Setup

| | Generation 1 | Generation 2 |
|---|---|---|
| Base model | meta-models/Muse-Glimmer-30B | same |
| Training data | `stateset_support.jsonl` (hand-built, 100% clean) | `curated.jsonl` from graded gen-1 transcripts (30 examples, 83% clean — see below) |
| Method | LoRA SFT (r=16, α=32, lr 1e-4, 3 epochs, max_length 512) | identical |
| Hardware | RunPod NVIDIA H100 80GB (SECURE), 160 GB container disk | identical |
| Adapter | `outputs/muse_glimmer_eval_flow` | `outputs/flywheel_gen2` |

## Step 1 — transcripts (synthetic, and why)

Chatting with the live gen-1 adapter for ~30 multi-turn conversations would
require a long-lived GPU pod (hours of billed H100 time) before the experiment
proper even started. Instead, 30 transcripts were **synthesized from gen-1's
known, verified behavior** (its eval outputs reproduce the training template
verbatim) in the exact `chat-remote` transcript format
(`{"messages": [...], "metadata": {...}}` JSONL —
`stateset_agents.remote.chat_session.ChatSession.transcript`). This is the
honest limitation of this v1: the "gen-1 conversations" are simulated, with a
**planted ground-truth mix**:

- **25 good assistant turns** — persona-correct Astra replies (signon
  "Thanks for reaching out to StateSet Support!", concrete resolution, signoff
  "— Astra @ StateSet") over NEW order numbers (20xxx/70xxx, disjoint from
  gen-1's 10xxx training range).
- **15 flawed turns** across four flavors: `rude` (wrong tone, personal
  attack), `curt` ("ok, done." — no resolution), `missing_signoff` (correct
  resolution, persona dropped), `vague` (polite-adjacent deflection, no
  resolution).

Generator: `scratchpad/flywheel/gen_transcripts.py`, seed 20260813
(30 conversations, 40 assistant turns; 1/3 are two-turn conversations).

## Step 2 — the real pipeline: ingest → improve

Actual shipped commands, nothing reimplemented:

```bash
stateset-agents ingest --format openai --input transcripts.jsonl --output ingested/
stateset-agents improve run --transcripts ingested/ --reward customer_support --output improve_out/
python scripts/prepare_sft_dataset.py --input improve_out/curated.jsonl \
    --format chat --output sft_train.jsonl --min-score 0.7 --dedup
```

### Curation quality vs planted ground truth

Graded 30 transcripts / 40 assistant turns, mean score 0.625, threshold 0.7.
Curated 30 examples.

| Metric | Value |
|---|---|
| True positives (good kept) | 25 |
| False positives (flawed kept) | 5 |
| False negatives (good dropped) | 0 |
| **Precision** | **0.833** |
| **Recall** | **1.000** |

- All `rude` turns (safety gate → 0.0) and all `curt` turns (length penalty)
  were correctly dropped.
- All 3 `missing_signoff` and both `vague` turns **slipped through** at score
  0.75 — the same context-free rule-based grader gap
  `benchmarks/improvement_loop.py` documents (its measured precision on its
  own corpus is ~0.82; this experiment reproduces the gap on persona-flavored
  flaws). The rule-based `customer_support` reward cannot see persona
  requirements (signoff) or resolution quality without scenario context.

Net effect: the gen-2 training set is **30 examples, 25 clean / 5 flawed
(83%)** — versus gen-1's 100%-clean hand-built set. That contamination level
is itself part of the experiment: does the flywheel degrade the persona?

## Step 3 — train-remote (real H100, real pipeline)

`RunPodExecutor(wheel=dist/stateset_agents-0.38.0-py3-none-any.whl,
ready_timeout_s=900)` with a `RemoteJobSpec` matching gen-1's hyperparameters
exactly, plus assertion-gated eval prompts on three **held-out** order numbers
(#88121, #88342, #88515 — appear nowhere in either training set):

```python
{"prompt": "Where is my order #88121?",
 "expect": ["StateSet Support", "Astra @ StateSet", "#88121"]}
# ... damaged-item and cancellation variants likewise
```

Launcher: `scratchpad/flywheel/run_gen2.py`.

## Step 4 — gen-1 vs gen-2

Gen-2 job: `JobStatus.SUCCEEDED`, adapter + `eval_results.json` in
`outputs/flywheel_gen2/` (12 optimizer steps, 3 epochs over 30 examples).
Gen-1's eval predates assertion-gated prompts, so the same three checks
(`expect: ["StateSet Support", "Astra @ StateSet", "#<order>"]`) were applied
retroactively to `outputs/muse_glimmer_eval_flow/eval_results.json`.

| Metric (finetuned completions) | Gen 1 | Gen 2 |
|---|---|---|
| Eval assertion pass rate (held-out order numbers) | 3/3 (retroactive) | **3/3 (gated in-job)** |
| Signon "Thanks for reaching out to StateSet Support!" | 3/3 | 3/3 |
| Signoff "— Astra @ StateSet" | 3/3 | 3/3 |
| Correct order number echoed | 3/3 | 3/3 |
| Concrete resolution (tracking ETA / replacement / cancellation) | 3/3 | 3/3 |
| Base model (same prompts, no adapter) | 0/3 — rambling `to=self` deliberation, refuses to act | 0/3 — same |

Gen-2's completions are template-perfect Astra replies on all three held-out
prompts — e.g. for `#88342` (damaged item) it produced the free-replacement
resolution with signon and signoff intact. Notably, **none of the five flawed
training examples bled through**: no missing signoffs, no vague deflections,
despite 17% of the gen-2 training set carrying exactly those flaws.

## Limitations (read before citing)

1. **Synthetic transcripts.** Gen-1's conversational behavior was simulated
   from its verified eval outputs, not sampled live. The curation
   precision/recall numbers are exact (ground truth is planted); the claim
   "gen-2 trained on gen-1's real chat logs" is approximated.
2. **Same base model, one seed, three eval prompts.** This measures whether
   the loop *preserves and transfers* the behavior through a noisy curated
   set, not statistically significant improvement.
3. **Rule-based grader gap.** 5/30 curated examples are planted flaws the
   `customer_support` reward cannot detect without scenario context
   (`must_acknowledge`/persona checks). An LLM-judge reward would likely
   close this; it is deliberately out of scope for the offline loop.
4. **Template task.** The Astra task is intentionally template-like, so
   assertion evals are crisp — but it is far easier than open-ended support.

## Cost

One live RunPod run (of the two authorized). Pod
`stateset-sft-675a49697c6d` (NVIDIA H100 80GB HBM3, SECURE, $3.29/hr) lived
~13 minutes end-to-end — model download, LoRA SFT, base-vs-tuned eval —
for a total of **~$0.75**. The executor terminated the pod on completion;
the RunPod REST API confirmed **zero pods** remaining afterwards.

## Verdict

**The flywheel closes and holds, within this experiment's limits.**
Transcripts of gen-1's behavior went through the real shipped pipeline
(`ingest` → `improve run` → `prepare_sft_dataset` → `train-remote` on a real
H100), and the resulting gen-2 adapter reproduced the target persona with a
100% assertion pass rate on held-out order numbers — matching gen-1 despite
training on a noisier, machine-curated dataset (83% clean vs 100%).

What this proves: the loop **preserves and transfers** behavior end-to-end
with zero manual data work, and the curation stage's measured filtering
(P=0.833, R=1.0) is good enough that residual noise did not degrade the
model. What it does not prove: that gen-2 *beats* gen-1 — both saturate this
template task at 3/3, so the ceiling was hit, not raised. A harder task, live
(non-synthetic) gen-1 transcripts, and an LLM-judge grading pass are the
natural next iteration.
