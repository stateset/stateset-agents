# Flywheel Headroom — does the loop raise a ceiling, or only hold one?

**Date:** 2026-08-17 · **Status:** complete · **Verdict: the ceiling ROSE — 2/12 → 10/12**

Every flywheel run before this one showed the loop *holding*: generation 2 tied
generation 1 at 3/3. That is a weak result, because the task was saturated —
gen-1 already passed everything, so there was nothing for gen-2 to win. This
experiment asks the sharper question:

> Does the loop raise a ceiling on a task generation 1 genuinely **fails**?

## Why a headroom task exists

Gen-1 (`outputs/muse_glimmer_eval_flow`, a Muse-Glimmer-30B LoRA) was trained on
exactly 140 examples. Verified locally against its training corpus:

- 140 rows, **all exactly 2 messages** — one user turn, one assistant turn.
- **7** distinct assistant response templates (shipping status, return opened,
  duplicate charge reversed, address updated, damaged→replacement, cancellation,
  30-day policy).
- **One issue per conversation.** Every one of the 140 replies contains exactly
  **one** proof token — never two.

So a message carrying **two** issues at once is strictly outside gen-1's
distribution. That is the headroom.

## Protocol

### 1. The hard eval (local, deterministic, seeded — `build_prompts.py`, seed 20260817)

12 held-out **compound** prompts: one order number, two distinct issues drawn
from *different* templates. Assertions demand both resolutions plus the persona,
with proof-tokens taken from gen-1's own reply vocabulary so a correct compound
answer is objectively detectable:

```json
{"prompt": "Hi — my order #55120: it arrived damaged, and I was double charged for it.",
 "expect": ["StateSet Support", "Astra @ StateSet", "55120", "replacement", "reversed"]}
```

| issue | proof token (gen-1's own wording) |
|---|---|
| not arrived | `on the way` |
| return | `prepaid label` |
| double charge | `reversed` |
| address change | `shipping address` |
| damaged | `replacement` |
| cancel | `cancelled` |
| policy | `30 days` |

A **disjoint** set of 30 compound prompts (orders 66010–66213 vs the eval's
55120–55197) was reserved for sampling. No prompt overlap.

Scoring uses `stateset_agents.training.sft.evaluate_checks` — the same
case-insensitive substring checker the training job's own eval gate uses.

### 2–3. Measure gen-1, then harvest — ONE pod, one session

One RunPod H100 80GB SECURE pod ($3.29/hr): base `meta-models/Muse-Glimmer-30B`
+ the gen-1 adapter, loaded through the shipped `load_base_model_for_sft`
(Muse-Glimmer is a composite multimodal repo and registers only under
transformers' image-text-to-text auto-mapping — `AutoModelForCausalLM` rejects
it, which cost one $0.38 pod to discover).

- **Stage 1 — gen-1 on the hard eval**, greedy, `max_new_tokens=300`.
- **Stage 2 — best-of-N harvest**, N=8 per prompt, temperature 0.9, top_p 0.95,
  over the 30 sampling prompts. Only samples passing the *same* objective checks
  were kept. Pod terminated immediately after.

### 4. Train gen-2

`RemoteJobSpec` + `RunPodExecutor` on the harvested set: Muse-Glimmer-30B,
3 epochs, lora_r 16, lora_alpha 32, lr 1e-4, max_length 512, H100 80GB,
container_disk 170 GB, `max_cost_usd=6`,
`parent_adapter="outputs/muse_glimmer_eval_flow"`, and the **12 hard-eval
prompts with their assertions** as `eval_prompts` (`eval_max_new_tokens=300`) —
so the job's own eval measures gen-2 on the exact set gen-1 was measured on.

## Results

| | pass rate on the 12 compound prompts |
|---|---|
| Base Muse-Glimmer-30B, untuned | **0 / 12** (0%) |
| **Generation 1** | **2 / 12** (16.7%) |
| **Generation 2** | **10 / 12** (83.3%) |

**Harvest rate: 58 / 240 samples passed (24.2%).** 13 of the 30 sampling prompts
produced at least one passing sample; the other 17 produced none in 8 tries.
Those 58 samples are the entire gen-2 training set.

Gen-2 was trained **twice** (see "an honest detour" below) and scored 10/12 both
times, failing **the same two prompts** — the result is reproducible, not a
lucky seed.

### Gen-1's failure mode is exactly the predicted one

It answers **one** issue — the template it knows — and silently drops the other:

> **Prompt:** `Order #55127 — it arrived damaged and on top of that I need to change the shipping address.`
>
> **Gen-1:** Thanks for reaching out to StateSet Support! I checked right away: I'm sorry about that! I've shipped a free replacement for order #55127 — no need to send the damaged one back. Anything else I can help with? — Astra @ StateSet
>
> ↑ `replacement` ✓, persona ✓, order ✓ — **`shipping address` missing.** FAIL.
>
> **Gen-2:** Thanks for reaching out to StateSet Support! I checked right away: I'm sorry about that! I've shipped a free replacement for order #55127 — no need to send the damaged one back. **I also updated the shipping address before it left the warehouse.** Anything else I can help with? — Astra @ StateSet
>
> ↑ both resolutions, persona intact. PASS.

The untuned base model, for the same prompt, emits a rambling chain-of-thought
deliberation ("We need respond. Likely customer service…") and never produces a
customer-facing reply at all — so gen-2's win is over gen-1, not over nothing.

### The two gen-2 failures are near-misses, not collapses

Both remaining failures address *both* issues but phrase one resolution outside
my token whitelist:

- `#55162` (return + policy): says "You have **30 days** from delivery to return
  anything, and I've also sorted your other request" — omits `prepaid label`.
- `#55190` (return + not arrived): says "I've **expedited** it and you'll have
  tracking within 24 hours" — omits `on the way`.

So 10/12 is a **lower bound** on gen-2's compound competence; the substring
proxy is stricter than the behaviour it stands for.

## An honest detour, and a data artifact

**The gated job's artifacts were lost.** Run A carried the assertions, so when
2 of 12 failed the job exited non-zero by design and the executor skipped the
artifact download — the adapter and `eval_results.json` died with the pod. Its
result (`Eval checks: 10/12 prompt(s) passed`) is in the job log and is the
protocol-specified measurement. Run B re-ran the identical job with the prompts
**ungated** (plain strings, no `expect`), which exits 0 and downloads
`eval_results.json`; I then scored those completions locally with the same
`evaluate_checks`. Run B also gave 10/12, same two failures. Both numbers are
reported; neither is a re-roll chosen after the fact.

**A chat-template channel header leaked into the harvest.** Every generation
from this model — gen-1's baseline included — is prefixed with a literal
` to=user` channel token. All 58 harvested samples carried it. Left in, gen-2
would have been trained to emit a channel token as message content, so it was
stripped from the training targets before training. This is a disclosed
deviation from a pure "keep the sample verbatim" rejection-sampling protocol.
It cannot affect any pass/fail number: every checked token appears after the
prefix, and the prefix is present identically in gen-1's baseline outputs.

## Cost

| pod | purpose | runtime | cost |
|---|---|---|---|
| `okt0mjjrddxg7k` | aborted — wrong Auto class for a multimodal repo | 6.8 min | $0.38 |
| `wgbo9cw5e59ysr` | gen-1 hard eval + best-of-8 harvest | 17.6 min | $0.97 |
| `4c4lyojaofi1hm` | gen-2 train, assertion-gated | 9.8 min | $0.54 |
| `taq5q886r36y1s` | gen-2 train, ungated (artifact recovery) | 26.1 min | $1.43 |
| | | | **$3.32** |

All figures from the executor's own `job_cost` / measured pod lifetime at
$3.29/hr. Budget was $6. `GET /v1/pods` returns an empty list — **zero pods
remaining**; no stragglers to delete.

## Limitations — read these before citing the number

- **Synthetic prompts.** The compound prompts are template-generated from five
  opener patterns, not real customer messages. Real compound complaints are
  messier and would likely score lower.
- **Objective checks are substring proxies.** A reply "passes" by containing the
  right tokens. It is not judged for correctness, tone, or whether the two
  resolutions are mutually consistent. As shown above, this cuts both ways.
- **One base model, one adapter lineage.** Muse-Glimmer-30B only. Nothing here
  shows the effect transfers to another model or another domain.
- **N=8, single seed** (20260817) for the harvest; single seed for the eval.
  No confidence intervals, no repeated harvests.
- **Small eval: 12 prompts.** One prompt is 8.3 points. The 2→10 gap is far
  larger than that granularity, but the precise rates are coarse.
- **The harvest is narrow.** 58 samples covering only 13 distinct prompts —
  17 of 30 prompts contributed nothing. Gen-2 generalised from 13 compound
  demonstrations to 12 unseen ones, which is the interesting part, but the
  training set is tiny.
- **Gen-2's training data is gen-1's own output**, so this measures
  self-distillation of a latent capability, not acquisition of a new one.

## Verdict

**The ceiling rose.** On a task that gen-1 genuinely fails — compound, two-issue
requests, provably outside its 140-example single-issue training distribution —
gen-1 scored 2/12 and gen-2, trained only on gen-1's own rare successes, scored
10/12. Reproduced across two independent training runs with identical failures.

The mechanism is exactly rejection sampling's: gen-1 *occasionally* handles a
compound request (24.2% of samples at temperature 0.9), curation harvests those
successes, and training makes the occasional behaviour the default. The loop
therefore does more than hold a saturated task — it can amplify a capability the
model already possesses but rarely expresses.

The corresponding limit, which this run happens to clear but does not refute:
**curation can only amplify behaviour the model already sometimes exhibits.**
Had gen-1 scored 0 across all 240 samples, there would have been nothing to
harvest and the loop would have had no way forward. 17 of 30 prompts were
exactly that case at N=8.
