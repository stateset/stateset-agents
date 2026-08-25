# Getting started: train your model on River — zero machines rented

Fine-tune, self-improve, and RL-train language models through
[River](https://river.ai)'s remote-autograd service using `stateset-agents`
— no GPUs to rent, no SSH, no disks to size. Everything below is written
from runs that actually happened (20+ live training campaigns; see
[`PROOFS.md`](PROOFS.md)), including the failure modes and what they look
like when they bite you.

## 1. Setup (five minutes)

River's SDK requires **Python ≥ 3.12** (stateset-agents itself supports
3.10+, so use a side venv if your project runs older):

```bash
uv venv --python 3.12 .river-venv
uv pip install --python .river-venv/bin/python river-client stateset-agents jinja2
export RIVER_API_KEY=rv_...        # from the River console
```

Sanity checks — your key's trainable models, and their service health:

```bash
.river-venv/bin/python -c "
import os, river_client as river
c = river.Client(api_key=os.environ['RIVER_API_KEY'])
print(c.health_check(), c.get_capabilities())"
```

`get_capabilities()` is authoritative: River scopes models per account.

## 2. Your first fine-tune (one command, ~10 minutes)

Data is chat-format JSONL — one conversation per line, loss lands on every
`assistant` turn:

```json
{"messages": [{"role": "user", "content": "Ticket #30001 — I'm locked out."}, {"role": "assistant", "content": "Thanks for contacting IT! I've sent a secure reset link. — Byte"}]}
```

```bash
export STATESET_RIVER_VERBOSE=1    # watch progress live (step ticks + loss)
.river-venv/bin/python -m stateset_agents.cli train-remote \
  --provider river \
  --dataset train.jsonl \
  --base-model Qwen/Qwen3.5-9B \
  --output-dir outputs/my_model \
  --num-epochs 3 --learning-rate 1e-4
```

What you get back: **not weights** — the trained LoRA lives on River. Your
output directory holds `river_checkpoint.json` (the `river://` URI) and a
`stateset_manifest.json` (dataset sha256, hyperparameters, lineage). Talk
to the trained model through their API:

```python
import os, json, river_client as river
ckpt = json.load(open("outputs/my_model/river_checkpoint.json"))["checkpoint"]
c = river.Client(api_key=os.environ["RIVER_API_KEY"])
r = c.chat_complete_from_checkpoint(
    [{"role": "user", "content": "Ticket #9001 — I'm locked out."}],
    checkpoint_path=ckpt, base_model="Qwen/Qwen3.5-9B", max_tokens=300,
)
```

First sample from a fresh checkpoint can cold-start for **5+ minutes** —
that's their replica spin-up, not a hang.

Measured expectations (from our runs): 140 rows × 3 epochs ≈ 10 minutes on
the 9B, similar on the 35B MoE when their pool is healthy. A held-out
persona fine-tune scored 3/3 on canonical wording at this size —
repeatedly, across three domains.

## 3. Evals that gate the run

Add `--eval-prompts` with objective assertions and the job scores itself
greedily after training, writing `eval_results.json`:

```json
[{"prompt": "Ticket #66010 — I'm locked out.", "expect": ["reset link", "66010"], "forbid": []}]
```

Use held-out reference numbers so passing proves learning, not memorizing.
For real-world data without canonical phrases, use judges instead:
`{"prompt": ..., "judge": "customer_support", "min_judge_score": 0.7}`.

## 4. The flywheel: self-improvement in one command

Where your model *sometimes* succeeds at something checkable, the flywheel
converts rarity into reliability: sample best-of-N, keep only passing
outputs, train the next generation on them, measure, repeat — stopping on
plateau, perfection, or a dry harvest:

```bash
.river-venv/bin/python -m stateset_agents.cli flywheel \
  --provider river \
  --base-model Qwen/Qwen3.5-9B \
  --initial-adapter outputs/my_model \
  --harvest-prompts harvest.json --eval-prompts eval.json \
  --generations 2 --best-of 8
```

Live results this produced: 7/12 → 11/12 (9B), 7/12 → **12/12 in one
generation** (35B). The operating regime, measured: it lifts when the
harvest rate sits near ~60% with greedy headroom; when temperature
sampling already succeeds ~everywhere, it honestly plateaus — make the
task harder instead (see `eval_ladder`, which generates train/eval kits at
parameterized difficulty, including multi-turn episode scripts that score
cross-turn memory objectively).

Multi-turn: pass episode scripts (`{"turns": [...], "turn_expect":
[[...], ...], "forbid": [...]}`) as harvest/eval files and whole
conversations are rolled out, scored per turn, and trained on. Live
result: 8/12 → 12/12 on conversations requiring context carryover.

## 5. RL, when imitation isn't enough

```bash
... flywheel --provider river --algorithm cispo --rounds 4 --best-of 8 ...
```

Every sample is graded (expected-token fraction + a completeness bonus −
a full point per violated refusal), advantages are group-relative, and
training uses River's clipped importance-sampling losses on their
sampler's own logprobs. Two hard-won warnings baked into the defaults:

- **Reward shaping matters.** Our first graded reward Goodharted live —
  mean reward climbed while the real eval fell, because partial credit
  out-earned full passes. The completeness bonus exists because of that
  incident. Watch `rl_report.json`'s per-round eval, not just the reward.
- RL needs **variance**: groups where every sample scores the same are
  skipped as gradient-free. If most groups are zero-variance, your task is
  too easy or too hard for the current model.

## 6. When things go wrong (they did for us — here's what it looks like)

| symptom | meaning | what happens |
|---|---|---|
| `ALREADY_EXISTS ... use a fresh model_seq_id` | a slow create raced a client retry | auto-retried with a fresh session (3 attempts, backoff) |
| `NOT_FOUND - Request not found` mid-training | their pool lost an in-flight request | auto-retried; training restarts from scratch (sessions aren't durable) |
| steps crawling (~12s/step vs ~3s) | pool under load | it finishes; watch with `STATESET_RIVER_VERBOSE=1` |
| `No route to host` / health check hangs | full outage (we've seen one, ~12h) | job fails cleanly after retries; checkpoints already saved are safe — resume later |
| first checkpoint sample times out at 5 min | replica cold start | wait; use a longer `timeout` |
| `Cannot load optimizer from an inference checkpoint` | resuming from a `mode="inference"` save | handled: checkpoints are typed automatically |

Cost note: River bills per token and quotes no price to the SDK — the
cost ledger records these runs as **unknown, never zero**, and `--max-cost`
cannot gate them.

## 7. Where to go next

- [`RIVER_PROVIDER.md`](RIVER_PROVIDER.md) — provider reference.
- [`FLYWHEEL_DOMAIN2.md`](FLYWHEEL_DOMAIN2.md) — the full replication study.
- [`PROOFS.md`](PROOFS.md) — every claim above, with its evidence class.
