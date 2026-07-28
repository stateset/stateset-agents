# StateSet Agents Cookbook

Copy-paste recipes for the workflows that actually come up. Pairs with [`PLATFORM_TOUR.md`](./PLATFORM_TOUR.md) (the structured journey) and [`WHITEPAPER.md`](./WHITEPAPER.md) (the algorithmic details).

Each recipe is **self-contained** — no required context from earlier sections — and includes:
- The shell commands to run (no Python edits needed)
- What artifacts land where
- What to check if it doesn't work

---

## Recipe 1 — Your first fine-tune in 4 hours

**You want:** a fine-tuned LoRA adapter on a small Qwen model for your client's customer-support use case, tonight.

```bash
# 1. Install + scaffold (30s)
pip install 'stateset-agents[training,api]'
stateset-agents starter customer-support ./client-acme --client-name "Acme Corp"
cd client-acme

# 2. Edit the bundled 8-scenario corpus to match your client's intents
${EDITOR} scenarios.jsonl
# (same schema: {"intent": "...", "user_query": "...", "must_acknowledge": [...], "must_avoid": [...]})

# 3. Train (Colab A100 recommended; ~3 hours)
pip install -r requirements.txt
python train.py

# 4. Sanity-check at the REPL
stateset-agents chat --model Qwen/Qwen3.5-0.8B --checkpoint outputs/acme_corp_v1 \
                     --grade customer_support

# 5. Serve
./serve.sh outputs/acme_corp_v1
# Then:
curl -X POST http://localhost:8000/agents/default/messages \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"I want a refund"}]}'
```

**Artifacts:**
- `outputs/acme_corp_v1/` — LoRA adapter
- W&B project `acme_corp` (auto-named by `--client-name`)

**If it didn't work:** `stateset-agents doctor` — checks GPU, peft, transformers, checkpoint env vars.

---

## The improvement loop in one command

**You want:** grade -> curate -> retrain without stitching together `grade_transcript.py`, `summarize_graded_batch.py`, and `sft_from_curated.py` by hand every time.

```bash
stateset-agents improve run --transcripts transcripts/ \
                             --reward customer_support \
                             --output improved/ \
                             --threshold 0.7
```

This does, in one step, what Recipe 2 below does manually: grades every transcript in `transcripts/` with the named reward, aggregates a summary (mean score, per-reward-component breakdown, count above threshold), curates the turns that clear `--threshold` into `improved/curated.jsonl`, and writes `improved/next_steps.md` with the exact `sft_from_curated.py`/`finetune_gspo.py` commands to train on it. Machine-readable results live in `improved/improve_summary.json`.

Bringing logs from a different agent framework? Ingest first in the same call:

```bash
stateset-agents improve run --transcripts openai_logs.jsonl --format openai \
                             --reward customer_support --output improved/
```

Check a previous run's numbers without re-grading:

```bash
stateset-agents improve status --output improved/
```

`improve` is offline-friendly — the reward functions (`gsm8k`, `customer_support`, `tool_calling`) are rule-based, no LLM judge or API key required; an LLM-judge reward name fails fast with a clear message.

Use `improve` for the fast path. Reach for the manual steps in Recipe 2 when you need finer control — e.g. inspecting per-transcript markdown reports before curating, or grading with a `--context-file` of ground-truth scenarios.

Driving this loop from an MCP client (Claude Code/Desktop, another agent) instead of a shell? See [`docs/MCP_SERVER.md`](MCP_SERVER.md) — `stateset-agents mcp` exposes `improve_run`, `improve_status`, `ingest_transcripts`, `grade_transcript`, `list_rewards`, `list_model_presets`, and `dry_run_finetune` as MCP tools (`pip install 'stateset-agents[mcp]'`).

---

## Recipe 2 — Iterate from production conversation logs

**You want:** take real customer transcripts from production, identify what your model gets wrong, and produce a better fine-tune.

```bash
# Assume production logs are stored as JSONL with {role, content} per line.
# Place them under transcripts/:
ls transcripts/
# session_2026_01.jsonl  session_2026_02.jsonl  session_2026_03.jsonl

# 1. Grade every transcript with the same reward used during training
make grade-batch DIR=transcripts/ REWARD=customer_support \
                 CURATED=curated.jsonl THRESHOLD=0.7

# 2. Get the cross-session summary
make grade-batch-summary GRADED_DIR=transcripts/graded \
                         OUTPUT=transcripts/SUMMARY.md

# 3. Look at the per-transcript table — sessions with low mean score
#    are where the model disagrees with your reward function.
cat transcripts/SUMMARY.md

# 4. Use the high-scoring examples (the ones model + reward agree on)
#    as new SFT training data
stateset-agents fine-tune curated.jsonl \
                          --base-model Qwen/Qwen3.5-0.8B \
                          --output-dir outputs/sft_v2

# 5. Sanity-check the new model on the originally-low-scoring sessions
stateset-agents chat --model Qwen/Qwen3.5-0.8B --checkpoint outputs/sft_v2 \
                     --grade customer_support \
                     --replay transcripts/session_2026_01.jsonl
```

**Iteration:** each round shifts curated.jsonl. The dedup in step 1 ensures duplicate (prompt, response) pairs are skipped — safe to re-run as your reward function evolves.

**Why this works:** the curated set captures what your model + reward function **agree** is good. Training on that set tightens the model toward the reward; spot-checking via chat tightens the reward toward your judgment.

---

## Recipe 2b — Bring your own agent's logs

**You want:** you already have a production agent — built with the OpenAI SDK directly, or with LangChain/LangGraph, anything — and you want its conversation logs to feed the same grade -> curate -> retrain loop as Recipe 2, without rewriting the agent or hand-converting the logs.

```bash
# Your OpenAI-format logs: one conversation per line, {"messages": [...], "reward"?}
# (bare message lists also work — see the module docstring for exact shapes)
stateset-agents ingest --format openai --input openai_logs.jsonl --output transcripts/session_ingest.jsonl

# Or a LangChain/LangGraph message dump (flat {"type": ..., "data": {...}}
# or dumpd/dumps constructor form — see the module docstring)
stateset-agents ingest --format langchain --input lc_run.json --output transcripts/

# From here it's the same loop as Recipe 2 — grade what came out:
make grade-batch DIR=transcripts/ REWARD=customer_support \
                 CURATED=curated.jsonl THRESHOLD=0.7
```

Programmatic equivalent, if you'd rather wire it into your own pipeline:

```python
from stateset_agents.data import from_openai_jsonl, to_grading_history

trajectories = from_openai_jsonl("openai_logs.jsonl")
for traj in trajectories:
    # each dict is exactly what scripts/grade_transcript.py's loader expects
    history = to_grading_history(traj)
```

**Why this works:** `stateset_agents.data.trajectory_ingest` maps OpenAI chat-completions messages and LangChain message dumps onto the framework's own `ConversationTurn`/`MultiTurnTrajectory` types faithfully — tool calls are preserved in turn metadata, multimodal content is flattened to text (with skipped parts noted, not silently dropped), and a per-conversation `reward`/`score` field in the source is carried through if present. `to_grading_history()` emits the plain `{"role", "content"}` dicts the grading loop already reads, so ingested logs are indistinguishable from transcripts captured via `stateset-agents chat --history`.

---

## Recipe 3 — Reproduce a whitepaper number

**You want:** confirm a specific published `eval_pass_at_1` from the whitepaper on your own hardware.

```bash
# 1. Check out the exact commit the paper pins
git clone https://github.com/stateset/stateset-agents
cd stateset-agents
git checkout 14c0e65    # the commit named in the whitepaper

# 2. Install
pip install -e ".[training]"

# 3. Reproduce one row of the table — three seeds for variance bars
make benchmark-phase0 TRAINER=gspo SEED=42 -- --train --output benchmark_results/whitepaper_v1/gspo_seed42.json
make benchmark-phase0 TRAINER=gspo SEED=1337 -- --train --output benchmark_results/whitepaper_v1/gspo_seed1337.json
make benchmark-phase0 TRAINER=gspo SEED=2026 -- --train --output benchmark_results/whitepaper_v1/gspo_seed2026.json

# 4. Aggregate + apply publication gates
make release-whitepaper-v1-strict
# Exits non-zero if your seeds don't pass: σ ≤ 0.10, +0.03 improvement, single commit.

# 5. Inspect the auto-generated table
cat docs/WHITEPAPER_SECTION_11_7.md
```

**Hardware budget:** ~6 hours on a single A100 for one (trainer, task) combination at 3 seeds.
**Full 9-run matrix** (3 trainers × 3 seeds, 1 task): `make benchmark-phase0-all`.

---

## Recipe 4 — Build a tool-using agent

**You want:** an agent that learns to invoke real APIs — your Slack, your CRM, your billing system.

```bash
# 1. Scaffold from the tool-calling template
stateset-agents starter tool-calling-agent ./tool-agent
cd tool-agent

# 2. Replace the 3 stub tools with calls to your real APIs.
#    Edit tools.py — add entries to CUSTOM_TOOLS following the same JSON-schema shape.
${EDITOR} tools.py

# 3. Replace scenarios.jsonl with your client's user queries.
#    Each row: {"user_query": "...", "expected_tool": "...", "expected_params": {...}, "expected_outcome": "..."}
${EDITOR} scenarios.jsonl

# 4. Train + serve
pip install -r requirements.txt
python train.py
stateset-agents chat --model Qwen/Qwen3.5-0.8B --checkpoint outputs/tool_agent_v1
```

**The reward:** the shipped `ToolCallReward` parses the JSON tool-call block from each response and scores against expected tool + expected parameters + expected outcome (substring match). 60% / 30% / 10% by default — tune in `config.yaml`.

**For the Colab walk-through:** [`notebooks/tool_calling_agent_demo.ipynb`](../notebooks/tool_calling_agent_demo.ipynb).

---

## Recipe 5 — Run a batch evaluation against a trained checkpoint

**You want:** a single command that scores a fine-tuned checkpoint against a fixed eval set and produces a markdown report. The same one you'd run nightly or in a PR check.

```bash
# 1. Use the bundled sample eval set (10 scenarios across 4 intents)
#    Or replace with your own JSONL — same schema as scenarios.jsonl.
ls examples/sample_eval_set.jsonl

# 2. Score every scenario against the customer-support reward
stateset-agents evaluate \
    --checkpoint outputs/acme_corp_v1 \
    --scenarios examples/sample_eval_set.jsonl \
    --reward customer_support \
    --output eval_report.md \
    --threshold 0.7

# 3. Read the report
cat eval_report.md
```

**Output shape** (excerpt):

```markdown
# Batch evaluation — `customer_support`
**Scenarios:** 10
**Mean score:** 0.74 ± 0.18
**Pass rate (≥ 0.7):** 7/10 (70.0%)

| # | Score | Query | Response (head) |
|---|-------|-------|-----------------|
| 0 | ✅ 0.92 | I want a refund for order #4521 | I'd be happy to help with your refund... |
| 1 | ⚠️  0.43 | The app crashes every time...   | Let me look into that for you...      |
| ... |
```

**Use it in CI:** the markdown report makes the threshold check grep-able. For a PR gate that fails when mean score regresses, see [`docs/PLATFORM_TOUR.md` §FAQ](./PLATFORM_TOUR.md).

**Iteration:** failing scenarios become candidates for the next curation pass — capture them in a transcript, grade them, curate the corrected examples, re-train.

---

## Recipe 6 — Debug a stuck reward

**You want:** your training run isn't improving over baseline. What now?

```bash
# 1. Check the doctor first
stateset-agents doctor
# Verify: torch + CUDA + peft + transformers all present, GPU detected, checkpoint env vars right.

# 2. Look at the live reward distribution
stateset-agents chat --grade customer_support \
                     --replay your_test_set.jsonl
# If the reward gives 1.0 to obviously bad responses, that's reward gaming.
# If the reward gives 0.0 to obviously good responses, the reward signal is wrong.

# 3. Profile each training phase
python scripts/run_phase0_benchmark.py \
    --trainer gspo --task customer_support \
    --train --output /tmp/profile.json
# Check the JSON — long rollout_seconds → vLLM helps. Long score_seconds → reward is bottleneck.

# 4. Common fixes from the FAQ in PLATFORM_TOUR:
#   - GSPO clip range too tight → widen clip_range_left / clip_range_right
#   - DAPO dynamic sampling filtering everything → check accuracy_threshold
#   - Reward collapsing within group → add a length penalty / diversity reward
```

**Triage table:** see [`docs/WHITEPAPER.md` §11.5 Failure Modes and Diagnostics](./WHITEPAPER.md#115-failure-modes-and-diagnostics).

---

## Recipe 7 — Hand off to a colleague

**You want:** your colleague to reproduce a result without a knowledge handoff.

```bash
# 1. Capture your environment provenance
stateset-agents version --json > version.json
# {
#   "version": "0.12.1",
#   "git_commit": "14c0e65",
#   "python": "3.10.16",
#   "dependencies": {"torch": "2.6.0+cu124", ...}
# }

# 2. Capture your training config
ls config.yaml scenarios.jsonl reward.py train.py
# These were created by `stateset-agents starter` — already self-contained.

# 3. Capture the result + provenance
make release-whitepaper-v1   # generates summary.md + summary.csv + figures + manifest

# 4. Send them this directory. They run:
#    git checkout 14c0e65 && pip install -e ".[training]" && python train.py
#    The same seed (in config.yaml.training.seed) reproduces the numbers.
```

**Sources of nondeterminism we control:** `set_all_seeds()` covers Python random, NumPy, PyTorch (CPU + CUDA), Transformers in one call. The whitepaper's canonical seed is `42`.

---

## Recipe 8 — Run the demos for a stakeholder

**You want:** show a colleague or prospect what the platform does, in ~10 seconds, no GPU needed.

```bash
# Benchmark pipeline (scaffold + smoke + aggregate + plot + release)
make demo

# Curation pipeline (chat → grade → curate)
make demo-curation

# SFT closure (curated → prepare → SFT dry-run)
make demo-full-loop
```

Each produces formatted output suitable for a screen share or asciinema cast, with real artifacts in `/tmp/stateset_*_demo/`. The synthetic data is obvious from the table (all stds at 0.000).

---

## Where each recipe lives in the code

| Recipe | Primary modules |
|--------|-----------------|
| 1 — First fine-tune | `stateset_agents.scaffolding`, `stateset_agents.training.gspo_trainer` |
| 2 — Iterate from logs | `scripts/grade_transcript.py`, `scripts/sft_from_curated.py`, `stateset_agents.cli.fine_tune` |
| 3 — Reproduce whitepaper | `scripts/run_phase0_benchmark.py`, `scripts/release_v1_whitepaper.py` |
| 4 — Tool-using agent | `stateset_agents.data.tool_calling_bench`, `stateset_agents.core.tool_agent` |
| 5 — Batch evaluation | `stateset_agents.cli.evaluate` (batch mode), `examples/sample_eval_set.jsonl` |
| 6 — Debug stuck reward | `stateset_agents.utils.cli.doctor`, whitepaper §11.5 |
| 7 — Hand off | `stateset_agents.utils.reproducibility`, `stateset_agents.cli.version` |
| 8 — Demos | `Makefile` targets `demo`, `demo-curation`, `demo-full-loop`, `demo-all` |

---

*Bugs in a recipe are bugs in the platform — please open an issue at [github.com/stateset/stateset-agents](https://github.com/stateset/stateset-agents/issues).*
