# StateSet Agents – Quickstart (the boring path)

One loop you can run right away — no GPU, no API keys:

- pip install → ingest sample logs → improve (grade/curate) → one GRPO dry‑run (stub/HF) → serve.

All commands below are real and use files/scripts that live in this repository.

---

## 1) Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install stateset-agents
# Optional (for `serve`): FastAPI + Uvicorn gateway
pip install 'stateset-agents[api]'
```

Verify:

```bash
python -c "import stateset_agents; print(stateset_agents.__version__)"
```

---

## 2) Ingest sample logs and Improve (grade → curate)

Easiest: run the maintained example that writes three OpenAI‑format logs, ingests them, grades every assistant turn with the customer‑support rubric, and writes a curated JSONL:

```bash
PYTHON=python examples/five_minute_demo.sh
```

It prints the location of the curated set (curated.jsonl). You can also run the same loop explicitly:

```bash
# 2a. Write three chat-completions transcripts (from the example)
WORKDIR="$(mktemp -d)"; LOGS="$WORKDIR/logs.jsonl"
python - <<'PY'
import json, os
logs = [
  {"messages":[{"role":"user","content":"My package is late, can I get an update?"},{"role":"assistant","content":"Thanks for flagging this delay — I’ll check your order and email you the latest ETA."}]},
  {"messages":[{"role":"user","content":"I was charged twice for my order"},{"role":"assistant","content":"Sorry about that — I’ll refund the duplicate charge and confirm by email."}]},
  {"messages":[{"role":"user","content":"What’s included in the Premium Plan?"},{"role":"assistant","content":"Premium includes priority support and higher usage limits."}]},
]
with open(os.environ["LOGS"],"w") as f:
  for row in logs: f.write(json.dumps(row)+"\n")
print("wrote:", os.environ["LOGS"])
PY

# 2b. Ingest OpenAI logs → grading transcripts (one conversation per file)
python -m stateset_agents.cli ingest --format openai --input "$LOGS" --output "$WORKDIR/transcripts"

# 2c. Grade and curate with the customer-support reward (offline, no API keys)
python -m stateset_agents.cli improve run \
  --transcripts "$WORKDIR/transcripts" --format transcripts \
  --reward customer_support --output "$WORKDIR/improved"

echo "Curated set → $WORKDIR/improved/curated.jsonl"
```

---

## 3) One GRPO job (GPU‑free, stub/HF)

Run the unified GSPO driver in dry‑run mode (default). It builds the agent with the stub backend, resolves hyperparameters, and prints the plan — no downloads, no GPU:

```bash
python examples/finetune_gspo.py --model qwen3.5-0.8b
# Add --no-dry-run on a GPU host to actually train.
```

Prefer supervised fine‑tuning first? Turn the curated set into a CPU‑safe SFT plan (prints the training plan and exits 0 on CPU‑only machines):

```bash
python scripts/prepare_sft_dataset.py \
  --input "$WORKDIR/improved/curated.jsonl" --format chat \
  --output "$WORKDIR/improved/sft_train.jsonl" --min-score 0.7 --dedup

python scripts/sft_from_curated.py \
  --dataset "$WORKDIR/improved/sft_train.jsonl" \
  --base-model Qwen/Qwen3.5-0.8B \
  --output-dir "$WORKDIR/outputs/sft_v1" \
  --num-epochs 3 --lora-r 16
```

---

## 4) Remote training (default backend = Modal)

Modal is the recommended remote backend. Live transport remains unverified; jobs fail closed without credentials and never claim results.

Plan a run without spending:

```bash
# Default provider is Modal — --provider modal is optional
python -m stateset_agents.cli train-remote \
  --dataset "$WORKDIR/improved/sft_train.jsonl" \
  --base-model Qwen/Qwen3.5-0.8B \
  --plan-only
```

Run for real (requires Modal account + extras):

```bash
pip install 'stateset-agents[modal]'
python -m stateset_agents.cli train-remote \
  --dataset "$WORKDIR/improved/sft_train.jsonl" \
  --base-model Qwen/Qwen3.5-0.8B
```

Other providers (RunPod, Fireworks, River, …) remain available but are not the default.

---

## 5) Serve the API

Start the FastAPI gateway. On CPU‑only machines, use a dry run to preview the command; on GPU/after training, pass your checkpoint:

```bash
# Preview (no server starts)
stateset-agents serve --dry-run

# Serve a trained adapter (on a machine with the weights available)
stateset-agents serve --checkpoint "$WORKDIR/outputs/sft_v1" --host 0.0.0.0 --port 8000
```

---

## What keeps this from rotting?

- CI runs a CLI smoke over the same entrypoints used above.
- A docs‑contract test asserts the `stateset-agents` commands referenced here exist.
- The npm client’s test script is JS‑only and typechecks, with a guard that prevents accidental `*.ts` execution.

If something here stops working, please open an issue.

---

• Need a deeper tour? See `docs/PLATFORM_TOUR.md` and `docs/COOKBOOK.md`.*** End Patch``` }}}```?? Wait... The tool requires valid grammar exactly. Ensure no stray braces. The patch content should be plain. Let's correct. I included extra triple braces. I must re-submit. Let's produce proper ApplyPatch input. We'll craft again without stray delimiting.
