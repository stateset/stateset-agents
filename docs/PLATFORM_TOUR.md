# Platform Tour

A guided walk through the StateSet Agents platform, from `pip install` to a published v1.0 whitepaper revision. Twelve passes of platform work, in one document.

This tour is opinionated and linear. For reference docs, see the per-module README files. For the algorithm details, see [`docs/WHITEPAPER.md`](./WHITEPAPER.md).

---

## The 30-second pitch

You're a developer or consultant who wants to **fine-tune an LLM for a business use case** — customer support, math reasoning, function calling — and deploy it. StateSet Agents gives you a single tool that handles the full pipeline:

```
pip install                  scaffold              benchmark             release
     │                          │                     │                     │
     ▼                          ▼                     ▼                     ▼
stateset-agents[training]   starter <template>    benchmark phase0    release-whitepaper-v1
                            ./client-acme          --trainer gspo      → 6 artifacts
```

Three commands take you from clone to a defensible empirical result. The framework's distinctive value: **multi-turn RL** (group-based GRPO / GSPO / GEPO / DAPO / VAPO), **composable rewards**, and a **closed train → benchmark → serve** loop.

---

## Tour stop 1 — Install

```bash
pip install stateset-agents[training,api]
```

The package has a **deliberately lean core** (`numpy`, `pydantic`, `rich`, `typer`, `tqdm`). Everything else — `torch`, `transformers`, `peft`, `trl`, `fastapi`, `vllm` — is an optional extra. The `[training]` extra gets you the trainers; `[api]` gets the FastAPI gateway.

For the bleeding edge (matches this tour), install from source:

```bash
pip install -e 'git+https://github.com/stateset/stateset-agents@14c0e65#egg=stateset-agents[training,api]'
```

The current pinned commit is `14c0e65`, documented as v0.11.7 in [`CHANGELOG.md`](../CHANGELOG.md).

---

## Tour stop 2 — Scaffold a project

```bash
stateset-agents starter list
```

Returns the four bundled templates:

| Template | Purpose |
|----------|---------|
| `customer-support` | Multi-turn dialogue agent (the framework's differentiator) |
| `gsm8k-math` | Single-turn math reasoner with verifiable rewards |
| `tool-calling-agent` | Agent that learns to invoke tools/APIs |
| `minimal` | Bare scaffold — edit everything |

Each scaffolds 6–10 files: `config.yaml`, `scenarios.jsonl` (where applicable), `reward.py`, `train.py`, `eval.py`, `serve.sh`, `README.md`, `requirements.txt`, plus a `.stateset-agents-starter.json` provenance marker.

```bash
stateset-agents starter customer-support ./client-acme --client-name "Acme Corp"
```

`--client-name` slugifies into the `output_dir` paths and the `wandb_project` field throughout the scaffolded `config.yaml`. The scaffolded project lands ready for a named engagement.

---

## Tour stop 3 — Edit and train

Two files matter most after scaffolding:

```
client-acme/
├── config.yaml         # training + model + reward configuration
└── scenarios.jsonl     # your dataset (replace the 8 bundled rows)
```

Edit `scenarios.jsonl` to point at your data (same schema), then:

```bash
cd client-acme
pip install -r requirements.txt
python train.py
```

The training script uses **GSPO** by default — the framework's flagship trainer for multi-turn dialogue. Its tight `clip_range_left=3e-4 / clip_range_right=4e-4` defaults are intentional (sequence-level importance ratios are exp-of-a-small-per-token-quantity, so the effective clip needs to be much tighter than token-level PPO's 0.2). See [whitepaper §5.2](./WHITEPAPER.md#52-gspo-group-sequence-policy-optimization).

On a Colab A100, training the bundled 16-scenario customer-support corpus takes ~3 hours and costs ~$2.

---

## Tour stop 4 — Benchmark

Training alone isn't credible — you also need to show **the fine-tune improved over the base model, by how much, reproducibly**.

```bash
make benchmark-smoke              # 6-second pipeline health check (no GPU)
make benchmark-phase0 TRAINER=gspo SEED=42    # one configuration
make benchmark-phase0-all         # 9-run matrix: 3 trainers × 3 seeds (~6h on A100)
```

The runner emits schema-compliant JSON per [`benchmark_results/SCHEMA.md`](../benchmark_results/SCHEMA.md). Each run records the **git commit hash**, **all seeds applied** (Python, NumPy, PyTorch, Transformers — one call to `set_all_seeds()` handles them all), **per-phase wall-clock**, and **peak VRAM**.

The runner supports three trainers (`--trainer gspo | grpo | dapo`) and three tasks (`--task gsm8k | customer_support | tool_calling`). The matrix is intentional: GSM8K is verifiable (cheapest publishable signal), customer support shows the multi-turn differentiator, tool calling shows function-calling capability.

For real GPU training, add `--train`. Without GPU, the runner records `status: skipped_train_no_gpu` and falls through to baseline-only.

For faster rollouts on large batches:

```bash
python scripts/run_phase0_benchmark.py --trainer gspo --task gsm8k --train --vllm --output r.json
```

---

## Tour stop 5 — Aggregate, plot, publish

After 3 seeds × 3 trainers × 1 task = 9 result JSONs land in `benchmark_results/whitepaper_v1/`:

```bash
make release-whitepaper-v1
```

This one command does everything:

1. **Aggregate** — groups runs by (trainer, model), computes mean ± std, applies publication gates (3 seeds, σ ≤ 0.10, +0.03 improvement, single commit).
2. **Plot** — produces two PNG figures: pass@1 per trainer (with error bars + baseline line) and improvement ranking (with the +0.03 gate line). Falls back to a markdown text-table when matplotlib isn't installed.
3. **Render** — generates `docs/WHITEPAPER_SECTION_11_7.md` — the auto-generated markdown table that drops straight into §11.7 of the whitepaper.
4. **Stage figures** — copies the PNGs into `docs/figures/`.
5. **Manifest** — writes `benchmark_results/RELEASE_MANIFEST.json` with the full provenance (commit, seeds, trainers, tasks, artifacts).

Six artifacts, one command.

---

## Tour stop 6 — Serve

```bash
# Close the train → serve loop
make serve-trained CHECKPOINT=outputs/acme_v1 BASE_MODEL=Qwen/Qwen3.5-0.8B

# Or:
stateset-agents serve --checkpoint outputs/acme_v1 --base-model Qwen/Qwen3.5-0.8B --port 8000
```

The bundled FastAPI gateway exposes OpenAI-compatible endpoints:

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"trained","messages":[{"role":"user","content":"I want a refund for order #4521"}]}'
```

For production deployment, Helm chart + 10+ values overlays cover A100, H100, B200, Kimi-K2.5-finetuned, GLM 5.1 FP8 profiles. GKE Autopilot and Standard cluster types are supported with separate staging/production examples.

---

## Tour stop 7 — Local CI parity

```bash
make smoke
```

Runs everything: 140 unit tests across 7 modules + benchmark smoke + scaffold smoke (all 4 templates) + notebook JSON validation. About 18 seconds, no GPU required.

`make ci` is `smoke` plus repo-hygiene + lint + type-check + coverage. About 90 seconds. Match it on every PR.

`make release-prep` is the final gate before publishing: `smoke` → `python -m build` → `twine check`. About 2 minutes.

---

## Tour stop 8 — Three Colab notebooks for the three pillars

If you don't have local GPU, the Colab notebooks are the path to actual numbers:

| Notebook | Task | Runtime on A100 | Cost |
|----------|------|-----------------|------|
| [`whitepaper_v1_gsm8k_benchmark.ipynb`](../notebooks/whitepaper_v1_gsm8k_benchmark.ipynb) | GSM8K math | ~45 min | ~$0.50 |
| [`customer_support_4h.ipynb`](../notebooks/customer_support_4h.ipynb) | Multi-turn dialogue | ~3 h | ~$2 |
| [`tool_calling_agent_demo.ipynb`](../notebooks/tool_calling_agent_demo.ipynb) | Function calling | ~2 h | ~$1.20 |

All three pin to commit `14c0e65`, use `set_all_seeds(42)`, invoke `train_with_gspo`, and write a schema-compliant JSON result. Drop the JSONs into `benchmark_results/whitepaper_v1/` and run `make release-whitepaper-v1`.

---

## Tour stop 9 — Whitepaper

The platform's algorithmic foundations and design philosophy are in [`docs/WHITEPAPER.md`](./WHITEPAPER.md). It's ~12k words across 14 sections + 3 appendices, with implementation citations to specific line numbers in the code so every claim is verifiable. Sections most relevant to this tour:

- **§5 Algorithmic Foundations** — GRPO, GSPO, GEPO, DAPO, VAPO with their math + per-trainer defaults
- **§6.5 Training Data Flow** — the 5-phase pipeline this platform builds on
- **§7.5 Benchmark Methodology** — the test harness this platform implements
- **§11.5 Failure Modes and Diagnostics** — what to do when training won't converge
- **Appendix C Reproducibility Commands** — every command this tour walks through

---

## Tour stop 10 — Close the loop: curate → SFT → iterate

After serving and seeing real conversations, you'll spot disagreements between the model and what you'd say. Capture them with `stateset-agents chat --history` (tour stop 6), then run the curation pipeline:

```bash
# Grade your transcripts + collect high-scoring examples
make grade-batch DIR=transcripts/ REWARD=customer_support \
                 CURATED=curated.jsonl THRESHOLD=0.7

# One-shot SFT loop: prepare the dataset + train a new LoRA adapter
make full-loop INPUT=curated.jsonl BASE_MODEL=Qwen/Qwen3.5-0.8B \
               OUTPUT_DIR=outputs/sft_v2

# Chat with the new adapter
stateset-agents chat --model Qwen/Qwen3.5-0.8B --checkpoint outputs/sft_v2
```

The framework's idempotent curated.jsonl + dry-run-when-no-GPU posture means you can iterate this cycle indefinitely as your reward function evolves. Each round, the model and your reward tighten on each other.

For an end-to-end Colab walk-through: [`notebooks/sft_from_curated_demo.ipynb`](../notebooks/sft_from_curated_demo.ipynb).

---

## Tour stop 11 — Where to next

Once you've shipped a v1.0 fine-tune:

- **Add an `LLMJudgeReward` component** (`stateset_agents/rewards/llm_judge.py`) — replace the rule-based composite with an LLM-judge for production-grade reward quality.
- **Scale to a real dataset** — the bundled corpora are deliberate (24 customer-support scenarios, 8 tool-calling scenarios) for reproducibility. For production, swap in 500–2000+ trajectories.
- **Run 3 seeds** (42, 1337, 2026) and confirm σ < 0.10 on your eval set before publishing numbers.
- **Wire W&B** (`report_to: wandb` in config.yaml + `WANDB_PROJECT=<slug>`).
- **Continual learning** (`stateset_agents/training/continual_learning.py`) — when the model needs to learn new intents without forgetting old ones (EWC, LwF, replay).
- **Sim-to-real** (`stateset_agents/training/sim_to_real.py`) — when you've been training against a simulator and want to bridge to real users (DANN, MMD, CORAL).

---

## Summary in numbers

After 12 cron passes of platform work, the state:

| Category | Count |
|----------|-------|
| Unit tests | 140 (all passing in ~18s, no GPU) |
| Trainers wired in the runner | 3 (GSPO + GRPO + DAPO) |
| Task adapters | 3 (GSM8K + customer support + tool calling) |
| Starter templates | 4 |
| Colab notebooks | 3 |
| Makefile targets (benchmark/scaffold/release) | 13 |
| CLI subcommands added | 5 |
| New Python modules | 5 |
| CI workflows | 1 (~3 min, no GPU) |

---

## FAQ

**Q: How do I check if my install is healthy?**
A: Three commands cover the diagnostic surface from broadest to narrowest:

- `make health` — full sweep: version + commit + dependencies + CHANGELOG entry + all CLI subcommands + GPU detection + end-to-end GSM8K pipeline smoke. ~10 seconds, the one-command answer to "is everything wired correctly?"
- `stateset-agents doctor` — environment + dependencies + checkpoint env vars in detail.
- `stateset-agents version` — package version + git commit + key dep versions for bug reports.

**Q: Can I get up-arrow history in `stateset-agents chat`?**
A: Yes, automatic. The REPL wires up `readline` on Linux/macOS and persists input history at `${XDG_STATE_HOME:-$HOME/.local/state}/stateset-agents/chat_input_history`. Up-arrow recalls previous commands; the history survives across sessions.

**Q: I ran `stateset-agents serve --checkpoint <path>` but my requests aren't going to the trained model.**
A: Run `stateset-agents doctor` — it'll show the `STATESET_DEFAULT_CHECKPOINT` env var, whether the path exists, and whether `STATESET_DEFAULT_BASE_MODEL` is set. The registered agent lands under id `"default"`. Hit it via `/agents/default/messages`. If the startup hook failed to register, the API logs will tell you why and a gpt2 demo agent is used as a fallback (with a warning).

**Q: My `make benchmark-phase0` runs but the result JSON shows `status: skipped_train_no_gpu` even though I have a GPU.**
A: The runner checks `torch.cuda.is_available()`. If that returns False, `torch` either isn't installed, isn't CUDA-built, or your driver/CUDA mismatch. Run `stateset-agents doctor` to verify — it explicitly reports CUDA availability and the GPU name.

**Q: I'm getting `TypeError: AgentConfig.__init__() got an unexpected keyword argument 'peft_path'`.**
A: You're on an older version of the package than the tour describes. Either install from source (the tour pins `14c0e65`) or upgrade past `0.12.0`.

**Q: GSPO with `clip_range_left=0.2` gives weird results.**
A: GSPO ratios are length-normalized — exp of a small per-token quantity — so the effective clip needs to be tighter than token-level PPO's `0.2`. The whitepaper-recommended defaults are `clip_range_left=3e-4, clip_range_right=4e-4`. See [whitepaper §5.2](./WHITEPAPER.md#52-gspo-group-sequence-policy-optimization) and the "If you see no exploration, this is the first knob to widen" note.

**Q: `make demo` works locally but I want to use this in a real talk / screen share.**
A: It's designed for that. The output is formatted with Unicode box-drawing characters and uses arrows to mark steps. `make demo` runs in ~3 seconds without GPU and produces real artifacts in `/tmp/stateset_demo/`. The synthetic results are obvious in the table (all rows show `± 0.000` std).

**Q: My fine-tune isn't improving over baseline. What now?**
A: Three things to check:
  1. `policy/clip_fraction` in W&B — should be 0.05–0.20. Above 0.5 means clipping is dominant and effective step size is small (widen the clip range).
  2. `reward/std` collapsing toward 0 within a group means mode collapse. Add a length or diversity penalty to the composite reward.
  3. If you're using DAPO with `use_dynamic_sampling=True` and seeing zero gradient signal, check the train/eval data — the buffer may be filtering everything out as already-100%-correct or 0%-correct.

See [whitepaper §11.5 Failure Modes and Diagnostics](./WHITEPAPER.md#115-failure-modes-and-diagnostics) for the full triage table.

**Q: Can I use this with my own dataset?**
A: Yes — that's the point. Scaffold a project with `stateset-agents starter <template> <output>`, then replace `scenarios.jsonl` with your data using the same schema. Each template's `README.md` documents the schema.

**Q: Can I use a model that isn't Qwen 3.5 0.8B?**
A: Yes. Edit `config.yaml.model.name` in your scaffolded project — any Hugging Face causal LM compatible with `AutoModelForCausalLM` works. For larger models (7B+) add `use_4bit: true` to the config for QLoRA.

---

*This tour was generated alongside the platform itself. Bugs in the tour are bugs in the platform — please open an issue.*
