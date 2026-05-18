# Bundled Colab Notebooks

Six runnable Colab notebooks covering the full developer journey. Each pins to a specific commit, sets seeds via `set_all_seeds()`, and produces real artifacts. Open any of them in Colab via the badge inside.

## Pick the right notebook for what you want to do

```
                       ┌─────────────────────────────────────────────┐
                       │  Just installed.                            │
                       │  Want to ship a fine-tune tonight.          │
                       └────────────────────┬────────────────────────┘
                                            │
                                            ▼
                             quickstart_first_finetune.ipynb
                                  (~3 h on A100, $2)
                                            │
            ┌───────────────────────────────┼───────────────────────────────┐
            │                               │                               │
            ▼                               ▼                               ▼
   I want hard numbers          I want a multi-turn               I want function-
   for the whitepaper           customer-support agent            calling behavior
            │                               │                               │
            ▼                               ▼                               ▼
   whitepaper_v1_gsm8k_         customer_support_4h.ipynb       tool_calling_agent_
   benchmark.ipynb              (~3 h on A100)                  demo.ipynb (~2 h)
   (~45 min)                                                            │
            │                               │                               │
            └───────────────────────────────┼───────────────────────────────┘
                                            │
                                            ▼ (after training a model + serving)
                                grade_and_curate_demo.ipynb
                                  (chat → grade → curate)
                                            │
                                            ▼
                              sft_from_curated_demo.ipynb
                                (prepare → SFT → next adapter)
                                            │
                                            └──► back to top of pipeline
```

## The eight core notebooks

| Notebook | Stage | A100 runtime | Cost | What it produces |
|----------|-------|--------------|------|------------------|
| [`quickstart_first_finetune.ipynb`](./quickstart_first_finetune.ipynb) | First touch | ~3 h | ~$2 | Working LoRA adapter on a multi-turn customer-support corpus, REPL-tested, with provenance JSON for handoff. Mirrors [Cookbook Recipe 1](../docs/COOKBOOK.md). |
| [`customer_support_3seed_judge.ipynb`](./customer_support_3seed_judge.ipynb) | Canonical whitepaper result | ~25 min | ~$0.70 | **The whitepaper §11.7 publication-gate notebook.** Three seeds (42 / 1337 / 2026), both rubric and LLM-judge eval (local `Qwen2.5-1.5B-Instruct` judge — no API key), KL anchor enabled. Closes the keyword-rubric-blindness gap exposed by `customer_support_4h.ipynb`. |
| [`whitepaper_v1_gsm8k_benchmark.ipynb`](./whitepaper_v1_gsm8k_benchmark.ipynb) | RL train, verifiable (binary reward) | ~45 min | ~$0.50 | Schema-compliant benchmark JSON for §11.7 of the whitepaper. GSM8K math reasoning under GSPO with binary 0/1 reward. Produced [the proof-of-life result](../benchmark_results/whitepaper_v1/gsm8k_qwen3_5_0_8b_gspo_proof_of_life.json) — see issue #16 for the all-zero-groups pathology that motivated v2. |
| [`whitepaper_v1_gsm8k_benchmark_v2.ipynb`](./whitepaper_v1_gsm8k_benchmark_v2.ipynb) | RL train, verifiable (dense reward) | ~45 min | ~$0.50 | Same trainer/config/compute as v1 but swaps in `PartialCreditGSM8KReward` (0.0 / 0.2 / 0.5 / 1.0 tiers) so within-group variance stays non-zero on a weak base model. Clean A/B against v1 — only the reward shape changes. |
| [`customer_support_4h.ipynb`](./customer_support_4h.ipynb) | RL train, multi-turn dialogue | ~3 h | ~$2 | LoRA adapter for the 24-scenario customer-support corpus under GSPO. The framework's multi-turn differentiator. |
| [`tool_calling_agent_demo.ipynb`](./tool_calling_agent_demo.ipynb) | RL train, function calling | ~2 h | ~$1.20 | Tool-using agent with the bundled 3 sample tools (weather, calculator, search). |
| [`grade_and_curate_demo.ipynb`](./grade_and_curate_demo.ipynb) | Curate | ~5 min | <$0.10 | Curated JSONL of high-scoring (prompt, response) pairs from your captured transcripts. |
| [`sft_from_curated_demo.ipynb`](./sft_from_curated_demo.ipynb) | SFT closure | ~15 min | ~$0.30 | LoRA adapter trained on the curated examples. Closes the chat → grade → curate → SFT loop. |

Three legacy notebooks (`00_environment_setup.ipynb`, `01_qwen_support_agent_gspo.ipynb`, `02_qwen_sales_agent_gspo.ipynb`) remain for backward compatibility but are superseded by the eight above.

## Common patterns across notebooks

Every notebook:

- **Pins the framework** via `git clone --quiet` + `git checkout <commit>`. The commit is named explicitly in the first cell so users see what they're running.
- **Seeds all RNGs** via `set_all_seeds(42)` — covers Python `random`, NumPy, PyTorch (CPU + CUDA), Transformers in one call.
- **Uses `train_with_gspo()` as the trainer entry point** when training (the older `GSPOTrainer(...)` direct constructor is broken — see CHANGELOG 0.12.0).
- **Writes a schema-compliant JSON result** conforming to `benchmark_results/SCHEMA.md` when producing benchmark numbers.
- **Detects GPU** with `torch.cuda.is_available()` and falls through gracefully when missing.

## When to use what

| Situation | Notebook |
|-----------|----------|
| "I just want to see this work" | `quickstart_first_finetune` |
| "Reproduce a whitepaper benchmark (binary reward)" | `whitepaper_v1_gsm8k_benchmark` |
| "Reproduce a whitepaper benchmark (dense reward, A/B)" | `whitepaper_v1_gsm8k_benchmark_v2` |
| "Train for a client's customer support" | `customer_support_4h` |
| "My agent needs to call APIs" | `tool_calling_agent_demo` |
| "I have transcripts to grade" | `grade_and_curate_demo` |
| "I have curated examples to SFT on" | `sft_from_curated_demo` |
| "I want to see what the platform does in 5 seconds, no GPU" | Run `make demo-all` from a clone (not in Colab) |

## Running locally instead of Colab

Each notebook works locally too — same commands, but skip the `git clone /content/...` cell and run from the repo root:

```bash
pip install -e ".[training,trl]"
pip install jupyterlab ipykernel
jupyter lab
```

For GPU training, install a CUDA-enabled PyTorch build that matches your NVIDIA driver.

## CI gating

All eight core notebooks are JSON-validated in CI on every PR that touches them. See [`.github/workflows/benchmark-smoke.yml`](../.github/workflows/benchmark-smoke.yml).

For the algorithmic details behind these notebooks: [`docs/WHITEPAPER.md`](../docs/WHITEPAPER.md).
For the broader developer journey: [`docs/PLATFORM_TOUR.md`](../docs/PLATFORM_TOUR.md).
For copy-paste recipes that mirror these notebooks: [`docs/COOKBOOK.md`](../docs/COOKBOOK.md).
