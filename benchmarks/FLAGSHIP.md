# Flagship benchmark — the one number that sells the framework

**Goal:** a reproducible, headline-grade result on a real (7–8B) model on the
framework's differentiator task — multi-turn customer support — with receipts
that anyone can re-run from a fresh clone. The existing whitepaper result
(judge +0.079 on a 0.5B model, §11.7) proves the pipeline; this proves the
*product*.

## The run

Three seeds × GSPO on an 8B instruct model, multi-turn customer-support
composite reward, judge-scored eval before/after. One command per seed:

```bash
# ~1× H100/A100-80GB per seed. LoRA + bf16. Budget ≈ 3–5 h per seed.
for SEED in 42 43 44; do
  python scripts/run_phase0_benchmark.py \
    --trainer gspo --task customer_support \
    --model Qwen/Qwen3.5-8B-Instruct \
    --num-train-examples 500 --num-eval-examples 200 \
    --seed "$SEED" --train --vllm \
    --output "benchmark_results/flagship_v1/gspo_seed${SEED}_customer_support.json"
done

# Aggregate → markdown + CSV + figures + gate report
python scripts/aggregate_phase0_results.py \
  --input benchmark_results/flagship_v1 \
  --output benchmark_results/flagship_v1/summary
python scripts/plot_phase0_results.py \
  --input benchmark_results/flagship_v1 \
  --output benchmark_results/flagship_v1/figures
```

Or via make:

```bash
make flagship-benchmark SEED=42       # one seed
make flagship-benchmark-all           # all three, sequentially
```

## Publish gates (do not publish a number that fails these)

1. **Three seeds, all reported** — no seed selection. Mean ± std in the
   headline; per-seed JSONs committed under `benchmark_results/flagship_v1/`
   (the schema in `benchmark_results/SCHEMA.md` applies).
2. **Judge stability** — run `examples/testing/test_judge_stability.py`'s
   protocol against the eval judge first; a judge with >0.05 self-disagreement
   invalidates the comparison.
3. **Baseline parity** — the pre-training eval uses the identical prompt
   template, decoding params, and judge as the post-training eval
   (`--skip-baseline` is forbidden for the flagship).
4. **Provenance** — commit hash, model revision, dataset revision, and full
   config embedded in each result JSON (the runner does this; verify before
   publishing).
5. **A negative or null result still gets committed** to
   `benchmark_results/flagship_v1/` — the credibility of every other number
   in this repo depends on not silently discarding runs.

## Where the number goes once it exists

- README badge + "What's new" headline (replace the §11.7 badge).
- `docs/WHITEPAPER.md` new section, anchored to the exact commit.
- The PyPI project description (long_description picks up README).

## Hardware notes

- 8B + LoRA + bf16 + vLLM rollouts fits a single 80 GB card; on 40 GB use
  `--model Qwen/Qwen3.5-4B-Instruct` and label the result accordingly.
- CI never runs this (GPU); the nightly `-m slow` convergence test
  (`tests/e2e/test_gspo_convergence_tiny.py`) is the automated proxy that the
  training loop still learns.
