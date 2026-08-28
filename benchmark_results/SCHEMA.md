# Phase 0 Benchmark Result Schema

Every Phase 0 benchmark run produces a single JSON file conforming to this schema. The schema is intentionally minimal — what's the trainer, what hardware, what seed, what did it score — and stable: future versions only add fields, never remove or rename.

## Required fields

```jsonc
{
  // Identification
  "trainer": "grpo" | "gspo" | "dapo" | "vapo" | "gepo",
  "model": "Qwen/Qwen3.5-0.8B",       // HF model identifier
  "model_revision": "40-char immutable checkpoint commit",
  "seed": 42,                         // canonical seed for the v1.0 results
  "commit": "40-char git commit SHA", // exact source the script ran against
  "evidence_class": "measured",       // "synthetic" is preview-only
  "timestamp": "2026-05-13T11:30:00Z", // ISO-8601 UTC

  // Reproducibility
  "config": {                         // full trainer config dataclass as dict
    "learning_rate": 5e-6,
    "num_generations": 4,
    "clip_range_left": 3e-4,
    // ... all trainer fields
  },

  // Results
  "metrics": {
    "eval_pass_at_1": 0.41,               // post-training accuracy on GSM8K test
    "eval_pass_at_1_baseline": 0.32,      // pre-training accuracy (frozen model)
    "improvement": 0.09,                  // eval_pass_at_1 - baseline
    "train_reward_mean_final": 0.55,      // final reward at end of training
    "train_reward_std_final": 0.18,
    "wall_clock_seconds": 1842.5,
    "rollout_seconds": 1101.2,            // phase 2 of the training data flow
    "score_seconds": 412.0,
    "update_seconds": 322.0,
    "peak_vram_mb": 24317,
    "train_examples": 200,
    "eval_examples": 100,
    "max_grad_norm_ratio": 1.8,           // max / median non-zero grad norm
    "status": "trained"
  },

  // Provenance — where did this number come from?
  "wandb_run_url": "https://wandb.ai/stateset-agents/whitepaper-v1/runs/abc123",
  "hardware": {                       // required for publication
    "gpu": "NVIDIA A100-SXM4-80GB",
    "vram_gb": 80,
    "cuda": "12.4",
    "driver": "550.54.15"
  }
}
```

## Naming convention

```
benchmark_results/
  whitepaper_v1/
    {trainer}_seed{seed}_{model_slug}.json
```

Example: `benchmark_results/whitepaper_v1/gspo_seed42_qwen3_5_0_8b.json`

## Validation

`scripts/validate_phase0_results.py` (planned) will check that every published number in the whitepaper has a matching JSON file and that the schema is followed.

## How to aggregate

```bash
jq -s '.' benchmark_results/whitepaper_v1/*.json > all_runs.json
```

The aggregated `all_runs.json` is the input to the figure-generation scripts that produce the reward-curve plots in §11.7 of the whitepaper.

## Pass/fail gates for the v1.0 whitepaper

Synthetic/demo rows may be rendered only with `--allow-synthetic`; they can
never pass a publication gate. Measured rows require immutable 40-character
source and model revisions, `status=trained`, named hardware, positive peak
VRAM, and positive wall-clock evidence.

For a result to be publishable in the v1.0 whitepaper revision:

| Gate | Requirement |
|------|-------------|
| Reproducibility | 3 seeds with stddev < 0.1 on `eval_pass_at_1` |
| Improvement | `improvement > 0.03` (a 3-point absolute gain over baseline) |
| Stability | No `grad_norm` spike >10x during training |
| Wall-clock | Training completes in < 4 hours on a single A100 |

Runs that fail any gate are still recorded, but flagged in the aggregated report.
