# Framework comparison evidence schema

StateSet only publishes framework comparisons backed by measured, matched runs.
Synthetic timing, estimated memory, hard-coded rewards, and subjective numeric
feature scores are prohibited.

Each JSON file represents one seed from one framework:

```json
{
  "schema_version": 1,
  "measured": true,
  "framework": "stateset-agents",
  "framework_version": "0.42.3",
  "harness_commit": "025787625165fad81c0212733070c9dcbe6bc62d",
  "protocol": "agent-rl-shootout-v1",
  "cache_policy": "prewarmed-model-and-dataset-cache-v1",
  "algorithm": "gspo",
  "algorithm_revision": "published-objective-or-implementation-revision",
  "model": "Qwen/Qwen3.5-8B-Instruct",
  "model_revision": "FULL_IMMUTABLE_REVISION",
  "task": "customer-support-multiturn-v1",
  "dataset_revision": "FULL_IMMUTABLE_REVISION",
  "seed": 42,
  "timestamp": "2026-08-26T21:00:00Z",
  "command": "the exact command used for this run",
  "config": {"num_generations": 4, "learning_rate": 0.000005},
  "hardware": {
    "gpu": "NVIDIA H100 80GB HBM3",
    "gpu_count": 1,
    "cuda": "12.8"
  },
  "metrics": {
    "samples_per_second": 1.0,
    "wall_clock_seconds": 3600.0,
    "peak_vram_mb": 70000.0,
    "eval_score_baseline": 0.50,
    "eval_score_final": 0.60
  },
  "artifact_sha256": "64_HEXADECIMAL_CHARACTERS_FOR_THE_TRAINED_ARTIFACT"
}
```

`benchmarks/framework_comparison.py` fails closed unless:

- every document explicitly says `measured: true` and carries complete provenance;
- protocol, cache policy, algorithm revision, model revision, dataset revision, task, GPU
  model, and GPU count match;
- each framework uses one version and has at least three unique seeds;
- every required metric is finite, with throughput, wall time, and VRAM
  strictly positive; and
- at least two frameworks are present.

The output is descriptive. It does not declare an overall winner or turn
subjective features into numbers.
