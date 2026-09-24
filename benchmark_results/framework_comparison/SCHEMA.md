# Framework comparison evidence schema

StateSet only publishes framework comparisons backed by measured, matched runs.
Synthetic timing, estimated memory, hard-coded rewards, and subjective numeric
feature scores are prohibited.

Each JSON file represents one seed from one framework:

```json
{
  "schema_version": 2,
  "measured": true,
  "manifest_sha256": "64_lowercase_hex_characters_for_the_exact_shootout_manifest",
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
  "artifact_sha256": "64_HEXADECIMAL_CHARACTERS_FOR_THE_RETAINED_ARTIFACT",
  "artifact_path": "runs/stateset-agents-seed42/artifact"
}
```

`benchmarks/framework_comparison.py` fails closed unless:

- every document explicitly says `measured: true` and carries complete provenance;
- schema version, exact shootout-manifest digest, harness commit, protocol,
  cache policy, algorithm revision, model revision, dataset revision, task,
  canonical config, GPU model/count, and CUDA version match;
- each framework uses one version, has at least three unique seeds, and uses
  the identical seed set;
- every required metric is finite, with throughput, wall time, and VRAM
  strictly positive; and
- schema-v2 artifact paths are relative, remain within the evidence bundle,
  contain no symlinks, and hash to the declared digest; and
- at least two frameworks are present.

Schema-v1 rows remain readable for historical reports but lack independently
re-verifiable artifact paths. They cannot be mixed with schema-v2 rows or used
to resume a corrected shootout. The comparison CLI rejects them by default;
`--allow-legacy-schema-v1` is an explicit historical-reporting exception.

Publication profiles should also pass each expected implementation with the
repeatable `--required-framework` option. The A+ profile requires
`stateset-agents`, `trl`, `verl`, `nemo-rl`, and `openrlhf`; a smaller roster
remains valid evidence for its exact participants but cannot satisfy that gate.

For rented RunPod execution, add `--require-provider-cost` and pass exactly one
evidence directory. The directory must contain every launcher lifecycle record
(`runpod-provider.json`, `runpod-provider-2.json`, and so on). Each schema-v2
record is bound to the same manifest and harness revision, uses the
post-allocation provider rate, records observed pod lifetime, and proves
termination. The report totals the resulting provider-derived estimate. This
is auditable rate-times-lifetime accounting, not a claim that the value is a
settled provider invoice.

The output is descriptive. It does not declare an overall winner or turn
subjective features into numbers.
