# Shootout execution manifest

`benchmarks/shootout.py` executes framework adapters from a shared JSON
manifest. Commands are argument arrays and never pass through a shell. The
following whole-argument placeholders are available: `{seed}`, `{adapter_output}`,
`{phase0_output}`, `{artifact_dir}`, `{model}`, `{model_revision}`, `{dataset_revision}`, and
`{task}`. `{config_json}` is the canonical compact JSON form of the shared
configuration; adapters must apply it and attest its SHA-256 digest.

Each adapter must write this neutral result to `{adapter_output}`:

```json
{
  "status": "completed",
  "measured": true,
  "config_sha256": "sha256-of-canonical-manifest-config",
  "framework_version": "exact-installed-version",
  "artifact_path": "/absolute/path/to/non-empty/artifact",
  "hardware": {"gpu": "NVIDIA H100 80GB HBM3", "gpu_count": 1, "cuda": "12.8"},
  "metrics": {
    "samples_processed": 512,
    "peak_vram_mb": 72000,
    "eval_score_baseline": 0.50,
    "eval_score_final": 0.61
  }
}
```

The orchestrator—not the adapter—measures end-to-end wall time and computes
samples/second. It hashes the artifact, records the exact command and current
commit, rotates execution order across seeds, and validates the final evidence
before writing it.

Secrets must be supplied through environment variables, never manifest command
arguments. A failed or timed-out adapter retains logs and `failure.json`, but
cannot produce an accepted evidence document.

Copy `benchmarks/shootout_manifest.example.json`, replace every revision,
hardware, and installed-version marker with observed immutable values, then run
the orchestrator from a clean committed worktree.
