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
arguments. A failed or timed-out adapter retains logs and cannot produce an
accepted evidence document. The orchestrator continues through the remaining
framework/seed pairs and writes `_accounting/shootout-summary.json`; any failed attempt
makes the overall command fail after the full roster has been attempted.

Copy `benchmarks/shootout_manifest.example.json`, replace every revision,
hardware, and installed-version marker with observed immutable values, then run
the orchestrator from a clean committed worktree.

Before the full run, execute the first seed for every configured implementation
and require the complete competitive roster:

```bash
python benchmarks/shootout.py shootout-manifest.json \
  --preflight \
  --required-framework stateset-agents \
  --required-framework trl \
  --required-framework verl \
  --required-framework nemo-rl \
  --required-framework openrlhf \
  --output-dir benchmark_results/framework_comparison/preflight
```

Preflight proves adapter compatibility only. Its one-seed evidence is not
publication-complete and must not be mixed into the three-seed evidence folder.

After execution, validate the complete competitive roster explicitly:

```bash
python benchmarks/framework_comparison.py EVIDENCE \
  --required-framework stateset-agents \
  --required-framework trl \
  --required-framework verl \
  --required-framework nemo-rl \
  --required-framework openrlhf \
  --validate-only
```
