# Standard agent-quality evidence

This directory is reserved for measured, paired base-versus-trained evaluations
on tau-bench, BFCL, and SWE-bench Verified. It intentionally contains no result
claim until the complete matrix has run and passed.

## Collection

Copy `benchmarks/agent_quality_manifest.example.json` and
`benchmarks/agent_quality_harnesses.example.json` to experiment-specific files,
replace all placeholder revisions, checkout paths, and driver commands, commit
the harness, then run:

```bash
make benchmark-agent-quality-run \
  MANIFEST=benchmarks/agent_quality_manifest.json \
  OUTPUT_DIR=benchmark_results/agent_quality
```

The runner creates:

- `runs/<suite>-seed<seed>/`: upstream stdout, stderr, adapter result, raw
  artifacts, and retained failure details;
- `evidence/<suite>-seed<seed>.json`: validated schema-v2 evidence binding the
  baseline policy, trained policy, trained artifact, task set, configuration,
  harness, suite, cost, and raw artifact digest;
- `_accounting/matrix-summary.json`: one record for every requested attempt;
- `report.json`: emitted only after the full three-suite matrix passes.

`--preflight` executes one seed per suite and cannot emit a publication report.
The measured runner rejects a dirty worktree. A complete matrix still needs a
mean improvement of at least `+0.03` in every suite and a paired 95% confidence
bound above zero.

## Neutral adapter output

Each suite adapter receives all values as separate command arguments and must
write JSON to `{adapter_output}` with these fields:

```json
{
  "status": "completed",
  "measured": true,
  "suite": "tau-bench",
  "suite_revision": "FULL_40_CHARACTER_REVISION",
  "split": "test",
  "seed": 42,
  "framework_version": "0.47.0",
  "baseline_model": "organization/base-model",
  "baseline_model_revision": "FULL_40_CHARACTER_REVISION",
  "trained_model": "organization/trained-model",
  "trained_model_revision": "FULL_40_CHARACTER_REVISION",
  "evaluation_config_sha256": "LOWERCASE_SHA256",
  "paired_task_ids_sha256": "LOWERCASE_SHA256",
  "tasks": 100,
  "baseline_successful_episodes": 40,
  "trained_successful_episodes": 50,
  "baseline_score": 0.4,
  "trained_score": 0.5,
  "evaluation_cost_usd": 5.0,
  "cost_source": "provider-api",
  "artifact_path": "/absolute/path/inside/artifact_dir/results.jsonl"
}
```

The included `benchmarks/adapters/paired_agent_harness.py` produces this neutral
object. Each suite driver only writes ordered JSONL records containing
`task_id`, boolean `success`, and measured `cost_usd`. The paired adapter
requires the exact same unique ordered task IDs for both policies and derives
the digest, scores, counts, and combined cost itself. The outer runner does not
trust an adapter-supplied artifact digest: it hashes the retained path after
execution.
