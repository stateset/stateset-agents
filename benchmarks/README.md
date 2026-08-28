# StateSet Agents benchmarks

This directory separates measured evidence from smoke tests and synthetic
component tests. Only the strict evidence tools below may support comparative
performance claims.

## Measured framework shootout

Copy `shootout_manifest.example.json`, verify/update the pinned Hugging Face
revisions and framework versions, replace the exact GPU name, then commit the
harness before execution:

```bash
python benchmarks/shootout.py shootout-manifest.json \
  --output-dir benchmark_results/framework_comparison/evidence

python benchmarks/framework_comparison.py \
  benchmark_results/framework_comparison/evidence \
  --output-dir benchmark_results/framework_comparison/report
```

The orchestrator runs StateSet and the independent upstream-TRL adapter for
three matched seeds. It passes every protocol field directly, rotates execution
order, measures wall time outside each adapter, checks the exact installed
version and canonical config digest, verifies hardware, hashes artifacts, and
retains logs on failure. Dirty-worktree runs and unpinned model/dataset
revisions are rejected.

## Other strict evidence gates

```bash
# Matched algorithm evidence
python benchmarks/algorithm_comparison.py EVIDENCE --output-dir REPORT

# Complete 1/2/4/8-GPU matrices
python benchmarks/scaling_comparison.py EVIDENCE \
  --gpu-counts 1 2 4 8 --output-dir REPORT

# Worker, controller, and network recovery evidence
python benchmarks/reliability_evidence.py EVIDENCE --output REPORT.json
```

Schemas and collection guidance live under `benchmark_results/`. These tools
fail closed on missing seeds, mismatched protocols, estimated results, and
incomplete provenance.

## Non-comparative tests

- `benchmark_suite.py`: component latency/throughput checks, often with stubs.
- `performance_benchmarks.py`: local component regression measurements.
- `real_performance_benchmarks.py`: system measurements that are not a neutral
  cross-framework comparison.
- `improvement_loop.py`: deterministic synthetic-data quality regression test.
- `autoresearch_eval.py`: autoresearch evaluation helper.

These are useful engineering tests, but they do not establish model-quality,
framework-superiority, GPU-scaling, or production-capacity claims.

See [`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md) for supported claims and the
publication policy.
