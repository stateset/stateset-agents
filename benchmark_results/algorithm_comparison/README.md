# Measured algorithm comparisons

Algorithm shootouts use the same evidence fields and validation rules as
[`../framework_comparison/SCHEMA.md`](../framework_comparison/SCHEMA.md).
For an algorithm comparison, keep `framework`, framework version, harness,
protocol, model revision, dataset revision, task, and hardware identical while
varying `algorithm` and `algorithm_revision`.

At least three unique seeds are required for every algorithm. Synthetic reward
curves and estimated performance are not accepted.

```bash
python benchmarks/algorithm_comparison.py \
  benchmark_results/algorithm_comparison/evidence \
  --output-dir benchmark_results/algorithm_comparison/report
```
