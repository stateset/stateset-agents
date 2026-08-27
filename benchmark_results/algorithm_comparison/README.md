# Measured algorithm comparisons

Algorithm shootouts use the same evidence fields and validation rules as
[`../framework_comparison/SCHEMA.md`](../framework_comparison/SCHEMA.md).
For an algorithm comparison, keep `framework`, framework version, harness,
protocol, model revision, dataset revision, task, and hardware identical while
varying `algorithm` and `algorithm_revision`.

At least three identical unique seeds are required for every algorithm.
Synthetic reward curves and estimated performance are not accepted. CUDA must
also match; the A+ roster is enforced explicitly rather than inferred from any
two available algorithms.

```bash
python benchmarks/algorithm_comparison.py \
  benchmark_results/algorithm_comparison/evidence \
  --required-algorithm grpo \
  --required-algorithm gspo \
  --required-algorithm dapo \
  --required-algorithm vapo \
  --required-algorithm gepo \
  --output-dir benchmark_results/algorithm_comparison/report
```
