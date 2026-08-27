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

The executable contract is
[`benchmarks/algorithm_shootout.py`](../../benchmarks/algorithm_shootout.py)
with the checked-in
[`algorithm_shootout_manifest.example.json`](../../benchmarks/algorithm_shootout_manifest.example.json).
It rotates algorithm order by seed and retains per-run stdout, stderr, neutral
adapter results, normalized policies, and failure JSON. Each evidence document
contains separate shared-protocol and algorithm-objective configurations; both
are independently attested by the adapter before the orchestrator accepts a
run. The manifest deliberately rejects gradient accumulation above one until
all five implementations apply equivalent accumulation semantics.

The complete live matrix is published in [`evidence/`](evidence/) with its
strictly validated [descriptive report](report/comparison.md): 15 measured
runs, three seeds each for GRPO, GSPO, DAPO, VAPO, and GEPO on one RTX 5080.
The matrix uses pinned Qwen2.5-0.5B-Instruct and GSM8K revisions and contains
hashed normalized policies for every run.

Live preflights and rejected attempts are retained under
[`diagnostics/`](diagnostics/). They are deliberately excluded from the
publication evidence directory and cannot satisfy the three-seed gate.

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
