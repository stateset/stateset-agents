# Distributed-scaling evidence

Scaling runs use the measured-run schema in
[`../framework_comparison/SCHEMA.md`](../framework_comparison/SCHEMA.md) plus a
`workload_config_sha256` field. The digest must identify the canonicalized
model, data, optimizer, rollout, sequence-length, batching, and step-count
configuration shared by every topology.

The default publication matrix requires three matching seeds at 1, 2, 4, and
8 GPUs. `benchmarks/scaling_comparison.py` rejects missing topologies, mixed
seed sets, different GPU models, changed workload digests, or incomplete
provenance. It reports speedup and efficiency relative to one GPU without
inventing a pass threshold.

```bash
python benchmarks/scaling_comparison.py \
  benchmark_results/scaling/evidence \
  --gpu-counts 1 2 4 8 \
  --output-dir benchmark_results/scaling/report
```
