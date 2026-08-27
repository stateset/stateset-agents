# Distributed-scaling evidence

Scaling runs use the measured-run schema in
[`../framework_comparison/SCHEMA.md`](../framework_comparison/SCHEMA.md) plus a
`workload_config_sha256` field. The digest must identify the canonicalized
model, data, optimizer, rollout, sequence-length, batching, and step-count
configuration shared by every topology.

The default publication matrix requires three matching seeds at 1, 2, 4, and
8 GPUs. `benchmarks/scaling_comparison.py` rejects missing topologies, mixed
seed sets, different GPU models, changed workload digests, or incomplete
provenance. The publication gate is declared before execution: mean throughput
must increase at every topology and scaling efficiency must remain at least
50% at 2, 4, and 8 GPUs.

Generate the matrix on a host exposing eight identical CUDA devices:

```bash
python benchmarks/run_scaling_matrix.py \
  --gpu-counts 1 2 4 8 \
  --seeds 42 1337 2026 \
  --output-dir benchmark_results/scaling
```

The executable workload performs real BF16 forward/backward passes, computes
group-relative policy advantages, updates the policy, and synchronizes its
gradients through DDP. It uses a deterministic generated policy task so every
topology consumes the same global rows without model or dataset downloads.
This measures StateSet's single-node DDP training path; it is not LLM quality,
rollout-serving, multi-node, or end-to-end agent throughput evidence.

```bash
python benchmarks/scaling_comparison.py \
  benchmark_results/scaling/evidence \
  --gpu-counts 1 2 4 8 \
  --output-dir benchmark_results/scaling/report
```
