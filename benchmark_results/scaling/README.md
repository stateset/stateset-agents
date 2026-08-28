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

The default publication workload measures **weak scaling**: each GPU receives
the same fixed local batch and the global sample count grows with the topology.
The executable workload performs real BF16 forward/backward passes, computes
group-relative policy advantages, updates the policy, and synchronizes its
gradients through DDP. Its fixed gradient-accumulation window uses DDP
`no_sync` for intermediate microbatches and one synchronized update at the end,
matching StateSet's intended large-effective-batch execution. It uses a
deterministic generated policy task without model or dataset downloads. This
measures StateSet's single-node DDP training path; it is not strong-scaling,
LLM quality, rollout-serving, multi-node, or end-to-end agent throughput
evidence.

For a matched **strong-scaling** matrix, override only the mode and write to a
separate evidence directory:

```bash
python benchmarks/run_scaling_matrix.py \
  --gpu-counts 1 2 4 8 \
  --seeds 42 1337 2026 \
  --config-json '{"scaling_mode":"strong"}' \
  --output-dir benchmark_results/scaling_strong
```

The configured accumulation count is the one-GPU reference. Strong mode
divides it exactly by the world size, keeping the effective global batch and
number of optimizer updates fixed at every topology. A topology is rejected
when it cannot partition that work exactly.

```bash
python benchmarks/scaling_comparison.py \
  benchmark_results/scaling/evidence \
  --gpu-counts 1 2 4 8 \
  --output-dir benchmark_results/scaling/report
```
