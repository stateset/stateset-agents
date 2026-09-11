# Distributed-scaling evidence

Scaling runs use the measured-run schema in
[`../framework_comparison/SCHEMA.md`](../framework_comparison/SCHEMA.md) plus a
`workload_config_sha256` field. The digest must identify the canonicalized
model, data, optimizer, rollout, sequence-length, batching, and step-count
configuration shared by every topology.

The default publication matrix requires three matching seeds at 1, 2, 4, and
8 GPUs. `benchmarks/scaling_comparison.py` rejects missing topologies, mixed
seed sets, different GPU models, changed workload digests, or incomplete
provenance. The A+ publication gate is declared before execution: mean
throughput must increase at every topology and scaling efficiency must remain
at least 70% at 2, 4, and 8 GPUs.

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

Protocol v4 additionally records privacy-preserving physical-node identities
directly from every distributed rank, verifies the declared ranks per node,
and re-hashes the retained policy artifact. A provider driver can execute the
same matrix across physical machines through the shell-free
`scaling_launcher_manifest.example.json` contract:

```bash
make benchmark-scaling-multi-node-contract
make benchmark-scaling-multi-node-run \
  MANIFEST=benchmarks/scaling_launcher_manifest.json \
  OUTPUT_DIR=benchmark_results/scaling/multi_node
```

The example command name is intentionally a placeholder: the deployment must
provide an argv-only driver that provisions the declared topology, launches
the supplied workload under multi-node `torchrun`, and retrieves the JSON and
policy artifact to the requested output paths. The collector rejects an
eight-GPU topology that does not span at least two nodes, or any result whose
rank-observed topology differs from its manifest.
For A+ publication, every node identity must derive from the host DMI product
UUID; a container hostname or machine-ID fallback remains useful diagnostics
but cannot prove that pods occupy distinct physical machines.

The A+ path also requires a source-bound scaling container. Use
`make benchmark-scaling-image-plan` before any registry mutation and
`benchmark-scaling-image-push` only after supplying the exact confirmation.
The resulting `image-attestation.json` retains the registry-read SLSA
provenance, SPDX SBOM, OCI source/version labels, and manifest digest. Every
RunPod or Kubernetes lifecycle record in the matrix must name that exact
`repository@sha256:...` image; a tag alone cannot satisfy publication.

For RunPod, copy `scaling_launcher_runpod.example.json`. The concrete adapter
uses Secure Cloud global networking, pins every pod to one datacenter,
rendezvous through `POD_ID.runpod.internal`, verifies distinct provider
`machineId` values, applies catalog and authoritative post-allocation cost
ceilings, arms per-pod self-destruct watchdogs, retains lifecycle/cost records
outside the evidence directory, and terminates partial clusters. RunPod
documents standard Pod global networking at 100 Mbps, so this path is mainly
useful for functional or negative evidence; use an Instant Cluster or another
high-bandwidth provider for a credible 70%-efficiency run.

For CoreWeave CKS or Nebius Managed Kubernetes, start from
`scaling_launcher_kubernetes.example.json`. The provider-neutral adapter uses
a headless Service and stable Indexed-Job DNS, requests an explicit fabric
device such as `rdma/ib`, and applies hard pod anti-affinity on
`kubernetes.io/hostname`. It requires an image digest, exact kubeconfig
context/namespace, and a declared fabric; it retrieves rank-zero evidence,
hashes provider node identities in a lifecycle record, and deletes the Job
and Service even after admission or workload failure. Cluster-billed capacity
has no defensible per-Job price ceiling, so cost must be attested separately
by the provider billing export. Retain one raw-export-bound envelope per Job
using [`../../benchmarks/KUBERNETES_BILLING.md`](../../benchmarks/KUBERNETES_BILLING.md);
the A+ gate rejects incomplete attribution or line-item reuse and reports total
cost plus cost per measured optimizer step.

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
