# StateSet Agents benchmarks

This directory separates measured evidence from smoke tests and synthetic
component tests. Only the strict evidence tools below may support comparative
performance claims.

## External backend conformance

Run each external engine in its own dependency-compatible GPU image using a
completed copy of `backend_conformance_manifest.example.json`:

```bash
python benchmarks/backend_conformance.py backend-conformance.json \
  --output-dir benchmark_results/backend_conformance/nemo-rl \
  --timeout-seconds 1800

# RunPod: free catalog preflight (the default; creates no pod).
python benchmarks/runpod_backend_conformance.py backend-conformance.json \
  --dataset ./gsm8k.jsonl --plan-output runpod-plan.json

# RunPod: the only provisioning mode; repeat the manifest ceiling exactly.
RUNPOD_API_KEY=... python benchmarks/runpod_backend_conformance.py \
  backend-conformance.json --dataset ./gsm8k.jsonl --execute \
  --confirm-max-cost-usd 1.0 \
  --output-dir benchmark_results/backend_conformance/nemo-rl

# Revalidate the self-contained record and checkpoint bytes later.
python benchmarks/backend_conformance.py \
  --validate-evidence benchmark_results/backend_conformance/nemo-rl/conformance.json

# After collecting all three engine-specific directories, gate the roster.
python benchmarks/backend_conformance_suite.py \
  benchmark_results/backend_conformance \
  --output benchmark_results/backend_conformance/suite.json
```

The runner invokes the public StateSet backend protocol, requires a visible
NVIDIA GPU, records every GPU UUID/size/driver, verifies exact engine and
StateSet versions, rejects a dirty or mismatched harness checkout, and enforces
the manifest's exact GPU count/name and workload timeout before launch. Schema
v3 also requires an immutable container-image digest, provider/tier, container
disk size, a total billable-lifetime bound, and a finite positive spend ceiling.
The runner embeds and hashes that execution
contract, manifest, experiment, and resulting artifact, and writes either
`conformance.json` or `failure.json` without overwriting a prior attempt. Artifact
locations are evidence-relative, so copying the complete backend directory preserves
independent byte-level validation. The manifest contains no credentials; provide
Hub/provider credentials through the process environment. A provider wrapper
must additionally compare its live price quote with `max_cost_usd` before
renting hardware; recording a ceiling is not itself provider-side enforcement.
The RunPod launcher does this twice: first against the unauthenticated catalog
before creation, then against the authoritative whole-pod `costPerHr` returned
after allocation. Any drift terminates immediately. A local recovery lease and
an in-pod self-destruct cover launcher and client failure, and the downloaded
`runpod-provider.json` records lifetime, estimated spend, and confirmed cleanup.
The digest-pinned image must already contain the selected external engine and
its runtime dependencies. The launcher installs the exact committed StateSet
checkout without resolving dependencies, then cleans generated build metadata
before the harness performs its clean-worktree check.

The suite gate recursively discovers only `conformance.json` records, rehashes
every colocated checkpoint, and requires exactly one NeMo RL, OpenRLHF, and verl
record. It rejects duplicate or missing engines, mixed StateSet/harness versions,
and drift in the provider/GPU/time/cost envelope, algorithm, model revision,
seed, or semantic task. Its summary
is bound to the exact input documents and does not turn conformance into a
performance claim.

Conformance proves that an adapter can complete a real GPU training step and
emit a reusable checkpoint. It does not establish quality, throughput parity,
or the three-seed comparative evidence required below.

## Measured framework shootout

Copy `shootout_manifest.example.json`, verify/update the pinned Hugging Face
revisions and framework versions, replace the exact GPU name, then commit the
harness before execution:

```bash
# Cheap first-seed diagnostic across every configured framework.
python benchmarks/shootout.py shootout-manifest.json \
  --preflight \
  --required-framework stateset-agents \
  --required-framework trl \
  --required-framework verl \
  --required-framework nemo-rl \
  --required-framework openrlhf \
  --output-dir benchmark_results/framework_comparison/preflight

# Publication run: all frameworks and all three seeds.
python benchmarks/shootout.py shootout-manifest.json \
  --required-framework stateset-agents \
  --required-framework trl \
  --required-framework verl \
  --required-framework nemo-rl \
  --required-framework openrlhf \
  --output-dir benchmark_results/framework_comparison/evidence

python benchmarks/framework_comparison.py \
  benchmark_results/framework_comparison/evidence \
  --output-dir benchmark_results/framework_comparison/report
```

The checked-in example runs StateSet and the independent upstream-TRL adapter;
the competitive gate additionally requires verl, NeMo RL, and OpenRLHF adapter
commands. The orchestrator passes every protocol field directly, rotates
execution order, measures wall time outside each adapter, checks the exact
installed version and canonical config digest, verifies hardware, and hashes
artifacts. A failed framework no longer prevents the rest of the roster from
running: `_accounting/shootout-summary.json` accounts for every success and failure, and
the command returns nonzero after all attempts. Dirty-worktree runs and
unpinned model/dataset revisions are rejected. Preflight results are diagnostic
and cannot satisfy the three-seed publication gate.

## Other strict evidence gates

```bash
# Matched algorithm evidence
python benchmarks/algorithm_comparison.py EVIDENCE --output-dir REPORT

# Complete 1/2/4/8-GPU matrices
python benchmarks/scaling_comparison.py EVIDENCE \
  --gpu-counts 1 2 4 8 --output-dir REPORT

# Worker, controller, and network recovery evidence
python benchmarks/reliability_evidence.py EVIDENCE --output REPORT.json

# Two-node minimum, three-seed async fault matrix plus 12-hour soak
make benchmark-distributed-async-gate \
  INPUTS=benchmark_results/distributed_async/evidence \
  OUTPUT=benchmark_results/distributed_async/report.json

# Three-seed tau-bench, BFCL, and SWE-bench Verified quality matrix
make benchmark-agent-quality-gate \
  INPUTS=benchmark_results/agent_quality/evidence \
  OUTPUT=benchmark_results/agent_quality/report.json
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
