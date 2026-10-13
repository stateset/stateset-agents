# StateSet Agents benchmarks

This directory separates measured evidence from smoke tests and synthetic
component tests. Only the strict evidence tools below may support comparative
performance claims.

## Public product-use benchmark

The starter benchmark for teaching and evaluating agent interaction with the
StateSet product surface lives in [`product_use/`](product_use/). It contains
versioned MCP tasks, an execution-backed local scorer, an interactive JSONL
tool-session runner, and guidance for private generated transfer checks. Run
the public demonstration with:

```bash
python -m benchmarks.product_use.execution \
  --tasks benchmarks/product_use/tasks.v0.3.public.json \
  --submissions benchmarks/product_use/demonstrations.v0.3.json
```

The public task set measures improvement-loop usage, tool discovery, and
training configuration. Version 0.4 can generate evaluator-only task files and
prompts-only participant files from a private seed. These checks are separate
from the GPU performance evidence below:
tool-use scores must not be presented as training throughput or model-quality
claims.

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

### Five-framework matched GSPO matrix (v5)

`shootout_manifest_v6_five_frameworks.json` is the v4 protocol (same config
object, Qwen2.5-1.5B-Instruct and GSM8K revisions, seeds 42/1337/2026, one
NVIDIA L40S) extended to every framework `a_plus_gate.py` requires:
`stateset-agents-gspo`, `trl`, `verl`, `openrlhf`, `nemo-rl`. The three
competitor adapters keep the TRL adapter's CLI and result schema and import
its GSM8K split, prompt template, exact-numeric reward and evaluator
(`benchmarks/adapters/_matched_gsm8k.py`). Each one evaluates the base model,
trains in its framework's own environment as a subprocess, parses that
framework's own metrics, loads the trained LoRA adapter on the base model,
evaluates it with the shared evaluator, and only then writes the result.
A nonzero exit, a short run (fewer than `max_steps` optimizer steps), missing
reward metrics, a version mismatch, or a missing adapter fails closed with no
result file.

| Adapter | Pinned | GSPO switch | Native | Recorded deviations |
|---|---|---|---|---|
| `verl_gspo.py` | verl 0.9.1 (PyPI) | `actor.policy_loss.loss_mode=gspo`, `loss_agg_mode=seq-mean-token-mean` | sequence ratio, ε 3e-4/4e-4, β 0, group-normalized GRPO advantages, LoRA r16/α32 q,k,v,o | `lora_dropout` 0.0 (FSDP LoRA path has no dropout field) |
| `openrlhf_gspo.py` | OpenRLHF 0.11.2 (PyPI) | `--actor.policy_loss_type gspo` | sequence ratio, `--actor.eps_clip_low_high 3e-4 4e-4`, β 0 (no ref model), `group_norm`, LoRA r16/α32/dropout 0.05 | none |
| `nemo_rl_gspo.py` | NeMo-RL 0.7.0 (git tag `v0.7.0`, `81aa43d`) | `loss_fn.sequence_level_importance_ratios=true`, `token_level_loss=false` | sequence ratio, `ratio_clip_min/max`, β 0, mean/std group normalization, LoRA r16/α32/dropout 0.05 | LoRA A init is xavier (no kaiming option); LoRA weights are merged into the base for vLLM generation |

Every adapter result also carries `protocol` (the exact framework settings),
`protocol_deviations` and `protocol_notes`; `framework/launch.json` and the
framework's own log/metrics are kept in the hashed artifact directory. The
shootout evidence documents do not copy those fields, so read
`runs/<framework>-seed<N>/adapter-result.json` alongside the report.

Batch geometry follows the StateSet harness: `per_device_train_batch_size x
gradient_accumulation_steps` prompts per optimizer step (4), each with
`num_generations` completions (16 completions per step, one optimizer step per
rollout batch), and `samples_processed = steps x 16`. TRL 1.x instead counts
`per_device_train_batch_size` in completions (1 prompt x 4 completions per
step) while `trl_grpo.py` reports the same `steps x 16`, so the TRL leg does a
quarter of the declared work per step. That adapter is unchanged here; fix it
(and re-pin the protocol) before treating TRL throughput as comparable.

`hardware.cuda` is the harness venv's CUDA version, because the comparison
key requires one value across frameworks; the framework venv's CUDA runtime
is recorded as `hardware.framework_cuda`. Peak VRAM is the larger of the
shared evaluator's allocator peak and the training peak; verl reports a
trainer allocator peak, while OpenRLHF and NeMo-RL fall back to device-wide
`nvidia-smi` sampling (`peak_vram_source`), which includes vLLM's
preallocated KV cache, so VRAM is not comparable across those rows.

Framework environments (`benchmarks/adapters/framework_envs.py`; prints the
install script, `--execute` runs it):

| Framework | Environment | Est. disk | Est. install |
|---|---|---|---|
| verl | `uv venv` py3.12 at `/workspace/venvs/verl`, `verl[vllm,fsdp]==0.9.1` (torch 2.11, vLLM 0.24) | ~25 GB | ~15 min |
| OpenRLHF | `uv venv` py3.12 at `/workspace/venvs/openrlhf`, `openrlhf[vllm]==0.11.2` (DeepSpeed 0.19.6, vLLM 0.29, flash-attn 2.8.3) | ~30 GB | ~45 min (flash-attn may compile) |
| NeMo-RL | source checkout at `/workspace/frameworks/nemo-rl`, `uv sync --locked` (py3.13) + `prefetch_venvs.py` for the vLLM and DTensor-v2 workers | ~60 GB | ~40 min |

The harness venv keeps StateSet and `trl==1.12.0` (via `extra_pip`) for
`shootout.py`, the StateSet/TRL legs, and all evaluation. The framework
venvs pin torch 2.11 wheels, which need a host driver new enough for their
CUDA runtime; check `nvidia-smi` on the pod before a paid run.

```bash
# Local GPU box (after building the framework environments):
python benchmarks/adapters/framework_envs.py --execute
python benchmarks/shootout.py benchmarks/shootout_manifest_v6_five_frameworks.json \
  --preflight \
  --required-framework stateset-agents-gspo --required-framework trl \
  --required-framework verl --required-framework openrlhf \
  --required-framework nemo-rl \
  --output-dir benchmark_results/framework_comparison_v5/preflight

# RunPod: free plan first. Set harness_revision to the commit containing the
# adapters before --execute (the checked-in value is a placeholder).
python benchmarks/runpod_shootout.py benchmarks/runpod_shootout_v6_manifest.json
python benchmarks/runpod_shootout.py benchmarks/runpod_shootout_v6_manifest.json \
  --execute --confirm-max-cost-usd 45.0 \
  --output-dir benchmark_results/framework_comparison_v5/evidence
```

Cost estimate (planning, not measured): no v4 or v5 run has completed in this
tree, so per-run time is unknown. The v3 launcher budgeted at most 1.83 h per
1.5B/200-step run (11 h for 6 runs). Two greedy evaluations of 256 problems
with batch-1 `generate` are shared by every framework and likely take ~45-60
min of each run. Planning range: 1-2 GPU-hours per run, so 15 runs is 15-30
GPU-hours, plus ~2 h one-time environment install and cache prewarm: about
17-32 L40S-hours, roughly $15-28 at RunPod Secure $0.86/h. Per framework
(3 seeds): ~3-6 GPU-hours, ~$2.6-5.2. The launcher caps the pod at 48 h and
$45. A `--preflight` pass (one seed per framework, 5 runs) costs roughly a
third of that and is the recommended first spend. The environment install
runs inside one blocking SSH command during setup; a dropped session fails
the launch.

Status: the adapters are verified against the pinned upstream sources (config
keys, CLI flags, loss switches, reward hooks, log formats) and by
`tests/unit/test_competitor_adapters.py` with the subprocess and GPU mocked.
None has run end to end on a GPU yet. The least-verified pieces are the
NeMo-RL DTensor-v2 LoRA checkpoint layout (the adapter fails closed unless it
finds exactly one PEFT adapter) and the verl v1-trainer checkpoint path
consumed by `verl.model_merger`.

## Other strict evidence gates

```bash
# Matched algorithm evidence
python benchmarks/algorithm_comparison.py EVIDENCE --output-dir REPORT

# Complete 1/2/4/8-GPU matrices
python benchmarks/scaling_comparison.py EVIDENCE \
  --gpu-counts 1 2 4 8 --output-dir REPORT

# Validate, then run the provider-specific physical multi-node launcher
make benchmark-scaling-multi-node-contract
make benchmark-scaling-multi-node-run \
  MANIFEST=benchmarks/scaling_launcher_manifest.json OUTPUT_DIR=RESULTS

# Plan an immutable CUDA/PyTorch scaling image without building or pushing it
make benchmark-scaling-image-plan \
  BASE_IMAGE=registry/pytorch@sha256:<digest> \
  IMAGE=registry/stateset-scaling:0.54.0

# After review, build/push with maximal provenance + SBOM and retain the result
make benchmark-scaling-image-push \
  BASE_IMAGE=registry/pytorch@sha256:<digest> \
  IMAGE=registry/stateset-scaling:0.54.0 \
  CONFIRM_PUSH=registry/stateset-scaling:0.54.0 \
  OUTPUT=benchmark_results/scaling/image-attestation.json

# Concrete RunPod Pods adapter (copy, pin, and approve its explicit ceiling)
cp benchmarks/scaling_launcher_runpod.example.json runpod-scaling.json

# High-bandwidth CoreWeave/Nebius Kubernetes template
cp benchmarks/scaling_launcher_kubernetes.example.json kubernetes-scaling.json

# Worker, controller, and network recovery evidence
python benchmarks/reliability_evidence.py EVIDENCE --output REPORT.json

# Validate the multi-node collection contract without provisioning
make benchmark-distributed-async-contract

# Execute the two-node minimum, three-seed fault matrix plus 12-hour soak
make benchmark-distributed-async-run \
  MANIFEST=benchmarks/distributed_async_manifest.json \
  OUTPUT_DIR=benchmark_results/distributed_async

# Revalidate retained evidence independently
make benchmark-distributed-async-gate \
  INPUTS=benchmark_results/distributed_async/evidence \
  OUTPUT=benchmark_results/distributed_async/report.json

# Three-seed τ³-bench and BFCL V4 quality matrix (SWE-bench Verified optional)
make benchmark-agent-quality-contract
make benchmark-agent-quality-run \
  MANIFEST=benchmarks/agent_quality_manifest.json \
  OUTPUT_DIR=benchmark_results/agent_quality
make benchmark-agent-quality-gate \
  INPUTS=benchmark_results/agent_quality/evidence \
  OUTPUT=benchmark_results/agent_quality/report.json
```

Standard RunPod Pod global networking is documented at 100 Mbps. Its adapter
is suitable for validating orchestration and failure behavior, but is not
presented as a likely path to the A+ efficiency threshold; use high-bandwidth
cluster nodes for publication.

Once every measured bundle exists for one immutable release commit, run the
single leadership decision gate:

```bash
make benchmark-a-plus-gate \
  FRAMEWORK_EVIDENCE=benchmark_results/framework_comparison/evidence \
  SCALING_EVIDENCE=benchmark_results/scaling/multi_node/evidence \
  SCALING_PROVIDER_EVIDENCE=benchmark_results/scaling/multi_node/kubernetes-provider \
  SCALING_IMAGE_ATTESTATION=benchmark_results/scaling/image-attestation.json \
  SCALING_BILLING_EVIDENCE=benchmark_results/scaling/multi_node/billing \
  RELIABILITY_EVIDENCE=benchmark_results/reliability/evidence \
  DISTRIBUTED_ASYNC_EVIDENCE=benchmark_results/distributed_async/evidence \
  PROVIDER_EVIDENCE=benchmark_results/provider_canaries/current \
  AGENT_QUALITY_EVIDENCE=benchmark_results/agent_quality/evidence \
  FLAGSHIP_MANIFEST=benchmarks/flagship_manifest.json \
  FLAGSHIP_EVIDENCE=benchmark_results/flagship_v1/evidence \
  RELEASE_READINESS=publish-readiness-summary.json \
  OUTPUT=benchmark_results/a_plus/report.json
```

The command cannot pass on the existing partial evidence. It requires five
matched framework implementations including native StateSet GSPO, complete
provider-derived cost, a lifecycle record for every scaling row, one
commit/version-bound digest-pinned scaling image built with maximal provenance
and an SBOM request, raw provider billing bound to every Kubernetes Job with
matrix-level cost per optimizer step, 70%
strong-scaling efficiency across at least two
physical nodes, zero-loss recovery, a 12-hour multi-node soak, fresh passing
River/RunPod/Fireworks/CoreWeave/Nebius canaries, the required standard agent
suites (plus any optional suite that was run), the 7–9B flagship, and publish readiness for the same commit.

### Agent-quality suite policy

The agent-quality gate (`agent_quality_evidence.py`) splits suites into:

- **Required** (`REQUIRED_SUITES`): `tau3-bench` and `bfcl-v4`. Every matrix
  must include both. They match the flagship's domain: a 7–8B
  customer-support / tool-use policy.
- **Optional** (`OPTIONAL_SUITES`): `swe-bench-verified`. Repository-level
  software engineering is out of domain for the flagship, so it is not
  required. A matrix may still include it.

No suite is exempt once it is present. Every suite in the evidence must pass
the same per-suite gate: at least `+0.03` mean improvement, a paired 95%
confidence bound above zero, the same seed set, and one shared trained
artifact across all suites. When the collection runner gates its own output, the
evidence must cover exactly the manifest roster. A suite that is run but fails
cannot be dropped from the published matrix. Unknown suite names are
rejected.

The example manifests list only the two required suites. To add SWE-bench
Verified, make the same three additions to your experiment copies:

1. Append a `{"name": "swe-bench-verified", "revision": ..., "split": "test",
   "command": [...]}` entry to `suites` in the manifest. The command is the
   same `paired_agent_harness.py run ...` argv used by the other suites.
2. Add a `swe-bench-verified` pipeline under
   `evaluation_config.official_suite_pipelines`. It runs a pinned coding agent
   that writes `{predictions}` and `{cost_records}`, then
   `python -m swebench.harness.run_evaluation --dataset_name
   princeton-nlp/SWE-bench_Verified --split {split} --predictions_path
   {predictions} --max_workers {parallelism} --run_id {policy} --report_dir
   {official_results}`. Set `results_path: "official/report"`,
   `cost_records_path: "official/costs.jsonl"`, and
   `artifact_paths: {"predictions": "official/predictions.jsonl"}`. A
   `command_timeout_seconds` of 43200 is typical.
3. Add a `swe-bench-verified` entry to the harness config. It has the same
   shape as the other suites, plus a pinned SWE-bench checkout as
   `repository_path`.

### Planned τ³-bench + BFCL V4 run (served base vs. trained adapter)

`agent_quality_manifest.json` and `agent_quality_harnesses.json` are the
concrete plan: Qwen/Qwen2.5-7B-Instruct at `a09a3545…` (the flagship base)
versus the trained StateSet adapter, both served by one vLLM OpenAI server.
Upstreams are pinned to τ³-bench v1.0.1 (`sierra-research/tau2-bench@fc0055dc`)
and BFCL V4 (`ShishirPatil/gorilla@6ea57973`, the latest BFCL commit; no
release tag exists). The manifest fails closed until the operator fills
`trained_policy.{model,revision,artifact_sha256}` and
`budget.max_total_cost_usd` (the planned worst case is 6 attempts x $30).

```bash
# 0. Pinned upstream checkouts + their environments (public downloads, no spend)
export STATESET_AQ_UPSTREAM_ROOT=/srv/stateset-aq/upstream
python benchmarks/agent_quality_upstream.py install \
  --harness-config benchmarks/agent_quality_harnesses.json \
  --root "$STATESET_AQ_UPSTREAM_ROOT" \
  --manifest benchmarks/agent_quality_manifest.json

# 1. Fill the manifest placeholders (digest = hash_artifact of the adapter
#    directory) and commit: measured runs reject a dirty harness tree.
python -c "import sys; from pathlib import Path; \
from benchmarks.agent_quality_evidence import hash_artifact; \
print(hash_artifact(Path(sys.argv[1])))" /path/to/adapter

# 2. Secrets are environment references only; the τ³ user simulator is
#    gpt-4.1-2025-04-14 (the upstream default), so it needs OPENAI_API_KEY.
export OPENAI_API_KEY=...
python benchmarks/agent_quality_endpoint.py check-env \
  --manifest benchmarks/agent_quality_manifest.json

# 3. Serve both policies, run the matrix against them, tear down.
python benchmarks/agent_quality_endpoint.py serve \
  --manifest benchmarks/agent_quality_manifest.json \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --base-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --adapter /path/to/adapter --workdir /srv/stateset-aq/endpoint \
  --gpu-hourly-usd 2.69 \
  --vllm-arg=--enable-auto-tool-choice --vllm-arg=--tool-call-parser \
  --vllm-arg=hermes --vllm-arg=--max-lora-rank --vllm-arg=32 \
  -- python benchmarks/run_agent_quality_matrix.py \
       benchmarks/agent_quality_manifest.json \
       --output-dir benchmark_results/agent_quality
```

`agent_quality_endpoint.py serve` launches `vllm serve` with the pinned
`--revision`, registers the adapter with `--enable-lora` (or merges it through
`serving_artifacts` with `--trained-mode merged`, on a second port), checks the
adapter digest against the manifest, runs the `serve-remote` greedy
adapter-effect probe, and writes an endpoint record (served names, revisions,
adapter digest, vLLM argv, GPU rate; the API key stays in a 0600 file). Every
official stage runs through its `exec` shim, which receives `{model}` and
`{model_revision}` (the `command-argument` binding), resolves the served policy,
re-verifies the revision marker, adapter digest and live `/v1/models`, fails
closed on missing `--require-env` secrets, exports the endpoint to litellm
(`HOSTED_VLLM_*`) and BFCL (`REMOTE_OPENAI_*`), and retains a secret-free
attestation. Its `meter` stage writes the per-task cost rows (endpoint wall
time x the declared GPU rate, split equally, plus τ³'s own user-simulator
cost), so both suites report `cost_source: local-meter`. The runner stops
scheduling attempts once metered spend reaches `budget.max_total_cost_usd`.

Known constraints of this plan: τ³ runs the `retail` domain, `base` split;
BFCL runs `single_turn,multi_turn` (the V4 agentic web-search category needs a
paid SerpAPI key and is excluded); BFCL has no seed, so its three seeds are
near-identical; and BFCL has no registered Qwen2.5 handler at the pinned
commit, so `bfcl_model_handler` uses the Qwen FC prompt handler
(`Qwen/Qwen3-8B-FC`) for both policies. Absolute BFCL numbers are therefore
not leaderboard-comparable; the paired difference is. The A+ gate rejects any
bundle whose suite revisions differ from these pins, so a stub dry run
(`tests/unit/test_agent_quality_dry_run.py`) can never pass it.

Start from `agent_quality_manifest.example.json` and
`agent_quality_harnesses.example.json`. The included
`adapters/paired_agent_harness.py` executes both policies shell-free against the
same clean upstream checkout at the manifest-pinned revision. Its built-in
`adapters/official_suite_pipeline.py` executes each suite's configured official
CLI stages, retains a revision-bound execution manifest and per-command logs,
rejects stale or out-of-directory artifacts, and verifies that no command
modifies the pinned checkout. It then converts official τ³-bench, BFCL V4, or
(optional) SWE-bench Verified artifacts into ordered JSONL records with a non-empty string
`task_id`, a boolean `success`, and a finite, non-negative `cost_usd`.

The example contains runnable official τ³ and BFCL CLI stages. Replace its
explicit user-simulator model, provider-meter exporter, coding-agent command
(if you add SWE-bench Verified), model revisions, and checkout paths with immutable experiment values. The
meter exporter must write exactly one measured `task_id`/`cost_usd` JSONL row
per BFCL or SWE task; the SWE coding agent must also write the official
predictions JSONL. These are experiment inputs, not framework driver code.
The paired adapter rejects duplicate or reordered task sets and derives both
scores, success counts, task digest, and total cost from the normalized records
rather than trusting aggregate values reported by an upstream command.

The collection runner additionally binds both policy revisions, the trained
artifact digest, the canonical evaluation configuration, raw result artifacts,
and measured cost provenance. It accounts for every failed attempt before the
publication gate evaluates the complete matrix. Schemas and collection guidance
live under `benchmark_results/`. These tools fail closed on missing seeds,
mismatched protocols, estimated results, and incomplete provenance.

The pipeline invokes the same normalizers directly. They also remain available
as standalone commands for inspecting or importing existing official runs:

```bash
# τ³-bench uses its embedded per-simulation agent_cost by default.
python benchmarks/adapters/official_result_normalizer.py \
  --suite tau3-bench --results TAU_RESULTS --output tasks.jsonl

# BFCL V4 joins generated IDs to official incorrect-case score files.
python benchmarks/adapters/official_result_normalizer.py \
  --suite bfcl-v4 --results BFCL_RESULTS --scores BFCL_SCORES \
  --cost-records costs.jsonl --output tasks.jsonl

# SWE-bench requires the complete official schema-v2 run report.
python benchmarks/adapters/official_result_normalizer.py \
  --suite swe-bench-verified --results SWE_RESULTS \
  --cost-records costs.jsonl --output tasks.jsonl
```

`costs.jsonl` must contain exactly one `task_id` and measured `cost_usd` for
every evaluated task. Partial SWE-bench reports, BFCL category-count drift,
duplicate trials, missing costs, and extra cost IDs are rejected.

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

See [A_PLUS_RUNBOOK.md](A_PLUS_RUNBOOK.md) for the ordered, costed path to a passing `a_plus_gate.py`.
