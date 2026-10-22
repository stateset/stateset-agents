# StateSet Agents roadmap

Current tagged version: **v0.68.0 (strong beta; publication blocked)**.

The roadmap is evidence-driven. A checked box means the behavior exists and
has retained tests or live evidence; an unchecked box is not a product claim.

## Current foundation

- [x] GRPO, GSPO, DAPO, GEPO, VAPO, PPO, offline RL, and RLAIF surfaces
- [x] Single-turn and multi-turn environments, rewards, trajectories, and evals
- [x] QLoRA/SFT and real CUDA GSPO proof on RunPod
- [x] River remote-autograd training, sampling, and improvement loops
- [x] RunPod training, artifact retrieval, serving, leases, and cleanup controls
- [x] Fireworks adapter with durable job reconnection and artifact handling
- [x] FastAPI/OpenAI-compatible serving, observability, and deployment assets
- [x] Python 3.10–3.13 packaging and a growing deterministic test suite
- [x] Provenance-enforced measured algorithm/framework comparison schemas
- [x] Neutral StateSet/upstream-TRL runner with config attestation and artifact hashing
- [x] Strict 1/2/4/8-GPU scaling and three-scenario recovery validators
- [x] Versioned training-backend protocol with semantic digests, capability
  checks, secret-free requests, and shell-free external adapter execution
- [x] Fail-closed OpenRLHF PPO/GRPO/GSPO adapter with immutable inputs and
  content-pinned reward/agent code (live GPU conformance remains open)
- [x] Fail-closed verl PPO/GRPO adapter with immutable inputs, pinned reward
  code, explicit Hydra overrides, and forced checkpoint output (live GPU
  conformance remains open)
- [x] Fail-closed NeMo RL GRPO adapter with immutable inputs, an exact built-in
  math reward contract, pinned source/version checks, and forced checkpoint
  output (live GPU conformance remains open)
- [x] Strict external-backend conformance runner with immutable manifests,
  exact GPU identity, artifact hashing, retained failures, and non-overwriting
  evidence output
- [x] Portable conformance artifacts and a full-roster gate that revalidates
  checkpoint bytes and rejects missing, duplicate, or semantically drifted
  NeMo RL, verl, and OpenRLHF evidence
- [x] Immutable conformance execution envelopes with digest-pinned images,
  exact GPU topology/disk, workload and total-lifetime bounds, spend ceilings,
  and failure accounting
- [x] Fail-closed RunPod conformance launcher with zero-cost catalog planning,
  explicit spend confirmation, authoritative price recheck, remote
  self-destruct, local recovery lease, cost ledger, and portable retrieval

See [`RELEASE_EVIDENCE.md`](RELEASE_EVIDENCE.md) for the exact current claims.

## A+ evidence gates

The unreleased checkout has retained [2026-10-15 local verification](../benchmark_results/training_contracts/20261015/evaluation-isolation/README.md)
for evaluation RNG isolation, complete failure accounting, validation/test
separation and checkpoint-selection receipts. Its full suite passed 8,615 tests
with 57 skipped, and 217 combined contracts passed against the rebuilt wheel.
The [remaining evidence requirements](../benchmark_results/training_contracts/20261015/evaluation-isolation/a_plus_readiness.json)
remain open; these engineering checks do not close the live A+ gates below.

These are the blockers between the current strong beta and an independently
defensible A+ rating.

StateSet is not pursuing "most algorithms" as the definition of leadership.
The target is the best trace-to-policy control plane for deployed agents, with
specialized engines interchangeable behind stable trajectory, reward,
evaluation, and lineage contracts.

### Release and provider completion

- [ ] Publish the current release to PyPI through the release workflow
- [ ] Live-verify Fireworks training, artifact retrieval, serving, and cleanup
- [ ] Keep tag-triggered River, RunPod, and Fireworks canaries green
- [ ] Publish and verify signed provenance for the current Python distributions
- [ ] Publish provenance and SBOMs for container artifacts
- [x] Publish a machine-readable model/provider tier and certification matrix
- [x] Add non-billable RunPod resource planning with measured-vs-estimated labels
- [x] Require an explicit spend ceiling before estimated frontier plans execute
- [ ] Promote Qwen3.8-Flash-Next after retained train/eval/serve/cleanup proof
- [ ] Promote GLM-5.3-Flash after the same retained proof
- [ ] Live-verify Modal transport with a bounded ephemeral GPU job
- [x] Add approval-gated, cleanup-enforced Modal and Fireworks live workflows
- [x] Complete Modal dataset upload, isolated input/output Volume paths,
  output commit, named-Secret/region injection, and multi-GPU resource wiring

Earlier releases retain successful Python publication and attestations in
[`RELEASE_EVIDENCE.md`](RELEASE_EVIDENCE.md). The v0.68.0 publish workflow failed
before publication; its PyPI upload, GitHub release, and distribution
attestations remain unverified. Post-tag fixes do not change the immutable tag.

The tag pipeline generates GitHub build attestations for Python artifacts
and requests maximal provenance plus SBOMs for both container images. This gate
remains unchecked until a tagged run retains and verifies those attestations.
The scaling benchmark now has a separate digest-pinned BuildKit path and the
A+ gate binds its source/image attestation to every scaling lifecycle record;
the checkbox remains open until a real registry push retains the resulting
provenance and SBOM artifacts.

### Comparative evidence

- [ ] Recover and independently reproduce the Qwen3.5-9B v2 preflight-5
  reward/judge divergence; run v3 on fresh validation/test partitions with
  baseline-preserving judge checkpoint selection before a paid three-seed run.
  See [reward recovery](RUNPOD_REWARD_RECOVERY.md) for reported versus recovered
  evidence, implemented safeguards and the remaining paid-run gates.
- [ ] Complete the 8B, three-seed flagship multi-turn benchmark
- [x] Add a fail-closed flagship collector with immutable model, dataset, and
  judge revisions, provider cost ceilings, complete seed accounting, artifact
  hashes, external timing, judge stability, and paired significance
- [ ] Run matched StateSet versus TRL, verl, NeMo RL, and OpenRLHF comparisons
- [x] Complete matched GRPO/GSPO/DAPO/VAPO/GEPO comparisons
- [x] Publish negative/null runs alongside positive results
- [ ] Obtain at least one independent reproduction outside StateSet

### Scale and reliability

- [x] Match native DDP token-path updates to the combined-batch objective with
  global trajectory/token counts, separately synchronized entropy, complete and
  partial accumulation, and repeated inner updates; verify real CPU/Gloo peers,
  registered buffers and bounded, precision-safe gradient reductions through
  [288 source/wheel scenarios](../benchmark_results/training_contracts/20261015/native-ddp-weighting/summary.json)
- [x] Report native DDP loss/entropy/KL/ratio diagnostics using global counts,
  stable advantage moments, exact control counters and coordinated schema
  rejection; retain [294 source/wheel scenarios](../benchmark_results/training_contracts/20261015/native-ddp-metrics/summary.json)
- [x] Verify cross-rank batch-normalized and global reward-mean advantage
  construction, including frozen baselines across inner updates; retain
  [48 source/wheel population scenarios and 12 rejection/recovery cases](../benchmark_results/training_contracts/20261015/native-ddp-advantages/summary.json)
- [x] Verify native learner preflight agreement on loss configuration and update
  schedules, and coordinated rejection of invalid sampled-token metadata before
  old-policy or reference-model scoring; retain
  [40 source/wheel admission rejection and recovery cases](../benchmark_results/training_contracts/20261015/native-ddp-preflight/summary.json)
- [x] Compare initial/invalidated DDP replica values and scheduler callable
  bindings before scoring; retain
  [32 completed-checkpoint startup scenarios](../benchmark_results/training_contracts/20261015/native-ddp-replica-consensus/summary.json)
- [x] Verify checkpoint-bound policy/optimizer tensor agreement and scheduler
  implementation identity across native CPU/Gloo learner ranks before resumed
  updates; retain [source/wheel checkpoint evidence](../benchmark_results/training_contracts/20261015/native-ddp-checkpoint/summary.json)
- [x] Replace native DDP's legacy parallel writer with atomic shared state and
  rank-local gradients/banks, RNGs, buffers and estimator history; verify
  [84 two-rank resume and 14 rejection/recovery scenarios](../benchmark_results/training_contracts/20261015/native-ddp-checkpoint/README.md)
  per peer against source and installed wheel on CPU/Gloo
- [x] Weight native DDP evaluation by sample counts and agree on best-checkpoint
  decisions across unequal/empty local partitions; preserve training state and
  verify [28 source/wheel evaluation scenarios per peer](../benchmark_results/training_contracts/20261015/native-ddp-evaluation/README.md)
- [x] Consolidate native DDP Fisher/teacher memory with global accepted-example
  weights, staged commits and preserved local learning state; close task windows,
  budget their updates and retain completed-boundary receipts; verify
  [46 source/wheel task-memory and training/resume scenarios per peer](../benchmark_results/training_contracts/20261015/native-ddp-continual/README.md)
- [x] Coordinate native DDP local generation/callback failures and empty token
  batches; verify initial/completed resumes and interrupted-task episode cursors
  with [131 population and 17 loop/recovery scenarios per peer against source and wheel](../benchmark_results/training_contracts/20261015/native-ddp-empty-loop/README.md)
- [x] Coordinate and checkpoint native diagnostic guard history; preserve empty
  no-ops and callback references, reject divergent controls before learner work,
  and verify [8 source/wheel recovery scenarios per peer](../benchmark_results/training_contracts/20261015/native-ddp-diagnostics/README.md)
- [x] Report optimizer updates committed by task-boundary and final accumulation
  flushes, share committed-update cadence and preserve early stopping;
  verify [16 source/wheel reporting and recovery scenarios per peer](../benchmark_results/training_contracts/20261016/reward-recovery/README.md)
  with real schedulers and correctly matched initial reference architectures
- [x] Weight native episode callback rewards and training-log rollout summaries
  across unequal/empty workers; separate new and replay populations and preserve
  exact global medians with [21 source/wheel reporting scenarios per peer](../benchmark_results/training_contracts/20261015/native-ddp-rollout-metrics/README.md)
- [x] Bind native DDP model/runtime implementation revisions across workers and
  checkpoint restore; verify fresh source reads, portable wheel identities,
  47 admission rejection/recovery scenarios and 99 checkpoint scenarios per peer
  against source and installed wheel on CPU/Gloo; retain
  [implementation evidence](../benchmark_results/training_contracts/20261017/native-ddp-implementation/summary.json)
- [ ] Verify the same native RL updates under NCCL and multi-rank Accelerate,
  including partial-window optimizer flush and checkpoint recovery
- [x] Bind Muse/frontier text SFT validation, source hashes and training to one
  captured read per split, with file-replacement race regressions
- [x] Verify tiny serialized GLM FP8 attention/expert dequantization, BF16
  adapter optimizer updates, frozen weights and export/reload on CPU
- [x] Require rank agreement before Muse/frontier weight loading on source-bound
  datasets, tokenization, labels, settings, runtime versions and StateSet code;
  verify agreement, rejection and recovery with two actual CPU/Gloo ranks
- [ ] Verify NCCL consensus and adapter synchronization under actual load-time
  FSDP/EP execution on GPUs

- [x] Measure 1/2/4/8-GPU throughput, memory, and weak/strong scaling efficiency
- [x] Define native policy-versioned async queue and runtime, staleness bounds,
  importance-correction evidence, publication ordering, backpressure, failure
  propagation, and audit counters
- [x] Define transport-neutral remote worker leases, heartbeats, generation
  fencing, policy-exact admission, health counters, and checkpoint recovery
- [x] Expose the control plane through authenticated, principal-isolated HTTP
  routes with strict schemas, bounded bodies, and stable failure semantics
- [x] Bind policy versions and rollouts to content-addressed weight artifacts
  with atomic publication, local verification, and restart recovery
- [x] Add a fail-closed multi-node async evidence contract covering a 12-hour
  soak, fault recovery, policy lag, weight-sync tail latency, integrity,
  throughput, cleanup, and cost per accepted rollout
- [x] Add a shell-free multi-node collection runner with immutable manifests,
  external wall-time verification, rotated fault order, retained artifacts,
  complete failure accounting, and automatic publication-gate execution
- [ ] Verify multi-node training on a retained, reproducible configuration
- [x] Add a measured-evidence gate for worker, network, and controller recovery
- [ ] Execute and retain the complete live three-seed fault-injection matrix
- [ ] Report cost per rollout, training step, and measured eval improvement
- [ ] Add soak tests for long-running remote training and serving sessions

The implementation now calculates cost per accepted rollout for distributed
async evidence, cost per successful episode for agent suites, and
raw-provider-exported cost per measured optimizer step for scaling. The item
remains open until the live matrices populate all three metrics alongside
measured evaluation improvement.

### Stable product surface

- [x] Freeze a v1 public API and publish compatibility/migration guarantees
- [x] Govern beta graduation with explicit, machine-validated per-component
  maturity criteria and retained evidence references
- [ ] Complete standard agent benchmarks for tool use and long horizon (SWE tasks
  optional for the customer-support / tool-use flagship)
- [x] Add a fail-closed τ³-bench/BFCL V4 evidence gate (SWE-bench Verified
  optional, gated identically when present) with paired significance, immutable revisions, artifacts, and cost accounting
- [x] Add an executable, manifest-driven τ³-bench/BFCL V4/SWE collection runner
  that binds both policies, the trained artifact, paired task IDs, raw outputs,
  failures, and cost provenance before invoking the evidence gate
- [x] Add a shell-free paired upstream-harness adapter that verifies pinned,
  clean suite checkouts and derives scores and cost from identical ordered
  per-task records
- [x] Add a shell-free official-suite pipeline for multi-stage τ³/BFCL V4/SWE
  commands with fresh confined artifacts, checkout-mutation detection,
  revision-bound manifests, retained command logs, and direct normalization
- [x] Establish machine-validated public security response SLAs and safe harbor
- [ ] Complete and retain an independent third-party security review

## Leadership scorecard

The project may describe itself as a leading agent-RL framework only when the
following measurements are retained and reproducible. These are gates, not
estimated targets in marketing material.

| Dimension | Acceptance gate |
|---|---|
| Agent quality | Statistically supported held-out improvement at equal compute on the 8B flagship |
| Competitive efficiency | Throughput within 10% of the fastest matched framework and complete cost accounting |
| Distributed scale | At least 70% efficiency on a retained multi-node configuration |
| Reliability | No lost or duplicate optimizer updates across the declared recovery matrix and soak test |
| Agent capability | Reproducible long-horizon, tool-use, and SWE benchmark integrations |
| Portability | One StateSet experiment contract can execute through native and external training backends |
| Trust | Signed provenance, stable v1 compatibility policy, public security process, independent reproduction |

All locally machine-verifiable rows above must pass together through
`benchmarks/a_plus_gate.py`. The gate has no diagnostic opt-out: it revalidates
raw artifacts, requires native StateSet GSPO plus TRL/verl/NeMo RL/OpenRLHF,
enforces StateSet throughput within 10% of the fastest, requires strong scaling
across at least two physical nodes at 70% efficiency, permits zero lost or
duplicate optimizer updates, requires the 12-hour soak and five fresh provider
canaries, and binds every benchmark and publish-readiness result to one commit.

## Next implementation sequence

1. Execute the checked-in corrected 1.5B native-GSPO versus direct-TRL v4
   protocol and retain all six framework/seed pairs. Its non-billable dry run
   passes, and both sides are pinned to the same sequence-level importance
   ratio, asymmetric GSPO clipping, group-normalized rewards, and aggregation
   semantics. The legacy v3 directory retains two direct-TRL seeds, but exact
   manifest, harness, evidence-schema, and artifact binding prevents those rows
   from entering or resuming the corrected matrix.
2. Fill the immutable flagship manifest, run its bounded preflight, then run
   the complete 7–9B three-seed matrix and report quality, throughput, memory,
   and provider-derived cost per successful held-out episode.
3. Live-preflight the implemented NeMo RL, verl, and OpenRLHF adapters, then run
   the matched three-seed roster.
   The orchestrator enforces a required roster and accounts for every attempt
   after errors.
4. Complete native and external engine coverage on the versioned backend
   protocol so StateSet environments, rewards, evals, and lineage target TRL,
   verl, NeMo RL, or OpenRLHF execution without semantic drift.
5. Prove multi-node asynchronous rollout/training, weight synchronization,
   staleness bounds, checkpoint recovery, and a long-running soak.
6. Add reproducible tool-use and long-horizon agent suites (starting with
   the required τ³-bench/BFCL V4 suites, with SWE-bench Verified optional) and
   obtain an independent reproduction.
7. Bootstrap `@stateset/agents` with an npm automation token, configure
   Fireworks and Docker credentials, and remove the scoped PyPI-token fallback
   after the trusted-publisher identity is independently verified.

## Contribution policy

New roadmap work should include deterministic unit tests and, where applicable,
an integration or live-evidence path. Performance changes need before/after raw
artifacts on identical configurations. Feature requests and RFCs belong in the
GitHub issue tracker.

Last updated: 2026-10-15.
