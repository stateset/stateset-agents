# StateSet Agents roadmap

Current release line: **v0.47.x (beta)**.

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
- [x] Python 3.10–3.13 packaging and a 4,600+ test default suite
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

These are the blockers between the current strong beta and an independently
defensible A+ rating.

StateSet is not pursuing "most algorithms" as the definition of leadership.
The target is the best trace-to-policy control plane for deployed agents, with
specialized engines interchangeable behind stable trajectory, reward,
evaluation, and lineage contracts.

### Release and provider completion

- [ ] Publish the current release to PyPI through a working trusted publisher
- [ ] Live-verify Fireworks training, artifact retrieval, serving, and cleanup
- [ ] Keep tag-triggered River, RunPod, and Fireworks canaries green
- [x] Publish signed provenance for Python distributions
- [ ] Publish provenance and SBOMs for container artifacts
- [x] Publish a machine-readable model/provider tier and certification matrix
- [x] Add non-billable RunPod resource planning with measured-vs-estimated labels
- [x] Require an explicit spend ceiling before estimated frontier plans execute
- [ ] Promote Qwen3.8-Flash-Next after retained train/eval/serve/cleanup proof
- [ ] Promote GLM-5.3-Flash after the same retained proof
- [ ] Live-verify Modal transport with a bounded ephemeral GPU job
- [x] Add approval-gated, cleanup-enforced Modal and Fireworks live workflows

The tag pipeline now generates GitHub build attestations for Python artifacts
and requests maximal provenance plus SBOMs for both container images. This gate
remains unchecked until a tagged run retains and verifies those attestations.

### Comparative evidence

- [ ] Complete the 8B, three-seed flagship multi-turn benchmark
- [ ] Run matched StateSet versus TRL, verl, NeMo RL, and OpenRLHF comparisons
- [x] Complete matched GRPO/GSPO/DAPO/VAPO/GEPO comparisons
- [x] Publish negative/null runs alongside positive results
- [ ] Obtain at least one independent reproduction outside StateSet

### Scale and reliability

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
- [ ] Verify multi-node training on a retained, reproducible configuration
- [x] Add a measured-evidence gate for worker, network, and controller recovery
- [x] Execute and retain the complete three-seed fault-injection matrix
- [ ] Report cost per rollout, training step, and measured eval improvement
- [ ] Add soak tests for long-running remote training and serving sessions

### Stable product surface

- [ ] Freeze a v1 public API and publish compatibility/migration guarantees
- [ ] Graduate beta components using explicit per-component maturity criteria
- [ ] Complete standard agent benchmarks for tool use, long horizon, and SWE tasks
- [x] Add a fail-closed tau-bench/BFCL/SWE-bench Verified evidence gate with
  paired significance, immutable revisions, artifacts, and cost accounting
- [ ] Establish public security response SLAs and third-party review

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

## Next implementation sequence

1. Merge the v0.47.0 release commit into `master`, configure npm trusted
   publishing (or its scoped-token fallback), and configure Fireworks and
   Docker credentials so every tag job completes green.
2. Live-preflight the implemented NeMo RL, verl, and OpenRLHF adapters, then run
   the matched three-seed roster.
   The orchestrator enforces a required roster and accounts for every attempt
   after errors.
3. Run the 8B multi-turn flagship and report quality, throughput, memory, and
   cost per successful held-out episode.
4. Complete native and external engine coverage on the versioned backend
   protocol so StateSet environments, rewards, evals, and lineage target TRL,
   verl, NeMo RL, or OpenRLHF execution without semantic drift.
5. Prove multi-node asynchronous rollout/training, weight synchronization,
   staleness bounds, checkpoint recovery, and a long-running soak.
6. Add reproducible tool-use and long-horizon agent suites (starting with
   tau-bench/BFCL and a SWE task) and obtain an independent reproduction.
7. Configure PyPI trusted publishing and rerun the release workflow without
   its scoped-token fallback.

## Contribution policy

New roadmap work should include deterministic unit tests and, where applicable,
an integration or live-evidence path. Performance changes need before/after raw
artifacts on identical configurations. Feature requests and RFCs belong in the
GitHub issue tracker.

Last updated: 2026-08-31.
