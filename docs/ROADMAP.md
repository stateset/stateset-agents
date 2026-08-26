# StateSet Agents roadmap

Current release line: **v0.42.x (beta)**.

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

See [`RELEASE_EVIDENCE.md`](RELEASE_EVIDENCE.md) for the exact current claims.

## A+ evidence gates

These are the blockers between the current strong beta and an independently
defensible A+ rating.

### Release and provider completion

- [ ] Publish the current release to PyPI through a working trusted publisher
- [ ] Live-verify Fireworks training, artifact retrieval, serving, and cleanup
- [ ] Keep tag-triggered River, RunPod, and Fireworks canaries green
- [ ] Publish signed/SLSA provenance for Python and container artifacts

### Comparative evidence

- [ ] Complete the 8B, three-seed flagship multi-turn benchmark
- [ ] Run matched StateSet versus TRL, verl, NeMo RL, and OpenRLHF comparisons
- [ ] Complete matched GRPO/GSPO/DAPO/VAPO/GEPO comparisons
- [ ] Publish negative/null runs alongside positive results
- [ ] Obtain at least one independent reproduction outside StateSet

### Scale and reliability

- [ ] Measure 1/2/4/8-GPU throughput, utilization, memory, and scaling efficiency
- [ ] Verify multi-node training on a retained, reproducible configuration
- [x] Add a measured-evidence gate for worker, network, and controller recovery
- [ ] Execute and retain the complete three-seed fault-injection matrix
- [ ] Report cost per rollout, training step, and measured eval improvement
- [ ] Add soak tests for long-running remote training and serving sessions

### Stable product surface

- [ ] Freeze a v1 public API and publish compatibility/migration guarantees
- [ ] Graduate beta components using explicit per-component maturity criteria
- [ ] Complete standard agent benchmarks for tool use, long horizon, and SWE tasks
- [ ] Establish public security response SLAs and third-party review

## Next implementation sequence

1. Execute the three-seed StateSet/upstream-TRL manifest, then extend the same
   neutral adapter contract to verl, NeMo RL, and OpenRLHF.
2. Execute the existing scaling and recovery matrices and retain partial and
   failed runs as auditable evidence.
3. Configure PyPI trusted publishing and Fireworks credentials externally;
   rerun the existing release/provider workflows without code-side exceptions.
4. Promote only gates backed by retained evidence into release claims.

## Contribution policy

New roadmap work should include deterministic unit tests and, where applicable,
an integration or live-evidence path. Performance changes need before/after raw
artifacts on identical configurations. Feature requests and RFCs belong in the
GitHub issue tracker.

Last updated: 2026-08-26.
