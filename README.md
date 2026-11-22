<div align="center">

# StateSet Agents

**The RL improvement control plane for deployed AI agents.**

> **Renaming to TraceForge RL (Phase 1, non-breaking):** the new
> `import traceforge_rl` package and `traceforge-rl` CLI are available
> alongside `stateset_agents` / `stateset-agents`, which keep working
> unchanged. See the phased plan and deprecation timeline in
> [docs/RENAME_TRACEFORGE.md](docs/RENAME_TRACEFORGE.md).

[![PyPI version](https://img.shields.io/pypi/v/stateset-agents.svg)](https://pypi.org/project/stateset-agents/)
[![CI](https://github.com/stateset/stateset-agents/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/stateset/stateset-agents/actions/workflows/ci.yml)
[![Security](https://github.com/stateset/stateset-agents/actions/workflows/security.yml/badge.svg?branch=master)](https://github.com/stateset/stateset-agents/actions/workflows/security.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: BUSL-1.1](https://img.shields.io/badge/License-BUSL--1.1-green.svg)](LICENSE)
[![Whitepaper v0.13.4](https://img.shields.io/badge/whitepaper-v0.13.4-blue)](docs/WHITEPAPER.md)
[![First-party result](https://img.shields.io/badge/§11.7-judge%20%2B0.079%20%E2%9C%93-brightgreen)](benchmark_results/whitepaper_v1/customer_support_3seed_judge_qwen25_05b_instruct.json)

</div>

**v0.74.0 is on PyPI.** It adds finite-sample bounds for CQL proposal sampling
and fixes how PPO and GRPO use recorded sampling probabilities.
[Release highlights](#whats-new) · [Installation](#installation) · [How to evaluate StateSet](#how-to-evaluate-stateset)

- **If you run a support agent**, grade sample conversations locally and inspect
  which replies become training data. Then try the same steps on a redacted
  conversation from your own agent. [Quickstart](QUICKSTART.md)
- **If you already use Hugging Face TRL**, score two refund replies on CPU
  and use the same reward function in your existing GRPO trainer.
  [Try the TRL example](docs/COMPARISONS.md#try-a-stateset-reward-in-an-existing-trl-run)

The v0.74.0 support-agent pilot runs through December 19, 2026. Try either
workflow and share which step you reached and what helped or blocked you. Keep transcripts
and customer details local. [Join the 30-day pilot](https://github.com/stateset/stateset-agents/issues/136)

## Fan out RL training to StateSet Swarm

Give three GRPO seeds the same 20-update budget. Each sandbox trains a policy
for a synthetic refund workflow, evaluates it on 128 cases, and reloads its
checkpoint. The coordinator collects the three results on a local Git branch.

From this checkout, with the host CLI installed and `COORD` pointing to the
built StateSet Sandbox coordinator, run the following. Planning is free.
The `node` command launches three hosted tasks with a $1 model budget each,
plus sandbox compute. [Setup and local replay](examples/swarm/README.md)

```bash
RUN=$(mktemp -d "${TMPDIR:-/tmp}/stateset-rl-XXXXXXXX")
stateset-agents swarm plan rl-seeds > "$RUN.plan.json"
node "$COORD" run "$RUN.plan.json" "$RUN" --max-parallel 3 --publish
git worktree add --detach "$RUN-results" "swarm/${RUN##*/}"
stateset-agents swarm report rl-seeds --results "$RUN-results/swarm-results"
```

In a local CPU replay, successful cases went from **67 to 86** for seed 42,
**63 to 89** for seed 43, and **64 to 41** for seed 44. All three checkpoints
reloaded. The `starter` recipe pairs one RL run with a verified weight edit.
[Per-seed results](examples/swarm/evidence/local-cpu/) · [Swarm recipe library](examples/swarm/README.md)

---

StateSet Agents is a production‑oriented RL stack for training and serving LLM‑backed agents that improve through **multi‑turn interaction**. The library provides:

- An **improvement loop** that turns your agent's own logs into a better agent: `ingest` → `improve` (grade → curate) → fine‑tune.
- Async‑first **agent APIs** (`MultiTurnAgent`, `ToolAgent`) with Hugging Face and stub backends.
- **Environments** for conversational and task‑oriented episodes.
- **Trajectories** and value/advantage utilities tailored to dialogue.
- Composable **reward functions** (heuristic, domain, multi‑objective, neural, LLM‑judge, and a proof‑backed [StateSet NSR verifier](docs/NSR_INTEGRATION.md) for RLVR‑style verifiable rewards).
- A family of **group‑based policy‑optimization trainers** (GRPO, GSPO, GEPO, DAPO, VAPO), plus PPO and standalone RLAIF. RLAIF optimizes reward-weighted response log likelihood. The declarative [`PolicyObjective`](docs/OBJECTIVES.md) (GRPO, Dr. GRPO, BNPO, DAPO, GSPO, GSPO-token, GEPO, RLOO, REINFORCE++-baseline, CISPO, PPO) is verified against loop references, Hypothesis invariants, and TRL 1.12.
- **Offline RL algorithms** for learning from logged conversations (BCQ, BEAR, CQL, IQL, Decision Transformer).
- **Sim‑to‑Real transfer** for training in simulation and deploying to real users (domain randomization, system identification, progressive transfer).
- **Continual learning + long‑term planning** utilities (replay/LwF/EWC, plan context injection).
- An **MCP server** so Claude Code/Desktop — or any MCP client — can drive the loop conversationally.
- Optional **performance layers** (vLLM generation, Rust acceleration, distributed training, HPO, FastAPI service).
- A native, policy-versioned [asynchronous rollout control plane](docs/ASYNC_ROLLOUTS.md)
  with bounded backpressure, explicit staleness limits, sampler log-probability
  evidence, importance correction, auditable drop/reject counters, and a
  failure-safe multi-producer/serialized-learner runtime.
- A versioned, capability-checked [training backend protocol](docs/TRAINING_BACKENDS.md)
  for delegating the same experiment to specialized engines without silently
  changing its model, data, environment, reward, or algorithm configuration.
- A fail-closed OpenRLHF adapter for version-pinned PPO, GRPO, and GSPO engine
  execution; live GPU conformance remains an explicit evidence gate.
- A fail-closed verl adapter for version-pinned PPO/GRPO execution with
  content-addressed inputs and explicit Hydra configuration; live GPU
  conformance remains an explicit evidence gate.
- A fail-closed NeMo RL adapter for version-pinned, distributed math GRPO from
  a pinned source checkout, with immutable model/data inputs and forced
  reusable checkpoints; live GPU conformance remains an explicit evidence gate.
- A portable external-backend conformance gate that rehashes copied checkpoint
  bytes and requires one semantically aligned NeMo RL, OpenRLHF, and verl record
  before the adapter roster can be called complete.
- Immutable conformance execution envelopes that bind provider tier,
  digest-pinned images, exact GPU topology/disk, workload and total pod-lifetime
  bounds, and spend ceilings. The RunPod launcher performs a free public-catalog
  plan, requires explicit spend confirmation, rechecks the authoritative pod
  price after allocation, arms a remote self-destruct, and always retains a
  cost record while keeping a cleanup lease until termination is confirmed.

StateSet's focus is the complete improvement loop: preserve the agent's
multi-turn production traces, turn them into reproducible rewards and
evaluations, train through the most appropriate backend, and deploy the result
with lineage. Multi-turn generation is now available in several trainer
libraries; the differentiator here is keeping that whole loop coherent and
measurable.

For the Qwen3.5-9B RunPod reward regression, see the
[reward recovery investigation](docs/RUNPOD_REWARD_RECOVERY.md). It separates
reported results from recovered evidence and documents v3 rewards,
independent judge checkpoint selection and the requirements for another run.
The [2026-10-16 targeted acceptance](benchmark_results/training_contracts/20261016/reward-recovery/README.md)
passed 2,173 source regressions (seven skips), 906 installed-wheel regressions,
typing for 416 modules, and twelve two-worker CPU/Gloo contracts against source
and wheel. The three recovered reward examples are diagnostic evidence;
representative quality gains and the complete 200-pair re-score remain unverified.

### Verification evidence

The v0.74.0 [verification coverage](formal/README.md) lists the properties
checked for each algorithm. The results below apply to the named release or experiment.
The [improvement pipeline guide](docs/IMPROVEMENT_PIPELINE.md) describes logged
policy training, data governance, diagnostics, and production promotion.

The [2026-10-15 local verification](benchmark_results/training_contracts/20261015/evaluation-isolation/summary.json)
of the unreleased checkout based on `v0.66.1` passed **8,615 tests**, with
**57 skipped**, and typing for **386 modules**. Its wheel executed real SFT and
IQL policy updates, export, reload, local inference, and native sampled RL across
three seeds with gradient accumulation and verified checkpoint recovery.
Independent references cover uneven prompt groups, accumulated updates,
partial windows, AMP recovery, and clipping order. The retained
reports identify the exact wheel and package sources. All 217 combined evaluation,
recovery and Fisher-estimation cases also passed against that wheel, covering
isolated sampling, failed-attempt accounting, validation/test separation, reward
statistics, replay RNGs, frozen task references, EWC state, independent causal
gradients and low-precision statistics. An earlier wheel's [RTX A5000 device check](benchmark_results/training_contracts/20261014/native-offline-device/README.md)
verified actual policy updates, exact checkpoint recovery and CUDA inference
after reload. These synthetic checks do not establish representative quality,
GPU backend parity, or production approval.

The isolated `v0.58.0` release tree completed its local Linux Python 3.12 suite on
**2026-10-07** with **6,020 passed, 18 skipped, and 65.00% coverage**. Its release
checks passed lint, typing, API compatibility, governance, security scanning,
distribution metadata, and isolated wheel installation. The
[tag publication workflow](https://github.com/stateset/stateset-agents/actions/runs/37660763697)
published Python distributions, documentation, and the GitHub release. The npm
client was unpublished at the time. The broader Linux Python 3.10–3.13 and
Windows Python 3.10/3.13 matrix evidence remains historical (`v0.54.0`), and
should not be read as certification of every newer checkpoint or platform.
See the retained
[proof ledger](docs/PROOFS.md) for the evidence behind individual training,
serving, and provider claims.

- **RL objectives:** every native trainer evaluates one declarative
  [`PolicyObjective`](docs/OBJECTIVES.md); the eleven presets are verified by
  explicit-loop references, Hypothesis invariants, a numeric pin against TRL
  1.12, and deterministic per-trainer goldens. The GRPO trainers train on the
  exact generated token ids with per-token ratios, KL, entropy, and optional
  PPO-style inner updates; before `v0.50.0` they recorded no log-probs and
  trained as unclipped REINFORCE on re-tokenised text.
- **Rollout engine synchronization.** `MultiTurnAgent.set_rollout_backend`
  samples through vLLM, the trainers push the policy's weights into the engine
  after every optimizer step, and `old_logprobs_source="sampler"` uses the
  recorded sampling probabilities at each visited prefix. Proven live on vLLM 0.28 / A40 for
  full weights and a LoRA adapter: the engine-vs-policy log-prob gap of the
  sampled tokens returns to bf16 noise after the sync (retained in
  [`benchmark_results/vllm_rollout_check/`](benchmark_results/vllm_rollout_check/README.md)).
- **Measured comparisons:** two retained three-seed A40 shootouts against
  direct TRL at 0.5B (a four-step and a 48-step protocol) establish
  throughput parity within run-to-run noise (StateSet's TRL-backed GRPO 7%
  slower than direct TRL) and **no learning-quality signal for either
  implementation at that scale** (see
  [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)). The native-GSPO rows of those
  matrices are **withdrawn**: a harness bug trained them on a placeholder
  prompt with a reward that never saw the ground truth (identically zero at
  every step). Versions `0.53.0` and `0.54.0` fixed the prompt/reward wiring,
  on-policy reference log-probs, batched sampling, inference-mode generation,
  and zero-signal detection. The legacy 1.5B v3 matrix remains incomplete:
  two direct-TRL seeds are retained, but no corrected native-GSPO seed is yet
  publishable. Unreleased resume hardening now binds evidence to the exact
  manifest and harness, so the corrected protocol will rerun every pair in a
  fresh matrix rather than mix those legacy rows. No quality superiority is
  claimed.
- **Auditable rented-compute cost:** schema-v2 comparison rows are bound to one
  exact shootout manifest. The publication gate can require every RunPod pod
  lifecycle record, verify its post-allocation rate, observed lifetime,
  manifest/harness identity, unique pod ID, and confirmed termination, and
  report the total provider-derived estimate alongside quality and throughput.
- **One fail-closed A+ decision:** `benchmarks/a_plus_gate.py` revalidates the
  competitive, strong multi-node scaling, recovery, 12-hour soak, five-provider,
  standard-agent, 7–9B flagship, and publish-readiness bundles together and
  requires every harness result to match one immutable release commit.
- **Source-bound provider canaries:** schema-v2 River, RunPod, Fireworks,
  CoreWeave, and Nebius reports now carry the exact clean harness commit and
  framework version. Historical schema-v1 or unrelated fresh canaries cannot
  satisfy the A+ gate.
- **Current Python candidate:** `stateset-agents==0.77.0`, publication pending.
- **Latest verified PyPI release:** `stateset-agents==0.74.0`. The wheel and source
  distribution on PyPI match the SHA-256 digests of the locally verified builds.
  [Published artifacts](https://pypi.org/project/stateset-agents/0.74.0/#files)
- **Node client.** npm lists `0.74.0` as the `latest` version of
  `@stateset/agents`. [npm package](https://www.npmjs.com/package/@stateset/agents/v/0.74.0) · [Installation](#nodejs-api-client)
- **Live providers:** RunPod and River passed the `v0.54.0` tag canaries and
  have retained end-to-end evidence. Fireworks failed closed without
  `FIREWORKS_API_KEY` and `FIREWORKS_ACCOUNT_ID`, creating no billable
  resources. Container publication likewise remains owner-config blocked.
  CoreWeave CKS/Dedicated Inference and Nebius Serverless AI are fully
  integrated and unit-pinned; their live certification remains open alongside
  Fireworks lifecycle and Modal transport. None are promoted automatically.
- **Frontier models:** GLM-5.3-Flash and Qwen3.8-Flash-Next are integrated and
  smoke-tested; their production certification still requires retained live
  training and serving evidence.
- **Release governance:** `make release-governance` validates the canonical
  [component maturity](contracts/component_maturity_v1.json) and
  [security response](contracts/security_response_v1.json) contracts. CI and
  package publication fail closed if evidence, limitations, graduation
  criteria, stable API references, or response targets drift.
- **Standard agent benchmarks:** a manifest-driven τ³-bench, BFCL V4, and
  SWE-bench Verified runner now executes paired base-versus-trained matrices,
  drives multi-stage official suite commands without a shell, verifies clean
  pinned upstream checkouts and fresh confined artifacts, derives scores from
  identical ordered per-task records, proves that each evaluator receives its
  immutable model revision (or verifies it from a local checkpoint marker),
  binds both policy revisions and the trained artifact digest, retains raw
  results and failures in portable bundles, re-hashes those retained bytes at
  publication time, and rejects
  publication unless every suite clears the three-seed significance and
  cost-accounting gate. The suites still require measured execution; an
  implemented runner is not a benchmark result.
- **Public product-use benchmark:** [`benchmarks/product_use/`](benchmarks/product_use/)
  provides versioned MCP tasks, example traces, and a scorer that runs the
  real tools for teaching and evaluating how agents use StateSet. Public scores
  are separate from GPU performance evidence; private holdouts are required
  for a leaderboard.
- **Distributed evidence collection:** a shell-free manifest runner executes
  the two-node, three-seed soak and fault matrix, rotates scenario order,
  measures wall time outside provider adapters, hashes retained artifacts,
  accounts for failures, and invokes the strict publication gate. Live
  execution and the 12-hour soak remain pending.

### How to evaluate StateSet

For a team running a support agent, the practical question is how many more
cases it can resolve correctly at the same total cost. In the retained
48-step comparison, StateSet's TRL-backed trainer was about 7% slower than
direct TRL. Neither showed a measurable learning-quality gain at that scale.
[Comparison results and limitations](docs/BENCHMARKS.md#48-step-three-framework-comparison-v2-protocol)

A separate matched 200-step GSM8K trial on Qwen2.5-1.5B scored **31.4%** for
StateSet native GSPO and **31.1%** for direct TRL. The paired 95% interval
for the difference spans −7.5 to +8.0 percentage points across three seeds.
[Matched framework results](https://github.com/stateset/stateset-agents/blob/4164974a853bca6c42fab94cbc9142c506a16ebd/benchmark_results/framework_comparison_v5_1/README.md)

In a post-v0.74.0 CPU experiment, balancing refund eligibility within each
training batch raised final success from **561/1,024 (54.8%)** to **836/1,024
(81.6%)** across eight held-out training seeds. Each policy faced the same two
synthetic refund states. This compares two StateSet training recipes.
[Preregistered protocol, paired uncertainty and raw runs](benchmark_results/training_contracts/20261119/sampled-workflow-batching/)

A support team can test whether those gains transfer to its own workflow.

1. **Agree on a win before training.** Have an independent support team hold
   back fresh cases and define a meaningful improvement, a total cost ceiling,
   and a maximum rate of policy violations. Count a refund as resolved only
   when the correct ledger action occurs.
2. **Compare credible alternatives.** Include the team's current system, a
   supervised fine-tune, and a tuned RL baseline such as direct TRL or verl.
   Match starting weights, training data, tool access, hardware and tuning
   budget across training frameworks. Include failed runs, retries and serving
   costs at the same declared traffic volume.
3. **Publish enough to challenge the result.** Freeze configurations before
   opening the holdout. Score the same cases blind, report paired uncertainty
   across cases and training seeds, and show the result for each team and each
   comparator. Ask a second team to repeat the winning workflow on fresh data.
   [Public pilot](https://github.com/stateset/stateset-agents/issues/136)

Broader framework claims also depend on the following engineering evidence:

1. Repeat the matched 1.5B comparison on the frozen release, then run the
   7–9B three-seed flagship and require a
   positive paired 95% confidence bound on held-out quality.
2. Run the same experiment contract through verl, NeMo RL, and OpenRLHF and
   report throughput, peak memory, total provider cost, and quality together.
3. Retain a multi-node training run, the complete fault matrix, and a 12-hour
   soak proving no lost or duplicate updates and bounded policy staleness.
4. Execute the official τ³-bench, BFCL V4, and SWE-bench Verified pipelines on
   base and trained policies rather than treating implemented adapters as
   benchmark results.
5. Close container, Fireworks, Modal, CoreWeave, and Nebius live evidence,
   then obtain an independent reproduction and security review.

The multi-node scaling path is now executable rather than aspirational:
`benchmark-scaling-multi-node-run` accepts a shell-free provider launcher,
binds every run to the clean harness commit, gathers hashed physical-node IDs
from the distributed ranks, verifies ranks per node, and re-hashes each saved
policy. Live provider execution remains the evidence still to collect.
The repository also includes a concrete budget-bounded RunPod adapter with
private-DNS rendezvous, distinct-machine attestation, per-node watchdogs,
unconditional cleanup, and retained per-pod cost records. Standard RunPod Pod
networking is only 100 Mbps, so the adapter is documented as a
functional/negative-evidence path rather than a shortcut to a passing claim.
For the publication path, a provider-neutral Kubernetes adapter now supports
CoreWeave CKS and Nebius clusters using digest-pinned images, Indexed-Job DNS,
hard node anti-affinity, an enforced RDMA/fabric resource, immutable commit
checkout, rank-zero artifact retrieval, and unconditional resource cleanup.

The acceptance gates and exact evidence still missing are tracked in
[`docs/ROADMAP.md`](docs/ROADMAP.md). Until they pass, StateSet claims a
well-tested trace-to-policy control plane—not general RL-framework leadership.

---

## The improvement loop

Your agent already produces conversation logs. They are a training set.

```bash
pip install stateset-agents

# 1. Bring your own logs — OpenAI chat format or LangChain traces
stateset-agents ingest --format openai --input my_agent_logs.jsonl --output transcripts/

# 2. Grade every conversation, curate the best turns, get your next command
stateset-agents improve run \
  --transcripts transcripts/ \
  --reward customer_support \
  --output improved/

# 3. Train on what worked (the exact command is printed in improved/next_steps.md)
python scripts/sft_from_curated.py --dataset improved/curated.jsonl --base-model <model>
```

**No GPU?** Step 3 is the only part that needs one. `train-remote` rents one,
trains, proves the model learned, and gives the hardware back:

```bash
stateset-agents train-remote --provider runpod --gpu "NVIDIA H100 80GB HBM3" \
  --dataset improved/curated.jsonl --base-model meta-models/Muse-Glimmer-30B \
  --container-disk-gb 160 --eval-prompts held_out.txt --max-cost 5
```

The pod is terminated on every exit path — success, failure, timeout, or your
laptop dying mid-run — and `--max-cost` refuses to start a run that could
exceed your ceiling. Alongside the adapter you get `eval_results.json`: the
base model's answers next to the fine-tuned model's, on prompts it never
trained on, with pass/fail assertions if you wrote them.

**Then talk to what you trained:**

```bash
stateset-agents chat-remote --base-model meta-models/Muse-Glimmer-30B \
  --adapter outputs/sft_v1
```

Every conversation is saved in the format `ingest` accepts — so the loop
closes: chat → ingest → improve → train → chat.

### Getting started — pick your substrate

| you have | start here |
|---|---|
| **Nothing but an API key** (no GPUs, ever) | [`docs/GETTING_STARTED_RIVER.md`](docs/GETTING_STARTED_RIVER.md) — train, self-improve, and RL-tune through River's remote autograd; zero machines rented |
| **A few dollars for rented GPUs** | [`docs/GETTING_STARTED_API.md`](docs/GETTING_STARTED_API.md) — zero to calling your own fine-tuned model over an OpenAI-compatible API |
| **Bigger jobs on rented GPUs** | [`docs/RUNPOD_GUIDE.md`](docs/RUNPOD_GUIDE.md) — GPU/disk sizing, spot pricing, multi-GPU sharding, merged serving for hybrid models, and every failure mode we hit |
| **Five minutes and curiosity** | `bash examples/five_minute_demo.sh` — the whole loop offline, no GPU, no key |

### Choose a training provider

The provider changes where compute runs and where the trained result lives; it
does not change the dataset or experiment contract. Start with RunPod when you
want model weights back, or River when you want managed remote-autograd RL
without operating a GPU machine.

| Provider | Compute model | Result | Current evidence |
|---|---|---|---|
| **RunPod** | Rented GPU machine | Local adapter and evaluation artifacts | Live end-to-end training, serving, cost, and cleanup evidence |
| **River AI** | Managed remote autograd | Hosted `river://` checkpoint | Live SFT/RL training and held-out sampling evidence |
| **CoreWeave** | Bare-metal CKS training; Dedicated Inference | Local adapters; hosted BYOW model | Integration and cleanup unit-pinned; live certification pending ([guide](docs/COREWEAVE_PROVIDER.md)) |
| **Nebius** | Serverless AI jobs and endpoints | Local adapters; hosted vLLM model | Integration and cleanup unit-pinned; live certification pending ([guide](docs/NEBIUS_PROVIDER.md)) |
| **Fireworks AI** | Managed fine-tuning and serving | Hosted LoRA, with optional local artifacts | Integration complete; live full-lifecycle certification pending |
| **Tinker** | Managed remote autograd | Hosted sampler and training-state pointers | Inkling-Small SFT loop unit-pinned; live certification pending ([guide](docs/MANAGED_TRAINING_PROVIDERS.md)) |
| **Prime Intellect Lab** | Managed verifiers/OpenEnv RL | Hosted Prime run pointer | Config/CLI lifecycle unit-pinned; live certification pending ([guide](docs/MANAGED_TRAINING_PROVIDERS.md)) |
| **Hugging Face** | GPU Jobs and Inference Endpoints | Writable bucket artifact pointer; hosted endpoint | Job/endpoint lifecycles unit-pinned; live certification pending ([guide](docs/MANAGED_TRAINING_PROVIDERS.md)) |
| **Together AI** | Managed supervised fine-tuning | Hosted tuned-model pointer | Upload/train/poll/cancel lifecycle unit-pinned; live certification pending ([guide](docs/MANAGED_TRAINING_PROVIDERS.md)) |
| **Modal** | Rented serverless GPU | Local artifacts | Dataset upload, named Secrets, region/multi-GPU wiring, output commit/download, and per-job Volume cleanup implemented; live certification pending |
| **Local** | Your machine or cluster | Local artifacts | Unit-tested reference path |

Inspect capabilities and model evidence from the installed release:

```bash
stateset-agents remote-providers --json
stateset-agents model-support --json
```

Before paying for a frontier-model run, ask RunPod for a non-billable plan:

```bash
stateset-agents train-remote --provider runpod \
  --dataset data/train.jsonl --base-model Qwen/Qwen3.8-Flash-Next \
  --plan-only
```

`--plan-only` never contacts or provisions RunPod. Estimated frontier plans
must also have an explicit `--max-cost` before they can execute. The full
evidence rules and promotion stages are in
[`docs/MODEL_PROVIDER_CERTIFICATION.md`](docs/MODEL_PROVIDER_CERTIFICATION.md).

Both guides are written from live runs — the pitfalls listed are ones we
actually paid for. What the loop has proven, with numbers:
[`docs/rl-vibe.md`](docs/rl-vibe.md); every claim's evidence status:
[`docs/PROOFS.md`](docs/PROOFS.md).

### It works. Here is the receipt.

140 support conversations, three epochs, about a dollar of rented H100 —
answering an order number it had never seen:

| | |
|---|---|
| **Base model** | `to=self` … *"We need to respond. No context. Probably we don't have access to order tracking…"* — never answers the customer |
| **Fine-tuned** | *"Thanks for reaching out to StateSet Support! I checked right away: your order #77701 is on the way — it left our warehouse and should arrive within 3 business days. Anything else I can help with? — Astra @ StateSet"* |

Every row below is a run that actually happened, on rented hardware, with
the artifact returned — not a mock and not a plan:

| Proven | Evidence |
|---|---|
| Muse Glimmer 30B (63GB, multimodal) | 258MB adapter; persona learned from 140 examples; base model could not answer at all |
| Nemotron 3.5 Lightning (hybrid Mamba/MoE) | Learns the same task, but needs 8 epochs to hold a brand name where Muse needs 3 |
| Nemotron 3.5 Super VL NVFP4 on River | One CISPO optimizer update with a finite nonzero gradient norm and hosted step-1 checkpoint; validation tied the baseline, so no quality gain is claimed ([evidence](benchmark_results/training_contracts/20261015/nemotron-vl/README.md)) |
| Qwen3.8-27B (hybrid, multimodal) | Fine-tuned the week it shipped: 2/2 held-out assertions, $0.96 |
| A model too big for one card | 63GB checkpoint sharded across two 48GB cards (`0=24 module(s), 1=36 module(s)`) |
| The RL core, not just SFT | GSPO on a real GPU: target-token probability 2.8e‑05 → 0.125 in 40 steps, re-proved weekly |
| Multi-turn memory | `chat-remote` resolved *"I got double charged for it"* to the order number from the previous turn |
| **The loop raises a ceiling** | On compound requests gen‑1 was never trained on: base 0/12, gen‑1 **2/12**, gen‑2 — trained only on gen‑1's machine-curated sampled successes — **10/12**, reproduced twice ([`FLYWHEEL_HEADROOM.md`](docs/FLYWHEEL_HEADROOM.md)) |
| **Serve it over HTTPS** | `serve-remote`'s endpoint answered live: an authenticated `POST /v1/chat/completions` against `https://<pod-id>-8000.proxy.runpod.net` returned a completion from a vLLM server the platform brought up on a rented H100 |
| **The loop replicates — as a product** | Second domain (IT helpdesk), 35× smaller model (0.8B), one `stateset-agents flywheel` command, reproduced twice: 0/12 → 7/12 and 0/12 → **11/12**, harvest rate rising to 42.7% and 57.9%, ~$3/run ([`FLYWHEEL_DOMAIN2.md`](docs/FLYWHEEL_DOMAIN2.md)) |
| The loop holds under contamination | Machine-curated data trains a second generation with no manual data work ([`FLYWHEEL_EXPERIMENT.md`](docs/FLYWHEEL_EXPERIMENT.md), limitations included) |

**Every claim above is tracked in [`docs/PROOFS.md`](docs/PROOFS.md)** —
what kind of proof backs it (scheduled re-verification, a retained live run,
or a pinned test) and which rows are still automation-pending.

What is **not** yet proven is labelled as such throughout — the starter
table's ✅ column marks which models have actually been trained rather than
merely wired up. The River provider is live-verified as of 2026-08-18 —
their SDK finally landed on PyPI (Python ≥3.12) and the first real run
trained and sampled a checkpoint through our executor unchanged.

`improve` writes three things: `improve_summary.json` (machine‑readable scores
and per‑reward breakdown), `curated.jsonl` (the turns above your threshold, ready
to train on), and `next_steps.md` (runnable training commands — regression‑tested
against the real CLI so they never drift).

**Try it in five minutes, offline, no GPU and no API key:**

```bash
bash examples/five_minute_demo.sh
```

It writes sample logs, runs the whole loop, and shows you the graded output.
Colab version: [`notebooks/improve_your_agent_5min.ipynb`](notebooks/improve_your_agent_5min.ipynb).

**Or let an agent drive it:**

```bash
pip install "stateset-agents[mcp]"
claude mcp add stateset-agents -- stateset-agents mcp
```

Seven MCP tools (`list_rewards`, `ingest_transcripts`, `grade_transcript`,
`improve_run`, `improve_status`, `list_model_presets`, `dry_run_finetune`) — see
[`docs/MCP_SERVER.md`](docs/MCP_SERVER.md).

Check what has actually been proven—not merely registered—with
`stateset-agents model-support`. Its schema-versioned output separates unit
coverage, live hardware attempts, and successful inference by provider.

---

## What's new

**v0.77.0 (latest candidate; publication pending):**

- **3D Weight Studio v2 realism.** Adds real-time planar optical-bench reflections, 1024×1024 PCF soft shadow maps, depth-crevice SSAO, optical Bokeh Depth-of-Field, 60 FPS GPU liquid-crystal morphing, 3D spatial rulers, and GQA attention-head dividers.
- **Mechanistic interpretability & advanced surgery.** Adds real PyTorch `register_forward_hook` execution on local causal LMs, optional live vLLM/OpenAI endpoint probing, Layer-by-Layer Logit Lens, Causal Activation Patching, TIES/DARE/SLERP multi-adapter merging with 3D Sign-Conflict view, Wanda+Fisher importance heatmaps, Per-Head Attention Surgery, ROME Rank-1 edits with protected-prompt drift checks, and `merge_adapter` effect probes.
- **Scale & curated datasets.** Adds multi-shard `model.safetensors.index.json` lazy discovery, binary `Float32Array` 128×128 Microscope streaming with deep-zoom cropping, curated JSONL dataset & reward trace inspection, and `finetune_workspace` validation.

**v0.76.0:**

- **3D Weight Studio.** `stateset-agents studio` (or `python -m stateset_agents.weight_studio`) launches a browser-based, custom WebGL2 PBR 3D workbench for exploring local `.safetensors` checkpoints and model presets across four 3D views (Architecture Tower, Microscope 64², SVD Spectral, and Trajectory 4D).
- **Surgical weight editing.** Apply non-destructive Scale, Magnitude Prune, Percentile Clamp, SVD Rank Truncation, Gaussian Perturbation, 3D Brush Region Edits, and Checkpoint Blending with full Undo/Redo history and SHA-256 reload-verified `.safetensors` export.
- **Interactive finetuning & model probing.** Configure GSPO/GRPO/DAPO/GEPO/VAPO/PPO/SFT runs with live 3D LoRA target highlighting, VRAM/provider cost estimates, live SSE-streamed PyTorch micro-finetuning updates, and Base-vs-Edited A/B inference with Token Logit Lens and 3D activation waves.

**v0.75.0:**

- **Benchmarks as environments.** τ-bench retail and BFCL V4 function calling
  can be used for training and evaluation. The three-arm τ-bench retail study
  runner has no published result yet, and BFCL scores use a seeded split that
  is not comparable with leaderboard numbers.
  [τ-bench retail](benchmarks/TAU_RETAIL.md) · [BFCL](benchmarks/BFCL.md)
- **Token loss.** `bounded_token_backward` backpropagates a few turns at a
  time and now works on the distributed token path, and
  `shared_prefix_min_tokens` encodes a shared prompt prefix once. Both are
  off by default and validated on CPU only. `strict_rewards` aborts on a
  failed rollout or reward. [Changelog](CHANGELOG.md)
- **Environment registry.** Build environments by name and discover ones
  published by other packages through an entry-point group.
- **Inkling 2 Mini starter.** `stateset-agents inkling-2-mini` configures
  attention-only LoRA for `thinkingmachines/Inkling-2-Mini`. It is a
  configuration integration; no training run has been verified.
  [Starter guide](docs/inkling2_mini_starter.rst)

**v0.74.0:**

- **Offline RL.** For a fixed critic, the Lean proofs bound how the number
  of sampled action proposals affects CQL's estimated soft value and penalty
  gradient under the stated sampling assumptions. CPU tests check IQL and CQL
  critic updates and checkpoint replay. [Proofs and assumptions](formal/README.md#rl-mathematics)
- **Online training.** PPO records token probabilities from the weights used
  to sample each rollout. Native and continual GRPO reuse the recorded sampler
  probabilities across repeated updates. DAPO and VAPO apply their configured
  entropy and KL terms during training. [Changelog](CHANGELOG.md)
- **Verification scope.** The formal suite covers 38 Lean roots, 49 finite
  TLA+ model configurations, and 57 equations extracted from PyTorch.
  End-to-end proofs for every trainer remain open. [Verification coverage](formal/README.md)

The tag follows the 0.67.0 source line. The 0.73.x releases have a separate
history. [Tagged source](https://github.com/stateset/stateset-agents/tree/v0.74.0)

Highlights from earlier releases are in
[docs/RELEASE_HIGHLIGHTS.md](docs/RELEASE_HIGHLIGHTS.md); every change is in
[CHANGELOG.md](CHANGELOG.md).

## Renting GPUs, and knowing what it cost

Model/provider support is now tiered as `default`, `frontier-preview`, or
`compatibility`, with separate configured, smoke, training, serving, and
production certification stages. `stateset-agents model-support --json`
returns the auditable catalog and its dated evidence. See
[`docs/MODEL_PROVIDER_CERTIFICATION.md`](docs/MODEL_PROVIDER_CERTIFICATION.md).

Before renting anything, resolve a non-billable RunPod plan:

```bash
stateset-agents train-remote --provider runpod \
  --dataset data/train.jsonl --base-model Qwen/Qwen3.8-Flash-Next \
  --plan-only
```

Known measured models automatically select catalog GPU/count/disk defaults.
Frontier or unknown plans are explicitly labeled estimates and require a
`--max-cost` ceiling before execution. `--plan-only` provisions nothing;
`--dry-run` runs on the selected provider and may still allocate compute.

Four commands cover the whole rented-hardware lifecycle. Every pod is
terminated on every exit path — success, failure, timeout, or your laptop
dying mid-run — and every pod records what it cost.

| Command | What it does |
|---|---|
| `train-remote` | Rents a GPU, fine-tunes, returns the adapter plus `eval_results.json`, gives the hardware back |
| `chat-remote` | Multi-turn conversation with a tuned model; saves every transcript in the format `ingest` accepts |
| `serve-remote` | Persistent vLLM endpoint with a bearer token, `--max-hours` self-destruct, `--list` / `--stop` |
| `costs` | What every remote run actually cost — model, hardware, duration, dollars |

Cost control is built in rather than bolted on: `--max-cost N` refuses to
start a run whose worst case would exceed N dollars, and a pod the provider
will not price is refused rather than rented. `--cloud-type COMMUNITY` uses
spot pricing, `--network-volume-id` keeps checkpoints alive across pod death,
and `--gpu-count 2` shards a checkpoint too large for one card (verified: 63GB
across two 48GB cards).

Adapters are not anonymous either — each carries a manifest with its base
model, dataset **content hash**, hyperparameters, eval outcome, and parent
adapter, which `stateset-agents adapters` reads back as a family tree.

See [`docs/RUNPOD_GUIDE.md`](docs/RUNPOD_GUIDE.md) for sizing, capacity
behaviour, and a troubleshooting table covering every failure mode this
project actually hit.

## Why group‑based optimization?

Traditional RLHF/PPO trains on one sampled response at a time. In long conversations this leads to high‑variance updates and brittle behavior.  
StateSet Agents implements **group‑relative methods**:

- **GRPO (Group Relative Policy Optimization)**: sample a group of trajectories per prompt, compute advantages relative to the group baseline, then apply clipped policy‑gradient updates.
- **GSPO (Group Sequence Policy Optimization)**: a more stable sequence‑level variant (Alibaba Qwen team) that avoids token‑level collapse on long outputs and MoE models.

The result is steadier learning for dialogue tasks.

---

## Core concepts

- **Agent**: wraps a causal LM and exposes `initialize()` and `generate_response()`.
  - `MultiTurnAgent` handles conversation history and state.
  - `ToolAgent` adds function/tool calling.
- **Environment**: defines episode reset/step logic and optional reward hooks.
  - `ConversationEnvironment` ships with scenario‑driven multi‑turn conversations.
  - `TaskEnvironment` is for goal‑oriented tasks.
- **Trajectory**: a multi‑turn record of turns, rewards, and metadata (`MultiTurnTrajectory`).
- **Rewards**: `RewardFunction` subclasses and factories; combined via `CompositeReward` or multi‑objective reward models.
- **Training**: trainers in `stateset_agents.training` implement GRPO‑family updates, GAE/value heads, KL regularization, LoRA support, and optional distributed/vLLM execution.

---

## Reward semantics

Reward functions can be evaluated per-step or only at episode end. Set
`reward_type` on your `RewardFunction` to control how the environment applies it:

- `RewardType.IMMEDIATE` or `RewardType.DENSE`: compute per-step rewards only.
- `RewardType.CUMULATIVE` or `RewardType.SPARSE`: compute a final reward only.

If you pass a custom reward without `reward_type`, the environment assumes legacy
behavior and may compute both step and final rewards. For new rewards, always
set `reward_type` explicitly to avoid double counting.

---

## Tool calling (ToolAgent)

`ToolAgent` lets a model request a tool via a JSON block, which the agent executes:

```python
import asyncio
from stateset_agents.core.agent import AgentConfig, ToolAgent

def add(a: int, b: int) -> int:
    return a + b

async def main():
    agent = ToolAgent(
        AgentConfig(model_name="stub://tools", use_stub_model=True),
        tools=[
            {
                "name": "add",
                "description": "Add two integers",
                "parameters": {"a": "int", "b": "int"},
                "function": add,
            }
        ],
    )
    await agent.initialize()
    # The model should respond with a JSON tool call like:
    # {"tool": "add", "parameters": {"a": 1, "b": 2}}
    print(await agent.generate_response("Please calculate 1 + 2"))

asyncio.run(main())
```

---

## Installation

### Core (lightweight, stub‑ready)

To install the Python package from PyPI, run

```bash
pip install "stateset-agents==0.74.0" # latest verified PyPI release
```

To install from the corresponding Git tag, run

```bash
pip install "stateset-agents @ git+https://github.com/stateset/stateset-agents.git@v0.77.0"
```

That's enough for the [five-minute demo](#the-improvement-loop), the stub
backend, and the CLI. Training real models needs `[training]` below.

> To install the development version on `master`, run
>
> ```bash
> pip install "git+https://github.com/stateset/stateset-agents@master"
> ```

### Training / real models

```bash
pip install "stateset-agents[training]"
```

For a pinned training and API environment from this checkout on Python 3.12/Linux:

```bash
make install-training-locked
```

`requirements-training-lock.txt` contains the resolved runtime dependencies.
`make lock` regenerates all three locks, and `make lock-check` verifies them.
Verification resolves temporary copies, preserves working locks on failure or
interruption, and rejects input changes made while it runs.
CI audits the runtime lock independently and tests its installed wheel. A passing
dependency audit does not establish GPU compatibility or model quality.

### Optional extras

```bash
pip install "stateset-agents[auto-research]" # Autonomous experiment loop + Optuna
pip install "stateset-agents[trl]"           # TRL GRPO integration + bitsandbytes
pip install "stateset-agents[vllm]"          # vLLM generation backend
pip install "stateset-agents[hpo]"           # Optuna/Ray Tune HPO
pip install "stateset-agents[api]"           # FastAPI service
pip install "stateset-agents[distributed]"   # DeepSpeed / multi‑GPU helpers
pip install "stateset-agents[coreweave]"     # CKS training + Dedicated Inference
pip install "stateset-agents[nebius]"        # Serverless AI jobs + endpoints
pip install "stateset-agents[rust]"          # Rust-accelerated GAE/advantage kernels (stateset-rl-core)
pip install "stateset-agents[full]"          # Most extras in one go
```

### Node.js API client

The typed Node.js client requires Node 18+ and has no runtime dependencies.
It supports Messages and OpenAI-compatible calls, including SSE streaming.
From your Node application's directory, install the client and check the import.

```bash
npm install @stateset/agents@0.74.0
node --input-type=module -e "import { StateSet } from '@stateset/agents'; console.log(new StateSet().constructor.name)"
```

The last command prints `StateSet` without connecting to a server.
[Messages and streaming examples](npm/README.md)

> This repository also contains an internal, unpublished Rust crate at the repo root (a StateSet
> commerce daemon) that is unrelated to the `stateset-rl-core` accelerator behind `[rust]` above.
> See `docs/RUST_CRATES.md` for how the two Rust crates in this repo relate.

### Model starter paths

One driver covers every supported model — `--dry-run` is the default, so nothing
trains until you ask:

```bash
pip install "stateset-agents[training,trl]"
python examples/finetune_gspo.py --list-models              # 20 presets
python examples/finetune_gspo.py --model qwen3.5-0.8b       # dry run: show the resolved config
python examples/finetune_gspo.py --model qwen3.5-0.8b --no-dry-run   # actually train
```

Useful flags: `--starter-profile {balanced,memory,quality}` (the `memory` profile
uses 4‑bit quantization and smaller context/group sizes), `--use-lora/--no-lora`,
`--use-4bit/--use-8bit`, `--use-vllm`, `--wandb`, `--export-merged`,
`--write-config PATH`.

Twelve models ship a dedicated starter with tuned defaults, and two newly
released composite models have architecture-aware unified presets. The ✅
column marks what has actually been fine-tuned on rented hardware, not merely
wired up:

| Model | Training entry point | Live-verified | Notes |
|---|---|---|---|
| `ibm-granite/granite-4.0-micro` | `stateset-agents granite-4-micro` | | LoRA/QLoRA; [guide](examples/README.md#additional-small-models-and-functiongemma) |
| `meta-llama/Llama-3.2-1B-Instruct` | `stateset-agents llama-3-2-1b` | | LoRA/QLoRA; [guide](examples/README.md#additional-small-models-and-functiongemma) |
| `meta-llama/Llama-3.2-3B-Instruct` | `stateset-agents llama-3-2-3b` | | LoRA/QLoRA; [guide](examples/README.md#additional-small-models-and-functiongemma) |
| `Qwen/Qwen3-4B-Instruct-2507` | `stateset-agents qwen3-4b-instruct` | | LoRA/QLoRA; [guide](examples/README.md#additional-small-models-and-functiongemma) |
| `allenai/Olmo-3-7B-Instruct` | `stateset-agents olmo-3-7b` | | LoRA/QLoRA; [guide](examples/README.md#additional-small-models-and-functiongemma) |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | `stateset-agents deepseek-r1-1-5b` | | LoRA/QLoRA; [guide](examples/README.md#additional-small-models-and-functiongemma) |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | `stateset-agents deepseek-r1-7b` | | LoRA/QLoRA; [guide](examples/README.md#additional-small-models-and-functiongemma) |
| `google/functiongemma-270m-it` | `python -m stateset_agents.training.sft --base-model google/functiongemma-270m-it ...` | | Tool-calling SFT only; [guide](examples/README.md#additional-small-models-and-functiongemma) |
| `Qwen/Qwen3.5-0.8B` | `stateset-agents qwen3-5-0-8b` | ✅ | Cheapest path to a first run (~$0.30) |
| `meta-models/Muse-Glimmer-30B` | `stateset-agents muse-glimmer` | ✅ | Meta's open agentic model; dense 30B, multimodal, 131K ctx, Apache‑2.0 |
| `meta-models/Muse-Spark-Open-816B-A42B` | `stateset-agents muse-spark-open` | GPU unverified | Distributed BF16 text LoRA SFT; separate text GSPO preset; gated access and cluster-scale memory; [guide](docs/MUSE_SPARK_OPEN.md) |
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | `stateset-agents nemotron-3-5` | ✅ | Hybrid Mamba‑2 + MoE reasoning model, 3B active params |
| `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` | `stateset-agents nemotron-3-5-super-vl` | CPU contracts | Image-conditioned LoRA SFT; full-size GPU training unverified; [guide](docs/NEMOTRON_3_5_VL.md) |
| `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16` | `stateset-agents nemotron-3-5-bolt` | CPU contracts | Image-conditioned LoRA SFT; BF16 training uses more memory than the 24GB inference artifact; [guide](docs/NEMOTRON_3_5_VL.md) |
| `Qwen/Qwen3.8-27B` | `stateset-agents qwen3-8-27b` | ✅ | Hybrid linear/standard attention, multimodal, 256K ctx, Apache‑2.0 — ~56GB BF16 |
| `Qwen/Qwen3.8-Flash-Next` | `python examples/finetune_gspo.py --model qwen3.8-flash-next` | | Qwen4 architecture preview; 125B main / 6B active + 51B n-gram embeddings; native multimodal, 262K ctx |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct` | `stateset-agents qwen3-coder` | | 128 experts / 8 active, 256K ctx, Apache‑2.0 |
| `openai/gpt-oss-20b` | `stateset-agents gpt-oss` | | 32 experts / 4 active, 128K ctx, Apache‑2.0 |
| `deepseek-ai/DeepSeek-V4-Flash` | `stateset-agents deepseek-v4` | | MLA attention, 256 experts, 1M ctx, MIT — QLoRA + vLLM |
| `thinkingmachines/Inkling-2-Mini` | `stateset-agents inkling-2-mini` | | Looped MoE, 128 experts / 8 active, 1M ctx, Apache‑2.0 — attention-only LoRA on the NVFP4 checkpoint; training unverified |
| `google/gemma-4-31B-it` | `stateset-agents gemma-4-31b` | | Use `--starter-profile memory` on tighter GPU budgets |
| `google/gemma-4-E2B-it` | `stateset-agents gemma-4-e2b` | | Text-only LoRA/QLoRA; install `[gemma4]`; [guide](examples/README.md#small-gemma-models-e2b-and-e4b) |
| `google/gemma-4-E4B-it` | `stateset-agents gemma-4-e4b` | | Same profiles; E4B includes about 8B parameters with embeddings |
| `Qwen/Qwen3.5-2B` | `stateset-agents qwen3-5-2b` | | Text-only LoRA/QLoRA; install `[small-models]`; [guide](examples/README.md#more-small-models-qwen-smollm-phi-ministral-and-liquid) |
| `Qwen/Qwen3.5-4B` | `stateset-agents qwen3-5-4b` | | Text-only LoRA/QLoRA; install `[small-models]`; [guide](examples/README.md#more-small-models-qwen-smollm-phi-ministral-and-liquid) |
| `Qwen/Qwen3.5-9B` | `stateset-agents qwen3-5-9b` | | Text-only LoRA/QLoRA; install `[small-models]`; [guide](examples/README.md#more-small-models-qwen-smollm-phi-ministral-and-liquid) |
| `HuggingFaceTB/SmolLM3-3B` | `stateset-agents smollm3-3b` | | Text-only LoRA/QLoRA; install `[small-models]`; [guide](examples/README.md#more-small-models-qwen-smollm-phi-ministral-and-liquid) |
| `microsoft/Phi-4-mini-instruct` | `stateset-agents phi-4-mini` | | Text-only LoRA/QLoRA; install `[small-models]`; [guide](examples/README.md#more-small-models-qwen-smollm-phi-ministral-and-liquid) |
| `mistralai/Ministral-3-3B-Instruct-2512-BF16` | `stateset-agents ministral-3-3b` | | Text-only LoRA/QLoRA; install `[small-models]`; [guide](examples/README.md#more-small-models-qwen-smollm-phi-ministral-and-liquid) |
| `mistralai/Ministral-3-8B-Instruct-2512-BF16` | `stateset-agents ministral-3-8b` | | Text-only LoRA/QLoRA; install `[small-models]`; [guide](examples/README.md#more-small-models-qwen-smollm-phi-ministral-and-liquid) |
| `LiquidAI/LFM2.5-2.6B` | `stateset-agents lfm2-5-2-6b` | | Text-only LoRA/QLoRA; install `[small-models]`; [guide](examples/README.md#more-small-models-qwen-smollm-phi-ministral-and-liquid) |
| `moonshotai/Kimi-K2.6` | `stateset-agents kimi-k2-6` | | |
| `moonshotai/Kimi-K3` | `stateset-agents frontier-models --model kimi-k3` | | Released MXFP4 checkpoint pinned; training loader blocked; see [frontier guide](docs/FRONTIER_FINETUNING.md) |
| `zai-org/GLM-5.1` | `python examples/finetune_glm5_1_gspo.py` | | 754B MoE, QLoRA‑only + vLLM |
| `zai-org/GLM-5.2` | `python examples/finetune_glm5_2_gspo.py` | | 754B MoE, QLoRA‑only + vLLM |
| `zai-org/GLM-5.3-Flash` | `python examples/finetune_gspo.py --model glm5.3-flash` | | 320B / 18B active, native multimodal, FP8, 1M ctx |
| `thinkingmachines/Inkling-Small` | `stateset-agents train-remote --provider tinker --base-model thinkingmachines/Inkling-Small ...` | | Tinker-native 276B / 12B-active multimodal MoE; remote-autograd SFT/RL, hosted sampler/state pointers |

Every CLI starter accepts the same flags: `--json-output`, `--list-profiles`,
`--starter-profile NAME`, `--write-config PATH`, `--config PATH --no-dry-run`.
The GLM starters are importable too (`from stateset_agents.training.glm5_2_starter
import get_glm5_2_config, run_glm5_2_config`), as are the others.

### Supported models

First-class starters ship for **Qwen 3.5 0.8B**, **Muse Glimmer 30B**, **Nemotron 3.5 Lightning**, **Qwen3.8 27B**, **Qwen3-Coder 30B**, **gpt-oss 20B**, **DeepSeek V4 Flash**, **Inkling 2 Mini** *(config integration; training unverified)*, **Gemma 4 31B IT**, **Kimi-K2.6**, **Kimi-K3** *(released; packed-weight training loader blocked)*, **GLM 5.1**, and **GLM 5.2**. Architecture-aware unified presets additionally cover **GLM-5.3-Flash** and **Qwen3.8-Flash-Next**. Reference examples and hosting plans cover Qwen 3.5 27B, Qwen 3, Qwen 2.5, Kimi-K2.5, Gemma 3 / Gemma 2 27B IT, Llama 3, Llama 2 7B, and Mistral 7B. Compatible Hugging Face causal LMs work through the generic flow; native multimodal conditional-generation repositories use StateSet's composite-loader fallback for text-only RL.

See [`docs/SUPPORTED_MODELS.md`](docs/SUPPORTED_MODELS.md) for the full matrix, algorithm compatibility, and instructions for adding a new starter.

### Experimental namespace

Research-grade modules (neural architecture search, multimodal processing,
long-term planning, few-shot adaptation, the intelligent orchestrator,
adaptive learning controller, and multi-agent coordination) live in
`stateset_agents.experimental`. They carry **no API-stability guarantees**
and may change or be removed in any release. The former
`stateset_agents.core.<module>` import paths still work for one deprecation
cycle and emit a `DeprecationWarning`.

### Stable v1 compatibility contract

The supported top-level Python exports and production HTTP operations are
frozen in [`contracts/public_api_v1.json`](contracts/public_api_v1.json).
Every pull request and release runs `make api-compatibility`; an unreviewed
addition, removal, signature-schema change, or response-schema change fails
closed. See [`docs/API_COMPATIBILITY.md`](docs/API_COMPATIBILITY.md) for the
covered surface, deprecation window, and migration process. Experimental and
opt-in training-lab APIs are intentionally outside this guarantee.

### Component maturity and security response

Every canonical product domain has an evidence-backed Stable, Beta, or
Experimental designation in the
[`component maturity policy`](docs/COMPONENT_MATURITY.md). CI rejects missing
tests, documentation, evidence, limitations, graduation criteria, or invalid
stable-contract references. The public [`security policy`](SECURITY.md) defines
private reporting, severity-based response targets, coordinated disclosure,
and safe harbor; the independent third-party review remains explicitly pending.
Run the same fail-closed validation locally with:

```bash
make release-governance
```

### Dashboard and mobile app (separate repos)

The React + Vite dashboard and Expo mobile app — working clients for the
simulator-backed `/api/lab/*` "Training Lab" router — live in their own
repositories:
[stateset-agents-dashboard](https://github.com/stateset/stateset-agents-dashboard)
and
[stateset-agents-mobile](https://github.com/stateset/stateset-agents-mobile)
(extracted from this repo with full history, 2026-08-11). Both are runnable
locally but have no deployment path today — the router is gated behind auth
and the `API_ENABLE_TRAINING_LAB` flag (off by default).

### API serving (/v1/messages)

For hosted **Qwen4-Max**, set `INFERENCE_BACKEND=qwencloud` and supply a
Singapore `DASHSCOPE_API_KEY`. The preset selects `qwen4-max` and QwenCloud's
Singapore API endpoint. Use `/v1/chat/completions` to preserve reasoning history,
tool calls, and multimodal inputs. See the [Qwen4-Max guide](docs/QWEN4_MAX.md)
for gateway and direct Python examples.

```bash
export INFERENCE_BACKEND=vllm
export INFERENCE_BACKEND_URL=http://localhost:8001
export INFERENCE_DEFAULT_MODEL=moonshotai/Kimi-K2.5
# Optional: ask the backend to include token usage in streaming chunks when supported.
export INFERENCE_STREAM_INCLUDE_USAGE=true
```

```bash
curl http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "moonshotai/Kimi-K2.5",
    "max_tokens": 128,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

OpenAI-compatible endpoint:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "moonshotai/Kimi-K2.5",
    "max_tokens": 128,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Helm deployment

```bash
helm upgrade --install stateset-agents deployment/helm/stateset-agents \
  --namespace stateset-agents
```

---

## Quick start

### 1) Stub hello world (no downloads)

Runs without Torch/transformers and is ideal for CI or prototyping.

```python
import asyncio
from stateset_agents import MultiTurnAgent
from stateset_agents.core.agent import AgentConfig

async def main():
    agent = MultiTurnAgent(AgentConfig(model_name="stub://demo"))
    await agent.initialize()
    reply = await agent.generate_response([{"role": "user", "content": "Hi!"}])
    print(reply)

asyncio.run(main())
```

### 2) Chat with a real model

```python
import asyncio
from stateset_agents import MultiTurnAgent
from stateset_agents.core.agent import AgentConfig

async def main():
    agent = MultiTurnAgent(
        AgentConfig(
            model_name="your-real-model-id",
            max_new_tokens=128,
            temperature=0.7,
        )
    )
    await agent.initialize()
    messages = [{"role": "user", "content": "What is GRPO?"}]
    print(await agent.generate_response(messages))

asyncio.run(main())
```

For the zero-download onboarding path, run `python examples/quick_start.py`.

---

## Fine-tune from a Herdr terminal

```bash
stateset-agents tui --workspace outputs/finetune
# Preview a phone-sized terminal without launching work:
stateset-agents tui --snapshot --columns 40 --rows 24
```

Phone-sized terminals work from 32 columns by 16 rows. Compact controls use
`2`/`8` to move, `3`/`9` to page, `0` to go back and `4` to switch the main views.
Digits remain literal in editors and filters; reviewed launches still require
Enter. Resizes that hide the interface pause actions until it can be displayed.

Choose a model and dataset. Press `b` to select local CUDA, RunPod, Modal or
River. Local supports SFT/RL; remote studio backends support SFT. Preview inputs
with `p`, inspect local readiness with `i`, and review a launch with `r`.
Press `v` to see all configuration issues and jump to a field to fix it.
Press `u` for dataset statistics and duplicate checks; `e` explicitly reveals a
bounded private excerpt in that report.
On Runs, `l` opens wrapped logs with source selection, pause/follow and literal filtering.
Use `a` to pin a reference run and `x` to compare settings and reported metrics.
Switch to retained runs with `Tab`; `/` searches, `t` filters statuses, and `g`
opens reported metric trends and progress. Logs and metrics reconnect when you reopen the workspace; closing the pane
keeps jobs running. Local training uses an installed Buildr queue and requires
CUDA. See the [terminal studio guide](docs/FINETUNING_TUI.md) for data formats,
keyboard controls, resource settings and paid-run behavior. AI agents can use
`stateset-agents finetune schema` to discover the JSON interface and share this
workspace; see the [agent protocol](docs/FINETUNING_AGENTS.md).
On Runs, `r` reviews and launches an agent-prepared run with its existing ID,
saved plan checksum and retained dataset; `c` copies settings for a new run.

## Train a multi‑turn agent with GRPO

The high‑level `train(...)` helper chooses single‑turn vs multi‑turn GRPO automatically.

```python
import asyncio
from stateset_agents import (
    MultiTurnAgent,
    ConversationEnvironment,
    CompositeReward,
    HelpfulnessReward,
    SafetyReward,
    train,
)
from stateset_agents.core.agent import AgentConfig

async def main():
    # 1) Agent
    agent = MultiTurnAgent(
        AgentConfig(
            model_name="stub://quickstart",
            use_stub_model=True,
            system_prompt="You are a helpful customer support assistant.",
        )
    )
    await agent.initialize()

    # 2) Environment
    scenarios = [
        {
            "id": "refund",
            "topic": "refunds",
            "context": "User wants a refund for a delayed order.",
            "user_responses": [
                "My order is late.",
                "I'd like a refund.",
                "Thanks for your help.",
            ],
        }
    ]
    env = ConversationEnvironment(scenarios=scenarios, max_turns=6)

    # 3) Reward
    reward_fn = CompositeReward(
        [HelpfulnessReward(weight=0.7), SafetyReward(weight=0.3)]
    )

    # 4) Train
    trained_agent = await train(
        agent=agent,
        environment=env,
        reward_fn=reward_fn,
        num_episodes=4,
        profile="balanced",
        training_mode="single_turn",
        save_path="./outputs/refund_agent",
    )

    # 5) Try the trained model
    resp = await trained_agent.generate_response(
        [{"role": "user", "content": "My order was delayed, what can you do?"}]
    )
    print(resp)

asyncio.run(main())
```

With a real model, `train(...)` asks the agent for full turns
(`MultiTurnAgent.generate_turn`), so each trajectory carries the exact
prompt and response token ids the policy sampled. The loss then takes one
padded forward pass per trajectory group and evaluates the configured
[`PolicyObjective`](docs/OBJECTIVES.md) on per-token log-probs; stub and
text-only agents fall back to the sequence-level path. Pick any preset and
PPO-style inner updates through the config:

```python
trained_agent = await train(
    agent=agent,
    environment=env,
    reward_fn=reward_fn,
    num_episodes=4,
    config_overrides={
        "objective": "dapo",          # or dr_grpo, cispo, rloo, gspo, ...
        "objective_overrides": {"clip_high": 0.3},
        "num_gradient_updates": 2,    # inner updates against frozen old log-probs
    },
)
```

Rollouts can come from an inference engine instead of Hugging Face
`generate`: `agent.set_rollout_backend(VLLMGenerator(...))`. The trainers
push the policy's weights into the engine after every optimizer step
(`rollout_sync`), and `old_logprobs_source="sampler"` uses the recorded token probabilities
when forming ratios on fresh and replayed groups. [Replay gradient assumptions](formal/README.md#replayed-group-gradients)

`stateset-agents train --list-objectives` prints every preset with its
advantage estimator, ratio level, clip, aggregation, and KL estimator.

More end‑to‑end scripts live in `examples/complete_grpo_training.py` and `examples/production_ready_customer_service.py`.

---

## Continual learning + long‑term planning (optional)

Enable planning context and replay/LwF in the trainer with config overrides:

```python
agent = MultiTurnAgent(
    AgentConfig(
        model_name="stub://quickstart",
        use_stub_model=True,
        enable_planning=True,
        planning_config={"max_steps": 4},
    )
)

trained_agent = await train(
    agent=agent,
    environment=env,
    reward_fn=reward_fn,
    num_episodes=4,
    training_mode="single_turn",
    # resume_from_checkpoint="./outputs/checkpoint-100",
    config_overrides={
        "continual_strategy": "replay_lwf",
        "continual_kl_beta": 0.1,
        "replay_buffer_size": 500,
        "replay_ratio": 0.3,
        "replay_sampling": "balanced",
        "task_id_key": "task_id",
        "task_schedule": ["task_a", "task_b"],
        "task_switch_steps": 25,
    },
)

context = {"conversation_id": "demo-trip", "goal": "Plan a 4-day trip to Kyoto"}
resp = await trained_agent.generate_response(
    [{"role": "user", "content": "Can you draft a plan?"}],
    context=context,
)

followup = await trained_agent.generate_response(
    [{"role": "user", "content": "Great. What should we do next?"}],
    context={"conversation_id": "demo-trip", "plan_update": {"action": "advance"}},
)

# To update the plan goal explicitly:
# context={"conversation_id": "demo-trip", "plan_goal": "Plan a 4-day trip to Osaka"}
```

---

## Other training algorithms

All algorithms are available under `stateset_agents.training` when training deps are installed:

- **GSPO**: stable sequence‑level GRPO variant (`GSPOTrainer`, `GSPOConfig`, `train_with_gspo`)
- **GEPO**: expectation‑based group optimization for heterogeneous/distributed setups
- **DAPO**: decoupled clip + dynamic sampling for reasoning‑heavy tasks
- **VAPO**: value‑augmented group optimization (strong for math/reasoning)
- **PPO**: experimental token-exact online training with retained behavior
  likelihoods, critic values and rollout receipts, plus external-text reference
  updates. Online training currently requires one process and full-vocabulary
  sampling (`top_p=1`, `top_k=0`). See the
  [PPO limits](docs/OBJECTIVES.md#ppo-reference-update-limits).
- **RLAIF**: RL from AI feedback via judge/reward models

Every trainer above evaluates a named [`PolicyObjective`](docs/OBJECTIVES.md)
and accepts `objective="<preset>"` on its config to swap the advantage
estimator, ratio level, clipping, aggregation, and KL estimator; trainers
whose rollouts store sequence-level log-probs reject token-level presets at
construction with the compatible names.

Minimal GSPO sketch:

```python
from stateset_agents.training import get_config_for_task, GSPOConfig, train_with_gspo
from stateset_agents.rewards.multi_objective_reward import create_customer_service_reward

base_cfg = get_config_for_task("customer_service", model_name="your-real-model-id")
gspo_cfg = GSPOConfig.from_training_config(base_cfg, num_outer_iterations=5)

trained_agent = await train_with_gspo(
    config=gspo_cfg,
    agent=agent,
    environment=env,
    reward_model=create_customer_service_reward(),
)
```

See `docs/GSPO_GUIDE.md`, `docs/ADVANCED_RL_ALGORITHMS.md`, and `examples/train_with_gspo.py` for full configs.

---

## Scaffold a fine‑tuning project in 30 seconds

If you're building a fine‑tune for a client, start from a template instead of from scratch:

```bash
# See what's available
stateset-agents starter list

# Multi-turn customer support agent (the framework's differentiator)
stateset-agents starter customer-support ./my-client

# Single-turn math reasoner with verifiable rewards
stateset-agents starter gsm8k-math ./math-bench

# Agent that learns to invoke tools/APIs (weather, calculator, search stubs)
stateset-agents starter tool-calling-agent ./tool-agent

# Bare scaffold — edit everything
stateset-agents starter minimal ./hack
```

Each scaffold lands a runnable project: `config.yaml`, `scenarios.jsonl` (where applicable), `reward.py`, `train.py`, `eval.py`, `serve.sh`, plus a tailored `README.md`. From clone to running endpoint in three commands:

```bash
cd my-client
pip install -r requirements.txt
python train.py                          # trains on the bundled sample data
./serve.sh outputs/customer_support_v1   # serves via FastAPI gateway
```

Replace `scenarios.jsonl` with your client's data — same schema — and you're consulting.

---

## Chat with your fine‑tune locally

```bash
# Interactive REPL — no API server needed, exits cleanly with /quit or Ctrl+D
stateset-agents chat --model Qwen/Qwen3.5-0.8B --checkpoint outputs/acme_v1

# With live reward grading — see scores after every assistant turn
stateset-agents chat --grade customer_support --history conversation.jsonl
```

The chat REPL is the fastest path from "did my fine-tune even load?" to "let me feel how it behaves on the queries that matter." The optional `--history` flag captures every turn to JSONL for later grading or replay; `--grade` shows live composite-reward scores so you can spot reward-function disagreements with your intuition in real time.

## Curate good examples — build the next training set

After capturing many conversations, score them with the same reward function used during training, and curate the high-scoring ones as new training data:

```bash
# Grade every transcript in a directory + collect good examples into one JSONL
make grade-batch DIR=transcripts/ REWARD=customer_support \
                 CURATED=curated.jsonl THRESHOLD=0.7

# One-shot summary across all graded sessions
make grade-batch-summary GRADED_DIR=transcripts/graded
```

The curated file is **idempotent across reruns** — duplicate (prompt, response) pairs are skipped, so you can re-grade as your reward function evolves without polluting the curated set.

This closes the **human-in-the-loop curation cycle**: train → eval → chat → capture → grade → curate → train again.

## Benchmark your fine‑tune

After training, you usually want a defensible number: *did this actually improve over the base model, by how much, and is it reproducible?* The framework ships a Phase‑0 benchmark pipeline across **three tasks** (GSM8K, the bundled customer‑support corpus, and the tool‑calling corpus). Results become publication-grade only after the strict provenance and replication gates below pass.

**Quick path:** open one of the bundled Colab notebooks. The whitepaper §11.7 canonical result was produced by `customer_support_3seed_judge.ipynb` — judge improvement **+0.079** with three-seed agreement on Qwen2.5-0.5B-Instruct ([artifact](benchmark_results/whitepaper_v1/customer_support_3seed_judge_qwen25_05b_instruct.json)).

| Notebook | Task | Runtime on A100 |
|---|---|---|
| **`notebooks/customer_support_3seed_judge.ipynb`** | **Whitepaper §11.7 publication-gate notebook — 3 seeds × dual eval (rubric + LLM judge)** | ~25 min |
| `notebooks/whitepaper_v1_comparative_trainers.ipynb` | TRL GRPO vs GSPO vs DAPO head-to-head on §11.7 protocol | ~45 min |
| `notebooks/whitepaper_v1_gsm8k_benchmark.ipynb` | GSM8K (single‑turn math) — binary reward | ~45 min |
| `notebooks/whitepaper_v1_gsm8k_benchmark_v2.ipynb` | GSM8K — dense-reward A/B variant | ~45 min |
| `notebooks/customer_support_4h.ipynb` | Multi‑turn customer support (single-seed) | ~3 h |
| `notebooks/vllm_speedup_benchmark.ipynb` | HF generate vs vLLM throughput sweep for §6.4 | ~20 min |

See [`notebooks/README.md`](notebooks/README.md) for all ten core notebooks (the four above plus quickstart, tool-calling, curate, SFT-closure, and the standard GSM8K variant). Every notebook is JSON-validated **and lint-checked** in CI via [`scripts/lint_notebooks.py`](scripts/lint_notebooks.py) — pre-flighting against the eight foot-gun patterns from [issue #16](https://github.com/stateset/stateset-agents/issues/16) (asyncio.run in Jupyter, abstract `Agent` base, flash-attn defaults, etc.).

**CLI path** (local A100 / H100):

```bash
# 6-second pipeline health check (no GPU)
make benchmark-smoke

# Run one configuration
make benchmark-phase0 TRAINER=gspo SEED=42

# Full matrix: 3 trainers × 3 seeds × 1 task = 9 runs
make benchmark-phase0-all

# Aggregate JSONs → markdown + CSV + PNG figures + gate report
make release-whitepaper-v1
```

The pipeline:

- **Reproducibility.** `set_all_seeds()` covers Python random, NumPy, PyTorch (CPU + CUDA), and Transformers in one call. Publication evidence carries full source, model, and dataset commits.
- **Schema.** Each run produces a single JSON conforming to `benchmark_results/SCHEMA.md`. Every published number traces back to a file.
- **Publication gates.** Three unique seeds, σ ≤ 0.10, +0.03 improvement, one full source/model revision, real hardware/VRAM/wall-clock evidence, and bounded gradient stability. Synthetic demo rows are excluded by default and can never pass.
- **Figures.** `make benchmark-plot` produces two whitepaper‑ready PNGs (pass@1 per trainer, improvement ranking) plus a matplotlib‑free text fallback.
- **One‑shot release.** `make release-whitepaper-v1` now fails closed unless every gate passes, then aggregates → plots → generates the whitepaper §11.7 markdown snippet → copies figures into `docs/figures/` → writes a release manifest.
- **Neutral comparison.** `benchmarks/shootout.py` executes framework adapters
  from one attested configuration, rotates run order, attempts the entire
  matrix even after individual failures, and writes complete attempt
  accounting. StateSet-versus-TRL evidence is retained today; verl, NeMo RL,
  and OpenRLHF remain required evidence gates rather than implied wins. See
  [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

See `benchmark_results/README.md` for the full pipeline reference.

---

## Offline RL: Learn from logged conversations

Train agents from historical conversation logs without online interaction. Useful when:
- You have existing customer service transcripts
- Online training is expensive or risky
- You want to bootstrap before online fine‑tuning

### Available Algorithms

| Algorithm | Best For | Key Innovation |
|-----------|----------|----------------|
| **BCQ** | Conservative learning | VAE‑constrained action space |
| **BEAR** | Distribution matching | MMD kernel regularization |
| **CQL** | Pessimistic Q‑values | Conservative Q‑function penalty |
| **IQL** | Expectile regression | Implicit value learning |
| **Decision Transformer** | Sequence modeling | Return‑conditioned generation |

### Quick Start

```python
from stateset_agents.data import ConversationDataset, ConversationDatasetConfig
from stateset_agents.training import BCQTrainer, BCQConfig

# Load historical conversations
config = ConversationDatasetConfig(quality_threshold=0.7)
dataset = ConversationDataset.from_jsonl("conversations.jsonl", config)

# Train with BCQ
bcq_config = BCQConfig(
    hidden_dim=256,
    latent_dim=64,
    num_epochs=100,
)
trainer = BCQTrainer(bcq_config)
await trainer.train(dataset)
```

### Hybrid Offline + Online Training

Combine offline pretraining with online GRPO fine‑tuning:

```python
from stateset_agents.training import OfflineGRPOTrainer, OfflineGRPOConfig

config = OfflineGRPOConfig(
    offline_algorithm="cql",
    offline_pretrain_steps=1000,
    online_ratio=0.3,  # 30% online, 70% offline
)
trainer = OfflineGRPOTrainer(config)
trained = await trainer.train(agent, env, reward_fn, offline_dataset=dataset)
```

See `docs/OFFLINE_RL_SIM_TO_REAL_GUIDE.md` for complete documentation.

---

## Sim‑to‑Real Transfer

Train in simulation, deploy to real users. The framework provides:

### Domain Randomization

Generate diverse training scenarios with randomized user personas:

```python
from stateset_agents.training import DomainRandomizer, DomainRandomizationConfig

config = DomainRandomizationConfig(
    persona_variation=0.3,
    topic_variation=0.2,
    style_variation=0.2,
)
randomizer = DomainRandomizer(config)

# Randomize during training
persona = randomizer.sample_persona()
scenario = randomizer.sample_scenario(topic="returns")
```

### Conversation Simulator

Calibratable simulator with adjustable realism:

```python
from stateset_agents.environments import ConversationSimulator, ConversationSimulatorConfig

simulator = ConversationSimulator(ConversationSimulatorConfig(
    base_model="gpt2",
    realism_level=0.8,
))

# Calibrate to real data
await simulator.calibrate(real_conversations)

# Measure sim‑to‑real gap
gap = simulator.compute_sim_real_gap(real_data, sim_data)
```

### Progressive Transfer

Gradually transition from simulation to real interactions:

```python
from stateset_agents.training import SimToRealTransfer, SimToRealConfig

transfer = SimToRealTransfer(SimToRealConfig(
    transfer_schedule="cosine",  # linear, exponential, step
    warmup_steps=100,
    total_steps=1000,
))

# Get current sim/real mixing ratio
sim_ratio = transfer.get_sim_ratio(current_step)
```

See `docs/OFFLINE_RL_SIM_TO_REAL_GUIDE.md` for complete documentation.

---

## Hyperparameter optimization (HPO)

Install with `stateset-agents[hpo]`, then:

```python
from stateset_agents.training import TrainingConfig, TrainingProfile
from stateset_agents.training.hpo import quick_hpo

base_cfg = TrainingConfig.from_profile(
    TrainingProfile.BALANCED, num_episodes=100
)

summary = await quick_hpo(
    agent=agent,
    environment=env,
    reward_function=reward_fn,
    base_config=base_cfg,
    n_trials=30,
)
print(summary.best_params)
```

See `docs/HPO_GUIDE.md` and `examples/hpo_training_example.py`.

---

## Custom rewards

Use the decorator for quick experiments:

```python
from stateset_agents.core.reward import reward_function

@reward_function(weight=0.5)
async def politeness_reward(turns, context=None) -> float:
    return 1.0 if any("please" in t.content.lower() for t in turns) else 0.0
```

Combine with built‑ins via `CompositeReward`.

---

## Custom environments

Subclass `Environment` for task‑specific dynamics:

```python
from stateset_agents.core.environment import Environment, EnvironmentState
from stateset_agents.core.trajectory import ConversationTurn

class MyEnv(Environment):
    async def reset(self, scenario=None) -> EnvironmentState:
        ...

    async def step(
        self, state: EnvironmentState, action: ConversationTurn
    ):
        ...
```

---

## Checkpoints

- `train(..., save_path="...")` saves an agent checkpoint.
- Load later:

```python
from stateset_agents.core.agent import load_agent_from_checkpoint

agent = await load_agent_from_checkpoint("./outputs/refund_agent")
```

---

## Auto‑Research

Run autonomous hyperparameter experiments overnight. The loop proposes configurations, trains with a time budget, evaluates on held‑out scenarios, and keeps only improvements.

```bash
# Quick test (no GPU)
stateset-agents auto-research --stub --max-experiments 5

# Real training with smart proposer
stateset-agents auto-research --proposer smart --improvement-patience 10

# From a config file
stateset-agents auto-research --config config.yaml
```

7 proposer strategies (perturbation, smart, adaptive, random, grid, bayesian, LLM), 5 search spaces, early abort on bad experiments, resume from checkpoint, W&B logging, and post‑run analysis with parameter importance.

```python
# Load and analyze results after a run
from stateset_agents.training.auto_research import ExperimentTracker, compare_runs
tracker = ExperimentTracker.load("./auto_research_results")
tracker.print_summary()
print(compare_runs("./run_a", "./run_b"))
```

See `docs/AUTO_RESEARCH_GUIDE.md` for the full guide.

---

## CLI

The CLI is a thin wrapper around the Python API:

```bash
stateset-agents version
stateset-agents doctor
stateset-agents train --stub
stateset-agents train --config ./config.yaml --no-dry-run --save ./outputs/ckpt
stateset-agents evaluate --checkpoint ./outputs/ckpt --message "Hello"
stateset-agents serve --host 0.0.0.0 --port 8001
stateset-agents auto-research --proposer smart --max-experiments 50
```

For complex runs prefer the Python API and the examples folder.

---

## Examples and docs

**Start here:**
- [`docs/WHITEPAPER.md`](docs/WHITEPAPER.md) — the v0.13.4 technical whitepaper. Anchored to a specific git commit; every claim is verifiable via Appendix C.
- [`docs/WHITEPAPER_ERRATA.md`](docs/WHITEPAPER_ERRATA.md) — corrections published after each whitepaper revision.
- [`docs/PLATFORM_TOUR.md`](docs/PLATFORM_TOUR.md) — a guided walk from `pip install` to a published v1.0 whitepaper revision (linear, journey-style).
- [`docs/COOKBOOK.md`](docs/COOKBOOK.md) — copy-paste recipes for 8 common workflows (look up what you need).
- [`notebooks/README.md`](notebooks/README.md) — a map of the **ten bundled Colab notebooks**: which to open when.
- [`benchmark_results/whitepaper_v1/`](benchmark_results/whitepaper_v1/) — first-party result artifacts including the §11.7 canonical positive result.
- [`CHANGELOG.md`](CHANGELOG.md) — what changed in each release (latest candidate `v0.77.0`).
- [`docs/RELEASE_EVIDENCE.md`](docs/RELEASE_EVIDENCE.md) — retained test,
  provider, GPU, and publication records for earlier releases.

Other entry points:

- **[`examples/getting_started/`](examples/getting_started/)** — **start here after `pip install`**: five small examples (stub hello, custom reward, first GSPO fine-tune, LLM-judge eval, serve via FastAPI). All target the published PyPI version; the GPU-free three smoke-test the install end-to-end. Run `make getting-started-smoke` to verify all three at once.
- `examples/finetune_gspo.py` – **unified finetune driver**: `--model <preset>` over the 20-model registry (`--list-models`), safe `--dry-run` by default, `--no-dry-run` to train
- `examples/hello_world.py` – stub mode walkthrough
- `examples/quick_start.py` – stub-backed onboarding example with training + smoke test
- `examples/complete_grpo_training.py` – end‑to‑end GRPO training
- `examples/train_with_gspo.py` – GSPO + GSPO‑token training
- `examples/train_with_trl_grpo.py` – Hugging Face TRL GRPO integration
- `examples/auto_research_quickstart.py` – autonomous experiment loop

Key docs:

- `docs/AUTO_RESEARCH_GUIDE.md`
- `docs/RL_FRAMEWORK_GUIDE.md` — canonical usage guide
- `docs/GSPO_GUIDE.md`
- `docs/OFFLINE_RL_SIM_TO_REAL_GUIDE.md`
- `docs/HPO_GUIDE.md`
- `docs/CLI_REFERENCE.md`
- `docs/ARCHITECTURE.md`

---

## Related Projects

- [stateset-nsr](https://github.com/stateset/stateset-nsr) - Neuro‑symbolic reasoning engine for explainable tools.
- [stateset-api](https://github.com/stateset/stateset-api) - Commerce/operations API that agents can drive.
- [stateset-sync-server](https://github.com/stateset/stateset-sync-server) - Multi‑tenant orchestration and integrations.
- [core](https://github.com/stateset/core) - Cosmos SDK blockchain for on‑chain commerce.
- Public API docs: https://docs.stateset.com

---

## Industry fine-tuning

Create task-specific training data for financial services, retail, travel,
healthcare, services, media, telecommunications, technology, and the public sector:

```bash
stateset-agents industry init retail ./retail-source
stateset-agents industry prepare retail ./retail-source/examples.jsonl ./retail-run
stateset-agents industry train ./retail-run --dry-run
```

The starters contain synthetic format demonstrations. Replace them with reviewed
conversations; preparation validates tool calls, groups related cases, and records
dataset hashes. Paired evaluation checks base-versus-adapter reference and tool
agreement, retains per-case regressions, and gates on independent source groups
and optional latency/cost limits. Synthetic starter data cannot pass this gate.
See [Industry fine-tuning](docs/INDUSTRY_FINETUNING.md) for training, prediction
callbacks, CLI evaluation, and the distinction from autonomous task success.

## Private improvement workflow

Use `stateset-agents workflow --help` for reviewed dataset intake, River experiments,
paired quality gates, and staging deployment recovery. Customer data and artifacts
must live outside Git checkouts. Start with the offline synthetic fixture in the
[private workflow guide](docs/private-improvement-workflow.md). The public evidence
catalog preserves failed replications and unknown costs alongside successful pilots.

## Contributing

See `CONTRIBUTING.md`. Please run `pytest -q` and format with `black`/`isort` before opening a PR.

---

## License

Business Source License 1.1. Non‑production use permitted until **2029‑09‑03**, then transitions to Apache 2.0. See `LICENSE`.

Pinned frontier checkpoint support now covers MiMo-V2.6-Pro (RL and MOPD),
GLM-5.3, Kimi K3, GLM-5.3-Flash, DeepSeek V4.1 Flash, Qwen3.8 2.4T A95B and
Qwen3.8-Flash-Next. `stateset-agents frontier-models` reports precise readiness.
The four native GLM/Qwen paths have distributed text LoRA implementations;
MiMo, Kimi and DeepSeek retain explicit training-loader blockers. Full-size GPU
training remains unverified. See [the guide](docs/FRONTIER_FINETUNING.md).
