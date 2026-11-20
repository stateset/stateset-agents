# Release highlights before v0.74.0

Moved from the README. The authoritative record is
[CHANGELOG.md](../CHANGELOG.md).

**Platform verification improvements included in v0.67.0:**

- **Evidence-first README refresh.** Release, installation, provider, and
  benchmark status now match `v0.54.0`; the incomplete corrected v3 comparison
  and the remaining A+ gates are stated explicitly.
- **Release-version drift prevention.** README install commands and the
  current-release declaration are guarded against package-version drift and
  updated automatically by the release script.
- **Optional-runtime isolation.** `stateset-agents version --json` reads
  dependency metadata without importing CUDA runtimes, and an incompatible
  installed vLLM now preserves the Hugging Face fallback instead of breaking
  CLI output or package import.
- **Exact benchmark resume identity.** Measured shootout files now carry the
  complete manifest digest, and both local and RunPod resume paths require the
  same manifest, implementation version, seed, and harness commit. Corrected
  experiments cannot silently reuse stale evidence. New evidence also carries
  a portable raw-artifact path that is re-hashed before reporting or resume.
- **Corrected v4 comparison contract.** The pinned 1.5B native-GSPO versus
  direct-TRL manifest validates and enumerates all six rotated pairs without
  provisioning hardware; measured execution still requires an explicitly
  confirmed RunPod spend ceiling.
- **Fail-closed flagship evidence.** A measured flagship run now rejects
  template placeholders and stale StateSet versions before execution, while
  judge-derived scores are bound to the declared judge model, immutable
  revision, and rubric digest.
- **Exact scale/recovery provenance.** Local scaling and fault-injection
  matrices now require a clean checkout and refuse a claimed harness revision
  that differs from the code actually executing.
- **Source-bound scaling images.** A dedicated BuildKit path refuses mutable
  base images, labels the exact commit and package version, requests maximal
  provenance plus an SBOM, then reads SLSA/SPDX/label/manifest evidence back
  from the registry. The A+ gate verifies it and requires every provider
  lifecycle record to use that exact digest.
- **Provider-exported scaling cost.** Kubernetes lifecycle evidence now carries
  exact Job start/finish timestamps. The A+ gate requires a raw-export-bound
  CoreWeave/Nebius billing allocation for every Job, prevents line-item reuse,
  and reports total cost plus cost per measured optimizer step.
- **Artifact-backed release readiness.** Readiness now installs the newly built
  wheel offline into a clean virtual environment outside the checkout, reusing
  the dependency set that passed the preceding gates, and emits portable schema-v2
  evidence for both distributions, coverage XML, and Bandit/pip-audit reports.
  The A+ gate re-hashes and parses those files instead of trusting a bare
  `status: passed` marker.
- **Cross-format backend identity.** A NeMo RL/OpenRLHF/verl conformance roster
  must bind the same canonical dataset-content digest even when the physical
  JSONL and Parquet artifacts necessarily have different byte hashes.
- **Official-suite execution safety.** Measured τ³-bench, BFCL, and
  SWE-bench collection now rejects template commands, zero revisions, stale
  package versions, unbound evaluator model revisions, and dirty harness
  checkouts before invoking an evaluator. Every model-consuming stage is bound,
  and standalone execution verifies the checkout's exact suite revision.

**v0.67.0:**

- Add the fine-tuning terminal studio, agent CLI and HTTP controls, with durable run ownership, checkpoint inspection and validation/test separation.
- Add synthetic ecommerce, logistics and SaaS dataset generation with isolated splits and retained provenance; independent domain review remains required.
- Harden native RL, distributed training, checkpoint recovery, dataset lineage and provider submissions.
- Complete six native River CISPO updates: RL scored 960/960 versus the SFT parent's 954/960 on shared synthetic test cases. Four seeds improved and two tied; the replication gate failed (p = 0.125), with no promotion or production quality certification.
- Retain source-bound training evidence and sampler-stream audits. The documented dependency readiness blocker remains subject to the publication gates.

**v0.66.1:**

- Fix lease checks under database lock contention and use PostgreSQL server time across worker hosts.
- Reject invalid lease durations and expired renewals before changing queue state.
- Keep durable workers processing jobs after lease loss, and finish heartbeat cleanup during repeated cancellation.
- Add regression tests for stale leases, clock differences, and worker shutdown, including live PostgreSQL checks.

**v0.66.0:**

- Add strict rollout contracts, shared trainer checkpoints, outcome rewards, and validation before compute allocation.
- Run training through durable SQLite/PostgreSQL queues and independent workers, with idempotency and fenced leases.
- Add sampled CPU/GPU regressions, reviewed-case studies, installed-wheel compatibility checks, and rollout performance metrics.
- Synthetic CPU/GPU checks pass. The pretrained synthetic refund diagnostic showed no lift; representative quality still requires reviewed cases.

**v0.56.0:**

- Added TLA+ models for rollout admission, policy publication, and
  auto-research checkpoint recovery, checked in CI.
- Added Lean proofs for clipped objectives and reward composition, with Python
  refinement tests.
- Fixed stale rollout admission, timeout accounting, checkpoint recovery, and
  invalid score handling uncovered while modeling these paths.

**v0.55.0:**

- Added one fail-closed A+ decision gate spanning competitive frameworks,
  physical multi-node scaling, reliability, provider canaries, official agent
  suites, flagship quality, and release readiness.
- Added reproducible, provenance/SBOM-bound scaling images plus budgeted RunPod
  and CoreWeave/Nebius Kubernetes launchers with cleanup and cost evidence.
- Upgraded competitive, provider, flagship, and release artifacts with exact
  source/version binding, portable hashes, semantic validation, and adversarial
  regression coverage.

**v0.54.0:**

- Rollouts are always sampled in inference mode: `MultiTurnAgent` wraps every `generate` in `core.generation_mode.inference_mode` (eval mode with the KV cache, the trainer's mode restored afterwards). Sampling in train mode with LoRA dropout and gradient checkpointing active produced garbage rollouts on a real 1.5B model, which is why every retained native-GSPO run had zero reward even after the prompt-wiring and log-prob fixes in 0.53.0.
- `training.callbacks.ZeroSignalGuard` aborts a run whose reward stays identically zero for five consecutive steps; the phase0 harness arms it for native GSPO so a mis-wired run fails within minutes instead of burning hours of GPU.
- The remaining placeholder-prompt query derivation in `train_with_gspo_token` and the auto-research DAPO/VAPO paths now goes through `queries_from_scenarios` as well.

**v0.53.0:**

- Native GSPO trained on a placeholder prompt with a blind reward: `train_with_gspo` derived queries from `scenario["context"]` (fallback "Hello"), only from the first few scenarios, and passed the reward no scenario fields, so every retained native-GSPO shootout run had identically zero reward and loss. Queries now come from the real prompt field with the whole scenario as reward context and rotate through every scenario; the phase0 harness passes GSPO the same task prompts the TRL path uses. The v2 native-GSPO rows are withdrawn (`benchmark_results/framework_comparison_v2/RETRACTION.md`).
- Native GSPO gated ~90% of on-policy samples: the old log-prob came from a separate unbatched forward whose numerical gap to the batched current pass exceeds GSPO's clip band. The old log-probs are now the current pass detached (exactly on-policy, the TRL convention) with the gap reported as `generation_log_prob_gap`.
- `MultiTurnAgent.generate_turns(messages, n)` samples a whole group in one batched generate with exact token ids and sampler log-probs; native GSPO's HF path uses it instead of sequential generation plus rescoring.
- Evidence is fail-closed on learning signal: adapters record `train_reward_{mean,std_mean,zero_fraction,steps}` and both `benchmarks/shootout.py` and `benchmarks/framework_comparison.py` reject a run whose training reward was identically zero at every step.

**v0.52.0:**

- Both GRPO trainers push the policy's weights into the attached rollout backend after every optimizer step (`TrainingConfig.rollout_sync`); turns record the policy version that sampled them, and a backend that cannot take weights is reported once as stale instead of silently sampling from an old policy. `TrainingConfig.old_logprobs_source="sampler"` uses the recorded conditional token probabilities across learner updates.
- Proven live on vLLM 0.28 / NVIDIA A40 for full weights and a LoRA adapter (`benchmark_results/vllm_rollout_check/`): the engine-vs-policy log-prob gap of the sampled tokens returns to bf16 noise after `VLLMGenerator.sync_weights`. `VLLMGenerator` finds the engine model through any vLLM layout, keeps the V1 engine core in-process, and maps PEFT's `.base_layer` weight names.
- Fixed: `from stateset_agents.training import train` returned the `train` submodule once anything had imported it (the API service and CLI do).
- Paid comparison runs are observable while they run (live per-run logs, a status line per launcher poll) and resumable after a pod lifetime cut (`runpod_shootout.py --resume`); resume now accepts only exact-manifest and exact-harness evidence. The legacy v3 protocol (Qwen2.5-1.5B, 200 steps, 256-problem eval) and its L40S envelopes are retained in `benchmarks/`; the corrected matrix must use a fresh protocol and rerun every pair.

**v0.51.0:**

- Retained the validated 48-step, three-seed A40 comparison of StateSet's TRL-backed GRPO, native GSPO, and direct TRL 1.9.1 on Qwen2.5-0.5B / GSM8K (`benchmark_results/framework_comparison_v2/`): throughput parity within run-to-run noise and no learning-quality signal for any implementation at that scale, stated as such.
- `benchmarks/runpod_shootout.py` runs the remote shootout detached and streams every finished evidence file back incrementally, so a dropped launcher session cannot lose completed runs; `framework_comparison.py` skips launcher records by `kind`.
- `MultiTurnAgent.set_rollout_backend(engine)`: route generation through any engine exposing `generate_with_logprobs` (`VLLMGenerator` now does); turns carry the engine's exact token ids and per-token log-probs for the GRPO per-token path.
- Import-time module stubs in the test suite are scoped to the file that installs them, removing cross-file leakage of fake `vllm`, `trl`, and `peft` modules.

**v0.50.0:**

- Per-token GRPO: `MultiTurnAgent.generate_turn` returns the assistant turn with the exact prompt token ids, sampled response ids, and the model's log-probs of those ids; both GRPO trainers request turns and train on one padded forward pass per group with per-token ratios, any token-level objective preset, k3 KL, and entropy on the same logits (sequence-level fallback preserved). Previously the GRPO trainers recorded no log-probs and trained as unclipped REINFORCE on re-tokenised text.
- `TrainingConfig.num_gradient_updates`: PPO/DAPO-style inner updates against frozen old-policy log-probs on the GRPO token path. With sampler scores, clipping can engage on the first update. Metrics report `inner_updates`, `ratio_mean`, `ratio_mean_last`, `clip_fraction`.
- One `StarterSpec` per packaged model family (`training/starter_factory.py`): the twelve starters shrink from ~5,000 to ~2,000 lines with every public name, default, and validation rule unchanged, and a contract test enforces the per-family surface.
- `benchmarks/runpod_shootout.py`: a budget-bounded, self-destructing RunPod launcher for `benchmarks/shootout.py`, with the v2 48-step three-seed Qwen2.5-0.5B GSM8K protocol comparing StateSet's TRL-backed GRPO, native GSPO, and direct TRL.

**v0.49.0:**

- One declarative `PolicyObjective` library (`stateset_agents.training.objectives`) with eleven presets — GRPO, Dr. GRPO, BNPO, DAPO, GSPO, GSPO-token, GEPO, RLOO, REINFORCE++-baseline, CISPO, PPO — verified by explicit-loop references, Hypothesis property tests, and a numeric pin against TRL 1.12 (`docs/OBJECTIVES.md`).
- Every native trainer evaluates its objective through the library; golden regression pins prove the migration changed no numerics except the documented PPO KL fix (k3 estimator, clamped ratios).
- Select any preset from `TrainingConfig.objective` / `objective_overrides`, `stateset-agents train --objective` / `--list-objectives`, and `--objective` on every per-model starter command.

**v0.48.0:**

- **Publication-grade flagship evidence pipeline.** A strict, shell-free runner
  validates immutable 7–9B GSPO manifests, executes exactly three measured
  seeds, retains failures and hashed artifacts, binds provider billing and
  hardware, and rejects publication unless improvement, stability, judge
  agreement, time, and cost gates all pass.
- **CI-enforced benchmark contract.** The example manifest, Make targets,
  maturity contract, tests, benchmark documentation, roadmap, and release
  evidence now share one executable protocol. Dry-run preflight evidence is
  explicitly non-publishable, while live GPU execution remains an honest open
  evidence gate.

**v0.47.2:**

- **Executable multi-node evidence collection.** Added a shell-free collector
  with immutable manifests, exact topology checks, independently measured
  duration, rotated fault order, artifact hashing, retained failures, and
  automatic matrix validation. Preflight output cannot satisfy the publication
  gate.
- **Node 24 release compatibility.** Runtime tests now select the JavaScript
  client suite explicitly while TypeScript remains covered by its compiler
  gate, preventing compile-only fixtures from making network requests during
  npm publication.

**v0.47.1:**

- Completed Modal dataset transport through isolated per-job Volume input and output paths with explicit output commits.
- Added current Modal GPU and multi-GPU resource wiring, named Secret references, regional placement, safe artifact extraction, and failure-artifact recovery.
- Strengthened the approval-gated live Modal canary and added cross-platform behavioral coverage for container mounts and artifact paths.

**v0.47.0:**

- **Restartable distributed rollout control plane.** Remote workers now share
  transport-neutral registration, renewable leases, generation fencing,
  policy-exact admission, capacity reclamation, secret-free health snapshots,
  auditable rejection counters, and quiescent checkpoint recovery. An
  authenticated FastAPI transport adds principal-scoped worker isolation,
  role-gated fleet observability, strict schemas, deterministic error mapping,
  and fixed/chunked request-size enforcement. Content-addressed publication
  binds each policy version to an immutable URI, byte count, and SHA-256;
  workers verify downloaded weights and rollouts retain that digest. Live
  multi-node GPU evidence remains an explicit gate rather than an inferred
  claim.

**v0.46.0:**

- **Policy-safe asynchronous rollouts.** Native producers and learners can
  run independently through a bounded coordinator and executable runtime. It
  rejects future-policy samples, evicts rollouts beyond a configured policy-lag
  window, retains sampler token log-probabilities, deduplicates producer retries
  across checkpoint restore, computes clipped importance weights in log space,
  propagates worker failures, and exposes a new policy version only after its
  weights are published. The initial proof is deterministic and in-process;
  multi-node GPU throughput and recovery remain explicit evidence gates.

**v0.45.0:**

- **Four new managed substrates.** Tinker remote autograd, Prime Intellect Lab
  verifiers/OpenEnv RL, Hugging Face GPU Jobs plus Inference Endpoints, and
  Together managed LoRA fine-tuning now share StateSet's provider lifecycle
  without hiding their different artifact and durability semantics.
- **Inkling-Small support.** `thinkingmachines/Inkling-Small` is cataloged as a
  Tinker-native frontier model with assistant-masked SFT, token-aligned
  importance-sampling RL, sampler checkpoints, resumable state, and honest
  unit-tested/live-pending evidence.
- **Provider-aware CLI and canaries.** `train-remote` accepts explicit job modes
  and provider options, while credential-safe probes cover Tinker, Hugging
  Face, and Together without creating billable resources.

**v0.44.0:**

- **CoreWeave and Nebius are first-class providers.** The same packaged SFT
  job now runs on CoreWeave CKS or Nebius Serverless AI with durable handles,
  S3-compatible artifact exchange, secret references, cancellation, cleanup,
  and fail-closed capability reporting. Separate managed-inference commands
  deploy complete weights to CoreWeave Dedicated Inference or Nebius vLLM
  endpoints without misrepresenting a LoRA adapter as a standalone model.
- **Typed Node.js client.** `@stateset/agents` covers Messages, OpenAI-compatible
  chat completions, model discovery, health checks, JSON/SSE responses,
  authentication, timeouts, cancellation, and structured API errors without
  runtime dependencies.
- **One tag, two registries.** The release workflow verifies Python and Node
  versions against the Git tag, publishes both packages with provenance, and
  creates or updates the GitHub Release explicitly against this repository.
- **Cross-platform documentation guards.** README CLI snippets are checked
  against exact Typer option metadata, independent of terminal width, while
  still executing every referenced command's help path.

**v0.43.0:**

- **Model/provider certification is explicit.** A machine-readable catalog
  separates default, frontier-preview, and compatibility models across
  configured, smoke-tested, training-verified, serving-verified, and
  production-certified stages without promoting architecture support into a
  live claim.
- **RunPod plans before it rents.** `train-remote --plan-only` resolves
  model-aware GPU/count/disk recommendations without provisioning hardware;
  estimated frontier and unknown plans require an explicit `--max-cost` before
  execution.
- **Managed-provider cleanup fails closed.** Modal deletes per-job persistent
  Volumes across every terminal path, Fireworks rolls back deployments after
  addon-load failures, and approval-gated workflows retain frontier, Modal,
  and Fireworks lifecycle evidence.

**v0.42.6:**

- **Cross-platform release confidence.** The complete PR gate now passes on
  Python 3.10–3.13 plus Windows 3.10/3.13, with deterministic API router tests,
  normalized paths, UTF-8 subprocess handling, bounded test timeouts, and a
  stable 61.9% coverage floor.
- **Credential-safe remote execution.** API-key identities use
  domain-separated fingerprints and constant-time WebSocket comparisons, while
  RunPod self-destruct keys stream directly to root-only remote files instead
  of touching plaintext local staging files. Python and Rust security locks
  were refreshed; both RustSec workspace audits report zero vulnerabilities.
- **Reproducible evaluation inputs.** GSM8K downloads default to the validated
  immutable dataset commit, and mutable revision identifiers fail closed.
- **Verified distribution.** The tag-built wheel and sdist passed the complete
  publish-readiness gate, Twine validation, and isolated wheel installation;
  their public PyPI SHA-256 digests match the retained workflow artifacts.

**v0.42.5:**

- **Five native RL objectives, one measured protocol.** A complete 15-run
  RTX 5080/CUDA 12.9 matrix compares GRPO, GSPO, DAPO, VAPO, and GEPO over
  three rotated seeds with pinned Qwen2.5-0.5B-Instruct/GSM8K revisions,
  external wall time, actual completion counts, peak VRAM, held-out scores,
  exact config attestations, and hashed normalized policies. The strict
  [descriptive report](../benchmark_results/algorithm_comparison/report/comparison.md)
  accepts all 15 runs; rejected attempts remain published separately.
- **VAPO now fits the matched 16 GB workload.** The first full attempt exposed
  deterministic graph-retention OOMs at 15.28/15.47 GiB. Prompt-normalized
  gradient accumulation preserves one optimizer step and the same mean
  objective while releasing each forward graph before the next. The corrected
  three-seed VAPO runs average 6,078 MiB peak VRAM and all complete.

**v0.42.4:**

- **Publication-grade comparison gates.** Framework, algorithm, scaling, and
  fault-recovery reports now fail closed on synthetic data, unpinned revisions,
  mismatched hardware/configuration, duplicate seeds, or missing artifacts.
- **Neutral StateSet-versus-TRL execution.** A framework-neutral shootout
  rotates three matched seeds, measures wall time outside each adapter, hashes
  saved models, and independently invokes upstream TRL GRPO. On the exact
  four-step Qwen2.5-0.5B/GSM8K A40 protocol, StateSet and TRL measured the same
  throughput within noise (`0.326` vs `0.327` samples/s), wall time (`196.4s`
  vs `196.0s`), and VRAM (`3436.7` vs `3437.0` MiB); see the
  [three-seed report](../benchmark_results/framework_comparison/report/comparison.md).
- **Live-discovered GRPO fixes.** RunPod execution replaced the optional
  FlashAttention default with SDPA, corrected Phase-0 agent construction,
  bridged TRL's sync reward callback to async rewards, and made returned trained
  agents immediately usable for inference. A rejected diagnostic shootout also
  exposed and fixed seed propagation, short-run warmup, and asymmetric
  evaluation before any comparison result could be published.
- **Measured 1/2/4/8-GPU weak scaling.** Three matched seeds on eight identical
  RTX 5080s reached `2.085×`, `4.004×`, and `8.080×` throughput at 2, 4, and 8
  GPUs (`104.3%`, `100.1%`, and `101.0%` weak-scaling efficiency), with a fixed
  per-device policy workload and `589.8 MiB` peak VRAM per GPU. The
  [validated report](../benchmark_results/scaling/report/scaling.md) labels this
  as weak scaling; the failed fixed-global-batch
  [strong-scaling diagnostic](../benchmark_results/scaling/diagnostics/f6e7478-strong/report/scaling.md)
  is retained rather than hidden.
- **Measured 1/2/4/8-GPU strong scaling.** A corrected fixed-work protocol holds
  the effective global batch at `196,608` samples and divides its 96 reference
  microbatches exactly across ranks. Three matched seeds reached `1.973×`,
  `3.676×`, and `6.642×` throughput at 2, 4, and 8 GPUs (`98.7%`, `91.9%`, and
  `83.0%` strong-scaling efficiency); the [validated strong-scaling report](../benchmark_results/scaling/strong/report/scaling.md)
  binds every throughput value to measured work and wall time.
- **Measured recovery matrix.** Nine CUDA runs injected worker exit, controller
  SIGKILL, and live TCP control-plane interruption across three seeds. Every
  run resumed from its atomic optimizer checkpoint, reached the final step,
  recorded zero lost or duplicate updates, and left zero child resources; see
  the [retained report](../benchmark_results/reliability/report.json).
- **Fresh provider authentication.** Read-only River and RunPod canaries passed
  on 2026-08-27 with zero billable resources and verified cleanup. Fireworks is
  explicitly retained as skipped because its key and account id are absent.

**v0.42.3:**

- **Neutral StateSet-versus-TRL evidence.** Three matched A40 seeds compare the
  same pinned Qwen2.5-0.5B/GSM8K GRPO protocol, including external wall time,
  peak VRAM, and hashed model artifacts.
- **Fail-closed publication evidence.** Comparison reports reject synthetic or
  incomplete inputs, mismatched configurations, duplicate seeds, and unpinned
  model, dataset, algorithm, or framework revisions.
- **RunPod-discovered training fixes.** The release corrected attention
  defaults, Phase-0 construction, asynchronous reward bridging, seed and
  warmup propagation, symmetric evaluation, and trained-agent inference.

**v0.42.2:**

- **Real GPU proof.** The packaged RunPod path completed QLoRA SFT with 2/2
  held-out checks and CUDA GSPO with verified target-probability improvement.
- **Honest verification.** Missing credentials now fail live workflows,
  provider canaries run for release tags, and retained evidence distinguishes
  unit, authentication, hardware, training, inference, and publication claims.
- **Current RunPod provisioning.** Pod requests use explicit GPU compute and
  availability priority, while low-cost SFT/RL lanes run serially to avoid
  competing for scarce Community Cloud capacity.

**v0.42.1:**

- **Deterministic release validation.** The 4,613-test default suite now uses
  bounded parallelism, per-test timeouts, and CI job ceilings, while Redis and
  synchronous health checks fail fast instead of stalling teardown.
- **Auditable model evidence.** `stateset-agents model-support` distinguishes
  framework unit coverage, hardware attempts, and verified inference in
  stable human-readable and JSON output.
- **Consistent deployment artifacts.** Package, Helm, Kubernetes, wheel, and
  documentation version surfaces advance together and are regression-tested.

**v0.42.0:**

- **Bounded multi-GPU RunPod serving.** Day-zero vLLM images support explicit
  GPU counts, tensor parallelism, cost ceilings, readiness deadlines, and an
  independent pod-side watchdog.
- **Persistent caches and safer diagnostics.** Existing network volumes can
  retain Hugging Face weights, while authenticated startup-log tails and
  sanitized provisioning errors improve failure evidence without leaking
  serving tokens.
- **Live evidence stays honest.** Bounded Qwen3.8 and small-model control runs
  cleaned up every pod and isolated dedicated-image/container startup as the
  remaining RunPod bottleneck; no successful-inference claim is made.

**v0.41.0:**

- **Day-one GLM-5.3-Flash and Qwen3.8-Flash-Next support.** Both native
  multimodal composite checkpoints have architecture-verified LoRA targets,
  dedicated dependency extras, safe text-only RL paths, and explicit
  multimodal serving guidance.
- **Composite model loading is shared across agents and RL.** Native
  conditional-generation repositories now fall back from
  `AutoModelForCausalLM` to Transformers' multimodal auto classes in the core
  agent, SFT, GSPO, DAPO, GEPO, and VAPO paths.
- **Provider evidence now stays fresh.** Read-only River, RunPod, and Fireworks
  canaries emit schema-versioned JSON, fail strict CI on missing credentials or
  leaked canary resources, and run alongside a bounded weekly slow/E2E lane.

**v0.40.0:**

- **Remote execution is now explicit and restart-safe.** Provider capability
  discovery prevents unsupported job submissions, Fireworks jobs reconnect
  across process restarts, and RunPod leases make orphan cleanup auditable.
- **API and smoke-test reliability improved.** Redis connections are bounded,
  FastAPI dependencies avoid worker-pool deadlocks, and benchmark smoke mode
  is deterministic and fully offline.

**v0.39.0:**

- **GEPO and VAPO get a real trust region.** Both trainers recomputed
  their "old" log-probs from the current model in the same step, so the
  importance ratio was identically 1 and clipping never fired. They now
  snapshot sampler log-probs (and VAPO's values) at rollout and honor a
  new `num_gradient_updates` (default 1 — the on-policy path is the same
  formula as before; opting in changes the dynamics, since ratios are no
  longer identically 1). Scheduler/`global_step` cadence is now the same
  across DAPO, GEPO and VAPO.
- **No more silent NaN steps.** The GSPO-token trust-region gate selects
  with `torch.where` instead of multiplying (`0 * inf` is gone), and every
  per-token or per-sequence ratio goes through `rl_losses.safe_exp_ratio`,
  which clamps the log-ratio to `±20` (`±30` for GEPO) before `exp`. Note
  the clamp also zeroes the gradient beyond its bounds, on the unclipped
  branch too.
- **One loss implementation, for real.** `GSPOTrainer.compute_group_advantages`,
  the GSPO-token gather and `gspo_generation`'s scorer all call
  `rl_losses`; degenerate groups now return zero advantages and
  `std_reward: 0.0` instead of NaN; `gather_token_logprobs` builds its mask
  in fp32; `VAPOTrainer.compute_token_log_probs(..., response_mask=None)`
  scores only response positions; `logprob_dtype`
  (`"bf16"`/`"fp16"`/`"fp32"`) on DAPO/VAPO configs controls the
  log-softmax precision/memory trade-off.
- **CI that tells the truth.** Benchmarks are collected and run again
  (serially, `-n0`, under the xdist default — they were silently disabled),
  the PyPI upload is split into its own `pypi-publish` job so docs/Docker
  publish regardless, coverage floor 61, bandit's B614 is skipped globally
  (the real control is the AST guard in `tests/unit/test_checkpoint_trust.py`),
  and docs version strings self-maintain via `release.py` with a new
  `test_docs_version_freshness` tripwire.
- **Leaner package, faster suite.** Zero-reference modules removed
  (`utils/advanced_dashboard`, `api/services/request_batcher`), and
  `docs/IMPROVEMENTS_TO_10.md` deleted; the two bottom-of-file import cycles
  in `core/agent.py` are gone; starter-script tests assert forwarding by AST
  instead of spawning a ~9 s interpreter each, and duplicate starter spawns
  are marked `-m slow`, cutting suite wall time substantially.

**v0.38.0:**

- **One loss spine, five trainers.** `training/rl_losses.py` now owns the
  token-logprob gather, group advantages, the k3 KL estimator, the clip
  gate and masked means; GSPO, GSPO-token, GRPO, DAPO, GEPO and VAPO all
  call it. Four silent bugs fell out: GSPO's KL penalty had a
  zero-expectation gradient (no pull toward the reference), GSPO-token had
  no clip gate, GRPO's `token_level_loss` divided by length twice (1/L²),
  and DAPO/GEPO produced NaN advantages on groups of one. Property tests
  pin each: zero advantage ⇒ zero grad, KL ≥ 0 and restoring, out-of-region
  sequences contribute no gradient.
- **Checkpoints are untrusted by default.** Every `torch.load` in the
  package goes through `core/checkpoint_io.load_checkpoint_file` with
  `weights_only=True`; a pickled-object checkpoint raises a `ModelError`
  that names the `trusted=True` opt-in. Configs are saved as plain dicts
  and rebuilt on load. An AST guard keeps it that way.
- **Torch-free entry points, enforced.** `import stateset_agents`,
  `.core`, `.training`, `.cli` no longer import torch (`chat --help`
  3 s → 0.6 s); a committed allowlist + meta-test ratchets every other
  module. `core` no longer imports `experimental` at module level; the
  root re-exports of `Planning*` warn.
- **`cli_train.py` 2851 → 533 lines.** The ten per-model commands are
  generated from `core/model_presets.py` (moved into the package from
  `examples/`); `--help` is byte-identical, and tests pin every
  preset↔starter symbol.
- **Gates that stay green.** Suite runs under xdist by default (~4 min,
  4489 tests, coverage 59.6%); `mypy stateset_agents` at zero errors; a
  README/QUICKSTART snippet test runs every documented CLI command; the
  litellm/wandb atexit traceback is gone; `sitecustomize.py` is gone.

**v0.37.0:**

- **Distillation breaks walls self-training cannot: 9/12 → 11/12.**
  `flywheel --teacher-base-model/--teacher-adapter` — a FIXED teacher
  harvests, the student trains on its successes, the student's evals drive
  the stops. Live: the 9B that walled at 9/12 under both its own SFT and
  RL reached **11/12 in one generation** on the 35B teacher's 97% harvest.
  Patterns the student never samples at any temperature, bought once at
  big-model prices and owned forever at small-model serving cost — through
  River, zero machines rented.
- **The rarity controller** (`--target-harvest-rate`): per-generation
  temperature probes keep the harvest inside the measured ~60% operating
  window. Live on first attempt: probed three temperatures, chose 0.7, and
  the real harvest landed at **59%**.
- **Tool-use episodes — agents that do things, with proof.** Ladder issues
  declare `{tool, args}` actions; training rows teach emitting fenced json
  blocks; episode scoring verifies them deterministically (parse + tool
  name + args subset). Prose that merely claims the action always fails.
  Live: a ladder-trained 9B scored 5/12 on tool-gated episodes.
- **And the failure mode that found: unchecked turns are not
  unconstrained.** Turns without a tool requirement emitted invented tools
  and malformed multi-object json; 113 such episodes were harvested and
  the model trained on them stopped emitting valid actions entirely
  (5/12 → 0/12). Episodes now declare `known_tools` and any junk block
  anywhere fails the episode — verified by rerun: the same start held
  **5/12 instead of collapsing**, with the harvest tightening to 51% clean
  episodes ([`PROOFS.md`](PROOFS.md)).

**v0.36.1:**

- **The wall, cleared by scale: 2/12 → 11/12 → 12/12.** The rung-5
  difficulty (five-turn episodes, 60% refusals, final-turn summaries) that
  held the curriculum-trained 9B at 9/12 under both SFT and RL fell to an
  episode-naive 35B MoE in two flywheel turns — including the refusal
  episode every smaller model left standing, with the harvest rate arcing
  23% → 93% exactly as the measured operating regime predicts. The
  capacity study is complete: the ladder finds each model's ceiling, the
  flywheel drives to it, scale moves it. The run survived a ~12-hour River
  outage mid-experiment via an auto-resuming serving probe.
- **[`docs/GETTING_STARTED_RIVER.md`](GETTING_STARTED_RIVER.md)** —
  train, self-improve, and RL-tune with zero machines rented, written
  entirely from 20+ live runs, including a failure-mode table where every
  row actually happened to us.
- **[`docs/rl-vibe.md`](rl-vibe.md)** — the state of the framework,
  told straight: headline numbers, the compressed experimental arc, what
  broke and taught, and what's honestly not done.

**v0.36.0:**

- **The multi-turn curriculum, climbed and mapped.** N-turn episode
  scripts (`build_episode_ladder(turns=N)`) whose final turn demands a
  summary of everything done: rung 3 lifted 8/12 → **12/12**; the rung-3
  adapter transferred UPWARD to rung 4 at 11/12 and topped off at
  **12/12**; rung 5 found the wall (9/12, held by both methods) — a
  transferable skill, a climbable ladder, and an honestly mapped boundary.
- **Multi-turn RL, live.** Episode-level advantages over whole
  conversations (per-turn prompt ids tokenized client-side, sampler
  logprobs verbatim, shaped episode rewards with completeness bonus and
  refusal penalty). First campaign confirmed the rung-5 wall from the
  second direction: where imitation regressed, RL held stable — a genuine
  model-capacity ceiling, mapped from both sides
  ([`PROOFS.md`](PROOFS.md)).
- **You can watch River runs now.** `STATESET_RIVER_VERBOSE=1` streams
  every executor log line and step ticks (with loss) live — built after a
  slow provider pool turned two runs into silent 45-minute mysteries, one
  of which was a retry loop nobody could see. Scalars are read from
  `ForwardResult.metrics`; step counters reset per retry attempt; harvest
  retries transients like training does.

**v0.35.1:**

- **The first multi-turn ceiling-raise: 8/12 → 12/12 in one generation.**
  N-turn episode scripts (`build_episode_ladder(turns=N)`) where the final
  turn demands a summary of everything done — every earlier resolution
  token plus the never-repeated account reference must reappear in the
  last reply. At rung 3 (three turns, 40% refusals) the
  single-turn-trained 9B scored 8/12 greedy with a 60% episode harvest,
  and one flywheel turn on the 115 passing conversations produced a
  **perfect 12/12** — cross-turn memory and self-summarization, trained
  by the loop, with zero machines rented. The rarity doctrine now stands
  confirmed from both sides: rung 2 (9/12 baseline, 65% harvest)
  honestly plateaued; rung 3 lifted to perfection
  ([`PROOFS.md`](PROOFS.md)).

**v0.35.0:**

- **The multi-turn episode flywheel — the founding claim enters the
  loop.** `build_episode_ladder` generates two-turn scripts where the
  user's second turn raises a new issue and asks for confirmation of the
  first **without ever repeating the account reference** — context
  carryover becomes an objective per-turn check (refusal variants forbid
  the declined remedy episode-wide). River rollouts branch best-of-N per
  script, batched per turn; passing episodes become whole-conversation
  training rows (every assistant turn loss-weighted); harvest summaries,
  post-train evals, and spec validation all speak episodes.
- **First live run — machinery verified, doctrine confirmed.** The
  single-turn-trained 9B already scores 9/12 greedy on two-turn carryover
  (65% episode harvest rate); gen-2 trained on 93 passing conversations
  scored 8/12, and the loop plateau-stopped, keeping gen-1. The honest
  lesson is the flywheel's own doctrine seen from the other side: the
  mechanism amplifies RARE successes — at 65% there was little latent
  capability left to convert at this difficulty. Harder rungs (three-turn
  scripts, higher refusal fractions) are one command away
  ([`PROOFS.md`](PROOFS.md)).

**v0.34.2:**

- **A reward hack, caught and fixed live — the objective/eval separation
  doing its job.** On a fresh ladder-generated domain (GreenGrid Energy,
  Qwen3.5-9B, gen-1 3/3 on held-out accounts), the RL flywheel's v1 graded
  reward Goodharted: mean reward climbed 0.67 → 0.84 while the
  all-or-nothing greedy eval FELL 6/12 → 4/12 — the failure anatomy showed
  the exploit exactly (resolve one issue confidently, drop the rest). A
  +1.0 completeness bonus made the full pass strictly dominant, and the
  A/B rerun from the same checkpoint fixed the collapse: 6/12 → 7/12 with
  reward 1.48/2.0 and no degenerate policy. Both trajectories retained;
  the incident lives in the reward function's docstring and as a
  [`PROOFS.md`](PROOFS.md) row: *RL reward hacking is detectable and
  fixable in-platform.*

**v0.34.1:**

- **The RL flywheel is live-verified — with an honest verdict.** First real
  run (35B MoE, depth-3 refusal kit, zero infrastructure): **7/12 → 10/12
  by round 1**, mean graded reward monotone 0.72 → 0.89 across 4 rounds.
  The head-to-head from the identical start on the identical kit: SFT on
  the 171-row harvest reached **11/12** — so untuned RL lands at rough
  parity, not superiority, and BOTH methods retain exactly one refusal
  violation. Verified, not oversold ([`PROOFS.md`](PROOFS.md)).
- **The ladder's rung 2 stands as its own result**: the depth-2-perfect
  (12/12) model scores 7/12 at depth-3 with refusals, and one SFT
  wheel-turn restores 11/12 — the residual failure being exactly a refusal
  violation, the skill neither method has fully learned.
- Hardening from the runs: River harvests retry transients like training
  does (a live 'Server unavailable' had killed a finished generation's
  run), and RL prompt ids are tokenized client-side and passed to the
  sampler (River echoes none for text prompts).

**v0.34.0:**

- **The eval difficulty ladder** (`stateset_agents.training.eval_ladder`):
  a 35B saturated the hand-written depth-2 eval in one wheel-turn — from
  then on it measured nothing. Difficulty is now a parameter: a declarative
  `DomainSpec` generates train/harvest/eval kits at any compound depth,
  with **refusal prompts** (the user declines one remedy; its proof token
  becomes a `forbid`) that punish template-spraying. First live rung:
  the 12/12 model scores **7/12 at depth-3 with refusals** — headroom
  restored, and the refusal skill isolated as the thing to learn.
- **The RL flywheel** (`flywheel --algorithm cispo`, `--provider river`):
  GRPO-style RL with zero infrastructure. Each round samples best-of-N per
  prompt, grades EVERY sample (expected-resolution fraction, minus a full
  point for violating a refusal), computes group-relative advantages, and
  trains with River's clipped importance-sampling losses on their own
  sampler logprobs in their pre-shifted RL datum layout — failures push
  probability mass away instead of being discarded. Prompt ids are
  tokenized client-side and passed to the sampler, so the datums carry
  exactly the ids generation used. Per-round eval trajectory in
  `rl_report.json`; zero-variance groups skipped. The SFT-vs-RL
  head-to-head on the refusal ladder is running as this ships.

**v0.33.1:**

- **First perfect score: 7/12 → 12/12 in one generation.** A new domain
  (Starlight Travel concierge) on a new model class — Qwen3.6-35B-A3B-FP8,
  a mixture-of-experts — trained and flywheeled entirely through River with
  zero machines rented: gen-1 passed 3/3 held-out bookings verbatim, and
  one wheel-turn on compound requests (harvest 177/240) hit **12/12**,
  firing the perfect-score stop in the wild for the first time. Fourth
  model/substrate/domain combination for the ceiling-raise — and the
  cleanest scaling datapoint yet: 0.8B needed two generations for 7–11/12,
  9B reached 11/12, the 35B MoE maxed the eval in one turn
  ([`FLYWHEEL_DOMAIN2.md`](FLYWHEEL_DOMAIN2.md), [`PROOFS.md`](PROOFS.md)).

**v0.33.0:**

- **The zero-infrastructure flywheel** (`flywheel --provider river`):
  harvest via River's sampling API, train via their remote autograd, no
  machines rented. Live-verified: the TechNest 9B checkpoint went 7/12 →
  **11/12** on compound tickets (harvest rate 67% → 88%) with a correct
  plateau-stop — the ceiling-raise's third substrate
  ([`FLYWHEEL_DOMAIN2.md`](FLYWHEEL_DOMAIN2.md)).
- **`serve-remote --merge` live-verified: hybrid fine-tunes actually
  serve.** The merged model answered greedy training-format tickets in
  full persona over the proxy. Six attempts surfaced five real defects —
  composite checkpoints loaded as themselves, text-trained adapter keys
  remapped (probe delta 0.0 before, real deltas after), processor
  artifacts saved, merge isolated from vLLM's deps, and the release-wheel
  gap — each fixed with tests.
- **Serving is self-verifying**: every adapter serve greedy-probes its own
  effect through the live endpoint (warn, or fail with `--strict`);
  `--merge` refuses on-pod to serve a merge with no observable effect.
  The silent no-op documented in [`docs/PROOFS.md`](PROOFS.md) is now
  structurally impossible to ship.
- **Judge-gated harvests** (`judge` + `min_judge_score` in specs) — the
  step toward real-data flywheels; an unavailable judge rejects rather
  than waving samples through.
- **`flywheel --repeats N`** — score distributions (min/mean/max) under
  one shared budget; motivated by the live 7/12-vs-11/12 seed spread.
- **Weekly flywheel smoke** workflow: one real turn of the wheel every
  Monday, armed the moment the `RUNPOD_API_KEY` secret exists.
- Hardening from live runs: RunPod REST 5xx retries, River transient
  recovery per their SDK's taxonomy with `train_step` preferred,
  readiness-failure evidence that keeps the engine's root cause.

**v0.32.1:**

- **River: training effect verified, not just mechanics.** `train-remote --provider river` trained the 140-row TechNest persona on Qwen3.5-9B (3 epochs, 210 steps through the executor's loop), and sampling the resulting `river://` checkpoint answered **3/3 held-out tickets** with the exact canonical resolutions, the persona signature, and the ticket numbers echoed — the same objective standard the RunPod training path cleared. A scale note for the record: 9B anchored canonical wording at 3 epochs where 0.8B needed 8.

**v0.32.0:**

- **`stateset-agents flywheel` — the improvement loop as one unattended
  command, live-verified and replicated.** Harvest the current generation's
  rare successes (best-of-N against objective checks), train the next
  generation on only those, measure, repeat — stopping on plateau, dry
  harvest, perfect score, or a hard `--max-cost` ceiling checked before each
  rental. Two independent runs in a NEW domain (IT helpdesk) at 35× smaller
  scale (Qwen3.5-0.8B): **0/12 → 7/12 and 0/12 → 11/12** on compound
  tickets, harvest rate rising to 42.7% and 57.9%, ~$3/run
  ([`FLYWHEEL_DOMAIN2.md`](FLYWHEEL_DOMAIN2.md)).
- **River AI: live-verified.** Their SDK landed on PyPI (Python ≥3.12) and
  the first real run worked through our blind-written executor essentially
  unchanged — session → LoRA model on Qwen3.5-9B → training step →
  `river://` checkpoint with lineage manifest → sampled the trained
  weights. Hardened on first contact with their own recovery taxonomy
  (transient backoff + session rebuild; `train_step` preferred).
- **Serve grows up**: repeatable `--adapter name=path` serves several
  fine-tunes on one endpoint for A/B; `stateset-agents deploy` is
  train-then-serve in one command; SSE streaming over the RunPod proxy
  verified live. **Honesty row**: vLLM silently does NOT apply
  hybrid-Qwen3.5 LoRA adapters (greedy A/B proof; `chat-remote` remains the
  verified path — see the DISPROVEN entry in
  [`docs/PROOFS.md`](PROOFS.md)).
- **Failed jobs no longer destroy their artifacts** — the RunPod executor
  salvages the output dir before terminating the pod on non-zero exits (the
  eval gate saves artifacts before failing), and `fetch()` accepts any
  terminal status.
- **`docs/PROOFS.md`** — every headline claim mapped to its evidence class
  (re-proved automatically / live-verified / unit-pinned / unverified /
  disproven), linked from the README.
- `STATESET_AGENTS_WHEEL` ships an unreleased build to rented pods; the
  harvest module moves the model to GPU (an H100 sat at 0% for an hour
  finding this).

**v0.31.0:**

- **The serve claim is now a receipt.** After nine failed attempts across five
  distinct failure modes, `serve-remote` answered an authenticated
  `POST /v1/chat/completions` over RunPod's proxy — first from a hand-driven
  verification run, then through the shipped CLI serving a real fine-tuned
  adapter (`--adapter`). *(Correction, found by a later greedy A/B probe:
  the transport is verified, but vLLM silently does not apply hybrid-Qwen3.5
  LoRA adapters — see [`docs/PROOFS.md`](PROOFS.md); `chat-remote`
  remains the verified way to talk to hybrid fine-tunes.)*
  Two live-only bugs fell out and are fixed with tests: a flashinfer
  annotation that crashes vLLM's engine on Python 3.11 (now patched in place
  post-install) and a shell-precedence bug that made the self-destruct arm
  hold the ssh channel for the pod's whole lifetime.
- **The flywheel raises a ceiling: 0/12 → 2/12 → 10/12.** On compound
  requests provably outside gen-1's training distribution, best-of-8
  rejection sampling harvested gen-1's rare successes and training gen-2 on
  only those took the eval from 2/12 to 10/12, reproduced across two
  independent trainings for $3.32 ([`FLYWHEEL_HEADROOM.md`](FLYWHEEL_HEADROOM.md)).
- **A failed eval gate no longer destroys the adapter it just judged** —
  `wait()` fetches artifacts best-effort on failure too (observed live: a
  10/12 run's adapter had to be retrained because fetch was success-only).
- **River executor aligned to the published docs** while their SDK remains
  uninstallable: canonical `client.session(project=...)` support and a
  one-argument flip (`shift_targets`) for the causal-shift assumption.
- **Fireworks AI provider (`train-remote --provider fireworks`) — code
  complete, *not* live-verified.** A managed fine-tuning service with a
  genuinely asynchronous job: the job id outlives your process. The tuned
  LoRA addon lives on Fireworks, with weights downloaded locally when the
  account allows it — the checkpoint pointer records which happened rather
  than promising a `serve --checkpoint` that would fail. `--deploy` rents
  on-demand hardware and serves the addon behind an OpenAI-compatible URL;
  `undeploy` tears it down, because it bills until deleted. Written against
  the real `fireworks-ai` 1.x SDK, so the call shapes come from the client
  rather than from prose; what is unverified is the service's behaviour,
  itemised with symptoms in [`docs/FIREWORKS_PROVIDER.md`](FIREWORKS_PROVIDER.md).

**v0.30.0:**

- **River AI provider (`train-remote --provider river`) — code complete, *not* live-verified.** River is a remote autograd service: you drive `forward_backward` / `optim_step` yourself. The substantive half of this integration is a pure tokenization layer (`remote/river_batches.py`) turning our chat rows into River's `{input_ids, target_tokens, weights}` with prompt tokens weighted 0.0, plus the `{input_ids, old_logprobs, advantages, attention_mask}` shape their `ppo`/`cispo` losses take — which is where our trainers' advantages would plug in. 92 tests drive it through an injectable client.
- **Why it is unverified, plainly:** `river-client` is not installable from PyPI and our account answers `402 Billing: insufficient_funds`, so no token has been trained. Every assumption is isolated and documented — notably whether `target_tokens` carries the causal shift, which is one function to flip.
- **What probing the live API did establish**, none of which the docs mention: there is a REST surface, it authenticates with `Authorization: Bearer rv_...` (401 without), and an unfunded account returns an OpenAI-shaped `insufficient_funds` envelope. Both account states now raise named, actionable errors rather than a generic training failure.

**v0.29.0:**

- **Qwen3.8-27B, fine-tuned on rented hardware the week it shipped.** `stateset-agents qwen3-8-27b` targets `Qwen/Qwen3.8-27B` (27.8B, multimodal, 256K ctx, Apache‑2.0). Verified live on an H100: 140 support conversations, 3 epochs, **2/2 held-out assertions passed**, 467MB adapter returned, **$0.96**, pod terminated.
- **LoRA inference learned about hybrid attention — and it mattered.** Qwen3.8 puts Mamba-style `linear_attn` (`in_proj_qkv`, `out_proj`, …) in most of its 64 text layers and standard `self_attn` in a minority. Our candidate list only knew llama-style names, so `train-remote` would have adapted the minority and silently skipped the rest. Both families are now targeted, proven on real weights, and the two-pass vision exclusion correctly keeps `out_proj` (it exists in the text stack) while dropping vision-only names. The same fix also improves Qwen3.5, which turns out to be hybrid too.

**v0.28.0:**

- **Every adapter now knows where it came from.** Training writes `stateset_manifest.json` beside the adapter — base model, dataset path *and content hash*, hyperparameters, eval outcome, parent adapter — and `stateset-agents adapters` reads them back as a family tree. Dataset bytes are hashed, not just named: two runs claiming the same file are only comparable if the bytes match.
- **`--gpu-count` is hardware-proven.** A 63GB checkpoint trained sharded across two 48GB cards (`Model sharded across devices: 0=24 module(s), 1=36 module(s)`). The first attempt passed and proved nothing — capacity had swapped in cards big enough to hold the model whole — so device placement is now logged on every run and the retry used a model too large for any single card available.
- **`serve-remote`: two real bugs found and fixed.** Five verification attempts failed identically because we requested the model port as `8000/tcp` and waited for a TCP mapping that RunPod was never going to publish — http ports are served through its proxy. Now requested as `8000/http`, addressed at the proxy URL, with `supportPublicIp` set so a community pod cannot silently start without an IP. Long installs then run detached with their exit code polled, so a dropped ssh link costs a poll instead of the run.
- **Costs are complete.** `chat-remote` and `serve-remote` pods now reach the cost ledger too — the docs had claimed it, only training did it, and serve pods (the ones that outlive their command) were the most expensive omission.
- **Two guides written entirely from runs that happened**: [`RUNPOD_GUIDE.md`](RUNPOD_GUIDE.md) (disk sizing with the exact error a too-small disk produces, GPU/model table, spot pricing, network volumes, multi-GPU, and every failure mode we hit) and [`GETTING_STARTED_API.md`](GETTING_STARTED_API.md) (raw logs to an OpenAI-compatible or Anthropic-style API call).

**v0.27.0:**

- **Money is accounted for.** Every remote run appends what it cost — model, hardware, pod lifetime, dollars — to a per-user ledger, read back with `stateset-agents costs`. `train-remote --max-cost` refuses a run whose worst case would exceed your ceiling, *before* any work starts; a pod the provider won't price is refused rather than rented, because an unknown cost must never render as free.
- **Curation stopped rewarding waffle.** The rule-based grader scored polite-but-useless replies 0.75 — above the curation threshold — which two live experiments measured as precision 0.818 and 0.833. A concreteness/resolution component plus optional persona-fidelity checks and a guarded LLM-judge take `make benchmark-loop` to **precision 1.000 / recall 1.000**, floors ratcheted to 0.95.
- **Durable checkpoints, live-verified.** `--network-volume-id` attaches a RunPod network volume at `/workspace`, so checkpoints survive pod death and the retry path resumes instead of restarting. Proven end to end on rented hardware: volume created, trained against, adapter fetched, volume deleted, zero pods and volumes left behind.
- **`--gpu-count`** for multi-GPU pods (`device_map="auto"` when torch sees more than one device). Shipped unproven and labelled as such; hardware-proven in v0.28.0 above.

**v0.26.0:**

- **The flywheel closes.** `chat-remote` saves every conversation as an ingest-ready transcript (chat → ingest → improve → train-remote); eval prompts gained pass/fail assertions (`expect`/`forbid`/judge scores) — a fine-tune that didn't take now fails the job while preserving artifacts.
- **RL verified on real GPUs.** GSPO ran live on rented hardware for the first time (target-token probability 2.8e-05 → 0.125 in 40 steps); a weekly `rl-live-smoke` job keeps it true.
- **Resilient, cheaper training.** Pod death auto-retries on a fresh pod; `--cloud-type COMMUNITY` (spot pricing) verified live; `--resume` for checkpoint restarts.
- **`serve-remote`** (vLLM OpenAI endpoint with token auth, `--max-hours` self-destruct — verified to terminate even with the client force-killed — `--stop`/`--list`); endpoint bring-up was verified live in v0.31.0.
- **`make release`** — this release was cut with it.

**v0.25.0:**

- **Talk to your fine‑tuned model.** `stateset-agents chat-remote --base-model X --adapter DIR` rents a GPU pod, loads base + adapter, and holds a multi‑turn conversation (SSH‑piped, pod terminated on every exit path; `--prompt` for scripted mode). Live‑verified: a Muse‑Glimmer‑30B adapter resolved "I got double charged for it" to the order number from the previous turn.
- **Dead pods fail fast.** SSH keepalives bound peer-loss detection to ~2 minutes (a pod restarting under a running job previously hung the executor indefinitely).

**v0.24.0:**

- **Fine‑tune and call it, one command.** `train-remote` gains `--eval-prompts FILE` (the job generates base‑vs‑finetuned completions for held‑out prompts, greedy for comparability, returning `eval_results.json` beside the adapter) and `--container-disk-gb` (pod disk sized to the checkpoint — a 63GB model needs ~160). Verified live on an H100: Muse‑Glimmer‑30B tuned on 140 support conversations answers held‑out order numbers in the trained persona while the base model stalls in its reasoning channel.
- **Vision‑exclusion that actually works.** LoRA target inference is two‑pass — leaf names existing only in vision/audio stacks are dropped (peft matches names model‑wide), shrinking multimodal adapters ~13%; base‑eval generation now runs on GPU.

**v0.23.1:**

- **`train-remote` handles large multimodal checkpoints** — four fixes found and verified by training `meta-models/Muse-Glimmer-30B` end-to-end on a rented H100 (63GB BF16 download, LoRA on the text stack, 258MB adapter returned): multimodal-architecture fallback in the SFT loader, configurable RunPod pod disk (`container_disk_gb`), transformers-5.x-proof `TrainingArguments` construction, and vision-tower exclusion from LoRA target inference.

**v0.23.0:**

- **Three new first-class starters.** `stateset-agents qwen3-coder` (`Qwen/Qwen3-Coder-30B-A3B-Instruct`, 256K ctx, Apache‑2.0), `stateset-agents gpt-oss` (`openai/gpt-oss-20b`, 128K, Apache‑2.0; 120B variant flagged multi‑GPU), and `stateset-agents deepseek-v4` (`deepseek-ai/DeepSeek-V4-Flash`, 1M ctx, MIT — GLM‑style QLoRA+vLLM path with MLA‑correct LoRA targets verified from the weight map).

**v0.22.0:**

- **Architecture consolidation.** The four flagship trainers (GSPO/DAPO/GEPO/VAPO) share one model-loading/checkpoint runtime (`training/trainer_runtime.py`); the seven model starters are thin definition layers over `training/starter_common.py`; research modules moved to `stateset_agents.experimental` (old paths warn for one cycle). All RL math and public APIs unchanged — the full suite passes unmodified.
- **NVIDIA Nemotron 3.5 Lightning starter.** `stateset-agents nemotron-3-5` targets `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (hybrid Mamba‑2 + MoE, 30B total / 3B active, 256K ctx, OpenMDW‑1.1) with Mamba-aware LoRA targets. [`docs/nemotron_3_5_starter.rst`](nemotron_3_5_starter.rst)
- **Quality gates that enforce themselves.** Coverage and mypy-allowlist ratchets fail CI when floors fall behind reality; the security workflow's scanners now actually gate (45 HIGH/CRITICAL findings cleared); `make benchmark-loop` scores the improvement loop against planted ground truth (precision 0.818 / recall 1.0); a weekly `gpu-verify` workflow re-proves the training job on rented hardware.
- **Slimmer repo.** `dashboard/` and `mobile/` moved to their own repositories.

**v0.21.0:**

- **Muse Glimmer 30B first‑class starter.** `stateset-agents muse-glimmer` targets `meta-models/Muse-Glimmer-30B` — Meta's open agentic model (Aug 2026; dense 30B, 131K ctx, Apache‑2.0) — with the standard balanced/memory/quality QLoRA profiles, `init --preset muse-glimmer`, and a `muse-glimmer` preset in the unified finetune driver. [`docs/muse_glimmer_starter.rst`](muse_glimmer_starter.rst)
- **RunPod provider for `train-remote`.** `--provider runpod` rents a GPU pod over SSH, runs the same packaged job every other provider runs, and copies the adapter back. GPU defaults are now per‑provider — `RemoteJobSpec.gpu` no longer hard‑codes a Modal‑specific name.

**v0.20.0:**

- **Run the fine‑tune step without a GPU.** `stateset-agents train-remote` runs the SFT job from `improve` on rented compute (`--provider local|modal`), closing the last gap in the improvement loop. The job itself is unchanged whichever provider runs it, and remote runs install a pinned published package rather than syncing your working tree. [`docs/CLI_REFERENCE.md`](CLI_REFERENCE.md)

**v0.19.0:**

- **MCP server.** `stateset-agents mcp` (`pip install stateset-agents[mcp]`) exposes the improvement loop as tools for any MCP client — Claude Code/Desktop or your own agent. Seven tools, stdio transport, dry‑run‑only training. [`docs/MCP_SERVER.md`](MCP_SERVER.md)

**v0.18.0:**

- **Bring your own agent's logs.** `stateset-agents ingest` converts OpenAI chat‑format and LangChain conversation dumps into framework trajectories — logs from agents built *anywhere* plug straight into the loop.
- **The improvement loop in one command.** `stateset-agents improve run` grades, curates, and emits verified‑runnable training commands (a regression test executes every suggestion against the real CLI parsers).
- **Flagship benchmark recipe.** `make flagship-benchmark-all` — a reproducible 3‑seed GSPO run on an 8B model with publish gates ([`benchmarks/FLAGSHIP.md`](../benchmarks/FLAGSHIP.md)).

**v0.16.0 – v0.17.3 (correctness and distribution):**

- **RL‑core correctness overhaul.** All five trainers fixed and behaviorally tested: DAPO freezes rollout‑time old log probs and honors µ inner updates; GEPO runs in log space; GSPO scores exactly the text it sampled and rescores vLLM rollouts; GSPO‑token regained its gradient path; VAPO clips values against rollout predictions with terminal‑token rewards and decoupled GAE; the GRPO loss path uses length‑normalized ratios. A cross‑trainer ratio‑invariant suite guards all of it.
- **Convergence proof in CI.** A nightly job trains a real (tiny) model and asserts the target‑token probability strictly increases — verified against zero‑signal and reversed‑reward controls.
- **API hardening.** Training‑lab routes auth‑gated behind `API_ENABLE_TRAINING_LAB` (off outside development), bounded in‑memory state, fail‑closed production config validation, constant‑time key comparison, identity‑keyed rate limiting.
- **Distribution repaired.** PyPI is current again, wheels ship the runtime config presets, [`stateset-rl-core`](https://pypi.org/project/stateset-rl-core/) is published so `[rust]` and `[full]` resolve, and CI is green across 4 Python versions + Windows.
- **Unified finetune driver.** `examples/finetune_gspo.py --model <preset>` replaced the per‑model script maze with a 12‑model preset registry.

Earlier highlights: v0.15.3 shipped Rust accelerator wheels (abi3‑py310) and model‑level Prometheus inference metrics; v0.15.0 added the getting‑started ladder; v0.13.2 shipped the whitepaper §11.7 three‑seed canonical benchmark (judge improvement **+0.079**, [artifact](../benchmark_results/whitepaper_v1/customer_support_3seed_judge_qwen25_05b_instruct.json)).

Full breakdown in [CHANGELOG.md](../CHANGELOG.md).

---
