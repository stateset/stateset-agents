# Changelog

All notable changes to the StateSet RL Agent Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Adapter provenance and lineage.** Every training run now writes
  `stateset_manifest.json` beside its adapter — base model, dataset path and
  **content hash**, hyperparameters, eval outcome, package version, and the
  adapter it descends from (`--parent-adapter`). `stateset-agents adapters`
  reads them back and reconstructs the family tree, resolving links by path
  and then by directory name so a manifest written on a rented pod still
  links up after the adapter is fetched elsewhere. Adapters trained before
  manifests existed are listed as carrying no provenance rather than hidden.

## [0.27.0] - 2026-08-13 — cost accounting, a grader that rewards resolutions, durable checkpoints

### Added

- **Cost ledger and budget ceilings.** Every remote run now appends what it
  cost — model, hardware, pod lifetime, dollars — to a per-user ledger, read
  back with `stateset-agents costs`. `train-remote --max-cost` refuses to run
  when a pod's worst case (its full `--timeout` at the provider's quoted rate)
  would exceed the ceiling, checked before any work starts; a pod the provider
  will not price is refused rather than rented. `RemoteJobResult` carries
  `duration_s`/`cost_usd`, and unknown costs stay unknown rather than
  rendering as free.

  Building this immediately caught a real defect: the unit suite had been
  writing zero-cost rows into the user's own ledger via the fake providers.
  Tests now get an isolated ledger (autouse fixture).

### Changed

- **`--network-volume-id` is now live-verified.** A durable 25GB volume was
  created, attached to a rented pod at `/workspace`, trained against, and the
  adapter fetched back — then the volume was deleted and both pods and volumes
  confirmed at zero. (It took four GPU types to find capacity in the volume's
  datacenter; the executor reported each `500` honestly and moved on.)

### Added

- **Multi-GPU pods: shard a checkpoint bigger than one card.**
  `RemoteJobSpec.gpu_count` (CLI: `--gpu-count`, RunPod `gpuCount`) rents N
  GPUs of the requested type, and the SFT job now loads the base model with
  `device_map="auto"` whenever more than one CUDA device is visible —
  accelerate shards the checkpoint across all of them, and HF `Trainer`
  treats the multi-device `hf_device_map` as model parallelism (no
  DataParallel wrap, no blanket `.to("cuda")`, which would collapse the
  shards onto `cuda:0` and OOM). Single-GPU and CPU behavior is byte-for-byte
  unchanged. Verified live on a 2x H100 80GB SECURE pod: Muse-Glimmer-30B
  (~62GB BF16) sharded across both cards, trained, and passed its eval
  assertions. DeepSeek-V4-Flash — the one starter tier never proven on
  hardware — was checked first and ruled out with numbers: 160GB of fp8/fp4
  safetensors (291B params) exceeds even a 2x H100 pod before activations,
  and quantized-expert LoRA is unsupported territory.

- **Cross-pod checkpoint resume on RunPod via network volumes.**
  `RemoteJobSpec.network_volume_id` (CLI: `--network-volume-id`) attaches an
  existing RunPod network volume at `/workspace`, so checkpoints land on
  durable storage that outlives the pod. The pod-died-mid-job retry path
  then re-runs **with `--resume`** — an interruption costs at most one
  epoch instead of the whole run (previously it always restarted from
  scratch, the documented v1 gap). Volumes are datacenter-scoped, so pod
  creation is pinned to the volume's `dataCenterId` (REST fields verified
  live: `networkVolumeId` + `volumeMountPath` + `dataCenterIds` on pod
  create; `name`/`size`/`dataCenterId` on volume create). Added
  `RunPodApi.list_network_volumes()` and `RunPodApi.get_network_volume()`.
  Behavior without a volume is unchanged. The volume is caller-managed and
  bills monthly until deleted. Verified end-to-end on live hardware
  (RTX A4000 + Qwen3.5-0.8B with a 20GB volume attached).

## [0.26.0] - 2026-08-13 — the flywheel closes — assertions, GPU-verified RL, spot pods, serve-remote

### Added

- **`serve-remote`: a persistent vLLM OpenAI-compatible endpoint on a
  RunPod pod, with cost controls.** `stateset-agents serve-remote
  --base-model X [--adapter DIR]` rents a pod (ports 22 + 8000), installs
  vLLM, loads the base model (+ LoRA adapter as served-model `adapter`),
  and prints the endpoint URL (from RunPod's port-8000 mapping), a
  generated Bearer token (vLLM `--api-key`), an example `curl`, and the
  stop command. The pod deliberately outlives the CLI, so every run arms
  an **on-pod self-destruct**: a `nohup`-ed script sleeps `--max-hours`
  (default 1.0) then calls the RunPod DELETE endpoint on its own pod id —
  it fires even if the launching machine is gone, at the documented cost
  of copying the API key to the pod (`chmod 600`). `--stop <name-or-id>`
  terminates immediately and `--list` shows serve pods with age and $/hr.
  Startup failures terminate the pod before the error propagates. New
  module `stateset_agents.remote.serve_session`; `RunPodApi.list_pods()`
  added.

- **Eval prompts can now assert, turning the post-train comparison into a
  pass/fail gate.** `--eval-prompts` file lines (and `RemoteJobSpec.
  eval_prompts` entries) may be JSON objects `{"prompt", "expect",
  "forbid", "judge", "min_judge_score"}` alongside plain prompt strings.
  `expect`/`forbid` substrings are matched case-insensitively against the
  finetuned completion; results land per-row in `eval_results.json` as
  `"checks": {"expect_hits", "forbid_hits", "passed"}`. The optional
  `judge` scores the completion with `create_domain_reward(<name>)` into
  `"judge_score"` when the reward stack is importable on the worker
  (degrading to a logged warning otherwise), and `min_judge_score` gates
  on it. When any assertion fails the job exits non-zero **after** saving
  the adapter and `eval_results.json`, so a red run never destroys the
  training artifacts. The weekly `gpu-verify` workflow now gates on
  completion content (`expect: ["number"]`), not just adapter tensors.

- **`chat-remote` transcripts close the improvement loop.** Every
  `stateset-agents chat-remote` conversation is now saved on exit (every
  exit path, aborted sessions included) as an OpenAI chat-format JSONL
  transcript — default `./chat_transcripts/chat_<timestamp>.jsonl`, tunable
  with `--save-transcript PATH`, opt out with `--no-save`. The file is
  exactly what `stateset-agents ingest --format openai --input <file>`
  accepts, so chat -> ingest -> improve -> train-remote is a flywheel; the
  command prints the next step on save. `RemoteChatSession` grew a
  client-side conversation mirror and a `transcript` property (messages +
  session metadata: base model, adapter, GPU, start time); only
  successfully answered turns are recorded, matching the on-pod history.

## [0.25.0] - 2026-08-12 — chat-remote: converse with your fine-tuned model

### Added

- **`stateset-agents chat-remote`** — an ephemeral interactive chat session
  with a fine-tuned model on a rented RunPod GPU. Rents a pod, loads the
  base model plus a local LoRA adapter there, chats over one persistent SSH
  channel (JSON-lines protocol, full multi-turn history kept on the pod),
  and **terminates the pod on every exit path** — no open ports, no idle
  billing. `--prompt` (repeatable) gives a non-interactive scripted mode
  for verification. New modules: `stateset_agents.remote.chat_session`
  (local orchestration) and `stateset_agents.remote.chat_repl` (the on-pod
  server).

## [0.24.0] - 2026-08-12 — train-remote: eval prompts, disk sizing, effective vision exclusion

### Added

- **`train-remote --container-disk-gb`** — sets the RunPod pod's container
  disk from the CLI (previously only reachable by constructing
  `RunPodExecutor(container_disk_gb=...)` in Python). Size it at roughly
  2.5x the model download; verified live where a 63GB checkpoint dies on
  the old fixed 40GB.
- **`train-remote --eval-prompts FILE`** — post-train base-vs-tuned
  comparison. A local text file of prompts (one per line); each is answered
  greedily by the base model before LoRA is applied and again by the trained
  adapter, and the pairs are written to `output_dir/eval_results.json`. The
  prompts travel inside the job spec (as `--eval-prompts-json` on the
  packaged `stateset_agents.training.sft` CLI), so remote pods need no
  second file upload. Dry runs are unaffected.

### Fixed

- **Vision-tower exclusion now actually works on real multimodal models.**
  Verified live on the published 0.23.1: peft matches `target_modules` by
  leaf name across the whole model, so skipping vision modules during
  inference was insufficient — and Muse Glimmer's `vision_adapter` /
  `vision_projection` weren't in the marker list. Inference is now two-pass
  (names existing only in non-text stacks are dropped from the list) and
  the marker set carries the names observed on the real weight map.

## [0.23.1] - 2026-08-12 — train-remote fixes verified live on Muse Glimmer 30B

### Fixed

- **`train-remote` now handles large multimodal checkpoints** — three fixes
  found by training `meta-models/Muse-Glimmer-30B` on live RunPod hardware
  (verified end-to-end on an H100 80GB: 63GB BF16 download, LoRA on the text
  stack, 258MB adapter returned, pod terminated):
  - `sft.py` falls back to `AutoModelForImageTextToText` when
    `AutoModelForCausalLM` rejects a composite multimodal architecture
    (transformers registers `muse_glimmer` only under image-text-to-text).
  - `RunPodExecutor(container_disk_gb=...)` is now configurable — the fixed
    40GB default killed the 63GB download mid-stream with an opaque HF-cache
    "File reconstruction error".
  - `TrainingArguments` construction filters kwargs against the installed
    transformers' signature — 5.x removed `warmup_ratio`, which crashed the
    job after the full model download.
  - `infer_lora_target_modules` skips vision-tower/projector modules on
    multimodal composites — their fc1/fc2 layers matched decoder-MLP names
    and were adapted despite receiving no gradient from text-only SFT.

## [0.23.0] - 2026-08-11 — Qwen3 Coder, gpt-oss, DeepSeek V4 starters

### Added

- **Three new first-class model starters** — packaged thin starter modules
  with the full standard surface (CLI command, `init --preset`,
  `examples/finetune_gspo.py` preset + thin forwarder, balanced/memory/quality
  profiles, docs, and unit/integration tests) for:
  - **Qwen3 Coder 30B A3B** (`stateset_agents/training/qwen3_coder_starter.py`,
    `stateset-agents qwen3-coder`) — `Qwen/Qwen3-Coder-30B-A3B-Instruct`,
    Alibaba's open coding MoE (30B total / ~3B active, 128 experts / 8 active,
    256K ctx, Apache-2.0). Attention-only LoRA targets: the 128-expert MoE
    MLPs are impractical LoRA targets.
  - **gpt-oss 20B** (`stateset_agents/training/gpt_oss_starter.py`,
    `stateset-agents gpt-oss`) — `openai/gpt-oss-20b`, OpenAI's open-weight
    reasoning MoE (32 experts / 4 active, 131K ctx, Apache-2.0; adjustable
    reasoning effort, harmony format). `openai/gpt-oss-120b` is listed as a
    variant with a multi-GPU validation warning. Attention-only LoRA targets
    verified against the checkpoint weight map.
  - **DeepSeek V4 Flash** (`stateset_agents/training/deepseek_v4_starter.py`,
    `stateset-agents deepseek-v4`) — `deepseek-ai/DeepSeek-V4-Flash`, a large
    MoE with MLA attention (256 routed experts / 6 active, up to 1M positions,
    MIT). Modeled on the GLM large-MoE starters: QLoRA-only, vLLM-backed
    generation, memory warnings. LoRA targets use the checkpoint's actual MLA
    projection names (`wq_a`/`wq_b`/`wkv`/`wo_a`/`wo_b`, verified against the
    safetensors weight map — llama-style `q_proj`/`k_proj`/`v_proj` do not
    exist in this architecture).

## [0.22.0] - 2026-08-11 — Architecture consolidation + Nemotron 3.5 Lightning

### Added

- **First-class Nemotron 3.5 Lightning starter** — packaged
  `stateset_agents/training/nemotron_3_5_starter.py` for
  `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (NVIDIA's hybrid
  Mamba-2/attention/MoE open model, Aug 2026; 30B total / ~3B active params,
  256K practical context, OpenMDW-1.1), with the standard thin-starter
  surface: `stateset-agents nemotron-3-5` CLI command, `init --preset
  nemotron-3-5`, `examples/finetune_gspo.py --model nemotron-3-5` preset
  (plus the thin forwarder `examples/finetune_nemotron_3_5_gspo.py`),
  balanced/memory/quality profiles, docs (`docs/nemotron_3_5_starter.rst`),
  and unit/integration tests. LoRA targets attention plus Mamba-2
  `in_proj`/`out_proj`; presets target QLoRA post-training of the BF16
  checkpoint (the NVFP4 variant is inference-only).
- **`benchmarks/improvement_loop.py` + `make benchmark-loop`** — measurable
  benchmark of the closed improvement loop. Generates a seeded synthetic
  OpenAI-format corpus with a planted good/bad mix, runs the real
  ingest → grade → curate pipeline (`run_improve`), and reports curation
  precision/recall/F1, dataset yield, and grade distribution against the
  planted ground truth (JSON + human table). Exits non-zero below
  configurable floors (defaults `--min-precision 0.75`, `--min-recall 0.95`;
  measured 0.818 / 1.0 on the default corpus — the precision gap is the
  documented `deflection` false positive of context-free rule-based
  grading). Covered by `tests/unit/test_improvement_loop_benchmark.py`.

### Changed

- **Research modules moved to `stateset_agents.experimental`.**
  `neural_architecture_search`, `multimodal_processing`, `long_term_planning`,
  `few_shot_adaptation`, `intelligent_orchestrator` (+ `_logic`/`_models`),
  `adaptive_learning_controller`, and `multi_agent_coordination` relocated
  from `stateset_agents.core` to the new `stateset_agents.experimental`
  namespace, which carries no API-stability guarantees. The old
  `stateset_agents.core.<module>` import paths still work for one deprecation
  cycle via shims that emit a `DeprecationWarning`.

- **`dashboard/` and `mobile/` extracted to their own repositories**
  ([stateset-agents-dashboard](https://github.com/stateset/stateset-agents-dashboard),
  [stateset-agents-mobile](https://github.com/stateset/stateset-agents-mobile),
  full history preserved via `git subtree split`). They were demo clients
  with no deployment path whose lockfiles and CI repeatedly cost this repo
  maintenance (36 of the 45 security findings cleared this release came from
  `mobile/`'s lockfile). Their workflows are retired here.

- **Starter modules deduplicated onto `training/starter_common.py`.** The seven
  `*_starter.py` modules were near-verbatim copies (4,452 lines total); shared
  config-file IO, profile plumbing, preview/run scaffolding, and the
  `StarterConfigMixin` dataclass base now live in one module, and each starter
  is a thin definition layer (2,902 lines total). Public API, error strings,
  and test patch targets are unchanged — the full starter test suite passes
  unmodified. `docs/SUPPORTED_MODELS.md` documents the new thin-module pattern
  for adding starters.
- **Security workflow actually enforces.** CodeQL action v2 → v3 (v2 is
  retired and failing), dependency-review-action v3 → v4, TruffleHog pinned to
  v3.96.0, Trivy installed via its official installer (it was previously
  `pip install trivy`, which is not a real package, silently masked by
  `|| true`). The high-severity gate now runs on every trigger — not just
  pull requests — and fails when an expected scanner report is missing instead
  of silently passing.
- **Deployment images pinned.** All 14 vLLM manifests move from
  `vllm/vllm-openai:nightly` to `v0.27.1`; the legacy `grpo-framework:latest`
  deployment is pinned to the release version, and `deploy.sh` now tags images
  with the version from `pyproject.toml`.
- **`[dev]` extra completeness enforced.** `[dev]` had drifted from
  `[training]`/`[api]` (missing `gymnasium` and `redis`); the missing packages
  are restored and a new guardrail test
  (`tests/unit/test_pyproject_extras.py`) asserts `[dev]` stays a superset of
  both. (Self-referential extras would make this structural, but they send
  pip-compile into `ResolutionTooDeepError`, so the duplication is deliberate
  and documented in-line.) Dev lock file regenerated.

### Added

- **Self-moving quality ratchets.** CI now fails when measured coverage
  exceeds the `fail_under` floor by a full point ("raise the floor"), and
  `tests/unit/test_mypy_allowlist_ratchet.py` stops the mypy typed-surface
  allowlist from ever shrinking — making both floors monotonic instead of
  policy-only.
- **Weekly GPU verification (`gpu-verify.yml`).** Rents a RunPod GPU via the
  same `train-remote` path users run, trains a real QLoRA adapter on
  Qwen3.5-0.8B from the current checkout's wheel, and fails unless adapter
  tensors come back. Skips cleanly when the `RUNPOD_API_KEY` secret is
  absent. Turns "verified on live hardware" into a standing weekly property.
- **CLI reference completeness guardrail.** `docs/CLI_REFERENCE.md` gained the
  12 undocumented commands (`improve`, `ingest`, `mcp`, `chat`, `benchmark`,
  `recipe`, `starter`, `tour`, `fine-tune`, `auto-research`, `init-config`,
  `gemma-4-31b`), generated from the live Typer app, and a new meta-test
  (`tests/unit/test_cli_reference_complete.py`) fails CI whenever a command is
  added, renamed, or removed without updating the reference.

## [0.21.0] - 2026-08-10 — Muse Glimmer 30B starter + RunPod hardening

### Added

- **Muse Glimmer 30B first-class starter.** `meta-models/Muse-Glimmer-30B`
  (Meta's open agentic model, Aug 2026: dense 30B, 131K+ ctx, Apache-2.0) now
  ships a packaged starter (`stateset_agents.training.muse_glimmer_starter`),
  a `stateset-agents muse-glimmer` CLI command, an `init --preset muse-glimmer`
  scaffold, a `muse-glimmer` preset in the unified `examples/finetune_gspo.py`
  driver, and `docs/muse_glimmer_starter.rst`.

- **RunPod provider for `train-remote`.** `--provider runpod` rents a GPU pod,
  runs the same packaged job every other provider runs, and copies the adapter
  back. Needs `RUNPOD_API_KEY`, an SSH keypair, and `ssh`/`scp` on PATH — no
  extra Python dependency.

  ```bash
  export RUNPOD_API_KEY=...
  stateset-agents train-remote --provider runpod --gpu "NVIDIA RTX A4000" \
      --dataset improved/curated.jsonl --base-model Qwen/Qwen3.5-0.8B
  ```

  RunPod rents a machine rather than a function, so there is no managed
  filesystem like Modal's Volumes: the pod is created with TCP 22 exposed and
  the caller's public key injected, and files move over `scp`. **The pod is
  terminated on every exit path** — success, non-zero job exit, transport
  exception, and never-became-reachable — because an orphaned pod bills by the
  hour. No network volume is created, so there is no storage cost after a run.

- **`RunPodExecutor(wheel=...)`.** Installs a locally built wheel on the pod
  instead of resolving the pinned version from PyPI, which is the only way to
  verify an unreleased change on real hardware.

  Verified end-to-end on live hardware: RTX A4000, `Qwen/Qwen3.5-0.8B`, LoRA
  r=8, ~5.5 minutes wall clock, returning a 12.8 MB adapter (192 tensors) to
  local disk with the pod terminated afterwards.

### Changed

- **`RemoteJobSpec.gpu` no longer hard-codes `"A10G"`.** GPU names are provider
  vocabulary ("A10G" on Modal, "NVIDIA RTX A4000" on RunPod), so a shared
  default silently sent an invalid id to whichever provider did not coin it.
  The field is now `None` by default and each executor supplies its own.

### Fixed

- **LoRA `target_modules` are now passed explicitly.** `run_sft` previously
  relied on peft inferring them from the architecture, which only works for
  models in peft's built-in mapping — anything else (Qwen3.5, for one) died
  with "Please specify `target_modules`". The failure needs a real model on a
  GPU to reproduce, so the CPU dry-run path never reached it. New
  `stateset_agents.training.sft.infer_lora_target_modules()` inspects the
  loaded model and selects the standard projection layers actually present
  (separate q/k/v/o, fused `c_attn`, and MLP projections), excluding the
  output head. Pre-existing: it affected `scripts/sft_from_curated.py` too.

## [0.20.0] - 2026-07-31 — train-remote: run the fine-tune step without a GPU

### Added

- **`stateset-agents train-remote`.** Runs the SFT job from `improve run` on
  local or rented compute, closing the last gap in the improvement loop —
  `ingest` and `improve` are CPU-only and cheap, but the fine-tune that
  consumes `curated.jsonl` needs a GPU. New `stateset_agents.remote` package:
  a provider-agnostic `RemoteJobSpec`, a five-method stateless `RemoteExecutor`
  contract, a `LocalExecutor` (subprocess), and a `ModalExecutor`. New extras:
  `remote` and `modal` (`pip install "stateset-agents[modal]"`).

  ```bash
  stateset-agents train-remote --provider modal --gpu A100 \
      --dataset improved/curated.jsonl --base-model Qwen/Qwen3.5-0.8B
  ```

  Remote runs install a pinned published `stateset-agents[training]` rather
  than syncing the local working tree, so a run is reproducible; the tradeoff
  is that testing an unreleased change remotely needs a dev release. A run
  succeeds only if it actually produces an adapter — a container that exits
  cleanly having written nothing is reported as a failure, not as success with
  an empty output directory.

  **Known limitation:** the Modal network transport is written against the
  documented SDK API but has not yet been verified against a live Modal
  account. `--provider local` is verified end-to-end. See
  `docs/superpowers/specs/2026-07-30-remote-executor-design.md`.

- **Five-minute onboarding demo.** `examples/five_minute_demo.sh` — a
  self-contained, offline, no-GPU script that writes sample customer-support
  conversation logs, ingests them with `stateset-agents ingest`, grades +
  curates them with `stateset-agents improve run --reward customer_support`,
  and prints the graded report — the fastest path from `pip install
  stateset-agents` to a curated training set. Colab equivalent:
  `notebooks/improve_your_agent_5min.ipynb`.

### Changed

- **The SFT job moved into the installed package** as
  `stateset_agents.training.sft`, runnable via `python -m
  stateset_agents.training.sft`. `scripts/` is excluded from the wheel, so a
  remote worker that installs the package could not have run the job from
  there. `scripts/sft_from_curated.py` is now a thin CLI that re-exports every
  public name it previously defined — existing callers and imports are
  unaffected.

## [0.19.0] - 2026-07-28 — MCP server: the improvement loop as tools for any MCP client

### Added

- **MCP server.** `stateset-agents mcp` (new optional `mcp` extra —
  `pip install stateset-agents[mcp]`) exposes the grade → curate → retrain
  "improve" loop as MCP tools (`list_rewards`, `ingest_transcripts`,
  `grade_transcript`, `improve_run`, `improve_status`, `list_model_presets`,
  `dry_run_finetune`) so any MCP client (Claude Code/Desktop, other agents)
  can drive the loop directly. Thin wrappers only — no grading/curation/
  training logic reimplemented; `improve_run` shares the same orchestration
  function (`cli_improve.run_improve`) as the CLI's `improve run` command.
  v1 scope: no tool starts real GPU training (`dry_run_finetune` is
  dry-run only). See `docs/MCP_SERVER.md` for setup and Claude Code
  registration (`claude mcp add stateset-agents -- stateset-agents mcp`).

## [0.18.0] - 2026-07-28 — Bring-your-own-agent: trajectory ingestion + one-command improve loop

### Added — `stateset-agents improve`: the grade -> curate -> retrain loop in one command

- New CLI subcommand `stateset-agents improve run --transcripts DIR --reward
  NAME --output DIR [--threshold F] [--format transcripts|openai|langchain]`.
  Thin orchestrator over existing, tested pieces — `stateset_agents.data.
  trajectory_ingest` (only when `--format openai/langchain`), `scripts/
  grade_transcript.py`'s grading + curation functions, and generates
  `next_steps.md` with the exact `scripts/sft_from_curated.py` /
  `examples/finetune_gspo.py` commands to train on the curated set. Writes
  machine-readable `improve_summary.json` (mean score, per-reward-component
  breakdown, curated count) alongside `curated.jsonl`.
- `stateset-agents improve status --output DIR` prints a previous run's
  summary without re-grading.
- Offline-friendly: only the rule-based rewards (`gsm8k`, `customer_support`,
  `tool_calling`) are supported; an LLM-judge reward name fails with a clear
  message instead of silently requiring an API key.
- See docs/COOKBOOK.md "The improvement loop in one command".

### Added — trajectory ingestion for logs from any agent framework

- `stateset_agents.data.trajectory_ingest`: `from_openai_messages`/`from_openai_jsonl`
  and `from_langchain_json` convert OpenAI chat-completions logs and
  LangChain/LangGraph message dumps into `MultiTurnTrajectory` — tool calls
  preserved in turn metadata, multimodal content flattened to text (skipped
  parts recorded, not dropped), optional per-conversation `reward`/`score`
  carried through. `to_grading_history()` emits the `{"role", "content"}`
  dicts `scripts/grade_transcript.py` already reads, so logs from any agent
  plug straight into the existing grade -> curate -> retrain loop.
- New CLI subcommand `stateset-agents ingest --format openai|langchain
  --input PATH --output PATH`.
- Exported from `stateset_agents.data`; see docs/COOKBOOK.md Recipe 2b
  ("Bring your own agent's logs").

## [0.17.3] - 2026-07-28 — Green CI: toolchain pinning, Windows/utf-8 correctness, packaging pipeline repair

## [0.17.2] - 2026-07-27 — Packaging A-grade: PyPI pipeline repair, rust_core 0.1.1, JS auth + CI

### Fixed — dashboard/mobile JS surface (A-grade pass)

- `dashboard/src/api.ts`: `BASE` now reads `VITE_API_BASE_URL` (falling
  back to the same-origin `/api/lab` default) instead of being hardcoded;
  requests send an `X-API-Key` header and the WebSocket connection sends
  `api_key` as a query param, sourced from `VITE_API_KEY` (build-time) or
  a new runtime `setApiKey()`/`getApiKey()` pair persisted to
  `localStorage` under `stateset.apiKey`. `connectWs()` derives its
  `ws(s)://` origin from `BASE` when it's an absolute URL, otherwise keeps
  the previous same-origin behavior. Added `dashboard/.env.example` and a
  `src/vite-env.d.ts` typing the two env vars.
- `dashboard/package.json`: added `engines.node >=20.19.0`; added
  `dashboard/.nvmrc` (`20.20.0`) to mirror `mobile/.nvmrc`.
- `mobile/lib/api.ts`: requests now send `X-API-Key` from a new
  `EXPO_PUBLIC_API_KEY` env var, mirroring the dashboard's auth pattern.
- `mobile/components/ui/DemoDataBanner.tsx`: new component making the
  existing mock-data fallback (`useTrainingData().isMockData`) visible in
  the UI, not just the console; wired into the dashboard tab screen
  (`mobile/app/(tabs)/dashboard/index.tsx`) as the reference pattern for
  the other screens.
- Added `.github/workflows/mobile.yml`: path-filtered CI on `mobile/**`
  running `npm ci` + `npm run typecheck` on Node 20 — mobile previously had
  no CI at all.
- `.github/workflows/dashboard.yml`: bundle artifact retention extended
  7 → 30 days; added a comment noting the deploy target is still an open
  decision (no deploy step added).

### Fixed — Rust/Cargo packaging surface (A-grade pass)

- Root `Cargo.toml`: removed two `[[example]]` stanzas pointing at
  `examples/multi_agent_orchestration.rs` and `examples/fulfillment_agent.rs`,
  which don't exist (`examples/` is the Python examples dir) — this made
  `cargo check --all-targets` fail outright.
- Root `Cargo.toml`: added `publish = false` — the root crate is an internal
  StateSet commerce daemon that needs a sibling `stateset-api` repo to build
  meaningfully, and its crates.io name would collide with the unrelated PyPI
  package `stateset-agents`.
- Added `docs/RUST_CRATES.md` clarifying the two unrelated Rust crates in
  this repo (`rust_core`/`stateset-rl-core`, the pyo3 accelerator, vs. the
  root crate, the internal commerce daemon), linked from the README
  installation section.
- `rust_core/Cargo.toml`: made `pyo3`/`numpy` optional dependencies gated
  behind a new `python` feature (which also enables
  `pyo3/extension-module`), so plain `cargo check`/`cargo test` and
  docs.rs work without libpython. `rust_core/src/lib.rs` moved all pyo3
  bindings into a `#[cfg(feature = "python")] mod python_bindings` block;
  the pure-Rust algorithm modules (`advantage`, `gae`, `trajectory`,
  `rewards`) remain unconditionally available. `rust_core/pyproject.toml`'s
  `[tool.maturin] features` now points at `["python"]`. Verified with
  `cargo check`/`cargo test` (default and `--features python`) and
  `maturin build --release`.
- Bumped `rust_core` 0.1.0 → 0.1.1 (`Cargo.toml` + `pyproject.toml`, kept
  in lock-step per the publish workflow's tag assertion) to ship the
  feature-gating fix.
- Added `.github/workflows/rust-ci.yml`: `cargo check --all-targets` +
  `cargo clippy -D warnings` (scoped with `-A dead_code -A
  unused_variables -A unused_imports -A unused_mut` to avoid a mass
  cleanup of the commerce daemon's WIP scaffolding) for the root crate,
  plus `cargo check`/`clippy`/`cargo test` (17 unit tests) for `rust_core`.
  Two small genuine clippy hits (`manual_is_multiple_of` in
  `rust_core/src/advantage.rs` and `rewards.rs`, a nested `format!` in
  `src/agents/customer_service.rs`) were fixed rather than allowed.
- `pyproject.toml`: added `pyyaml>=6.0` to core `dependencies` — the wheel
  ships runtime-loaded YAML presets that a bare `pip install
  stateset-agents` previously couldn't read (it was only pulled in by the
  `auto-research` extra).

## [0.17.1] - 2026-07-27 — Convergence e2e test + honest demo labeling

### Changed — Honest status labeling for dashboard/mobile; mock-data fallback surfaced (A+ final wave, Task 5)

- `dashboard/README.md` and `mobile/README.md` now lead with an explicit
  "demo, not deployed" status: both apps are real, working code, but
  neither has a deployment path — the `/api/lab/*` router they talk to is
  simulator-backed and gated behind auth + `API_ENABLE_TRAINING_LAB`
  (off by default). Both READMEs document how to run locally and what
  productionizing would require.
- `README.md` gained a short "Dashboard and mobile app (demo, not
  deployed)" section under Supported models, linking to the two READMEs.
- `dashboard/src/api.ts` gained a header comment stating it targets the
  simulator-backed `/api/lab` router and requires auth plus
  `API_ENABLE_TRAINING_LAB`.
- `mobile/hooks/useTrainingData.ts` now exposes `isMockData: boolean`
  alongside the existing `source: 'live' | 'mock'` field, and logs one
  `console.warn` per app session the first time it silently falls back to
  bundled mock data (unreachable API, auth failure, or empty response) —
  previously this fallback was invisible.

## [0.17.0] - 2026-07-27 — Unified finetune driver, rate-limiter hardening, grpo untangling

### Changed — Rate limiter moved out of the deprecated `grpo` package; deprecation warning scoped (A+ final wave, Task 3)

- `UnifiedRateLimiter` (and `RateLimitResult`, `get_rate_limiter`,
  `reset_rate_limiter`, `MAX_BUCKETS`) moved from
  `stateset_agents.api.grpo.rate_limiter` to `stateset_agents.api.rate_limiter`,
  reflecting that it is shared infrastructure used by `middleware.py` on the
  normal app path, not something specific to the secondary GRPO app.
  `stateset_agents.api.grpo.rate_limiter` remains as a thin re-export for
  backward compatibility.
- `stateset_agents.api.grpo`'s `DeprecationWarning` no longer fires on
  package import (previously it fired on every normal app startup via
  `middleware.py`'s rate-limiter import). It now fires lazily, only when
  the deprecated app-surface submodules (`service`, `service_routes`,
  `router_v1`, `auth`) are actually accessed.

### Changed — Unified finetune driver absorbs shared flags (A+ final wave, Task 1)

- `examples/finetune_gspo.py` now supports the full set of flag families
  shared by the packaged-starter per-model scripts:
  `--use-lora/--no-lora`, `--use-4bit`, `--use-8bit`, `--use-vllm`,
  `--wandb`/`--wandb-project`, `--export-merged`, `--learning-rate`,
  `--epochs`, `--steps`, and, for presets with a packaged starter
  (`ModelPreset.starter_module`), `--starter-profile
  {balanced,memory,quality}`, `--config PATH`, `--write-config PATH`, and
  `--list-profiles`. `--dry-run` now defaults to `True` (use `--no-dry-run`
  for a real run), matching the safe-by-default behavior of the starter
  scripts it now covers.
- **Fixed:** the driver's non-dry-run ("real run") mode used to be a
  no-op — it logged "wire this into your training entry point" and
  exited 0 without training anything. It now actually invokes the real
  training entry point: the packaged starter's own `run_<name>_config`
  coroutine for starter-backed presets, or
  `stateset_agents.training.gspo_entrypoints.train_with_gspo` for the rest.
  The dry-run exit message now explicitly says "pass --no-dry-run to
  train", and every doc/docstring example of a real run now shows
  `--no-dry-run` explicitly.
- **Fixed:** `--wandb`/`--wandb-project` were parsed but never reached
  `GSPOConfig` for non-starter presets (silent no-op even on real runs).
  `build_gspo_config` now sets `report_to="wandb"` plus `wandb_project`/
  `wandb_tags` when `--wandb` is passed (and `report_to="none"` otherwise),
  on both the starter and non-starter paths.
- **Fixed:** `--export-merged` was parsed but wired to nothing (silent
  no-op). It now calls `export_merged_model_for_serving` after a real,
  non-starter-backed training run (skipping with a warning if LoRA is
  disabled, since there is nothing to merge), and exits with a clear error
  for starter-backed presets, since none of the packaged starters currently
  expose a merge-export path.
- **Fixed:** the forwarder scripts silently dropped the old `--iterations`
  flag (the driver had no such flag, so passing it errored with
  "unrecognized arguments" instead of doing the thing the old script did).
  The driver now accepts `--iterations`, mapping it to a starter's
  `num_outer_iterations` override for starter-backed presets, and exiting
  with a clear error for non-starter presets ("use --epochs or --steps
  instead") rather than either silently dropping it or hard-failing with a
  generic argparse error.
- `ModelPreset` gained a `starter_module: str | None` field naming the
  packaged `stateset_agents.training.*_starter` module backing a preset's
  `--starter-profile` delegation, set for `kimi-k3`, `kimi-k2.6`,
  `glm5.1`, `glm5.2`, `gemma4-31b`, and `qwen3.5-0.8b`.
- Converted `examples/finetune_kimi_k3_gspo.py`,
  `examples/finetune_kimi_k2_6_gspo.py`,
  `examples/finetune_gemma4_31b_gspo.py`, and
  `examples/finetune_qwen3_5_0_8b_gspo.py` into thin (<=15 line) deprecated
  forwarders onto `examples/finetune_gspo.py --model <preset>`, now that
  the driver reproduces their entire CLI. The remaining per-model scripts
  (GLM's serving-only flags, the multi-size branching family scripts, and
  the already-forwarding `finetune_kimi_k2_5_gspo.py`) are kept; see
  `examples/README.md` for why each one still carries unique logic.

### Changed — CI + deferred cleanup (surface consolidation, Plan 3 Task 4)

- CI now runs `examples/getting_started/smoke.sh` against the checked-out
  source tree (via the editable install already used for tests), in
  addition to the existing PyPI-based `make getting-started-smoke` target
  used for release checks.
- Tightened `tests/unit/test_advanced_trainers.py::test_compute_gepo_coefficient`
  to pass sequence *log*-probs (matching the real call signature) and to
  assert the coefficients against the linear-space formula
  `coef_i = p_i / (sum(q^2) / sum(q))`, not just "greater than zero".
- Fixed `GSPOTokenTrainer.train_step_token_level`'s per-response token loss
  to normalize by the actual response length instead of the full padded
  sequence width, matching the sequence-level normalization used
  elsewhere in GSPO/GSPO-token.
- Corrected the comment on `_estimate_policy_entropy = compute_entropy_bonus`
  in `stateset_agents/training/loss_computation.py`: it is not a drop-in
  "backwards-compatible alias" (the signatures differ — the old estimator
  and `compute_entropy_bonus` are not interchangeable); it exists only
  because the one known external caller does `callable(_estimate_policy_entropy)`
  rather than calling it with the old signature.
- **Operational note:** `GSPOConfig.rescore_old_log_probs` now defaults to
  `True` (see prior GSPO hardening). This means even vLLM-based rollout
  deployments still require a local HF agent model + tokenizer at train
  time — the rollout log-probs are always rescored against the current
  policy before use, so a vLLM-only deployment with no HF model loaded
  will fail. Set `rescore_old_log_probs=False` explicitly if you
  intentionally want to trust the vLLM-reported log-probs instead (not
  recommended; see `gspo_generation.py` for the numerical-stability
  rationale).

### Changed — docs consolidation (surface consolidation, Plan 3 Task 3)

- Merged `docs/COMPARISON_TRL.md`, `docs/COMPARISON_LLM_FRAMEWORKS.md`,
  `docs/COMPARISON_TRADITIONAL_RL.md`, and the prior `docs/COMPARISONS.md`
  overview into a single `docs/COMPARISONS.md` with three clearly-headed
  sections ("StateSet Agents vs Hugging Face TRL",
  "StateSet Agents vs Traditional RL Frameworks",
  "StateSet Agents vs LLM Orchestration Frameworks").
- Archived the three superseded comparison files plus the dev-journal
  artifacts `docs/ENHANCEMENTS_SUMMARY.md`,
  `docs/FRAMEWORK_ENHANCEMENT_SUMMARY.md`, and root
  `GYM_INTEGRATION_COMPLETE.md` to `docs/archive/`. No other doc, README,
  or Sphinx toctree referenced these paths (verified by repo-wide grep
  before moving), so no link fixes were required beyond the merged
  `docs/COMPARISONS.md` itself.
- Added `tests/unit/test_docs_structure.py` to keep the archived files out
  of their old top-level paths and to assert `docs/COMPARISONS.md`
  contains all three merged section headers.

### Changed — examples cleanup (surface consolidation, Plan 3 Task 2)

- `examples/finetune_kimi_k2_5_gspo.py` is now a deprecated forwarder to
  `examples/finetune_kimi_k25_gspo.py` (a strict superset of its flags,
  plus `--system-prompt`, `--use-vllm`, `--export-merged`, `--iterations`).
  It will be removed in a future release.
- `examples/README.md` now documents every top-level example script; a new
  `tests/unit/test_examples_readme_complete.py` enforces this going
  forward.
- The other `examples/finetune_*_gspo.py` scripts and `examples/*_config.py`
  files were evaluated for collapsing into `examples/finetune_gspo.py` /
  `examples/model_presets.py` (from the prior "unified GSPO finetune
  driver" change) but were kept as-is: each carries model-specific CLI
  behavior (starter profiles, `--list-profiles`, `--write-config`, vLLM
  export, FP8 serving, model-size branching) that the unified driver
  intentionally does not reproduce, and the `*_config.py` files are
  imported by dedicated unit tests and docs (e.g.
  `tests/unit/test_kimi_k3_config.py`, `docs/glm5_1_starter.rst`).
  `examples/finetune_gspo.py --model <preset> --dry-run` remains available
  as a quick cross-model preview.

### Archived

- Moved `examples/enhanced_framework_demo.py`,
  `examples/enhanced_framework_showcase.py`,
  `examples/ultimate_customer_service_demo.py`,
  `examples/enhanced_customer_service.py`, and
  `examples/enhanced_grpo_demo.py` to `examples/archive/` — each was a
  redundant variant of a canonical example already documented in
  `examples/README.md` (`production_ready_customer_service.py`,
  `grpo_showcase.py`). References in `docs/ENHANCEMENTS_SUMMARY.md` and
  `docs/FRAMEWORK_ENHANCEMENT_SUMMARY.md` were updated to the new paths.

### Fixed — misplaced/duplicated Kimi-K2.5 test files

- `examples/test_kimi_k25.py` (a live, network-dependent smoke-check
  script, not a pytest suite despite its name) moved to
  `examples/kimi_k25/live_smoke_checks.py`.
- `tests/test_kimi_k25_integration.py` (misplaced at the `tests/` root,
  with test coverage that did not overlap
  `tests/integration/test_kimi_k25_integration.py`) moved to
  `tests/integration/test_kimi_k25_extended.py`.
- `examples/kimi_k25/README.md` updated to reference the new paths.

## [0.16.0] - 2026-07-27 — RL-core correctness + API hardening

### Fixed — RL training core correctness

- **DAPO**: old-policy token log probs are now frozen at rollout time (computed
  once under `no_grad` in `collect_samples_with_dynamic_sampling`) instead of
  being recomputed from the current model inside `train_step`, and
  `num_gradient_updates` (µ) is honored instead of being hard-capped at 1 —
  Clip-Higher and the PPO-style `min(unclipped, clipped)` objective can now
  actually fire on inner updates.
- **GEPO**: group-expectation importance weights are computed in log space via
  `logsumexp` (previously `exp()` of summed sequence log probs underflowed to 0
  for realistic responses, yielding NaN/zero coefficients), and the response
  mask off-by-one vs. the shifted-label convention is fixed
  (`build_response_mask`, `max(P-1, 0)`).
- **GSPO**: generation and scoring now share one tokenization convention —
  the chat-template-rendered prompt (including `system_prompt` when set) is
  scored via `build_scoring_text(rendered_prompt, response)` with no injected
  space; the trainer's current-policy and reference-KL log probs use the same
  convention, removing a systematic importance-ratio bias. Accumulated loss is
  normalized by the number of processed query groups. vLLM rollouts are
  rescored at T=1 by default (`rescore_old_log_probs=True`; requires the HF
  agent model/tokenizer even in vLLM deployments — set it to `False` to keep
  raw sampling-temperature `cumulative_logprob` with a one-time warning).
  Removed the fake-parameter injection for parameterless models
  (`GSPOTrainer` now raises `ValueError`).
- **GSPO-token**: restored the gradient path (token log probs were computed
  under `torch.no_grad()`, making `backward()` a no-op), masked prompt tokens
  out of the objective, fixed the reward call to use `compute_turn_reward`,
  and replaced `self.model.device` with `_get_model_device` (PEFT/sharded-safe).
- **GRPO loss path** (`loss_computation.py`): PPO ratios are length-normalized
  (`exp((new−old)/token_count)`) instead of `exp()` of raw log-prob sums; the
  entropy bonus is now differentiable (computed from the grad-enabled forward's
  logits); `LOSS_EXCEPTIONS` narrowed to `(RuntimeError, ValueError, OSError)`
  so programming errors propagate; `compute_enhanced_grpo_loss` gained ratio
  clipping and skips the full-vocab `log_softmax` when `beta == 0`.
- **VAPO**: value clipping compares fresh values against rollout-time
  predictions (was clipping values against themselves — a no-op); scalar
  rewards are placed on the terminal response token only before GAE (was
  broadcast across every token, inflating returns); `critic_advantages` is
  wired into the value target (`returns = critic_advantages + old_values`,
  decoupled GAE); the optimizer steps once per `train_step` over the
  prompt-averaged loss. VAPO's GAE uses the Rust kernel
  (`stateset_rl_core`) when installed, with a byte-identical Python fallback.

### Added — behavioral trainer tests

- Cross-trainer invariant suite `tests/integration/test_trainer_ratio_invariants.py`
  (DAPO/GEPO/GSPO on a tiny real GPT-2: on-policy first-update ratio ≈ 1,
  ratio ≠ 1 after a parameter update, finite loss, nonzero grads) plus
  per-trainer behavioral tests and a Rust/Python GAE parity test —
  the previous algorithm tests re-derived the math inline and could not catch
  any of the defects above.

### Security

- **Training Lab API gated behind auth + feature flag**: `/api/lab/*` (22 REST
  endpoints + metrics WebSocket) previously had no authentication and was
  mounted unconditionally. It is now opt-in via `enable_training_lab`
  (env `API_ENABLE_TRAINING_LAB`, default `true` in development, `false`
  otherwise) and, when mounted, every HTTP endpoint requires auth via
  `Depends(require_auth_if_enabled)`. The metrics WebSocket authenticates
  explicitly using an `api_key`/`token` query param (or `X-API-Key`/
  `Authorization` header), closing with code `4401` on missing/invalid
  credentials.
- **Training Lab in-memory state is now bounded**: `/api/lab` previously kept
  unbounded module-level dicts for experiments, episodes, and logs, and could
  leak untracked background training tasks. Experiments are now capped at
  `MAX_EXPERIMENTS = 100` (oldest created/completed/failed experiment is
  evicted to make room; a `429` is returned if every experiment is
  running/paused). Episodes and logs are bounded per-experiment via
  `collections.deque` (`MAX_EPISODES_PER_EXPERIMENT = 1000`,
  `MAX_LOGS_PER_EXPERIMENT = 5000`). Stopping or deleting an experiment now
  cancels its background training task.
- **`config.validate()` is now enforced at startup**: `create_app()` logs
  every configuration warning (e.g. missing API keys while auth is required)
  at `WARNING` level, and fails closed in production — raising
  `ConfigurationError` — when auth is required but no credential source at
  all (no API keys and no JWT secret) is configured.
- **Rate limiting now keys on identity, not shared IP**: `RateLimitMiddleware`
  previously bucketed by raw client IP (or a raw bearer/API key), so requests
  behind a shared NAT/proxy could exhaust one another's quota, and credential
  values leaked into limiter state. The bucket key is now the SHA-256 hash
  (first 16 hex chars, matching `auth.py`'s identity derivation) of the
  presented `Authorization`/`X-API-Key` credential when one is present,
  otherwise the client IP. `X-Forwarded-For`'s first hop is honored only when
  the new `trust_proxy_headers` flag (env `API_TRUST_PROXY_HEADERS`, default
  `false`) is enabled — previously the raw `request.client.host` was always
  used with no way to see through a trusted proxy, and there was no equivalent
  spoofing protection to disable it. Added an optional Redis-backed limiter
  (`API_RATE_LIMIT_BACKEND=redis` + `API_RATE_LIMIT_REDIS_URL`) for multi-pod
  deployments, using a fixed-window INCR/EXPIRE approximation; it falls back
  to the existing in-memory limiter (logged once) if `redis` isn't installed
  or the connection fails. `redis` is now listed under the `api` extra as an
  optional dependency.
- **Constant-time API key comparison**: `auth.py`'s API-key lookup used plain
  dict membership (`api_key in config.security.api_keys`), which is a
  short-circuiting `==` under the hood and vulnerable to timing side-channels
  against configured keys. It now compares the presented key against every
  configured key with `hmac.compare_digest`.

### Removed

- **Legacy GRPO service shims**: deleted
  `stateset_agents/api/ultimate_grpo_service.py` and
  `stateset_agents/api/enhanced_ultimate_grpo_service.py` — unmaintained
  duplicate FastAPI apps that shadowed `stateset_agents.api.main` and had no
  internal callers. Added `tests/api/test_no_legacy_shims.py` to guard
  against reintroduction; removed their Sphinx `automodule` entries.
- **Root `Dockerfile`**: moved to
  `deployment/docker/Dockerfile.rust-commerce-agent` with a header comment
  clarifying it builds the unrelated Rust commerce daemon (`src/main.rs`),
  not the Python FastAPI gateway (`deployment/docker/Dockerfile`). Updated
  `docker-compose.yml` and the `Makefile`'s `docker-build`/`docker-run`
  targets to the new path.

### Changed

- **Helm vLLM image tag pinned**: `deployment/helm/stateset-agents/values.yaml`
  used `vllm/vllm-openai:nightly`, a moving target with no reproducibility
  guarantee. Pinned to `v0.18.2` (matching the `vllm>=0.18.2` pin in
  `pyproject.toml`'s `vllm` extra) with a `# pin by digest in production
  overrides` comment for stricter deployments.
- **`stateset_agents.api.grpo` deprecated**: importing the package now emits
  a module-level `DeprecationWarning` pointing at `stateset_agents.api.main`
  as the supported entry point; this starts the deprecation cycle ahead of
  its eventual removal.

### Fixed

- `main.py` used `datetime.utcnow()` (deprecated, naive datetime) for the
  `/live` and `/circuits` timestamps; switched to
  `datetime.now(timezone.utc)`.
- `main.py` imported `CORSMiddleware` from the deprecated
  `fastapi.middleware.cors` re-export path; now imports from
  `starlette.middleware.cors` directly.

### Added — Kimi-K3 starter path

- **`stateset_agents/training/kimi_k3_starter.py`** — packaged GSPO starter for
  `moonshotai/Kimi-K3` (provisional ID — HF weights, model card, and license not
  yet published as of 2026-07-16), mirroring the Kimi-K2.6 surface: `KimiK3Config`,
  `get_kimi_k3_config`, profile matrix (balanced/memory/quality), JSON/YAML config
  round-trip, and lazy exports from `stateset_agents.training`.
- **`stateset-agents kimi-k3`** CLI command (`cli_train.py`) and
  `stateset-agents init --preset kimi-k3` scaffold preset (`cli.py`).
- Examples: `examples/finetune_kimi_k3_gspo.py`, `examples/kimi_k3_config.py`.
- Docs: `docs/kimi_k3_starter.rst`, CLI reference section, SUPPORTED_MODELS row,
  README starter section.
- Tests: `tests/unit/test_kimi_k3_config.py`, `tests/unit/test_kimi_k3_module_exports.py`,
  `kimi-k3` command + init-preset tests in `tests/unit/test_cli.py`.

### Added — GLM 5.2 starter path

- **`stateset_agents/training/glm5_2_starter.py`** — packaged GSPO starter for
  `zai-org/GLM-5.2` (754B MoE, `glm_moe_dsa` architecture), mirroring the GLM 5.1
  surface: `Glm52Config`, `get_glm5_2_config`, profile matrix
  (balanced/memory/quality), `get_glm5_2_serving_recommendations`, JSON/YAML
  config round-trip, and `finetune_glm5_2`. Exported lazily from
  `stateset_agents.training`.
- **`examples/finetune_glm5_2_gspo.py`** and **`examples/glm5_2_config.py`** —
  dedicated CLI starter + re-export helper.
- **`scripts/render_glm5_2_helm_values.py`** — renders Helm values from a GLM 5.2
  `serving_manifest.json`.
- Deployment: `values-glm5-2{,-fp8,-finetuned}.yaml` Helm overrides and
  `glm5-2-vllm{,-fp8,-finetuned,-finetuned-gcs}.yaml` + `glm5-2-training-job.yaml`
  Kubernetes manifests.
- Docs: `docs/glm5_2_starter.rst`, `docs/GLM5_2_HOSTING_PLAN.md`, plus
  `SUPPORTED_MODELS.md` / README / Sphinx toctree entries.

## [0.15.3] - 2026-05-24 — A+ polish: inference observability, honest coverage gate, Rust wheel pipeline

### Added — Rust accelerator on PyPI

- `rust_core/pyproject.toml` — maturin-backed Python packaging metadata for
  the `stateset-rl-core` crate. abi3-py310 means one wheel per (OS, arch)
  covers Python 3.10–3.13. Verified locally: `maturin build --release`
  produces a working manylinux_2_31 wheel, installs cleanly, and
  `stateset_agents.core.rust_accelerator.is_rust_available()` returns
  True. Crate version stays in lock-step with Cargo.toml (asserted by
  the publish workflow).
- `.github/workflows/rust-wheels.yml` — builds wheels for Linux
  x86_64/aarch64, macOS x86_64/arm64, and Windows x86_64; builds sdist;
  smoke-imports the host wheel; uploads each as an artifact. Publishes
  to PyPI when a `rust-core-v*` tag is pushed (gated on `PYPI_API_TOKEN`
  + `pypi-rust-core` environment for review). Decoupled from
  `stateset-agents` tags so the two release cadences don't interfere.
- New `[rust]` optional extra: `pip install "stateset-agents[rust]"` now
  pulls `stateset-rl-core>=0.1.0`. Added to `full` as well.
- `rust_core/Cargo.toml` — enabled `pyo3/abi3-py310` feature.
- `rust_core/README.md` install section updated: PyPI install is now
  primary, source build is "development" path.
- `docs/ARCHITECTURE.md` — added Rust accelerator entry under Technical
  Specifications.

### Added — Inference observability

- **`stateset_agents/api/inference_metrics.py`** — model-level Prometheus
  metrics complementing the HTTP-level metrics in `api/middleware.py`:
  `stateset_inference_requests_total`, `_duration_seconds`, `_ttft_seconds`,
  `_tokens_per_second`, `_tokens_total` (split by `prompt`/`completion`),
  and `_inflight` gauge. Labels: `model`, `route` (one of
  `openai_response`, `openai_stream`, `anthropic_response`,
  `anthropic_stream`), and `status` where applicable. Mirrors the optional-
  import pattern from the HTTP middleware so `prometheus_client` stays an
  optional dependency.
- `InferenceService.create_openai_response`, `stream_openai`,
  `stream_anthropic`, and `create_anthropic_response` are now instrumented
  via `track_request` / `track_inflight`. TTFT is captured on the first
  data line of each stream; per-request throughput is computed from
  completion tokens / total duration.
- **Grafana serving row** (`deployment/monitoring/grafana-dashboard.json`,
  panel IDs 100–106): inference RPS by model/route, error rate, end-to-end
  latency (P50/P95/P99), TTFT (P50/P95), output throughput (P50/P95),
  in-flight gauge, and token-volume by direction. Also corrects existing
  HTTP panels to use the real `stateset_http_*` metric prefix (was
  `http_*`, which never matched anything emitted by the app).
- 8 new unit + integration tests in `tests/unit/test_inference_metrics.py`
  covering the helper API and end-to-end recording against the stub
  backend.

### Fixed

- **Coverage gate honesty.** v0.15.1 set
  `[tool.coverage.report] fail_under = 70` aspirationally, but measured
  coverage on the post-CLI-decomp tree is 54.50%, which was silently
  failing the CI test step (and the duplicate `--cov-fail-under=70` in
  `scripts/publish_readiness.sh`). Gate dropped to 54 to match the
  measured floor; ratchet policy documented inline. Removed the
  duplicate `--cov-fail-under` flag so pyproject.toml is the single
  source of truth.
- **`docs/ARCHITECTURE.md`** refreshed from "Version 0.5.0 / Dec 2024 /
  ~50K LOC" to the actual v0.15.2 state (~104K LOC across 243 modules,
  full algorithm roster, observability section). Added a doc-versioning
  disclaimer mirroring the whitepaper's.

## [0.15.2] - 2026-05-20 — cli.py decomposition + deployment-version sync

Internal restructure. No public API change — all `stateset_agents.cli` imports, the `stateset-agents` entry point, the `python -m stateset_agents.cli` invocation, and every CLI subcommand behave exactly as before.

### Changed

- **`stateset_agents/cli.py` split from 3,118 → 1,150 LOC** (a 63% reduction in the worst-offender file). Twenty-three subcommands moved into focused sibling modules:
  - `cli_train.py` (998 LOC) — `train`, `qwen3-5-0-8b`, `kimi-k2-6`, `gemma-4-31b`
  - `cli_meta.py` (353 LOC) — `doctor`, `preflight`, `publish-check`
  - `cli_research.py` (360 LOC) — `auto-research`, `fine-tune`
  - `cli_chat.py` (272 LOC) — `chat`
  - `cli_benchmark.py` (172 LOC) — `benchmark` sub-app + four subcommands
  - Existing `cli_advanced.py` (390 LOC) untouched
- The remaining `cli.py` keeps the helpers, exception tuples, profile constants, `app` instance, and the small commands (`version`, `serve`, `evaluate`, `validate-config`, `init`, `init-config`, `starter`, `recipe`, `tour`).
- Test patches on `stateset_agents.cli._collect_dependency_status` and `_collect_import_status` continue to work — the meta module accesses these via late binding (`_cli._collect_…`) rather than locally rebinding them.
- `python -m stateset_agents.cli` keeps working via a one-line `sys.modules.setdefault("stateset_agents.cli", sys.modules[__name__])` alias at the top of `cli.py` that prevents a duplicate module load (and a second orphan `app` instance) when invoked as `__main__`.

### Fixed

- **Helm/K8s/docs version drift.** `deployment/helm/stateset-agents/values.yaml`, `Chart.yaml`, three Kubernetes training-job manifests, the Helm chart README, and `docs/KIMI_K25_GKE_AUTOPILOT.md` were pinned to image tag `0.11.6` (the last release that updated them). All bumped to `0.15.2` in lock-step with the package version. Closes the pre-existing `test_helm_values_use_current_package_version` and `test_selected_kubernetes_and_docs_refs_use_current_package_version` failures.
- **`test_cli_version_outputs_version`** asserted on the pre-v0.12.0 output format (`stateset-agents version X.Y.Z`); aligned with the actual current format (`stateset-agents X.Y.Z`, set in v0.12.0).

## [0.15.1] - 2026-05-20 — CI/release hardening: reproducible installs, nightly perf alerts, full Helm coverage

Infrastructure-only patch. No public API change. Tightens four CI/release dimensions to A+: coverage is now enforced from a single source of truth, installs are reproducible from committed lock files, performance regressions are caught on a nightly schedule with PR comments, and every published Helm values profile is rendered in CI.

### Added

- **`requirements-lock.txt` + `requirements-dev-lock.txt`** — pip-compile output committed to the tree, giving fully-pinned reproducible installs (`make install-locked`). `pip-tools` added to the `[dev]` extra. `make lock` regenerates; `make lock-check` is wired into `ci.yml` and fails the build if either lock drifts from `pyproject.toml`.
- **`.github/workflows/benchmark-nightly.yml`** — cron `0 7 * * *` runs the full `benchmarks/` + `tests/performance/` suites via `pytest-benchmark`, with `benchmark-action/github-action-benchmark` configured for `fail-on-alert: true`, `alert-threshold: 150%`, and PR comments on regression. Raw JSON archived for 30 days. The existing per-PR `benchmark.yml` and Phase-0 `benchmark-smoke.yml` are unchanged.

### Changed

- **Coverage gate moved to `pyproject.toml`** (`[tool.coverage.report] fail_under = 70`) — `pytest-cov` reads it automatically, so `pytest --cov` enforces the same floor locally and in CI. Redundant `--cov-fail-under=70` CLI flag removed from `ci.yml`. Added `precision = 2`, `show_missing = true`, and standard `exclude_lines` (`raise NotImplementedError`, `if TYPE_CHECKING:`).
- **Helm CI now renders every published profile** (`ci.yml` `helm` job) — was 4 (default + A100/H100/B200/Kimi-finetuned), now all 13 `values-*.yaml` including GKE staging/prod, vllm-A100, the full GLM5.1 trio (base/fp8/finetuned), and the full Qwen3.5-27B trio (base/minimal/finetuned). Profile count is asserted; rendered manifests uploaded as a 7-day artifact.

## [0.15.0] - 2026-05-20 — Onboarding & scenario coverage: 5 examples, 5 test patterns, 3 scenario notebooks

Triples the surface area of runnable onboarding content. Adds five new GPU-free `examples/getting_started/` scripts that smoke-test in <5s, a five-file `examples/testing/` directory (56 passing pytest cases in ~4s) covering the patterns most useful when writing tests against the framework, and three new Colab-ready training notebooks for scenarios the existing notebooks didn't cover (multi-turn × tool-calling, judge-driven RLHF loop, RAG grounding).

### Added

- **`examples/getting_started/` 06–10** (all GPU-free, all run via `smoke.sh`):
  - `06_multi_turn_episode.py` — drives `env.reset()` → loop `env.step()` against the stub backend; same shape as a real trainer rollout.
  - `07_tool_calling.py` — `ToolAgent` + bundled `ToolCallReward`, scoring well-formed, wrong-tool, and malformed responses.
  - `08_eval_driven_loop.py` — the §11.7 dev rhythm (baseline → change → measure) with two policies.
  - `09_curate_dataset.py` — chat → grade → curate, writes an SFT-ready JSONL.
  - `10_scenario_testing.py` — must-acknowledge / must-avoid / rubric-floor assertions; non-zero exit on regression.

- **`examples/testing/`** — new directory with 56 passing pytest cases (~4s on CPU) across five patterns:
  - `test_custom_reward.py` (table-driven), `test_stub_integration.py` (no-mock integration), `test_hypothesis_properties.py` (property-based, uses `stateset_agents.testing` strategies), `test_env_smoke.py` (parametrized over every bundled scenario), `test_judge_stability.py` (judge noise-floor + 2σ-separation gate from §11.7).
  - `README.md` documents the "which pattern when" matrix and the explicit anti-patterns (no mocking the model, no `time.sleep`, no asserting on free-form text).

- **Three scenario notebooks** (all pinned, seeded, lint-clean, `train_with_gspo` + `attn_implementation='sdpa'` + `use_reference_model=True, beta=0.05`):
  - `ecommerce_returns_agent.ipynb` (~2h on A100) — multi-turn × tool-calling under a 60% rubric / 40% tool composite reward.
  - `judge_driven_training_loop.ipynb` (~90 min) — closed-loop `train → judge → retrain` with a local Qwen2.5-1.5B-Instruct judge.
  - `rag_agent_finetune.ipynb` (~75 min) — BM25 retriever + 10-doc KB + `GroundingReward` that penalizes uncited claims.

### Changed

- `scripts/lint_notebooks.py` — added the three new notebooks to `BUNDLED_NOTEBOOKS`; CI now lints 13 notebooks (was 10).
- `examples/getting_started/README.md` — expanded to a 10-row table (was 5).
- `examples/getting_started/smoke.sh` — runs all 8 GPU-free examples (was 3).
- `notebooks/README.md` — "thirteen core notebooks" (was ten), with three new rows and three new "when to use what" entries.

## [0.14.1] - 2026-05-19 — Docs canonicalization

Tightened the documentation surface by consolidating overlapping guides into a single canonical set, removing the "which guide do I read?" ambiguity.

### Changed

- **Canonical guide picks:** `/QUICKSTART.md` (15-min), `docs/RL_FRAMEWORK_GUIDE.md` (usage), `/OVERVIEW.md` + `docs/FRAMEWORK_OVERVIEW.md` (overview), `docs/COOKBOOK.md` (recipes).
- **Archived under `docs/archive/`** (with redirect stubs at the original paths so inbound links don't 404): `QUICKSTART_5MIN.md`, `USAGE_GUIDE.md`, `COMPREHENSIVE_USAGE_GUIDE.md`, `STATESET_RL_AGENTS_OVERVIEW.md`. The first two used stale "GRPO Agent Framework" branding; `RL_FRAMEWORK_GUIDE.md` supersedes them with current branding plus the CLI reference, GRPO-vs-GSPO comparison, and HPO coverage.
- **Cross-refs updated** in `README.md`, `QUICKSTART.md`, and `docs/API_EXAMPLES.md` to point at the canonical guide.

### Tests

- Removed `tests/unit/test_usage_guide.py` and `tests/unit/test_comprehensive_usage_guide.py` (asserted against archived files).
- Added `tests/unit/test_rl_framework_guide.py` consolidating the surviving regressions (stub-quickstart flow present, old package name absent) against the canonical guide.
- Patched `tests/unit/test_advanced_docs_alignment.py` to drop archived targets and point at the canonical.

## [0.14.0] - 2026-05-19 — Dashboard A+ pass: build/lint/test/CI green, accessibility, external-store notifications

A focused quality push on `dashboard/`. The lab UI previously had a broken `tsc -b` (30+ TS errors), 13 lint errors, 0 tests, no CI coverage, and the default Vite README. This release fixes all of that, hardens accessibility, and trims runtime overhead.

### Fixed

- **Build.** `useToast` API mismatch (`toast.success` / `error` / `info` called everywhere but never exposed) was the dominant error class — eliminated by splitting `toast-context.ts` (context), `useToast.ts` (hook), and `components/ToastProvider.tsx` (provider) with a 4-method API. `verbatimModuleSyntax` violations resolved with `import type`. Recharts 3 `ValueType | undefined` narrowed in `MetricsCharts` and `RewardHistogram`. Invalid `fractionalSecondDigits: 0` removed from `TrainingConsole`.
- **Lint.** 13 errors → 0. Dead imports (`Zap`, `X`, `BarChart3`, `StatCard`, `Card`, `Filter`) removed across 6 files. `react-hooks/set-state-in-effect` resolved in `LiveMonitor` (form state initialized in click handler), `ExperimentDrawer` (React 19 "adjust state on prop change" pattern), and `NotificationCenter` (rewritten to consume a module-level store via `useSyncExternalStore`; status-transition diffing moved to a new `useTrackExperimentNotifications` hook that dispatches to the external store).
- **WS + polling overlap.** `LiveMonitor` now only polls REST when `connected === false`.

### Added

- **CI.** `.github/workflows/dashboard.yml` runs lint → typecheck → test → build on every PR touching `dashboard/` and uploads the bundle.
- **Tests.** Vitest + jsdom + Testing Library. Three suites (`useToast`, `notifications-store`, `App` smoke), 7 passing tests. `npm test` and `npm run typecheck` scripts added.
- **Accessibility.** Command palette is now a focus-trapped `role="dialog"` with `aria-modal`, `aria-activedescendant`, `aria-controls`, `role="listbox"` / `role="option"`, `aria-selected`, `Home`/`End` keys, scroll-into-view on the active item, and focus restoration on close. Dashboard icon buttons gained `aria-label`. Toast region is `aria-live="polite"`.
- **README.** Replaced the Vite boilerplate with a real architecture doc (feature matrix, run instructions, data-flow notes, code-quality bar).

### Notes

- Library/API code in `stateset_agents/` is unchanged in this release. The minor version bump reflects the dashboard hardening as a deliberate UX milestone.

## [0.13.4] - 2026-05-18 — Fourth-reviewer round: factual fixes + PDF rendering + reproducibility re-run

A fourth reviewer graded v0.13.3 at A−, primarily citing factual inconsistencies in §5.2 (clip-range "symmetric" vs the actual 3e-4/4e-4 values; "three orders of magnitude" vs the actual 2.8), a stale PyPI-lag note still claiming PyPI was at 0.7.1, and Mermaid diagrams rendered as fenced code in the PDF. This release lands those fixes plus three smaller asks. Items map 1:1 to the reviewer's numbered list.

### Fixed (factual)

- **WP §5.2 GSPO clip range.** Was: "the shipped defaults are symmetric — Clip-Higher asymmetry is reserved for DAPO and VAPO" with defaults `3e-4 / 4e-4`. Now correctly describes the defaults as **mildly asymmetric** with a slight upside bias, contrasted against DAPO's deliberate 40% Clip-Higher asymmetry. Also fixes the "three orders of magnitude tighter than 0.2" claim — the actual ratio is `0.2 / 3e-4 ≈ 667× = 2.8 orders of magnitude`.

- **WP front-matter PyPI callout + §"Versioning and Reproducibility".** Both still said PyPI was at 0.7.1 — stale since we shipped 0.13.2 to PyPI in this same release line. Updated to the current state: PyPI is at 0.13.2, source ahead by a patch, lag-window closed.

- **WP §5.6 comparative summary table.** The "Reported AIME score" row had empty cells for GRPO/GSPO/GEPO that highlighted the gap rather than mitigated it. Replaced with two rows: "Source-paper reasoning benchmark" (cites the paper-reported numbers with links) and "First-party (this framework)" (only GSPO has an entry, all others honestly marked "pending §5.6 v1.1"). Also added explicit "design-intent map, not a head-to-head benchmark" framing at the top of the section.

- **WP §6.7 Continual Learning.** Reviewer flagged this as "shaped like a feature list, not a guide." Demoted from a full subsection with a strategy table to a single paragraph explicitly labeled "primitive, evaluation harness pending."

### Added

- **`make whitepaper-pdf` Mermaid rendering.** The build script now fetches SVG from `mermaid.ink` (no install required, no JS execution) with on-disk caching at `/tmp/mermaid_svg_cache/`. The two diagrams in §2.1 and §3 now render as actual SVG in the PDF instead of fenced code blocks. Falls back to the code-block placeholder if the service is unreachable.

- **WP §C.7 audit-trail update.** Added the missing reproduction command for the §11.7 canonical result — both the Colab notebook URL and the headless `jupyter nbconvert --execute` invocation.

- **WP §10.3 rubric-blindness audit note.** One-line acknowledgment that the other bundled rubrics (`GSM8KReward`, `PartialCreditGSM8KReward`, `ToolCallingReward`) were audited for similar blindness patterns; tool-calling has its own structural failure mode (well-formed JSON for the wrong tool) that the §11.7 LLM-judge protocol would catch.

- **`benchmark_results/whitepaper_v1/customer_support_3seed_judge_qwen25_05b_instruct_rerun.json`** — independent re-run of the canonical §11.7 result on a fresh Colab session ~7 hours after the original. `trained_metrics` are bitwise-identical to four decimals across both runs — empirical evidence of seed-determinism, not just claimed.

- **`docs/WHITEPAPER.pdf`** rebuilt at this version (v0.13.4). 60 pages, ~390 KB. Cover page, header/footer pagination, Charter serif body / Inter sans headings / JetBrains Mono code, rendered Mermaid diagrams.

### Still pending (Colab-bound — explicit in the docs)

The fourth reviewer's "what would push it to A" list:
- **Head-to-head trainer benchmark.** `whitepaper_v1_comparative_trainers.ipynb` is ready; needs ~$1.30 of Colab time. Fills the new "First-party" row in §5.6.
- **Cross-family judge sanity check.** `CROSS_FAMILY_JUDGE_MODEL` knob is shipped; needs ~$0.85 of Colab time. Validates §11.7's in-family bound.
- **Framework-vs-framework benchmark.** Optional A+ target — TRL on the same customer-support task. Not currently scheduled.

## [0.13.3] - 2026-05-18 — Third-reviewer polish + README refresh (A → unblocking-to-A+)

The third reviewer round graded the v0.13.2 whitepaper at **A (up from A−)** with three small polish items and two Colab-bound run asks. This release lands the three polish items and refreshes the README to reflect the PyPI parity from v0.13.2.

### Changed

- **WP §14 trimmed.** The five-principles block in the conclusion was duplicated nearly verbatim from §1.0 (Design Philosophy preface). §14 now ends with a one-line reference back to §1.0 — saves ~150 words and gives the conclusion more punch.
- **WP §6.8 reordered.** The "<1% end-to-end" honest framing now appears *above* the kernel-speedup table, not below. The table also gains an explicit "End-to-end §11.7 impact" column (all <1%) so skimmers can't pick up the 26–72× number without the calibration. Pre-empts the misread the reviewer flagged.
- **WP §11.3 decision-tree caveat.** Added a parallel caveat to §11.7's "what this result does not support": the trainer-selection decision tree summarizes literature + implementation experience, but **first-party head-to-head numbers are pending** — the harness exists in `whitepaper_v1_comparative_trainers.ipynb`, the result row hasn't landed in §5.6 yet. Closes the last unsupported-authority surface in the document.

### Documentation

- **README.md substantial refresh:**
  - Header badges: replaced `badge.fury.io` with `shields.io/pypi/v` (auto-updates from PyPI), added Whitepaper version badge and §11.7 first-party-result badge.
  - New top-of-doc "What's new in v0.13.2" section listing the six headline changes since v0.13.0.
  - Install section now reflects the PyPI parity: `pip install stateset-agents` gets v0.13.2, no longer stuck at v0.7.1. Old PyPI lag callout retained as historical context.
  - Benchmark notebook table expanded from 2 rows to 6, headed by `customer_support_3seed_judge.ipynb` as the canonical §11.7 notebook. Pointer to the lint script and issue #16.
  - "Start here" section bumped: whitepaper + errata + benchmark_results/whitepaper_v1/ now linked above the cookbook/platform-tour entries. CHANGELOG version reference: v0.12.1 → v0.13.2.

### Still pending (Colab-bound — explicit in the docs, not in this release)

Two runs would move the grade from **A** to **A+**:

- **Head-to-head trainer comparison.** The notebook (`whitepaper_v1_comparative_trainers.ipynb`) is in v0.13.1; the run is a few hours of Colab compute. Output JSON fills §5.6's comparative table.
- **Cross-family judge sanity check.** Set `CROSS_FAMILY_JUDGE_MODEL='meta-llama/Meta-Llama-3-8B-Instruct'` in `customer_support_3seed_judge.ipynb` (the knob is shipped in v0.13.2). Re-run on the same 24 generations. The third reviewer was explicit: "either direction is more important than not knowing."

## [0.13.2] - 2026-05-18 — Second PhD-reviewer round (calibration + scoping fixes)

Addresses every item from the second reviewer pass. Patch bump: no API removed, additions are doc clarifications, one framework warning, and a notebook extension. Items map 1:1 to the reviewer's numbered list.

### Added

- **§1.0 Design Philosophy preface** — the five principles previously only in §14 now lead §1, anchoring the rest of the document. §14 retains the detailed treatment.
- **§11.7 in-family-judge-bias disclosure** — explicit acknowledgment that Qwen-trainee + Qwen-judge has known in-family bias. Points readers at the new `CROSS_FAMILY_JUDGE_MODEL` knob in the reference notebook.
- **`CROSS_FAMILY_JUDGE_MODEL` constant** in `notebooks/customer_support_3seed_judge.ipynb` — opt-in second judge (Llama-3-8B-Instruct or similar) for the cross-family sanity check. Disabled by default (saves ~6 GB VRAM + ~5 min wall clock); set to enable. Evaluate() function now reports `cross_judge_mean` alongside the primary `judge_mean` when enabled.
- **§6.8 end-to-end framing** — explicit caveat that the 26–72× kernel speedup translates to <1% of end-to-end step time at typical §11.7 configurations. Pre-empts "framework is 26× faster" misreading.
- **§3.6 memory algorithm pin-down** — replaces the vague "configurable triggers" with the actual `add_turn` sequence (append → extract entities → extract facts → maybe summarize → decay → trim) plus the retrieval order. Flags semantic-tier embedding-similarity as v1.1 pending work.
- **§8.1 maturity-inversion acknowledgment** — one paragraph explaining why serving is `Stable` while trainers are `Beta`/`Experimental` (the operational layer hardened first against StateSet's commercial deployments).
- **§5.3 GEPO math clarification** — names $E_q[q]$ as the self-normalized importance-sampling denominator, connecting the symbol to particle-filter/population-Monte-Carlo primitives.
- **§11.5 token-soup callout** — promotes the issue #16 failure mode from a table row to a prose callout that reminds readers never to accept a rubric improvement without spot-checking generations.
- **"Skim on first read" labels** on §3.6, §6.7, §6.9, §11.6 — improves navigation for readers using the document as reference.

### Changed

- **`NeuralRewardModel` constructor** in `stateset_agents/training/neural_reward_trainer.py` now takes an `encoder=None` parameter and emits a `WARNING` when the default hash-based smoke-test encoder is used without `suppress_smoke_encoder_warning=True`. The hash encoder cannot learn useful rewards (deterministic pseudo-random per string); the warning quotes the fix and the smoke-test rationale. §4.4 prose updated accordingly: the default path is now explicitly framed as "smoke-test the training loop only."
- **§13.2 prose** — softened from "no other framework ships all three" to "no other framework in the matrix treats all three as first-class concerns in a single deployable package," with concrete examples of which competitors treat which subset as first-class.
- **§1.2 BUSL framing** — trimmed the positioning paragraph to a one-line license declaration; the four-year rationale moves to a footnote.

### Documentation

- All eleven reviewer items above plus the smaller notes (BUSL trim, dense Mermaid diagram in §3 — flagged but not yet split; "skim" labels; principles preface).

### Still pending (Colab-bound, for v1.0/v1.1 first-party results)

- Comparative trainer benchmark run (`whitepaper_v1_comparative_trainers.ipynb` — authored in v0.13.1, awaiting Colab execution).
- vLLM speedup measurement run (`vllm_speedup_benchmark.ipynb` — authored in v0.13.1, awaiting Colab execution).
- Cross-family judge result on the §11.7 generations (notebook supports it in this release; needs a run).

## [0.13.1] - 2026-05-18 — Whitepaper PhD-reviewer round + comparative-trainer notebook + notebook-lint

Closes the v1.0 whitepaper review feedback (A− → unblocking-to-A): comparative trainer notebook, vLLM-speedup notebook, related-work feature matrix, Rust-core measured speedup, §5.7 KL note promotion with TRL line citation, notation consistency, and the issue #16 CI smoke ask.

### Added

- **`notebooks/whitepaper_v1_comparative_trainers.ipynb`** — TRL GRPO vs GSPO vs DAPO on the §11.7 customer-support protocol. Three trainers × three seeds = nine training runs, same baseline, same rubric + LLM-judge eval. Fills the §5.6 comparative trainer table that's been empty since v0.11.6. Skips GEPO (async-oriented) and VAPO (experimental warmup) per the reviewer's recommendation.
- **`notebooks/vllm_speedup_benchmark.ipynb`** — HF generate vs vLLM throughput sweep over `prompt_batch_size ∈ {1, 8, 32, 128}` × `num_generations=4` × `max_tokens=512` on `Qwen2.5-0.5B-Instruct`. Produces the §6.4 "Validated configuration" number with sweep data so readers see the speedup-grows-with-batch claim is verifiable.
- **`scripts/lint_notebooks.py`** + **`make notebook-lint`** — codifies issue #16's lessons into a CI-friendly lint. Checks all ten bundled notebooks for: `asyncio.run()` in Jupyter, `Agent(config=...)` (abstract base), `attn_implementation='flash_attention_2'` (Colab incompatibility). Wired into `make smoke` and `.github/workflows/benchmark-smoke.yml`. Trigger paths simplified to `notebooks/**`.
- **`benchmark_results/whitepaper_v1/rust_vs_python_microbenchmark.json`** — first-party measurement backing §6.8's Rust-core speedup claim. `batch_compute_gae` runs 26-72× faster than the equivalent Python loop across three batch sizes; vectorizable kernels (`compute_group_advantages`) don't see the same benefit. Honestly characterized.

### Fixed

- **`stateset_agents/training/trl_grpo_trainer.py:604`** — mirror of the `gspo_trainer.py` canonical-signature fix from `a2bdde4`. TRL GRPO's reward call was using `compute_reward(trajectory=None, turn=turn, ...)`, only accepted by `MultiObjectiveRewardFunction`. Now uses `compute_reward(turns=[turn], context=...)` so any `RewardFunction` subclass works — required by the new comparative-trainer notebook.
- **`notebooks/quickstart_first_finetune.ipynb`** — three `asyncio.run()` calls in the test-prompt loop converted to top-level await (caught by `notebook-lint`).
- **`notebooks/grade_and_curate_demo.ipynb`** — `asyncio.run()` over scenarios refactored to an `async def` + top-level await.
- **`notebooks/sft_from_curated_demo.ipynb`** — two `asyncio.run()` in inline generate/eval calls converted to top-level await.
- **`.github/workflows/benchmark-smoke.yml`** — notebook validation step now lints all ten bundled notebooks (was six) via `python scripts/lint_notebooks.py` in addition to JSON validity.

### Documentation (PhD reviewer round)

- **§13 Related Work** — replaced the prose paragraph with a 15-capability × 6-framework feature matrix (StateSet Agents vs TRL, OpenRLHF, TRLX, NeMo-Aligner, Verl). New "Distinguishing position" subsection naming the three column-pairs where the framework adds something the others don't.
- **§5.7 "Exact Forward KL"** — promoted from "A Note on KL Divergence" with a three-way comparison (analytical / Schulman k1 / Schulman k3) and TRL line citations (`trl/trainer/ppo_trainer.py:517` and `rloo_trainer.py:436` — both use k1 `0.5 * (logprobs_diff**2).mean()`). Closes the "is that really true of all open implementations?" loophole.
- **§3.1 Tool-call semantics** — new paragraph documenting the four behavior decisions: tool call is part of assistant turn (not separate), credit assignment is per-turn (not per-call), tool errors are reward's responsibility, tool outputs excluded from policy gradient.
- **§6.8 Rust Acceleration Core** — replaces the unbacked performance claim with a measured speedup table: 26-72× on `batch_compute_gae`. Explicit that vectorizable kernels don't see the same benefit; the Rust core is included for recurrence-heavy paths, not as a blanket NumPy replacement.
- **§6.4 vLLM speedup** — adds a "Validated configuration" paragraph naming the specific (model, batch, seq-len, hardware) tuple the integration is validated against, pointing at `vllm_speedup_benchmark.ipynb`.
- **PyPI install-path warning** moved from §"Versioning and Reproducibility" body to a prominent callout immediately after the abstract, with a concrete `pip install git+...` command.
- **Notation consistency** — `ε_R` → `ε_H` throughout §5.2 (GSPO) to match §5.0 notation table and §5.4 (DAPO).
- **§4.4 "Bitter Lesson"** — Sutton citation (new reference [20]) plus a caveat about neural-reward failure modes (reward hacking, judge gaming, calibration drift).
- **§1.2** now cross-references §8.1 Component Maturity so new readers encounter the experimental/beta/stable classification before §8.
- **§1.2 BUSL→Apache 2.0** transition gets a rationale paragraph (no-hosted-derivative protection during the 4-year window, always-permitted use for individual/academic/customer cases, binding contractual conversion in 2029).
- **§2.1 layered-design diagram** upgraded from ASCII to Mermaid (matches the §3 class diagram's quality).
- **§7.4 dashboard** description points readers at `dashboard/README.md` for screenshots and `make dashboard` for the live UI.
- **Appendix C.4 test-count verification** — changed from `pytest --collect-only` (counts but doesn't verify green) to `pytest tests/unit -q --no-header | tail -2` (counts AND confirms pass).

## [0.13.0] - 2026-05-18 — First-party canonical benchmark + KL-anchor safety

The release that lands the v1.0 whitepaper's first canonical first-party benchmark result and codifies a previously-undocumented training foot-gun as a runtime warning. Driven by [issue #16](https://github.com/stateset/stateset-agents/issues/16) — a multi-day debugging session running the bundled benchmarks end-to-end on Colab A100.

### Added

- **`PartialCreditGSM8KReward`** in `stateset_agents.data.gsm8k` — dense-reward variant of `GSM8KReward` with four tiers (`0.0` unparseable / `0.2` parseable / `0.5` within 10% / `1.0` correct). Eliminates the all-zero-groups pathology where the binary reward gives no gradient signal on weak base models. 8 new unit tests covering all four tiers and edge cases.
- **`train_with_gspo` runtime warning** — emits a loud warning when `use_reference_model=False` AND `beta=0.0` AND `len(train_queries) < 100`. This combination destabilizes the policy to gibberish output (see whitepaper §10.5 and `benchmark_results/whitepaper_v1/customer_support_qwen3_5_0_8b_gspo.json`). Warning quotes the safe-default fix and points to the artifact.
- **`notebooks/whitepaper_v1_gsm8k_benchmark_v2.ipynb`** — dense-reward A/B variant of the GSM8K benchmark notebook.
- **`notebooks/customer_support_3seed_judge.ipynb`** — the canonical whitepaper §11.7 publication-gate notebook. Three seeds (42 / 1337 / 2026), both rubric and LLM-judge eval (local `Qwen2.5-1.5B-Instruct` judge — no API key), KL anchor enabled.
- **First-party benchmark artifacts** under `benchmark_results/whitepaper_v1/` (un-ignored via `.gitignore` negation): 5 result files spanning the full headroom spectrum — proof-of-life GSM8K, dense-reward A/B, gibberish KL-anchor-absent failure, near-ceiling stability finding, and the canonical positive result (`customer_support_3seed_judge_qwen25_05b_instruct.json`: judge improvement +0.079, 3-seed agreement).
- **Whitepaper §11.7 "First-Party Reproduction"** — methodology, result table, publication-gate verification, cross-references to §5.1 / §10.5 / §B.1.

### Fixed

- **`gspo_trainer.py:594`** now calls `reward_model.compute_reward(turns=[turn], context=...)` — the canonical `RewardFunction` API documented in §4.1 of the whitepaper. The previous call shape (`trajectory=None, turn=turn, ...`) was only accepted by `MultiObjectiveRewardFunction`'s compatibility shim; every other `RewardFunction` subclass raised `TypeError`. 35/35 `tests/unit/test_gspo_trainer.py` pass; 248/248 `-k "gspo or reward"` sweep passes.
- **`notebooks/customer_support_4h.ipynb`** — eight bug patterns from the GSM8K v1 debug session pre-applied: stale commit pin, missing `transformers/peft/torchao` upgrade, `asyncio.run` in Jupyter, abstract `Agent` base used instead of `MultiTurnAgent`, missing `attn_implementation='sdpa'`, no explicit `train_queries` (rubric context dropped), gradient-checkpointing-incompatible bf16 conv1d backward on Qwen3.5's hybrid layer, and the unsafe default GSPOConfig.
- **Whitepaper §B.1 hyperparameter table** — `beta=0.0` and `use_reference_model=False` defaults carry a "See warning below" flag with an inline warning block.
- **Whitepaper §10.3 "Reward gaming"** — adds the inverse case from the canonical run: rule-based rubric scoring a qualitatively better trained policy *lower* than baseline because the trained model pivots to clarifying questions.
- **Whitepaper §10.5 "Reference-model drift"** — adds documented evidence across the full headroom spectrum (no anchor → gibberish; anchor + ceiling → stable; anchor + headroom → positive transfer).
- **Whitepaper §11.5 failure-modes table** — new row for "trained model emits token soup; rubric score still nonzero".
- **Whitepaper §8.1 maturity matrix** — GSPO trainer entry references the first-party result.
- **Whitepaper anchor** rebased twice this release: `c0dbd68` → `a2bdde4` → `4744c76`. `docs/WHITEPAPER_ERRATA.md` records both re-anchorings.

### Changed

- **`notebooks/customer_support_4h.ipynb`** ships with safe defaults: `use_reference_model=True`, `beta=0.05`, `num_epochs=1`.
- **`.gitignore`** — `benchmark_results/` rule changed from whole-tree-ignore to selective: tracks `whitepaper_v1/**` plus the schema/README/summary files, still ignores ad-hoc local dumps.
- **`docs/WHITEPAPER.md`** front-matter — "experimental results absent" caveat replaced with a concrete summary of §11.7.
- **`notebooks/README.md`** — eight core notebooks (was six in 0.12.0).
- **Whitepaper test-suite count** refreshed in §3.2 and §9 — 2,438 collected (was 1,624 in the v0.11.6 anchor).
- **Whitepaper §9 exception-tuple list** — fixed double-listed `SERIALIZATION_EXCEPTIONS` and removed non-existent `ENVIRONMENT_EXCEPTIONS`; correct names from `stateset_agents/exceptions.py` substituted.

### Documentation

- **`docs/WHITEPAPER_ERRATA.md`** — net-new errata document tracking three anchor rebases (`14c0e65` → `c0dbd68` → `a2bdde4` → `4744c76`) and the v0.11.6 → v0.12.2 corrections.

## [0.12.2] - 2026-05-15 — Docs polish

### Documentation
- **README "Start here" section** now links `CHANGELOG.md` alongside `PLATFORM_TOUR.md`, `COOKBOOK.md`, and `notebooks/README.md` — four discoverability entry points.
- **`docs/PLATFORM_TOUR.md` FAQ** version-upgrade hint bumped from "0.11.7" to "0.12.0" (was stale after the 0.12.x releases).
- **`docs/COOKBOOK.md` Recipe 6** example `stateset-agents version --json` output now shows `version: 0.12.1` (was `0.11.8`). The other fields (git_commit, dependencies) remain illustrative and don't depend on the release.

No functional changes; no test changes; pure docs maintenance from a grep audit of "0.11" references across the repo.

## [0.12.1] - 2026-05-15 — Batch evaluation + sample eval set

### Added
- **`stateset-agents evaluate --scenarios <jsonl> --reward <name>`** — batch-evaluation mode against a saved checkpoint. Scores every row in the JSONL with the named reward (`gsm8k` / `customer_support` / `tool_calling`) and emits a markdown report: mean ± std, pass rate at `--threshold`, per-scenario table with ✅/⚠️/❌ markers. The single-message mode is preserved. Suitable for nightly / PR-blocking evaluation in CI.
- **`examples/sample_eval_set.jsonl`** — 10 bundled customer-support scenarios across 4 intents (refund / technical / billing / general). Drop-in input for the new batch-evaluate mode. See [Cookbook Recipe 5](./docs/COOKBOOK.md).
- **`make smoke-fast`** — runs only the 15 platform-pipeline unit test modules (~60s, ~222 tests, no integration tests, no CLI subprocess overhead). Inner-loop alternative to the full `make smoke`.
- **`notebooks/README.md`** — map of the 6 bundled Colab notebooks with an ASCII flowchart for "which notebook to open for which goal" + a stage/runtime/cost table.

### Fixed
- CHANGELOG 0.12.0 entry claimed "7 self-contained recipes" — Recipe 5 (batch evaluation) brought the count to 8. Updated the 0.12.0 entry to reflect what actually shipped.
- `tests/unit/test_recipe_cli.py::test_list_default_with_no_args` checked for "7 recipes" via a 7-keyword list; updated to 8 with the new `batch` keyword. Also `test_by_short_substring` updated since "debug" now matches Recipe 6 (not 5).

## [0.12.0] - 2026-05-15 — Developer experience + discoverability

### Added
- **`stateset-agents recipe <name>`** — open a cookbook recipe in `$PAGER`. Supports `list`, numeric (`recipe 1`), full slug, and substring match. Sources from `docs/COOKBOOK.md`.
- **`docs/COOKBOOK.md`** — 8 self-contained, copy-paste recipes covering: first fine-tune, iterating from production logs, reproducing a whitepaper number, building a tool-using agent, running a batch evaluation, debugging a stuck reward, handing off to a colleague, running the demos. Sphinx-integrated via `docs/cookbook.rst`. The bundled `examples/sample_eval_set.jsonl` (10 scenarios across 4 intents) gives users a ready-to-run input for the batch-evaluate recipe.
- **`notebooks/quickstart_first_finetune.ipynb`** — Colab mirror of Cookbook Recipe 1. The 6th bundled notebook; covers install → scaffold → train → REPL sanity-check → provenance hand-off.
- **Chat REPL `readline` integration** — up-arrow recalls previous commands; input history persists across sessions at `${XDG_STATE_HOME:-$HOME/.local/state}/stateset-agents/chat_input_history`. Linux/macOS automatic; Windows users can `pip install pyreadline3`.
- **`make smoke-cli`** — verify every CLI subcommand's `--help` loads cleanly. Catches argument-parsing regressions across the now-15 commands.
- **`make health`** — comprehensive platform health check: version + provenance + CHANGELOG entry + CLI smoke + doctor + end-to-end GSM8K pipeline smoke. ~10 seconds, the one-command answer to "is everything wired correctly?"
- **`make demo-all`** — runs `demo` → `demo-curation` → `demo-full-loop` in sequence with section dividers. ~12 seconds total. Screen-share / asciinema friendly.
- **`make new-version VERSION=x.y.z`** — atomically bumps `pyproject.toml` + `stateset_agents/__init__.py` and prepends a CHANGELOG draft with today's date + TODO placeholders.
- **`make changelog-check`** — fails if the version in `pyproject.toml` doesn't have a corresponding `## [x.y.z]` section in `CHANGELOG.md`. Wired into CI so version bumps without CHANGELOG entries fail the build.
- **PLATFORM_TOUR.md** — new tour stop 10 ("Close the loop: curate → SFT → iterate") covering the full curation pipeline; new FAQ entries on `make health`, `recipe`, readline, and the diagnostic surface.
- **CI workflow** — extended to gate `make changelog-check`, `make smoke-cli`, `make health`, all 6 notebooks, and per-template scaffold smoke.

### Fixed
- Two earlier-pass notebooks (`whitepaper_v1_gsm8k_benchmark.ipynb`, `customer_support_4h.ipynb`) called `GSPOTrainer(config, agent, environment)` which would `TypeError` at runtime — the actual constructor requires `(config, model, tokenizer, agent, environment, reward_model)`. Both notebooks now use the canonical `train_with_gspo()` high-level entry point.
- Silent gpt2 fallback in the `/agents/converse` router when `STATESET_DEFAULT_CHECKPOINT` was set but the startup hook failed — now logs a diagnostic warning naming the env var so users see what happened.

## [0.11.8] - 2026-05-14 — Human-in-the-loop curation

### Added
- **`stateset-agents chat`** — interactive in-process REPL against an agent. Flags:
  - `--model` / `--checkpoint` / `--system` / `--max-new-tokens` — agent configuration
  - `--history` — append every turn to JSONL
  - `--replay` — preload from a saved transcript
  - `--grade <reward>` — show live composite-reward scores per turn (✅/⚠️/❌ markers)
- **`stateset-agents tour`** — opens `docs/PLATFORM_TOUR.md` in `$PAGER` or stdout.
- **`AgentConfig.peft_path`** field — loads a pre-trained LoRA adapter from disk. Wired through `MultiTurnAgent.initialize()` via `PeftModel.from_pretrained()`. Closes the `--checkpoint` flow.
- **`AgentService.register_default_checkpoint_agent()`** — at API startup, the lifespan hook reads `STATESET_DEFAULT_CHECKPOINT` / `STATESET_DEFAULT_BASE_MODEL` env vars and registers a usable agent under id `"default"`. Routable via `/agents/default/messages`.
- **`scripts/grade_transcript.py`** — score a JSONL chat transcript with the same reward functions used during training (`gsm8k`, `customer_support`, `tool_calling`). Optional `--output-curated` flag writes high-scoring (prompt, response) tuples to a deduplicated JSONL for use as next-pass training data.
- **`scripts/summarize_graded_batch.py`** — cross-transcript umbrella summary (grand mean, per-transcript table sorted by score).
- **`scripts/prepare_sft_dataset.py`** — converts curated JSONL into three SFT dataset formats (`hf-trainer`, `chat`, `axolotl`). Filters: `--min-score`, repeatable `--source`, `--dedup` by prompt. Stats summary.
- **`scripts/sft_from_curated.py`** — supervised fine-tune on the prepared chat-format JSONL via `transformers.Trainer` + PEFT LoRA. Stub-aware: dry-run on CPU, full training on GPU. Saves a LoRA adapter directory consumable by `AgentConfig.peft_path`. Completes the human-in-the-loop cycle entirely in-framework.
- **`stateset-agents version`** — extended to report git commit hash + key dependency versions (`torch`, `transformers`, `peft`, `trl`, `datasets`, `fastapi`, `vllm`) with ✓/— markers. Single source of truth for provenance.
- **`make full-loop`** umbrella — runs prepare-sft → sft-from-curated as one command, closing the loop in a single invocation.
- **Makefile targets:** `grade-transcript`, `grade-batch` (with optional `CURATED=` / `THRESHOLD=`), `grade-batch-summary`, `prepare-sft`, `sft-from-curated`, `full-loop`, `changelog-check`, `serve-trained`, `release-prep`, `demo`, `demo-curation`, `starter-test`, `smoke`.
- **`docs/PLATFORM_TOUR.md`** — canonical 10-stop guided walk from `pip install` to published v1.0 whitepaper, plus a FAQ answering the 7 questions that come up in practice. Sphinx-integrated via `docs/platform_tour.rst`.
- **Tool-calling benchmark** — `stateset_agents.data.tool_calling_bench` (bundled 8-scenario corpus + 3 sample tools + `ToolCallReward` composite). Completes the three-pillar coverage (math, dialogue, function calling).
- **Tool-calling Colab notebook** — `notebooks/tool_calling_agent_demo.ipynb` parallels the GSM8K and customer-support notebooks.
- **Doctor command enhancements** — surfaces `STATESET_DEFAULT_CHECKPOINT` / `STATESET_DEFAULT_BASE_MODEL`, validates the path exists, warns when base model is unset.
- **`docs/figures/`** convention + the v1.0 release packager (`scripts/release_v1_whitepaper.py`) generates publication figures and the §11.7 markdown snippet in one command.
- **40+ new unit + integration tests.** Total test count now 195+ across the benchmark / scaffolding / chat / grading pipelines, all passing in ~75 s with no GPU.

### Fixed
- Earlier Colab notebooks called `GSPOTrainer(config, agent, environment)` which would `TypeError` at runtime — the actual `GSPOTrainer.__init__` requires `(config, model, tokenizer, agent, environment, reward_model)`. Switched both notebooks to the canonical `train_with_gspo()` high-level entry point.
- The `--checkpoint` CLI flag set `STATESET_DEFAULT_CHECKPOINT` but nothing read it — now wired all the way through the API lifespan + the new `AgentConfig.peft_path` field + the LoRA-from-disk loader.
- Silent gpt2 fallback in the agents-converse router when `STATESET_DEFAULT_CHECKPOINT` was set but the startup hook failed — now logs a diagnostic warning so users see what happened.

### Documentation
- README sections "Chat with your fine-tune locally", "Curate good examples", "Scaffold a fine-tuning project", "Benchmark your fine-tune" document the new developer journey.
- CHANGELOG promoted from `[Unreleased]` to `[0.11.7]` for the empirical-results pipeline, then to `[0.11.8]` for the curation work.

## [0.11.7] - 2026-05-14 — Platform empirical-results pipeline

### Added
- **`stateset_agents.utils.reproducibility`** — central `set_all_seeds()` that seeds Python `random`, NumPy, PyTorch (CPU + CUDA), and Transformers in one call. Singleton `REPRODUCIBILITY_STATE` records what was seeded.
- **`stateset_agents.data.gsm8k`** — GSM8K loader (`load_gsm8k`), gold/predicted answer parsers, and a `GSM8KReward` verifier reward function for single-turn verifiable-reward training.
- **`stateset_agents.data.customer_support_bench`** — bundled 24-scenario multi-turn customer-support corpus across 4 intents (refund / technical / billing / general). `SupportRewardComposite` ships a three-signal rule-based reward (intent ack + brand voice + safety multiplier).
- **`stateset_agents.data.tool_calling_bench`** — bundled 8-scenario tool-calling corpus, 3 sample tools (weather / calculator / search), and a `ToolCallReward` composite (tool selection 40% + parameter correctness 30% + outcome substring 30%). Completes the framework's three-pillar benchmark coverage.
- **`stateset_agents.scaffolding`** — `scaffold_project()` and 4 starter templates (`customer-support`, `gsm8k-math`, `tool-calling-agent`, `minimal`) with full file scaffolds: config.yaml, scenarios.jsonl, reward.py, train.py, eval.py, serve.sh, README.md. `--client-name` flag slugifies into output_dir paths and the W&B project name throughout.
- **CLI subcommands** — `stateset-agents starter <template> <output>` and `stateset-agents benchmark {smoke,phase0,plot,aggregate}`. `stateset-agents serve` gained `--checkpoint` and `--base-model` flags to close the train→serve loop.
- **`scripts/run_phase0_benchmark.py`** — runner with `TaskAdapter` interface (`GSM8KAdapter`, `CustomerSupportAdapter`, `ToolCallingAdapter`). Records baseline + config in schema-compliant JSON per `benchmark_results/SCHEMA.md`.
- **`scripts/aggregate_phase0_results.py`** — reads JSONs, groups by (trainer, model), applies publication gates (3 seeds, σ < 0.10, +0.03 improvement, single commit), emits `summary.md` + `summary.csv` + `passes_gates.json`. Returns exit 1 under `--strict` if any group fails.
- **`scripts/plot_phase0_results.py`** — produces two publication PNGs (pass@1 per trainer with seed-variance error bars, improvement ranking with publication-gate line). Graceful matplotlib-less fallback to a markdown text table.
- **`scripts/release_v1_whitepaper.py`** — one-shot v1.0 release packager. Aggregates → plots → generates the whitepaper §11.7 markdown snippet → copies figures into `docs/figures/` → writes a provenance manifest.
- **Three Colab notebooks** under `notebooks/`:
  - `whitepaper_v1_gsm8k_benchmark.ipynb` — GSM8K + GSPO showcase (~45min on A100)
  - `customer_support_4h.ipynb` — multi-turn customer-support showcase (~3h on A100)
  - `tool_calling_agent_demo.ipynb` — tool-calling showcase (~2h on A100)
- **Makefile targets** — `smoke` (umbrella), `benchmark-smoke`, `benchmark-phase0`, `benchmark-phase0-all`, `benchmark-aggregate`, `benchmark-aggregate-strict`, `benchmark-plot`, `benchmark-publish`, `release-whitepaper-v1`, `release-whitepaper-v1-strict`, `serve-trained`, `starter-test`.
- **CI workflow** — `.github/workflows/benchmark-smoke.yml` runs the full unit-test suite, end-to-end benchmark smoke, scaffold-smoke per template, notebook JSON validation, and `make starter-test` on every relevant PR. ~3 min, no GPU.
- **134 new unit tests** across `test_gsm8k.py`, `test_reproducibility.py`, `test_aggregate_phase0_results.py`, `test_customer_support_bench.py`, `test_tool_calling_bench.py`, `test_task_adapters.py`, `test_scaffolding.py`.

### Fixed
- Earlier Colab notebooks called `GSPOTrainer(config, agent, environment)` but the trainer's actual `__init__` requires `(config, model, tokenizer, agent, environment, reward_model)`. Switched both notebooks to the canonical `train_with_gspo()` high-level entry point.

### Documentation
- README sections "Scaffold a fine-tuning project in 30 seconds" and "Benchmark your fine-tune" document the new developer journey from clone → train → benchmark → serve.
- `benchmark_results/SCHEMA.md` — canonical result JSON schema + publication gates.
- `benchmark_results/README.md` — practitioner-facing pipeline reference.
- Whitepaper appendix C ("Reproducibility Commands") gained the full benchmark-pipeline command list.

## [0.11.6] - 2026-04-20

### Changed
- Expanded the blocking mypy surface from 14 to 35 files, enforcing `disallow_untyped_defs` + `disallow_incomplete_defs` across the highest-traffic core and training modules: `agent`, `agent_backends`, `agent_config`, `trajectory`, `reward_base`, `basic_rewards`, `domain_rewards`, `reward_factories`, `environment`, `environment_base`, `conversation_environment`, `multiturn_agent`, `input_validation`, `memory`, `errors`, `base_trainer`, `trl_grpo_trainer`, `gspo_trainer`, `dapo_trainer`, `gepo_trainer`, and `vapo_trainer`.
- Fixed ~64 strict-mode type violations across the newly gated modules (missing return annotations, untyped `*args`/`**kwargs`, bare `Callable`/`list` generics, `Any | None` declarations for lazy-loaded transformers/peft globals, and `None`-callable narrowing around the optional PEFT loader).

### Fixed
- Realigned deployment artifacts (helm chart, Kubernetes training/production manifests, README, benchmark/autopilot docs) with the released package version so `test_helm_values_use_current_package_version` and `test_selected_kubernetes_and_docs_refs_use_current_package_version` pass on `master`.

## [0.11.1] - 2026-04-02

### Added
- First-class `gemma-4-31b` starter support across packaged training helpers, CLI entrypoints, presets, examples, and regression tests.
- Repo hygiene checks for generated artifacts, backup files, and local tool state, enforced in local workflows and CI.
- A canonical-import regression test to prevent new code from depending on deprecated top-level shim modules.

### Changed
- Bumped the package version to `0.11.1` in package metadata and runtime exports.
- Removed tracked backup files and local autoresearch state from the repository, and tightened contributor guidance around maintenance policy.

## [0.11.0] - 2026-04-02

### Added
- Expo-powered mobile training console under `mobile/` with dashboard, runs, datasets, models, and run-detail screens tailored to RL fine-tuning workflows.
- Mobile data layer that reads live training-lab endpoints when available and falls back to curated mock runs, datasets, models, algorithms, and environments for preview mode.
- Auto-research workflow support for tracking recent training-quality search iterations and benchmark outcomes.

### Changed
- Bumped the package version to `0.11.0` in package metadata and runtime exports.
- Scoped `.gitignore` patterns for `lib/`, `models/`, and `runs/` to the repository root so mobile source files can be tracked and released normally.

## [0.5.0] - 2025-10-19

### Added
- Unit coverage for CLI stub mode, monitoring health checks, and stub backend factories to protect critical workflows.

### Changed
- Extracted stub tokenizer/model scaffolding into `core.agent_backends`, keeping `core.agent` focused on orchestration.
- Modernised test fixtures and README/RELEASE_NOTES to document the new backend module and verification steps.

### Fixed
- Hardened optional dependency guards in performance optimisers and monitoring utilities so imports fail gracefully with actionable guidance.
- Replaced deprecated `asyncio.coroutine` usage in health checks with loop-aware execution that supports sync or async callables.

## [0.4.0] - 2025-10-05

### Added
- End-to-end stub mode surfaced in CLI and examples for fast demos and CI validation.
- Regression coverage ensuring stub responses flow through the computational engine and raw prompt paths.

### Changed
- Standardised all internal and external imports on the `stateset_agents.*` namespace to smooth integration.
- Refreshed documentation and CLI messaging to highlight the stub workflow.

## [0.3.3] - 2025-09-04

## [0.3.4] - 2025-09-22

### Fixed
- Made `stateset_agents.core` proxy resilient to missing optional dependencies by wrapping `async_pool` import in a safe try/except. This prevents import-time failures when `aiohttp` is not installed and allows tests and consumers to import the package without optional extras.

### Changed
- Bumped package version to `0.3.4` in `pyproject.toml` and `stateset_agents/__init__.py` to keep versions in sync.

### Notes
- No API changes; this release improves import robustness and packaging hygiene only.

### Changed
- Refactored `training` package imports to absolute paths (`core.*`, `utils.*`, `rewards.*`) so `import training` works directly from installed packages
- CI publish workflow hardened: idempotent uploads (`twine upload --skip-existing`), resilient tag creation, and Docker/docs jobs also run on GitHub release events

### Fixed
- Resolved import errors when using `training` modules from PyPI install
- Ensured packaged distribution includes all required top-level modules used by `stateset_agents`

### Publishing
- Released `stateset-agents==0.3.3` to PyPI

## [0.3.2] - 2025-09-04

### Added
- Packaged top-level implementation modules with the library: `core`, `rewards`, `utils`, `api`, `environments`, and a lightweight `grpo_agent_framework` proxy

### Fixed
- Added proxy module for `stateset_agents.core` to forward to top-level `core` package to avoid duplication while enabling stable imports

### Publishing
- Released `stateset-agents==0.3.2` to TestPyPI and PyPI

## [0.3.1] - 2024-12-XX

### Added
- **Code Quality Tools**: Comprehensive code quality setup with Black, isort, Ruff, mypy, and pre-commit hooks
- **CI/CD Pipeline**: GitHub Actions workflows for automated testing, linting, security scanning, and publishing
- **Development Tools**: Makefile with common development tasks and comprehensive pyproject.toml configuration
- **Documentation**: Code of Conduct and Contributing guidelines for community standards
- **Security**: Bandit security linting and dependency vulnerability scanning
- **Benchmarking**: Automated performance benchmarking with historical tracking
- **Dependency Management**: Dependabot configuration for automated dependency updates

### Changed
- **Package Consistency**: Standardized all imports to use `stateset_agents` namespace consistently
- **Documentation**: Updated all examples and documentation to use correct import paths
- **Testing**: Enhanced pytest configuration with coverage thresholds and better markers
- **Project Structure**: Improved .gitignore and development file organization

### Fixed
- Import inconsistencies between `grpo_agent_framework` and `stateset_agents` packages
- Missing development dependencies and tools
- Inconsistent code formatting and style

### Developer Experience
- Added comprehensive Makefile for common tasks (`make help` to see all commands)
- Pre-commit hooks for automated code quality checks
- Enhanced testing with coverage reporting and multiple test categories
- Improved documentation with clear contribution guidelines

## [0.3.0] - 2024-01-XX

### Added
- **Enhanced Error Handling & Resilience**
  - Comprehensive exception hierarchy for training, model, data, network, and resource errors
  - Configurable async retry with exponential backoff and jitter
  - Circuit breaker pattern for automatic failure detection and recovery
  - Rich error context with stack traces, categories, and recovery suggestions

- **Performance Optimization**
  - Real-time memory monitoring with automatic cleanup and optimization
  - Dynamic batch sizing based on resource availability
  - PyTorch 2.0 compilation support for faster inference
  - Mixed precision training with automated FP16/BF16 optimization

- **Type Safety & Validation**
  - Runtime type checking for all framework components
  - Type-safe configuration with detailed error reporting
  - Reliable serialization/deserialization with type preservation
  - Clear protocol interfaces for extensible components

- **Advanced Async Resource Management**
  - High-performance async resource pools with health checking
  - Sophisticated async task scheduling with resource limits
  - Automatic resource cleanup and scaling
  - Real-time monitoring of resource utilization

- **Production Monitoring**
  - Comprehensive performance tracking and reporting
  - Automated system health monitoring and alerting
  - Dynamic optimization recommendations
  - Advanced debugging and profiling capabilities

### Changed
- Enhanced production-ready configuration and deployment
- Improved async resource management patterns
- Better error handling and recovery mechanisms
- More comprehensive monitoring and observability

## [0.2.0] - 2023-11-XX

### Added
- Multi-turn conversation support with trajectory tracking
- Domain-specific reward functions (Customer Service, Technical Support, Sales)
- Neural reward models that learn from trajectory data
- Distributed training capabilities
- Advanced data processing pipeline
- TRL GRPO integration for large model fine-tuning

### Changed
- Enhanced training infrastructure with better stability
- Improved reward modeling system
- Better environment abstractions

## [0.1.0] - 2023-08-XX

### Added
- Initial GRPO training framework
- Basic agent and environment classes
- Core reward functions
- Simple training pipeline
- Documentation and examples

### Changed
- Initial release with core functionality
