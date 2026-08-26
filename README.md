<div align="center">

# StateSet Agents

**Reinforcement‑learning framework for multi‑turn conversational AI agents.**

[![PyPI version](https://img.shields.io/pypi/v/stateset-agents.svg)](https://pypi.org/project/stateset-agents/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: BUSL-1.1](https://img.shields.io/badge/License-BUSL--1.1-green.svg)](LICENSE)
[![Whitepaper v0.13.4](https://img.shields.io/badge/whitepaper-v0.13.4-blue)](docs/WHITEPAPER.md)
[![First-party result](https://img.shields.io/badge/§11.7-judge%20%2B0.079%20%E2%9C%93-brightgreen)](benchmark_results/whitepaper_v1/customer_support_3seed_judge_qwen25_05b_instruct.json)

</div>

StateSet Agents is a production‑oriented RL stack for training and serving LLM‑backed agents that improve through **multi‑turn interaction**. The library provides:

- An **improvement loop** that turns your agent's own logs into a better agent: `ingest` → `improve` (grade → curate) → fine‑tune.
- Async‑first **agent APIs** (`MultiTurnAgent`, `ToolAgent`) with Hugging Face and stub backends.
- **Environments** for conversational and task‑oriented episodes.
- **Trajectories** and value/advantage utilities tailored to dialogue.
- Composable **reward functions** (heuristic, domain, multi‑objective, neural, LLM‑judge, and a proof‑backed [StateSet NSR verifier](docs/NSR_INTEGRATION.md) for RLVR‑style verifiable rewards).
- A family of **group‑based policy‑optimization trainers** (GRPO, GSPO, GEPO, DAPO, VAPO) plus PPO and RLAIF.
- **Offline RL algorithms** for learning from logged conversations (BCQ, BEAR, CQL, IQL, Decision Transformer).
- **Sim‑to‑Real transfer** for training in simulation and deploying to real users (domain randomization, system identification, progressive transfer).
- **Continual learning + long‑term planning** utilities (replay/LwF/EWC, plan context injection).
- An **MCP server** so Claude Code/Desktop — or any MCP client — can drive the loop conversationally.
- Optional **performance layers** (vLLM generation, Rust acceleration, distributed training, HPO, FastAPI service).

If you want a framework that treats conversations as first‑class RL episodes (rather than single turns), this is it.

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

**v0.42.1 (latest release; PyPI publication pending):**

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
  episodes ([`PROOFS.md`](docs/PROOFS.md)).

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
- **[`docs/GETTING_STARTED_RIVER.md`](docs/GETTING_STARTED_RIVER.md)** —
  train, self-improve, and RL-tune with zero machines rented, written
  entirely from 20+ live runs, including a failure-mode table where every
  row actually happened to us.
- **[`docs/rl-vibe.md`](docs/rl-vibe.md)** — the state of the framework,
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
  ([`PROOFS.md`](docs/PROOFS.md)).
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
  ([`PROOFS.md`](docs/PROOFS.md)).

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
  ([`PROOFS.md`](docs/PROOFS.md)).

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
  [`PROOFS.md`](docs/PROOFS.md) row: *RL reward hacking is detectable and
  fixable in-platform.*

**v0.34.1:**

- **The RL flywheel is live-verified — with an honest verdict.** First real
  run (35B MoE, depth-3 refusal kit, zero infrastructure): **7/12 → 10/12
  by round 1**, mean graded reward monotone 0.72 → 0.89 across 4 rounds.
  The head-to-head from the identical start on the identical kit: SFT on
  the 171-row harvest reached **11/12** — so untuned RL lands at rough
  parity, not superiority, and BOTH methods retain exactly one refusal
  violation. Verified, not oversold ([`PROOFS.md`](docs/PROOFS.md)).
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
  ([`FLYWHEEL_DOMAIN2.md`](docs/FLYWHEEL_DOMAIN2.md), [`PROOFS.md`](docs/PROOFS.md)).

**v0.33.0:**

- **The zero-infrastructure flywheel** (`flywheel --provider river`):
  harvest via River's sampling API, train via their remote autograd, no
  machines rented. Live-verified: the TechNest 9B checkpoint went 7/12 →
  **11/12** on compound tickets (harvest rate 67% → 88%) with a correct
  plateau-stop — the ceiling-raise's third substrate
  ([`FLYWHEEL_DOMAIN2.md`](docs/FLYWHEEL_DOMAIN2.md)).
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
  The silent no-op documented in [`docs/PROOFS.md`](docs/PROOFS.md) is now
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
  ([`FLYWHEEL_DOMAIN2.md`](docs/FLYWHEEL_DOMAIN2.md)).
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
  [`docs/PROOFS.md`](docs/PROOFS.md)).
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
  LoRA adapters — see [`docs/PROOFS.md`](docs/PROOFS.md); `chat-remote`
  remains the verified way to talk to hybrid fine-tunes.)*
  Two live-only bugs fell out and are fixed with tests: a flashinfer
  annotation that crashes vLLM's engine on Python 3.11 (now patched in place
  post-install) and a shell-precedence bug that made the self-destruct arm
  hold the ssh channel for the pod's whole lifetime.
- **The flywheel raises a ceiling: 0/12 → 2/12 → 10/12.** On compound
  requests provably outside gen-1's training distribution, best-of-8
  rejection sampling harvested gen-1's rare successes and training gen-2 on
  only those took the eval from 2/12 to 10/12, reproduced across two
  independent trainings for $3.32 ([`FLYWHEEL_HEADROOM.md`](docs/FLYWHEEL_HEADROOM.md)).
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
  itemised with symptoms in [`docs/FIREWORKS_PROVIDER.md`](docs/FIREWORKS_PROVIDER.md).

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
- **Two guides written entirely from runs that happened**: [`RUNPOD_GUIDE.md`](docs/RUNPOD_GUIDE.md) (disk sizing with the exact error a too-small disk produces, GPU/model table, spot pricing, network volumes, multi-GPU, and every failure mode we hit) and [`GETTING_STARTED_API.md`](docs/GETTING_STARTED_API.md) (raw logs to an OpenAI-compatible or Anthropic-style API call).

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
- **NVIDIA Nemotron 3.5 Lightning starter.** `stateset-agents nemotron-3-5` targets `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (hybrid Mamba‑2 + MoE, 30B total / 3B active, 256K ctx, OpenMDW‑1.1) with Mamba-aware LoRA targets. [`docs/nemotron_3_5_starter.rst`](docs/nemotron_3_5_starter.rst)
- **Quality gates that enforce themselves.** Coverage and mypy-allowlist ratchets fail CI when floors fall behind reality; the security workflow's scanners now actually gate (45 HIGH/CRITICAL findings cleared); `make benchmark-loop` scores the improvement loop against planted ground truth (precision 0.818 / recall 1.0); a weekly `gpu-verify` workflow re-proves the training job on rented hardware.
- **Slimmer repo.** `dashboard/` and `mobile/` moved to their own repositories.

**v0.21.0:**

- **Muse Glimmer 30B first‑class starter.** `stateset-agents muse-glimmer` targets `meta-models/Muse-Glimmer-30B` — Meta's open agentic model (Aug 2026; dense 30B, 131K ctx, Apache‑2.0) — with the standard balanced/memory/quality QLoRA profiles, `init --preset muse-glimmer`, and a `muse-glimmer` preset in the unified finetune driver. [`docs/muse_glimmer_starter.rst`](docs/muse_glimmer_starter.rst)
- **RunPod provider for `train-remote`.** `--provider runpod` rents a GPU pod over SSH, runs the same packaged job every other provider runs, and copies the adapter back. GPU defaults are now per‑provider — `RemoteJobSpec.gpu` no longer hard‑codes a Modal‑specific name.

**v0.20.0:**

- **Run the fine‑tune step without a GPU.** `stateset-agents train-remote` runs the SFT job from `improve` on rented compute (`--provider local|modal`), closing the last gap in the improvement loop. The job itself is unchanged whichever provider runs it, and remote runs install a pinned published package rather than syncing your working tree. [`docs/CLI_REFERENCE.md`](docs/CLI_REFERENCE.md)

**v0.19.0:**

- **MCP server.** `stateset-agents mcp` (`pip install stateset-agents[mcp]`) exposes the improvement loop as tools for any MCP client — Claude Code/Desktop or your own agent. Seven tools, stdio transport, dry‑run‑only training. [`docs/MCP_SERVER.md`](docs/MCP_SERVER.md)

**v0.18.0:**

- **Bring your own agent's logs.** `stateset-agents ingest` converts OpenAI chat‑format and LangChain conversation dumps into framework trajectories — logs from agents built *anywhere* plug straight into the loop.
- **The improvement loop in one command.** `stateset-agents improve run` grades, curates, and emits verified‑runnable training commands (a regression test executes every suggestion against the real CLI parsers).
- **Flagship benchmark recipe.** `make flagship-benchmark-all` — a reproducible 3‑seed GSPO run on an 8B model with publish gates ([`benchmarks/FLAGSHIP.md`](benchmarks/FLAGSHIP.md)).

**v0.16.0 – v0.17.3 (correctness and distribution):**

- **RL‑core correctness overhaul.** All five trainers fixed and behaviorally tested: DAPO freezes rollout‑time old log probs and honors µ inner updates; GEPO runs in log space; GSPO scores exactly the text it sampled and rescores vLLM rollouts; GSPO‑token regained its gradient path; VAPO clips values against rollout predictions with terminal‑token rewards and decoupled GAE; the GRPO loss path uses length‑normalized ratios. A cross‑trainer ratio‑invariant suite guards all of it.
- **Convergence proof in CI.** A nightly job trains a real (tiny) model and asserts the target‑token probability strictly increases — verified against zero‑signal and reversed‑reward controls.
- **API hardening.** Training‑lab routes auth‑gated behind `API_ENABLE_TRAINING_LAB` (off outside development), bounded in‑memory state, fail‑closed production config validation, constant‑time key comparison, identity‑keyed rate limiting.
- **Distribution repaired.** PyPI is current again, wheels ship the runtime config presets, [`stateset-rl-core`](https://pypi.org/project/stateset-rl-core/) is published so `[rust]` and `[full]` resolve, and CI is green across 4 Python versions + Windows.
- **Unified finetune driver.** `examples/finetune_gspo.py --model <preset>` replaced the per‑model script maze with a 12‑model preset registry.

Earlier highlights: v0.15.3 shipped Rust accelerator wheels (abi3‑py310) and model‑level Prometheus inference metrics; v0.15.0 added the getting‑started ladder; v0.13.2 shipped the whitepaper §11.7 three‑seed canonical benchmark (judge improvement **+0.079**, [artifact](benchmark_results/whitepaper_v1/customer_support_3seed_judge_qwen25_05b_instruct.json)).

Full breakdown in [CHANGELOG.md](CHANGELOG.md).

---

## Renting GPUs, and knowing what it cost

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

```bash
pip install stateset-agents          # latest release (v0.42.1)
```

That's enough for the [five-minute demo](#the-improvement-loop), the stub
backend, and the CLI. Training real models needs `[training]` below.

> PyPI tracks the release tags. For unreleased work on master:
>
> ```bash
> pip install "git+https://github.com/stateset/stateset-agents@master"
> ```

### Training / real models

```bash
pip install "stateset-agents[training]"
```

### Optional extras

```bash
pip install "stateset-agents[auto-research]" # Autonomous experiment loop + Optuna
pip install "stateset-agents[trl]"           # TRL GRPO integration + bitsandbytes
pip install "stateset-agents[vllm]"          # vLLM generation backend
pip install "stateset-agents[hpo]"           # Optuna/Ray Tune HPO
pip install "stateset-agents[api]"           # FastAPI service
pip install "stateset-agents[distributed]"   # DeepSpeed / multi‑GPU helpers
pip install "stateset-agents[rust]"          # Rust-accelerated GAE/advantage kernels (stateset-rl-core)
pip install "stateset-agents[full]"          # Most extras in one go
```

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
| `Qwen/Qwen3.5-0.8B` | `stateset-agents qwen3-5-0-8b` | ✅ | Cheapest path to a first run (~$0.30) |
| `meta-models/Muse-Glimmer-30B` | `stateset-agents muse-glimmer` | ✅ | Meta's open agentic model; dense 30B, multimodal, 131K ctx, Apache‑2.0 |
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | `stateset-agents nemotron-3-5` | ✅ | Hybrid Mamba‑2 + MoE reasoning model, 3B active params |
| `Qwen/Qwen3.8-27B` | `stateset-agents qwen3-8-27b` | ✅ | Hybrid linear/standard attention, multimodal, 256K ctx, Apache‑2.0 — ~56GB BF16 |
| `Qwen/Qwen3.8-Flash-Next` | `python examples/finetune_gspo.py --model qwen3.8-flash-next` | | Qwen4 architecture preview; 125B main / 6B active + 51B n-gram embeddings; native multimodal, 262K ctx |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct` | `stateset-agents qwen3-coder` | | 128 experts / 8 active, 256K ctx, Apache‑2.0 |
| `openai/gpt-oss-20b` | `stateset-agents gpt-oss` | | 32 experts / 4 active, 128K ctx, Apache‑2.0 |
| `deepseek-ai/DeepSeek-V4-Flash` | `stateset-agents deepseek-v4` | | MLA attention, 256 experts, 1M ctx, MIT — QLoRA + vLLM |
| `google/gemma-4-31B-it` | `stateset-agents gemma-4-31b` | | Use `--starter-profile memory` on tighter GPU budgets |
| `moonshotai/Kimi-K2.6` | `stateset-agents kimi-k2-6` | | |
| `moonshotai/Kimi-K3` | `stateset-agents kimi-k3` | | **Provisional** — HF weights unpublished as of 2026‑07‑16 |
| `zai-org/GLM-5.1` | `python examples/finetune_glm5_1_gspo.py` | | 754B MoE, QLoRA‑only + vLLM |
| `zai-org/GLM-5.2` | `python examples/finetune_glm5_2_gspo.py` | | 754B MoE, QLoRA‑only + vLLM |
| `zai-org/GLM-5.3-Flash` | `python examples/finetune_gspo.py --model glm5.3-flash` | | 320B / 18B active, native multimodal, FP8, 1M ctx |

Every CLI starter accepts the same flags: `--json-output`, `--list-profiles`,
`--starter-profile NAME`, `--write-config PATH`, `--config PATH --no-dry-run`.
The GLM starters are importable too (`from stateset_agents.training.glm5_2_starter
import get_glm5_2_config, run_glm5_2_config`), as are the others.

### Supported models

First-class starters ship for **Qwen 3.5 0.8B**, **Muse Glimmer 30B**, **Nemotron 3.5 Lightning**, **Qwen3.8 27B**, **Qwen3-Coder 30B**, **gpt-oss 20B**, **DeepSeek V4 Flash**, **Gemma 4 31B IT**, **Kimi-K2.6**, **Kimi-K3** *(provisional)*, **GLM 5.1**, and **GLM 5.2**. Architecture-aware unified presets additionally cover **GLM-5.3-Flash** and **Qwen3.8-Flash-Next**. Reference examples and hosting plans cover Qwen 3.5 27B, Qwen 3, Qwen 2.5, Kimi-K2.5, Gemma 3 / Gemma 2 27B IT, Llama 3, Llama 2 7B, and Mistral 7B. Compatible Hugging Face causal LMs work through the generic flow; native multimodal conditional-generation repositories use StateSet's composite-loader fallback for text-only RL.

See [`docs/SUPPORTED_MODELS.md`](docs/SUPPORTED_MODELS.md) for the full matrix, algorithm compatibility, and instructions for adding a new starter.

### Experimental namespace

Research-grade modules (neural architecture search, multimodal processing,
long-term planning, few-shot adaptation, the intelligent orchestrator,
adaptive learning controller, and multi-agent coordination) live in
`stateset_agents.experimental`. They carry **no API-stability guarantees**
and may change or be removed in any release. The former
`stateset_agents.core.<module>` import paths still work for one deprecation
cycle and emit a `DeprecationWarning`.

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
- **PPO baseline**: standard PPO trainer for comparison
- **RLAIF**: RL from AI feedback via judge/reward models

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

After training, you usually want a defensible number: *did this actually improve over the base model, by how much, and is it reproducible?* The framework ships a Phase‑0 benchmark pipeline that produces publication‑grade results across **three tasks** (GSM8K, the bundled customer‑support corpus, and the tool‑calling corpus).

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

- **Reproducibility.** `set_all_seeds()` covers Python random, NumPy, PyTorch (CPU + CUDA), and Transformers in one call. Every result JSON carries the git commit hash.
- **Schema.** Each run produces a single JSON conforming to `benchmark_results/SCHEMA.md`. Every published number traces back to a file.
- **Publication gates.** 3 seeds, σ < 0.10, +0.03 improvement, single commit. Use `make benchmark-aggregate-strict` in CI to enforce.
- **Figures.** `make benchmark-plot` produces two whitepaper‑ready PNGs (pass@1 per trainer, improvement ranking) plus a matplotlib‑free text fallback.
- **One‑shot release.** `make release-whitepaper-v1` aggregates → plots → generates the whitepaper §11.7 markdown snippet → copies figures into `docs/figures/` → writes a release manifest. Six artifacts in one command.

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
- [`CHANGELOG.md`](CHANGELOG.md) — what changed in each release (latest release `v0.42.1`).
- [`docs/RELEASE_EVIDENCE.md`](docs/RELEASE_EVIDENCE.md) — exact test,
  provider, GPU, cleanup, and publication claims for the current release.

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

## Contributing

See `CONTRIBUTING.md`. Please run `pytest -q` and format with `black`/`isort` before opening a PR.

---

## License

Business Source License 1.1. Non‑production use permitted until **2029‑09‑03**, then transitions to Apache 2.0. See `LICENSE`.
