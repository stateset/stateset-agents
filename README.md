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
- Composable **reward functions** (heuristic, domain, multi‑objective, neural, LLM‑judge).
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

---

## What's new

**v0.19.0 (latest release — [live on PyPI](https://pypi.org/project/stateset-agents/)):**

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
pip install stateset-agents          # latest release (v0.19.0)
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
python examples/finetune_gspo.py --list-models              # 12 presets
python examples/finetune_gspo.py --model qwen3.5-0.8b       # dry run: show the resolved config
python examples/finetune_gspo.py --model qwen3.5-0.8b --no-dry-run   # actually train
```

Useful flags: `--starter-profile {balanced,memory,quality}` (the `memory` profile
uses 4‑bit quantization and smaller context/group sizes), `--use-lora/--no-lora`,
`--use-4bit/--use-8bit`, `--use-vllm`, `--wandb`, `--export-merged`,
`--write-config PATH`.

Six models also ship a dedicated starter with tuned defaults and a hosting plan:

| Model | Dedicated entry point | Notes |
|---|---|---|
| `Qwen/Qwen3.5-0.8B` | `stateset-agents qwen3-5-0-8b` | Cheapest path to a first run — see `docs/QWEN3_FINETUNING_GUIDE.md` |
| `google/gemma-4-31B-it` | `stateset-agents gemma-4-31b` | Use `--starter-profile memory` on tighter GPU budgets |
| `moonshotai/Kimi-K2.6` | `stateset-agents kimi-k2-6` | |
| `moonshotai/Kimi-K3` | `stateset-agents kimi-k3` | **Provisional** — HF weights unpublished as of 2026‑07‑16; presets mirror Kimi‑K2.6 |
| `zai-org/GLM-5.1` | `python examples/finetune_glm5_1_gspo.py` | 754B MoE, QLoRA‑only + vLLM; `docs/GLM5_1_HOSTING_PLAN.md` |
| `zai-org/GLM-5.2` | `python examples/finetune_glm5_2_gspo.py` | 754B MoE, QLoRA‑only + vLLM; `docs/GLM5_2_HOSTING_PLAN.md` |

Every CLI starter accepts the same flags: `--json-output`, `--list-profiles`,
`--starter-profile NAME`, `--write-config PATH`, `--config PATH --no-dry-run`.
The GLM starters are importable too (`from stateset_agents.training.glm5_2_starter
import get_glm5_2_config, run_glm5_2_config`), as are the others.

### Supported models

First-class starters ship for **Qwen 3.5 0.8B**, **Gemma 4 31B IT**, **Kimi-K2.6**, **Kimi-K3** *(provisional)*, **GLM 5.1**, and **GLM 5.2**. Reference examples and hosting plans cover Qwen 3.5 27B, Qwen 3, Qwen 2.5, Kimi-K2.5, Gemma 3 / Gemma 2 27B IT, Llama 3, Llama 2 7B, and Mistral 7B. Any HuggingFace causal LM compatible with `AutoModelForCausalLM` + TRL GRPO is supported through the generic flow.

See [`docs/SUPPORTED_MODELS.md`](docs/SUPPORTED_MODELS.md) for the full matrix, algorithm compatibility, and instructions for adding a new starter.

### Dashboard and mobile app (demo, not deployed)

[`dashboard/`](dashboard/) (React + Vite) and [`mobile/`](mobile/) (Expo)
are working clients for a simulator-backed `/api/lab/*` "Training Lab"
router. Both are real code you can run locally, but neither has a
deployment path today — the router is gated behind auth and the
`API_ENABLE_TRAINING_LAB` flag (off by default), and there is no hosted
instance of either app. See [`dashboard/README.md`](dashboard/README.md)
and [`mobile/README.md`](mobile/README.md) for status and how to run them.

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
stateset-agents train --config ./config.yaml --dry-run false --save ./outputs/ckpt
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
- [`CHANGELOG.md`](CHANGELOG.md) — what changed in each release (latest release `v0.19.0`).

Other entry points:

- **[`examples/getting_started/`](examples/getting_started/)** — **start here after `pip install`**: five small examples (stub hello, custom reward, first GSPO fine-tune, LLM-judge eval, serve via FastAPI). All target the published PyPI version; the GPU-free three smoke-test the install end-to-end. Run `make getting-started-smoke` to verify all three at once.
- `examples/finetune_gspo.py` – **unified finetune driver**: `--model <preset>` over the 12-model registry (`--list-models`), safe `--dry-run` by default, `--no-dry-run` to train
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
