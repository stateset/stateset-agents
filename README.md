<div align="center">

# StateSet Agents

**Reinforcement‑learning framework for multi‑turn conversational AI agents.**

[![PyPI version](https://badge.fury.io/py/stateset-agents.svg)](https://pypi.org/project/stateset-agents/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: BUSL-1.1](https://img.shields.io/badge/License-BUSL--1.1-green.svg)](LICENSE)

</div>

StateSet Agents is a production‑oriented RL stack for training and serving LLM‑backed agents that improve through **multi‑turn interaction**. The library provides:

- Async‑first **agent APIs** (`MultiTurnAgent`, `ToolAgent`) with Hugging Face and stub backends.
- **Environments** for conversational and task‑oriented episodes.
- **Trajectories** and value/advantage utilities tailored to dialogue.
- Composable **reward functions** (heuristic, domain, multi‑objective, neural).
- A family of **group‑based policy‑optimization trainers** (GRPO, GSPO, GEPO, DAPO, VAPO) plus PPO and RLAIF.
- Optional **performance layers** (vLLM generation, Rust acceleration, distributed training, HPO, FastAPI service).

If you want a framework that treats conversations as first‑class RL episodes (rather than single turns), this is it.

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

## Installation

### Core (lightweight, stub‑ready)

```bash
pip install stateset-agents
```

### Training / real models

```bash
pip install "stateset-agents[training]"
```

### Optional extras

```bash
pip install "stateset-agents[trl]"        # TRL GRPO integration + bitsandbytes
pip install "stateset-agents[vllm]"       # vLLM generation backend
pip install "stateset-agents[hpo]"        # Optuna/Ray Tune HPO
pip install "stateset-agents[api]"        # FastAPI service
pip install "stateset-agents[distributed]"# DeepSpeed / multi‑GPU helpers
pip install "stateset-agents[full]"       # Most extras in one go
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
        AgentConfig(model_name="gpt2", max_new_tokens=128, temperature=0.7)
    )
    await agent.initialize()
    messages = [{"role": "user", "content": "What is GRPO?"}]
    print(await agent.generate_response(messages))

asyncio.run(main())
```

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
    agent = MultiTurnAgent(AgentConfig(model_name="gpt2"))
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
        num_episodes=50,
        profile="balanced",
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

base_cfg = get_config_for_task("customer_service", model_name="gpt2")
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

## CLI

The CLI is a thin wrapper around the Python API:

```bash
stateset-agents version
stateset-agents doctor
stateset-agents train --stub
stateset-agents train --config ./config.yaml --dry-run false --save ./outputs/ckpt
stateset-agents evaluate --checkpoint ./outputs/ckpt --message "Hello"
stateset-agents serve --host 0.0.0.0 --port 8001
```

For complex runs prefer the Python API and the examples folder.

---

## Examples and docs

Good starting points:

- `examples/hello_world.py` – stub mode walkthrough
- `examples/quick_start.py` – basic agent + environment
- `examples/complete_grpo_training.py` – end‑to‑end GRPO training
- `examples/train_with_gspo.py` – GSPO + GSPO‑token training
- `examples/train_with_trl_grpo.py` – Hugging Face TRL GRPO integration

Key docs:

- `docs/USAGE_GUIDE.md`
- `docs/RL_FRAMEWORK_GUIDE.md`
- `docs/GSPO_GUIDE.md`
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
