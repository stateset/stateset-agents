# StateSet Agents vs Hugging Face TRL

This document compares StateSet Agents to Hugging Face TRL (Transformer Reinforcement Learning). Both are used for training LLMs with reinforcement learning, but they sit at different layers of the stack.

## What TRL is best at

TRL provides reusable, low‑level trainers and utilities for RLHF‑style optimization of transformer models (e.g., PPO/DPO/GRPO‑family). It is a good fit when you want:

- A lightweight library focused on the **optimizer/trainer layer**.
- Dataset‑driven **prompt→response** RLHF loops.
- Tight integration with Hugging Face `transformers`, `accelerate`, and PEFT/LoRA.

## What StateSet Agents is best at

StateSet Agents is a higher‑level framework focused on **multi‑turn agent learning** and production deployment. It is a good fit when you want:

- Native **multi‑turn conversation trajectories** and dialogue state.
- Group‑based LLM RL algorithms beyond standard PPO (GRPO, GSPO, GEPO, DAPO, VAPO).
- **Composable rewards** (rule‑based, neural reward models, LLM‑as‑judge, multi‑objective rewards).
- Async‑first training and orchestration designed for large‑scale agent rollouts.
- Built‑in observability, health checks, and API serving.

## Feature comparison

| Capability | StateSet Agents | TRL |
|---|---|---|
| Multi‑turn agent runtime | Yes (`core/multiturn_agent.py`) | No (user‑built) |
| Conversation/task environments | Yes (`core/environment.py`) | No (dataset + sampler) |
| Group‑based RL algorithms | Yes (GRPO/GSPO/GEPO/DAPO/VAPO) | Partial (GRPO/PPO/DPO‑family) |
| Reward composition | Yes (`rewards/`, `core/reward.py`) | Minimal (reward fn/model hook) |
| Distributed/async rollouts | Yes (`training/distributed_trainer.py`) | Via Accelerate/DeepSpeed |
| Production serving | Yes (`api/`) | No |
| HPO support | Built‑in (`training/hpo/`, Optuna/Ray/W&B extras) | External |

## When to choose which

Choose **TRL** if:

- You are doing a **single‑turn RLHF** pipeline and want to stay close to Hugging Face primitives.
- You already have custom rollout and reward infrastructure.
- You want the smallest possible dependency surface.

Choose **StateSet Agents** if:

- Your agent needs to learn across **multi‑turn dialogues or tool‑using workflows**.
- You want built‑in reward modeling, evaluation, monitoring, and serving.
- You need GRPO/GSPO‑style group training as a first‑class workflow.

## Using them together

StateSet Agents includes TRL integration (see `training/trl_grpo_trainer.py`). A common pattern:

1. Use **StateSet** to define the agent, environment, reward, and rollout strategy.
2. Delegate the **optimizer step** to TRL trainers when appropriate.

This gives you TRL’s mature transformer training utilities with StateSet’s agent‑level scaffolding.

## Migration notes

If you have a TRL script already:

- Move prompt generation and logging into a `ConversationEnvironment`.
- Wrap your model in a `MultiTurnAgent` (or a custom subclass).
- Port the reward function into `RewardFunction` or `CompositeReward`.
- Keep the TRL optimizer configuration; StateSet’s TRL trainer accepts equivalent settings.
