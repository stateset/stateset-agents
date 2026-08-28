# StateSet Agents Comparisons

This page explains how StateSet Agents relates to other reinforcement-learning
and LLM-agent frameworks. Feature statements were reviewed against upstream
documentation on **2026-08-28**. They are positioning context, not benchmark
results; measured claims are limited to [`BENCHMARKS.md`](BENCHMARKS.md).

## Quick positioning

- **StateSet Agents** is an improvement control plane for deployed agents:
  production traces, multi-turn environments, composable rewards, training,
  evaluation, lineage, remote execution, and serving are one workflow.
- **TRL** now supports tools, stateful per-rollout environments, and
  asynchronous GRPO in addition to its broad post-training trainer surface.
- **verl** targets high-throughput distributed post-training with FSDP/FSDP2,
  Megatron, vLLM/SGLang, multi-turn tool use, multimodality, and very large
  model/cluster configurations.
- **NeMo RL** targets scalable NVIDIA post-training with Ray, FSDP2/Megatron,
  FP8, asynchronous RL, multimodality, and multi-node recipes.
- **OpenRLHF** targets distributed agentic RLHF through Ray, vLLM, and
  DeepSpeed, including multi-turn and multimodal workflows.
- **Traditional RL frameworks** (Ray RLlib, Stable‑Baselines3, CleanRL, Acme, TorchRL) are **general MDP toolkits** for games/robotics/control. They can be adapted to LLM agents but usually require extra scaffolding for dialogue state, trajectory grouping, and reward modeling.
- **LLM orchestration frameworks** (LangChain, LlamaIndex, DSPy) focus on **tool/RAG/flow composition** and evaluation. They do not provide gradient‑based RL training loops; StateSet provides the training layer.

## Comparison matrix (high‑level)

| Dimension | StateSet | TRL | verl | NeMo RL | OpenRLHF |
|---|---|---|---|---|---|
| Primary focus | Trace-to-policy agent improvement loop | Accessible model post-training | Flexible, high-throughput distributed RL | NVIDIA-scale post-training | Ray/vLLM/DeepSpeed RLHF |
| Multi-turn/tools | Native trajectory and environment model | Tools and stateful rollout environments | Multi-turn tool calling | Multi-turn and environment isolation | Multi-turn agent workflows |
| Distinctive strength | Rewards, evals, lineage, remote execution, and serving together | Hugging Face ecosystem and trainer UX | Placement, resharding, backend breadth, scale | Megatron/FSDP2/FP8 and large-model recipes | Colocated/hybrid distributed execution |
| Current StateSet evidence | — | Three-seed single-A40 parity | Not measured | Not measured | Not measured |
| Best current fit | Production-agent improvement lifecycle | Direct post-training scripts | Maximum distributed flexibility/throughput | NVIDIA multi-node/model scale | Ray-centric distributed RLHF |

The architecture is compositional: StateSet owns the stable trace, environment,
reward, evaluation, and lineage contracts while training may run through the
StateSet reference implementation or a specialized backend. The versioned
[backend protocol](TRAINING_BACKENDS.md) now enforces semantic digests and
capabilities. The executable OpenRLHF adapter is implemented but still awaits
live GPU conformance; executable verl and NeMo RL adapters remain roadmap work.
None are presented as benchmark wins today.

---

## StateSet Agents vs Hugging Face TRL

Both projects train language models with reinforcement learning, but StateSet
spans more of the operational lifecycle. TRL is no longer accurately described
as single-turn only.

### What TRL is best at

TRL provides reusable, low‑level trainers and utilities for RLHF‑style optimization of transformer models (e.g., PPO/DPO/GRPO‑family). It is a good fit when you want:

- A lightweight library focused on the **optimizer/trainer layer**.
- Direct dataset-driven and agent/environment GRPO workflows.
- Tight integration with Hugging Face `transformers`, `accelerate`, and PEFT/LoRA.
- An experimental asynchronous GRPO trainer that decouples vLLM generation
  from optimization.

### What StateSet Agents is best at

StateSet Agents is a higher‑level framework focused on **multi‑turn agent learning** and production deployment. It is a good fit when you want:

- Native **multi‑turn conversation trajectories** and dialogue state.
- Group‑based LLM RL algorithms beyond standard PPO (GRPO, GSPO, GEPO, DAPO, VAPO).
- **Composable rewards** (rule‑based, neural reward models, LLM‑as‑judge, multi‑objective rewards).
- Async‑first training and orchestration designed for large‑scale agent rollouts.
- Built‑in observability, health checks, and API serving.

### Feature comparison

| Capability | StateSet Agents | TRL |
|---|---|---|
| Multi‑turn agent runtime | Yes (`core/multiturn_agent.py`) | Yes in GRPO tool/environment workflows |
| Conversation/task environments | Yes (`core/environment.py`) | Stateful per-rollout environment factory |
| Group‑based RL algorithms | Yes (GRPO/GSPO/GEPO/DAPO/VAPO) | Partial (GRPO/PPO/DPO‑family) |
| Reward composition | Yes (`rewards/`, `core/reward.py`) | Minimal (reward fn/model hook) |
| Distributed/async rollouts | Yes (`training/distributed_trainer.py`) | Accelerate/DeepSpeed plus experimental async GRPO |
| Production serving | Yes (`api/`) | No |
| HPO support | Built‑in (`training/hpo/`, Optuna/Ray/W&B extras) | External |

### When to choose which

Choose **TRL** if:

- You want to stay close to Hugging Face primitives and trainer APIs.
- You already have custom rollout and reward infrastructure.
- You want the smallest possible dependency surface.

Choose **StateSet Agents** if:

- Your agent needs to learn across **multi‑turn dialogues or tool‑using workflows**.
- You want built‑in reward modeling, evaluation, monitoring, and serving.
- You need GRPO/GSPO‑style group training as a first‑class workflow.

### What the benchmark establishes

On the retained four-step Qwen2.5-0.5B/GSM8K A40 protocol, StateSet and direct
TRL have equivalent throughput, wall time, and peak VRAM within run-to-run
noise. The small, short experiment does not establish a learning-quality
winner. See the [validated report](../benchmark_results/framework_comparison/report/comparison.md).

### Using them together

StateSet Agents includes TRL integration (see `training/trl_grpo_trainer.py`). A common pattern:

1. Use **StateSet** to define the agent, environment, reward, and rollout strategy.
2. Delegate the **optimizer step** to TRL trainers when appropriate.

This gives you TRL's mature transformer training utilities with StateSet's agent‑level scaffolding.

### Migration notes

If you have a TRL script already:

- Move prompt generation and logging into a `ConversationEnvironment`.
- Wrap your model in a `MultiTurnAgent` (or a custom subclass).
- Port the reward function into `RewardFunction` or `CompositeReward`.
- Keep the TRL optimizer configuration; StateSet's TRL trainer accepts equivalent settings.

---

## StateSet Agents vs Traditional RL Frameworks

Traditional RL libraries such as Ray RLlib, Stable‑Baselines3 (SB3), CleanRL, DeepMind Acme, and TorchRL were designed for generic Markov‑decision‑process settings (games, robotics, control). StateSet Agents targets a different problem: **multi‑turn conversational LLM agents**.

### Scope differences

| Dimension | StateSet Agents | Traditional RL frameworks |
|---|---|---|
| Target domain | Conversational / tool‑using LLM agents | Generic MDPs |
| Episode structure | Multi‑turn dialogues; variable‑length trajectories | Step‑based rollouts |
| State representation | Dialogue history + structured context | Numerical/structured state tensors |
| Action space | Token sequences or tool calls | Discrete/continuous actions |
| Reward | Composed turn‑level or sequence‑level rewards; neural/LLM judges | Scalar from env step |
| Algorithms | Group‑based LLM RL (GRPO/GSPO/GEPO/DAPO/VAPO) + PPO/DPO/etc. | PPO/SAC/A2C/TD3/IMPALA/etc. |

### Practical implications

- **Environment work**: In RLlib/SB3/CleanRL, you typically build a Gym‑style env and flatten dialogue into steps. In StateSet, `ConversationEnvironment` and `MultiTurnTrajectory` are native.
- **Credit assignment**: Group‑based advantages and sequence‑level ratios are first‑class in StateSet; in traditional libs you must re‑implement this logic.
- **Tool‑using agents**: StateSet models tool calls as part of trajectories and rewards; traditional RL treats tools as part of the env you define.
- **Production concerns**: StateSet ships monitoring, health checks, and serving templates. Traditional frameworks focus on training only.

### When traditional frameworks win

Use RLlib/SB3/CleanRL/Acme/TorchRL when:

- You are training on **simulated environments** with well‑defined state tensors.
- You need a broad suite of classic RL algorithms or off‑policy methods.
- You already rely on Ray's distributed ecosystem or existing Gym benchmarks.

### When StateSet Agents wins

Use StateSet Agents when:

- The task is **language‑based, multi‑turn, or tool‑augmented**.
- Rewards come from **LLM judges, heuristics, or business multi‑objectives**, not a simulator.
- You want to train using **GRPO/GSPO‑family algorithms** with minimal boilerplate.
- You need an end‑to‑end path from training to serving.

### Interop patterns

If you are migrating a traditional RL setup:

- Keep your simulator or task logic, but expose it via a `ConversationEnvironment` or `TaskEnvironment`.
- Map each episode to a `MultiTurnTrajectory` (StateSet handles grouping and advantage computation).
- Reuse policy/value architecture ideas; StateSet's trainers accept custom heads and value functions.

---

## StateSet Agents vs LLM Orchestration Frameworks

Frameworks like LangChain, LlamaIndex, and DSPy help you *run* LLM agents (chains, tools, retrieval, evaluators). StateSet Agents helps you *train* those agents with reinforcement learning.

### Different layers of the stack

| Layer | StateSet Agents | LangChain / LlamaIndex / DSPy |
|---|---|---|
| Runtime orchestration | Basic agent runtime for training rollouts | Strong (tools, RAG, routing, memory) |
| Training / optimization | Strong (GRPO/GSPO‑family RL, reward models) | None (external) |
| Reward signals | Native reward abstractions, LLM‑as‑judge, multi‑objective | Evaluators / heuristics only |
| Goal | Improve policies through learning | Build and operate workflows |

### When to choose which

Choose **LangChain/LlamaIndex/DSPy** if:

- Your main problem is **tool/RAG orchestration** or agent workflow design.
- You do not need gradient‑based training; prompt iteration and evaluators are enough.

Choose **StateSet Agents** if:

- You want your agent to **learn from interaction rewards over time**.
- You need to optimize for long‑horizon dialogue behavior or business KPIs.

### Using them together

A common production pattern:

1. Build your agent workflow (tools, retrieval, routing) in LangChain/LlamaIndex/DSPy.
2. Wrap that workflow inside a StateSet `MultiTurnAgent` by overriding `generate_response` / `process_turn`.
3. Define a reward function that scores the workflow outputs.
4. Train with GRPO/GSPO to improve the underlying model or policy.

StateSet does not replace orchestration frameworks; it complements them by adding a learning layer.
