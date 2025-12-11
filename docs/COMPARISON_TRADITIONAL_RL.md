# StateSet Agents vs Traditional RL Frameworks

Traditional RL libraries such as Ray RLlib, Stable‑Baselines3 (SB3), CleanRL, DeepMind Acme, and TorchRL were designed for generic Markov‑decision‑process settings (games, robotics, control). StateSet Agents targets a different problem: **multi‑turn conversational LLM agents**.

## Scope differences

| Dimension | StateSet Agents | Traditional RL frameworks |
|---|---|---|
| Target domain | Conversational / tool‑using LLM agents | Generic MDPs |
| Episode structure | Multi‑turn dialogues; variable‑length trajectories | Step‑based rollouts |
| State representation | Dialogue history + structured context | Numerical/structured state tensors |
| Action space | Token sequences or tool calls | Discrete/continuous actions |
| Reward | Composed turn‑level or sequence‑level rewards; neural/LLM judges | Scalar from env step |
| Algorithms | Group‑based LLM RL (GRPO/GSPO/GEPO/DAPO/VAPO) + PPO/DPO/etc. | PPO/SAC/A2C/TD3/IMPALA/etc. |

## Practical implications

- **Environment work**: In RLlib/SB3/CleanRL, you typically build a Gym‑style env and flatten dialogue into steps. In StateSet, `ConversationEnvironment` and `MultiTurnTrajectory` are native.
- **Credit assignment**: Group‑based advantages and sequence‑level ratios are first‑class in StateSet; in traditional libs you must re‑implement this logic.
- **Tool‑using agents**: StateSet models tool calls as part of trajectories and rewards; traditional RL treats tools as part of the env you define.
- **Production concerns**: StateSet ships monitoring, health checks, and serving templates. Traditional frameworks focus on training only.

## When traditional frameworks win

Use RLlib/SB3/CleanRL/Acme/TorchRL when:

- You are training on **simulated environments** with well‑defined state tensors.
- You need a broad suite of classic RL algorithms or off‑policy methods.
- You already rely on Ray’s distributed ecosystem or existing Gym benchmarks.

## When StateSet Agents wins

Use StateSet Agents when:

- The task is **language‑based, multi‑turn, or tool‑augmented**.
- Rewards come from **LLM judges, heuristics, or business multi‑objectives**, not a simulator.
- You want to train using **GRPO/GSPO‑family algorithms** with minimal boilerplate.
- You need an end‑to‑end path from training to serving.

## Interop patterns

If you are migrating a traditional RL setup:

- Keep your simulator or task logic, but expose it via a `ConversationEnvironment` or `TaskEnvironment`.
- Map each episode to a `MultiTurnTrajectory` (StateSet handles grouping and advantage computation).
- Reuse policy/value architecture ideas; StateSet’s trainers accept custom heads and value functions.

