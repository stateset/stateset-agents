# StateSet Agents Comparisons

These documents explain how StateSet Agents relates to other reinforcement‑learning and LLM agent frameworks. The goal is clarity about scope, workflows, and trade‑offs (not to claim universal benchmarks).

## Quick positioning

- **StateSet Agents** is a production‑oriented RL framework for **multi‑turn conversational LLM agents**. It includes group‑based policy‑optimization algorithms (GRPO, GSPO, GEPO, DAPO, VAPO), composable reward modeling (including neural/LLM rewards), async‑first trainers, observability, and deployment templates.
- **Traditional RL frameworks** (Ray RLlib, Stable‑Baselines3, CleanRL, Acme, TorchRL) are **general MDP toolkits** for games/robotics/control. They can be adapted to LLM agents but usually require extra scaffolding for dialogue state, trajectory grouping, and reward modeling.
- **Hugging Face TRL** is a **low‑level RLHF/LLM optimization library**. StateSet integrates with TRL for GRPO training while providing higher‑level agent/environment abstractions.
- **LLM orchestration frameworks** (LangChain, LlamaIndex, DSPy) focus on **tool/RAG/flow composition** and evaluation. They do not provide gradient‑based RL training loops; StateSet provides the training layer.

## Comparison matrix (high‑level)

| Dimension | StateSet Agents | TRL | RLlib | SB3 / CleanRL | LangChain / LlamaIndex / DSPy |
|---|---|---|---|---|---|
| Primary focus | Multi‑turn conversational agents + RLHF‑style training | Transformer RLHF trainers | Generic RL at scale | Research‑grade RL algorithms | Agent/tool/RAG orchestration |
| Typical tasks | Dialogue policy improvement, reasoning RL, multi‑objective business RL | Prompt→response RLHF, preference fine‑tuning | Simulators, multi‑agent MDPs | Gym‑style environments | Building agent workflows |
| Algorithms shipped | GRPO, GSPO, GEPO, DAPO, VAPO, PPO, DPO, A2C, TRPO | PPO, DPO/ORPO‑style methods, GRPO variants | Wide algorithm zoo (PPO, IMPALA, SAC, etc.) | PPO/A2C/SAC/TD3/etc. | None (training is external) |
| Environment model | Conversation + task environments; multi‑turn trajectories | Dataset‑driven RLHF loops | Gym/RL‑Env abstractions | Gym/RL‑Env abstractions | Tool/RAG graphs |
| Multi‑turn support | Native | Limited / user‑built | User‑built | User‑built | Native (runtime only) |
| Reward modeling | Composable rewards, neural/LLM rewards, multi‑objective | Reward models as callouts | Reward from env step | Reward from env step | Heuristics/eval scores |
| Distributed training | Built‑in async + Accelerate; HPO extras | Accelerate/DeepSpeed integrations | Strong distributed story | Limited (single‑node focus) | N/A |
| Production serving | FastAPI services, deployment templates | Not a serving framework | Ray Serve optional | Not a serving framework | Runtime serving integrations |
| Best when | You need RL for real multi‑turn agents, end‑to‑end | You want a lightweight RLHF trainer | You already live in Ray/sim RL | You want minimal RL baselines | You want orchestration, not RL |

## Detailed comparisons

- `docs/COMPARISON_TRL.md` — StateSet Agents vs Hugging Face TRL.
- `docs/COMPARISON_TRADITIONAL_RL.md` — StateSet Agents vs RLlib, SB3, CleanRL, Acme, TorchRL.
- `docs/COMPARISON_LLM_FRAMEWORKS.md` — StateSet Agents vs LangChain, LlamaIndex, DSPy.

If you want a comparison to another specific framework, open an issue or PR and we can add it here.

