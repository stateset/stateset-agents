# StateSet Agents vs LLM Orchestration Frameworks

Frameworks like LangChain, LlamaIndex, and DSPy help you *run* LLM agents (chains, tools, retrieval, evaluators). StateSet Agents helps you *train* those agents with reinforcement learning.

## Different layers of the stack

| Layer | StateSet Agents | LangChain / LlamaIndex / DSPy |
|---|---|---|
| Runtime orchestration | Basic agent runtime for training rollouts | Strong (tools, RAG, routing, memory) |
| Training / optimization | Strong (GRPO/GSPO‑family RL, reward models) | None (external) |
| Reward signals | Native reward abstractions, LLM‑as‑judge, multi‑objective | Evaluators / heuristics only |
| Goal | Improve policies through learning | Build and operate workflows |

## When to choose which

Choose **LangChain/LlamaIndex/DSPy** if:

- Your main problem is **tool/RAG orchestration** or agent workflow design.
- You do not need gradient‑based training; prompt iteration and evaluators are enough.

Choose **StateSet Agents** if:

- You want your agent to **learn from interaction rewards over time**.
- You need to optimize for long‑horizon dialogue behavior or business KPIs.

## Using them together

A common production pattern:

1. Build your agent workflow (tools, retrieval, routing) in LangChain/LlamaIndex/DSPy.
2. Wrap that workflow inside a StateSet `MultiTurnAgent` by overriding `generate_response` / `process_turn`.
3. Define a reward function that scores the workflow outputs.
4. Train with GRPO/GSPO to improve the underlying model or policy.

StateSet does not replace orchestration frameworks; it complements them by adding a learning layer.

