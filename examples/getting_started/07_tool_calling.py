"""07 — Tool-calling with `ToolAgent` and the bundled tool benchmark.

Demonstrates the function-calling pillar of the framework: an agent that
decides which of several registered tools to invoke, with what parameters,
and produces a response containing the expected outcome. Uses the bundled
``ToolCallReward`` to score each of the three signals.

Stub-backed, GPU-free. The same agent + reward + scenarios drop straight
into a `GSPOConfig` training run — see `notebooks/tool_calling_agent_demo.ipynb`
for the trained version.

Install:
    pip install stateset-agents

Run:
    python 07_tool_calling.py

Expected output:
    Registered 3 tools.
    Loaded 8 tool-call scenarios.
    --- well-formed call --- score=1.00 (tool 1.0, params 1.0, outcome 1.0)
    --- wrong tool ---       score=0.30 (tool 0.0, params 0.0, outcome 1.0)
    --- malformed JSON ---   score=0.30 (tool 0.0, params 0.0, outcome 1.0)
"""

import asyncio

from stateset_agents import ToolAgent  # use the lazy export to avoid a circular import
from stateset_agents.core.agent_config import AgentConfig
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.data import SAMPLE_TOOLS, ToolCallReward, load_tool_call_scenarios


# Three handcrafted response strings cover the three failure modes a trainer
# will see during rollout: well-formed, wrong tool, malformed JSON.
GOOD = """I'll check that for you.
```json
{"tool": "calculator", "parameters": {"expression": "17 * 24"}}
```
The answer is 408."""

WRONG_TOOL = """Let me search for that.
```json
{"tool": "search", "parameters": {"query": "17 * 24"}}
```
I found 408."""

MALFORMED = """Sure! `calculator(17 * 24)` returns 408 obviously."""


async def main() -> None:
    agent = ToolAgent(
        config=AgentConfig(model_name="stub://tools", use_stub_model=True),
        tools=SAMPLE_TOOLS,
    )
    await agent.initialize()
    print(f"Registered {len(agent.tools)} tools.")

    scenarios = load_tool_call_scenarios()
    print(f"Loaded {len(scenarios)} tool-call scenarios.")

    # Take one scenario (calculator: 17*24=408) and score three candidate responses.
    target = next(s for s in scenarios if s.expected_tool == "calculator")
    context = target.to_scenario()

    reward = ToolCallReward()
    cases = [("well-formed call", GOOD), ("wrong tool", WRONG_TOOL), ("malformed JSON", MALFORMED)]
    for label, response in cases:
        turns = [ConversationTurn(role="assistant", content=response)]
        result = await reward.compute_reward(turns, context=context)
        print(f"--- {label} ---")
        print(f"score={result.score:.2f}  breakdown={result.breakdown}")


if __name__ == "__main__":
    asyncio.run(main())
