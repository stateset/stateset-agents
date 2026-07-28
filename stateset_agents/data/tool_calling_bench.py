"""
Tool-calling benchmark — completes the framework's three-pillar showcase.

GSM8K (single-turn, verifiable) and ``customer_support_bench`` (multi-turn,
dialogue) cover two of the framework's three pillars. This module covers the
third: **function calling**. An agent learns to invoke the right tool, pass
the right parameters, and produce a response that contains the expected
outcome.

This module mirrors ``customer_support_bench`` in shape:

* :class:`ToolCallScenario` — a parsed scenario with an expected tool, params,
  and outcome string.
* :func:`load_tool_call_scenarios` — bundled 8-scenario corpus across 3 tools
  (weather, calculator, search), deterministic and reproducible without
  external API calls.
* :class:`ToolCallReward` — composite reward: tool selection (40%) +
  parameter correctness (30%) + outcome substring match (30%).

Usage::

    from stateset_agents.data.tool_calling_bench import (
        load_tool_call_scenarios,
        ToolCallReward,
        SAMPLE_TOOLS,
    )
    from stateset_agents.core.tool_agent import ToolAgent
    from stateset_agents.core import ConversationEnvironment

    scenarios = load_tool_call_scenarios()
    agent = ToolAgent(config=..., tools=SAMPLE_TOOLS)
    env = ConversationEnvironment(
        scenarios=[s.to_scenario() for s in scenarios],
        reward_fn=ToolCallReward(),
        max_turns=1,
    )

The agent emits responses with JSON tool-call blocks in the standard format::

    ```json
    {"tool": "calculator", "parameters": {"expression": "17 * 24"}}
    ```

``ToolCallReward`` parses the first such block per assistant turn.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..core.reward_base import RewardFunction, RewardResult, RewardType
from ..core.trajectory import ConversationTurn

# ---------------------------------------------------------------------------
# Sample tool registry
# ---------------------------------------------------------------------------

SAMPLE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "The city name."}},
            "required": ["city"],
        },
    },
    {
        "name": "calculator",
        "description": "Evaluate a math expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A math expression like '17 * 24'.",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "search",
        "description": "Search a knowledge base.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."}
            },
            "required": ["query"],
        },
    },
]


# ---------------------------------------------------------------------------
# Bundled scenarios
# ---------------------------------------------------------------------------

_BUNDLED_SCENARIOS: list[dict[str, Any]] = [
    {
        "user_query": "What's the weather in San Francisco?",
        "expected_tool": "get_weather",
        "expected_params": {"city": "San Francisco"},
        "expected_outcome": "63",
    },
    {
        "user_query": "Calculate 17 * 24",
        "expected_tool": "calculator",
        "expected_params": {"expression": "17 * 24"},
        "expected_outcome": "408",
    },
    {
        "user_query": "Look up the population of Tokyo",
        "expected_tool": "search",
        "expected_params": {"query": "population of Tokyo"},
        "expected_outcome": "13.96",
    },
    {
        "user_query": "Find recent papers on diffusion models",
        "expected_tool": "search",
        "expected_params": {"query": "diffusion models"},
        "expected_outcome": "papers",
    },
    {
        "user_query": "Calculate the square root of 144",
        "expected_tool": "calculator",
        "expected_params": {"expression": "sqrt(144)"},
        "expected_outcome": "12",
    },
    {
        "user_query": "What's the weather forecast for Paris tomorrow?",
        "expected_tool": "get_weather",
        "expected_params": {"city": "Paris"},
        "expected_outcome": "58",
    },
    {
        "user_query": "How many calories in 200g of rice?",
        "expected_tool": "search",
        "expected_params": {"query": "calories in 200g rice"},
        "expected_outcome": "260",
    },
    {
        "user_query": "Compute 2 to the power of 10",
        "expected_tool": "calculator",
        "expected_params": {"expression": "2 ** 10"},
        "expected_outcome": "1024",
    },
]


@dataclass
class ToolCallScenario:
    """A single tool-calling scenario with an expected outcome."""

    user_query: str
    expected_tool: str
    expected_params: dict[str, Any] = field(default_factory=dict)
    expected_outcome: str = ""

    def to_scenario(self) -> dict[str, Any]:
        return {
            "user_query": self.user_query,
            "expected_tool": self.expected_tool,
            "expected_params": dict(self.expected_params),
            "expected_outcome": self.expected_outcome,
        }


def load_tool_call_scenarios(
    tool_filter: str | None = None, limit: int | None = None
) -> list[ToolCallScenario]:
    """Load the bundled tool-calling scenarios.

    Args:
        tool_filter: If set, return only scenarios whose ``expected_tool``
            matches this value.
        limit: If set, return at most this many scenarios.
    """
    scenarios = [
        ToolCallScenario(
            user_query=s["user_query"],
            expected_tool=s["expected_tool"],
            expected_params=dict(s["expected_params"]),
            expected_outcome=s["expected_outcome"],
        )
        for s in _BUNDLED_SCENARIOS
    ]
    if tool_filter is not None:
        scenarios = [s for s in scenarios if s.expected_tool == tool_filter]
    if limit is not None:
        scenarios = scenarios[:limit]
    return scenarios


def make_tool_call_scenarios(scenarios: list[ToolCallScenario]) -> list[dict[str, Any]]:
    """Convert ``ToolCallScenario`` objects to ``ConversationEnvironment`` scenarios."""
    return [s.to_scenario() for s in scenarios]


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

_TOOL_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def extract_tool_call(response: str) -> dict[str, Any] | None:
    """Parse the first ``{"tool": ..., "parameters": ...}`` JSON block in ``response``.

    Returns None if no block is present, the JSON is malformed, or the block
    doesn't have a ``tool`` key.
    """
    if not response:
        return None
    match = _TOOL_BLOCK_RE.search(response)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "tool" not in data:
        return None
    return data


class ToolCallReward(RewardFunction):
    """Composite reward for tool-calling agents.

    Three signals, weighted:

    * **Tool selection** (default 0.4) — did the agent invoke the
      ``expected_tool``?
    * **Parameter correctness** (default 0.3) — fraction of expected parameter
      keys whose stringified values match (case-insensitive after stripping).
    * **Outcome substring** (default 0.3) — does the response anywhere contain
      ``expected_outcome`` as a substring?

    The reward parses the first JSON tool-call block in the response and
    expects ``context`` to contain ``expected_tool``, ``expected_params``,
    and ``expected_outcome``.
    """

    name = "tool_call_composite"

    def __init__(
        self,
        weight: float = 1.0,
        tool_selection_weight: float = 0.4,
        param_correctness_weight: float = 0.3,
        outcome_weight: float = 0.3,
    ) -> None:
        super().__init__(weight=weight, reward_type=RewardType.SPARSE, name=self.name)
        self.tool_selection_weight = tool_selection_weight
        self.param_correctness_weight = param_correctness_weight
        self.outcome_weight = outcome_weight

    async def compute_reward(
        self,
        turns: list[ConversationTurn],
        context: dict[str, Any] | None = None,
    ) -> RewardResult:
        if not turns:
            return RewardResult(score=0.0, breakdown={"no_response": 1.0})

        full_response = "\n".join(
            t.content for t in turns if t.role == "assistant" and t.content
        )
        ctx = context or {}
        expected_tool = ctx.get("expected_tool", "")
        expected_params = ctx.get("expected_params", {}) or {}
        expected_outcome = str(ctx.get("expected_outcome", ""))

        call = extract_tool_call(full_response)

        # 1) Tool selection.
        if call is None:
            tool_score = 0.0
        elif call.get("tool", "") == expected_tool:
            tool_score = 1.0
        else:
            tool_score = 0.0

        # 2) Parameter correctness.
        if call is None or not expected_params:
            param_score = 1.0 if not expected_params else 0.0
        else:
            params = call.get("parameters", {}) or {}
            if not isinstance(params, dict):
                param_score = 0.0
            else:
                matched = sum(
                    1
                    for k, v in expected_params.items()
                    if str(params.get(k, "")).strip().lower() == str(v).strip().lower()
                )
                param_score = matched / max(len(expected_params), 1)

        # 3) Outcome substring.
        if expected_outcome:
            outcome_score = (
                1.0 if expected_outcome.lower() in full_response.lower() else 0.0
            )
        else:
            outcome_score = 1.0

        composite = (
            self.tool_selection_weight * tool_score
            + self.param_correctness_weight * param_score
            + self.outcome_weight * outcome_score
        )

        return RewardResult(
            score=float(composite),
            breakdown={
                "tool_selection": tool_score,
                "param_correctness": param_score,
                "outcome": outcome_score,
                "tool_called": (call or {}).get("tool", ""),
            },
            explanation=(
                f"expected_tool={expected_tool} got={(call or {}).get('tool', 'none')} "
                f"params={param_score:.2f} outcome={outcome_score:.2f}"
            ),
        )


__all__ = [
    "SAMPLE_TOOLS",
    "ToolCallReward",
    "ToolCallScenario",
    "extract_tool_call",
    "load_tool_call_scenarios",
    "make_tool_call_scenarios",
]
