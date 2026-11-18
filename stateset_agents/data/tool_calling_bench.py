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
from collections import Counter
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


def normalize_tool_call(call: Any) -> Any:
    """Normalize OpenAI and LangChain calls for chat templates, retaining IDs."""
    if not isinstance(call, dict):
        return call
    function = call.get("function")
    if isinstance(function, dict):
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                return call
            return {**call, "function": {**function, "arguments": arguments}}
    elif "name" in call and "args" in call:
        return {
            **({"id": call["id"]} if "id" in call else {}),
            "type": "function",
            "function": {"name": call["name"], "arguments": call["args"]},
        }
    return call


def _same_json_value(left: Any, right: Any) -> bool:
    """Compare JSON trees, treating 42 and 42.0 as numbers and true as a boolean."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same_json_value(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _same_json_value(a, b) for a, b in zip(left, right, strict=True)
        )
    return isinstance(left, bool) == isinstance(right, bool) and bool(left == right)


def tool_call_references(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Read the exact tool calls required to grade one assistant action or reply."""
    if "expected_tool_calls" not in context:
        tool = context.get("expected_tool")
        if not isinstance(tool, str) or not tool.strip():
            raise ValueError("Missing expected_tool")
        if not isinstance(context.get("expected_params"), dict):
            raise ValueError("expected_params must be an object")
        ref = {"tool": tool, "args": context["expected_params"]}
        if "expected_tool_result" in context:
            ref["result"] = context["expected_tool_result"]
        return [ref]

    if {"expected_tool", "expected_params", "expected_tool_result"} & context.keys():
        raise ValueError("Choose either expected_tool_calls or the single-tool fields")
    refs = context["expected_tool_calls"]
    if not isinstance(refs, list) or not refs:
        raise ValueError("expected_tool_calls must be a nonempty list")
    for i, ref in enumerate(refs):
        if (
            not isinstance(ref, dict)
            or set(ref) - {"tool", "args", "result"}
            or not isinstance(ref.get("tool"), str)
            or not ref["tool"].strip()
            or not isinstance(ref.get("args"), dict)
        ):
            raise ValueError(
                f"expected_tool_calls[{i}] requires tool, args and optionally result"
            )
    if any(("result" in ref) != ("result" in refs[0]) for ref in refs):
        raise ValueError("Supply a result for every expected call when grading a reply")
    return refs


def _preceding_tool_batch(
    turns: list[ConversationTurn],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Pair the immediately preceding batch with its contiguous result messages."""
    index = len(turns) - 2
    results: dict[str, str] = {}
    while index >= 0 and turns[index].role == "tool":
        turn = turns[index]
        call_id = turn.metadata.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id.strip() or call_id in results:
            raise ValueError("Tool result has a missing or duplicate call ID")
        if not isinstance(turn.content, str):
            raise ValueError("Missing tool result content")
        results[call_id] = turn.content
        index -= 1
    if (
        not results
        or index < 0
        or turns[index].role != "assistant"
        or turns[-1].role != "assistant"
        or turns[-1].tool_calls
    ):
        raise ValueError("Missing linked tool batch for this reply")
    calls = turns[index].tool_calls or []
    ids = [call.get("id") if isinstance(call, dict) else None for call in calls]
    if any(not isinstance(i, str) or not i.strip() for i in ids):
        raise ValueError("Tool call is missing its ID")
    if len(set(ids)) != len(calls):
        raise ValueError("Duplicate tool call ID in the batch")
    if set(ids) != results.keys():
        raise ValueError(
            "Tool results must match every call ID in the preceding batch exactly once"
        )
    return calls, results


def _result_matches(expected: Any, content: str) -> bool:
    if isinstance(expected, str):
        return expected == content
    try:
        return _same_json_value(expected, json.loads(content))
    except ValueError:
        return False


def _matches_batch(
    refs: list[dict[str, Any]], observed: list[dict[str, Any]], *, check_results: bool
) -> bool:
    if len(refs) != len(observed):
        return False
    remaining = observed.copy()
    # Raw string expectations have a narrower match than parsed JSON (e.g.
    # "42" versus 42). Reserve exact text matches first for repeated calls.
    for ref in sorted(refs, key=lambda r: not isinstance(r.get("result"), str)):
        for index, call in enumerate(remaining):
            if (
                ref["tool"] == call["tool"]
                and _same_json_value(ref["args"], call["args"])
                and (
                    not check_results or _result_matches(ref["result"], call["result"])
                )
            ):
                remaining.pop(index)
                break
        else:
            return False
    return True


def _grade_tool_batch(
    turns: list[ConversationTurn],
    refs: list[dict[str, Any]],
    expected_outcome: str,
    passing_score: float,
) -> RewardResult:
    verify_results = "result" in refs[0]
    results: dict[str, str] = {}
    calls = turns[-1].tool_calls or []
    if verify_results:
        try:
            calls, results = _preceding_tool_batch(turns)
        except ValueError as exc:
            return RewardResult(score=0, metadata={"ungradable_reason": str(exc)})

    observed = []
    for raw in calls:
        native = normalize_tool_call(raw)
        fn = native.get("function") if isinstance(native, dict) else None
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            return RewardResult(score=0, explanation="Malformed tool call")
        action = {"tool": fn["name"], "args": fn.get("arguments")}
        if verify_results:
            action["result"] = results[raw["id"]]
        observed.append(action)

    breakdown = {
        "tool_selection": float(
            Counter(r["tool"] for r in refs) == Counter(c["tool"] for c in observed)
        ),
        "param_correctness": float(_matches_batch(refs, observed, check_results=False)),
        "outcome": float(expected_outcome.lower() in (turns[-1].content or "").lower()),
    }
    if verify_results:
        breakdown["tool_results"] = float(
            _matches_batch(refs, observed, check_results=True)
        )
    failed = [component for component, score in breakdown.items() if not score]
    return RewardResult(
        score=0.0 if failed else passing_score,
        breakdown=breakdown,
        metadata={"tool_called": observed[0]["tool"]} if len(observed) == 1 else {},
        explanation=(
            "Mismatch in " + ", ".join(failed)
            if failed
            else "All referenced tool calls matched"
        ),
    )


def _parameter_score(
    call: dict[str, Any] | None,
    expected: dict[str, Any],
    *,
    structured: bool,
    per_turn: bool,
) -> float:
    params = call.get("parameters") if call else None
    if per_turn:
        return float(isinstance(params, dict) and _same_json_value(params, expected))
    if not expected:
        return 1.0
    if not isinstance(params, dict):
        return 0.0
    if structured:
        matches = sum(
            k in params and _same_json_value(params[k], v) for k, v in expected.items()
        )
    else:
        matches = sum(
            str(params.get(k, "")).strip().lower() == str(v).strip().lower()
            for k, v in expected.items()
        )
    return matches / len(expected)


class ToolCallReward(RewardFunction):
    """Composite reward for tool-calling agents.

    Three signals, weighted:

    * **Tool selection** (default 0.4) — did the agent invoke the
      ``expected_tool``?
    * **Parameter correctness** (default 0.3) — fraction of expected parameter
      keys whose values match. Native calls compare typed JSON values. Fenced
      text calls retain their case-insensitive string comparison.
    * **Outcome substring** (default 0.3) — does the response anywhere contain
      ``expected_outcome`` as a substring?

    The reward reads native tool calls or a JSON tool-call block and
    expects ``context`` to contain ``expected_tool``, ``expected_params``,
    and an optional ``expected_outcome``. In per-turn grading, supplying
    ``expected_tool_result`` verifies the linked call and result immediately
    preceding the final text reply. Use ``expected_tool_calls`` for a parallel
    batch, with one tool, args and optional result per call.
    """

    name = "tool_call_composite"

    def __init__(
        self,
        weight: float = 1.0,
        tool_selection_weight: float = 0.4,
        param_correctness_weight: float = 0.3,
        outcome_weight: float = 0.3,
        *,
        per_turn: bool = False,
    ) -> None:
        super().__init__(weight=weight, reward_type=RewardType.SPARSE, name=self.name)
        self.tool_selection_weight = tool_selection_weight
        self.param_correctness_weight = param_correctness_weight
        self.outcome_weight = outcome_weight
        self.per_turn = per_turn

    async def compute_reward(
        self,
        turns: list[ConversationTurn],
        context: dict[str, Any] | None = None,
    ) -> RewardResult:
        if not turns:
            return RewardResult(score=0.0, breakdown={"no_response": 1.0})

        ctx = context or {}
        if self.per_turn and (
            "expected_tool_calls" in ctx or "expected_tool_result" in ctx
        ):
            try:
                refs = tool_call_references(ctx)
            except ValueError as exc:
                return RewardResult(score=0, metadata={"ungradable_reason": str(exc)})
            return _grade_tool_batch(
                turns,
                refs,
                str(ctx.get("expected_outcome", "")),
                self.tool_selection_weight
                + self.param_correctness_weight
                + self.outcome_weight,
            )

        replies = [t for t in turns if t.role == "assistant"]
        if self.per_turn:
            replies = replies[-1:]
        full_response = "\n".join(t.content for t in replies if t.content)
        expected_tool = ctx.get("expected_tool", "")
        expected_params = ctx.get("expected_params", {}) or {}
        expected_outcome = str(ctx.get("expected_outcome", ""))

        native_calls = [call for reply in replies for call in (reply.tool_calls or [])]
        if native_calls:
            native = normalize_tool_call(native_calls[0])
            function = native.get("function") if isinstance(native, dict) else None
            if self.per_turn and len(native_calls) != 1:
                return RewardResult(
                    score=0.0, explanation="Expected a single tool action"
                )
            call = (
                {"tool": function.get("name"), "parameters": function.get("arguments")}
                if isinstance(function, dict)
                else None
            )
        else:
            call = extract_tool_call(full_response)

        tool_score = float(call is not None and call.get("tool") == expected_tool)
        param_score = _parameter_score(
            call, expected_params, structured=bool(native_calls), per_turn=self.per_turn
        )

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
        if self.per_turn and not (tool_score and param_score and outcome_score):
            composite = 0.0

        tool_called = call["tool"] if call else ""
        breakdown: dict[str, Any] = {
            "tool_selection": tool_score,
            "param_correctness": param_score,
            "outcome": outcome_score,
        }
        if not self.per_turn:
            # Existing benchmark reports include the tool name alongside numeric scores.
            breakdown["tool_called"] = tool_called
        return RewardResult(
            score=float(composite),
            breakdown=breakdown,
            metadata={"tool_called": tool_called},
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
