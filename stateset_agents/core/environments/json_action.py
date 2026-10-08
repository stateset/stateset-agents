"""Unambiguous JSON action parsing for executable benchmark sandboxes."""

from __future__ import annotations

import json
import math
from typing import Any, NoReturn

from stateset_agents.core.trajectory import ConversationTurn


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Action JSON must not contain duplicate keys")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    raise ValueError("Action JSON must not contain nonfinite numbers")


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        _reject_constant(value)
    return result


def parse_json_action(action: ConversationTurn) -> tuple[str, dict[str, Any]]:
    """Require one assistant JSON action without duplicate keys or tool calls.

    Malformed and excessively nested model output raises ValueError so the
    sandbox scores it as an invalid action, rather than an infrastructure error.
    Tool-specific argument validation remains the environment's responsibility.
    """
    if (
        action.role != "assistant"
        or not isinstance(action.content, str)
        or action.tool_calls
    ):
        raise ValueError(
            "Expected an assistant JSON action without separate tool calls"
        )
    try:
        parsed = json.loads(
            action.content,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except RecursionError as exc:
        raise ValueError("Action JSON is too deeply nested") from exc
    if not isinstance(parsed, dict) or set(parsed) != {"tool", "args"}:
        raise ValueError("Expected JSON with exactly tool and args keys")
    tool, args = parsed["tool"], parsed["args"]
    if not isinstance(tool, str) or not isinstance(args, dict):
        raise ValueError("Expected a tool name and an args object")
    return tool, args
