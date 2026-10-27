"""Parse the tool calls a policy writes into its reply.

Two written forms are recognized, plus structured calls on the turn:

* the chat-template form that Hermes/Qwen-style templates render and models
  are trained to emit, one or more
  ``<tool_call>{"name": "calculator", "arguments": {"expression": "12*7"}}</tool_call>``
  blocks anywhere in the reply (``arguments`` may also be a JSON string);
* the JSON-action form the benchmark sandboxes already use
  (:func:`stateset_agents.core.environments.json_action.parse_json_action`):
  the whole reply is ``{"tool": "calculator", "args": {...}}``. Only a JSON
  object that names a tool (a ``"tool"`` key, or ``"name"`` with
  ``"arguments"``) is a call; any other whole-reply JSON, such as
  ``{"answer": "84"}``, is a final answer;
* ``ConversationTurn.tool_calls`` in the OpenAI shape
  (``{"id": ..., "function": {"name": ..., "arguments": "<json>"}}``), parsed
  with :meth:`stateset_agents.core.function_calling.ToolCall.parsed_arguments`.
  A call's own ``id`` is kept (``has_own_id``); calls without one are
  numbered ``call_<n>``.

Parsing never raises on model output. A call that cannot be read (bad JSON,
duplicate keys, non-finite numbers, a missing name, an unterminated
``<tool_call>`` block) becomes a :class:`ToolCallError`, which the
environment turns into a structured error observation. JSON is parsed
strictly (:func:`~stateset_agents.core.environments.json_action.strict_json_loads`),
so an ambiguous call is never resolved to one reading.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .environments.json_action import parse_json_action, strict_json_loads
from .function_calling import ToolCall
from .trajectory import ConversationTurn

__all__ = [
    "TOOL_CALL_CLOSE",
    "TOOL_CALL_OPEN",
    "ToolCallError",
    "ToolCallParse",
    "ToolCallRequest",
    "parse_tool_calls",
    "strip_tool_calls",
]

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
_BLOCK = re.compile(
    re.escape(TOOL_CALL_OPEN) + r"(.*?)" + re.escape(TOOL_CALL_CLOSE), re.S
)
#: Keep at most this much of a malformed call in its error.
_RAW_LIMIT = 200
#: Keys that make an unparseable whole-reply JSON object a call attempt.
_TOOL_KEY = re.compile(r'"tool"\s*:')
_NAME_KEY = re.compile(r'"name"\s*:')
_ARGUMENTS_KEY = re.compile(r'"arguments"\s*:')


@dataclass(frozen=True)
class ToolCallRequest:
    """One well-formed call: a tool name and its JSON object arguments.

    ``has_own_id`` says ``call_id`` is the call's own id (a structured call's
    ``"id"``), which the observation answering it must keep; otherwise the
    parser numbered the call and an environment may renumber it.
    """

    name: str
    arguments: dict[str, Any]
    call_id: str
    source: str
    has_own_id: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.call_id,
            "name": self.name,
            "arguments": self.arguments,
            "source": self.source,
        }


@dataclass(frozen=True)
class ToolCallError:
    """A call the policy attempted but wrote unreadably (``has_own_id`` as for
    :class:`ToolCallRequest`)."""

    message: str
    source: str
    raw: str
    call_id: str
    has_own_id: bool = False


@dataclass(frozen=True)
class ToolCallParse:
    """Calls and malformed calls in reply order, and the reply's other text."""

    items: tuple[ToolCallRequest | ToolCallError, ...] = ()
    text: str = ""

    @property
    def calls(self) -> tuple[ToolCallRequest, ...]:
        return tuple(i for i in self.items if isinstance(i, ToolCallRequest))

    @property
    def errors(self) -> tuple[ToolCallError, ...]:
        return tuple(i for i in self.items if isinstance(i, ToolCallError))

    @property
    def attempted(self) -> bool:
        """Whether the reply tried to call a tool (well-formed or not)."""
        return bool(self.items)


def _call_from_body(body: str, call_id: str) -> ToolCallRequest | ToolCallError:
    """A ``<tool_call>`` body: ``{"name": str, "arguments": object | str}``."""
    raw = body.strip()[:_RAW_LIMIT]
    try:
        parsed = strict_json_loads(body)
    except ValueError as error:
        return ToolCallError(f"invalid JSON: {error}", "chat_template", raw, call_id)
    if not isinstance(parsed, dict):
        return ToolCallError(
            "a tool call must be a JSON object", "chat_template", raw, call_id
        )
    unknown = set(parsed) - {"name", "arguments"}
    if unknown:
        return ToolCallError(
            f"unexpected keys {sorted(unknown)}; use name and arguments",
            "chat_template",
            raw,
            call_id,
        )
    return _request(parsed.get("name"), parsed.get("arguments", {}), call_id, raw)


def _request(
    name: Any,
    arguments: Any,
    call_id: str,
    raw: str,
    source: str = "chat_template",
    has_own_id: bool = False,
) -> ToolCallRequest | ToolCallError:
    if not isinstance(name, str) or not name.strip():
        return ToolCallError("missing tool name", source, raw, call_id, has_own_id)
    if isinstance(arguments, str):
        try:
            arguments = ToolCall(
                id=call_id, function={"name": name, "arguments": arguments}
            ).parsed_arguments()
        except ValueError as error:
            return ToolCallError(str(error), source, raw, call_id, has_own_id)
    if not isinstance(arguments, dict):
        return ToolCallError(
            "arguments must be a JSON object", source, raw, call_id, has_own_id
        )
    return ToolCallRequest(name.strip(), arguments, call_id, source, has_own_id)


def _own_id(call: Mapping[str, Any]) -> str | None:
    """A structured call's own ``"id"`` (a non-empty string or an integer)."""
    value = call.get("id")
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return None


def _structured_calls(tool_calls: Any) -> list[ToolCallRequest | ToolCallError]:
    items: list[ToolCallRequest | ToolCallError] = []
    for index, call in enumerate(tool_calls or []):
        call_id = f"call_{index}"
        if not isinstance(call, Mapping):
            items.append(
                ToolCallError("tool call must be an object", "structured", "", call_id)
            )
            continue
        # The call's own id is what the observation answering it carries;
        # only a call without one is numbered.
        own = _own_id(call)
        call_id = own if own is not None else call_id
        has_own_id = own is not None
        function = call.get("function") if "function" in call else call
        if not isinstance(function, Mapping):
            items.append(
                ToolCallError(
                    "function must be an object", "structured", "", call_id, has_own_id
                )
            )
            continue
        arguments = function.get("arguments", {})
        if isinstance(arguments, Mapping):
            # Re-serialized so the same strict checks (finite numbers, depth)
            # apply as to a JSON string.
            try:
                arguments = json.dumps(dict(arguments))
            except (TypeError, ValueError) as error:
                items.append(
                    ToolCallError(str(error), "structured", "", call_id, has_own_id)
                )
                continue
        raw = str(function.get("name"))[:_RAW_LIMIT]
        items.append(
            _request(
                function.get("name"), arguments, call_id, raw, "structured", has_own_id
            )
        )
    return items


def _template_calls(text: str, offset: int) -> tuple[list[Any], str]:
    """``<tool_call>`` blocks in ``text`` and the text outside them."""
    items: list[ToolCallRequest | ToolCallError] = []
    outside: list[str] = []
    cursor = 0
    for match in _BLOCK.finditer(text):
        outside.append(text[cursor : match.start()])
        items.append(_call_from_body(match.group(1), f"call_{offset + len(items)}"))
        cursor = match.end()
    rest = text[cursor:]
    if TOOL_CALL_OPEN in rest or TOOL_CALL_CLOSE in rest:
        marker = rest.find(TOOL_CALL_OPEN)
        marker = rest.find(TOOL_CALL_CLOSE) if marker < 0 else marker
        items.append(
            ToolCallError(
                f"unterminated or unopened {TOOL_CALL_OPEN} block",
                "chat_template",
                rest[marker : marker + _RAW_LIMIT],
                f"call_{offset + len(items)}",
            )
        )
        rest = rest[:marker]
    outside.append(rest)
    return items, "".join(outside).strip()


def _names_a_tool(text: str) -> bool:
    """Whether a whole-reply JSON object is shaped as a tool call.

    It names a tool: it has a ``"tool"`` key (the JSON-action shape), or
    ``"name"`` and ``"arguments"`` keys (a function call written without
    ``<tool_call>`` tags). Text that is not valid strict JSON counts when it
    spells such a key. Any other JSON (``{"answer": "84"}``,
    ``{"final_answer": 84}``) is a final answer, not a call.
    """
    try:
        parsed = strict_json_loads(text)
    except ValueError:
        return bool(
            _TOOL_KEY.search(text)
            or (_NAME_KEY.search(text) and _ARGUMENTS_KEY.search(text))
        )
    return isinstance(parsed, dict) and (
        "tool" in parsed or {"name", "arguments"} <= set(parsed)
    )


def _json_action(content: str, offset: int) -> ToolCallRequest | ToolCallError | None:
    """The whole reply as a ``{"tool": ..., "args": ...}`` action, if it is one.

    Only a JSON object that names a tool (:func:`_names_a_tool`) is an
    attempted call; other JSON is the reply's answer text.
    """
    stripped = content.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    if not _names_a_tool(stripped):
        return None
    call_id = f"call_{offset}"
    try:
        name, args = parse_json_action(
            ConversationTurn(role="assistant", content=stripped)
        )
    except ValueError as error:
        return ToolCallError(
            f"invalid JSON action: {error}",
            "json_action",
            stripped[:_RAW_LIMIT],
            call_id,
        )
    if not name.strip():
        return ToolCallError("missing tool name", "json_action", stripped, call_id)
    return ToolCallRequest(name.strip(), args, call_id, "json_action")


def parse_tool_calls(
    reply: ConversationTurn | str, *, json_actions: bool = True
) -> ToolCallParse:
    """Every tool call in ``reply``, in order, and the text around them.

    Structured ``tool_calls`` on the turn come first, then ``<tool_call>``
    blocks. Only when neither is present, and ``json_actions`` is on, is a
    reply that is wholly a JSON object read as a JSON action (a malformed one
    is still an attempted call). Never raises for model output.
    """
    if isinstance(reply, ConversationTurn):
        content = reply.content if isinstance(reply.content, str) else ""
        items: list[ToolCallRequest | ToolCallError] = _structured_calls(
            reply.tool_calls
        )
    else:
        content, items = str(reply), []
    template_items, text = _template_calls(content, len(items))
    items.extend(template_items)
    if not items and json_actions:
        action = _json_action(content, 0)
        if action is not None:
            return ToolCallParse((action,), "")
    return ToolCallParse(tuple(items), text)


def strip_tool_calls(text: str, *, json_actions: bool = True) -> str:
    """``text`` without its tool calls: what is left to read as an answer."""
    return parse_tool_calls(str(text), json_actions=json_actions).text
