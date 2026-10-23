"""Typed tool registry: schemas, argument validation and timed execution.

A :class:`ToolSpec` names a deterministic Python callable, its JSON-schema
parameters and a per-call timeout. :class:`ToolRegistry` validates a parsed
call's arguments against the schema (a JSON Schema subset: ``type`` of each
property, ``required``, ``additionalProperties: false``, ``enum``), runs the
callable and returns a :class:`ToolObservation`, which becomes a ``tool``-role
turn. Every failure (unknown tool, invalid arguments, a raised exception, a
timeout, an unserializable result) is an observation with a typed error, never
an exception in the environment.

The timeout bounds how long the environment *waits*: a synchronous callable
runs in a worker thread and an async one is cancelled, but Python cannot
preempt a thread holding the GIL in a long computation, and an abandoned
thread keeps running. Tools must therefore be deterministic and bounded;
untrusted code belongs in a process sandbox
(:mod:`stateset_agents.rewards.code_verifier`).
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import math
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from .errors import ConfigurationError
from .function_calling import FunctionDefinition
from .math_expressions import (
    ExpressionError,
    evaluate_expression,
    normalize_answer_text,
    parse_expression,
)
from .tool_calls import ToolCallError, ToolCallRequest
from .trajectory import ConversationTurn

__all__ = [
    "ToolObservation",
    "ToolRegistry",
    "ToolSpec",
    "calculator_tool",
    "validate_arguments",
]

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")
#: Observation text beyond this many characters is truncated.
DEFAULT_MAX_OBSERVATION_CHARS = 2000
_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


@dataclass(frozen=True)
class ToolSpec:
    """A tool the policy may call.

    ``function`` receives the validated arguments as keyword arguments and
    must be deterministic (the same arguments give the same result) and
    return JSON-serializable data. ``timeout_seconds`` bounds each call.
    """

    name: str
    parameters: Mapping[str, Any]
    function: Callable[..., Any]
    description: str = ""
    timeout_seconds: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME.match(self.name):
            raise ConfigurationError(f"invalid tool name {self.name!r}")
        if (
            not isinstance(self.parameters, Mapping)
            or self.parameters.get("type", "object") != "object"
        ):
            raise ConfigurationError(
                f"tool {self.name!r} parameters must be a JSON schema of type object"
            )
        if not callable(self.function):
            raise ConfigurationError(f"tool {self.name!r} function is not callable")
        timeout = self.timeout_seconds
        if isinstance(timeout, bool) or not (
            isinstance(timeout, (int, float)) and math.isfinite(timeout) and timeout > 0
        ):
            raise ConfigurationError(
                f"tool {self.name!r} timeout_seconds must be a positive number"
            )

    def schema(self) -> dict[str, Any]:
        """The OpenAI-format tool schema (pass to ``apply_chat_template(tools=...)``)."""
        return FunctionDefinition(
            name=self.name,
            description=self.description,
            parameters=dict(self.parameters),
        ).to_openai_format()


@dataclass(frozen=True)
class ToolObservation:
    """The environment's answer to one tool call, as a ``tool`` turn."""

    name: str
    call_id: str
    ok: bool
    result: Any = None
    error_type: str | None = None
    error_message: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def failure(
        cls,
        name: str,
        call_id: str,
        error_type: str,
        message: str,
        arguments: dict[str, Any] | None = None,
    ) -> ToolObservation:
        return cls(
            name=name,
            call_id=call_id,
            ok=False,
            error_type=error_type,
            error_message=message,
            arguments=dict(arguments or {}),
        )

    @classmethod
    def malformed(cls, error: ToolCallError) -> ToolObservation:
        return cls.failure("", error.call_id, "malformed_call", error.message)

    def payload(self) -> dict[str, Any]:
        if self.ok:
            return {"name": self.name, "result": self.result}
        return {
            "name": self.name,
            "error": {"type": self.error_type, "message": self.error_message},
        }

    def content(self, max_chars: int = DEFAULT_MAX_OBSERVATION_CHARS) -> str:
        """The observation as JSON, with an oversized result truncated."""
        text = json.dumps(self.payload(), sort_keys=True)
        if len(text) <= max_chars:
            return text
        clipped = {**self.payload(), "truncated": True}
        key = "result" if self.ok else "error"
        clipped[key] = text[: max(0, max_chars - 64)]
        return json.dumps(clipped, sort_keys=True)

    def to_message(
        self, max_chars: int = DEFAULT_MAX_OBSERVATION_CHARS
    ) -> dict[str, str]:
        return {"role": "tool", "content": self.content(max_chars)}

    def to_turn(
        self, max_chars: int = DEFAULT_MAX_OBSERVATION_CHARS
    ) -> ConversationTurn:
        return ConversationTurn(
            role="tool",
            content=self.content(max_chars),
            metadata={"tool_call_id": self.call_id, "name": self.name, "ok": self.ok},
        )

    def to_log(self) -> dict[str, Any]:
        """The record kept in the environment state (executed evidence)."""
        return {
            "id": self.call_id,
            "name": self.name,
            "arguments": copy.deepcopy(self.arguments),
            "ok": self.ok,
            "result": copy.deepcopy(self.result),
            "error_type": self.error_type,
        }


def _type_matches(value: Any, expected: Any) -> bool:
    names = expected if isinstance(expected, list) else [expected]
    for name in names:
        types = _JSON_TYPES.get(str(name))
        if types is None:
            return True  # an unsupported keyword is not checked
        if isinstance(value, bool) and name in ("integer", "number"):
            continue  # JSON booleans are not numbers
        if isinstance(value, float) and name == "integer":
            continue
        if isinstance(value, types):
            return True
    return False


def validate_arguments(
    schema: Mapping[str, Any], arguments: Mapping[str, Any]
) -> str | None:
    """The first way ``arguments`` violate ``schema``, or ``None``.

    Checks the JSON Schema subset tools use: required properties,
    ``additionalProperties: false``, and each property's ``type`` and
    ``enum``. Nested schemas are not checked beyond their top-level type.
    """
    properties = schema.get("properties") or {}
    if not isinstance(properties, Mapping):
        properties = {}
    for name in schema.get("required") or []:
        if name not in arguments:
            return f"missing required argument {name!r}"
    if schema.get("additionalProperties", True) is False:
        unknown = sorted(set(arguments) - set(properties))
        if unknown:
            return f"unexpected arguments {unknown}"
    for name, value in arguments.items():
        rule = properties.get(name)
        if not isinstance(rule, Mapping):
            continue
        if "type" in rule and not _type_matches(value, rule["type"]):
            return f"argument {name!r} must be of type {rule['type']}"
        if "enum" in rule and value not in rule["enum"]:
            return f"argument {name!r} must be one of {rule['enum']}"
    return None


class ToolRegistry:
    """The tools an environment exposes, keyed by name."""

    def __init__(self, tools: Iterable[ToolSpec] = ()) -> None:
        self._tools: dict[str, ToolSpec] = {}
        for spec in tools:
            self.register(spec)

    def register(self, spec: ToolSpec) -> ToolSpec:
        if not isinstance(spec, ToolSpec):
            raise ConfigurationError("register() takes a ToolSpec")
        if spec.name in self._tools:
            raise ConfigurationError(f"tool {spec.name!r} is already registered")
        self._tools[spec.name] = spec
        return spec

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[ToolSpec]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """OpenAI-format schemas of every tool, for chat templates."""
        return [spec.schema() for spec in self._tools.values()]

    def describe(self) -> str:
        """One line per tool: name, description and parameter schema."""
        return "\n".join(
            f"- {spec.name}: {spec.description} Parameters: "
            f"{json.dumps(dict(spec.parameters), sort_keys=True)}"
            for spec in self._tools.values()
        )

    async def execute(self, call: ToolCallRequest) -> ToolObservation:
        """Validate and run ``call``; every failure is an error observation."""
        spec = self._tools.get(call.name)
        if spec is None:
            return ToolObservation.failure(
                call.name,
                call.call_id,
                "unknown_tool",
                f"unknown tool {call.name!r}; available: {self.names}",
                call.arguments,
            )
        problem = validate_arguments(spec.parameters, call.arguments)
        if problem is not None:
            return ToolObservation.failure(
                call.name, call.call_id, "invalid_arguments", problem, call.arguments
            )
        try:
            result = await asyncio.wait_for(
                _invoke(spec.function, copy.deepcopy(call.arguments)),
                timeout=spec.timeout_seconds,
            )
        except asyncio.TimeoutError:
            return ToolObservation.failure(
                call.name,
                call.call_id,
                "timeout",
                f"tool did not finish within {spec.timeout_seconds}s",
                call.arguments,
            )
        except Exception as error:  # noqa: BLE001 - a tool failure is an observation
            return ToolObservation.failure(
                call.name,
                call.call_id,
                "tool_error",
                f"{type(error).__name__}: {error}"[:500],
                call.arguments,
            )
        try:
            json.dumps(result, allow_nan=False)
        except (TypeError, ValueError) as error:
            return ToolObservation.failure(
                call.name,
                call.call_id,
                "unserializable_result",
                str(error)[:500],
                call.arguments,
            )
        return ToolObservation(
            call.name, call.call_id, True, result, arguments=dict(call.arguments)
        )


async def _invoke(function: Callable[..., Any], arguments: dict[str, Any]) -> Any:
    if inspect.iscoroutinefunction(function):
        return await function(**arguments)
    result = await asyncio.to_thread(function, **arguments)
    if inspect.isawaitable(result):
        return await result
    return result


def _calculate(expression: str) -> int | str:
    """Exact value of an arithmetic expression: an int, or ``"p/q"``."""
    try:
        value: Fraction = evaluate_expression(
            parse_expression(normalize_answer_text(str(expression)))
        )
    except ExpressionError as error:
        raise ValueError(str(error)) from error
    return value.numerator if value.denominator == 1 else str(value)


def calculator_tool(
    *, name: str = "calculator", timeout_seconds: float = 1.0
) -> ToolSpec:
    """A deterministic exact calculator over ``+ - * / ^`` and parentheses.

    Evaluates with :mod:`fractions` (no floating-point rounding, no ``eval``):
    ``"12*7"`` gives ``84`` and ``"1/3 + 1/6"`` gives ``"1/2"``. Symbols,
    functions and oversized powers are tool errors.
    """
    return ToolSpec(
        name=name,
        description="Evaluate an arithmetic expression exactly.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic using numbers, + - * / ^ and ().",
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
        function=_calculate,
        timeout_seconds=timeout_seconds,
    )
