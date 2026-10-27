"""Typed tool registry: schemas, argument validation and timed execution.

A :class:`ToolSpec` names a deterministic Python callable, its JSON-schema
parameters and a per-call timeout. :class:`ToolRegistry` validates a parsed
call's arguments against the schema (a JSON Schema subset: ``type`` of each
property, ``required``, ``additionalProperties: false``, ``enum``), runs the
callable and returns a :class:`ToolObservation`, which becomes a ``tool``-role
turn. Every failure (unknown tool, invalid arguments, a raised exception, a
timeout, an unserializable result) is an observation with a typed error, never
an exception in the environment.

The timeout bounds the tool's own run time. An async callable is awaited
and cancelled at the timeout. A synchronous one runs in a thread of its own,
started for that call (never queued behind other calls in a shared
executor), and its timeout is measured from the moment it starts running in
that thread, so concurrent calls and a busy event loop never turn a healthy
call into a ``timeout``. Python cannot preempt a thread holding the GIL in a
long computation, and an abandoned thread keeps running (the interpreter
waits for it at exit). Tools must therefore be deterministic and bounded;
untrusted code belongs in a process sandbox
(:mod:`stateset_agents.rewards.code_verifier`).
"""

from __future__ import annotations

import asyncio
import contextvars
import copy
import functools
import inspect
import json
import math
import re
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
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
#: The smallest observation bound: room for a truncated observation's JSON
#: (name, error type, ``"truncated": true``).
MIN_OBSERVATION_CHARS = 128
#: A truncated observation keeps at most this much of the tool name (a
#: registered name is at most 64 characters; a policy-invented one may not be).
_NAME_LIMIT = 64
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
        """The observation as JSON, never longer than ``max_chars`` characters.

        An oversized observation is truncated and stays valid JSON, marked
        ``"truncated": true``: a result becomes the longest prefix of its JSON
        text that fits, an error keeps its ``type`` and the longest prefix of
        its message that fits, and the tool name is clipped to what fits
        (at most 64 characters). The bound holds after JSON escaping
        (quotes, backslashes, ``\\uXXXX`` for non-ASCII text).
        ``max_chars`` must be at least :data:`MIN_OBSERVATION_CHARS`.
        """
        if isinstance(max_chars, bool) or max_chars < MIN_OBSERVATION_CHARS:
            raise ValueError(
                f"max_chars must be at least {MIN_OBSERVATION_CHARS}, got {max_chars}"
            )
        text = json.dumps(self.payload(), sort_keys=True)
        if len(text) <= max_chars:
            return text
        body = (
            json.dumps(self.result, sort_keys=True)
            if self.ok
            else str(self.error_message or "")
        )
        name = _longest_prefix(
            str(self.name)[:_NAME_LIMIT], lambda n: self._clipped(n, ""), max_chars
        )
        kept = _longest_prefix(body, lambda b: self._clipped(name, b), max_chars)
        clipped = self._clipped(name, kept)
        if len(clipped) > max_chars:  # an unusually long custom error type
            return json.dumps({"truncated": True})
        return clipped

    def _clipped(self, name: str, body: str) -> str:
        """The truncated observation with ``name`` and ``body`` as JSON."""
        payload: dict[str, Any] = {"name": name, "truncated": True}
        if self.ok:
            payload["result"] = body
        else:
            payload["error"] = {"type": self.error_type, "message": body}
        return json.dumps(payload, sort_keys=True)

    def to_message(
        self, max_chars: int = DEFAULT_MAX_OBSERVATION_CHARS
    ) -> dict[str, str]:
        """The ``tool`` chat message: content, the id of the call it answers
        (``tool_call_id``) and the tool ``name`` (when there is one)."""
        message = {
            "role": "tool",
            "content": self.content(max_chars),
            "tool_call_id": self.call_id,
        }
        if self.name:
            message["name"] = self.name
        return message

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
            result = await _invoke(spec, copy.deepcopy(call.arguments))
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


def _longest_prefix(text: str, render: Callable[[str], str], limit: int) -> str:
    """The longest prefix of ``text`` whose rendering is at most ``limit``
    characters (rendering length never shrinks as the prefix grows, and each
    character renders to at least one, so no prefix longer than ``limit``
    fits)."""
    low, high = 0, min(len(text), limit)
    while low < high:
        middle = (low + high + 1) // 2
        if len(render(text[:middle])) <= limit:
            low = middle
        else:
            high = middle - 1
    return text[:low]


async def _invoke(spec: ToolSpec, arguments: dict[str, Any]) -> Any:
    """Run ``spec``'s callable under its timeout (``asyncio.TimeoutError``).

    An async callable is awaited and cancelled at the timeout. A synchronous
    one runs in a thread of its own (:func:`_run_in_own_thread`), its timeout
    bounding its own run time.
    """
    function, timeout = spec.function, float(spec.timeout_seconds)
    if inspect.iscoroutinefunction(function):
        return await asyncio.wait_for(function(**arguments), timeout=timeout)
    result, elapsed = await _run_in_own_thread(spec.name, function, arguments, timeout)
    if inspect.isawaitable(result):
        return await asyncio.wait_for(result, timeout=max(timeout - elapsed, 0.0))
    return result


@dataclass
class _ThreadRun:
    """When one synchronous tool call ran, as its own thread measured it."""

    started: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)
    start: float = 0.0
    elapsed: float = 0.0

    def call(self, function: Callable[..., Any], arguments: dict[str, Any]) -> Any:
        self.start = time.monotonic()
        self.started.set()
        try:
            return function(**arguments)
        finally:
            self.elapsed = time.monotonic() - self.start
            self.finished.set()

    def remaining(self, timeout: float) -> float:
        """Run time left before the timeout; the whole timeout until the call
        has started (waiting to start is not run time)."""
        if not self.started.is_set():
            return timeout
        return self.start + timeout - time.monotonic()


async def _run_in_own_thread(
    name: str, function: Callable[..., Any], arguments: dict[str, Any], timeout: float
) -> tuple[Any, float]:
    """Run a synchronous tool in a new thread; return its result and run time.

    Each call gets its own thread, so no call ever waits behind others (a
    shared executor saturated by concurrent episodes, or by stuck tools,
    used to turn healthy calls into timeouts). The call times out only when
    its own run time, measured in its thread, exceeds ``timeout``; a call
    that finished in time counts however long the event loop took to notice.
    """
    loop = asyncio.get_running_loop()
    run = _ThreadRun()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"tool-{name}")
    try:
        future = loop.run_in_executor(
            executor,
            functools.partial(
                contextvars.copy_context().run, run.call, function, arguments
            ),
        )
    finally:
        executor.shutdown(wait=False)  # the thread exits once the call returns
    try:
        while not future.done() and not run.finished.is_set():
            remaining = run.remaining(timeout)
            if remaining <= 0:
                break
            await asyncio.wait({future}, timeout=remaining)
        if run.finished.is_set() and run.elapsed <= timeout:
            return await future, run.elapsed
        raise asyncio.TimeoutError(f"tool {name!r} ran longer than {timeout}s")
    finally:
        # A timed-out or cancelled call's thread keeps running; its outcome
        # is dropped, also when it arrives after the loop has closed.
        if not future.done():
            future.cancel()


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
