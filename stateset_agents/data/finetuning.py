"""Strict, dependency-free preparation of text and tool-calling SFT datasets."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

__all__ = [
    "load_finetuning_data",
    "validate_finetuning_row",
    "split_finetuning_data",
    "check_finetuning_overlap",
    "group_finetuning_data",
]


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def validate_finetuning_row(row: Any) -> dict[str, Any]:
    """Validate a text chat record without dropping metadata or typed arguments.

    Checks roles, tool declarations, argument objects, and tool-result IDs.
    Arbitrary JSON Schema constraints and model-specific templates are not
    evaluated here. A terminal assistant tool call is a valid training target.
    """
    if not isinstance(row, dict):
        raise ValueError("Expected a JSON object")
    _json(row)  # Reject NaN/Infinity even for callers supplying Python objects.
    if "synthetic" in row and type(row["synthetic"]) is not bool:
        raise ValueError("synthetic must be a boolean when supplied")
    if "group_id" in row and (
        not isinstance(row["group_id"], str) or not row["group_id"].strip()
    ):
        raise ValueError("group_id must be a nonempty string")
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    tools = row.get("tools", [])
    if not isinstance(tools, list):
        raise ValueError("tools must be a list")
    names: set[str] = set()
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict) or tool.get("type") != "function":
            raise ValueError(
                "Each tool must declare type=function and a function object"
            )
        name = function.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("Tool names must be nonempty and unique")
        if not isinstance(function.get("parameters", {}), dict):
            raise ValueError("Tool parameters must be a JSON Schema object")
        names.add(name)

    pending: set[str] = set()
    seen_calls: set[str] = set()
    has_user = False
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"message {index} must be an object")
        role = message.get("role")
        if not isinstance(role, str) or role not in {
            "system",
            "developer",
            "user",
            "assistant",
            "tool",
        }:
            raise ValueError(f"message {index} has an unsupported role")
        calls = message.get("tool_calls", [])
        if not isinstance(calls, list) or (calls and role != "assistant"):
            raise ValueError(f"message {index}: tool_calls must be an assistant list")
        content = message.get("content")
        if content is None and role == "assistant" and calls:
            pass
        elif not isinstance(content, str) or (not content.strip() and not calls):
            raise ValueError(f"message {index}: nonempty text content is required")
        if pending and role != "tool":
            raise ValueError(
                f"message {index}: missing results for preceding tool calls"
            )
        if role in {"system", "developer"} and has_user:
            raise ValueError("System/developer messages must precede the conversation")
        if role == "user":
            has_user = True
        if role == "assistant" and not has_user:
            raise ValueError("An assistant target must follow a user message")
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in pending:
                raise ValueError(f"message {index}: unknown or repeated tool_call_id")
            pending.remove(call_id)
        for call in calls:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict) or call.get("type") != "function":
                raise ValueError("Tool calls must contain type=function and a function")
            if (
                not isinstance(function.get("name"), str)
                or function["name"] not in names
            ):
                raise ValueError("Tool call references an undeclared function")
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments, parse_constant=_reject_constant)
                except ValueError as exc:
                    raise ValueError("Tool arguments must contain valid JSON") from exc
            if not isinstance(arguments, dict):
                raise ValueError(
                    "Tool arguments must be an object or JSON object string"
                )
            call_id = call.get("id")
            if "id" not in call and index == len(messages) - 1:
                # Native tool-only templates may omit IDs on terminal targets.
                continue
            if not isinstance(call_id, str) or not call_id or call_id in seen_calls:
                raise ValueError(
                    "Tool call IDs must be nonempty and unique per conversation"
                )
            seen_calls.add(call_id)
            pending.add(call_id)
    if not has_user or messages[-1]["role"] != "assistant":
        raise ValueError("A training conversation must end with an assistant target")
    return row


def load_finetuning_data(path: str | Path) -> list[dict[str, Any]]:
    """Read UTF-8 JSONL, failing with a line number on any malformed record."""
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=_reject_constant)
                rows.append(validate_finetuning_row(row))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}: line {line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"{path}: dataset is empty")
    return rows


def _identity_keys(row: dict[str, Any]) -> set[str]:
    # Conservatively keep repeated initial requests together even when their
    # answers, policies, or tool schemas differ. group_id also links a customer,
    # case, or source conversation across differently worded examples.
    prompt = next(m["content"] for m in row["messages"] if m["role"] == "user")
    keys = {"prompt:" + " ".join(prompt.split()).casefold()}
    if "group_id" in row:
        keys.add("group:" + row["group_id"])
    return keys


def check_finetuning_overlap(
    train: list[dict[str, Any]], validation: list[dict[str, Any]]
) -> None:
    """Reject shared initial requests or explicit source groups across splits."""
    train_keys = set().union(*(_identity_keys(row) for row in train))
    validation_keys = set().union(*(_identity_keys(row) for row in validation))
    if train_keys & validation_keys:
        raise ValueError("Train/validation overlap: shared initial request or group_id")


def group_finetuning_data(
    rows: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Deduplicate and group transitively linked source IDs and initial prompts.

    Preparation and evaluation share this boundary so repeated turns or related
    records cannot inflate the number of independent evaluation groups.
    """
    unique = {_json(validate_finetuning_row(row)): row for row in rows}
    ordered = [unique[key] for key in sorted(unique)]
    parents = list(range(len(ordered)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owners: dict[str, int] = {}
    for index, row in enumerate(ordered):
        for key in sorted(_identity_keys(row)):
            if key in owners:
                parents[root(index)] = root(owners[key])
            owners[key] = index
    groups: dict[int, list[dict[str, Any]]] = {}
    for index, row in enumerate(ordered):
        groups.setdefault(root(index), []).append(row)
    return list(groups.values())


def split_finetuning_data(
    rows: list[dict[str, Any]], validation_fraction: float = 0.2, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplicate and deterministically split connected prompt/source groups.

    The fraction applies to groups, so the row fraction may differ. Exact
    duplicate records are removed. Transitive links through group_id and initial
    requests stay in one split. Input order does not affect the resulting files.
    """
    if not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    groups = group_finetuning_data(rows)
    if len(groups) < 2:
        raise ValueError("At least two independent prompt/source groups are required")
    ranked = sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{seed}:{_json(group)}".encode()).hexdigest(),
    )
    count = max(1, min(len(ranked) - 1, round(len(ranked) * validation_fraction)))
    validation = [row for group in ranked[:count] for row in group]
    train = [row for group in ranked[count:] for row in group]
    check_finetuning_overlap(train, validation)
    return train, validation
