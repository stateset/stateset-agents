"""Run a product-use task as a line-delimited JSON tool session.

The evaluator writes one task event, accepts up to four tool-call lines, and
writes a tool-result event after each call. A ``{"done": true}`` line ends the
session early. The final line is the observed score. Only allowlisted StateSet
MCP functions can run; participant input is data, never executable code.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Any

from .execution import (
    DISCOVERY_TOOLS,
    SUPPORTED_SCHEMA_VERSIONS,
    TOOLS,
    _call_tool,
    _check_generated_result,
    _check_result,
    _prepare_fixture,
    _prepare_generated_fixture,
    _read_array,
)

MAX_CALLS = 4
MAX_INPUT_BYTES = 65536


class InteractiveSession:
    """Execute a bounded sequence of real MCP calls in one task workspace."""

    def __init__(self, task: dict[str, Any], root: Path) -> None:
        """Bind a generated fixture and score checker to a temporary workspace."""
        self.task = task
        self.root = root
        self.trace: list[dict[str, Any]] = []

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        """Run one allowlisted tool and retain its actual result for scoring."""
        if len(self.trace) >= MAX_CALLS:
            raise ValueError("tool-call limit reached")
        if self.task["schema_version"] == "0.2" and name in DISCOVERY_TOOLS:
            result: Any = {"error": "tool is unavailable in this task version"}
        else:
            try:
                with redirect_stdout(io.StringIO()):
                    result = await _call_tool(name, arguments, self.root)
            except (TypeError, ValueError, OSError) as exc:
                result = {"error": str(exc)}
        self.trace.append({"name": name, "arguments": arguments, "result": result})
        return result

    def report(self) -> dict[str, Any]:
        """Score the observed workspace and tool transcript."""
        passed = (
            _check_generated_result(self.task, self.trace, self.root)
            if self.task["schema_version"] == "0.4"
            else _check_result(self.task, self.trace, self.root)
        )
        return {
            "task_id": self.task["id"],
            "schema_version": self.task["schema_version"],
            "evaluation_mode": "interactive",
            "score": int(passed),
            "tool_names": [item["name"] for item in self.trace],
        }


@contextmanager
def open_task(task: dict[str, Any]) -> Iterator[InteractiveSession]:
    """Create and dispose a fresh workspace for one interactive task."""
    with tempfile.TemporaryDirectory(prefix="stateset-product-use-") as directory:
        root = Path(directory)
        if task["schema_version"] == "0.4":
            _prepare_generated_fixture(root, task)
        else:
            _prepare_fixture(root, task["fixture"])
        yield InteractiveSession(task, root)


def _emit(value: dict[str, Any]) -> None:
    """Write one JSON protocol event without buffering."""
    print(json.dumps(value, sort_keys=True), flush=True)


def main() -> int:
    """Serve one task over stdin/stdout using JSON lines."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--tools", type=Path, required=True)
    args = parser.parse_args()
    tasks = _read_array(args.tasks, "tasks")
    matches = [task for task in tasks if task["id"] == args.task_id]
    if len(matches) != 1:
        parser.error("task ID is not present in the task file")
    task = matches[0]
    version = task.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        parser.error("unsupported task schema version")
    catalog = json.loads(args.tools.read_text(encoding="utf-8"))
    if catalog.get("schema_version") != version or not isinstance(
        catalog.get("tools"), list
    ):
        parser.error("tool catalog does not match task schema version")
    names = {entry.get("name") for entry in catalog["tools"] if isinstance(entry, dict)}
    expected = set(TOOLS) - DISCOVERY_TOOLS if version == "0.2" else set(TOOLS)
    if names != expected or len(catalog["tools"]) != len(expected):
        parser.error("tool catalog does not match the allowlisted MCP functions")

    with open_task(task) as session:
        _emit(
            {
                "type": "task",
                "task_id": task["id"],
                "schema_version": version,
                "prompt": task["prompt"],
                "tools": catalog["tools"],
                "max_calls": MAX_CALLS,
            }
        )
        for _ in range(MAX_CALLS):
            raw = sys.stdin.buffer.readline(MAX_INPUT_BYTES + 1)
            if not raw:
                break
            if len(raw) > MAX_INPUT_BYTES:
                _emit({"type": "error", "message": "input line is too large"})
                break
            try:
                decision = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                _emit({"type": "error", "message": "input line must be JSON"})
                break
            if decision == {"done": True}:
                break
            if (
                not isinstance(decision, dict)
                or not isinstance(decision.get("name"), str)
                or not isinstance(decision.get("arguments"), dict)
            ):
                _emit({"type": "error", "message": "expected a tool-call object"})
                break
            result = asyncio.run(session.call(decision["name"], decision["arguments"]))
            _emit(
                {
                    "type": "tool_result",
                    "name": decision["name"],
                    "result": result,
                    "remaining_calls": MAX_CALLS - len(session.trace),
                }
            )
        _emit({"type": "report", **session.report()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
