"""Execute public product-use tasks against the real StateSet MCP tool functions.

Submissions are JSON tool-call traces. This module accepts data only: it never
executes a submitted command or imports a submitted module. Each task receives
a new temporary workspace and only the five listed MCP functions are callable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import tempfile
from pathlib import Path
from typing import Any

from stateset_agents import mcp_server

SCHEMA_VERSION = "0.2"
PATH_ARGUMENTS = frozenset(
    {"input_path", "history_path", "transcripts_dir", "output_dir"}
)
TOOLS = {
    "ingest_transcripts": mcp_server.ingest_transcripts,
    "grade_transcript": mcp_server.grade_transcript,
    "improve_run": mcp_server.improve_run,
    "improve_status": mcp_server.improve_status,
    "dry_run_finetune": mcp_server.dry_run_finetune,
}
ALLOWED_ARGUMENTS = {
    "ingest_transcripts": frozenset({"input_path", "format", "output_dir"}),
    "grade_transcript": frozenset({"history_path", "reward"}),
    "improve_run": frozenset(
        {"transcripts_dir", "reward", "output_dir", "threshold", "format"}
    ),
    "improve_status": frozenset({"output_dir"}),
    "dry_run_finetune": frozenset({"model_preset"}),
}


def _read_array(path: Path, label: str) -> list[dict[str, Any]]:
    """Load a JSON array with unique string task IDs."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or any(
        not isinstance(row, dict) for row in payload
    ):
        raise ValueError(f"{label} must be a JSON array of objects")
    ids = [row.get("task_id" if label == "submissions" else "id") for row in payload]
    if any(not isinstance(item, str) or not item for item in ids) or len(ids) != len(
        set(ids)
    ):
        raise ValueError(f"{label} must have unique nonempty task IDs")
    return payload


def _workspace_path(root: Path, value: Any) -> str:
    """Resolve one relative path without allowing traversal or absolute paths."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("workspace paths must be nonempty relative POSIX paths")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts or "." in candidate.parts:
        raise ValueError("workspace path escapes the task workspace")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("workspace path escapes the task workspace")
    return str(resolved)


def _prepare_fixture(root: Path, fixture: str) -> None:
    """Create synthetic input conversations for one task."""
    if fixture not in {"openai_support", "transcripts_support", "none"}:
        raise ValueError(f"unknown fixture: {fixture}")
    good = [
        {"role": "user", "content": "Please refund order 1234."},
        {
            "role": "assistant",
            "content": "I can help with a refund for order 1234 right away.",
        },
    ]
    bad = [
        {"role": "user", "content": "Please refund order 9999."},
        {"role": "assistant", "content": "idk"},
    ]
    if fixture == "openai_support":
        (root / "logs.jsonl").write_text(
            "".join(json.dumps({"messages": turns}) + "\n" for turns in (good, bad)),
            encoding="utf-8",
        )
    elif fixture == "transcripts_support":
        directory = root / "transcripts"
        directory.mkdir()
        for name, turns in (("good.jsonl", good), ("bad.jsonl", bad)):
            (directory / name).write_text(
                "".join(json.dumps(turn) + "\n" for turn in turns),
                encoding="utf-8",
            )


async def _call_tool(name: str, arguments: dict[str, Any], root: Path) -> Any:
    """Dispatch an allowlisted MCP function with confined path arguments."""
    if name not in TOOLS:
        raise ValueError(f"unknown tool: {name}")
    if not isinstance(arguments, dict) or set(arguments) - ALLOWED_ARGUMENTS[name]:
        raise ValueError(f"invalid arguments for {name}")
    bound = {
        key: _workspace_path(root, value) if key in PATH_ARGUMENTS else value
        for key, value in arguments.items()
    }
    result = TOOLS[name](**bound)
    if asyncio.iscoroutine(result):
        result = await result
    return result


def _check_result(
    task: dict[str, Any], trace: list[dict[str, Any]], root: Path
) -> bool:
    """Check real outputs and files for the requested workflow."""
    goal = task["goal"]
    if not trace or trace[-1]["name"] != goal:
        return False
    actual = trace[-1]["arguments"]
    result = trace[-1]["result"]
    if not isinstance(result, dict) or "error" in result:
        return False
    if goal == "ingest_transcripts":
        paths = sorted((root / "transcripts").glob("conversation_*.jsonl"))
        return (
            actual
            == {
                "input_path": "logs.jsonl",
                "format": "openai",
                "output_dir": "transcripts",
            }
            and result.get("conversation_count") == 2
            and len(paths) == 2
            and all(path.read_text(encoding="utf-8").strip() for path in paths)
        )
    if goal == "grade_transcript":
        score = result.get("mean_score")
        return (
            actual
            == {
                "history_path": "transcripts/good.jsonl",
                "reward": "customer_support",
            }
            and result.get("reward") == "customer_support"
            and result.get("assistant_turn_count") == 1
            and isinstance(score, (float, int))
            and math.isfinite(score)
            and 0 <= score <= 1
        )
    if goal == "improve_run":
        curated = root / "improved" / "curated.jsonl"
        summary = root / "improved" / "improve_summary.json"
        return (
            actual.get("transcripts_dir") == "transcripts"
            and actual.get("reward") == "customer_support"
            and actual.get("output_dir") == "improved"
            and actual.get("threshold") == 0.7
            and isinstance(result.get("curated_count"), int)
            and result["curated_count"] > 0
            and curated.is_file()
            and bool(curated.read_text(encoding="utf-8").strip())
            and summary.is_file()
        )
    if goal == "improve_status":
        return (
            actual == {"output_dir": "improved"}
            and len(trace) >= 2
            and trace[-2]["name"] == "improve_run"
            and trace[-2]["arguments"].get("transcripts_dir") == "transcripts"
            and trace[-2]["arguments"].get("reward") == "customer_support"
            and trace[-2]["arguments"].get("output_dir") == "improved"
            and trace[-2]["arguments"].get("threshold") == 0.7
            and result.get("reward") == "customer_support"
            and isinstance(result.get("curated_count"), int)
            and result["curated_count"] > 0
        )
    if goal == "dry_run_finetune":
        return result.get("model_preset") == task["preset"] and isinstance(
            result.get("config"), dict
        )
    raise ValueError(f"unknown goal: {goal}")


async def score_task(
    task: dict[str, Any], submission: dict[str, Any]
) -> dict[str, Any]:
    """Execute a submitted trace and grade its observed result."""
    calls = submission.get("calls")
    if not isinstance(calls, list) or len(calls) > 4:
        return {
            "task_id": task["id"],
            "score": 0,
            "error": "calls must be a list of at most four",
        }
    with tempfile.TemporaryDirectory(prefix="stateset-product-use-") as directory:
        root = Path(directory)
        _prepare_fixture(root, task["fixture"])
        trace: list[dict[str, Any]] = []
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get("name"), str):
                return {"task_id": task["id"], "score": 0, "error": "invalid tool call"}
            try:
                result = await _call_tool(call["name"], call.get("arguments", {}), root)
            except (TypeError, ValueError, OSError) as exc:
                return {"task_id": task["id"], "score": 0, "error": str(exc)}
            trace.append(
                {
                    "name": call["name"],
                    "arguments": call.get("arguments", {}),
                    "result": result,
                }
            )
            if isinstance(result, dict) and "error" in result:
                break
        passed = _check_result(task, trace, root)
        return {
            "task_id": task["id"],
            "score": 1 if passed else 0,
            "tool_names": [item["name"] for item in trace],
        }


def run(tasks_path: Path, submissions_path: Path) -> dict[str, Any]:
    """Score all version 0.2 tasks; omitted tasks receive zero."""
    tasks = _read_array(tasks_path, "tasks")
    submissions = _read_array(submissions_path, "submissions")
    if any(task.get("schema_version") != SCHEMA_VERSION for task in tasks):
        raise ValueError(f"tasks must use schema_version {SCHEMA_VERSION}")
    task_ids = {task["id"] for task in tasks}
    by_id = {row["task_id"]: row for row in submissions}
    if set(by_id) - task_ids:
        raise ValueError("submissions contain unknown task IDs")
    results = [
        (
            asyncio.run(score_task(task, by_id[task["id"]]))
            if task["id"] in by_id
            else {"task_id": task["id"], "score": 0, "error": "missing submission"}
        )
        for task in tasks
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_count": len(tasks),
        "score": (
            round(sum(item["score"] for item in results) / len(tasks), 4)
            if tasks
            else 0.0
        ),
        "results": results,
    }


def main() -> int:
    """Run the benchmark from JSON files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--submissions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.tasks, args.submissions)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
