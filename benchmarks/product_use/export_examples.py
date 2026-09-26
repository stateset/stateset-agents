"""Export verified public task demonstrations as model-agnostic JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .execution import run


def export_examples(
    tasks_path: Path, demonstrations_path: Path, tools_path: Path
) -> list[dict[str, Any]]:
    """Join public prompts and successful traces after replaying every example."""
    report = run(tasks_path, demonstrations_path)
    version = report["schema_version"]
    if report["score"] != 1.0:
        failed = [row["task_id"] for row in report["results"] if row["score"] != 1]
        raise ValueError(f"demonstrations failed execution: {', '.join(failed)}")
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    demonstrations = {
        row["task_id"]: row
        for row in json.loads(demonstrations_path.read_text(encoding="utf-8"))
    }
    catalog = json.loads(tools_path.read_text(encoding="utf-8"))
    if catalog.get("schema_version") != version or not isinstance(
        catalog.get("tools"), list
    ):
        raise ValueError("tool catalog schema version does not match the benchmark")
    return [
        {
            "schema_version": version,
            "task_id": task["id"],
            "interface": task["interface"],
            "prompt": task["prompt"],
            "tools": catalog["tools"],
            "calls": demonstrations[task["id"]]["calls"],
            "verified_score": 1,
        }
        for task in tasks
    ]


def main() -> int:
    """Write verified demonstration rows to a JSONL file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--demonstrations", type=Path, required=True)
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = export_examples(args.tasks, args.demonstrations, args.tools)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
