"""Interactive product-use sessions must expose real tool feedback and scores."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

from benchmarks.product_use.generate import generate
from benchmarks.product_use.interactive import open_task


def test_discovery_result_can_drive_next_tool_call() -> None:
    task = generate("interactive-test-seed")[0][-1]
    with open_task(task) as session:
        listed = asyncio.run(session.call("list_model_presets", {}))
        names = {row["name"] for row in listed["presets"]}
        assert task["params"]["preset"] in names
        preview = asyncio.run(
            session.call("dry_run_finetune", {"model_preset": task["params"]["preset"]})
        )
        assert preview["model_preset"] == task["params"]["preset"]
        assert session.report()["score"] == 1


def test_interactive_session_can_recover_from_rejected_path() -> None:
    task = generate("interactive-test-seed")[0][1]
    with open_task(task) as session:
        rejected = asyncio.run(
            session.call(
                "grade_transcript",
                {"history_path": "../escape.jsonl", "reward": "customer_support"},
            )
        )
        assert "error" in rejected
        accepted = asyncio.run(
            session.call(
                "grade_transcript",
                {
                    "history_path": task["params"]["history_path"],
                    "reward": "customer_support",
                },
            )
        )
        assert accepted["reward"] == "customer_support"
        assert session.report()["score"] == 1


def test_jsonl_cli_returns_tool_feedback_and_observed_score() -> None:
    call = {"name": "dry_run_finetune", "arguments": {"model_preset": "qwen3.5-0.8b"}}
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.product_use.interactive",
            "--tasks",
            "benchmarks/product_use/tasks.v0.3.public.json",
            "--task-id",
            "preview-qwen-training",
            "--tools",
            "benchmarks/product_use/tools.v0.3.json",
        ],
        input=json.dumps(call) + "\n" + json.dumps({"done": True}) + "\n",
        text=True,
        capture_output=True,
        check=True,
        timeout=40,
        cwd=Path.cwd(),
    )
    events = [json.loads(line) for line in completed.stdout.splitlines()]
    assert [event["type"] for event in events] == ["task", "tool_result", "report"]
    assert events[0]["task_id"] == "preview-qwen-training"
    assert {tool["name"] for tool in events[0]["tools"]} >= {"dry_run_finetune"}
    assert events[1]["result"]["model_preset"] == "qwen3.5-0.8b"
    assert events[2]["score"] == 1
    assert events[2]["evaluation_mode"] == "interactive"
