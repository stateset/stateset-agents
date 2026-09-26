"""The public execution score must follow observed StateSet tool behavior."""

import asyncio
import json
from pathlib import Path

from benchmarks.product_use.execution import run, score_task

TASKS = Path("benchmarks/product_use/tasks.v0.2.public.json")
DEMONSTRATIONS = Path("benchmarks/product_use/demonstrations.v0.2.json")


def test_real_tool_demonstrations_pass() -> None:
    report = run(TASKS, DEMONSTRATIONS)
    assert report["score"] == 1.0
    assert report["task_count"] == 5


def test_forged_artifact_does_not_pass() -> None:
    task = json.loads(TASKS.read_text(encoding="utf-8"))[0]
    result = asyncio.run(
        score_task(
            task,
            {
                "task_id": task["id"],
                "calls": [],
                "artifacts": {"output_dir": "transcripts"},
            },
        )
    )
    assert result["score"] == 0


def test_wrong_transcript_does_not_pass() -> None:
    task = json.loads(TASKS.read_text(encoding="utf-8"))[1]
    result = asyncio.run(
        score_task(
            task,
            {
                "task_id": task["id"],
                "calls": [
                    {
                        "name": "grade_transcript",
                        "arguments": {
                            "history_path": "transcripts/bad.jsonl",
                            "reward": "customer_support",
                        },
                    }
                ],
            },
        )
    )
    assert result["score"] == 0


def test_path_traversal_is_rejected() -> None:
    task = json.loads(TASKS.read_text(encoding="utf-8"))[0]
    result = asyncio.run(
        score_task(
            task,
            {
                "task_id": task["id"],
                "calls": [
                    {
                        "name": "ingest_transcripts",
                        "arguments": {
                            "input_path": "logs.jsonl",
                            "format": "openai",
                            "output_dir": "../escape",
                        },
                    }
                ],
            },
        )
    )
    assert result["score"] == 0
    assert "escapes" in result["error"]


def test_missing_tasks_reduce_aggregate(tmp_path: Path) -> None:
    submissions = tmp_path / "one.json"
    submissions.write_text(
        json.dumps(json.loads(DEMONSTRATIONS.read_text(encoding="utf-8"))[:1]),
        encoding="utf-8",
    )
    report = run(TASKS, submissions)
    assert report["score"] == 0.2
    assert sum(row["score"] for row in report["results"]) == 1
