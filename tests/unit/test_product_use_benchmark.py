"""Tests for the public product-use benchmark scorer."""

from pathlib import Path

from benchmarks.product_use.runner import run, score_submission


def test_score_submission_requires_tools_and_artifacts() -> None:
    task = {
        "id": "demo",
        "rubric": {
            "required_tool_calls": ["ingest_transcripts"],
            "required_artifacts": ["output_dir"],
        },
    }
    result = score_submission(
        task,
        {
            "task_id": "demo",
            "tool_calls": [{"name": "ingest_transcripts"}],
            "artifacts": {"output_dir": "out"},
        },
    )
    assert result["passed"] is True
    assert result["score"] == 1.0


def test_run_reports_missing_tasks() -> None:
    report = run(
        Path("benchmarks/product_use/tasks.public.json"),
        Path("tests/fixtures/product_use_submissions.json"),
    )
    assert report["task_count"] == 5
    assert report["submitted_count"] == 1
    assert report["missing_task_ids"] == [
        "curate-successes",
        "grade-customer-support",
        "preview-training",
        "reject-unsafe-path",
    ]
