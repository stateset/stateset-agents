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
    assert report["mean_score"] == 0.2
    assert report["missing_task_ids"] == [
        "curate-successes",
        "grade-customer-support",
        "preview-training",
        "reject-unsafe-path",
    ]


def test_run_rejects_unknown_task_ids(tmp_path: Path) -> None:
    submissions = tmp_path / "submissions.json"
    submissions.write_text('[{"task_id": "unknown"}]', encoding="utf-8")
    try:
        run(Path("benchmarks/product_use/tasks.public.json"), submissions)
    except ValueError as exc:
        assert "unknown task ids" in str(exc)
    else:
        raise AssertionError("unknown task id was accepted")
