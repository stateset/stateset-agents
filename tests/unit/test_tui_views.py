"""Studio views render retained records without a Studio instance."""

from __future__ import annotations

from stateset_agents.finetune_workspace import FinetuneConfig
from stateset_agents.tui_views import (
    backend_lines,
    comparison_lines,
    metric_lines,
    model_lines,
    progress_lines,
    resource_lines,
    review_lines,
    run_details_lines,
    stop_dialog_lines,
    validation_lines,
)


def _row(tmp_path, **state):
    return {
        "id": "run-1",
        "path": str(tmp_path),
        "plan": {"config": {"method": "sft", "epochs": 4, "provider": "local"}},
        "state": {"status": "running", **state},
    }


def test_views_describe_missing_selections(tmp_path):
    assert run_details_lines(None, tmp_path) == ["No retained run is selected."]
    assert metric_lines(None, 80) == ["No retained run selected."]
    assert comparison_lines(None) == ["No comparison snapshot. Esc returns to Runs."]
    assert review_lines(None, {}) == [
        "No validated launch plan. Esc returns to configuration."
    ]


def test_progress_shows_reported_epochs_against_the_budget(tmp_path):
    history = [{"step": 3, "metrics": {"epoch": 2.0, "loss": 0.5}, "updated_at": None}]
    rows = progress_lines(_row(tmp_path), history, 80)
    assert rows[0].startswith("Reported epoch: 2/4 [") and rows[0].endswith("50%")
    assert rows[-1] == "loss=0.5"


def test_details_name_the_run_and_its_agent_command(tmp_path):
    text = "\n".join(run_details_lines(_row(tmp_path), tmp_path / "workspace"))
    assert "Run: run-1" in text and "finetune status run-1" in text


def test_resources_follow_the_draft_without_a_plan():
    draft = {**FinetuneConfig().__dict__, "provider": "local"}
    assert resource_lines(None, draft)


def test_selection_lists_clamp_their_index_for_the_caller():
    rows, index = model_lines([("a/model", "configured")], 5, "", 60, 20)
    assert index == 0 and any(row.startswith(" > a/model") for row in rows)
    issue = type("Issue", (), {"field": "epochs", "code": "range", "message": "bad"})
    rows, index = validation_lines([issue()], 9, True, 60, 20, {"epochs": "Epochs"})
    assert index == 0 and "Selected: Epochs" in rows
    assert any("Settings changed" in row for row in rows)


def test_stop_dialog_explains_local_and_runpod_stops(tmp_path):
    assert stop_dialog_lines(tmp_path, "d" * 64, remote=False)[0] == (
        "REVIEW LOCAL STOP REQUEST"
    )
    remote = stop_dialog_lines(tmp_path, "d" * 64, remote=True)
    assert remote[0] == "REVIEW RUNPOD STOP REQUEST"
    assert backend_lines(1)[0].startswith("TRAINING BACKENDS")
