"""Regression coverage for complete history, responsive planning and action discovery."""

import json
import threading
import time

import pytest
from rich.cells import cell_len

from stateset_agents import tui
from stateset_agents.finetune_workspace import FinetuneConfig, list_runs, plan_run


@pytest.fixture
def studio(tmp_path):
    dataset = tmp_path / "chat.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "How can I help?"},
                ]
            }
        )
        + "\n"
    )
    return tui.Studio(tmp_path / "workspace", FinetuneConfig(dataset=str(dataset)))


def test_all_retained_runs_remain_searchable_and_active(tmp_path):
    for index in range(205):
        run = tmp_path / f"run-{index:04}"
        run.mkdir()
        (run / "plan.json").write_text(
            json.dumps(
                {
                    "config": {
                        "model": "old-active" if index == 0 else "new-model",
                        "method": "sft",
                        "provider": "local",
                    }
                }
            )
        )
        (run / "state.json").write_text(
            json.dumps(
                {
                    "status": "queued" if index == 0 else "succeeded",
                }
            )
        )
    assert len(list_runs(tmp_path)) == 100  # Existing API default is preserved.
    assert len(list_runs(tmp_path, limit=None)) == 205
    studio = tui.Studio(tmp_path)
    studio.tab = "runs"
    assert len(studio.all_runs) == 205
    studio.run_query = "old-active"
    studio.refresh()
    assert [row["id"] for row in studio.runs] == ["run-0000"]
    studio.run_query, studio.run_filter = "", "active"
    studio.refresh()
    assert [row["id"] for row in studio.runs] == ["run-0000"]
    assert "RUNS 1/205" in "\n".join(studio.lines())


@pytest.mark.parametrize("key", ["p", "r"])
def test_planning_is_nonblocking_single_flight_and_requires_confirmation(
    studio, monkeypatch, settle_preview, key
):
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    calls = []
    plan = plan_run(studio.config())

    def slow_plan(config):
        calls.append(config)
        started.set()
        assert release.wait(5), "Planning blocked the terminal thread"
        finished.set()
        return plan

    monkeypatch.setattr(tui, "plan_run", slow_plan)
    monkeypatch.setattr(tui, "launch_run", lambda *_: pytest.fail("Unexpected launch"))
    try:
        studio.handle(key)
        assert started.wait(2) and not finished.is_set()
        assert studio.overlay == "planning" and studio.plan is None
        assert "PREPARING PREVIEW" in "\n".join(studio.lines(40, 24))
        for command in ("ENTER", "r", "p", "ENTER"):
            studio.handle(command)
        assert len(calls) == 1 and studio.plan is None
        assert studio.handle("QUIT") is False
    finally:
        release.set()
    settle_preview(studio)
    assert studio.plan is plan
    assert studio.overlay == ("review" if key == "r" else None)
    assert not studio.workspace.exists()


@pytest.mark.parametrize("cancel", [True, False])
def test_cancelled_or_stale_preview_cannot_restore_a_launch_review(
    studio, monkeypatch, settle_preview, cancel
):
    release = threading.Event()
    plan = plan_run(studio.config())

    def slow_plan(config):
        assert release.wait(5)
        return plan

    monkeypatch.setattr(tui, "plan_run", slow_plan)
    try:
        studio.handle("r")
        if cancel:
            studio.lines(40, 24)
            studio.handle("0")
            assert studio.overlay is None
            studio.handle("p")
            assert "still finishing" in studio.message
            studio.handle("m")
        else:
            studio.values["epochs"] += 1
    finally:
        release.set()
    settle_preview(studio)
    assert studio.plan is None
    assert studio.overlay == ("models" if cancel else None)
    assert not studio.workspace.exists()


def test_failed_preview_is_recoverable_and_redacts_secrets(
    studio, monkeypatch, settle_preview
):
    monkeypatch.setenv("HF_TOKEN", "test-private-preview-token")

    def fail(config):
        raise RuntimeError("Cannot read test-private-preview-token")

    monkeypatch.setattr(tui, "plan_run", fail)
    studio.handle("r")
    settle_preview(studio)
    assert studio.plan is None and studio.overlay is None
    assert "[REDACTED]" in studio.message and "test-private" not in studio.message
    monkeypatch.setattr(tui, "plan_run", plan_run)
    studio.handle("r")
    settle_preview(studio)
    assert studio.overlay == "review" and studio.plan is not None


@pytest.mark.parametrize("size", [(31, 16), (39, 24), (71, 22), (110, 32), (140, 40)])
@pytest.mark.parametrize("tab", ["configure", "runs"])
def test_actions_are_visible_and_scrollable_at_every_supported_size(studio, size, tab):
    studio.tab = tab
    studio.lines(*size)
    studio.handle(":")
    assert studio.overlay == "actions"
    assert studio.action_menu is not None
    for index in range(len(studio.action_menu.actions)):
        frame = studio.lines(*size)
        assert len(frame) == size[1]
        assert all(cell_len(line) <= size[0] for line in frame)
        assert any(line.startswith(" > ") for line in frame)
        assert f"ACTIONS {index + 1}/" in frame[2 if studio.compact_layout else 3]
        studio.handle("DOWN")
    studio.handle("ESC")
    assert studio.overlay is None and not studio.workspace.exists()


def test_actions_open_existing_review_without_launch(
    studio, monkeypatch, settle_preview
):
    monkeypatch.setattr(tui, "launch_run", lambda *_: pytest.fail("Unexpected launch"))
    studio.handle(":")
    assert studio.action_menu is not None
    while studio.action_menu.actions[studio.action_menu.index].key != "r":
        studio.handle("DOWN")
    studio.handle("ENTER")
    settle_preview(studio)
    assert studio.overlay == "review" and not studio.workspace.exists()
    studio.handle("ESC")
    assert studio.overlay is None


def test_action_menu_does_not_redirect_to_a_different_run(studio):
    studio.all_runs = [{"id": "old", "state": {"status": "prepared"}, "plan": {}}]
    studio.runs = list(studio.all_runs)
    studio.tab = "runs"
    studio.handle(":")
    studio.runs = [{"id": "new", "state": {"status": "prepared"}, "plan": {}}]
    studio.handle("ENTER")
    assert studio.overlay is None and "Selected run changed" in studio.message


def test_history_load_does_not_block_navigation_and_preserves_selection(
    tmp_path, monkeypatch
):
    started, release = threading.Event(), threading.Event()
    rows = [{"id": "old", "state": {"status": "queued"}, "plan": {}}]

    def slow_history(workspace, *, limit):
        assert limit is None
        started.set()
        assert release.wait(5), "History blocked the terminal thread"
        return list(rows)

    monkeypatch.setattr(tui, "list_runs", slow_history)
    studio = tui.Studio(tmp_path, background_history=True)
    try:
        assert started.wait(2)
        studio.handle("TAB")
        assert "Loading retained runs" in "\n".join(studio.lines())
        studio.handle(":")
        assert studio.overlay == "actions"
        studio.handle("ESC")
    finally:
        release.set()

    def settle_history():
        deadline = time.monotonic() + 5
        while studio.history_task.busy and time.monotonic() < deadline:
            studio.refresh(history=False)
            time.sleep(0.005)
        assert not studio.history_task.busy

    settle_history()
    assert studio.history_loaded and studio.runs[0]["id"] == "old"
    rows.insert(0, {"id": "new", "state": {"status": "prepared"}, "plan": {}})
    studio.refresh()
    settle_history()
    assert studio.runs[studio.run_index]["id"] == "old"

    # A newly submitted run may arrive after an older in-flight history snapshot.
    studio.pending_run_id = "submitted"
    studio.refresh(history=False)
    assert studio.pending_run_id == "submitted"
    rows.insert(0, {"id": "submitted", "state": {"status": "prepared"}, "plan": {}})
    studio.refresh()
    settle_history()
    assert studio.runs[studio.run_index]["id"] == "submitted"
    assert studio.pending_run_id is None

    def failed_history(*args, **kwargs):
        raise PermissionError("Unavailable workspace")

    monkeypatch.setattr(tui, "list_runs", failed_history)
    studio.refresh()
    settle_history()
    assert studio.history_error and len(studio.all_runs) == 3
    assert "Cannot refresh retained runs" in "\n".join(studio.lines())
