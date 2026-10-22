"""Background report panels own their task, report, inputs and rendering."""

from __future__ import annotations

import threading
import time

from stateset_agents.tui_reports import (
    BackgroundReport,
    DataReport,
    QualityReport,
    ReadinessReport,
    WorkflowReport,
)


def settle(panel, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not panel.poll() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert not panel.busy


def test_report_runs_off_thread_single_flight_and_keeps_its_inputs():
    panel = BackgroundReport()
    release, calls = threading.Event(), []

    def produce():
        calls.append(1)
        assert release.wait(5)
        return {"value": 1}

    assert panel.start("inputs-a", produce)
    assert panel.busy and not panel.start("inputs-b", produce)
    release.set()
    settle(panel)
    assert calls == [1]
    assert (panel.signature, panel.report) == ("inputs-a", {"value": 1})


def test_failed_operation_keeps_the_previous_report():
    panel = BackgroundReport()
    panel.set("old", {"kept": True})

    def explode():
        raise RuntimeError("synthetic")

    panel.start("new", explode)
    settle(panel)
    assert (panel.signature, panel.report) == ("old", {"kept": True})


def test_readiness_marks_reports_taken_for_other_settings():
    panel = ReadinessReport()
    values = {"model": "m", "provider": "local"}
    assert panel.lines("now", values)[-1] == "Press i to inspect."
    panel.set("then", {"failed": 0, "warnings": 0, "checks": []})
    assert any("Settings changed" in line for line in panel.lines("now", values))
    assert not any("Settings changed" in line for line in panel.lines("then", values))


def test_workflow_selects_only_current_actions():
    panel = WorkflowReport()
    steps = [
        {"name": f"s{i}", "status": "ready", "detail": f"d{i}", "action": f"a{i}"}
        for i in range(5)
    ]
    panel.set("now", {"steps": steps, "notes": []})
    panel.move(-1)
    assert panel.index == 4 and panel.action("now") == "a4"
    assert panel.action("stale") is None
    panel.set("now", {"error": "failed"})
    assert panel.action("now") is None


def test_data_samples_toggle_only_for_a_usable_report():
    panel = DataReport()
    assert not panel.toggle_samples()
    panel.set("now", {"error": "bad"})
    assert not panel.toggle_samples() and not panel.step_sample(1)
    panel.set("now", {"samples": [{}, {}, {}]})
    assert panel.toggle_samples() and panel.show_samples
    assert panel.step_sample(-1) and panel.sample_index == 2


def test_quality_names_its_captured_run(tmp_path):
    panel = QualityReport()
    assert panel.lines()[1] == "Run: none"
    panel.run = tmp_path / "run-1"
    assert panel.lines()[1] == "Run: run-1"


def test_recovery_panel_selects_only_available_checkpoints(tmp_path):
    from stateset_agents.tui_reports import RecoveryPanel

    panel = RecoveryPanel()
    assert not panel.select()
    panel.report = {
        "checkpoints": [
            {"checkpoint": "checkpoint-1", "available": False, "reason": "gone"},
            {"checkpoint": "checkpoint-2", "available": True},
        ]
    }
    assert not panel.select()
    panel.move(5)
    assert panel.index == 1 and panel.select()
    assert panel.choice == {"checkpoint": "checkpoint-2", "available": True}
    panel.run = tmp_path / "parent"
    assert "Parent: parent" in panel.review_lines()


def test_recovery_authorized_steps_outlive_the_terminal(monkeypatch):
    from stateset_agents import tui_tasks
    from stateset_agents.tui_reports import RecoveryPanel

    started = []

    class Recorder:
        def __init__(self, *, target, daemon, name):
            started.append((daemon, name))

        def start(self):
            pass

    monkeypatch.setattr(tui_tasks.threading, "Thread", Recorder)
    for step, authorized in (("inspection", False), ("launch", True)):
        panel = RecoveryPanel()
        assert panel.start(dict, step=step, authorized=authorized)
    assert started == [
        (True, "stateset-recovery-inspection"),
        (False, "stateset-recovery-launch"),
    ]
