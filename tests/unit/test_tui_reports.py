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
