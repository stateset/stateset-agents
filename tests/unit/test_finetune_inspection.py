"""Read-only studio checks and truthful progress without GPUs or provider calls."""

import json
import os
import subprocess
import threading
import time
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
from rich.cells import cell_len

from stateset_agents import finetune_inspection as inspection
from stateset_agents import tui
from stateset_agents.finetune_worker import _metrics
from stateset_agents.finetune_workspace import FinetuneConfig, read_json, write_json


@pytest.fixture
def local(tmp_path, monkeypatch):
    dataset = tmp_path / "data.jsonl"
    # Readiness inspects metadata, not contents or training compatibility.
    dataset.write_text("contents deliberately unvalidated")
    monkeypatch.setattr(
        inspection.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 2**30)
    )
    monkeypatch.setattr(inspection.shutil, "which", lambda _: None)
    monkeypatch.setattr(inspection.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(inspection, "queue_executable", lambda: None)
    return FinetuneConfig(dataset=str(dataset))


def checks(report):
    return {row["name"]: row for row in report["checks"]}


def test_local_readiness_does_not_read_dataset_or_import_training(
    local, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        inspection.subprocess,
        "run",
        lambda *_a, **_k: pytest.fail("Unexpected process"),
    )
    report = inspection.inspect_readiness(local, tmp_path / "missing" / "workspace")
    rows = checks(report)
    assert rows["Dataset"]["status"] == "pass"
    assert "validate its contents" in rows["Dataset"]["detail"]
    assert rows["NVIDIA hardware"]["status"] == "warning"
    assert rows["Shared queue"]["status"] == "warning"
    assert not report["provisions_hardware"] and report["failed"] == 0
    assert not (tmp_path / "missing").exists()


def test_missing_local_dependencies_and_dataset_are_failures(
    local, tmp_path, monkeypatch
):
    monkeypatch.setattr(inspection.importlib.util, "find_spec", lambda _: None)
    report = inspection.inspect_readiness(replace(local, dataset=""), tmp_path)
    assert checks(report)["Dataset"]["status"] == "fail"
    assert checks(report)["Training packages"]["status"] == "fail"
    assert "peft" in checks(report)["Training packages"]["detail"]


@pytest.mark.parametrize("present", [False, True])
def test_remote_check_never_authenticates_or_retains_key(
    local, tmp_path, monkeypatch, present
):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    if present:
        monkeypatch.setenv("RUNPOD_API_KEY", "test-secret-not-a-real-key")
    monkeypatch.setattr(
        inspection.subprocess,
        "run",
        lambda *_a, **_k: pytest.fail("Provider/local subprocess forbidden"),
    )
    report = inspection.inspect_readiness(replace(local, provider="runpod"), tmp_path)
    assert checks(report)["RunPod credential"]["status"] == (
        "pass" if present else "fail"
    )
    assert "No provider request" in checks(report)["Provider readiness"]["detail"]
    assert "test-secret-not-a-real-key" not in json.dumps(report)
    assert "Training packages" not in checks(report)


@pytest.mark.parametrize("oversized", [False, True])
def test_readiness_uses_bounded_readonly_gpu_and_queue_commands(
    local, tmp_path, monkeypatch, oversized
):
    monkeypatch.setattr(inspection.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(inspection, "queue_executable", lambda: "/queue/buildr")
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        assert kwargs == {
            "capture_output": True,
            "text": True,
            "timeout": 3,
            "check": False,
        }
        if argv[0].endswith("nvidia-smi"):
            return SimpleNamespace(returncode=0, stdout="Example GPU, 8192, 4096\n")
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "config": {"cpu_slots": 6, "memory_gb": 32},
                    "counts": {"running": 1, "queued": 2},
                }
            ),
        )

    monkeypatch.setattr(inspection.subprocess, "run", run)
    config = replace(local, cpu_slots=7 if oversized else 2, memory_gb=8)
    rows = checks(inspection.inspect_readiness(config, tmp_path))
    assert calls[1] == ["/queue/buildr", "status", "--summary", "--json"]
    assert rows["Shared queue"]["status"] == ("fail" if oversized else "pass")
    assert "Snapshot only" in rows["Shared queue"]["detail"]
    assert (
        "compatibility and model fit are unchecked" in rows["NVIDIA hardware"]["detail"]
    )


@pytest.mark.parametrize(
    "error",
    [
        OSError("private detail"),
        subprocess.TimeoutExpired("query", 3),
        ValueError("bad response"),
    ],
)
def test_queue_inspection_failure_is_advisory(local, tmp_path, monkeypatch, error):
    monkeypatch.setattr(inspection, "queue_executable", lambda: "/queue/buildr")

    def run(*_a, **_k):
        raise error

    monkeypatch.setattr(inspection.subprocess, "run", run)
    row = checks(inspection.inspect_readiness(local, tmp_path))["Shared queue"]
    assert row["status"] == "warning"
    assert "private detail" not in row["detail"]


def test_metric_events_preserve_history_latest_and_private_permissions(tmp_path):
    _metrics(
        tmp_path,
        2,
        {
            "loss": 1.5,
            "epoch": 0.25,
            "flag": True,
            "invalid": float("nan"),
            "text": "secret",
        },
    )
    _metrics(tmp_path, 4, {"loss": 1.0, "epoch": 0.5})
    history = inspection.metric_history(tmp_path)
    assert [row["step"] for row in history] == [2, 4]
    assert history[0]["metrics"] == {"loss": 1.5, "epoch": 0.25}
    assert history[-1] == read_json(tmp_path / "metrics.json")
    if os.name != "nt":
        assert (tmp_path / "metrics.jsonl").stat().st_mode & 0o077 == 0
    assert inspection.metric_history(tmp_path, limit=1) == history[-1:]


def test_readiness_surfaces_queue_disk_blocking(local, tmp_path, monkeypatch):
    monkeypatch.setattr(inspection, "queue_executable", lambda: "/queue/buildr")
    snapshot = {
        "config": {"cpu_slots": 6, "memory_gb": 16},
        "counts": {"running": 1, "queued": 4},
        "queue_health": {
            "status": "disk_blocked",
            "summary": "10.5 GiB free / 11.2 required",
        },
    }
    monkeypatch.setattr(
        inspection.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(returncode=0, stdout=json.dumps(snapshot)),
    )
    report = inspection.inspect_readiness(local, tmp_path)
    row = checks(report)["Shared queue"]
    assert row["status"] == "warning"
    assert "10.5 GiB free / 11.2 required" in row["detail"]


def test_readiness_rejects_oversized_data_without_reading_it(local, tmp_path):
    with open(local.dataset, "r+b") as stream:
        stream.truncate(inspection.MAX_DATA_BYTES + 1)
    assert (
        checks(inspection.inspect_readiness(local, tmp_path))["Dataset"]["status"]
        == "fail"
    )


def test_history_partial_corrupt_and_stale_latest_are_tolerated(tmp_path):
    event = {
        "step": 5,
        "metrics": {"loss": 2, "invalid": float("inf"), "flag": True, "huge": 10**400},
        "updated_at": 10,
    }
    (tmp_path / "metrics.jsonl").write_text(
        "invalid\n[]\n" + json.dumps(event) + '\n{"step":'
    )
    write_json(tmp_path / "metrics.json", {"step": 4, "metrics": {"loss": 3}})
    assert inspection.metric_history(tmp_path) == [
        {"step": 5, "metrics": {"loss": 2.0}, "updated_at": 10}
    ]


def test_history_bounds_tail_and_falls_back_to_legacy_snapshot(tmp_path):
    event = {"step": 10, "metrics": {"reward": 0.5}}
    write_json(tmp_path / "metrics.json", event)
    assert inspection.metric_history(tmp_path)[0]["step"] == 10
    (tmp_path / "metrics.jsonl").write_text(
        "x" * (256 * 1024) + "\n" + json.dumps({"step": 11, "metrics": {}})
    )
    assert inspection.metric_history(tmp_path)[0]["step"] == 11
    (tmp_path / "metrics.jsonl").write_text(json.dumps({"step": True, "metrics": {}}))
    assert inspection.metric_history(tmp_path)[0]["step"] == 10


@pytest.mark.parametrize("filename", ["metrics.jsonl", "metrics.json"])
@pytest.mark.parametrize("kind", ["symlink", "fifo", "directory"])
def test_metric_files_reject_links_and_special_files_without_blocking(
    tmp_path, monkeypatch, filename, kind
):
    path = tmp_path / filename
    if kind == "symlink":
        target = tmp_path / "private-metrics.json"
        write_json(target, {"step": 999, "metrics": {"private": 100}})
        path.symlink_to(target)
    elif kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("Named pipes are unavailable")
        os.mkfifo(path)
    else:
        path.mkdir()

    expected = []
    if filename == "metrics.jsonl":
        write_json(tmp_path / "metrics.json", {"step": 7, "metrics": {"loss": 0.5}})
        expected = [{"step": 7, "metrics": {"loss": 0.5}, "updated_at": None}]
    original_open = os.open

    def safe_open(candidate, flags, *args, **kwargs):
        if os.fspath(candidate) == os.fspath(path):
            # A regression fails before an actual FIFO open could hang pytest.
            assert kind != "symlink", "A metric symlink must not be opened"
            if hasattr(os, "O_NONBLOCK"):
                assert flags & os.O_NONBLOCK
        return original_open(candidate, flags, *args, **kwargs)

    monkeypatch.setattr(inspection.os, "open", safe_open)
    assert inspection.metric_history(tmp_path) == expected


@pytest.mark.parametrize("filename", ["metrics.jsonl", "metrics.json"])
def test_metric_open_rejects_a_file_replaced_by_a_symlink(
    tmp_path, monkeypatch, filename
):
    if not hasattr(os, "O_NOFOLLOW"):
        pytest.skip("Atomic no-follow opens are unavailable")
    path = tmp_path / filename
    write_json(path, {"step": 1, "metrics": {}})
    target = tmp_path / "private-metrics.json"
    write_json(target, {"step": 999, "metrics": {"private": 100}})
    original_open = os.open

    def replaced_open(candidate, flags, *args, **kwargs):
        if os.fspath(candidate) == os.fspath(path):
            assert flags & os.O_NOFOLLOW
            path.unlink()
            path.symlink_to(target)
        return original_open(candidate, flags, *args, **kwargs)

    monkeypatch.setattr(inspection.os, "open", replaced_open)
    assert inspection.metric_history(tmp_path) == []


def test_metric_history_rejects_a_symlinked_run_directory(tmp_path):
    run = tmp_path / "original"
    run.mkdir()
    write_json(run / "metrics.json", {"step": 999, "metrics": {"private": 100}})
    link = tmp_path / "alias"
    link.symlink_to(run, target_is_directory=True)
    assert inspection.metric_history(link) == []


@pytest.mark.parametrize("filename", ["metrics.jsonl", "metrics.json"])
def test_metric_reads_keep_the_initial_window_when_a_file_grows(
    tmp_path, monkeypatch, filename
):
    path = tmp_path / filename
    path.write_text(json.dumps({"step": 5, "metrics": {"loss": 0.5}}))
    original_fdopen = os.fdopen
    appended = False

    class GrowingReader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def fileno(self):
            return self.stream.fileno()

        def seek(self, offset, whence=os.SEEK_SET):
            nonlocal appended
            position = self.stream.seek(offset, whence)
            if whence == os.SEEK_END and not appended:
                appended = True
                with path.open("ab") as writer:
                    writer.write(b'\n{"step":999,"metrics":{"loss":100}}')
            return position

        def readline(self, size):
            return self.stream.readline(size)

        def read(self, size):
            return self.stream.read(size)

    monkeypatch.setattr(
        inspection.os,
        "fdopen",
        lambda *args, **kwargs: GrowingReader(original_fdopen(*args, **kwargs)),
    )
    assert inspection.metric_history(tmp_path) == [
        {"step": 5, "metrics": {"loss": 0.5}, "updated_at": None}
    ]
    assert appended


@pytest.mark.parametrize("extra_byte", [0, 1])
def test_metric_snapshot_enforces_the_complete_file_byte_limit(tmp_path, extra_byte):
    value = {"step": 5, "metrics": {"loss": 0.5}, "padding": ""}
    value["padding"] = "x" * (128 * 1024 + extra_byte - len(json.dumps(value)))
    raw = json.dumps(value)
    assert len(raw) == 128 * 1024 + extra_byte
    (tmp_path / "metrics.json").write_text(raw)
    expected = (
        []
        if extra_byte
        else [{"step": 5, "metrics": {"loss": 0.5}, "updated_at": None}]
    )
    assert inspection.metric_history(tmp_path) == expected


@pytest.mark.parametrize("limit", [0, -1, True, 1.2])
def test_history_rejects_invalid_limit(tmp_path, limit):
    with pytest.raises(ValueError, match="positive integer"):
        inspection.metric_history(tmp_path, limit=limit)


def test_deeply_nested_metric_json_is_skipped(tmp_path):
    raw = "[" * 2000 + "]" * 2000
    (tmp_path / "metrics.jsonl").write_text(raw)
    (tmp_path / "metrics.json").write_text(raw)
    assert inspection.metric_history(tmp_path) == []


@pytest.mark.parametrize(
    "values,width,expected",
    [
        ([], 44, ""),
        ([1, 1], 44, "=="),
        ([float("nan")], 44, ""),
        ([1, 2], 0, ""),
        ([0, 1], 44, ".@"),
        ([-1e308, 1e308], 44, ".@"),
        ([1, 2, 3, 4], 2, ".@"),
        ([1, 2, 3], 1, "="),
    ],
)
def test_sparkline_only_samples_reported_values(values, width, expected):
    assert inspection.metric_sparkline(values, width=width) == expected


def test_inspection_is_background_reentrant_and_marks_changed_draft(
    local, tmp_path, monkeypatch
):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def inspect(config, workspace):
        calls.append((config, workspace))
        entered.set()
        assert release.wait(3)
        finished.set()
        return {
            "model": config.model,
            "provider": config.provider,
            "failed": 0,
            "warnings": 0,
            "checks": [],
        }

    monkeypatch.setattr(tui, "inspect_readiness", inspect)
    studio = tui.Studio(tmp_path / "studio", local)
    try:
        studio.handle("i")
        assert entered.wait(2) and studio.readiness_panel.busy
        assert (
            "background"
            in " ".join(
                studio.readiness_panel.lines(
                    studio._settings_signature(), studio.values
                )
            ).lower()
        )
        studio.handle("i")
        studio.handle("ESC")
        studio.values["model"] = "another/model"
        assert len(calls) == 1 and studio.overlay is None
    finally:
        release.set()
    assert finished.wait(2)
    deadline = time.monotonic() + 2
    while studio.readiness_panel.busy and time.monotonic() < deadline:
        studio.refresh()
        time.sleep(0.005)
    assert not studio.readiness_panel.busy
    assert (
        "changed"
        in " ".join(
            studio.readiness_panel.lines(studio._settings_signature(), studio.values)
        ).lower()
    )


def test_invalid_numeric_draft_readiness_does_not_start_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tui, "inspect_readiness", lambda *_: pytest.fail("Invalid draft")
    )
    studio = tui.Studio(tmp_path)
    studio.values["steps"] = "invalid"
    studio.handle("i")
    assert (
        not studio.readiness_panel.busy and studio.readiness_panel.report["failed"] == 1
    )


def test_background_inspection_exception_is_sanitized(local, tmp_path, monkeypatch):
    class ImmediateThread:
        def __init__(self, *, target, **kwargs):
            self.target = target

        def start(self):
            self.target()

    def fail(*_):
        raise OSError("sensitive subprocess detail")

    monkeypatch.setattr("stateset_agents.tui_tasks.threading.Thread", ImmediateThread)
    monkeypatch.setattr(tui, "inspect_readiness", fail)
    studio = tui.Studio(tmp_path, local)
    studio.handle("i")
    studio.refresh()
    assert not studio.readiness_panel.busy
    assert studio.readiness_panel.report["model"] == local.model
    text = " ".join(
        studio.readiness_panel.lines(studio._settings_signature(), studio.values)
    )
    assert "OSError" in text and "sensitive" not in text
    assert all(cell_len(line) <= 74 for line in studio.lines(74, 24))


@pytest.fixture
def runs(tmp_path, monkeypatch):
    rows = [
        {
            "id": name,
            "path": str(tmp_path / name),
            "state": {"status": status},
            "plan": {
                "config": asdict(replace(FinetuneConfig(), model=model, method=method))
            },
        }
        for name, status, model, method in [
            ("one", "failed", "Qwen/Small", "sft"),
            ("two", "queued", "Qwen/Small", "gspo"),
            ("three", "succeeded", "Other/Model", "sft"),
        ]
    ]
    for row in rows:
        (tmp_path / row["id"]).mkdir()
    monkeypatch.setattr(tui, "list_runs", lambda _, **kwargs: rows)
    studio = tui.Studio(tmp_path)
    studio.handle("TAB")
    return studio


def test_run_search_status_and_reset(runs):
    runs.handle("/")
    for key in "QWEN GSPO":
        runs.handle(key)
    assert runs.run_query == ""
    runs.handle("ENTER")
    assert [row["id"] for row in runs.runs] == ["two"]
    runs.handle("t")
    assert runs.run_filter == "active" and len(runs.runs) == 1
    runs.handle("t")
    assert not runs.runs and "Backspace" in " ".join(runs.lines(74, 44))
    runs.handle("BACKSPACE")
    assert runs.run_query == "" and len(runs.runs) == 3
    runs.handle("/")
    runs.handle("x")
    runs.handle("ESC")
    assert runs.run_query == "" and len(runs.runs) == 3


def test_filter_refresh_preserves_selected_identity(runs):
    runs.run_index = 1
    runs.handle("t")
    assert runs.runs[runs.run_index]["id"] == "two"
    runs.handle("BACKSPACE")
    assert runs.runs[runs.run_index]["id"] == "two"


def test_completed_run_does_not_silently_switch_metric_overlay(runs):
    runs.handle("t")
    runs.handle("g")
    assert runs.overlay == "metrics" and runs.runs[0]["id"] == "two"
    runs.all_runs[1]["state"]["status"] = "succeeded"
    runs.refresh()
    assert runs.overlay is None and not runs.runs
    assert "left this filter" in runs.message
    runs.handle("BACKSPACE")
    assert len(runs.runs) == 3


def test_progress_reports_commits_or_epoch_never_trainer_step_as_budget(runs):
    row = runs.all_runs[1]
    history = [{"step": 999, "metrics": {"reward": 0.9}, "updated_at": time.time()}]
    assert "not reported" in runs._progress_lines(row, history, 74)[0]
    row["state"]["committed_updates"] = 10
    row["plan"]["config"]["steps"] = 20
    assert "10/20" in runs._progress_lines(row, history, 74)[0]
    assert "50%" in runs._progress_lines(row, history, 74)[0]
    row = runs.all_runs[0]
    assert "not reported" in runs._progress_lines(row, history, 74)[0]
    history[0]["metrics"]["epoch"] = 0.5
    assert "Reported epoch: 0.5/3" in runs._progress_lines(row, history, 74)[0]


def test_metric_overlay_handles_empty_one_sample_and_scroll(runs):
    runs.handle("g")
    assert "History starts" in " ".join(runs._metric_lines(74))
    path = runs.workspace / "one"
    _metrics(path, 1, {"loss": 2, "epoch": 0.5})
    assert "One sample" in " ".join(runs._metric_lines(74))
    _metrics(path, 2, {"loss": 1, "epoch": 1})
    text = " ".join(runs._metric_lines(74))
    assert "2 retained events" in text and "No ETA" in text
    assert "not independent quality" in text
    for _ in range(30):
        runs.handle("PAGEDOWN")
    lines = runs.lines(74, 24)
    assert len(lines) <= 24 and all(cell_len(line) <= 74 for line in lines)
    assert "Esc" in " ".join(lines)


@pytest.mark.parametrize("counter", [True, -1, float("nan"), 10**400, "ten"])
def test_malformed_retained_counters_do_not_render_progress(runs, counter):
    row = runs.all_runs[1]
    row["state"]["committed_updates"] = counter
    history = [{"step": 1, "metrics": {}, "updated_at": counter}]
    lines = runs._progress_lines(row, history, 74)
    assert "not reported" in lines[0]
    assert "freshness unavailable" in lines[1]
