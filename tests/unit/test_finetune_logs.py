"""Bounded source-aware log reading and stable terminal inspection without training."""

import json
import os
from dataclasses import asdict

import pytest
from typer.testing import CliRunner

from stateset_agents import cli_finetune as api
from stateset_agents import finetune_workspace as workspace
from stateset_agents.finetune_workspace import (
    FinetuneConfig,
    create_run,
    plan_run,
    read_log_snapshot,
)
from stateset_agents.tui import Studio


@pytest.fixture
def retained(tmp_path):
    data = tmp_path / "chat.jsonl"
    data.write_text(
        '{"messages":[{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"}]}\n'
    )
    root = tmp_path / "studio"
    config = FinetuneConfig(dataset=str(data))
    run = create_run(root, plan_run(config))
    (run / "run.log").write_text(
        "training step 1\nWARNING customer-safe test fixture\n"
    )
    (run / "supervisor.log").write_text("supervisor start\nTRACE queue fixture\n")
    studio = Studio(root)
    studio.tab = "runs"
    return studio, run, config


def test_auto_source_falls_back_for_empty_training_but_explicit_source_does_not(
    tmp_path,
):
    (tmp_path / "run.log").touch()
    (tmp_path / "supervisor.log").write_text("startup traceback\n")
    auto = read_log_snapshot(tmp_path)
    assert auto.source == "supervisor" and auto.lines == ("startup traceback",)
    explicit = read_log_snapshot(tmp_path, source="training")
    assert explicit.source == "training" and explicit.available and explicit.lines == ()
    (tmp_path / "run.log").write_text("queue waiting\n")
    assert read_log_snapshot(tmp_path).source == "training"


@pytest.mark.parametrize("kind", ["missing", "symlink", "directory", "fifo"])
def test_missing_or_special_logs_are_reported_without_following_or_blocking(
    tmp_path, kind
):
    path = tmp_path / "run.log"
    if kind == "symlink":
        outside = tmp_path / "private-other-file"
        outside.write_text("PRIVATE CONTENT")
        path.symlink_to(outside)
    elif kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO unavailable")
        os.mkfifo(path)
    snapshot = read_log_snapshot(tmp_path, source="training")
    assert not snapshot.available and snapshot.bytes_read == 0
    assert snapshot.error == ("missing" if kind == "missing" else "unreadable")
    assert "PRIVATE" not in json.dumps(asdict(snapshot))


def test_large_tail_is_bounded_marked_and_terminal_safe(tmp_path):
    (tmp_path / "run.log").write_bytes(
        b"x" * (workspace.MAX_LOG_BYTES + 200)
        + b"\n\x1b[31mERROR\x1b[0m\x00\rnext\tvalue\n"
    )
    snapshot = read_log_snapshot(tmp_path)
    assert snapshot.available and snapshot.window_truncated
    assert snapshot.bytes_read == workspace.MAX_LOG_BYTES
    assert snapshot.lines[-2:] == ("ERROR", "next    value")
    assert not any("\x1b" in line or "\x00" in line for line in snapshot.lines)


def test_concurrent_append_does_not_expand_the_read_window(tmp_path, monkeypatch):
    path = tmp_path / "run.log"
    path.write_text("old")
    original = workspace.os.fdopen
    sizes = []

    class Stream:
        def __init__(self, inner):
            self.inner = inner

        def __enter__(self):
            self.inner.__enter__()
            return self

        def __exit__(self, *args):
            return self.inner.__exit__(*args)

        def fileno(self):
            return self.inner.fileno()

        def seek(self, *args):
            return self.inner.seek(*args)

        def read(self, size):
            sizes.append(size)
            with path.open("ab") as out:
                out.write(b"new" * workspace.MAX_LOG_BYTES)
            return self.inner.read(size)

    monkeypatch.setattr(
        workspace.os, "fdopen", lambda *a, **k: Stream(original(*a, **k))
    )
    snapshot = read_log_snapshot(tmp_path, source="training")
    assert sizes == [3] and snapshot.bytes_read == 3 and snapshot.lines == ("old",)


def test_pause_freezes_visible_window_across_appends_and_resizes(retained, monkeypatch):
    studio, run, _ = retained
    studio.handle("l")
    first = "\n".join(studio.lines(74, 44))
    assert "FOLLOW" in first and "training step 1" in first
    studio.handle(" ")
    (run / "run.log").write_text("NEW OUTPUT\n")
    original = workspace.read_log_snapshot
    monkeypatch.setattr(
        "stateset_agents.tui_panels.read_log_snapshot",
        lambda *a, **k: pytest.fail("paused view must not reread logs"),
    )
    for size in [(72, 22), (74, 44), (110, 32)]:
        text = "\n".join(studio.lines(*size))
        assert (
            "PAUSED" in text and "training step 1" in text and "NEW OUTPUT" not in text
        )
    monkeypatch.setattr("stateset_agents.tui_panels.read_log_snapshot", original)
    studio.handle("END")
    assert "NEW OUTPUT" in "\n".join(studio.lines(74, 44))
    studio.handle("s")
    studio.handle("s")
    text = "\n".join(studio.lines(74, 44))
    assert "supervisor.log" in text and "TRACE queue fixture" in text


def test_filter_is_literal_case_insensitive_scoped_and_does_not_change_run_search(
    retained,
):
    studio, _, _ = retained
    studio.handle("l")
    studio.lines(74, 44)
    studio.handle("/")
    for char in "warning":
        studio.handle(char)
    studio.handle("ENTER")
    text = "\n".join(studio.lines(74, 44))
    assert "WARNING customer-safe" in text and "training step 1" not in text
    assert (
        studio.log_viewer.query == "warning"
        and studio.run_query == ""
        and not studio.log_viewer.follow
    )
    studio.handle("/")
    studio.handle("CLEAR")
    for char in ".*":
        studio.handle(char)
    studio.handle("ENTER")
    assert "No matching lines in this tail window" in "\n".join(studio.lines(74, 44))
    studio.handle("/")
    studio.handle("CLEAR")
    studio.handle("ESC")
    assert studio.overlay == "logs" and studio.log_viewer.query == ".*"
    studio.handle("BACKSPACE")
    assert studio.log_viewer.query == ""
    studio.handle("ESC")
    assert studio.overlay is None and studio.log_viewer is None


def test_scrolling_and_live_refresh_remain_bound_to_selected_run(retained):
    studio, run, config = retained
    (run / "run.log").write_text("\n".join(f"line-{index:03}" for index in range(100)))
    studio.handle("ENTER")
    studio.handle("l")
    studio.lines(74, 44)
    studio.handle("HOME")
    assert (
        "line-000" in "\n".join(studio.lines(74, 44)) and not studio.log_viewer.follow
    )
    other = create_run(studio.workspace, plan_run(config))
    studio.refresh()
    studio.run_index = next(
        index for index, row in enumerate(studio.runs) if row["id"] == other.name
    )
    assert studio.log_viewer.run == run
    studio.handle("END")
    assert "line-099" in "\n".join(studio.lines(74, 44))
    studio.run_filter = "failed"
    studio.refresh()
    assert studio.overlay is None and studio.log_viewer is None
    assert "left this filter" in studio.message


@pytest.mark.parametrize("width,height", [(72, 22), (74, 44), (110, 32)])
def test_full_view_wraps_long_lines_and_limits_visible_scope(retained, width, height):
    studio, run, _ = retained
    (run / "run.log").write_text(
        "omitted-marker\n"
        + "short\n" * 2100
        + "WIDE "
        + "界" * 80
        + "\nEND-OF-LONG-LINE\n",
        encoding="utf-8",
    )
    studio.handle("l")
    frame = studio.lines(width, height)
    text = "\n".join(frame)
    assert len(frame) == height and "Earlier lines omitted" in text
    assert "END-OF-LONG-LINE" in text and "omitted-marker" not in text
    assert "Space pause/follow" in text and "End follow" in text


def test_agent_logs_report_actual_source_bounds_and_redact_known_keys(
    retained, monkeypatch
):
    studio, run, _ = retained
    monkeypatch.setenv("RUNPOD_API_KEY", "synthetic-log-secret")
    (run / "supervisor.log").write_text("Trace synthetic-log-secret\n")
    result = CliRunner().invoke(
        api.app,
        [
            "logs",
            run.name,
            "--workspace",
            str(studio.workspace),
            "--source",
            "supervisor",
        ],
    )
    assert result.exit_code == 0, result.stdout
    report = json.loads(result.stdout)["result"]
    assert report["source"] == "supervisor" and report["filename"] == "supervisor.log"
    assert report["available"] and not report["window_truncated"]
    assert report["bytes_read"] == (run / "supervisor.log").stat().st_size
    assert report["lines"] == ["Trace [REDACTED]"]
    before = (run / "supervisor.log").read_bytes()
    result = CliRunner().invoke(
        api.app,
        [
            "logs",
            run.name,
            "--workspace",
            str(studio.workspace),
            "--source",
            "../../private",
        ],
    )
    assert (
        result.exit_code == 1
        and json.loads(result.stdout)["error"]["code"] == "invalid_source"
    )
    assert (run / "supervisor.log").read_bytes() == before


def test_reading_logs_never_mutates_worker_or_saved_plan(retained):
    studio, run, _ = retained
    before = {path.name: path.read_bytes() for path in run.iterdir() if path.is_file()}
    studio.handle("l")
    studio.lines(74, 44)
    studio.handle("PAGEUP")
    studio.handle("PAGEDOWN")
    studio.handle("s")
    studio.handle("ESC")
    assert {
        path.name: path.read_bytes() for path in run.iterdir() if path.is_file()
    } == before
    assert not (run / "launch.lock").exists()
