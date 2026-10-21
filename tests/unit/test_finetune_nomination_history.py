"""Saved nominations can be reopened without duplicate receipts or inference."""

import copy
import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from typer.testing import CliRunner

from stateset_agents import cli_finetune as cli
from stateset_agents import finetune_evaluation as evaluation
from stateset_agents import finetune_nomination as nomination
from stateset_agents import finetune_workspace as ws
from stateset_agents.evaluation.agent_runs import content_hash
from stateset_agents.finetune_evaluation_tui import EvaluationPanel
from stateset_agents.tui import Studio
from tests.unit import test_finetune_evaluation as evaluation_tests
from tests.unit.test_finetune_checkpoint_evaluation import published
from tests.unit.test_finetune_nomination import candidate


def inventory(run):
    return {
        p.relative_to(run).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in run.rglob("*")
        if p.is_file()
    }


def test_history_status_and_config_read_no_weights_cases_or_private_reports(
    tmp_path, monkeypatch
):
    run, _, _, _, receipt = candidate(tmp_path, monkeypatch)
    before = inventory(run)

    def forbidden(*args, **kwargs):
        raise AssertionError("Metadata must not read model bytes")

    monkeypatch.setattr(evaluation, "_inventory", forbidden)
    monkeypatch.setattr(evaluation, "inspect_evaluation_checkpoint", forbidden)
    original = ws.read_json

    def metadata_only(path, **kwargs):
        assert Path(path).name not in {
            "cases.json",
            "paired-reports.json",
            "baseline.json",
            "candidate.json",
        }
        return original(path, **kwargs)

    monkeypatch.setattr(nomination, "read_json", metadata_only)
    rows = nomination.list_nominations(run)["nominations"]
    assert [r["nomination_id"] for r in rows] == [receipt["nomination_id"]]
    status = nomination.nomination_status(run, receipt["nomination_id"])
    restored = nomination.nomination_config(
        run, receipt["nomination_id"], expected_plan_sha256=status["plan_sha256"]
    )
    assert restored["config"]["cases"] == "" and restored["config"]["split"] == "test"
    assert restored["config"]["nomination_id"] == receipt["nomination_id"]
    assert not restored["prepared"] and not status["bytes_verified"]
    assert "PRIVATE" not in json.dumps(restored)
    assert inventory(run) == before


@pytest.mark.parametrize("change", ["checksum", "extra_field", "missing", "symlink"])
def test_broken_rows_remain_visible_without_exposing_private_values(
    tmp_path, monkeypatch, change
):
    run, _, _, _, receipt = candidate(tmp_path, monkeypatch)
    path = run / "nominations" / receipt["nomination_id"] / "receipt.json"
    if change == "missing":
        path.unlink()
    elif change == "symlink":
        original = tmp_path / "PRIVATE-receipt.json"
        path.rename(original)
        path.symlink_to(original)
    else:
        value = ws.read_json(path)
        value["plan"]["PRIVATE"] = "PRIVATE case or credential value"
        if change == "extra_field":
            value["plan_sha256"] = content_hash(value["plan"])
        ws.write_json(path, value)
    row = nomination.list_nominations(run)["nominations"][0]
    assert row == {
        "nomination_id": receipt["nomination_id"],
        "state": "unavailable",
        "bytes_verified": False,
    }
    assert "PRIVATE" not in json.dumps(row)
    with pytest.raises((ValueError, OSError)):
        nomination.nomination_status(run, receipt["nomination_id"])


def test_listing_caches_selection_and_code_identity_for_repeated_receipts(
    tmp_path, monkeypatch
):
    run, _, _, _, receipt = candidate(tmp_path, monkeypatch)
    value = ws.read_json(
        run / "nominations" / receipt["nomination_id"] / "receipt.json"
    )
    for index in range(3):
        path = run / "nominations" / f"20261018-000000-{index:08x}"
        path.mkdir()
        ws.write_json(path / "receipt.json", value)
    selection, identity = nomination._selection_plan, nomination.package_implementation
    calls = {"selection": 0, "identity": 0}

    def selected(*args):
        calls["selection"] += 1
        return selection(*args)

    def implementation():
        calls["identity"] += 1
        return identity()

    monkeypatch.setattr(nomination, "_selection_plan", selected)
    monkeypatch.setattr(nomination, "package_implementation", implementation)
    rows = nomination.list_nominations(run)["nominations"]
    assert len(rows) == 4 and all(row["state"] == "available" for row in rows)
    assert calls == {"selection": 1, "identity": 1}
    assert [r["nomination_id"] for r in rows] == sorted(
        [r["nomination_id"] for r in rows], reverse=True
    )


def test_metadata_never_claims_changed_weights_are_verified(tmp_path, monkeypatch):
    run, config, _, _, receipt = candidate(tmp_path, monkeypatch)
    (run / "artifacts/checkpoint-1/adapter_model.safetensors").write_bytes(b"changed")
    status = nomination.nomination_status(run, receipt["nomination_id"])
    assert status["state"] == "available" and not status["bytes_verified"]
    restored = nomination.nomination_config(
        run, receipt["nomination_id"], expected_plan_sha256=status["plan_sha256"]
    )
    values = dict(restored["config"], cases=config.cases)
    with pytest.raises(ValueError):
        evaluation.plan_evaluation(run, evaluation.EvaluationConfig(**values))


def test_stale_code_or_exact_selection_is_rejected(tmp_path, monkeypatch):
    run, _, _, _, receipt = candidate(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="differs from the selected"):
        nomination.nomination_config(
            run, receipt["nomination_id"], expected_plan_sha256="0" * 64
        )
    monkeypatch.setattr(nomination, "package_implementation", lambda: {"changed": True})
    assert nomination.list_nominations(run)["nominations"][0]["state"] == "unavailable"
    with pytest.raises(ValueError):
        nomination.nomination_status(run, receipt["nomination_id"])


def test_empty_history_is_read_only(tmp_path):
    run, _, _ = published(tmp_path)
    before = inventory(run)
    assert nomination.list_nominations(run) == {
        "nominations": [],
        "bytes_verified": False,
    }
    assert inventory(run) == before and not (run / "nominations").exists()


def test_oversize_history_rejects_scan_but_specific_status_still_works(
    tmp_path, monkeypatch
):
    run, _, _, _, receipt = candidate(tmp_path, monkeypatch)
    for index in range(100):
        (run / "nominations" / f"extra-{index}").mkdir()
    with pytest.raises(ValueError, match="exceeds 100"):
        nomination.list_nominations(run)
    assert (
        nomination.nomination_status(run, receipt["nomination_id"])["state"]
        == "available"
    )


def test_fresh_panel_restores_saved_nomination_without_creating_another(
    tmp_path, monkeypatch
):
    run, _, _, _, receipt = candidate(tmp_path, monkeypatch)
    before = inventory(run)
    panel = EvaluationPanel(run)
    panel.handle("h")
    evaluation_tests.wait(panel)
    assert panel.mode == "nominations" and "SAVED CHECKPOINT NOMINATIONS" in "\n".join(
        panel.lines()
    )
    panel.handle("ENTER")
    evaluation_tests.wait(panel)
    assert panel.mode == "form" and panel.values["cases"] == ""
    assert panel.values["nomination_id"] == receipt["nomination_id"]
    assert panel.values["checkpoint"] == "checkpoint-1"
    assert panel.review is None and panel.prepared_path is None
    assert inventory(run) == before


def test_browser_escape_and_stale_row_preserve_unsaved_form(tmp_path, monkeypatch):
    run, _, paths, _, _ = candidate(tmp_path, monkeypatch)
    panel = EvaluationPanel(run)
    panel.handle("n")
    panel.values["judge_model"] = "unsaved/judge"
    before = copy.deepcopy(panel.values)
    panel.handle("h")
    evaluation_tests.wait(panel)
    panel.handle("ESC")
    assert panel.mode == "form" and panel.values == before
    panel.handle("h")
    evaluation_tests.wait(panel)
    path = paths[0] / "plan.json"
    value = ws.read_json(path)
    value["config"]["seed"] += 1
    ws.write_json(path, value)
    panel.handle("ENTER")
    evaluation_tests.wait(panel)
    assert panel.mode == "nominations" and panel.values == before
    panel.handle("r")
    evaluation_tests.wait(panel)
    assert panel.nominations[0]["state"] == "unavailable"
    panel.handle("ENTER")
    assert not panel.busy


def test_late_restore_cannot_steal_an_editor_after_escape(tmp_path, monkeypatch):
    run, _, _, _, _ = candidate(tmp_path, monkeypatch)
    panel = EvaluationPanel(run)
    panel.handle("n")
    panel.handle("h")
    evaluation_tests.wait(panel)
    original = nomination.nomination_config
    started, release = threading.Event(), threading.Event()

    def delayed(*args, **kwargs):
        started.set()
        assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "stateset_agents.finetune_evaluation_tui.nomination_config", delayed
    )
    panel.handle("ENTER")
    assert started.wait(10)
    panel.handle("ESC")
    panel.values["judge_model"] = "changed/judge"
    release.set()
    evaluation_tests.wait(panel)
    assert panel.mode == "form" and panel.values["judge_model"] == "changed/judge"


def test_agent_lists_inspects_and_restores_exact_saved_selection(tmp_path, monkeypatch):
    run, _, _, _, receipt = candidate(tmp_path, monkeypatch)
    before = inventory(run)
    runner = CliRunner()
    common = ["--workspace", str(run.parent)]
    for command, args in [
        ("nominations", []),
        ("nomination-status", [receipt["nomination_id"]]),
        (
            "nomination-config",
            [
                receipt["nomination_id"],
                "--expected-plan-sha256",
                receipt["plan_sha256"],
            ],
        ),
    ]:
        result = runner.invoke(cli.app, [command, run.name, *args, *common])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["ok"] and "PRIVATE" not in result.output
        if command == "nomination-config":
            restored = json.loads(result.output)["result"]
            assert restored["config"]["cases"] == "" and not restored["prepared"]
    assert inventory(run) == before


@pytest.mark.parametrize("size", [(72, 22), (74, 44), (110, 32)])
def test_saved_nomination_navigation_keeps_selected_row_visible(
    tmp_path, monkeypatch, size
):
    run, _, _, _, receipt = candidate(tmp_path, monkeypatch)
    value = ws.read_json(
        run / "nominations" / receipt["nomination_id"] / "receipt.json"
    )
    for index in range(35):
        path = run / "nominations" / f"20261018-000000-{index:08x}"
        path.mkdir()
        ws.write_json(path / "receipt.json", value)
    studio = Studio(run.parent)
    studio.handle("s")
    before = (run.parent / "draft.json").read_bytes()
    panel = EvaluationPanel(run)
    studio.evaluation_panel, studio.overlay = panel, "evaluation"
    panel.handle("h")
    evaluation_tests.wait(panel)
    for key in ["END", "PAGEUP", "HOME", "PAGEDOWN"]:
        panel.handle(key)
        screen = "\n".join(studio.lines(*size))
        assert panel.nominations[panel.nomination_index]["nomination_id"] in screen
    assert (run.parent / "draft.json").read_bytes() == before


@pytest.mark.skipif(
    sys.platform != "linux", reason="Saved nomination curses flow uses a Linux PTY"
)
@pytest.mark.parametrize("size", [(74, 44), (110, 32)])
def test_real_terminal_restores_reviews_and_exits_without_new_records(
    tmp_path, monkeypatch, size
):
    import fcntl
    import pty
    import select
    import struct
    import termios
    import time

    run, config, _, _, _ = candidate(tmp_path, monkeypatch)
    Studio(run.parent).handle("s")
    before, draft = inventory(run), (run.parent / "draft.json").read_bytes()
    terminal, child = pty.openpty()
    fcntl.ioctl(
        terminal, termios.TIOCSWINSZ, struct.pack("HHHH", size[1], size[0], 0, 0)
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "stateset_agents.cli",
            "tui",
            "--workspace",
            str(run.parent),
        ],
        stdin=child,
        stdout=child,
        stderr=child,
        env=dict(
            os.environ, TERM="xterm-256color", COLUMNS=str(size[0]), LINES=str(size[1])
        ),
    )
    os.close(child)
    captured, full = bytearray(), bytearray()

    def wait_for(marker):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if marker.encode() in captured:
                return
            ready, _, _ = select.select([terminal], [], [], 0.1)
            if ready:
                try:
                    chunk = os.read(terminal, 65536)
                except OSError:
                    break
                captured.extend(chunk)
                full.extend(chunk)
        pytest.fail(f"Missing terminal marker {marker!r}: {bytes(captured)[-1500:]!r}")

    def send(keys, marker):
        captured.clear()
        os.write(terminal, keys)
        wait_for(marker)

    try:
        wait_for("FINE-TUNING STUDIO")
        send(b"\t", "RUNS 1/1")
        send(b"j", "LOCAL PAIRED EVALUATION")
        send(b"h", "SAVED CHECKPOINT NOMINATIONS")
        send(b"\r", "ready. Choose")
        send(b"\r", "Edit:")
        send(str(config.cases).encode() + b"\r", "Arrows/Tab field")
        send(b"r", "Evaluation SHA256")
        send(b"\x1b", "n new evaluation")
        send(b"\x1b", "j evaluate")
        os.write(terminal, b"q")
        assert process.wait(timeout=15) == 0
        assert (
            inventory(run) == before
            and (run.parent / "draft.json").read_bytes() == draft
        )
        assert b"PRIVATE" not in full
    finally:
        os.close(terminal)
        if process.poll() is None:
            process.kill()
            process.wait()
