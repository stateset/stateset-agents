"""Dataset statistics, explicit private excerpts and responsive read-only inspection."""

import hashlib
import json
import os
import subprocess
import sys
import threading
import time

import pytest
from typer.testing import CliRunner

from stateset_agents import cli_finetune as api
from stateset_agents import finetune_workspace as workspace
from stateset_agents.data.text_sft import DatasetFormatError
from stateset_agents.finetune_inspection import inspect_training_data
from stateset_agents.finetune_workspace import FinetuneConfig, read_training_data
from stateset_agents.tui import Studio


def write_chats(path, replies):
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": str(i),
                    "messages": [
                        {"role": "user", "content": "PRIVATE QUESTION"},
                        {"role": "assistant", "content": reply},
                    ],
                }
            )
            for i, reply in enumerate(replies)
        )
        + "\n"
    )
    return path


def wait_for_scan(studio):
    deadline = time.monotonic() + 5
    while studio.data_inspecting and time.monotonic() < deadline:
        studio.refresh()
        time.sleep(0.005)
    assert not studio.data_inspecting


def test_sft_statistics_count_characters_and_duplicates_without_exposing_text(tmp_path):
    path = write_chats(tmp_path / "chat.jsonl", ["界", "界", "界", "different"])
    report = inspect_training_data(path)
    assert report["examples"] == 4 and report["duplicate_examples"] == 2
    assert report["role_counts"] == {"user": 4, "assistant": 4}
    assert report["characters_per_example"] == {
        "min": 17,
        "median": 17,
        "p95": 25,
        "max": 25,
    }
    assert report["turns_per_example"] == {"min": 1, "median": 1, "p95": 1, "max": 1}
    assert report["dataset_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["bytes"] == path.stat().st_size
    assert len(report["warnings"]) == 2
    assert "PRIVATE" not in json.dumps(report) and "samples" not in report
    assert "not tokens" in report["notes"][0]
    assert len(path.read_text().splitlines()) == 4


@pytest.mark.parametrize("method", ["grpo", "gspo", "dapo"])
def test_native_rl_statistics_define_duplicates_and_scripted_turns(tmp_path, method):
    row = {
        "id": "one",
        "context": "PRIVATE context",
        "policy_facts": ["PRIVATE fact"],
        "expected_resolution": "PRIVATE resolution",
        "user_responses": ["one", "two"],
        "must_avoid": ["never"],
    }
    path = tmp_path / "rl.json"
    path.write_text(
        json.dumps([row, {**row, "id": "two", "extra_metadata": "PRIVATE"}])
    )
    report = inspect_training_data(path, method)
    assert report["examples"] == 2 and report["duplicate_examples"] == 1
    assert report["turn_definition"] == "scripted user responses"
    assert report["turns_per_example"]["max"] == 2
    assert report["role_counts"] == {} and "PRIVATE" not in json.dumps(report)
    assert "core scenario" in report["duplicate_definition"]
    sample = inspect_training_data(path, method, include_samples=True)["samples"][0]
    assert "PRIVATE context" in sample["text"] and "policy_facts" in sample["text"]


def test_samples_are_explicit_bounded_and_tied_to_the_report_hash(tmp_path):
    path = write_chats(tmp_path / "chat.jsonl", ["PRIVATE " + "界" * 6000] * 4)
    report = inspect_training_data(path, include_samples=True)
    assert len(report["samples"]) == 3 and report["samples_included"]
    assert all(
        len(sample["text"]) <= 4096 and sample["truncated"]
        for sample in report["samples"]
    )
    captured = report["dataset_sha256"]
    path.write_text("replaced after scan")
    assert report["dataset_sha256"] == captured
    assert "PRIVATE" in report["samples"][0]["text"]
    with pytest.raises(ValueError, match="boolean"):
        inspect_training_data(path, include_samples="false")


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"[]",
        b"{",
        b"\xff",
        b"[" * 2000,
        b'{"messages":[{"role":"user","content":"PRIVATE"}]}',
        b'{"messages":[],"unknown":"PRIVATE"}',
    ],
)
def test_invalid_data_is_rejected_without_reflecting_private_contents(tmp_path, raw):
    path = tmp_path / "bad.jsonl"
    path.write_bytes(raw)
    with pytest.raises(ValueError) as error:
        inspect_training_data(path)
    assert "PRIVATE" not in str(error.value)
    result = CliRunner().invoke(api.app, ["inspect-data", "--dataset", str(path)])
    assert (
        result.exit_code == 1
        and json.loads(result.stdout)["error"]["code"] == "invalid_dataset"
    )
    assert "PRIVATE" not in result.stdout


@pytest.mark.parametrize("kind", ["missing", "directory", "fifo", "oversize"])
def test_reader_rejects_nonregular_or_oversized_inputs_without_blocking(
    tmp_path, monkeypatch, kind
):
    path = tmp_path / "input"
    if kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO unavailable")
        os.mkfifo(path)
    elif kind == "oversize":
        monkeypatch.setattr(workspace, "MAX_DATA_BYTES", 8)
        path.write_bytes(b"x" * 9)
    with pytest.raises((OSError, ValueError)):
        read_training_data(path)


def test_regular_dataset_symlinks_preserve_existing_training_support(tmp_path):
    path = write_chats(tmp_path / "real.jsonl", ["ok"])
    alias = tmp_path / "alias.jsonl"
    alias.symlink_to(path)
    assert read_training_data(alias) == path.read_bytes()
    assert (
        inspect_training_data(alias)["dataset_sha256"]
        == inspect_training_data(path)["dataset_sha256"]
    )


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_preparation_keeps_existing_invalid_dataset_error_contract(tmp_path, kind):
    path = tmp_path / "input"
    if kind == "directory":
        path.mkdir()
    with pytest.raises(ValueError, match="Dataset must be a file"):
        workspace.plan_run(FinetuneConfig(dataset=str(path)))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"dataset": str(path)}))
    result = CliRunner().invoke(
        api.app,
        ["prepare", "--config", str(config), "--workspace", str(tmp_path / "unused")],
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout)["error"]["code"] == "invalid_dataset_or_plan"
    assert not (tmp_path / "unused").exists()


def test_concurrent_growth_remains_bounded_for_inspection_and_preparation(
    tmp_path, monkeypatch
):
    path = tmp_path / "growing"
    path.write_bytes(b"x")
    monkeypatch.setattr(workspace, "MAX_DATA_BYTES", 8)
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

        def read(self, size):
            sizes.append(size)
            with path.open("ab") as out:
                out.write(b"x" * 100)
            return self.inner.read(size)

    monkeypatch.setattr(
        workspace.os, "fdopen", lambda *a, **k: Stream(original(*a, **k))
    )
    with pytest.raises(ValueError, match="exceeds"):
        read_training_data(path)
    assert sizes == [9]
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match="exceeds"):
        workspace.plan_run(FinetuneConfig(dataset=str(path)))
    assert sizes == [9, 9]


def test_cli_redacts_opt_in_samples_and_runs_without_training_imports(
    tmp_path, monkeypatch
):
    path = write_chats(tmp_path / "chat.jsonl", ["synthetic-dataset-secret"])
    monkeypatch.setenv("RUNPOD_API_KEY", "synthetic-dataset-secret")
    hidden = CliRunner().invoke(api.app, ["inspect-data", "--dataset", str(path)])
    assert (
        hidden.exit_code == 0 and "samples" not in json.loads(hidden.stdout)["result"]
    )
    shown = CliRunner().invoke(
        api.app, ["inspect-data", "--dataset", str(path), "--include-samples"]
    )
    assert shown.exit_code == 0 and "[REDACTED]" in shown.stdout
    assert "synthetic-dataset-secret" not in shown.stdout
    schema = json.loads(CliRunner().invoke(api.app, ["schema"]).stdout)["result"]
    assert schema["effects"]["inspect-data"] == "reads_dataset"
    assert schema["dataset_inspection"] == {
        "sample_flag": "--include-samples",
        "max_samples": 3,
        "max_sample_characters": 4096,
        "length_unit": "unicode_characters",
        "requires_config": False,
    }
    invalid = CliRunner().invoke(
        api.app, ["inspect-data", "--dataset", str(path), "--method", "invalid"]
    )
    assert (
        invalid.exit_code == 1
        and json.loads(invalid.stdout)["error"]["code"] == "invalid_method"
    )
    program = """
import sys,json
from typer.testing import CliRunner
from stateset_agents.cli import app
r=CliRunner().invoke(app,['finetune','inspect-data','--dataset',sys.argv[1]])
assert r.exit_code == 0, r.stdout
assert json.loads(r.stdout)['result']['examples'] == 1
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("width,height", [(72, 22), (74, 44), (110, 32)])
def test_dataset_view_hides_private_text_scrolls_and_preserves_draft(
    tmp_path, width, height
):
    path = write_chats(
        tmp_path / "chat.jsonl", ["FIRST PRIVATE REPLY", "SECOND PRIVATE REPLY"]
    )
    studio = Studio(tmp_path / "unused", FinetuneConfig(dataset=str(path), epochs=0))
    before = dict(studio.values)
    studio.handle("u")
    wait_for_scan(studio)
    assert (
        "error" not in studio.data_report
    )  # Other draft settings need not be launch-valid.
    frame = studio.lines(width, height)
    assert len(frame) == height and "PRIVATE REPLY" not in "\n".join(frame)
    assert "e show/hide" in "\n".join(frame)
    studio.handle("e")
    assert "FIRST PRIVATE REPLY" in "\n".join(studio.lines(width, height))
    studio.handle("RIGHT")
    assert "SECOND PRIVATE REPLY" in "\n".join(studio.lines(width, height))
    for _ in range(100):
        studio.handle("PAGEDOWN")
    assert "after this snapshot" in "\n".join(studio.lines(width, height))
    studio.handle("ENTER")
    assert studio.plan is None and not studio.workspace.exists()
    studio.handle("ESC")
    assert not studio.data_show_samples and studio.values == before
    studio.handle("u")
    wait_for_scan(studio)
    assert "PRIVATE REPLY" not in "\n".join(studio.lines(width, height))


def test_background_scan_stays_responsive_and_reports_changed_settings(
    tmp_path, monkeypatch
):
    path = write_chats(tmp_path / "chat.jsonl", ["PRIVATE"])
    original = inspect_training_data
    entered, release = threading.Event(), threading.Event()

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr("stateset_agents.tui.inspect_training_data", blocked)
    studio = Studio(tmp_path / "unused", FinetuneConfig(dataset=str(path)))
    try:
        studio.handle("u")
        assert entered.wait(2)
        assert "Reading and validating" in "\n".join(studio.lines(74, 44))
        studio.handle("u")
        assert "already running" in studio.message
        studio.handle("ESC")
        assert studio.overlay is None and studio.data_inspecting
        studio.values["dataset"] = str(tmp_path / "new.jsonl")
    finally:
        release.set()
    wait_for_scan(studio)
    studio.overlay = "data"
    studio.handle("e")
    text = "\n".join(studio.lines(74, 44))
    assert "method changed" in text and "PRIVATE" not in text
    assert not studio.workspace.exists()


def test_unexpected_background_errors_are_controlled_and_retryable(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "stateset_agents.tui.inspect_training_data",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("PRIVATE FAILURE")),
    )
    studio = Studio(tmp_path / "unused", FinetuneConfig(dataset="some-path"))
    studio.handle("u")
    wait_for_scan(studio)
    text = "\n".join(studio.lines(74, 44))
    assert "Cannot inspect dataset" in text and "PRIVATE FAILURE" not in text
    studio.handle("u")
    wait_for_scan(studio)
    assert "error" in studio.data_report


@pytest.mark.parametrize(
    "raw,method,code,location",
    [
        (b'\r\n\r\n{"messages": []}', "sft", "invalid_conversation", {"line": 3}),
        (b'\n{"PRIVATE":}', "sft", "invalid_json", {"line": 2, "column": 12}),
        (b"ok\xff", "sft", "invalid_encoding", {"byte": 3}),
        (b"\n\r\n", "sft", "empty_dataset", {}),
        (b'[\n{"PRIVATE":}]', "grpo", "invalid_json", {"line": 2, "column": 12}),
        (b"[{}]", "gspo", "invalid_scenario", {"example": 1, "field": "id"}),
        (b'["PRIVATE"]', "dapo", "invalid_scenario", {"example": 1}),
    ],
)
def test_format_diagnostics_share_safe_one_based_locations(
    tmp_path, raw, method, code, location
):
    path = tmp_path / "PRIVATE-path"
    path.write_bytes(raw)
    with pytest.raises(DatasetFormatError) as captured:
        inspect_training_data(path, method)
    diagnostic = captured.value.diagnostic
    assert diagnostic["code"] == code and diagnostic["location"] == location
    assert "PRIVATE" not in json.dumps(diagnostic)
    result = CliRunner().invoke(
        api.app, ["inspect-data", "--dataset", str(path), "--method", method]
    )
    assert result.exit_code == 1
    error = json.loads(result.stdout)["error"]
    assert error["code"] == "invalid_dataset" and error["diagnostic"] == diagnostic
    assert "PRIVATE" not in result.stdout
    studio = Studio(
        tmp_path / "unused", FinetuneConfig(dataset=str(path), method=method)
    )
    studio.handle("u")
    wait_for_scan(studio)
    assert studio.data_report["diagnostic"] == diagnostic
    assert "scan failed" in studio.message
    text = "\n".join(studio.lines(74, 44))
    assert "PRIVATE" not in text
    for key, value in location.items():
        assert f"{key}: {value}" in text
    assert not studio.workspace.exists()


@pytest.mark.parametrize("method", ["grpo", "gspo", "dapo"])
@pytest.mark.parametrize("problem", ["duplicate", "responses", "avoid"])
def test_scenario_diagnostics_identify_record_and_field(tmp_path, method, problem):
    row = {
        "id": "PRIVATE",
        "context": "PRIVATE",
        "expected_resolution": "PRIVATE",
        "policy_facts": ["PRIVATE"],
        "user_responses": ["PRIVATE"],
    }
    rows = [dict(row), dict(row, id="second")]
    field = {"duplicate": "id", "responses": "user_responses", "avoid": "must_avoid"}[
        problem
    ]
    rows[1][field] = "PRIVATE" if problem != "responses" else [None]
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps(rows))
    with pytest.raises(DatasetFormatError) as captured:
        inspect_training_data(path, method)
    assert captured.value.diagnostic["location"] == {"example": 2, "field": field}
    assert "PRIVATE" not in str(captured.value)


@pytest.mark.parametrize("outcome", ["complete", "changed", "other_action"])
def test_completed_background_scan_updates_only_its_own_pending_message(
    tmp_path, outcome
):
    studio = Studio(tmp_path / "unused", FinetuneConfig(dataset="old.jsonl"))
    signature = studio._data_settings_signature()
    studio.data_inspecting = True
    studio.message = "Reading one dataset snapshot; no run is prepared or started."
    if outcome == "changed":
        studio.values["dataset"] = "new.jsonl"
    elif outcome == "other_action":
        studio.message = "Saved the draft."
    studio._data_results.put((signature, {"examples": 1}))
    studio.refresh()
    assert not studio.data_inspecting
    assert (
        studio.message
        == {
            "complete": "Dataset scan complete; no run was prepared or started.",
            "changed": "Dataset settings changed; press u to scan again.",
            "other_action": "Saved the draft.",
        }[outcome]
    )
