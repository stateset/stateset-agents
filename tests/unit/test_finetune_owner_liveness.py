"""One owner-liveness classifier backs run status, recovery and validation."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from stateset_agents import finetune_workspace as workspace

PID = 200001


def _probe(outcome):
    def probe(_pid):
        if outcome == "gone":
            raise ProcessLookupError()
        if outcome == "denied":
            raise PermissionError()

    return probe


@pytest.mark.parametrize(
    "token,probe,evidence,alive",
    [
        ("recorded", "unused", "alive", True),
        ("someone-else", "unused", "reused", False),
        (None, "gone", "gone", False),
        (None, "denied", "denied", None),
        (None, "answers", "running", None),
    ],
)
def test_owner_evidence_classifies_every_outcome(
    tmp_path, monkeypatch, token, probe, evidence, alive
):
    record = tmp_path / "execution.json"
    workspace.write_json(record, {"pid": PID, "process_token": "recorded"})
    monkeypatch.setattr(workspace, "process_token", lambda _pid: token)
    monkeypatch.setattr(workspace, "probe_process", _probe(probe))
    assert workspace.owner_evidence(record) == evidence
    assert workspace.owner_alive(record) is alive


@pytest.mark.parametrize("pid", [0, -1, "1", True, None])
def test_a_record_without_a_valid_pid_is_undecided(tmp_path, monkeypatch, pid):
    record = tmp_path / "execution.json"
    workspace.write_json(record, {"pid": pid, "process_token": "recorded"})
    monkeypatch.setattr(
        workspace, "probe_process", lambda _pid: pytest.fail("no probe")
    )
    assert workspace.owner_evidence(record) == "malformed"
    assert workspace.owner_alive(record) is None
    assert workspace._owner_confirmed(record) is False


def test_unreadable_records_raise_for_callers_to_decide(tmp_path):
    record = tmp_path / "execution.json"
    record.write_text("{not json")
    with pytest.raises(ValueError):
        workspace.owner_alive(record)
    assert workspace._owner_confirmed(record) is False


@pytest.mark.parametrize(
    "platform,token,probe,confirmed",
    [
        # Linux trusts only a matching start-time token and never probes.
        ("linux", "recorded", "fail", True),
        ("linux", "someone-else", "fail", False),
        ("linux", None, "fail", False),
        # Without tokens, an answering or uninspectable PID is the owner.
        ("darwin", "ignored", "answers", True),
        ("darwin", "ignored", "denied", True),
        ("darwin", "ignored", "gone", False),
    ],
)
def test_run_status_owner_check_keeps_its_platform_semantics(
    tmp_path, monkeypatch, platform, token, probe, confirmed
):
    record = tmp_path / "supervisor.json"
    workspace.write_json(record, {"pid": PID, "process_token": "recorded"})
    monkeypatch.setattr(workspace, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(
        workspace,
        "process_token",
        (
            (lambda _pid: token)
            if platform == "linux"
            else (lambda _pid: pytest.fail("tokens are Linux-only"))
        ),
    )
    monkeypatch.setattr(
        workspace,
        "probe_process",
        (
            (lambda _pid: pytest.fail("Linux must not probe"))
            if probe == "fail"
            else _probe(probe)
        ),
    )
    assert workspace._owner_confirmed(record) is confirmed


@pytest.mark.skipif(sys.platform != "linux", reason="Uses Linux process tokens")
def test_real_processes_are_alive_then_gone(tmp_path):
    import os

    record = tmp_path / "execution.json"
    workspace.write_json(
        record,
        {"pid": os.getpid(), "process_token": workspace.process_token(os.getpid())},
    )
    assert workspace.owner_alive(record) is True
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    token = workspace.process_token(child.pid)
    child.wait()
    workspace.write_json(record, {"pid": child.pid, "process_token": token})
    assert workspace.owner_alive(record) is False
