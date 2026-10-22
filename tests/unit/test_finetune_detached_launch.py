"""Runs outlive the launching terminal, and an interrupted supervisor reports."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from stateset_agents import finetune_worker as worker
from stateset_agents import finetune_workspace as workspace


def test_windows_launches_get_their_own_hidden_process_group():
    flags = workspace.detached_process_options(windows=True)["creationflags"]
    assert flags & 0x00000200  # CREATE_NEW_PROCESS_GROUP
    assert flags & 0x08000000  # CREATE_NO_WINDOW: no shared console
    assert workspace.detached_process_options(windows=False) == {
        "start_new_session": True
    }


@pytest.fixture
def run(tmp_path, monkeypatch):
    source = tmp_path / "chat.jsonl"
    source.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ]
            }
        )
        + "\n"
    )
    monkeypatch.setattr(worker.os, "umask", lambda _mask: 0)
    config = workspace.FinetuneConfig(dataset=str(source))
    return workspace.create_run(tmp_path / "workspace", workspace.plan_run(config))


@pytest.mark.parametrize("platform_is_windows", [True, False])
def test_every_launch_path_uses_the_detached_options(
    run, monkeypatch, platform_is_windows
):
    sentinel = workspace.detached_process_options(windows=platform_is_windows)
    monkeypatch.setattr(workspace, "detached_process_options", lambda: sentinel)
    monkeypatch.setattr(worker, "detached_process_options", lambda: sentinel)
    seen = []

    def popen(argv, **kwargs):
        seen.append(kwargs)
        return SimpleNamespace(pid=1, poll=lambda: 0, wait=lambda **_: 0)

    monkeypatch.setattr(workspace.subprocess, "Popen", popen)
    workspace.launch_run(run)
    monkeypatch.setattr(worker, "queue_executable", lambda: None)
    worker.supervise(run)
    assert len(seen) == 2
    for kwargs in seen:
        assert {key: kwargs.get(key) for key in sentinel} == sentinel
        if platform_is_windows:
            assert "start_new_session" not in kwargs


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_interrupted_supervisor_stops_its_worker_and_records_an_outcome(
    run, monkeypatch, interruption
):
    """Ctrl+C used to bypass ``except Exception`` and leave no outcome."""
    stopped = []
    monkeypatch.setattr(worker, "queue_executable", lambda: None)

    class Worker:
        pid, returncode = 4242, None

        def wait(self, timeout=None):
            if not stopped:
                raise interruption()
            self.returncode = -15
            return -15

        def poll(self):
            return self.returncode

    monkeypatch.setattr(worker.subprocess, "Popen", lambda *a, **k: Worker())
    monkeypatch.setattr(
        worker, "_stop_worker_group", lambda process, *, force: stopped.append(force)
    )
    with pytest.raises(interruption):
        worker.supervise(run)
    assert stopped == [False]
    state = workspace.read_json(run / "state.json")
    assert state["status"] == "failed"
    failure = workspace.read_json(run / "failure.json")
    assert failure["code"] == "supervisor_stopped"
    assert workspace.failure_info(run, state)["code"] == "supervisor_stopped"
