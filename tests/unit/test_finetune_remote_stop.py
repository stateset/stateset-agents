"""A paid RunPod run can be stopped: its verified worker ends its own pod."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap

import pytest

from stateset_agents import finetune_worker as worker
from stateset_agents import finetune_workspace as workspace
from tests.unit import test_finetune_remote_submission as remote_submission

make_run = remote_submission.make_run
pytestmark = pytest.mark.skipif(os.name == "nt", reason="RunPod stop is POSIX-only")


def _running(run, pid, token):
    workspace.write_json(run / "state.json", {"status": "running"})
    workspace.write_json(run / "execution.json", {"pid": pid, "process_token": token})


@pytest.mark.parametrize("observed,signalled", [("worker", True), ("reused", False)])
def test_stop_signals_only_the_verified_runpod_worker(
    make_run, monkeypatch, observed, signalled
):
    run, _ = make_run()
    _running(run, 4242, "worker")
    monkeypatch.setattr(workspace, "process_token", lambda _pid: observed)
    kills = []
    monkeypatch.setattr(workspace.os, "kill", lambda pid, sig: kills.append((pid, sig)))
    digest = workspace.read_run_plan(run)[1]
    result = workspace.request_stop(run, expected_plan_sha256=digest)
    assert workspace.stop_requested(run)
    assert result["worker_signalled"] is signalled
    assert kills == ([(4242, signal.SIGTERM)] if signalled else [])


def test_other_remote_providers_stay_refused(make_run):
    run, _ = make_run()
    plan, _ = workspace.read_run_plan(run)
    plan["config"]["provider"] = "river"
    workspace.write_json(run / "plan.json", plan)
    with pytest.raises(ValueError, match="RunPod runs on POSIX"):
        workspace.request_stop(
            run, expected_plan_sha256=workspace.read_run_plan(run)[1]
        )
    assert not (run / "stop.json").exists()


def test_windows_cannot_signal_a_detached_worker(monkeypatch):
    assert workspace.remote_stop_supported("runpod")
    monkeypatch.setattr(workspace.os, "name", "nt")
    assert not workspace.remote_stop_supported("runpod")


def test_signalled_worker_records_the_requested_stop(make_run, monkeypatch):
    run, config = make_run()
    digest = workspace.read_run_plan(run)[1]

    def stopped_mid_submission(*_args):
        workspace.write_json(
            run / "stop.json", {"schema_version": 1, "plan_sha256": digest}
        )
        raise SystemExit(128 + signal.SIGTERM)

    monkeypatch.setattr(worker, "_train_runpod", stopped_mid_submission)
    with pytest.raises(SystemExit):
        worker.execute(run)
    state = workspace.read_json(run / "state.json")
    assert state["status"] == "cancelled"
    assert workspace.read_json(run / "failure.json")["code"] == "cancelled"


def test_unrequested_termination_leaves_the_outcome_to_the_supervisor(
    make_run, monkeypatch
):
    run, _ = make_run()

    def killed(*_args):
        raise SystemExit(143)

    monkeypatch.setattr(worker, "_train_runpod", killed)
    with pytest.raises(SystemExit):
        worker.execute(run)
    assert workspace.read_json(run / "state.json")["status"] == "running"
    assert not (run / "failure.json").exists()


WORKER = textwrap.dedent("""
    import sys, time
    from pathlib import Path
    from stateset_agents import finetune_worker as worker

    run = Path(sys.argv[1])

    def submit_and_train(*_args):
        try:
            print("training", flush=True)
            time.sleep(60)
        finally:
            (run / "pod-terminated").write_text("yes")

    worker._train_runpod = submit_and_train
    sys.argv = ["finetune_worker", "--execute", str(run)]
    worker.main()
    """)


def test_real_worker_is_stopped_and_terminates_its_pod(make_run, tmp_path):
    run, _ = make_run()
    script = tmp_path / "worker.py"
    script.write_text(WORKER)
    child = subprocess.Popen(
        [sys.executable, str(script), str(run)],
        stdout=subprocess.PIPE,
        text=True,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
    )
    try:
        assert child.stdout.readline().strip() == "training"
        result = workspace.request_stop(
            run, expected_plan_sha256=workspace.read_run_plan(run)[1]
        )
        assert result["worker_signalled"] is True
        child.wait(timeout=30)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    assert child.returncode == 128 + signal.SIGTERM
    assert (run / "pod-terminated").read_text() == "yes"
    assert workspace.run_status(run)["status"] == "cancelled"


def test_tui_reviews_a_runpod_stop_with_runpod_wording(make_run, monkeypatch):
    from stateset_agents.tui import Studio

    run, _ = make_run()
    _running(run, 4242, "worker")
    studio = Studio(run.parent)
    studio.tab = "runs"
    studio.refresh()
    studio.handle("k")
    assert studio.overlay == "stop"
    frame = "\n".join(studio.lines(110, 32))
    assert "REVIEW RUNPOD STOP REQUEST" in frame
    assert "Enter signals the RunPod worker to stop" in frame
    assert "requests local stop" not in frame
