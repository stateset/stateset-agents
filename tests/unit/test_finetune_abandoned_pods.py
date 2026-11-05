"""A worker killed mid-submission must not leave its RunPod pod billing."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap

import pytest

from stateset_agents import finetune_provider_submission as submission
from stateset_agents import finetune_worker as worker
from stateset_agents import finetune_workspace as workspace
from stateset_agents.remote import runpod
from tests.unit import test_finetune_remote_submission as remote_submission

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX process signals")
JOB_ID, POD_ID = remote_submission.JOB_ID, remote_submission.POD_ID
make_run = remote_submission.make_run

CHILD = textwrap.dedent("""
    import json, os, signal, sys, time
    from pathlib import Path
    from stateset_agents.finetune_provider_submission import SubmissionJournal

    run, leases, job, pod, stage = (
        Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
    )
    journal = SubmissionJournal(run, "runpod", durable_supported=True)
    base = {"schema_version": 1, "provider": "runpod", "job_id": job,
            "resource_kind": "pod", "resource_name": "stateset-sft-" + job,
            "attempt": 1}
    journal.publish({**base, "event": "allocation_intent"})
    if stage == "allocated":
        leases.mkdir(parents=True, exist_ok=True)
        (leases / (pod + ".json")).write_text(
            json.dumps({"provider": "runpod", "pod_id": pod, "job_id": job})
        )
        journal.publish({**base, "event": "allocated", "resource_id": pod})
    print("submitting", flush=True)
    os.kill(os.getpid(), signal.SIGKILL)
    """)

# SIGHUP does not exist on Windows; parametrize is evaluated at import time.
POSIX_TERMINATION_SIGNALS = [signal.SIGTERM] + (
    [signal.SIGHUP] if hasattr(signal, "SIGHUP") else []
)


class Api:
    def __init__(self, pods=(), fail=False):
        self.terminated, self.pods, self.fail = [], list(pods), fail

    def terminate_pod(self, pod_id):
        self.terminated.append(pod_id)
        if self.fail:
            raise ConnectionError("provider unreachable")

    def list_pods(self):
        return self.pods


@pytest.fixture
def killed_worker(make_run, tmp_path, monkeypatch):
    """Run the real supervisor around a worker that SIGKILLs itself mid-submit."""
    leases = tmp_path / "leases"
    monkeypatch.setattr(runpod, "DEFAULT_RUNPOD_LEASE_DIR", leases)
    script = tmp_path / "killed_worker.py"
    script.write_text(CHILD)
    real_popen = subprocess.Popen

    def supervise(stage, api):
        run, _ = make_run()
        monkeypatch.setattr(runpod.RunPodExecutor, "_require_api", lambda self: api)

        def spawn(_command, **kwargs):
            kwargs["env"] = dict(kwargs["env"], PYTHONPATH=os.pathsep.join(sys.path))
            return real_popen(
                [
                    sys.executable,
                    str(script),
                    str(run),
                    str(leases),
                    JOB_ID,
                    POD_ID,
                    stage,
                ],
                **kwargs,
            )

        monkeypatch.setattr(worker.subprocess, "Popen", spawn)
        assert worker.supervise(run) == -signal.SIGKILL
        return run

    return supervise, leases


def test_supervisor_terminates_the_pod_a_killed_worker_left_billing(killed_worker):
    supervise, leases = killed_worker
    api = Api()
    run = supervise("allocated", api)
    assert api.terminated == [POD_ID]
    assert not (leases / f"{POD_ID}.json").exists()
    report = submission.inspect_submission(run)
    assert [entry["state"] for entry in report["resources"]] == ["cleanup_acknowledged"]
    assert workspace.read_json(run / "state.json")["status"] == "failed"
    assert workspace.read_json(run / "failure.json")["code"] == "worker_exit"


def test_pod_created_before_its_identity_was_journaled_is_found_by_name(
    killed_worker,
):
    supervise, _ = killed_worker
    api = Api(
        pods=[
            {"id": "pod-late", "name": "stateset-sft-" + JOB_ID},
            {"id": "someone-elses", "name": "stateset-sft-otherjob"},
        ]
    )
    supervise("intent", api)
    assert api.terminated == ["pod-late"]


def test_unconfirmed_reclamation_is_journaled_and_keeps_the_lease(killed_worker):
    supervise, leases = killed_worker
    api = Api(fail=True)
    run = supervise("allocated", api)
    assert (leases / f"{POD_ID}.json").exists()
    report = submission.inspect_submission(run)
    assert report["resources"][0]["state"] == "cleanup_unconfirmed"
    assert report["cleanup_required_or_unconfirmed"] is True
    assert workspace.read_json(run / "state.json")["status"] == "failed"


def test_cleaned_up_submission_needs_no_provider_call(make_run, monkeypatch):
    run, _ = make_run()
    journal = submission.SubmissionJournal(run, "runpod", durable_supported=True)
    base = {
        "schema_version": 1,
        "provider": "runpod",
        "job_id": JOB_ID,
        "resource_kind": "pod",
        "resource_name": "stateset-sft-" + JOB_ID,
    }
    journal.publish({**base, "event": "allocation_intent"})
    journal.publish({**base, "event": "allocated", "resource_id": POD_ID})
    journal.publish({**base, "event": "cleanup_acknowledged", "resource_id": POD_ID})
    monkeypatch.setattr(
        runpod.RunPodExecutor,
        "_require_api",
        lambda self: pytest.fail("no provider call is needed"),
    )
    assert submission.reclaim_abandoned_pods(run) == {"reclaimed": [], "failed": {}}


WORKER_MAIN = textwrap.dedent("""
    import sys, time
    from stateset_agents import finetune_worker as worker

    def execute(run, supervisor_token=None):
        try:
            print("training", flush=True)
            time.sleep(60)
        finally:
            print("cleanup ran", flush=True)

    worker.execute = execute
    sys.argv = ["finetune_worker", "--execute", sys.argv[1]]
    worker.main()
    """)


@pytest.mark.skipif(os.name == "nt", reason="POSIX termination signals")
@pytest.mark.parametrize("signum", POSIX_TERMINATION_SIGNALS)
def test_signalled_worker_runs_its_cleanup_before_exiting(tmp_path, signum):
    script = tmp_path / "worker_main.py"
    script.write_text(WORKER_MAIN)
    child = subprocess.Popen(
        [sys.executable, str(script), str(tmp_path)],
        stdout=subprocess.PIPE,
        text=True,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
    )
    try:
        assert child.stdout.readline().strip() == "training"
        child.send_signal(signum)
        output, _ = child.communicate(timeout=30)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    assert "cleanup ran" in output
    assert child.returncode == 128 + signum
