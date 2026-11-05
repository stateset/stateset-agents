"""Pods are reclaimed per lease, and a stopped submitter still terminates its pod."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import requests
from typer.testing import CliRunner

from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.remote.runpod import RunPodApi, RunPodExecutor


def _http_error(status):
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(f"HTTP {status}", response=response)


class LeaseApi:
    def __init__(self, failures):
        self.failures, self.calls = failures, []

    def terminate_pod(self, pod_id):
        self.calls.append(pod_id)
        if pod_id in self.failures:
            raise self.failures[pod_id]


def dead_owner():
    """The PID and start token of a process that has since exited.

    Without a start token (Windows without psutil) only the PID identifies
    the owner, and Windows hands exited PIDs out again quickly; a reused PID
    is rightly reported ``alive``. Retry until the PID is verifiably free.
    """
    from stateset_agents.utils.process_probe import probe_process, process_start_token

    if sys.platform == "win32" and process_start_token(os.getpid()) is None:
        # Windows PIDs are multiples of 4, so this one is never allocated: it
        # proves "no such process" without racing a reused exited PID.
        return 0xFFFFFFFB, None
    for _ in range(20):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        token = process_start_token(child.pid)
        child.kill()
        child.wait()
        if token is not None:
            return child.pid, token
        try:
            probe_process(child.pid)
        except ProcessLookupError:
            return child.pid, token
    pytest.fail("every exited child PID was reused immediately")


def _leased(tmp_path, *pod_ids, owner=None):
    import socket

    pid, token = owner or dead_owner()
    lease_dir = tmp_path / "leases"
    lease_dir.mkdir()
    for pod_id in pod_ids:
        (lease_dir / f"{pod_id}.json").write_text(
            json.dumps(
                {
                    "provider": "runpod",
                    "pod_id": pod_id,
                    "owner_pid": pid,
                    "owner_token": token,
                    "owner_host": socket.gethostname(),
                }
            )
        )
    return lease_dir


def test_cleanup_attempts_every_lease_and_treats_404_as_already_gone(tmp_path):
    lease_dir = _leased(tmp_path, "aaa-gone", "bbb-billing", "ccc-billing")
    api = LeaseApi({"aaa-gone": _http_error(404)})
    executor = RunPodExecutor(
        api=api, lease_dir=lease_dir, allow_environment_wheel=False
    )
    assert executor.cleanup_orphans() == ["aaa-gone", "bbb-billing", "ccc-billing"]
    assert api.calls == ["aaa-gone", "bbb-billing", "ccc-billing"]
    assert executor.leases() == []


def test_failures_are_reported_together_and_keep_only_their_leases(tmp_path):
    lease_dir = _leased(tmp_path, "aaa-down", "bbb-billing", "ccc-denied")
    api = LeaseApi(
        {"aaa-down": _http_error(503), "ccc-denied": requests.ConnectionError()}
    )
    executor = RunPodExecutor(
        api=api, lease_dir=lease_dir, allow_environment_wheel=False
    )
    with pytest.raises(RemoteExecutionError) as caught:
        executor.cleanup_orphans()
    message = str(caught.value)
    assert "aaa-down: HTTPError (HTTP 503)" in message
    assert "ccc-denied: ConnectionError" in message
    assert "terminated bbb-billing" in message
    assert api.calls == ["aaa-down", "bbb-billing", "ccc-denied"]
    assert sorted(p.stem for p in lease_dir.glob("*.json")) == [
        "aaa-down",
        "ccc-denied",
    ]


def test_orphans_command_reports_each_pod_and_fails_on_any_failure(
    tmp_path, monkeypatch
):
    import stateset_agents.cli_remote as cli_remote
    from stateset_agents.cli_remote import app

    lease_dir = _leased(tmp_path, "aaa-down", "bbb-billing")
    api = LeaseApi({"aaa-down": _http_error(500)})
    executor = RunPodExecutor(
        api=api, lease_dir=lease_dir, allow_environment_wheel=False
    )
    monkeypatch.setattr(cli_remote, "get_executor", lambda provider: executor)
    result = CliRunner().invoke(app, ["runpod-orphans", "--terminate"])
    assert result.exit_code == 1
    assert "Terminated bbb-billing" in result.output
    assert "aaa-down" in result.output and "lease kept" in result.output


def test_termination_retries_transient_provider_errors(monkeypatch):
    calls, sleeps = [], []

    class Response:
        def __init__(self, status):
            self.status_code = status

        def raise_for_status(self):
            if self.status_code >= 400:
                raise _http_error(self.status_code)

    statuses = iter([503, 429, 200])
    monkeypatch.setattr(
        requests,
        "delete",
        lambda url, **kwargs: calls.append(url) or Response(next(statuses)),
    )
    monkeypatch.setattr("stateset_agents.remote.runpod.time.sleep", sleeps.append)
    RunPodApi("key").terminate_pod("pod-1")
    assert len(calls) == 3 and sleeps == [2.0, 4.0]


def test_submission_cleanup_counts_an_already_deleted_pod_as_acknowledged(tmp_path):
    from stateset_agents.remote.job import RemoteJobSpec
    from tests.unit.test_remote_runpod_executor import FakePodApi, FakeSsh

    data = tmp_path / "train.jsonl"
    data.write_text('{"messages":[]}\n')
    api = FakePodApi(ready_after=1)

    def gone(pod_id):
        api.terminated.append(pod_id)
        raise _http_error(404)

    api.terminate_pod = gone
    executor = RunPodExecutor(
        api=api,
        ssh=FakeSsh(),
        public_key="ssh-rsa TEST",
        lease_dir=tmp_path / "leases",
        ledger_path=tmp_path / "ledger.jsonl",
        poll_interval_s=0,
    )
    handle = executor.submit(
        RemoteJobSpec(
            dataset=data, base_model="example/model", output_dir=tmp_path / "out"
        )
    )
    assert executor.status(handle).value == "succeeded"
    assert executor.leases() == []


def test_cleanup_never_reclaims_a_live_jobs_pod(tmp_path):
    """Reclaiming every lease used to end a running job's pod (double billing)."""
    import threading

    from stateset_agents.remote.job import RemoteJobSpec

    training, release = threading.Event(), threading.Event()

    class Api:
        def __init__(self):
            self.created, self.terminated = 0, []

        def create_pod(self, **kwargs):
            self.created += 1
            return {"id": f"livepod{self.created}", "costPerHr": 0.5}

        def get_pod(self, pod_id):
            return {
                "desiredStatus": "RUNNING",
                "publicIp": "1.2.3.4",
                "portMappings": {"22": 22},
            }

        def terminate_pod(self, pod_id):
            self.terminated.append(pod_id)

    class Ssh:
        def wait_until_reachable(self, *args):
            pass

        def upload(self, *args):
            pass

        def run(self, command, timeout_s=None):
            if "training.sft" in command:
                training.set()
                assert release.wait(10)
                return (255, "") if api.terminated else (0, "")
            return 0, ""

        def download_dir(self, remote, local):
            Path(local).mkdir(parents=True, exist_ok=True)
            (Path(local) / "adapter_model.safetensors").write_bytes(b"x")
            return [Path(local) / "adapter_model.safetensors"]

    api, leases = Api(), tmp_path / "leases"
    data = tmp_path / "d.jsonl"
    data.write_text('{"messages": []}\n')
    live = RunPodExecutor(
        api=api,
        ssh=Ssh(),
        public_key="ssh-ed25519 AAAA",
        lease_dir=leases,
        ledger_path=tmp_path / "ledger.jsonl",
        allow_environment_wheel=False,
    )
    handles = []
    job = threading.Thread(
        target=lambda: handles.append(
            live.submit(
                RemoteJobSpec(
                    dataset=data,
                    base_model="example/model",
                    output_dir=tmp_path / "out",
                    timeout_s=600,
                )
            )
        )
    )
    job.start()
    try:
        assert training.wait(10)
        other = RunPodExecutor(api=api, lease_dir=leases, allow_environment_wheel=False)
        assert [lease["owner_state"] for lease in other.leases()] == ["alive"]
        assert other.orphaned_leases() == [] and other.cleanup_orphans() == []
    finally:
        release.set()
        job.join(10)
    assert api.created == 1 and api.terminated == ["livepod1"]
    assert live.status(handles[0]).value == "succeeded"


def test_only_provably_dead_owners_are_reclaimed_without_force(tmp_path):
    import os

    from stateset_agents.utils.process_probe import process_start_token

    lease_dir = _leased(tmp_path, "dead-1")
    (tmp_path / "live").mkdir()
    alive = _leased(
        tmp_path / "live",
        "live-1",
        owner=(os.getpid(), process_start_token(os.getpid())),
    )
    for path in alive.glob("*.json"):
        path.rename(lease_dir / path.name)
    (lease_dir / "legacy-1.json").write_text(
        json.dumps({"provider": "runpod", "pod_id": "legacy-1"})
    )
    api = LeaseApi({})
    executor = RunPodExecutor(
        api=api, lease_dir=lease_dir, allow_environment_wheel=False
    )
    states = {lease["pod_id"]: lease["owner_state"] for lease in executor.leases()}
    assert states == {"dead-1": "gone", "live-1": "alive", "legacy-1": "unknown"}
    assert sorted(lease["pod_id"] for lease in executor.orphaned_leases()) == [
        "dead-1",
        "legacy-1",
    ]
    assert executor.cleanup_orphans() == ["dead-1"]
    assert sorted(executor.cleanup_orphans(force=True)) == ["legacy-1", "live-1"]
    assert api.calls == ["dead-1", "legacy-1", "live-1"]


def test_orphans_command_keeps_live_pods_unless_forced(tmp_path, monkeypatch):
    import os

    import stateset_agents.cli_remote as cli_remote
    from stateset_agents.cli_remote import app
    from stateset_agents.utils.process_probe import process_start_token

    lease_dir = _leased(
        tmp_path, "live-1", owner=(os.getpid(), process_start_token(os.getpid()))
    )
    api = LeaseApi({})
    executor = RunPodExecutor(
        api=api, lease_dir=lease_dir, allow_environment_wheel=False
    )
    monkeypatch.setattr(cli_remote, "get_executor", lambda provider: executor)
    result = CliRunner().invoke(app, ["runpod-orphans", "--terminate"])
    assert result.exit_code == 0 and api.calls == []
    assert "submitting process still running" in result.output
    assert "use --force" in result.output
    result = CliRunner().invoke(app, ["runpod-orphans", "--terminate", "--force"])
    assert result.exit_code == 0 and api.calls == ["live-1"]


def test_leases_record_their_owner_process(tmp_path):
    import os
    import socket

    from stateset_agents.remote.job import RemoteJobSpec
    from tests.unit.test_remote_runpod_executor import FakePodApi, FakeSsh

    executor = RunPodExecutor(
        api=FakePodApi(), ssh=FakeSsh(), public_key="ssh-rsa T", lease_dir=tmp_path
    )
    (tmp_path / "d.jsonl").write_text('{"messages": []}\n')
    executor._write_lease(
        "pod-1",
        "job-1",
        RemoteJobSpec(
            dataset=tmp_path / "d.jsonl", base_model="m", output_dir=tmp_path
        ),
        1.0,
    )
    lease = json.loads((tmp_path / "pod-1.json").read_text())
    assert lease["owner_pid"] == os.getpid()
    assert lease["owner_host"] == socket.gethostname()
