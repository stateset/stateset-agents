"""Pods are reclaimed per lease, and a stopped submitter still terminates its pod."""

from __future__ import annotations

import json

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


def _leased(tmp_path, *pod_ids):
    lease_dir = tmp_path / "leases"
    lease_dir.mkdir()
    for pod_id in pod_ids:
        (lease_dir / f"{pod_id}.json").write_text(
            json.dumps({"provider": "runpod", "pod_id": pod_id})
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
    assert executor.orphaned_leases() == []


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
    assert executor.orphaned_leases() == []
