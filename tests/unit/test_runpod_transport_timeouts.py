"""Bound client waits without exposing secrets or replaying uncertain jobs."""

from __future__ import annotations

import subprocess
import sys
import traceback
from types import SimpleNamespace

import pytest

from stateset_agents.remote import runpod


class Clock:
    def __init__(self, value=0.0):
        self.value = value

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds

    def time(self):
        return 1_700_000_000 + self.value


def connected(**kwargs):
    transport = runpod.SshTransport(**kwargs)
    transport._host, transport._port = "example.invalid", 22
    return transport


@pytest.mark.parametrize(
    "timeout", [0, -1, True, None, "10", float("nan"), float("inf"), 604801, 10**400]
)
def test_invalid_timeouts_reject_before_transport(timeout):
    with pytest.raises(ValueError, match="SSH timeout"):
        runpod.SshTransport(operation_timeout_s=timeout)


@pytest.mark.parametrize("operation", ["run", "upload", "secret", "download"])
def test_every_subprocess_has_a_finite_timeout(monkeypatch, tmp_path, operation):
    seen = []

    def execute(argv, **kwargs):
        seen.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(runpod.subprocess, "run", execute)
    transport = connected(operation_timeout_s=13)
    if operation == "run":
        assert transport.run("true") == (0, "ok")
    elif operation == "upload":
        transport.upload(tmp_path / "data.jsonl", "/workspace/data.jsonl")
    elif operation == "secret":
        transport.upload_secret("private-value", "/workspace/.key")
    else:
        transport.download_dir("/workspace/out", tmp_path / "out")
    assert len(seen) == 1 and seen[0][1]["timeout"] == 13
    assert "BatchMode=yes" in seen[0][0] and "ConnectTimeout=10" in seen[0][0]
    if operation == "secret":
        assert seen[0][1]["input"] == "private-value"
        assert "private-value" not in " ".join(seen[0][0])
    else:
        assert seen[0][1]["stdin"] == subprocess.DEVNULL


@pytest.mark.parametrize("stuck", [False, True])
def test_readiness_probes_and_backoff_respect_the_total_budget(monkeypatch, stuck):
    clock, limits = Clock(), []
    monkeypatch.setattr(runpod, "time", clock)

    def execute(argv, **kwargs):
        limits.append(kwargs["timeout"])
        if stuck:
            clock.value += kwargs["timeout"]
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return SimpleNamespace(returncode=255, stdout="", stderr="not ready")

    monkeypatch.setattr(runpod.subprocess, "run", execute)
    with pytest.raises(runpod.RemoteExecutionError, match="did not answer"):
        runpod.SshTransport().wait_until_reachable("example.invalid", 22, 12)
    assert clock.value == 12
    assert limits == ([10] if stuck else [10, 7, 2])


def test_readiness_reply_after_deadline_cannot_report_success(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(runpod, "time", clock)

    def execute(argv, **kwargs):
        clock.value += 4
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runpod.subprocess, "run", execute)
    with pytest.raises(runpod.RemoteExecutionError, match="did not answer"):
        runpod.SshTransport().wait_until_reachable("example.invalid", 22, 3)


def test_attempt_deadlines_and_endpoints_are_independent(monkeypatch):
    clock, limits = Clock(), []
    monkeypatch.setattr(runpod, "time", clock)
    base = connected(operation_timeout_s=5)
    first, second = base._for_deadline(25), base._for_deadline(45)
    first._host, first._port = "first.invalid", 1001
    second._host, second._port = "second.invalid", 1002

    def execute(argv, **kwargs):
        limits.append((argv[-2], kwargs["timeout"]))
        clock.value += 4
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runpod.subprocess, "run", execute)
    first.run("train")
    first.run("train")
    second.run("train")
    assert limits == [
        ("root@first.invalid", 25),
        ("root@first.invalid", 21),
        ("root@second.invalid", 37),
    ]
    assert base._host == "example.invalid" and base._deadline is None
    clock.value = 26
    with pytest.raises(runpod.RemoteExecutionError, match="deadline expired"):
        first.run("must not run")
    assert len(limits) == 3


@pytest.mark.parametrize("failure", ["exit", "timeout"])
def test_secret_upload_error_redacts_secret_and_remote_output(monkeypatch, failure):
    secret = "private-upload-value"

    def execute(argv, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, 1, output=secret, stderr=secret)
        return SimpleNamespace(returncode=1, stdout=secret, stderr=secret)

    monkeypatch.setattr(runpod.subprocess, "run", execute)
    with pytest.raises(runpod.RemoteExecutionError) as caught:
        connected(operation_timeout_s=1).upload_secret(secret, "/workspace/.key")
    assert secret not in "".join(traceback.format_exception(caught.value))
    assert secret not in str(caught.value)


def test_timeout_terminates_a_real_blocked_client_subprocess(monkeypatch):
    original_run, original_popen = subprocess.run, subprocess.Popen
    children = []

    def popen(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child

    def blocked_client(argv, **kwargs):
        return original_run(
            [sys.executable, "-c", "import time; time.sleep(60)"], **kwargs
        )

    monkeypatch.setattr(runpod.subprocess, "Popen", popen)
    monkeypatch.setattr(runpod.subprocess, "run", blocked_client)
    with pytest.raises(runpod.RemoteExecutionError, match="remote outcome is unknown"):
        connected(operation_timeout_s=0.1).run("never returns")
    assert len(children) == 1 and children[0].poll() is not None


def test_timeout_requests_cleanup_without_replacement_training(tmp_path):
    from stateset_agents.remote.job import RemoteJobSpec
    from tests.unit.test_remote_runpod_executor import (
        FakePodApi,
        FakeSsh,
    )

    data = tmp_path / "train.jsonl"
    data.write_text(
        '{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\n'
    )
    api, ssh, calls = FakePodApi(ready_after=1), FakeSsh(), []

    def uncertain(command):
        calls.append(command)
        raise runpod._SshTimeout("remote outcome is unknown", provider="runpod")

    ssh.run = uncertain
    executor = runpod.RunPodExecutor(
        api=api,
        ssh=ssh,
        public_key="ssh-rsa TEST",
        max_provision_attempts=3,
        lease_dir=tmp_path / "leases",
        ledger_path=tmp_path / "ledger.jsonl",
    )
    with pytest.raises(runpod.RemoteExecutionError, match="remote outcome is unknown"):
        executor.submit(
            RemoteJobSpec(
                dataset=data, base_model="example/model", output_dir=tmp_path / "out"
            )
        )
    assert len(api.created) == len(calls) == 1
    assert api.terminated == ["pod-abc"]


def test_builtin_executor_shares_one_timeout_across_setup_and_training(
    monkeypatch, tmp_path
):
    from stateset_agents.remote.job import RemoteJobSpec
    from tests.unit.test_remote_runpod_executor import FakePodApi

    clock, timeouts = Clock(), []
    monkeypatch.setattr(runpod, "time", clock)

    def execute(argv, **kwargs):
        timeouts.append(kwargs["timeout"])
        if "training.sft" in argv[-1]:
            clock.value += kwargs["timeout"]
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        clock.value += 2
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runpod.subprocess, "run", execute)
    api, transport = FakePodApi(ready_after=1), runpod.SshTransport()
    executor = runpod.RunPodExecutor(
        api=api,
        ssh=transport,
        public_key="ssh-rsa TEST",
        max_provision_attempts=3,
        lease_dir=tmp_path / "leases",
        ledger_path=tmp_path / "ledger.jsonl",
    )
    data = tmp_path / "train.jsonl"
    data.write_text(
        '{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\n'
    )
    with pytest.raises(runpod.RemoteExecutionError, match="remote outcome is unknown"):
        executor.submit(
            RemoteJobSpec(
                dataset=data,
                base_model="example/model",
                output_dir=tmp_path / "out",
                timeout_s=60,
                dry_run=True,
            )
        )
    assert timeouts == [10, 58, 56, 54]
    assert clock.value == 60
    assert len(api.created) == 1 and api.terminated == ["pod-abc"]
    assert transport._host is None and transport._deadline is None


def test_invalid_scoped_transport_still_cleans_up_its_known_allocation(tmp_path):
    from stateset_agents.remote.job import RemoteJobSpec
    from tests.unit.test_remote_runpod_executor import FakePodApi

    api, transport = FakePodApi(ready_after=1), runpod.SshTransport()
    transport.operation_timeout_s = 0
    executor = runpod.RunPodExecutor(
        api=api,
        ssh=transport,
        public_key="ssh-rsa TEST",
        max_provision_attempts=3,
        lease_dir=tmp_path / "leases",
        ledger_path=tmp_path / "ledger.jsonl",
    )
    data = tmp_path / "train.jsonl"
    data.write_text(
        '{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\n'
    )
    with pytest.raises(ValueError, match="SSH timeout"):
        executor.submit(
            RemoteJobSpec(
                dataset=data, base_model="example/model", output_dir=tmp_path / "out"
            )
        )
    assert len(api.created) == 1 and api.terminated == ["pod-abc"]
