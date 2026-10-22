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


def _run_budget_fixture(tmp_path):
    from stateset_agents.remote.job import RemoteJobSpec

    data = tmp_path / "train.jsonl"
    data.write_text(
        '{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\n'
    )
    return RemoteJobSpec(
        dataset=data,
        base_model="example/model",
        output_dir=tmp_path / "out",
        timeout_s=3600,
        max_cost_usd=1.0,
    )


class PricedPods:
    """Pods that become reachable at once, each quoted from ``prices``."""

    def __init__(self, *prices):
        self.prices, self.created, self.terminated, self.polls = list(prices), [], [], 0

    def create_pod(self, **kwargs):
        self.created.append(f"pod-{len(self.created) + 1}")
        return {"id": self.created[-1], "costPerHr": self.prices[len(self.created) - 1]}

    def get_pod(self, pod_id):
        self.polls += 1
        return {
            "desiredStatus": "RUNNING",
            "publicIp": "1.2.3.4",
            "portMappings": {"22": 40022},
        }

    def terminate_pod(self, pod_id):
        self.terminated.append(pod_id)


def test_replacement_pod_shares_the_run_deadline_and_max_cost(monkeypatch, tmp_path):
    """A pod dying mid-job must not buy its replacement a fresh ``timeout_s``.

    Training needs 3500s; the first pod dies (ssh 255) after 1800s. Before the
    run-level deadline, the replacement got a full new hour and the run spent
    ~$1.47 against a $1.00 ceiling.
    """
    from stateset_agents.remote.ledger import read_entries

    clock, training_limits = Clock(), []
    monkeypatch.setattr(runpod, "time", clock)

    def execute(argv, **kwargs):
        if "training.sft" in argv[-1]:
            training_limits.append(kwargs["timeout"])
            if len(training_limits) == 1:
                clock.value += 1800
                return SimpleNamespace(returncode=255, stdout="", stderr="")
            if kwargs["timeout"] < 3500:
                clock.value += kwargs["timeout"]
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
            clock.value += 3500
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        clock.value += 1
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runpod.subprocess, "run", execute)
    api = PricedPods(1.0, 1.0)
    executor = runpod.RunPodExecutor(
        api=api,
        ssh=runpod.SshTransport(),
        public_key="ssh-rsa TEST",
        lease_dir=tmp_path / "leases",
        ledger_path=tmp_path / "ledger.jsonl",
        poll_interval_s=0,
    )
    spec = _run_budget_fixture(tmp_path)
    with pytest.raises(runpod.RemoteExecutionError, match="remote outcome is unknown"):
        executor.submit(spec)

    assert api.created == api.terminated == ["pod-1", "pod-2"]
    # The replacement only gets what is left of the single run deadline.
    assert training_limits[1] <= 3600 - 1800
    assert clock.value <= 3600
    spent = sum(row["cost_usd"] for row in read_entries(tmp_path / "ledger.jsonl"))
    assert spent <= spec.max_cost_usd


def test_endpoint_wait_is_bounded_by_the_remaining_run_deadline(monkeypatch, tmp_path):
    """``ready_timeout_s`` must not extend a replacement past the run deadline."""
    clock = Clock()
    monkeypatch.setattr(runpod, "time", clock)
    api = PricedPods(1.0, 1.0)
    pending = {"desiredStatus": "PENDING", "publicIp": None}
    reachable = api.get_pod

    def get_pod(pod_id):
        return reachable(pod_id) if pod_id == "pod-1" else pending

    api.get_pod = get_pod

    class DyingTransport:
        def wait_until_reachable(self, host, port, timeout_s):
            pass

        def upload(self, local, remote):
            pass

        def run(self, command):
            if "training.sft" in command:
                clock.value += 3500
                return 255, "client_loop: send disconnect: Broken pipe"
            return 0, ""

    executor = runpod.RunPodExecutor(
        api=api,
        ssh=DyingTransport(),
        public_key="ssh-rsa TEST",
        lease_dir=tmp_path / "leases",
        ledger_path=tmp_path / "ledger.jsonl",
        poll_interval_s=7,
        ready_timeout_s=600,
    )
    spec = _run_budget_fixture(tmp_path)
    spec.max_cost_usd = None  # isolate the deadline from the spending ceiling
    with pytest.raises(runpod.RemoteExecutionError, match="never became reachable"):
        executor.submit(spec)
    assert api.created == api.terminated == ["pod-1", "pod-2"]
    # 100s of the run were left; the old per-pod wait ran 600s past them.
    assert clock.value <= 3600


def test_no_replacement_once_the_run_deadline_has_passed(monkeypatch, tmp_path):
    clock = Clock()
    monkeypatch.setattr(runpod, "time", clock)
    api = PricedPods(1.0, 1.0)

    class OverrunTransport:
        def wait_until_reachable(self, host, port, timeout_s):
            pass

        def upload(self, local, remote):
            pass

        def run(self, command):
            if "training.sft" in command:
                clock.value += 3700
                return 255, ""
            return 0, ""

    executor = runpod.RunPodExecutor(
        api=api,
        ssh=OverrunTransport(),
        public_key="ssh-rsa TEST",
        lease_dir=tmp_path / "leases",
        ledger_path=tmp_path / "ledger.jsonl",
        poll_interval_s=0,
    )
    with pytest.raises(runpod.RemoteExecutionError, match="time limit") as caught:
        executor.submit(_run_budget_fixture(tmp_path))
    assert caught.value.context.details["stage"] == "deadline"
    assert api.created == api.terminated == ["pod-1"]


def test_replacement_budget_counts_what_earlier_pods_already_spent(
    monkeypatch, tmp_path
):
    """Spend so far plus the replacement's worst case must fit ``max_cost_usd``."""
    clock = Clock()
    monkeypatch.setattr(runpod, "time", clock)
    api = PricedPods(0.5, 2.0)

    trainings = []

    class DyingTransport:
        def wait_until_reachable(self, host, port, timeout_s):
            pass

        def upload(self, local, remote):
            pass

        def run(self, command):
            if "training.sft" in command:
                trainings.append(command)
                clock.value += 1800
                return 255, ""
            return 0, ""

    executor = runpod.RunPodExecutor(
        api=api,
        ssh=DyingTransport(),
        public_key="ssh-rsa TEST",
        lease_dir=tmp_path / "leases",
        ledger_path=tmp_path / "ledger.jsonl",
        poll_interval_s=0,
    )
    with pytest.raises(runpod.RemoteExecutionError, match=r"\$0\.25 already spent"):
        executor.submit(_run_budget_fixture(tmp_path))
    # The pricier replacement is terminated before any work runs on it.
    assert api.created == api.terminated == ["pod-1", "pod-2"]
    assert len(trainings) == 1
