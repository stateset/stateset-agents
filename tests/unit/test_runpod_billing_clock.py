"""RunPod deadlines and spend keep counting while the host is suspended."""

from __future__ import annotations

import time

import pytest

from stateset_agents.remote import runpod
from stateset_agents.remote.job import RemoteJobSpec
from stateset_agents.remote.ledger import read_entries


class SuspendableClock:
    """Monotonic time pauses during a suspend; wall time keeps going."""

    def __init__(self):
        self.mono, self.wall = 100.0, 1_700_000_000.0

    def monotonic(self):
        return self.mono

    def time(self):
        return self.wall

    def sleep(self, seconds):
        self.advance(seconds)

    def advance(self, seconds):
        self.mono += seconds
        self.wall += seconds

    def suspend(self, seconds):
        self.wall += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = SuspendableClock()
    monkeypatch.setattr(runpod, "time", fake)
    return fake


def test_clock_counts_suspends_and_never_runs_backwards(clock):
    start = runpod._billing_now()
    clock.advance(10)
    clock.suspend(3600)
    assert runpod._billing_now() - start == pytest.approx(3610)
    clock.wall -= 7200  # wall clock stepped back by NTP
    assert runpod._billing_now() - start == pytest.approx(3610)
    clock.advance(5)
    assert runpod._billing_now() - start == pytest.approx(3615)


def test_scheduling_jitter_is_not_mistaken_for_a_suspend(clock):
    start = runpod._billing_now()
    clock.wall += 0.5
    assert runpod._billing_now() == start


@pytest.mark.skipif(
    not hasattr(time, "CLOCK_BOOTTIME"), reason="CLOCK_BOOTTIME is Linux-only"
)
def test_linux_uses_boottime():
    before = time.clock_gettime(time.CLOCK_BOOTTIME)
    now = runpod._billing_now()
    assert before <= now <= time.clock_gettime(time.CLOCK_BOOTTIME)


def test_suspend_mid_job_is_billed_and_rents_no_replacement(clock, tmp_path):
    """A lid closed for an hour used to look like one minute of spend."""

    class Api:
        def __init__(self):
            self.created, self.terminated = [], []

        def create_pod(self, **kwargs):
            self.created.append(f"pod-{len(self.created) + 1}")
            return {"id": self.created[-1], "costPerHr": 1.0}

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
                clock.advance(60)
                clock.suspend(3600)  # the laptop sleeps; the pod keeps billing
                return 255, "client_loop: send disconnect: Broken pipe"
            return 0, ""

    data = tmp_path / "train.jsonl"
    data.write_text('{"messages": []}\n')
    api = Api()
    executor = runpod.RunPodExecutor(
        api=api,
        ssh=Ssh(),
        public_key="ssh-rsa TEST",
        lease_dir=tmp_path / "leases",
        ledger_path=tmp_path / "ledger.jsonl",
        poll_interval_s=0,
        allow_environment_wheel=False,
    )
    with pytest.raises(runpod.RemoteExecutionError, match="time limit"):
        executor.submit(
            RemoteJobSpec(
                dataset=data,
                base_model="example/model",
                output_dir=tmp_path / "out",
                timeout_s=3600,
                max_cost_usd=2.0,
            )
        )
    assert api.created == api.terminated == ["pod-1"]
    (row,) = read_entries(tmp_path / "ledger.jsonl")
    assert row["duration_s"] >= 3660 and row["cost_usd"] >= 1.0
