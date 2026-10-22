"""A stopped or vanished submitter must not leave its RunPod pod billing."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from stateset_agents.remote.runpod import RunPodExecutor

CHILD = textwrap.dedent("""
    import sys, time
    from pathlib import Path
    from stateset_agents.remote.job import RemoteJobSpec
    from stateset_agents.remote.runpod import RunPodExecutor

    root = Path(sys.argv[1])

    class Api:
        def create_pod(self, **kw):
            return {"id": "pod-1", "costPerHr": 1.0}

        def get_pod(self, pod_id):
            return {"desiredStatus": "RUNNING", "publicIp": "1.2.3.4",
                    "portMappings": {"22": 22}}

        def terminate_pod(self, pod_id):
            (root / "terminated").write_text(pod_id)

    class Ssh:
        def wait_until_reachable(self, *a): pass
        def upload(self, *a): pass
        def download_dir(self, *a): return []
        def run(self, command, **kw):
            print("training", flush=True)
            time.sleep(60)
            return 0, ""

    (root / "data.jsonl").write_text('{"messages":[]}\\n')
    executor = RunPodExecutor(
        api=Api(), ssh=Ssh(), public_key="ssh-ed25519 AAAA x",
        lease_dir=root / "leases", ledger_path=root / "ledger.jsonl",
        allow_environment_wheel=False,
    )
    executor.submit(RemoteJobSpec(
        dataset=root / "data.jsonl", base_model="example/model",
        output_dir=root / "out", max_cost_usd=5.0, timeout_s=3600,
    ))
    """)

# SIGHUP does not exist on Windows; parametrize is evaluated at import time.
POSIX_TERMINATION_SIGNALS = [signal.SIGTERM] + (
    [signal.SIGHUP] if hasattr(signal, "SIGHUP") else []
)


@pytest.mark.skipif(os.name == "nt", reason="POSIX termination signals")
@pytest.mark.parametrize("signum", POSIX_TERMINATION_SIGNALS)
def test_signalled_submitter_terminates_its_pod_and_clears_the_lease(tmp_path, signum):
    """SIGTERM/SIGHUP mid-job used to exit without terminating the pod."""
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    child = subprocess.Popen(
        [sys.executable, str(script), str(tmp_path)],
        stdout=subprocess.PIPE,
        text=True,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
    )
    try:
        assert child.stdout.readline().strip() == "training"
        child.send_signal(signum)
        child.wait(timeout=30)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    assert child.returncode == 128 + signum
    assert (tmp_path / "terminated").read_text() == "pod-1"
    assert list((tmp_path / "leases").glob("*.json")) == []


def test_application_signal_handlers_are_left_alone(monkeypatch):
    from stateset_agents.remote import runpod

    def custom(*_args):
        return None

    previous = signal.signal(signal.SIGTERM, custom)
    try:
        with runpod._signals_end_submission():
            assert signal.getsignal(signal.SIGTERM) is custom
        assert signal.getsignal(signal.SIGTERM) is custom
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_default_handlers_are_restored_after_submission():
    from stateset_agents.remote import runpod

    previous = signal.signal(signal.SIGTERM, signal.SIG_DFL)
    try:
        with runpod._signals_end_submission():
            assert signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
    finally:
        signal.signal(signal.SIGTERM, previous)


class _DeleteRecorder:
    """A local stand-in for the RunPod REST endpoint."""

    def __enter__(self):
        import http.server
        import threading

        requests_seen = self.requests = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_DELETE(self):  # noqa: N802 - http.server API
                requests_seen.append((self.path, self.headers.get("Authorization")))
                self.send_response(200)
                self.end_headers()

            def log_message(self, *_args):
                pass

        self.server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.root = f"http://127.0.0.1:{self.server.server_port}/v1"
        return self

    def __exit__(self, *_exc):
        self.server.shutdown()


def _run_watchdog(root, env, expected="pod-1"):
    from stateset_agents.remote.runpod import _POD_WATCHDOG

    return subprocess.run(
        [sys.executable, "-c", _POD_WATCHDOG, "0", root, expected],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


@pytest.mark.skipif(os.name == "nt", reason="the watchdog forks on the pod (Linux)")
def test_pod_deadline_deletes_its_own_pod_with_the_pods_scoped_key():
    with _DeleteRecorder() as endpoint:
        env = dict(os.environ, RUNPOD_POD_ID="pod-1", RUNPOD_API_KEY="pod-scoped")
        result = _run_watchdog(endpoint.root, env)
        assert result.stdout.strip() == "pod deadline armed"
        for _ in range(200):
            if endpoint.requests:
                break
            time.sleep(0.05)
    assert endpoint.requests == [("/v1/pods/pod-1", "Bearer pod-scoped")]


@pytest.mark.skipif(os.name == "nt", reason="the watchdog forks on the pod (Linux)")
@pytest.mark.parametrize("pod", ["", "another-pod"])
def test_pod_deadline_never_arms_for_missing_or_foreign_identity(pod):
    with _DeleteRecorder() as endpoint:
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in {"RUNPOD_POD_ID", "RUNPOD_API_KEY"}
        }
        if pod:
            env.update(RUNPOD_POD_ID=pod, RUNPOD_API_KEY="pod-scoped")
        result = _run_watchdog(endpoint.root, env)
        time.sleep(0.2)
    assert result.stdout.strip() == "pod deadline unavailable"
    assert endpoint.requests == []


@pytest.mark.parametrize(
    "output", ["pod deadline armed\n", "pod deadline unavailable\n"]
)
def test_executor_arms_the_pod_deadline_at_the_run_deadline(
    tmp_path, monkeypatch, output
):
    from types import SimpleNamespace

    from stateset_agents.remote import runpod
    from stateset_agents.remote.job import RemoteJobSpec
    from tests.unit.test_remote_runpod_executor import FakePodApi
    from tests.unit.test_runpod_transport_timeouts import Clock

    clock, commands = Clock(), []
    monkeypatch.setattr(runpod, "time", clock)

    def execute(argv, **kwargs):
        commands.append(argv[-1])
        clock.value += 1
        watchdog = "pod deadline" in argv[-1]
        return SimpleNamespace(
            returncode=0, stdout=output if watchdog else "", stderr=""
        )

    monkeypatch.setattr(runpod.subprocess, "run", execute)
    data = tmp_path / "train.jsonl"
    data.write_text('{"messages":[]}\n')
    executor = RunPodExecutor(
        api=FakePodApi(ready_after=1),
        ssh=runpod.SshTransport(),
        public_key="ssh-rsa TEST",
        lease_dir=tmp_path / "leases",
        ledger_path=tmp_path / "ledger.jsonl",
        poll_interval_s=0,
    )
    handle = executor.submit(
        RemoteJobSpec(
            dataset=data,
            base_model="example/model",
            output_dir=tmp_path / "out",
            timeout_s=600,
            dry_run=True,
        )
    )
    import shlex

    armed = [shlex.split(command) for command in commands if "pod deadline" in command]
    assert len(armed) == 1
    # python3 -c <script> <delay> <api root> <pod id>; no credential on the line
    assert armed[0][3:] == ["599", runpod._API_ROOT, "pod-abc"]
    logs = "\n".join(executor.logs(handle))
    assert ("deadline armed" in logs) == ("armed" in output)
    assert executor.status(handle).value == "succeeded"


def test_deadline_is_armed_at_pod_creation_for_runpod_images(tmp_path):
    """A host crash during provisioning must not leave the pod billing forever."""
    import shlex

    from stateset_agents.remote import runpod
    from stateset_agents.remote.job import RemoteJobSpec
    from tests.unit.test_remote_runpod_executor import FakePodApi, FakeSsh

    data = tmp_path / "train.jsonl"
    data.write_text('{"messages":[]}\n')
    api = FakePodApi(ready_after=1)
    executor = RunPodExecutor(
        api=api,
        ssh=FakeSsh(),
        public_key="ssh-rsa TEST",
        lease_dir=tmp_path / "leases",
        ledger_path=tmp_path / "ledger.jsonl",
        poll_interval_s=0,
    )
    before = time.time()
    executor.submit(
        RemoteJobSpec(
            dataset=data,
            base_model="example/model",
            output_dir=tmp_path / "out",
            timeout_s=1800,
        )
    )
    created = api.created[0]
    assert created["docker_entrypoint"] == ["/bin/bash", "-c"]
    (command,) = created["docker_start_cmd"]
    assert command.endswith("& exec /start.sh")
    words = shlex.split(command.split(" >/tmp/stateset-deadline.log")[0])
    assert words[:2] == ["python3", "-c"] and words[2] == runpod._POD_WATCHDOG
    deadline = float(words[3].lstrip("@"))
    assert before + 1800 - 5 <= deadline <= time.time() + 1800 + 5
    assert words[4:] == [runpod._API_ROOT, "*"]
    # No user credential travels with the pod; it uses RunPod's injected key.
    assert set(created["env"]) == {"PUBLIC_KEY", "SSH_PUBLIC_KEY"}


def test_custom_images_keep_their_own_start_command(tmp_path):
    from stateset_agents.remote.job import RemoteJobSpec
    from tests.unit.test_remote_runpod_executor import FakePodApi, FakeSsh

    data = tmp_path / "train.jsonl"
    data.write_text('{"messages":[]}\n')
    api = FakePodApi(ready_after=1)
    RunPodExecutor(
        api=api,
        ssh=FakeSsh(),
        public_key="ssh-rsa TEST",
        image="example/custom:1",
        lease_dir=tmp_path / "leases",
        ledger_path=tmp_path / "ledger.jsonl",
        poll_interval_s=0,
    ).submit(RemoteJobSpec(dataset=data, base_model="m", output_dir=tmp_path / "out"))
    assert "docker_start_cmd" not in api.created[0]


@pytest.mark.skipif(os.name == "nt", reason="the watchdog forks on the pod (Linux)")
def test_absolute_deadline_fires_for_any_own_pod_identity():
    with _DeleteRecorder() as endpoint:
        env = dict(os.environ, RUNPOD_POD_ID="pod-9", RUNPOD_API_KEY="pod-scoped")
        from stateset_agents.remote.runpod import _POD_WATCHDOG

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                _POD_WATCHDOG,
                f"@{time.time():.0f}",
                endpoint.root,
                "*",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        assert result.stdout.strip() == "pod deadline armed"
        for _ in range(200):
            if endpoint.requests:
                break
            time.sleep(0.05)
    assert endpoint.requests == [("/v1/pods/pod-9", "Bearer pod-scoped")]


def test_submission_first_sweeps_pods_left_by_exited_submitters(tmp_path):
    import json
    import socket

    from stateset_agents.remote.job import RemoteJobSpec
    from tests.unit.test_remote_runpod_executor import FakePodApi, FakeSsh
    from tests.unit.test_runpod_pod_reclamation import dead_owner

    leases = tmp_path / "leases"
    leases.mkdir()
    pid, token = dead_owner()
    (leases / "crashed-pod.json").write_text(
        json.dumps(
            {
                "provider": "runpod",
                "pod_id": "crashed-pod",
                "owner_pid": pid,
                "owner_token": token,
                "owner_host": socket.gethostname(),
            }
        )
    )
    (leases / "legacy-pod.json").write_text(
        json.dumps({"provider": "runpod", "pod_id": "legacy-pod"})
    )
    data = tmp_path / "train.jsonl"
    data.write_text('{"messages":[]}\n')
    api = FakePodApi(ready_after=1)
    executor = RunPodExecutor(
        api=api,
        ssh=FakeSsh(),
        public_key="ssh-rsa TEST",
        lease_dir=leases,
        ledger_path=tmp_path / "ledger.jsonl",
        poll_interval_s=0,
    )
    handle = executor.submit(
        RemoteJobSpec(dataset=data, base_model="m", output_dir=tmp_path / "out")
    )
    assert api.terminated[0] == "crashed-pod" and "legacy-pod" not in api.terminated
    assert any("crashed-pod" in line for line in executor.logs(handle))
    assert sorted(p.stem for p in leases.glob("*.json")) == ["legacy-pod"]


def test_studio_start_sweeps_only_with_a_runpod_key(tmp_path, monkeypatch):
    from stateset_agents.remote import runpod
    from stateset_agents.tui import Studio

    calls = []
    monkeypatch.setattr(
        runpod.RunPodExecutor,
        "sweep_abandoned_pods",
        lambda self: calls.append(1) or (["crashed-pod"], {}),
    )
    studio = Studio(tmp_path)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    assert studio.start_lease_sweep() is False
    monkeypatch.setenv("RUNPOD_API_KEY", "synthetic")
    assert studio.start_lease_sweep() is True
    deadline = time.monotonic() + 5
    while studio.lease_sweep.busy and time.monotonic() < deadline:
        studio.refresh(history=False)
        time.sleep(0.01)
    studio.refresh(history=False)
    assert calls == [1] and "Ended 1 RunPod pod(s)" in studio.message
