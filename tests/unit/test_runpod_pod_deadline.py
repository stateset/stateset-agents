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
