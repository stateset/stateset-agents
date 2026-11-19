"""Tests for ``RemoteChatSession``, driven by behavioural fakes.

Same discipline as test_remote_runpod_executor.py: no network, no ssh
processes. The persistent channel is a fake process injected through the
``popen_factory`` seam, and — above all — every test that rents a pod
asserts it was terminated. A leaked pod bills by the hour.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import tarfile
from collections import deque
from pathlib import Path

import pytest

from stateset_agents.remote.chat_session import ChatSessionState, RemoteChatSession
from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.remote.runpod_recovery import recover_pods, session_lease_dir


class FakePodApi:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.terminated: list[str] = []

    def create_pod(self, **kwargs):
        self.created.append(kwargs)
        return {"id": "pod-chat", "desiredStatus": "RUNNING"}

    def get_pod(self, pod_id):
        return {
            "id": pod_id,
            "desiredStatus": "RUNNING",
            "publicIp": "1.2.3.4",
            "portMappings": {"22": 40022},
        }

    def terminate_pod(self, pod_id):
        self.terminated.append(pod_id)


class FakeSsh:
    def __init__(self, *, exit_code: int = 0):
        self.exit_code = exit_code
        self.commands: list[str] = []
        self.uploaded: list[tuple[Path, str]] = []
        self.connected_to: tuple[str, int] | None = None

    def wait_until_reachable(self, host, port, timeout_s):
        self.connected_to = (host, port)

    def upload(self, local: Path, remote: str) -> None:
        # Read the bytes so an already-deleted temp file would be caught.
        self.uploaded.append((Path(local).read_bytes() and Path(local), remote))

    def run(self, command: str) -> tuple[int, str]:
        self.commands.append(command)
        return self.exit_code, "ok"


class FakeStdin:
    def __init__(self, process: FakeChatProcess) -> None:
        self._process = process
        self.written: list[str] = []
        self.closed = False

    def write(self, text: str) -> None:
        self.written.append(text)
        self._process.on_request()

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class FakeStdout:
    def __init__(self) -> None:
        self.lines: deque[str] = deque()

    def readline(self) -> str:
        return self.lines.popleft() if self.lines else ""


class FakeChatProcess:
    """The persistent remote REPL: scripted stdout lines per stdin request."""

    def __init__(self, replies=None, startup=('{"ready": true}',)):
        self.stdout = FakeStdout()
        self.stdin = FakeStdin(self)
        self.replies = list(replies or [])
        self.killed = False
        for line in startup:
            self.stdout.lines.append(line + "\n")

    def on_request(self) -> None:
        if self.replies:
            batch = self.replies.pop(0)
            if isinstance(batch, str):
                batch = [batch]
            for line in batch:
                self.stdout.lines.append(line + "\n")

    def wait(self, timeout=None) -> int:
        return 0

    def kill(self) -> None:
        self.killed = True


@pytest.fixture
def api():
    return FakePodApi()


@pytest.fixture
def adapter_dir(tmp_path):
    directory = tmp_path / "adapter"
    directory.mkdir()
    (directory / "adapter_config.json").write_text("{}")
    (directory / "adapter_model.safetensors").write_bytes(b"WEIGHTS")
    return directory


@pytest.fixture
def make_session(api, tmp_path):
    sessions = []

    def build(process=None, ssh=None, **kwargs):
        process = process if process is not None else FakeChatProcess()
        launched: list[list[str]] = []

        def factory(cmd):
            launched.append(cmd)
            return process

        kwargs.setdefault("public_key", "ssh-rsa AAAATESTKEY")
        kwargs.setdefault("poll_interval_s", 0)
        kwargs.setdefault("lease_dir", tmp_path / "chat-leases")
        session = RemoteChatSession(
            api=api,
            ssh=ssh if ssh is not None else FakeSsh(),
            popen_factory=factory,
            **kwargs,
        )
        session.launched = launched  # test-only spy
        sessions.append(session)
        return session

    yield build
    for session in sessions:
        if session.state is ChatSessionState.READY:
            session.close()


def _crash_chat(root, boundary, ready, release):
    """Hold a real session at an allocation boundary, then exit without finalizers."""

    def crash():
        ready.set()
        os._exit(23 if release.wait(20) else 24)

    class Api(FakePodApi):
        def create_pod(self, **kwargs):
            pod = super().create_pod(**kwargs)
            (root / "provider.json").write_text(
                json.dumps({"name": kwargs["name"], "id": pod["id"]})
            )
            if boundary == "before_id":
                crash()
            return pod

    session = RemoteChatSession(
        Api(),
        FakeSsh(),
        lease_dir=root / "leases",
        public_key="fake",
        popen_factory=lambda cmd: FakeChatProcess(replies=['{"response":"hello"}']),
    )
    session.start("test/model")
    crash()


@pytest.mark.parametrize("boundary", ["before_id", "interactive"])
def test_recovery_skips_live_chat_and_cleans_it_once_after_process_exit(
    tmp_path, boundary
):
    ctx = multiprocessing.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    process = ctx.Process(target=_crash_chat, args=(tmp_path, boundary, ready, release))
    root = tmp_path / "leases"
    api = FakePodApi()
    api.list_pods = lambda: [json.loads((tmp_path / "provider.json").read_text())]
    process.start()
    try:
        assert ready.wait(15)
        rows = recover_pods(root, lambda: api, kind="chat", terminate=True)
        assert [r.status for r in rows] == ["busy"]
        assert api.terminated == []
        release.set()
        process.join(15)
        assert process.exitcode == 23
    finally:
        release.set()
        if process.is_alive():
            process.terminate()
            process.join(5)
    record = json.loads(next(root.glob("*.json")).read_text())
    assert record["pod_id"] == (None if boundary == "before_id" else "pod-chat")
    assert recover_pods(root, lambda: api, kind="chat")[0].status == "recoverable"
    assert api.terminated == []
    assert (
        recover_pods(root, lambda: api, kind="chat", terminate=True)[0].detail
        == "pod-chat"
    )
    assert recover_pods(root, lambda: api, kind="chat", terminate=True) == []
    assert api.terminated == ["pod-chat"]


def test_chat_recovery_cli_uses_chat_records(tmp_path, monkeypatch, api):
    from typer.testing import CliRunner

    from stateset_agents.cli import app
    from stateset_agents.remote.runpod_recovery import PodLease

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    name = "stateset-chat-123456789abc"
    lease = PodLease(session_lease_dir("chat") / f"{name}.json")
    with lease.starting():
        lease.record_id("pod-chat")
    monkeypatch.setattr(RemoteChatSession, "_require_api", lambda self: api)
    api.list_pods = lambda: [{"name": name, "id": "pod-chat"}]
    runner = CliRunner()
    assert (
        runner.invoke(
            app, ["runpod-orphans", "--serving", "--chat", "--terminate"]
        ).exit_code
        == 2
    )
    report = runner.invoke(app, ["runpod-orphans", "--chat"])
    assert report.exit_code == 0 and "recoverable  pod-chat" in report.output
    assert api.terminated == []
    assert (
        runner.invoke(app, ["runpod-orphans", "--chat", "--terminate"]).exit_code == 0
    )
    assert api.terminated == ["pod-chat"]


class TestStart:
    def test_missing_allocation_id_leaves_recoverable_intent(
        self, api, make_session, tmp_path
    ):
        def created_without_id(**kwargs):
            api.created.append(kwargs)
            return {"id": None}

        api.create_pod = created_without_id
        session = make_session()
        with pytest.raises(RemoteExecutionError, match="runpod-orphans --chat"):
            session.start("model")
        assert session.state is ChatSessionState.CLEANUP_PENDING
        assert api.terminated == []
        api.list_pods = lambda: [{"name": api.created[0]["name"], "id": "found-chat"}]
        results = recover_pods(
            tmp_path / "chat-leases", lambda: api, kind="chat", terminate=True
        )
        assert results[0].status == "termination accepted"
        assert api.terminated == ["found-chat"]

    def test_launches_the_repl_module_with_the_right_flags(
        self, make_session, adapter_dir
    ):
        session = make_session()
        session.start("Qwen/Qwen3.5-0.8B", adapter_dir=adapter_dir, gpu="H100")

        command = session.launched[0][-1]
        assert "python -m stateset_agents.remote.chat_repl" in command
        assert "--base-model Qwen/Qwen3.5-0.8B" in command
        assert "--adapter /workspace/adapter" in command

    def test_no_adapter_flag_without_an_adapter(self, make_session):
        session = make_session()
        session.start("Qwen/Qwen3.5-0.8B")

        assert "--adapter" not in session.launched[0][-1]

    def test_uploads_the_adapter_as_one_tarball_and_untars_it(
        self, make_session, adapter_dir
    ):
        ssh = FakeSsh()
        session = make_session(ssh=ssh)
        session.start("Qwen/Qwen3.5-0.8B", adapter_dir=adapter_dir)

        assert any(remote == "/workspace/adapter.tar.gz" for _, remote in ssh.uploaded)
        assert any("tar xzf /workspace/adapter.tar.gz" in cmd for cmd in ssh.commands)

    def test_the_tarball_actually_contains_the_adapter_files(
        self, make_session, adapter_dir, monkeypatch
    ):
        seen: list[list[str]] = []
        ssh = FakeSsh()

        original = ssh.upload

        def spy(local, remote):
            if str(remote).endswith(".tar.gz"):
                with tarfile.open(local) as tar:
                    seen.append(sorted(tar.getnames()))
            original(local, remote)

        ssh.upload = spy
        make_session(ssh=ssh).start("Qwen/Qwen3.5-0.8B", adapter_dir=adapter_dir)

        assert seen == [
            [
                "adapter",
                "adapter/adapter_config.json",
                "adapter/adapter_model.safetensors",
            ]
        ]

    def test_installs_the_pinned_published_package(self, make_session):
        ssh = FakeSsh()
        session = make_session(ssh=ssh, package_version="0.24.0")
        session.start("Qwen/Qwen3.5-0.8B")

        assert any(
            "pip install" in c and "stateset-agents[training]==0.24.0" in c
            for c in ssh.commands
        )

    def test_a_local_wheel_is_uploaded_and_installed_instead(
        self, make_session, tmp_path
    ):
        wheel = tmp_path / "stateset_agents-0.25.0-py3-none-any.whl"
        wheel.write_bytes(b"WHEELBYTES")
        ssh = FakeSsh()
        session = make_session(ssh=ssh, wheel=wheel)
        session.start("Qwen/Qwen3.5-0.8B")

        assert any(remote.endswith(wheel.name) for _, remote in ssh.uploaded)
        install = next(c for c in ssh.commands if "pip install" in c)
        assert wheel.name in install
        assert "stateset-agents[training]==" not in install

    def test_blocks_until_the_ready_line(self, make_session):
        process = FakeChatProcess(
            startup=('{"log": "loading model"}', '{"ready": true}')
        )
        session = make_session(process=process)
        session.start("Qwen/Qwen3.5-0.8B")  # would raise on EOF if ready was missed

    def test_startup_error_line_raises(self, make_session, api):
        process = FakeChatProcess(startup=('{"error": "no CUDA device"}',))

        with pytest.raises(RemoteExecutionError, match="no CUDA device"):
            make_session(process=process).start("Qwen/Qwen3.5-0.8B")

        assert api.terminated == ["pod-chat"]


class TestAsk:
    def _started(self, make_session, replies):
        session = make_session(process=FakeChatProcess(replies=replies))
        session.start("Qwen/Qwen3.5-0.8B")
        return session

    def test_writes_one_json_line_and_returns_the_response(self, make_session):
        session = self._started(make_session, ['{"response": "hello!"}'])

        assert session.ask("hi") == "hello!"
        sent = json.loads(session._process.stdin.written[0])
        assert sent == {"prompt": "hi"}

    def test_log_lines_are_skipped(self, make_session):
        session = self._started(
            make_session,
            [['{"log": "tokenizing"}', '{"log": "generating"}', '{"response": "ok"}']],
        )

        assert session.ask("hi") == "ok"

    def test_error_line_raises(self, make_session):
        session = self._started(make_session, ['{"error": "generation failed: oom"}'])

        with pytest.raises(RemoteExecutionError, match="oom"):
            session.ask("hi")

    def test_eof_raises(self, make_session):
        session = self._started(make_session, [])  # no reply queued → EOF

        with pytest.raises(RemoteExecutionError, match="ended unexpectedly"):
            session.ask("hi")

    def test_non_protocol_stdout_raises(self, make_session):
        session = self._started(make_session, ["Downloading shards: 100%"])

        with pytest.raises(RemoteExecutionError, match="non-protocol"):
            session.ask("hi")


class TestTranscript:
    """The client-side conversation mirror that feeds the improve flywheel."""

    def _started(self, make_session, replies):
        session = make_session(process=FakeChatProcess(replies=replies))
        session.start("Qwen/Qwen3.5-0.8B")
        return session

    def test_each_answered_turn_is_recorded_in_order(self, make_session):
        session = self._started(
            make_session, ['{"response": "hello!"}', '{"response": "bye!"}']
        )
        session.ask("hi")
        session.ask("later")

        assert session.transcript["messages"] == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello!"},
            {"role": "user", "content": "later"},
            {"role": "assistant", "content": "bye!"},
        ]

    def test_a_failed_turn_is_not_recorded(self, make_session):
        session = self._started(make_session, ['{"error": "generation failed: oom"}'])
        with pytest.raises(RemoteExecutionError):
            session.ask("hi")

        assert session.transcript["messages"] == []

    def test_metadata_describes_the_session(
        self, make_session, adapter_dir, monkeypatch
    ):
        monkeypatch.setattr(
            "stateset_agents.remote.chat_session.time.time", lambda: 1723500000.0
        )
        session = make_session()
        session.start("Qwen/Qwen3.5-0.8B", adapter_dir=adapter_dir, gpu="H100")

        meta = session.transcript["metadata"]
        assert meta == {
            "source": "chat-remote",
            "base_model": "Qwen/Qwen3.5-0.8B",
            "adapter": str(adapter_dir),
            "gpu": "H100",
            "started_at": 1723500000.0,
        }

    def test_transcript_survives_close(self, make_session):
        session = self._started(make_session, ['{"response": "hello!"}'])
        session.ask("hi")
        session.close()

        assert len(session.transcript["messages"]) == 2

    def test_transcript_round_trips_through_the_openai_ingest_parser(
        self, make_session, tmp_path
    ):
        """The contract behind chat -> ingest -> improve -> retrain."""
        from stateset_agents.data.trajectory_ingest import from_openai_jsonl

        session = self._started(
            make_session, ['{"response": "hello!"}', '{"response": "bye!"}']
        )
        session.ask("hi")
        session.ask("later")

        path = tmp_path / "chat.jsonl"
        path.write_text(json.dumps(session.transcript) + "\n")
        trajectories = from_openai_jsonl(path)

        assert len(trajectories) == 1
        assert [(t.role, t.content) for t in trajectories[0].turns] == [
            ("user", "hi"),
            ("assistant", "hello!"),
            ("user", "later"),
            ("assistant", "bye!"),
        ]
        assert trajectories[0].metadata["metadata"]["source"] == "chat-remote"


class TestClose:
    """A leaked pod bills by the hour. It must die exactly once, always."""

    def test_close_terminates_the_pod(self, make_session, api):
        session = make_session()
        session.start("Qwen/Qwen3.5-0.8B")
        session.close()

        assert api.terminated == ["pod-chat"]

    def test_close_is_idempotent(self, make_session, api):
        session = make_session()
        session.start("Qwen/Qwen3.5-0.8B")
        session.close()
        session.close()

        assert api.terminated == ["pod-chat"]

    def test_close_after_a_protocol_error_still_terminates_once(
        self, make_session, api
    ):
        session = make_session(process=FakeChatProcess(replies=[]))
        session.start("Qwen/Qwen3.5-0.8B")
        with pytest.raises(RemoteExecutionError):
            session.ask("hi")
        session.close()
        session.close()

        assert api.terminated == ["pod-chat"]

    def test_close_sends_eof_to_the_remote_repl(self, make_session):
        session = make_session()
        session.start("Qwen/Qwen3.5-0.8B")
        session.close()

        assert session._closed
        # stdin was closed → chat_repl sees EOF and exits 0 remotely.

    def test_context_manager_closes(self, make_session, api):
        with make_session() as session:
            session.start("Qwen/Qwen3.5-0.8B")

        assert api.terminated == ["pod-chat"]

    def test_pod_is_terminated_when_the_pip_install_fails(self, make_session, api):
        with pytest.raises(RemoteExecutionError, match="remote command failed"):
            make_session(ssh=FakeSsh(exit_code=1)).start("Qwen/Qwen3.5-0.8B")

        assert api.terminated == ["pod-chat"]

    def test_ask_after_close_is_rejected(self, make_session):
        session = make_session()
        session.start("Qwen/Qwen3.5-0.8B")
        session.close()

        with pytest.raises(RemoteExecutionError, match="not open"):
            session.ask("hi")


class TestLifecycle:
    """A session must retain ownership until pod termination succeeds."""

    def test_public_state_tracks_resource_ownership(
        self, make_session, api, monkeypatch
    ):
        session = make_session()
        assert session.state is ChatSessionState.NEW
        assert session.pod_id is None
        create, terminate = api.create_pod, api.terminate_pod

        def creating(**kwargs):
            assert session.state is ChatSessionState.STARTING
            with pytest.raises(RemoteExecutionError, match="new RemoteChatSession"):
                session.start("duplicate/model")
            return create(**kwargs)

        def terminating(pod_id):
            assert session.state is ChatSessionState.CLOSING
            assert session.pod_id == pod_id
            session.close()  # Reentrant cleanup cannot submit another deletion.
            terminate(pod_id)

        monkeypatch.setattr(api, "create_pod", creating)
        monkeypatch.setattr(api, "terminate_pod", terminating)
        session.start("some/model")
        assert session.state is ChatSessionState.READY
        assert json.loads(json.dumps({"state": session.state})) == {"state": "ready"}
        session.close()
        assert session.state is ChatSessionState.CLOSED
        assert session.pod_id is None
        assert len(api.created) == 1
        assert api.terminated == ["pod-chat"]

    def test_preflight_failure_allows_configuration_correction(
        self, make_session, api, monkeypatch
    ):
        session = make_session()
        require_key = session._require_public_key

        def missing_key():
            raise RemoteExecutionError("missing public key", provider="runpod")

        monkeypatch.setattr(session, "_require_public_key", missing_key)
        with pytest.raises(RemoteExecutionError, match="missing public key"):
            session.start("some/model")
        assert session.state is ChatSessionState.NEW
        assert api.created == []

        monkeypatch.setattr(session, "_require_public_key", require_key)
        session.start("some/model")
        session.close()
        assert api.terminated == ["pod-chat"]

    def test_submission_error_does_not_allow_replaying_creation(
        self, make_session, api, monkeypatch
    ):
        attempts = []

        def unavailable(**kwargs):
            attempts.append(kwargs)
            raise OSError("submission response lost")

        monkeypatch.setattr(api, "create_pod", unavailable)
        session = make_session()
        with pytest.raises(OSError, match="submission response lost"):
            session.start("some/model")
        with pytest.raises(RemoteExecutionError, match="new RemoteChatSession"):
            session.start("some/model")
        assert len(attempts) == 1

    def test_second_start_cannot_orphan_the_first_pod(self, make_session, api):
        session = make_session()
        session.start("some/model")
        try:
            with pytest.raises(RemoteExecutionError, match="new RemoteChatSession"):
                session.start("another/model")
            assert len(api.created) == 1
            assert session.transcript["metadata"]["base_model"] == "some/model"
        finally:
            session.close()
        assert api.terminated == ["pod-chat"]

    @pytest.mark.parametrize("started", [False, True])
    def test_start_after_close_never_provisions(self, make_session, api, started):
        session = make_session()
        if started:
            session.start("some/model")
        session.close()
        created = len(api.created)

        with pytest.raises(RemoteExecutionError, match="new RemoteChatSession"):
            session.start("another/model")

        assert len(api.created) == created

    def test_failed_termination_retains_cleanup_and_exit_hook(
        self, make_session, api, monkeypatch, caplog
    ):
        import atexit

        registered, unregistered = [], []
        monkeypatch.setattr(atexit, "register", registered.append)
        monkeypatch.setattr(atexit, "unregister", unregistered.append)
        terminate = api.terminate_pod

        def unavailable(pod_id):
            raise OSError("provider unavailable")

        monkeypatch.setattr(api, "terminate_pod", unavailable)
        session = make_session()
        session.start("some/model")
        session.close()

        assert session.state is ChatSessionState.CLEANUP_PENDING
        assert registered == [session.close]
        assert unregistered == []
        assert session.pod_id == "pod-chat"
        assert "pod-chat" in caplog.text
        with pytest.raises(RemoteExecutionError, match="not open"):
            session.ask("hi")
        with pytest.raises(RemoteExecutionError, match="new RemoteChatSession"):
            session.start("another/model")

        monkeypatch.setattr(api, "terminate_pod", terminate)
        session.close()
        session.close()
        assert api.terminated == ["pod-chat"]
        assert session.pod_id is None
        assert session.state is ChatSessionState.CLOSED
        assert unregistered == [session.close]

    def test_cleanup_failure_preserves_original_start_error(
        self, make_session, api, monkeypatch
    ):
        terminate = api.terminate_pod

        def unavailable(pod_id):
            raise OSError("provider unavailable")

        monkeypatch.setattr(api, "terminate_pod", unavailable)
        session = make_session(ssh=FakeSsh(exit_code=1))
        try:
            with pytest.raises(RemoteExecutionError, match="remote command failed"):
                session.start("some/model")
            assert session.pod_id == "pod-chat"
        finally:
            monkeypatch.setattr(api, "terminate_pod", terminate)
            session.close()
        assert api.terminated == ["pod-chat"]

    def test_cost_is_finalized_once_after_successful_cleanup(
        self, make_session, api, monkeypatch
    ):
        session = make_session()
        session.start("some/model")
        recorded = []
        monkeypatch.setattr(session, "_record_cost", recorded.append)
        terminate = api.terminate_pod

        def unavailable(pod_id):
            raise OSError("provider unavailable")

        monkeypatch.setattr(api, "terminate_pod", unavailable)
        session.close()
        session.close()
        assert recorded == []

        monkeypatch.setattr(api, "terminate_pod", terminate)
        session.close()
        session.close()
        assert recorded == ["pod-chat"]
        assert api.terminated == ["pod-chat"]

    def test_interrupted_termination_remains_retryable(
        self, make_session, api, monkeypatch
    ):
        session = make_session()
        session.start("some/model")
        terminate = api.terminate_pod

        def interrupted(pod_id):
            raise KeyboardInterrupt

        monkeypatch.setattr(api, "terminate_pod", interrupted)
        try:
            with pytest.raises(KeyboardInterrupt):
                session.close()
            assert session.state is ChatSessionState.CLEANUP_PENDING
            assert session.pod_id == "pod-chat"
        finally:
            monkeypatch.setattr(api, "terminate_pod", terminate)
            session.close()
        assert api.terminated == ["pod-chat"]

    def test_interrupted_process_cleanup_still_terminates_pod(
        self, make_session, api, monkeypatch
    ):
        process = FakeChatProcess()
        session = make_session(process=process)
        session.start("some/model")

        def interrupted(timeout):
            raise KeyboardInterrupt

        monkeypatch.setattr(process, "wait", interrupted)
        with pytest.raises(KeyboardInterrupt):
            session.close()
        assert api.terminated == ["pod-chat"]
        assert session.state is ChatSessionState.CLOSED

    def test_concurrent_starts_only_create_one_pod(
        self, make_session, api, monkeypatch
    ):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event

        session = make_session()
        creating, proceed, second_started = Event(), Event(), Event()
        create = api.create_pod

        def blocked_create(**kwargs):
            creating.set()
            assert proceed.wait(timeout=5)
            return create(**kwargs)

        def second_start():
            second_started.set()
            session.start("another/model")

        monkeypatch.setattr(api, "create_pod", blocked_create)
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(session.start, "some/model")
                try:
                    assert creating.wait(timeout=5)
                    second = executor.submit(second_start)
                    assert second_started.wait(timeout=5)
                finally:
                    proceed.set()
                first.result(timeout=5)
                with pytest.raises(RemoteExecutionError, match="new RemoteChatSession"):
                    second.result(timeout=5)
        finally:
            session.close()
        assert len(api.created) == 1
        assert api.terminated == ["pod-chat"]


class TestCostRecording:
    """A chat pod costs money exactly like a training pod; leaving it out of
    the ledger made `stateset-agents costs` under-report real spend."""

    def test_closing_records_the_pod_cost(self, make_session, tmp_path, monkeypatch):
        from stateset_agents.remote import ledger

        path = tmp_path / "ledger.jsonl"
        monkeypatch.setattr(ledger, "DEFAULT_LEDGER_PATH", path)

        session = make_session()
        session.start("some/model", gpu="NVIDIA RTX A4000")
        session.close()

        entries = ledger.read_entries(path)
        assert len(entries) == 1
        assert entries[0]["base_model"] == "some/model"
        assert entries[0]["gpu"] == "NVIDIA RTX A4000"
        assert entries[0]["status"] == "chat"

    def test_double_close_records_once(self, make_session, tmp_path, monkeypatch):
        from stateset_agents.remote import ledger

        path = tmp_path / "ledger.jsonl"
        monkeypatch.setattr(ledger, "DEFAULT_LEDGER_PATH", path)

        session = make_session()
        session.start("some/model")
        session.close()
        session.close()

        assert len(ledger.read_entries(path)) == 1
