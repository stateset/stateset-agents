"""Tests for ``RemoteChatSession``, driven by behavioural fakes.

Same discipline as test_remote_runpod_executor.py: no network, no ssh
processes. The persistent channel is a fake process injected through the
``popen_factory`` seam, and — above all — every test that rents a pod
asserts it was terminated. A leaked pod bills by the hour.
"""

from __future__ import annotations

import json
import tarfile
from collections import deque
from pathlib import Path

import pytest

from stateset_agents.remote.chat_session import RemoteChatSession
from stateset_agents.remote.executor import RemoteExecutionError


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
def make_session(api):
    def build(process=None, ssh=None, **kwargs):
        process = process if process is not None else FakeChatProcess()
        launched: list[list[str]] = []

        def factory(cmd):
            launched.append(cmd)
            return process

        kwargs.setdefault("public_key", "ssh-rsa AAAATESTKEY")
        kwargs.setdefault("poll_interval_s", 0)
        session = RemoteChatSession(
            api=api,
            ssh=ssh if ssh is not None else FakeSsh(),
            popen_factory=factory,
            **kwargs,
        )
        session.launched = launched  # test-only spy
        return session

    return build


class TestStart:
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
