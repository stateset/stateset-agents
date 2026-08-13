"""Unit tests for ``stateset-agents serve-remote``.

Faked at the ``RemoteServeSession``/``RunPodApi`` seams — no pods, no
network. What matters: the happy path prints URL + token + stop command,
bad inputs exit 2 before renting anything, and --stop/--list drive the
API without a session.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from stateset_agents.cli import app
from stateset_agents.remote import serve_session
from stateset_agents.remote.executor import RemoteExecutionError

runner = CliRunner()


class FakeSession:
    instances: list[FakeSession] = []
    start_raises: Exception | None = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started_with: dict | None = None
        self.token = "tok-test"
        self.endpoint_url = None
        self.pod_id = None
        self.pod_name = None
        FakeSession.instances.append(self)

    def start(self, base_model, adapter_dir=None, gpu=None, max_hours=1.0):
        if FakeSession.start_raises is not None:
            raise FakeSession.start_raises
        self.started_with = {
            "base_model": base_model,
            "adapter_dir": adapter_dir,
            "gpu": gpu,
            "max_hours": max_hours,
        }
        self.endpoint_url = "http://1.2.3.4:18000"
        self.pod_id = "pod-1"
        self.pod_name = "stateset-serve-abc"

    def _require_api(self):
        return FakeApi.instance

    def terminate(self):
        pass


class FakeApi:
    instance: FakeApi = None  # type: ignore[assignment]

    def __init__(self, pods=None):
        self.pods = pods or []
        self.terminated: list[str] = []
        FakeApi.instance = self

    def list_pods(self):
        return list(self.pods)

    def terminate_pod(self, pod_id):
        self.terminated.append(pod_id)


@pytest.fixture(autouse=True)
def fake_session(monkeypatch):
    FakeSession.instances = []
    FakeSession.start_raises = None
    FakeApi([])
    monkeypatch.setattr(serve_session, "RemoteServeSession", FakeSession)
    return FakeSession


def invoke(*args):
    return runner.invoke(app, ["serve-remote", *args])


class TestServe:
    def test_prints_endpoint_token_curl_and_stop_command(self):
        result = invoke("--base-model", "Qwen/Qwen3.5-0.8B")

        assert result.exit_code == 0, result.output
        assert "http://1.2.3.4:18000/v1" in result.output
        assert "tok-test" in result.output
        assert "curl http://1.2.3.4:18000/v1/chat/completions" in result.output
        assert "Bearer tok-test" in result.output
        assert "--stop stateset-serve-abc" in result.output

    def test_options_reach_start(self, tmp_path):
        adapter = tmp_path / "adapter"
        adapter.mkdir()

        result = invoke(
            "--base-model",
            "m",
            "--adapter",
            str(adapter),
            "--gpu",
            "NVIDIA H100 80GB HBM3",
            "--container-disk-gb",
            "120",
            "--max-hours",
            "2.5",
        )

        assert result.exit_code == 0, result.output
        session = FakeSession.instances[0]
        assert session.kwargs["container_disk_gb"] == 120
        assert session.started_with == {
            "base_model": "m",
            "adapter_dir": adapter,
            "gpu": "NVIDIA H100 80GB HBM3",
            "max_hours": 2.5,
        }
        # With an adapter, the curl example targets the served adapter model.
        assert '"model": "adapter"' in result.output

    def test_start_failure_exits_1(self):
        FakeSession.start_raises = RemoteExecutionError("no GPUs", provider="runpod")

        result = invoke("--base-model", "m")

        assert result.exit_code == 1
        assert "no GPUs" in result.output


class TestValidation:
    def test_missing_base_model_exits_2_before_renting(self):
        result = invoke()

        assert result.exit_code == 2
        assert "--base-model is required" in result.output
        assert all(s.started_with is None for s in FakeSession.instances)

    def test_missing_adapter_dir_exits_2(self, tmp_path):
        result = invoke("--base-model", "m", "--adapter", str(tmp_path / "absent"))

        assert result.exit_code == 2
        assert "does not exist" in result.output

    def test_nonpositive_max_hours_exits_2(self):
        result = invoke("--base-model", "m", "--max-hours", "0")

        assert result.exit_code == 2
        assert "max-hours" in result.output


class TestStopAndList:
    PODS = [
        {
            "id": "a1",
            "name": "stateset-serve-abc",
            "desiredStatus": "RUNNING",
            "costPerHr": 0.17,
            "createdAt": "2026-08-13T00:00:00Z",
        }
    ]

    def test_stop_terminates_by_name(self):
        FakeApi(self.PODS)

        result = invoke("--stop", "stateset-serve-abc")

        assert result.exit_code == 0, result.output
        assert FakeApi.instance.terminated == ["a1"]
        assert "Terminated" in result.output

    def test_stop_unknown_pod_exits_1(self):
        FakeApi([])

        result = invoke("--stop", "nope")

        assert result.exit_code == 1
        assert FakeApi.instance.terminated == []

    def test_list_shows_serve_pods(self):
        FakeApi(self.PODS)

        result = invoke("--list")

        assert result.exit_code == 0, result.output
        assert "stateset-serve-abc" in result.output
        assert "$0.17/hr" in result.output

    def test_list_with_no_pods_says_so(self):
        result = invoke("--list")

        assert result.exit_code == 0, result.output
        assert "No serve pods running" in result.output


class TestRegistration:
    def test_command_is_registered(self):
        names = {
            command.name or command.callback.__name__
            for command in app.registered_commands
        }
        assert "serve-remote" in names
