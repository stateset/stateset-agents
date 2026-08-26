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
        self.effect_warnings: list[str] = []
        self.endpoint_url = None
        self.pod_id = None
        self.pod_name = None
        FakeSession.instances.append(self)

    def start(
        self,
        base_model,
        adapter_dir=None,
        gpu=None,
        max_hours=1.0,
        adapters=None,
        merge=False,
        strict_effect=False,
        gpu_count=1,
        max_cost_usd=None,
        network_volume_id=None,
    ):
        if FakeSession.start_raises is not None:
            raise FakeSession.start_raises
        self.started_with = {
            "base_model": base_model,
            "adapter_dir": adapter_dir,
            "adapters": adapters,
            "gpu": gpu,
            "max_hours": max_hours,
            "gpu_count": gpu_count,
            "max_cost_usd": max_cost_usd,
            "network_volume_id": network_volume_id,
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
        assert '"model": "Qwen/Qwen3.5-0.8B"' in result.output

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
            "--ready-timeout",
            "900",
            "--max-hours",
            "2.5",
            "--gpu-count",
            "4",
            "--max-cost",
            "7",
            "--network-volume-id",
            "vol-1",
        )

        assert result.exit_code == 0, result.output
        session = FakeSession.instances[0]
        assert session.kwargs["container_disk_gb"] == 120
        assert session.kwargs["ready_timeout_s"] == 900
        assert session.started_with == {
            "base_model": "m",
            "adapter_dir": None,
            "adapters": {"adapter": adapter},
            "gpu": "NVIDIA H100 80GB HBM3",
            "max_hours": 2.5,
            "gpu_count": 4,
            "max_cost_usd": 7.0,
            "network_volume_id": "vol-1",
        }
        # With an adapter, the curl example targets the served adapter model.
        assert '"model": "adapter"' in result.output

    def test_prebuilt_vllm_image_and_args_reach_session(self):
        result = invoke(
            "--base-model",
            "Qwen/Qwen3.8-Flash-Next-FP8",
            "--gpu-count",
            "4",
            "--vllm-image",
            "vllm/vllm-openai:qwen38-flash-next",
            "--vllm-arg=--tensor-parallel-size",
            "--vllm-arg=4",
        )

        assert result.exit_code == 0, result.output
        session = FakeSession.instances[0]
        assert session.kwargs["direct_vllm_image"] is True
        assert session.kwargs["image"] == "vllm/vllm-openai:qwen38-flash-next"
        assert session.kwargs["vllm_args"] == ["--tensor-parallel-size", "4"]
        assert session.started_with["gpu_count"] == 4

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


class TestMultiAdapter:
    def test_named_adapters_reach_start_under_their_names(self, tmp_path):
        a, b = tmp_path / "gen1", tmp_path / "gen2"
        for d in (a, b):
            d.mkdir()

        result = invoke(
            "--base-model",
            "m",
            "--adapter",
            f"champion={a}",
            "--adapter",
            f"challenger={b}",
        )

        assert result.exit_code == 0, result.output
        started = FakeSession.instances[0].started_with
        assert started["adapters"] == {"champion": a, "challenger": b}
        assert "served-model name: champion" in result.output
        assert "served-model name: challenger" in result.output

    def test_duplicate_adapter_names_are_refused(self, tmp_path):
        a = tmp_path / "a"
        a.mkdir()

        result = invoke(
            "--base-model", "m", "--adapter", f"x={a}", "--adapter", f"x={a}"
        )

        assert result.exit_code != 0
        assert "duplicate" in result.output

    def test_bare_path_serves_under_the_default_name(self, tmp_path):
        a = tmp_path / "a"
        a.mkdir()

        result = invoke("--base-model", "m", "--adapter", str(a))

        assert result.exit_code == 0, result.output
        assert FakeSession.instances[0].started_with["adapters"] == {"adapter": a}


class TestDeploy:
    def test_deploy_is_registered(self):
        names = {
            command.name or command.callback.__name__
            for command in app.registered_commands
        }
        assert "deploy" in names

    def test_trains_then_serves_the_fresh_adapter(self, tmp_path, monkeypatch):
        """The zero-to-API story as one invocation: a successful training
        job's output_dir becomes the served adapter."""
        from stateset_agents.remote.job import JobHandle, JobStatus, RemoteJobResult

        dataset = tmp_path / "d.jsonl"
        dataset.write_text('{"messages": []}\n')
        out = tmp_path / "adapter"
        out.mkdir()

        class FakeExecutor:
            def submit(self, spec):
                FakeExecutor.spec = spec
                return JobHandle(provider="runpod", job_id="1")

            def wait(self, handle):
                return RemoteJobResult(
                    handle=handle,
                    status=JobStatus.SUCCEEDED,
                    output_dir=out,
                    cost_usd=1.5,
                )

        monkeypatch.setattr(
            "stateset_agents.cli_remote.get_executor", lambda name: FakeExecutor()
        )
        result = runner.invoke(
            app,
            [
                "deploy",
                "--dataset",
                str(dataset),
                "--base-model",
                "m",
                "--output-dir",
                str(tmp_path / "adapter"),
            ],
        )

        assert result.exit_code == 0, result.output
        assert FakeSession.instances[0].started_with["adapters"] == {"adapter": out}
        assert "Endpoint ready" in result.output
        assert "$1.50" in result.output

    def test_failed_training_does_not_serve(self, tmp_path, monkeypatch):
        from stateset_agents.remote.job import JobHandle, JobStatus, RemoteJobResult

        dataset = tmp_path / "d.jsonl"
        dataset.write_text('{"messages": []}\n')

        class FailingExecutor:
            def submit(self, spec):
                return JobHandle(provider="runpod", job_id="1")

            def wait(self, handle):
                return RemoteJobResult(
                    handle=handle,
                    status=JobStatus.FAILED,
                    output_dir=None,
                    logs=["boom"],
                )

        monkeypatch.setattr(
            "stateset_agents.cli_remote.get_executor",
            lambda name: FailingExecutor(),
        )
        result = runner.invoke(
            app,
            ["deploy", "--dataset", str(dataset), "--base-model", "m"],
        )

        assert result.exit_code == 1
        assert FakeSession.instances == []
        assert "not serving" in result.output
