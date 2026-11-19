"""Unit tests for ``stateset-agents serve-remote``.

Faked at the ``RemoteServeSession``/``RunPodApi`` seams — no pods, no
network. What matters: the happy path prints URL + token + stop command,
bad inputs exit 2 before renting anything, and --stop/--list drive the
API without a session.
"""

from __future__ import annotations

import json

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
def fake_session(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
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

    @pytest.mark.parametrize(
        ("flag", "value"),
        [
            ("--max-hours", "0"),
            ("--max-hours", "nan"),
            ("--max-hours", "inf"),
            ("--max-cost", "0"),
            ("--max-cost", "nan"),
            ("--max-cost", "inf"),
        ],
    )
    def test_invalid_limits_exit_before_rental(self, flag, value):
        result = invoke("--base-model", "m", flag, value)

        assert result.exit_code == 2
        assert flag in result.output
        assert FakeSession.instances == []


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
        assert "accepted termination" in result.output

    def test_unconfirmed_stop_prints_pod_id(self, monkeypatch):
        api = FakeApi(self.PODS)

        def unavailable(pod_id):
            raise OSError("provider unavailable")

        monkeypatch.setattr(api, "terminate_pod", unavailable)
        result = invoke("--stop", "stateset-serve-abc")
        assert result.exit_code == 1
        assert "Billing may continue" in result.output
        assert "--stop a1" in result.output

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


class TestServeOrphans:
    name = "stateset-serve-123456789abc"

    def pending(self, pod_id):
        from stateset_agents.remote.serve_recovery import serving_lease_dir

        root = serving_lease_dir()
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{self.name}.json"
        path.write_text(json.dumps({"name": self.name, "pod_id": pod_id}))
        return path

    def inspect(self, *flags):
        return runner.invoke(app, ["runpod-orphans", "--serving", *flags])

    def test_inspect_then_terminate_exact_recovered_pod(self):
        path = self.pending(None)
        api = FakeApi(
            [{"name": self.name, "id": "one"}, {"name": self.name + "1", "id": "two"}]
        )
        report = self.inspect()
        assert report.exit_code == 0
        assert "recoverable  one" in report.output
        assert api.terminated == []
        assert path.exists()

        result = self.inspect("--terminate")
        assert result.exit_code == 0
        assert "termination accepted  one" in result.output
        assert api.terminated == ["one"]
        assert not path.exists()

    @pytest.mark.parametrize(
        ("recorded", "pod_ids"),
        [
            (None, []),
            (None, ["one", "two"]),
            ("old", ["new"]),
            (None, [None]),
        ],
    )
    def test_inconclusive_matches_remain_pending(self, recorded, pod_ids):
        path = self.pending(recorded)
        api = FakeApi([{"name": self.name, "id": i} for i in pod_ids])
        result = self.inspect("--terminate")
        assert result.exit_code == 1
        assert "unresolved" in result.output
        assert path.exists()
        assert api.terminated == []

    def test_unconfirmed_delete_retains_the_matched_id(self, monkeypatch):
        path = self.pending(None)
        api = FakeApi([{"name": self.name, "id": "one"}])
        with monkeypatch.context() as patch:

            def fail(pod_id):
                raise OSError("response lost")

            patch.setattr(api, "terminate_pod", fail)
            result = self.inspect("--terminate")
        assert result.exit_code == 1
        assert json.loads(path.read_text())["pod_id"] == "one"
        assert self.inspect("--terminate").exit_code == 0
        assert api.terminated == ["one"]
        assert not path.exists()

    def test_recovery_in_progress_excludes_another_delete(self, monkeypatch):
        path = self.pending("one")
        api = FakeApi([{"name": self.name, "id": "one"}])
        from stateset_agents.remote.serve_recovery import recover_serving

        def terminate(pod_id):
            results = recover_serving(path.parent, lambda: api, terminate=True)
            assert [row.status for row in results] == ["busy"]
            api.terminated.append(pod_id)

        monkeypatch.setattr(api, "terminate_pod", terminate)
        assert self.inspect("--terminate").exit_code == 0
        assert api.terminated == ["one"]

    def test_conflicting_provider_ids_are_left_for_inspection(self):
        path = self.pending(None)
        api = FakeApi(
            [{"name": self.name, "id": "one"}, {"name": "other", "id": "one"}]
        )
        assert self.inspect("--terminate").exit_code == 1
        assert path.exists()
        assert api.terminated == []

    @pytest.mark.parametrize("recorded_id", ["one", None])
    def test_stop_by_id_retires_matching_local_record(self, recorded_id):
        path = self.pending(recorded_id)
        FakeApi([{"name": self.name, "id": "one"}])
        assert invoke("--stop", "one").exit_code == 0
        assert not path.exists()

    def test_duplicate_name_does_not_retire_name_only_record_on_manual_stop(self):
        path = self.pending(None)
        api = FakeApi(
            [{"name": self.name, "id": "one"}, {"name": self.name, "id": "two"}]
        )
        assert invoke("--stop", self.name).exit_code == 1
        assert api.terminated == []
        assert invoke("--stop", "one").exit_code == 0
        assert path.exists()
        assert api.terminated == ["one"]

    def test_corrupt_record_does_not_block_independent_cleanup(self):
        bad = self.pending(None)
        bad.write_text("{")
        good_name = "stateset-serve-abcdefabcdef"
        good = bad.with_name(f"{good_name}.json")
        good.write_text(json.dumps({"name": good_name, "pod_id": "good"}))
        api = FakeApi(
            [{"name": self.name, "id": "bad"}, {"name": good_name, "id": "good"}]
        )
        result = self.inspect("--terminate")
        assert result.exit_code == 1
        assert bad.exists()
        assert not good.exists()
        assert api.terminated == ["good"]


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
                "--max-cost",
                "5",
                "--max-hours",
                "0.25",
            ],
        )

        assert result.exit_code == 0, result.output
        assert FakeSession.instances[0].started_with["adapters"] == {"adapter": out}
        assert FakeExecutor.spec.max_cost_usd == 5
        assert FakeSession.instances[0].started_with["max_cost_usd"] == 5
        assert FakeSession.instances[0].started_with["max_hours"] == 0.25
        assert "Endpoint ready" in result.output
        assert "$1.50" in result.output

    @pytest.mark.parametrize(
        ("flag", "value"),
        [
            ("--max-hours", "0"),
            ("--max-hours", "nan"),
            ("--max-cost", "inf"),
        ],
    )
    def test_invalid_serving_limits_exit_before_training(
        self, flag, value, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            "stateset_agents.cli_remote.get_executor",
            lambda _: pytest.fail("invalid limits must be rejected before training"),
        )
        dataset = tmp_path / "data.jsonl"
        dataset.touch()
        result = runner.invoke(
            app, ["deploy", "--dataset", str(dataset), "--base-model", "m", flag, value]
        )
        assert result.exit_code == 2
        assert flag in result.output
        assert FakeSession.instances == []

    @pytest.mark.parametrize("preview", [False, True])
    def test_failed_training_or_preview_does_not_serve(
        self, tmp_path, monkeypatch, preview
    ):
        from stateset_agents.remote.job import JobHandle, JobStatus, RemoteJobResult

        dataset = tmp_path / "d.jsonl"
        dataset.write_text('{"messages": []}\n')

        class FailingExecutor:
            def submit(self, spec):
                return JobHandle(provider="runpod", job_id="1")

            def wait(self, handle):
                return RemoteJobResult(
                    handle=handle,
                    status=JobStatus.SUCCEEDED if preview else JobStatus.FAILED,
                    output_dir=tmp_path if preview else None,
                    logs=["boom"],
                    is_preview=preview,
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
        assert ("returned a preview" if preview else "not serving") in result.output
