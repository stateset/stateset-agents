"""Behavioural tests for Nebius Serverless jobs and endpoints."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from stateset_agents.remote.deployment import DeploymentSpec
from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.remote.job import JobHandle, JobStatus, RemoteJobSpec
from stateset_agents.remote.nebius import (
    NebiusCli,
    NebiusEndpointProvider,
    NebiusExecutor,
)


class FakeArtifacts:
    def __init__(self):
        self.uploaded = []
        self.deleted = []

    def upload_file(self, source, uri):
        self.uploaded.append((Path(source), uri))

    def download_prefix(self, uri, destination):
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / "adapter_config.json"
        target.write_text("{}")
        return [target]

    def delete_prefix(self, uri):
        self.deleted.append(uri)


class FakeCli:
    def __init__(self):
        self.created = []
        self.cancelled = []
        self.deleted_endpoints = []

    def create_job(self, args):
        self.created.append(args)
        return "job-1"

    def get_job(self, job_id):
        return {"metadata": {"id": job_id}, "status": {"state": "COMPLETED"}}

    def logs(self, job_id):
        return "loaded dataset\ntraining complete"

    def cancel(self, job_id):
        self.cancelled.append(job_id)

    def probe(self):
        return {"authenticated": True}

    def create_endpoint(self, args):
        self.created.append(args)
        return {
            "spec": {"image": {"id": "image-1", "url": "https://registry.example"}},
            "metadata": {"id": "endpoint-1"},
            "status": {
                "private_endpoints": ["https://private"],
                "public_endpoints": ["tls://tcp:443", "https://e"],
            },
        }

    def get_endpoint(self, endpoint_id):
        return {"metadata": {"id": endpoint_id}, "status": "RUNNING"}

    def delete_endpoint(self, endpoint_id):
        self.deleted_endpoints.append(endpoint_id)


@pytest.fixture
def spec(tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(json.dumps({"messages": []}) + "\n")
    return RemoteJobSpec(
        dataset=dataset,
        base_model="Qwen/test",
        output_dir=tmp_path / "out",
        package_version="0.43.0",
    )


@pytest.fixture
def configured_env(monkeypatch):
    monkeypatch.setenv("NEBIUS_S3_ACCESS_KEY_SECRET", "mb-access")
    monkeypatch.setenv("NEBIUS_S3_SECRET_KEY_SECRET", "mb-secret")


@pytest.mark.parametrize("dry_run", [False, True])
def test_submit_status_logs_and_fetch(spec, tmp_path, configured_env, dry_run) -> None:
    from stateset_agents.remote.worker import decode_spec

    spec.dry_run = dry_run
    cli = FakeCli()
    artifacts = FakeArtifacts()
    executor = NebiusExecutor(
        cli=cli,
        artifact_store=artifacts,
        bucket="bucket",
        subnet_id="subnet",
        state_dir=tmp_path / "state",
        poll_interval_s=0,
    )

    handle = executor.submit(spec)
    assert artifacts.uploaded[0][0] == spec.dataset
    rendered = cli.created[0]
    assert "--env-secret" in rendered
    assert "AWS_ACCESS_KEY_ID=mb-access" in rendered
    assert executor.status(handle) is JobStatus.SUCCEEDED
    assert "training complete" in list(executor.logs(handle))
    encoded = next(arg for arg in rendered if arg.startswith("STATESET_SPEC_B64="))
    assert decode_spec(encoded.split("=", 1)[1])["dry_run"] is dry_run
    result = executor.wait(handle)
    assert result.is_preview is dry_run
    if dry_run:
        assert result.output_dir is None
        assert not spec.output_dir.exists()
    else:
        assert (result.output_dir / "adapter_config.json").exists()
    assert artifacts.deleted


def test_rejected_create_removes_uploaded_prefix(
    spec, tmp_path, configured_env
) -> None:
    artifacts = FakeArtifacts()

    def runner(command, **kwargs):
        return subprocess.CompletedProcess(command, 25, stderr="no capacity")

    executor = NebiusExecutor(
        cli=NebiusCli(runner=runner),
        artifact_store=artifacts,
        bucket="bucket",
        subnet_id="subnet",
        state_dir=tmp_path / "state",
    )
    with pytest.raises(RemoteExecutionError, match="no capacity"):
        executor.submit(spec)
    assert artifacts.deleted == [
        "s3://bucket/stateset-agents/jobs/"
        + artifacts.uploaded[0][1].split("/jobs/")[1].split("/input/")[0]
    ]
    assert list((tmp_path / "state").iterdir()) == []


@pytest.mark.parametrize(
    "outcome", ["timeout", "operation_only", "unavailable", "invalid_json"]
)
def test_lost_create_reply_can_reconnect_without_resubmitting(
    spec, tmp_path, configured_env, outcome
) -> None:
    artifacts = FakeArtifacts()
    calls = []
    name = ""

    def runner(command, **kwargs):
        nonlocal name
        action = command[3]
        calls.append(action)
        if action == "create":
            name = command[command.index("--name") + 1]
            if outcome == "timeout":
                raise subprocess.TimeoutExpired(command, 120)
            if outcome == "unavailable":
                return subprocess.CompletedProcess(command, 20, stderr="unavailable")
            if outcome == "invalid_json":
                return subprocess.CompletedProcess(command, 0, stdout="truncated")
            payload = {"id": "operation-1", "request": {"id": "unrelated"}}
        elif action == "get-by-name":
            assert command[command.index("--name") + 1] == name
            payload = {"metadata": {"id": "job-1", "name": name}}
        elif action == "get":
            assert command[4] == "job-1"
            payload = {"status": {"state": "COMPLETED"}}
        else:
            assert action == "logs" and command[4] == "job-1"
            return subprocess.CompletedProcess(command, 0, stdout="finished")
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload))

    options = {
        "cli": NebiusCli(runner),
        "artifact_store": artifacts,
        "bucket": "bucket",
        "subnet_id": "subnet",
        "state_dir": tmp_path / "state",
    }
    with pytest.raises(RemoteExecutionError, match="Reconnect") as error:
        NebiusExecutor(**options).submit(spec)
    assert f"--job-id {name} --wait" in str(error.value)
    assert not artifacts.deleted
    spec.dataset.unlink()
    handle = JobHandle("nebius", name)
    result = NebiusExecutor(**options).wait(handle)
    assert result.status is JobStatus.SUCCEEDED
    assert (result.output_dir / "adapter_config.json").exists()
    assert "finished" in result.logs
    assert NebiusExecutor(**options).fetch(handle) == spec.output_dir
    assert calls == ["create", "get-by-name", "get", "logs"]
    assert len(artifacts.deleted) == 1


@pytest.mark.parametrize("final", ["CANCELLED", "COMPLETED", "FAILED"])
def test_cancel_preserves_artifacts_until_remote_confirmation(
    spec, tmp_path, configured_env, final
) -> None:
    cli = FakeCli()
    states = iter(["RUNNING", "CANCELLING", final])
    cli.get_job = lambda _: {"status": {"state": next(states)}}
    artifacts = FakeArtifacts()
    options = {
        "cli": cli,
        "artifact_store": artifacts,
        "bucket": "bucket",
        "subnet_id": "subnet",
        "state_dir": tmp_path,
    }
    executor = NebiusExecutor(**options)
    handle = executor.submit(spec)
    executor.cancel(handle)
    assert cli.cancelled == ["job-1"]
    assert not artifacts.deleted
    resumed = NebiusExecutor(**options)
    if final == "CANCELLED":
        assert resumed.status(handle) is JobStatus.CANCELLED
        assert len(artifacts.deleted) == 1
        assert resumed.status(handle) is JobStatus.CANCELLED
        assert len(artifacts.deleted) == 1
    else:
        expected = JobStatus.SUCCEEDED if final == "COMPLETED" else JobStatus.FAILED
        assert resumed.status(handle) is expected
        assert not artifacts.deleted
        assert resumed.fetch(handle) == spec.output_dir


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("STATE_UNSPECIFIED", JobStatus.PENDING),
        ("PROVISIONING", JobStatus.PENDING),
        ("IMAGE_PULLING", JobStatus.PENDING),
        ("STARTING", JobStatus.PENDING),
        ("DELETING", JobStatus.RUNNING),
        ("ERROR", JobStatus.FAILED),
    ],
)
def test_status_uses_job_state(spec, tmp_path, configured_env, state, expected):
    cli = FakeCli()
    cli.get_job = lambda _: {
        "spec": {"state": "FAILED"},
        "status": {"instances": [{"state": "RUNNING"}], "state": state},
    }
    executor = NebiusExecutor(
        cli, FakeArtifacts(), bucket="b", subnet_id="s", state_dir=tmp_path
    )
    assert executor.status(executor.submit(spec)) is expected


@pytest.mark.parametrize(
    "status", [{"instances": [{"state": "COMPLETED"}]}, {"state": "NEW_STATE"}]
)
def test_unknown_job_state_stops_wait(spec, tmp_path, configured_env, status):
    cli = FakeCli()
    cli.get_job = lambda _: {"status": status}
    artifacts = FakeArtifacts()
    executor = NebiusExecutor(
        cli, artifacts, bucket="b", subnet_id="s", state_dir=tmp_path
    )
    with pytest.raises(RemoteExecutionError, match=r"status\.state"):
        executor.wait(executor.submit(spec))
    assert not artifacts.deleted


def test_legacy_cancellation_is_rechecked_without_local_dataset(
    spec, tmp_path, configured_env
):
    cli = FakeCli()
    artifacts = FakeArtifacts()
    executor = NebiusExecutor(
        cli, artifacts, bucket="b", subnet_id="s", state_dir=tmp_path
    )
    handle = executor.submit(spec)
    path = tmp_path / f"{handle.job_id}.json"
    record = json.loads(path.read_text())
    del record["resource_id"], record["cleanup_on_cancel"]
    record["status"] = "cancelled"
    path.write_text(json.dumps(record))
    spec.dataset.unlink()
    assert executor.status(handle) is JobStatus.SUCCEEDED
    executor.cancel(handle)
    assert not cli.cancelled and not artifacts.deleted
    assert executor.fetch(handle) == spec.output_dir


@pytest.mark.parametrize("seconds", [300, 1805, 3601, 168 * 3600])
def test_timeout_is_forwarded_exactly(spec, tmp_path, configured_env, seconds):
    spec.timeout_s = seconds
    cli = FakeCli()
    executor = NebiusExecutor(
        cli, FakeArtifacts(), bucket="b", subnet_id="s", state_dir=tmp_path
    )
    executor.submit(spec)
    args = cli.created[0]
    assert args[args.index("--timeout") + 1] == f"{seconds}s"


@pytest.mark.parametrize("seconds", [299, 168 * 3600 + 1])
def test_unsupported_timeout_fails_before_upload(
    spec, tmp_path, configured_env, seconds
):
    spec.timeout_s = seconds
    cli = FakeCli()
    artifacts = FakeArtifacts()
    executor = NebiusExecutor(
        cli, artifacts, bucket="b", subnet_id="s", state_dir=tmp_path
    )
    with pytest.raises(RemoteExecutionError, match="timeout"):
        executor.submit(spec)
    assert not artifacts.uploaded and not cli.created


def test_recovery_requires_exact_job_name():
    def runner(command, **kwargs):
        payload = {"metadata": {"id": "other-job", "name": "other-name"}}
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload))

    with pytest.raises(RemoteExecutionError, match="no matching job"):
        NebiusCli(runner).job_id_by_name("expected-name")


def test_fetch_retries_cleanup_after_restart_without_a_second_download(
    spec, tmp_path, configured_env, monkeypatch
):
    artifacts = FakeArtifacts()
    executor = NebiusExecutor(
        FakeCli(), artifacts, bucket="input", subnet_id="s", state_dir=tmp_path
    )
    handle = executor.submit(spec)

    def cleanup_reply_lost(uri):
        artifacts.deleted.append(uri)
        raise RemoteExecutionError("cleanup reply lost")

    with monkeypatch.context() as patch:
        patch.setattr(artifacts, "delete_prefix", cleanup_reply_lost)
        with pytest.raises(RemoteExecutionError, match="cleanup reply lost"):
            executor.fetch(handle)
    monkeypatch.setattr(
        artifacts, "download_prefix", lambda *_: pytest.fail("repeat download")
    )
    resumed = NebiusExecutor(FakeCli(), artifacts, state_dir=tmp_path)
    assert resumed.fetch(handle) == spec.output_dir
    prefix = artifacts.uploaded[0][1].rsplit("/input/", 1)[0]
    assert artifacts.deleted == [prefix, prefix]
    assert (spec.output_dir / "adapter_config.json").is_file()


def test_overlapping_reconnects_cannot_overwrite_fetch_receipt(
    spec, tmp_path, configured_env, monkeypatch
):
    artifacts = FakeArtifacts()
    first = NebiusExecutor(
        FakeCli(), artifacts, bucket="b", subnet_id="s", state_dir=tmp_path
    )
    handle = first.submit(spec)
    second = NebiusExecutor(FakeCli(), artifacts, state_dir=tmp_path)
    download = artifacts.download_prefix

    def overlapping_download(uri, dest):
        with pytest.raises(RemoteExecutionError, match="another process"):
            second.fetch(handle)
        return download(uri, dest)

    monkeypatch.setattr(artifacts, "download_prefix", overlapping_download)
    assert first.fetch(handle) == spec.output_dir
    assert second.fetch(handle) == spec.output_dir
    assert len(artifacts.deleted) == 1


def test_train_remote_prints_durable_id_before_poll_failure(
    spec, tmp_path, configured_env, monkeypatch
):
    from typer.testing import CliRunner

    from stateset_agents import cli_remote

    cli = FakeCli()
    cli.get_job = lambda _: {"status": {"state": "UNKNOWN"}}
    executor = NebiusExecutor(
        cli, FakeArtifacts(), bucket="b", subnet_id="s", state_dir=tmp_path
    )
    monkeypatch.setattr(cli_remote, "get_executor", lambda _: executor)
    result = CliRunner().invoke(
        cli_remote.app,
        [
            "train-remote",
            "--provider",
            "nebius",
            "--dataset",
            str(spec.dataset),
            "--base-model",
            spec.base_model,
        ],
    )
    assert result.exit_code == 1
    assert "Submitted nebius job job-1" in result.output
    assert "unrecognized status.state" in result.output


def test_max_cost_fails_closed(spec, tmp_path, configured_env) -> None:
    spec.max_cost_usd = 1
    executor = NebiusExecutor(
        cli=FakeCli(),
        artifact_store=FakeArtifacts(),
        bucket="bucket",
        subnet_id="subnet",
        state_dir=tmp_path,
    )
    with pytest.raises(RemoteExecutionError, match="cannot be enforced"):
        executor.submit(spec)


def test_gpu_count_must_match_nebius_preset(spec, tmp_path, configured_env) -> None:
    spec.gpu_count = 2
    executor = NebiusExecutor(
        cli=FakeCli(),
        artifact_store=FakeArtifacts(),
        bucket="bucket",
        subnet_id="subnet",
        preset="1gpu-8vcpu-32gb",
        state_dir=tmp_path,
    )
    with pytest.raises(RemoteExecutionError, match="does not represent"):
        executor.submit(spec)


def test_endpoint_lifecycle(monkeypatch) -> None:
    monkeypatch.setenv("NEBIUS_S3_PROFILE_SECRET", "mb-profile")
    monkeypatch.setenv("NEBIUS_ENDPOINT_TOKEN_SECRET", "mb-token")
    cli = FakeCli()
    provider = NebiusEndpointProvider(cli=cli, subnet_id="subnet")
    spec = DeploymentSpec(
        name="model-endpoint",
        model_name="support-model",
        weights_uri="s3://models/qwen/merged",
        gpu="gpu-h100-sxm",
    )

    handle = provider.deploy(spec)
    assert handle.deployment_id == "endpoint-1"
    assert handle.endpoint == "https://e"
    assert "s3://models:/models:ro:default@mb-profile" in cli.created[0]
    assert "mb-token" in cli.created[0]
    assert "--public" not in cli.created[0]
    assert provider.status(handle)["status"] == "RUNNING"
    provider.delete(handle)
    assert cli.deleted_endpoints == ["endpoint-1"]


def test_endpoint_rejects_unrepresentable_scaling(monkeypatch) -> None:
    monkeypatch.setenv("NEBIUS_S3_PROFILE_SECRET", "mb-profile")
    monkeypatch.setenv("NEBIUS_ENDPOINT_TOKEN_SECRET", "mb-token")
    provider = NebiusEndpointProvider(cli=FakeCli(), subnet_id="subnet")
    spec = DeploymentSpec("name", "model", "s3://bucket/path", "gpu", max_replicas=2)
    with pytest.raises(RemoteExecutionError, match="provider-managed"):
        provider.deploy(spec)


def test_nebius_cli_uses_profile_and_machine_readable_output() -> None:
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"id": "operation-1", "resource_id": "job-1"}),
            stderr="",
        )

    cli = NebiusCli(runner=runner, profile="stateset")
    assert cli.create_job(["--name", "job"]) == "job-1"
    command, kwargs = calls[0]
    assert command == [
        "nebius",
        "--profile",
        "stateset",
        "ai",
        "job",
        "create",
        "--async",
        "--name",
        "job",
        "--format",
        "json",
    ]
    assert kwargs["timeout"] == 120


def test_endpoint_requires_recoverable_token(monkeypatch) -> None:
    monkeypatch.setenv("NEBIUS_S3_PROFILE_SECRET", "mb-profile")
    monkeypatch.delenv("NEBIUS_ENDPOINT_TOKEN_SECRET", raising=False)
    provider = NebiusEndpointProvider(cli=FakeCli(), subnet_id="subnet")
    spec = DeploymentSpec("name", "model", "s3://bucket/path", "gpu")
    with pytest.raises(RemoteExecutionError, match="TOKEN_SECRET"):
        provider.deploy(spec)
