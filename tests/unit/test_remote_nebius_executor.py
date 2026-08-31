"""Behavioural tests for Nebius Serverless jobs and endpoints."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from stateset_agents.remote.deployment import DeploymentSpec
from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.remote.job import JobStatus, RemoteJobSpec
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
        return {"metadata": {"id": job_id}, "status": "SUCCEEDED"}

    def logs(self, job_id):
        return "loaded dataset\ntraining complete"

    def cancel(self, job_id):
        self.cancelled.append(job_id)

    def probe(self):
        return {"authenticated": True}

    def create_endpoint(self, args):
        self.created.append(args)
        return {"metadata": {"id": "endpoint-1"}, "status": {"url": "https://e"}}

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


def test_submit_status_logs_and_fetch(spec, tmp_path, configured_env) -> None:
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
    assert (executor.fetch(handle) / "adapter_config.json").exists()
    assert artifacts.deleted


def test_submit_failure_removes_uploaded_prefix(spec, tmp_path, configured_env) -> None:
    cli = FakeCli()
    artifacts = FakeArtifacts()

    def fail(_):
        raise RemoteExecutionError("no capacity", provider="nebius")

    cli.create_job = fail
    executor = NebiusExecutor(
        cli=cli,
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


def test_cancel_checks_remote_state_and_cleans_artifacts(
    spec, tmp_path, configured_env
) -> None:
    cli = FakeCli()
    cli.get_job = lambda job_id: {"metadata": {"id": job_id}, "status": "RUNNING"}
    artifacts = FakeArtifacts()
    executor = NebiusExecutor(
        cli=cli,
        artifact_store=artifacts,
        bucket="bucket",
        subnet_id="subnet",
        state_dir=tmp_path,
    )
    handle = executor.submit(spec)
    executor.cancel(handle)
    assert cli.cancelled == ["job-1"]
    assert artifacts.deleted


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
            stdout=json.dumps({"metadata": {"id": "job-1"}}),
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
