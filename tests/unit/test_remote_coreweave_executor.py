"""Behavioural tests for CoreWeave CKS and Dedicated Inference."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from stateset_agents.remote.coreweave import (
    CoreWeaveExecutor,
    CoreWeaveInferenceApi,
    CoreWeaveInferenceProvider,
    KubectlApi,
)
from stateset_agents.remote.deployment import DeploymentHandle, DeploymentSpec
from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.remote.job import JobStatus, RemoteJobSpec


class FakeArtifacts:
    def __init__(self):
        self.uploaded = []
        self.deleted = []

    def upload_file(self, source, uri):
        self.uploaded.append((Path(source), uri))

    def download_prefix(self, uri, destination):
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / "adapter_model.safetensors"
        target.write_bytes(b"weights")
        return [target]

    def delete_prefix(self, uri):
        self.deleted.append(uri)


class FakeKube:
    def __init__(self):
        self.manifests = []
        self.deleted = []
        self.allowed = True

    def apply_job(self, manifest):
        self.manifests.append(manifest)

    def get_job(self, name):
        return {
            "metadata": {"name": name},
            "status": {"conditions": [{"type": "Complete", "status": "True"}]},
        }

    def logs(self, name):
        return "training complete"

    def delete_job(self, name):
        self.deleted.append(name)

    def can_create_jobs(self):
        return self.allowed


@pytest.fixture
def spec(tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(json.dumps({"messages": []}) + "\n")
    return RemoteJobSpec(
        dataset=dataset,
        base_model="Qwen/test",
        output_dir=tmp_path / "out",
        gpu="H100",
        gpu_count=2,
        package_version="0.43.0",
    )


def test_cks_submit_status_logs_fetch_and_manifest(spec, tmp_path) -> None:
    kube = FakeKube()
    artifacts = FakeArtifacts()
    executor = CoreWeaveExecutor(
        kube=kube,
        artifact_store=artifacts,
        bucket="bucket",
        storage_secret="s3-creds",
        state_dir=tmp_path / "state",
        poll_interval_s=0,
    )

    handle = executor.submit(spec)
    manifest = kube.manifests[0]
    pod = manifest["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert pod["nodeSelector"] == {"gpu.nvidia.com/class": "H100"}
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 2
    assert container["envFrom"] == [{"secretRef": {"name": "s3-creds"}}]
    assert not any("SECRET" in item.get("value", "") for item in container["env"])
    assert executor.status(handle) is JobStatus.SUCCEEDED
    assert "training complete" in list(executor.logs(handle))
    assert (executor.fetch(handle) / "adapter_model.safetensors").exists()


def test_cks_scratch_disk_sets_worker_tempdir(spec, tmp_path) -> None:
    spec.container_disk_gb = 80
    kube = FakeKube()
    executor = CoreWeaveExecutor(
        kube=kube,
        artifact_store=FakeArtifacts(),
        bucket="bucket",
        storage_secret="secret",
        state_dir=tmp_path,
    )
    executor.submit(spec)
    pod = kube.manifests[0]["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert {"name": "TMPDIR", "value": "/stateset-scratch"} in container["env"]
    assert container["volumeMounts"][0]["mountPath"] == "/stateset-scratch"


def test_cks_cancel_deletes_kubernetes_job(spec, tmp_path) -> None:
    kube = FakeKube()
    kube.get_job = lambda name: {"metadata": {"name": name}, "status": {"active": 1}}
    executor = CoreWeaveExecutor(
        kube=kube,
        artifact_store=FakeArtifacts(),
        bucket="bucket",
        storage_secret="secret",
        state_dir=tmp_path,
    )
    handle = executor.submit(spec)
    executor.cancel(handle)
    assert kube.deleted == [handle.job_id]


def test_cks_cancel_does_not_delete_completed_job(spec, tmp_path) -> None:
    kube = FakeKube()
    executor = CoreWeaveExecutor(
        kube=kube,
        artifact_store=FakeArtifacts(),
        bucket="bucket",
        storage_secret="secret",
        state_dir=tmp_path,
    )
    handle = executor.submit(spec)
    executor.cancel(handle)
    assert kube.deleted == []


def test_cks_canary_is_read_only(tmp_path) -> None:
    kube = FakeKube()
    executor = CoreWeaveExecutor(
        kube=kube,
        artifact_store=FakeArtifacts(),
        bucket="bucket",
        storage_secret="secret",
        state_dir=tmp_path,
    )
    assert executor.canary()["can_create_jobs"] is True
    assert not kube.manifests


def test_cks_max_cost_fails_closed(spec, tmp_path) -> None:
    spec.max_cost_usd = 1
    executor = CoreWeaveExecutor(
        kube=FakeKube(),
        artifact_store=FakeArtifacts(),
        bucket="bucket",
        storage_secret="secret",
        state_dir=tmp_path,
    )
    with pytest.raises(RemoteExecutionError, match="cluster-billed"):
        executor.submit(spec)


class FakeInferenceApi:
    def __init__(self):
        self.created_gateways = []
        self.created_deployments = []
        self.deleted = []
        self.deleted_gateways = []

    def create_gateway(self, name, zone):
        self.created_gateways.append((name, zone))
        return {
            "gateway": {
                "spec": {"id": "gateway-1"},
                "status": {"endpoints": ["https://gateway"]},
            }
        }

    def get_gateway(self, gateway_id):
        return {
            "gateway": {
                "spec": {"id": gateway_id},
                "status": {"endpoints": ["https://gateway"]},
            }
        }

    def create_deployment(self, payload):
        self.created_deployments.append(payload)
        return {"deployment": {"spec": {"id": "deployment-1"}}}

    def get_deployment(self, deployment_id):
        return {
            "deployment": {
                "spec": {"id": deployment_id},
                "status": {"status": "STATUS_READY"},
            }
        }

    def delete_deployment(self, deployment_id):
        self.deleted.append(deployment_id)

    def delete_gateway(self, gateway_id):
        self.deleted_gateways.append(gateway_id)


def test_coreweave_inference_full_lifecycle() -> None:
    api = FakeInferenceApi()
    provider = CoreWeaveInferenceProvider(api=api)
    spec = DeploymentSpec(
        name="support",
        model_name="support-model",
        weights_uri="s3://weights/models/support",
        gpu="gd-8xh100ib-i128",
        gpu_count=2,
        min_replicas=1,
        max_replicas=3,
        zone="US-WEST-04A",
        runtime="dynamo-vllm",
        runtime_config={"max-model-len": "8192"},
    )

    handle = provider.deploy(spec)

    assert handle.deployment_id == "deployment-1"
    assert handle.endpoint == "https://gateway"
    payload = api.created_deployments[0]
    assert payload["model"] == {
        "name": "support-model",
        "bucket": "weights",
        "path": "models/support",
    }
    assert payload["resources"]["gpuCount"] == 2
    assert payload["runtime"]["engine"] == "dynamo-vllm"
    assert payload["autoscaling"]["max"] == 3
    assert provider.status(handle)["deployment"]["status"]["status"] == "STATUS_READY"
    provider.delete(handle)
    assert api.deleted == ["deployment-1"]
    assert api.deleted_gateways == ["gateway-1"]


def test_coreweave_inference_rejects_bucket_root() -> None:
    provider = CoreWeaveInferenceProvider(api=FakeInferenceApi())
    spec = DeploymentSpec("name", "model", "s3://weights", "gpu", zone="zone")
    with pytest.raises(RemoteExecutionError, match="model directory"):
        provider.deploy(spec)


def test_coreweave_rolls_back_auto_created_gateway_on_deploy_failure() -> None:
    api = FakeInferenceApi()

    def fail(_payload):
        raise RemoteExecutionError("no capacity", provider="coreweave-inference")

    api.create_deployment = fail
    provider = CoreWeaveInferenceProvider(api=api)
    spec = DeploymentSpec("name", "model", "s3://weights/path", "gpu", zone="zone")
    with pytest.raises(RemoteExecutionError, match="no capacity"):
        provider.deploy(spec)
    assert api.deleted_gateways == ["gateway-1"]


def test_kubectl_transport_uses_context_namespace_and_stdin() -> None:
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    api = KubectlApi(runner=runner, context="cks", namespace="agents")
    api.apply_job({"kind": "Job", "metadata": {"name": "job"}})
    command, kwargs = calls[0]
    assert command == [
        "kubectl",
        "--context",
        "cks",
        "--namespace",
        "agents",
        "apply",
        "-f",
        "-",
    ]
    assert json.loads(kwargs["input"])["kind"] == "Job"


def test_coreweave_http_transport_sets_bearer_and_timeout() -> None:
    class Response:
        status_code = 200
        content = b"{}"

        def raise_for_status(self):
            return None

        def json(self):
            return {"zones": ["zone-a"]}

    class Session:
        def __init__(self):
            self.calls = []

        def request(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return Response()

    session = Session()
    api = CoreWeaveInferenceApi("token", session=session)
    assert api.gateway_parameters() == {"zones": ["zone-a"]}
    args, kwargs = session.calls[0]
    assert args == (
        "GET",
        "https://api.coreweave.com/v1alpha1/inference/gateways/parameters",
    )
    assert kwargs["headers"]["Authorization"] == "Bearer token"
    assert kwargs["timeout"] == 30


def test_inference_handle_provider_mismatch_fails_closed() -> None:
    provider = CoreWeaveInferenceProvider(api=FakeInferenceApi())
    handle = DeploymentHandle("nebius", "deployment-1", "model")
    with pytest.raises(RemoteExecutionError, match="provider mismatch"):
        provider.delete(handle)


def test_coreweave_rejects_invalid_deployment_name() -> None:
    provider = CoreWeaveInferenceProvider(api=FakeInferenceApi())
    spec = DeploymentSpec(
        "Invalid.Name", "model", "s3://weights/path", "gpu", zone="zone"
    )
    with pytest.raises(RemoteExecutionError, match="DNS label"):
        provider.deploy(spec)
