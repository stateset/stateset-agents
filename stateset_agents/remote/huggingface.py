"""Hugging Face Jobs training and Inference Endpoints integration."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from stateset_agents import __version__
from stateset_agents.remote.deployment import (
    DeploymentHandle,
    DeploymentSpec,
    InferenceDeploymentProvider,
)
from stateset_agents.remote.executor import RemoteExecutionError, RemoteExecutor
from stateset_agents.remote.job import JobHandle, JobStatus, RemoteJobSpec

HF_TOKEN_ENV = "HF_TOKEN"
HF_JOBS_BUCKET_ENV = "HF_JOBS_BUCKET"
HF_JOBS_NAMESPACE_ENV = "HF_JOBS_NAMESPACE"
HF_JOB_POINTER = "huggingface_job.json"

_STAGES = {
    "SCHEDULING": JobStatus.PENDING,
    "RUNNING": JobStatus.RUNNING,
    "COMPLETED": JobStatus.SUCCEEDED,
    "CANCELED": JobStatus.CANCELLED,
    "ERROR": JobStatus.FAILED,
    "DELETED": JobStatus.CANCELLED,
}


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


class HuggingFaceJobsExecutor(RemoteExecutor):
    """Run the packaged StateSet worker on Hugging Face Jobs."""

    name = "huggingface"
    supported_job_kinds = frozenset({"sft", "harvest"})
    durable_handles = True
    result_kind = "hosted_pointer"
    compute_model = "managed-container-jobs"
    verification_status = "unit-tested-live-lifecycle-pending"

    def __init__(
        self,
        api: Any | None = None,
        *,
        namespace: str | None = None,
        bucket: str | None = None,
        image: str = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime",
    ) -> None:
        self._api = api
        self.namespace = namespace or os.environ.get(HF_JOBS_NAMESPACE_ENV)
        self.bucket = bucket or os.environ.get(HF_JOBS_BUCKET_ENV)
        self.image = image
        self._specs: dict[str, RemoteJobSpec] = {}
        self._urls: dict[str, str | None] = {}

    def _client(self) -> Any:
        if self._api is not None:
            return self._api
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RemoteExecutionError.wrap(
                exc,
                "the Hugging Face provider needs huggingface_hub>=1.0; "
                "install 'stateset-agents[huggingface]'",
                provider=self.name,
            ) from exc
        self._api = HfApi(token=os.environ.get(HF_TOKEN_ENV))
        return self._api

    def submit(self, spec: RemoteJobSpec) -> JobHandle:
        self.validate_spec(spec)
        api = self._client()
        if not self.bucket:
            raise RemoteExecutionError(
                f"set {HF_JOBS_BUCKET_ENV} to a writable Hub bucket "
                "(for example 'my-org/stateset-jobs')",
                provider=self.name,
            )
        remote_root = f"/workspace/state/{spec.output_dir.name}"
        try:
            # Stage exactly the requested dataset. Uploading its parent could
            # accidentally include credentials or unrelated corpora.
            with tempfile.TemporaryDirectory(prefix="stateset-hf-job-") as staging:
                staged = Path(staging) / spec.dataset.name
                shutil.copy2(spec.dataset, staged)
                volume = api.sync_job_volume(
                    source=staging,
                    mount_path="/workspace/state",
                    remote_name=self.bucket,
                    read_only=False,
                    namespace=self.namespace,
                )
            dataset = f"/workspace/state/{spec.dataset.name}"
            module = (
                "stateset_agents.training.harvest"
                if spec.job_kind == "harvest"
                else "stateset_agents.training.sft"
            )
            version = spec.package_version or __version__
            package = f"stateset-agents[training]=={version}"
            args = spec.to_cli_args()
            args = [
                (
                    dataset
                    if arg == str(spec.dataset)
                    else remote_root if arg == str(spec.output_dir) else arg
                )
                for arg in args
            ]
            worker = " ".join(
                ["python", "-m", module] + [shlex.quote(arg) for arg in args]
            )
            command = [
                "bash",
                "-lc",
                f"python -m pip install --quiet {shlex.quote(package)} && {worker}",
            ]
            job = api.run_job(
                image=self.image,
                command=command,
                flavor=spec.gpu or "a10g-large",
                timeout=spec.timeout_s,
                name=f"stateset-{spec.job_kind}-{spec.output_dir.name}",
                labels={"framework": "stateset-agents", "job_kind": spec.job_kind},
                volumes=[volume],
                namespace=self.namespace,
            )
        except Exception as exc:  # noqa: BLE001
            raise RemoteExecutionError.wrap(
                exc, "could not submit Hugging Face Job", provider=self.name
            ) from exc
        job_id = str(_field(job, "id"))
        self._specs[job_id] = spec
        self._urls[job_id] = _field(job, "url")
        return JobHandle(self.name, job_id)

    def status(self, handle: JobHandle) -> JobStatus:
        try:
            job = self._client().inspect_job(handle.job_id, namespace=self.namespace)
            status = _field(job, "status")
            stage = str(_field(status, "stage", "SCHEDULING")).upper()
            return _STAGES.get(stage, JobStatus.PENDING)
        except Exception as exc:  # noqa: BLE001
            raise RemoteExecutionError.wrap(
                exc, "could not inspect Hugging Face Job", job_id=handle.job_id
            ) from exc

    def logs(self, handle: JobHandle) -> Iterator[str]:
        try:
            yield from self._client().fetch_job_logs(
                handle.job_id, namespace=self.namespace, follow=False
            )
        except Exception as exc:  # noqa: BLE001
            raise RemoteExecutionError.wrap(
                exc, "could not fetch Hugging Face Job logs", job_id=handle.job_id
            ) from exc

    def fetch(self, handle: JobHandle, dest: Path | None = None) -> Path:
        if self.status(handle) is not JobStatus.SUCCEEDED:
            raise RemoteExecutionError("Hugging Face Job has not succeeded")
        target = Path(dest or f"outputs/huggingface-{handle.job_id}")
        target.mkdir(parents=True, exist_ok=True)
        spec = self._specs.get(handle.job_id)
        payload = {
            "provider": self.name,
            "job_id": handle.job_id,
            "job_url": self._urls.get(handle.job_id),
            "bucket": self.bucket,
            "artifact_uri": (
                f"hf://buckets/{self.bucket}/{spec.output_dir.name}"
                if self.bucket and spec
                else None
            ),
        }
        (target / HF_JOB_POINTER).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return target

    def cancel(self, handle: JobHandle) -> None:
        try:
            self._client().cancel_job(handle.job_id, namespace=self.namespace)
        except Exception as exc:  # noqa: BLE001
            raise RemoteExecutionError.wrap(
                exc, "could not cancel Hugging Face Job", job_id=handle.job_id
            ) from exc


class HuggingFaceEndpointProvider(InferenceDeploymentProvider):
    """Manage dedicated Hugging Face Inference Endpoints."""

    name = "huggingface"

    def __init__(self, api: Any | None = None, *, namespace: str | None = None) -> None:
        self._api = api
        self.namespace = namespace or os.environ.get(HF_JOBS_NAMESPACE_ENV)

    def _client(self) -> Any:
        if self._api is None:
            try:
                from huggingface_hub import HfApi
            except ImportError as exc:
                raise RemoteExecutionError.wrap(
                    exc, "install 'stateset-agents[huggingface]'", provider=self.name
                ) from exc
            self._api = HfApi(token=os.environ.get(HF_TOKEN_ENV))
        return self._api

    def deploy(self, spec: DeploymentSpec) -> DeploymentHandle:
        endpoint = self._client().create_inference_endpoint(
            name=spec.name,
            repository=spec.weights_uri,
            framework=spec.runtime_config.get("framework", "pytorch"),
            task=spec.runtime_config.get("task", "text-generation"),
            accelerator="gpu",
            instance_size=spec.runtime_config.get(
                "instance_size", f"x{spec.gpu_count}"
            ),
            instance_type=spec.gpu,
            region=spec.zone or "us-east-1",
            vendor=spec.runtime_config.get("vendor", "aws"),
            min_replica=spec.min_replicas,
            max_replica=spec.max_replicas,
            namespace=self.namespace,
        )
        name = str(_field(endpoint, "name", spec.name))
        return DeploymentHandle(
            provider=self.name,
            deployment_id=name,
            model_name=spec.model_name,
            endpoint=_field(endpoint, "url"),
        )

    def status(self, handle: DeploymentHandle) -> dict[str, Any]:
        endpoint = self._client().get_inference_endpoint(
            handle.deployment_id, namespace=self.namespace
        )
        raw = _field(endpoint, "raw", {})
        return (
            raw
            if isinstance(raw, dict)
            else {"state": str(_field(endpoint, "status", raw))}
        )

    def delete(self, handle: DeploymentHandle) -> None:
        endpoint = self._client().get_inference_endpoint(
            handle.deployment_id, namespace=self.namespace
        )
        endpoint.delete()


__all__ = ["HuggingFaceEndpointProvider", "HuggingFaceJobsExecutor"]
