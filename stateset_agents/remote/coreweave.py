"""CoreWeave Kubernetes training and Dedicated Inference integrations."""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stateset_agents.remote.artifacts import S3ArtifactStore, parse_s3_uri
from stateset_agents.remote.deployment import (
    DeploymentHandle,
    DeploymentSpec,
    InferenceDeploymentProvider,
)
from stateset_agents.remote.executor import RemoteExecutionError, RemoteExecutor
from stateset_agents.remote.job import JobHandle, JobStatus, RemoteJobSpec

__all__ = [
    "CoreWeaveExecutor",
    "CoreWeaveInferenceApi",
    "CoreWeaveInferenceProvider",
    "KubectlApi",
]

_DEFAULT_IMAGE = "pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime"
_STATE_SCHEMA_VERSION = 1


def _default_state_dir() -> Path:
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "stateset-agents" / "coreweave-jobs"


def _encoded_spec(spec: RemoteJobSpec) -> str:
    payload = json.dumps(spec.to_dict(), separators=(",", ":"), sort_keys=True)
    return base64.urlsafe_b64encode(payload.encode()).decode()


class KubectlApi:
    """Injectable kubectl transport suitable for any conforming cluster."""

    def __init__(
        self,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        *,
        context: str | None = None,
        namespace: str = "default",
    ) -> None:
        self._runner = runner or subprocess.run
        self.context = context
        self.namespace = namespace

    def _command(self, args: list[str]) -> list[str]:
        command = ["kubectl"]
        if self.context:
            command.extend(["--context", self.context])
        command.extend(["--namespace", self.namespace, *args])
        return command

    def _run(self, args: list[str], *, stdin: str | None = None) -> str:
        try:
            result = self._runner(
                self._command(args),
                input=stdin,
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
        except FileNotFoundError as exc:
            raise RemoteExecutionError.wrap(
                exc,
                "kubectl is not installed; CoreWeave training requires a "
                "configured CKS kubeconfig",
                provider="coreweave",
            ) from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise RemoteExecutionError.wrap(
                exc, "could not execute kubectl", provider="coreweave"
            ) from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "unknown error").strip()
            raise RemoteExecutionError(
                f"kubectl failed ({result.returncode}): {detail}",
                provider="coreweave",
            )
        return result.stdout

    def apply_job(self, manifest: dict[str, Any]) -> None:
        self._run(["apply", "-f", "-"], stdin=json.dumps(manifest))

    def get_job(self, name: str) -> dict[str, Any]:
        output = self._run(["get", "job", name, "-o", "json"])
        try:
            payload = json.loads(output)
        except json.JSONDecodeError as exc:
            raise RemoteExecutionError(
                "kubectl returned invalid job JSON", provider="coreweave"
            ) from exc
        if not isinstance(payload, dict):
            raise RemoteExecutionError(
                "kubectl returned a non-object job", provider="coreweave"
            )
        return payload

    def logs(self, name: str) -> str:
        return self._run(["logs", f"job/{name}"])

    def delete_job(self, name: str) -> None:
        self._run(
            [
                "delete",
                "job",
                name,
                "--ignore-not-found=true",
                "--wait=true",
            ]
        )

    def can_create_jobs(self) -> bool:
        output = self._run(["auth", "can-i", "create", "jobs.batch"])
        return output.strip().lower() == "yes"


@dataclass
class _CoreWeaveJob:
    spec: RemoteJobSpec
    input_uri: str
    output_uri: str
    status: JobStatus = JobStatus.PENDING
    logs: list[str] = field(default_factory=list)
    fetched: Path | None = None


class CoreWeaveExecutor(RemoteExecutor):
    """Run packaged StateSet SFT jobs as Kubernetes Jobs on CKS."""

    name = "coreweave"
    supported_job_kinds = frozenset({"sft"})
    durable_handles = True
    managed_deployments = True
    result_kind = "local_artifacts"
    compute_model = "managed-bare-metal-kubernetes"
    verification_status = "code-complete-live-certification-pending"
    DEFAULT_GPU = "L40"

    def __init__(
        self,
        kube: KubectlApi | None = None,
        artifact_store: S3ArtifactStore | None = None,
        *,
        bucket: str | None = None,
        storage_secret: str | None = None,
        image: str | None = None,
        service_account: str | None = None,
        state_dir: Path | None = None,
        poll_interval_s: float = 10.0,
    ) -> None:
        local_endpoint = os.environ.get("COREWEAVE_S3_ENDPOINT_URL")
        self.kube = kube or KubectlApi(
            context=os.environ.get("COREWEAVE_KUBE_CONTEXT"),
            namespace=os.environ.get("COREWEAVE_KUBE_NAMESPACE", "default"),
        )
        self.artifacts = artifact_store or S3ArtifactStore(endpoint_url=local_endpoint)
        self.bucket = bucket or os.environ.get("COREWEAVE_S3_BUCKET")
        self.storage_secret = storage_secret or os.environ.get(
            "COREWEAVE_STORAGE_SECRET"
        )
        self.image = image or os.environ.get("COREWEAVE_JOB_IMAGE") or _DEFAULT_IMAGE
        self.service_account = service_account or os.environ.get(
            "COREWEAVE_SERVICE_ACCOUNT"
        )
        self.state_dir = Path(state_dir or _default_state_dir())
        self.poll_interval_s = poll_interval_s
        self._jobs: dict[str, _CoreWeaveJob] = {}

    def _require_configuration(self) -> tuple[str, str]:
        missing = [
            name
            for name, value in (
                ("COREWEAVE_S3_BUCKET", self.bucket),
                ("COREWEAVE_STORAGE_SECRET", self.storage_secret),
            )
            if not value
        ]
        if missing:
            raise RemoteExecutionError(
                "CoreWeave configuration is incomplete; set " + ", ".join(missing),
                provider=self.name,
            )
        assert self.bucket and self.storage_secret
        return self.bucket, self.storage_secret

    def _state_path(self, job_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", job_id)
        return self.state_dir / f"{safe}.json"

    def _persist(self, job_id: str, job: _CoreWeaveJob) -> None:
        target = self._state_path(job_id)
        temporary = target.with_suffix(".tmp")
        payload = {
            "schema_version": _STATE_SCHEMA_VERSION,
            "provider": self.name,
            "job_id": job_id,
            "spec": job.spec.to_dict(),
            "input_uri": job.input_uri,
            "output_uri": job.output_uri,
            "status": job.status.value,
            "logs": job.logs,
            "fetched": str(job.fetched) if job.fetched else None,
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(target)

    def _job(self, handle: JobHandle) -> _CoreWeaveJob:
        if handle.provider != self.name:
            raise RemoteExecutionError(
                "job handle provider mismatch", provider=self.name
            )
        if handle.job_id in self._jobs:
            return self._jobs[handle.job_id]
        try:
            payload = json.loads(self._state_path(handle.job_id).read_text())
            if payload.get("schema_version") != _STATE_SCHEMA_VERSION:
                raise ValueError("unsupported state schema")
            job = _CoreWeaveJob(
                spec=RemoteJobSpec.from_dict(payload["spec"]),
                input_uri=str(payload["input_uri"]),
                output_uri=str(payload["output_uri"]),
                status=JobStatus(payload.get("status", "pending")),
                logs=[str(line) for line in payload.get("logs", [])],
                fetched=Path(payload["fetched"]) if payload.get("fetched") else None,
            )
        except FileNotFoundError as exc:
            raise RemoteExecutionError(
                f"unknown CoreWeave job: {handle.job_id}", provider=self.name
            ) from exc
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RemoteExecutionError(
                f"invalid durable metadata for CoreWeave job {handle.job_id}: {exc}",
                provider=self.name,
            ) from exc
        self._jobs[handle.job_id] = job
        return job

    def _manifest(
        self,
        spec: RemoteJobSpec,
        name: str,
        input_uri: str,
        output_uri: str,
        storage_secret: str,
    ) -> dict[str, Any]:
        from stateset_agents import __version__

        version = spec.package_version or __version__
        package = f"stateset-agents[training,cloud]=={version}"
        endpoint = os.environ.get("COREWEAVE_JOB_S3_ENDPOINT_URL", "http://cwlota.com")
        script = (
            'python -m pip install --quiet "$STATESET_PACKAGE" && exec '
            "python -m stateset_agents.remote.worker "
            '--spec-b64 "$STATESET_SPEC_B64" '
            '--input-uri "$STATESET_INPUT_URI" '
            '--output-uri "$STATESET_OUTPUT_URI" '
            '--s3-endpoint-url "$STATESET_S3_ENDPOINT_URL"'
        )
        container: dict[str, Any] = {
            "name": "trainer",
            "image": self.image,
            "command": ["bash", "-lc"],
            "args": [script],
            "env": [
                {"name": "STATESET_PACKAGE", "value": package},
                {"name": "STATESET_SPEC_B64", "value": _encoded_spec(spec)},
                {"name": "STATESET_INPUT_URI", "value": input_uri},
                {"name": "STATESET_OUTPUT_URI", "value": output_uri},
                {"name": "STATESET_S3_ENDPOINT_URL", "value": endpoint},
            ],
            "envFrom": [{"secretRef": {"name": storage_secret}}],
            "resources": {
                "limits": {"nvidia.com/gpu": spec.gpu_count},
                "requests": {"nvidia.com/gpu": spec.gpu_count},
            },
        }
        if hf_secret := os.environ.get("COREWEAVE_HF_SECRET"):
            container["envFrom"].append({"secretRef": {"name": hf_secret}})
        pod_spec: dict[str, Any] = {
            "restartPolicy": "Never",
            "containers": [container],
            "nodeSelector": {"gpu.nvidia.com/class": spec.gpu or self.DEFAULT_GPU},
        }
        if spec.container_disk_gb is not None:
            container["volumeMounts"] = [{"name": "scratch", "mountPath": "/tmp"}]
            pod_spec["volumes"] = [
                {
                    "name": "scratch",
                    "emptyDir": {"sizeLimit": f"{spec.container_disk_gb}Gi"},
                }
            ]
        if self.service_account:
            pod_spec["serviceAccountName"] = self.service_account
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": name,
                "labels": {
                    "app.kubernetes.io/name": "stateset-agents",
                    "stateset.ai/job-kind": spec.job_kind,
                },
            },
            "spec": {
                "backoffLimit": 0,
                "activeDeadlineSeconds": spec.timeout_s,
                "ttlSecondsAfterFinished": 86400,
                "template": {
                    "metadata": {"labels": {"stateset.ai/job": name}},
                    "spec": pod_spec,
                },
            },
        }

    def submit(self, spec: RemoteJobSpec) -> JobHandle:
        self.validate_spec(spec)
        if spec.max_cost_usd is not None:
            raise RemoteExecutionError(
                "CoreWeave CKS capacity is cluster-billed, so a per-job "
                "--max-cost cannot be enforced by this executor",
                provider=self.name,
            )
        bucket, storage_secret = self._require_configuration()
        local_id = uuid.uuid4().hex[:12]
        name = f"stateset-sft-{local_id}"
        prefix = f"stateset-agents/jobs/{local_id}"
        input_uri = f"s3://{bucket}/{prefix}/input/{spec.dataset.name}"
        output_uri = f"s3://{bucket}/{prefix}/output"
        self.artifacts.upload_file(spec.dataset, input_uri)
        try:
            self.kube.apply_job(
                self._manifest(spec, name, input_uri, output_uri, storage_secret)
            )
        except Exception:
            self.artifacts.delete_prefix(f"s3://{bucket}/{prefix}")
            raise
        job = _CoreWeaveJob(
            spec=spec,
            input_uri=input_uri,
            output_uri=output_uri,
            logs=[f"CoreWeave Kubernetes job {name} submitted"],
        )
        self._jobs[name] = job
        self._persist(name, job)
        return JobHandle(provider=self.name, job_id=name)

    def status(self, handle: JobHandle) -> JobStatus:
        job = self._job(handle)
        if job.status.is_terminal:
            return job.status
        payload = self.kube.get_job(handle.job_id)
        status = payload.get("status", {})
        conditions = status.get("conditions", [])
        if any(
            c.get("type") == "Complete" and c.get("status") == "True"
            for c in conditions
        ):
            job.status = JobStatus.SUCCEEDED
        elif any(
            c.get("type") == "Failed" and c.get("status") == "True" for c in conditions
        ):
            job.status = JobStatus.FAILED
        elif int(status.get("active", 0) or 0) > 0:
            job.status = JobStatus.RUNNING
        else:
            job.status = JobStatus.PENDING
        self._persist(handle.job_id, job)
        return job.status

    def logs(self, handle: JobHandle) -> Iterator[str]:
        job = self._job(handle)
        try:
            remote = self.kube.logs(handle.job_id).splitlines()
        except RemoteExecutionError as exc:
            remote = [f"log retrieval failed: {exc}"]
        yield from [*job.logs, *remote]

    def fetch(self, handle: JobHandle, dest: Path | None = None) -> Path:
        job = self._job(handle)
        if not job.status.is_terminal:
            raise RemoteExecutionError(
                f"CoreWeave job {handle.job_id} is not terminal", provider=self.name
            )
        destination = Path(dest or job.spec.output_dir)
        if job.fetched == destination and destination.exists():
            return destination
        written = self.artifacts.download_prefix(job.output_uri, destination)
        if not written and not job.spec.dry_run:
            raise RemoteExecutionError(
                f"CoreWeave job {handle.job_id} produced no artifacts",
                provider=self.name,
            )
        job.fetched = destination
        self.artifacts.delete_prefix(job.input_uri.split("/input/", 1)[0])
        self._persist(handle.job_id, job)
        return destination

    def cancel(self, handle: JobHandle) -> None:
        job = self._job(handle)
        if not job.status.is_terminal:
            self.status(handle)
        if not job.status.is_terminal:
            self.kube.delete_job(handle.job_id)
            job.status = JobStatus.CANCELLED
            job.logs.append("Kubernetes job deleted")
            self._persist(handle.job_id, job)
        if job.status is JobStatus.CANCELLED:
            # Idempotent so a caller can retry cleanup after an object-store
            # failure without issuing a second provider cancellation.
            self.artifacts.delete_prefix(job.input_uri.split("/input/", 1)[0])

    def wait(self, handle: JobHandle, poll_interval_s: float | None = None):
        return super().wait(
            handle,
            poll_interval_s=(
                self.poll_interval_s if poll_interval_s is None else poll_interval_s
            ),
        )

    def canary(self) -> dict[str, Any]:
        """Check CKS authorization without creating a workload."""
        started = time.time()
        allowed = self.kube.can_create_jobs()
        if not allowed:
            raise RemoteExecutionError(
                "the configured Kubernetes identity cannot create Jobs",
                provider=self.name,
            )
        return {
            "provider": self.name,
            "authenticated": True,
            "can_create_jobs": True,
            "latency_ms": round((time.time() - started) * 1000, 1),
        }


class CoreWeaveInferenceApi:
    """REST client for CoreWeave Dedicated Inference."""

    def __init__(
        self,
        api_token: str,
        *,
        root: str = "https://api.coreweave.com/v1alpha1/inference",
        session: Any | None = None,
    ) -> None:
        self.api_token = api_token
        self.root = root.rstrip("/")
        self._session = session

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> Any:
        if self._session is None:
            import requests

            self._session = requests.Session()
        try:
            response = self._session.request(
                method,
                self.root + path,
                headers={
                    "Authorization": f"Bearer {self.api_token}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            if response.status_code == 204 or not response.content:
                return {}
            return response.json()
        except Exception as exc:  # noqa: BLE001 - requests-compatible clients
            raise RemoteExecutionError.wrap(
                exc,
                f"CoreWeave Inference API {method} {path} failed",
                provider="coreweave-inference",
            ) from exc

    def deployment_parameters(self) -> dict[str, Any]:
        return dict(self._request("GET", "/deployments/parameters"))

    def gateway_parameters(self) -> dict[str, Any]:
        return dict(self._request("GET", "/gateways/parameters"))

    def create_gateway(self, name: str, zone: str) -> dict[str, Any]:
        return dict(
            self._request(
                "POST",
                "/gateways",
                {
                    "name": name,
                    "zones": [zone],
                    "coreWeaveAuth": {},
                    "bodyBasedRouting": {"apiType": "API_TYPE_OPENAI"},
                },
            )
        )

    def get_gateway(self, gateway_id: str) -> dict[str, Any]:
        return dict(self._request("GET", f"/gateways/{gateway_id}"))

    def delete_gateway(self, gateway_id: str) -> None:
        self._request("DELETE", f"/gateways/{gateway_id}")

    def create_deployment(self, payload: dict[str, Any]) -> dict[str, Any]:
        return dict(self._request("POST", "/deployments", payload))

    def get_deployment(self, deployment_id: str) -> dict[str, Any]:
        return dict(self._request("GET", f"/deployments/{deployment_id}"))

    def delete_deployment(self, deployment_id: str) -> None:
        self._request("DELETE", f"/deployments/{deployment_id}")


def _resource(payload: dict[str, Any], kind: str) -> dict[str, Any]:
    value = payload.get(kind, payload)
    return value if isinstance(value, dict) else {}


def _resource_id(payload: dict[str, Any], kind: str) -> str:
    resource = _resource(payload, kind)
    spec = resource.get("spec", {})
    value = spec.get("id") or resource.get("id")
    if not value:
        raise RemoteExecutionError(
            f"CoreWeave returned no {kind} id", provider="coreweave"
        )
    return str(value)


class CoreWeaveInferenceProvider(InferenceDeploymentProvider):
    """Create and tear down CoreWeave BYOW inference deployments."""

    name = "coreweave"

    def __init__(self, api: CoreWeaveInferenceApi | None = None) -> None:
        if api is None:
            token = os.environ.get("COREWEAVE_API_TOKEN", "").strip()
            if not token:
                raise RemoteExecutionError(
                    "COREWEAVE_API_TOKEN is not set", provider=self.name
                )
            api = CoreWeaveInferenceApi(token)
        self.api = api

    def _ensure_gateway(self, spec: DeploymentSpec) -> tuple[str, str | None, bool]:
        if spec.gateway_id:
            gateway = self.api.get_gateway(spec.gateway_id)
            resource = _resource(gateway, "gateway")
            endpoints = resource.get("status", {}).get("endpoints", [])
            return spec.gateway_id, str(endpoints[0]) if endpoints else None, False
        if not spec.zone:
            raise RemoteExecutionError(
                "CoreWeave deployment needs gateway_id or zone",
                provider=self.name,
            )
        created = self.api.create_gateway(f"{spec.name}-gateway", spec.zone)
        gateway_id = _resource_id(created, "gateway")
        resource = _resource(created, "gateway")
        endpoints = resource.get("status", {}).get("endpoints", [])
        return gateway_id, str(endpoints[0]) if endpoints else None, True

    def deploy(self, spec: DeploymentSpec) -> DeploymentHandle:
        if spec.runtime not in {"vllm", "dynamo-vllm"}:
            raise RemoteExecutionError(
                "CoreWeave Dedicated Inference runtime must be vllm or " "dynamo-vllm",
                provider=self.name,
            )
        if spec.gpu_count not in {1, 2, 4, 8, 16}:
            raise RemoteExecutionError(
                "CoreWeave Dedicated Inference gpu_count must be one of "
                "1, 2, 4, 8, or 16",
                provider=self.name,
            )
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", spec.name):
            raise RemoteExecutionError(
                "CoreWeave deployment name must be a lowercase DNS label of at "
                "most 63 characters",
                provider=self.name,
            )
        if not 4 <= len(spec.model_name) <= 63:
            raise RemoteExecutionError(
                "CoreWeave model_name must contain 4 to 63 characters",
                provider=self.name,
            )
        bucket, path = parse_s3_uri(spec.weights_uri)
        if not path:
            raise RemoteExecutionError(
                "weights_uri must identify a model directory", provider=self.name
            )
        gateway_id, endpoint, owns_gateway = self._ensure_gateway(spec)
        runtime: dict[str, Any] = {
            "engine": spec.runtime,
            "engineConfig": dict(spec.runtime_config),
        }
        if spec.runtime_version:
            runtime["version"] = spec.runtime_version
        payload = {
            "name": spec.name,
            "gatewayIds": [gateway_id],
            "runtime": runtime,
            "resources": {"instanceType": spec.gpu, "gpuCount": spec.gpu_count},
            "model": {"name": spec.model_name, "bucket": bucket, "path": path},
            "autoscaling": {
                "min": spec.min_replicas,
                "max": spec.max_replicas,
            },
            "traffic": {"weight": 100},
        }
        try:
            created = self.api.create_deployment(payload)
        except Exception:
            if owns_gateway:
                try:
                    self.api.delete_gateway(gateway_id)
                except Exception:  # noqa: BLE001 - preserve deployment failure
                    pass
            raise
        deployment_id = _resource_id(created, "deployment")
        return DeploymentHandle(
            provider=self.name,
            deployment_id=deployment_id,
            model_name=spec.model_name,
            endpoint=endpoint,
            gateway_id=gateway_id,
            owns_gateway=owns_gateway,
        )

    def status(self, handle: DeploymentHandle) -> dict[str, Any]:
        if handle.provider != self.name:
            raise RemoteExecutionError(
                "deployment handle provider mismatch", provider=self.name
            )
        return self.api.get_deployment(handle.deployment_id)

    def delete(self, handle: DeploymentHandle) -> None:
        if handle.provider != self.name:
            raise RemoteExecutionError(
                "deployment handle provider mismatch", provider=self.name
            )
        self.api.delete_deployment(handle.deployment_id)
        if handle.owns_gateway and handle.gateway_id:
            self.api.delete_gateway(handle.gateway_id)
