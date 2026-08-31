"""Nebius Serverless AI executor.

The official ``nebius`` CLI owns authentication and the job control plane.
Datasets and adapters travel through Nebius' S3-compatible Object Storage;
credentials are injected into the remote container from SecretStash selectors
(called MysteryBox by some API surfaces), never serialized into the job
specification or local durable state.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shlex
import subprocess
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stateset_agents.remote.artifacts import S3ArtifactStore
from stateset_agents.remote.deployment import (
    DeploymentHandle,
    DeploymentSpec,
    InferenceDeploymentProvider,
)
from stateset_agents.remote.executor import RemoteExecutionError, RemoteExecutor
from stateset_agents.remote.job import JobHandle, JobStatus, RemoteJobSpec

__all__ = ["NebiusCli", "NebiusEndpointProvider", "NebiusExecutor"]

_DEFAULT_IMAGE = "pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime"
_STATE_SCHEMA_VERSION = 1
_STATUS_MAP = {
    "pending": JobStatus.PENDING,
    "creating": JobStatus.PENDING,
    "queued": JobStatus.PENDING,
    "starting": JobStatus.PENDING,
    "running": JobStatus.RUNNING,
    "succeeded": JobStatus.SUCCEEDED,
    "success": JobStatus.SUCCEEDED,
    "completed": JobStatus.SUCCEEDED,
    "failed": JobStatus.FAILED,
    "error": JobStatus.FAILED,
    "cancelled": JobStatus.CANCELLED,
    "canceled": JobStatus.CANCELLED,
}


def _default_state_dir() -> Path:
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "stateset-agents" / "nebius-jobs"


def _encode_spec(spec: RemoteJobSpec) -> str:
    payload = json.dumps(spec.to_dict(), separators=(",", ":"), sort_keys=True)
    return base64.urlsafe_b64encode(payload.encode()).decode()


def _find_string(value: Any, names: frozenset[str]) -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in names and isinstance(child, (str, int)):
                return str(child)
        for child in value.values():
            found = _find_string(child, names)
            if found:
                return found
    if isinstance(value, list):
        for child in value:
            found = _find_string(child, names)
            if found:
                return found
    return None


class NebiusCli:
    """Small, injectable wrapper around the documented Nebius AI job CLI."""

    def __init__(
        self,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        *,
        profile: str | None = None,
    ) -> None:
        self._runner = runner or subprocess.run
        self.profile = profile

    def _run(self, args: list[str], *, json_output: bool = False) -> Any:
        command = ["nebius"]
        if self.profile:
            command.extend(["--profile", self.profile])
        command.extend(args)
        if json_output:
            command.extend(["--format", "json"])
        try:
            result = self._runner(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
        except FileNotFoundError as exc:
            raise RemoteExecutionError.wrap(
                exc,
                "the Nebius CLI is not installed; follow "
                "https://docs.nebius.com/cli/install",
                provider="nebius",
            ) from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise RemoteExecutionError.wrap(
                exc, "could not execute the Nebius CLI", provider="nebius"
            ) from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "unknown error").strip()
            raise RemoteExecutionError(
                f"Nebius CLI failed ({result.returncode}): {detail}",
                provider="nebius",
            )
        if not json_output:
            return result.stdout
        try:
            return json.loads(result.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise RemoteExecutionError(
                "Nebius CLI returned invalid JSON",
                provider="nebius",
            ) from exc

    def create_job(self, args: list[str]) -> str:
        payload = self._run(["ai", "job", "create", *args], json_output=True)
        job_id = _find_string(
            payload, frozenset({"id", "job_id", "resource_id", "resourceid"})
        )
        if not job_id:
            raise RemoteExecutionError(
                "Nebius job creation returned no job id", provider="nebius"
            )
        return job_id

    def get_job(self, job_id: str) -> dict[str, Any]:
        payload = self._run(["ai", "job", "get", job_id], json_output=True)
        if not isinstance(payload, dict):
            raise RemoteExecutionError(
                "Nebius job lookup returned a non-object", provider="nebius"
            )
        return payload

    def logs(self, job_id: str) -> str:
        return str(self._run(["ai", "job", "logs", job_id]))

    def cancel(self, job_id: str) -> None:
        self._run(["ai", "job", "cancel", job_id])

    def probe(self) -> dict[str, Any]:
        payload = self._run(["ai", "job", "list"], json_output=True)
        return {"authenticated": True, "response_type": type(payload).__name__}

    def create_endpoint(self, args: list[str]) -> dict[str, Any]:
        payload = self._run(["ai", "endpoint", "create", *args], json_output=True)
        if not isinstance(payload, dict):
            raise RemoteExecutionError(
                "Nebius endpoint creation returned a non-object", provider="nebius"
            )
        return payload

    def get_endpoint(self, endpoint_id: str) -> dict[str, Any]:
        payload = self._run(["ai", "endpoint", "get", endpoint_id], json_output=True)
        if not isinstance(payload, dict):
            raise RemoteExecutionError(
                "Nebius endpoint lookup returned a non-object", provider="nebius"
            )
        return payload

    def delete_endpoint(self, endpoint_id: str) -> None:
        self._run(["ai", "endpoint", "delete", endpoint_id])


@dataclass
class _NebiusJob:
    spec: RemoteJobSpec
    input_uri: str
    output_uri: str
    status: JobStatus = JobStatus.PENDING
    logs: list[str] = field(default_factory=list)
    fetched: Path | None = None


class NebiusExecutor(RemoteExecutor):
    """Run packaged StateSet SFT jobs on Nebius Serverless AI."""

    name = "nebius"
    supported_job_kinds = frozenset({"sft"})
    durable_handles = True
    managed_deployments = True
    result_kind = "local_artifacts"
    compute_model = "serverless-container-job"
    verification_status = "code-complete-live-certification-pending"
    DEFAULT_GPU = "gpu-l40s-a"

    def __init__(
        self,
        cli: NebiusCli | None = None,
        artifact_store: S3ArtifactStore | None = None,
        *,
        bucket: str | None = None,
        subnet_id: str | None = None,
        platform: str | None = None,
        preset: str | None = None,
        image: str | None = None,
        state_dir: Path | None = None,
        poll_interval_s: float = 15.0,
    ) -> None:
        endpoint = os.environ.get("NEBIUS_S3_ENDPOINT_URL")
        self.cli = cli or NebiusCli(profile=os.environ.get("NEBIUS_PROFILE"))
        self.artifacts = artifact_store or S3ArtifactStore(endpoint_url=endpoint)
        self.bucket = bucket or os.environ.get("NEBIUS_S3_BUCKET")
        self.subnet_id = subnet_id or os.environ.get("NEBIUS_SUBNET_ID")
        self.platform = (
            platform or os.environ.get("NEBIUS_PLATFORM") or self.DEFAULT_GPU
        )
        self.preset = preset or os.environ.get("NEBIUS_PRESET") or "1gpu-8vcpu-32gb"
        self.image = image or os.environ.get("NEBIUS_JOB_IMAGE") or _DEFAULT_IMAGE
        self.state_dir = Path(state_dir or _default_state_dir())
        self.poll_interval_s = poll_interval_s
        self._jobs: dict[str, _NebiusJob] = {}

    def _require_configuration(self) -> tuple[str, str]:
        missing = [
            name
            for name, value in (
                ("NEBIUS_S3_BUCKET", self.bucket),
                ("NEBIUS_SUBNET_ID", self.subnet_id),
                (
                    "NEBIUS_S3_ACCESS_KEY_SECRET",
                    os.environ.get("NEBIUS_S3_ACCESS_KEY_SECRET"),
                ),
                (
                    "NEBIUS_S3_SECRET_KEY_SECRET",
                    os.environ.get("NEBIUS_S3_SECRET_KEY_SECRET"),
                ),
            )
            if not value
        ]
        if missing:
            raise RemoteExecutionError(
                "Nebius configuration is incomplete; set " + ", ".join(missing),
                provider=self.name,
            )
        assert self.bucket and self.subnet_id
        return self.bucket, self.subnet_id

    def _state_path(self, job_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", job_id)
        return self.state_dir / f"{safe}.json"

    def _persist(self, job_id: str, job: _NebiusJob) -> None:
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

    def _job(self, handle: JobHandle) -> _NebiusJob:
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
            job = _NebiusJob(
                spec=RemoteJobSpec.from_dict(payload["spec"]),
                input_uri=str(payload["input_uri"]),
                output_uri=str(payload["output_uri"]),
                status=JobStatus(payload.get("status", "pending")),
                logs=[str(line) for line in payload.get("logs", [])],
                fetched=Path(payload["fetched"]) if payload.get("fetched") else None,
            )
        except FileNotFoundError as exc:
            raise RemoteExecutionError(
                f"unknown Nebius job: {handle.job_id}", provider=self.name
            ) from exc
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RemoteExecutionError(
                f"invalid durable metadata for Nebius job {handle.job_id}: {exc}",
                provider=self.name,
            ) from exc
        self._jobs[handle.job_id] = job
        return job

    def submit(self, spec: RemoteJobSpec) -> JobHandle:
        self.validate_spec(spec)
        if spec.max_cost_usd is not None:
            raise RemoteExecutionError(
                "Nebius does not expose an authoritative pre-allocation price "
                "through the job CLI; --max-cost cannot be enforced safely",
                provider=self.name,
            )
        preset_gpu_count = re.match(r"^(\d+)gpu-", self.preset)
        if not preset_gpu_count or int(preset_gpu_count.group(1)) != spec.gpu_count:
            raise RemoteExecutionError(
                f"Nebius preset {self.preset!r} does not represent requested "
                f"gpu_count={spec.gpu_count}; set NEBIUS_PRESET to an exact "
                "provider preset",
                provider=self.name,
            )
        bucket, subnet_id = self._require_configuration()
        local_id = uuid.uuid4().hex[:12]
        prefix = f"stateset-agents/jobs/{local_id}"
        input_uri = f"s3://{bucket}/{prefix}/input/{spec.dataset.name}"
        output_uri = f"s3://{bucket}/{prefix}/output"
        self.artifacts.upload_file(spec.dataset, input_uri)

        from stateset_agents import __version__

        version = spec.package_version or __version__
        package = f"stateset-agents[training,cloud]=={version}"
        endpoint = os.environ.get("NEBIUS_S3_ENDPOINT_URL", "")
        command = (
            'python -m pip install --quiet "$STATESET_PACKAGE" && '
            "python -m stateset_agents.remote.worker "
            '--spec-b64 "$STATESET_SPEC_B64" '
            '--input-uri "$STATESET_INPUT_URI" '
            '--output-uri "$STATESET_OUTPUT_URI"'
        )
        if endpoint:
            command += ' --s3-endpoint-url "$STATESET_S3_ENDPOINT_URL"'
        timeout_hours = max(1, min(168, (spec.timeout_s + 3599) // 3600))
        args = [
            "--name",
            f"stateset-sft-{local_id}",
            "--image",
            self.image,
            "--container-command",
            "bash",
            "--args",
            f"-lc {command}",
            "--platform",
            spec.gpu or self.platform,
            "--preset",
            self.preset,
            "--timeout",
            f"{timeout_hours}h",
            "--subnet-id",
            subnet_id,
            "--env",
            f"STATESET_PACKAGE={package}",
            "--env",
            f"STATESET_SPEC_B64={_encode_spec(spec)}",
            "--env",
            f"STATESET_INPUT_URI={input_uri}",
            "--env",
            f"STATESET_OUTPUT_URI={output_uri}",
        ]
        if spec.container_disk_gb is not None:
            args.extend(["--disk-size", f"{spec.container_disk_gb}Gi"])
        if endpoint:
            args.extend(["--env", f"STATESET_S3_ENDPOINT_URL={endpoint}"])
        secret_env = {
            "AWS_ACCESS_KEY_ID": "NEBIUS_S3_ACCESS_KEY_SECRET",
            "AWS_SECRET_ACCESS_KEY": "NEBIUS_S3_SECRET_KEY_SECRET",
            "AWS_SESSION_TOKEN": "NEBIUS_S3_SESSION_TOKEN_SECRET",
            "HF_TOKEN": "NEBIUS_HF_TOKEN_SECRET",
        }
        for env_name, selector_name in secret_env.items():
            if selector := os.environ.get(selector_name):
                args.extend(["--env-secret", f"{env_name}={selector}"])
        try:
            job_id = self.cli.create_job(args)
        except Exception:
            self.artifacts.delete_prefix(f"s3://{bucket}/{prefix}")
            raise
        job = _NebiusJob(
            spec=spec,
            input_uri=input_uri,
            output_uri=output_uri,
            logs=[f"Nebius Serverless AI job {job_id} submitted"],
        )
        self._jobs[job_id] = job
        self._persist(job_id, job)
        return JobHandle(provider=self.name, job_id=job_id)

    def status(self, handle: JobHandle) -> JobStatus:
        job = self._job(handle)
        if job.status.is_terminal:
            return job.status
        payload = self.cli.get_job(handle.job_id)
        raw = (
            _find_string(payload, frozenset({"status", "state", "phase"})) or "pending"
        )
        normalized = raw.lower().removeprefix("job_status_").removeprefix("status_")
        job.status = _STATUS_MAP.get(normalized, JobStatus.PENDING)
        self._persist(handle.job_id, job)
        return job.status

    def logs(self, handle: JobHandle) -> Iterator[str]:
        job = self._job(handle)
        try:
            remote = self.cli.logs(handle.job_id).splitlines()
        except RemoteExecutionError as exc:
            remote = [f"log retrieval failed: {exc}"]
        yield from [*job.logs, *remote]

    def fetch(self, handle: JobHandle, dest: Path | None = None) -> Path:
        job = self._job(handle)
        if not job.status.is_terminal:
            raise RemoteExecutionError(
                f"Nebius job {handle.job_id} is not terminal", provider=self.name
            )
        destination = Path(dest or job.spec.output_dir)
        if job.fetched == destination and destination.exists():
            return destination
        written = self.artifacts.download_prefix(job.output_uri, destination)
        if not written and not job.spec.dry_run:
            raise RemoteExecutionError(
                f"Nebius job {handle.job_id} produced no artifacts",
                provider=self.name,
            )
        job.fetched = destination
        root_uri = job.input_uri.split("/input/", 1)[0]
        self.artifacts.delete_prefix(root_uri)
        self._persist(handle.job_id, job)
        return destination

    def cancel(self, handle: JobHandle) -> None:
        job = self._job(handle)
        if not job.status.is_terminal:
            self.status(handle)
        if not job.status.is_terminal:
            self.cli.cancel(handle.job_id)
            job.status = JobStatus.CANCELLED
            job.logs.append("cancellation requested")
            self._persist(handle.job_id, job)
        if job.status is JobStatus.CANCELLED:
            self.artifacts.delete_prefix(job.input_uri.split("/input/", 1)[0])

    def wait(self, handle: JobHandle, poll_interval_s: float | None = None):
        return super().wait(
            handle,
            poll_interval_s=(
                self.poll_interval_s if poll_interval_s is None else poll_interval_s
            ),
        )

    def canary(self) -> dict[str, Any]:
        """Perform a read-only authentication probe without allocating compute."""
        started = time.time()
        result = self.cli.probe()
        return {
            "provider": self.name,
            "latency_ms": round((time.time() - started) * 1000, 1),
            **result,
        }


class NebiusEndpointProvider(InferenceDeploymentProvider):
    """Deploy complete model weights as an authenticated Nebius vLLM endpoint."""

    name = "nebius"

    def __init__(
        self,
        cli: NebiusCli | None = None,
        *,
        subnet_id: str | None = None,
        image: str | None = None,
        preset: str | None = None,
    ) -> None:
        self.cli = cli or NebiusCli(profile=os.environ.get("NEBIUS_PROFILE"))
        self.subnet_id = subnet_id or os.environ.get("NEBIUS_SUBNET_ID")
        self.image = (
            image or os.environ.get("NEBIUS_VLLM_IMAGE") or "vllm/vllm-openai:v0.18.2"
        )
        self.preset = (
            preset or os.environ.get("NEBIUS_ENDPOINT_PRESET") or "1gpu-8vcpu-32gb"
        )

    def deploy(self, spec: DeploymentSpec) -> DeploymentHandle:
        from stateset_agents.remote.artifacts import parse_s3_uri

        if spec.runtime != "vllm":
            raise RemoteExecutionError(
                "Nebius endpoint integration currently supports vllm",
                provider=self.name,
            )
        if spec.min_replicas != 1 or spec.max_replicas != 1:
            raise RemoteExecutionError(
                "Nebius Serverless endpoint replicas are provider-managed; "
                "request min_replicas=max_replicas=1",
                provider=self.name,
            )
        if not self.subnet_id:
            raise RemoteExecutionError(
                "NEBIUS_SUBNET_ID is not set", provider=self.name
            )
        profile_secret = os.environ.get("NEBIUS_S3_PROFILE_SECRET", "").strip()
        if not profile_secret:
            raise RemoteExecutionError(
                "NEBIUS_S3_PROFILE_SECRET is not set; it must select SecretStash "
                "AWS profile credentials for the model bucket",
                provider=self.name,
            )
        bucket, path = parse_s3_uri(spec.weights_uri)
        if not path:
            raise RemoteExecutionError(
                "weights_uri must identify a model directory", provider=self.name
            )
        runtime_args = [
            "--model",
            f"/models/{path}",
            "--served-model-name",
            spec.model_name,
            "--host",
            "0.0.0.0",  # nosec B104 -- endpoint container must bind externally
            "--port",
            "8000",
        ]
        for key, value in sorted(spec.runtime_config.items()):
            runtime_args.extend([f"--{key}", value])
        args = [
            "--name",
            spec.name,
            "--image",
            self.image,
            "--container-command",
            "python",
            "--args",
            "-m vllm.entrypoints.openai.api_server " + shlex.join(runtime_args),
            "--container-port",
            "8000",
            "--platform",
            spec.gpu,
            "--preset",
            self.preset,
            "--subnet-id",
            self.subnet_id,
            "--volume",
            f"s3://{bucket}:/models:ro:default@{profile_secret}",
            "--auth",
            "token",
        ]
        token_secret = os.environ.get("NEBIUS_ENDPOINT_TOKEN_SECRET", "").strip()
        if not token_secret:
            raise RemoteExecutionError(
                "NEBIUS_ENDPOINT_TOKEN_SECRET is not set; requiring an explicit "
                "SecretStash token prevents an unrecoverable generated credential",
                provider=self.name,
            )
        args.extend(["--token-secret", token_secret])
        payload = self.cli.create_endpoint(args)
        endpoint_id = _find_string(
            payload,
            frozenset({"id", "endpoint_id", "resource_id", "resourceid"}),
        )
        if not endpoint_id:
            raise RemoteExecutionError(
                "Nebius endpoint creation returned no endpoint id",
                provider=self.name,
            )
        endpoint = _find_string(
            payload, frozenset({"url", "endpoint", "public_url", "publicurl"})
        )
        return DeploymentHandle(
            provider=self.name,
            deployment_id=endpoint_id,
            model_name=spec.model_name,
            endpoint=endpoint,
        )

    def status(self, handle: DeploymentHandle) -> dict[str, Any]:
        if handle.provider != self.name:
            raise RemoteExecutionError(
                "deployment handle provider mismatch", provider=self.name
            )
        return self.cli.get_endpoint(handle.deployment_id)

    def delete(self, handle: DeploymentHandle) -> None:
        if handle.provider != self.name:
            raise RemoteExecutionError(
                "deployment handle provider mismatch", provider=self.name
            )
        self.cli.delete_endpoint(handle.deployment_id)
