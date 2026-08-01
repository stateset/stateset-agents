"""Run the fine-tune job on a RunPod GPU pod.

RunPod rents a machine, not a function — there is no managed filesystem handed
to you the way Modal's Volumes are. So the transport here is plain SSH: the
pod is created with TCP 22 exposed and the caller's public key injected, the
dataset is copied in with ``scp``, the job runs over ``ssh``, and the adapter
is copied back out. No persistent storage is created, so nothing bills after
the pod dies.

**The pod is terminated on every exit path**, including exceptions and the
never-becomes-reachable case. A leaked pod bills by the hour until someone
notices, which makes termination the most important behaviour in this file —
not an afterthought in a happy-path ``finally``.

Uses ``requests`` (already a dependency) and the system ``ssh``/``scp``
binaries, so there is no extra install beyond ``[training]``.
"""

from __future__ import annotations

import os
import subprocess
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stateset_agents.remote.executor import RemoteExecutionError, RemoteExecutor
from stateset_agents.remote.job import JobHandle, JobStatus, RemoteJobSpec

__all__ = ["RunPodApi", "RunPodExecutor", "SshTransport"]

_API_ROOT = "https://rest.runpod.io/v1"
#: Official RunPod PyTorch image — ships a preconfigured sshd that reads
#: PUBLIC_KEY, so no custom start command is needed.
_DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
_REMOTE_WORKDIR = "/workspace"
_REMOTE_OUTPUT = "/workspace/out"


class RunPodApi:
    """Thin HTTP wrapper over the RunPod REST pod endpoints."""

    def __init__(self, api_key: str, root: str = _API_ROOT) -> None:
        self.api_key = api_key
        self.root = root

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def create_pod(
        self,
        *,
        name: str,
        image: str,
        gpu_type_id: str,
        ports: list[str],
        env: dict[str, str],
        container_disk_gb: int = 40,
    ) -> dict[str, Any]:
        import requests

        response = requests.post(
            f"{self.root}/pods",
            headers=self._headers(),
            json={
                "name": name,
                "imageName": image,
                "gpuTypeIds": [gpu_type_id],
                "gpuCount": 1,
                "cloudType": "SECURE",
                "containerDiskInGb": container_disk_gb,
                "ports": ports,
                "env": env,
            },
            timeout=60,
        )
        response.raise_for_status()
        return dict(response.json())

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        import requests

        response = requests.get(
            f"{self.root}/pods/{pod_id}", headers=self._headers(), timeout=60
        )
        response.raise_for_status()
        return dict(response.json())

    def terminate_pod(self, pod_id: str) -> None:
        import requests

        response = requests.delete(
            f"{self.root}/pods/{pod_id}", headers=self._headers(), timeout=60
        )
        response.raise_for_status()


class SshTransport:
    """File and command transport over the system ``ssh``/``scp`` binaries."""

    def __init__(self, user: str = "root", key_path: Path | None = None) -> None:
        self.user = user
        self.key_path = key_path
        self._host: str | None = None
        self._port: int | None = None

    def _base_opts(self) -> list[str]:
        opts = [
            # The pod is created fresh for this job and destroyed after, so
            # its host key is new every time and pinning it is meaningless.
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "LogLevel=ERROR",
        ]
        if self.key_path:
            opts += ["-i", str(self.key_path)]
        return opts

    def wait_until_reachable(self, host: str, port: int, timeout_s: int) -> None:
        """Block until sshd answers, or raise once ``timeout_s`` elapses."""
        self._host, self._port = host, port
        deadline = time.monotonic() + timeout_s
        last_error = ""
        while time.monotonic() < deadline:
            probe = subprocess.run(
                ["ssh", *self._base_opts(), "-p", str(port), "-o",
                 "ConnectTimeout=10", f"{self.user}@{host}", "true"],
                capture_output=True,
                text=True,
                check=False,
            )
            if probe.returncode == 0:
                return
            last_error = (probe.stderr or "").strip()
            time.sleep(5)
        raise RemoteExecutionError(
            f"pod ssh at {host}:{port} did not answer within {timeout_s}s: "
            f"{last_error}",
            provider="runpod",
        )

    def upload(self, local: Path, remote: str) -> None:
        self._scp(str(local), f"{self.user}@{self._host}:{remote}")

    def download_dir(self, remote_dir: str, local_dir: Path) -> list[Path]:
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        self._scp(
            f"{self.user}@{self._host}:{remote_dir.rstrip('/')}/.",
            str(local_dir),
            recursive=True,
        )
        return [p for p in Path(local_dir).rglob("*") if p.is_file()]

    def _scp(self, src: str, dest: str, recursive: bool = False) -> None:
        cmd = ["scp", *self._base_opts(), "-P", str(self._port)]
        if recursive:
            cmd.append("-r")
        cmd += [src, dest]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RemoteExecutionError(
                f"scp failed: {(result.stderr or '').strip()}", provider="runpod"
            )

    def run(self, command: str) -> tuple[int, str]:
        result = subprocess.run(
            [
                "ssh",
                *self._base_opts(),
                "-p",
                str(self._port),
                f"{self.user}@{self._host}",
                command,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode, (result.stdout or "") + (result.stderr or "")


@dataclass
class _RunPodJob:
    spec: RemoteJobSpec
    status: JobStatus
    logs: list[str] = field(default_factory=list)


class RunPodExecutor(RemoteExecutor):
    """Executes the job on a RunPod GPU pod, over SSH."""

    name = "runpod"

    def __init__(
        self,
        api: Any = None,
        ssh: Any = None,
        *,
        public_key: str | None = None,
        image: str = _DEFAULT_IMAGE,
        ready_timeout_s: int = 600,
        poll_interval_s: float = 10.0,
    ) -> None:
        self._api = api
        self._ssh = ssh
        self._public_key = public_key
        self.image = image
        self.ready_timeout_s = ready_timeout_s
        self.poll_interval_s = poll_interval_s
        self._jobs: dict[str, _RunPodJob] = {}

    # -- lazily resolved collaborators ------------------------------------

    def _require_api(self) -> Any:
        if self._api is not None:
            return self._api
        key = os.environ.get("RUNPOD_API_KEY", "").strip()
        if not key:
            raise RemoteExecutionError(
                "RUNPOD_API_KEY is not set; create a key at "
                "https://console.runpod.io/user/settings and export it",
                provider=self.name,
            )
        self._api = RunPodApi(key)
        return self._api

    def _require_public_key(self) -> str:
        if self._public_key:
            return self._public_key
        for candidate in ("id_ed25519.pub", "id_rsa.pub"):
            path = Path.home() / ".ssh" / candidate
            if path.exists():
                self._public_key = path.read_text().strip()
                return self._public_key
        raise RemoteExecutionError(
            "no SSH public key found (~/.ssh/id_ed25519.pub or id_rsa.pub); "
            "RunPod needs one to grant access to the pod",
            provider=self.name,
        )

    def _require_ssh(self) -> Any:
        if self._ssh is None:
            self._ssh = SshTransport()
        return self._ssh

    # -- lifecycle ---------------------------------------------------------

    def _wait_for_ssh_endpoint(self, api: Any, pod_id: str) -> tuple[str, int]:
        """Poll until the pod is RUNNING and has published its SSH port."""
        deadline = time.monotonic() + self.ready_timeout_s
        while True:
            pod = api.get_pod(pod_id)
            ip = pod.get("publicIp")
            port = (pod.get("portMappings") or {}).get("22")
            if pod.get("desiredStatus") == "RUNNING" and ip and port:
                return str(ip), int(port)
            if time.monotonic() >= deadline:
                raise RemoteExecutionError(
                    f"pod {pod_id} never became reachable within "
                    f"{self.ready_timeout_s}s (last status "
                    f"{pod.get('desiredStatus')!r})",
                    provider=self.name,
                )
            time.sleep(self.poll_interval_s)

    def _remote_commands(self, spec: RemoteJobSpec, dataset_remote: str) -> list[str]:
        version = spec.package_version
        pin = f"stateset-agents[training]=={version}" if version else (
            "stateset-agents[training]"
        )
        args = " ".join(
            [
                "--dataset", dataset_remote,
                "--base-model", spec.base_model,
                "--output-dir", _REMOTE_OUTPUT,
                "--num-epochs", str(spec.num_epochs),
                "--lora-r", str(spec.lora_r),
                "--lora-alpha", str(spec.lora_alpha),
                "--learning-rate", str(spec.learning_rate),
                "--max-length", str(spec.max_length),
                "--per-device-batch-size", str(spec.per_device_batch_size),
                "--gradient-accumulation-steps",
                str(spec.gradient_accumulation_steps),
            ]
        )
        if spec.dry_run:
            args += " --dry-run"
        return [
            f"pip install --quiet '{pin}'",
            f"python -m stateset_agents.training.sft {args}",
        ]

    def submit(self, spec: RemoteJobSpec) -> JobHandle:
        api = self._require_api()
        public_key = self._require_public_key()
        ssh = self._require_ssh()

        job_id = uuid.uuid4().hex[:12]
        handle = JobHandle(provider=self.name, job_id=job_id)
        logs: list[str] = []

        pod = api.create_pod(
            name=f"stateset-sft-{job_id}",
            image=self.image,
            gpu_type_id=spec.gpu,
            ports=["22/tcp"],
            env={"PUBLIC_KEY": public_key, "SSH_PUBLIC_KEY": public_key},
        )
        pod_id = str(pod["id"])
        logs.append(f"created pod {pod_id} ({spec.gpu})")

        # Everything from here must terminate the pod, whatever happens.
        try:
            host, port = self._wait_for_ssh_endpoint(api, pod_id)
            logs.append(f"pod reachable at {host}:{port}")
            ssh.wait_until_reachable(host, port, self.ready_timeout_s)

            dataset_remote = f"{_REMOTE_WORKDIR}/{spec.dataset.name}"
            ssh.upload(spec.dataset, dataset_remote)
            logs.append(f"uploaded {spec.dataset.name}")

            exit_code = 0
            for command in self._remote_commands(spec, dataset_remote):
                exit_code, output = ssh.run(command)
                logs.extend(output.splitlines())
                if exit_code != 0:
                    break

            if exit_code != 0:
                logs.append(f"remote job exited {exit_code}")
                self._jobs[job_id] = _RunPodJob(spec, JobStatus.FAILED, logs)
                return handle

            if spec.dry_run:
                self._jobs[job_id] = _RunPodJob(spec, JobStatus.SUCCEEDED, logs)
                return handle

            downloaded = ssh.download_dir(_REMOTE_OUTPUT, spec.output_dir)
            if not downloaded:
                logs.append(
                    "job exited cleanly but produced no artifacts — "
                    f"nothing was written to {_REMOTE_OUTPUT}"
                )
                self._jobs[job_id] = _RunPodJob(spec, JobStatus.FAILED, logs)
                return handle

            logs.append(f"downloaded {len(downloaded)} file(s) to {spec.output_dir}")
            self._jobs[job_id] = _RunPodJob(spec, JobStatus.SUCCEEDED, logs)
            return handle

        except RemoteExecutionError:
            raise
        except Exception as exc:
            raise RemoteExecutionError.wrap(
                exc, "RunPod job failed", provider=self.name, pod_id=pod_id
            ) from exc
        finally:
            # Unconditional: an orphaned pod bills by the hour.
            try:
                api.terminate_pod(pod_id)
            except Exception:  # never mask the original failure
                pass

    # -- executor interface -----------------------------------------------

    def _job(self, handle: JobHandle) -> _RunPodJob:
        try:
            return self._jobs[handle.job_id]
        except KeyError:
            raise RemoteExecutionError(
                f"unknown job: {handle.job_id}", provider=self.name
            ) from None

    def status(self, handle: JobHandle) -> JobStatus:
        return self._job(handle).status

    def logs(self, handle: JobHandle) -> Iterator[str]:
        yield from self._job(handle).logs

    def fetch(self, handle: JobHandle, dest: Path | None = None) -> Path:
        job = self._job(handle)
        if job.status is not JobStatus.SUCCEEDED:
            raise RemoteExecutionError(
                f"job {handle.job_id} is not finished successfully; nothing to fetch",
                provider=self.name,
            )
        return dest or job.spec.output_dir

    def cancel(self, handle: JobHandle) -> None:
        job = self._job(handle)
        if not job.status.is_terminal:
            job.status = JobStatus.CANCELLED
