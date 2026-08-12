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

import json
import os
import shlex
import subprocess
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stateset_agents.remote.executor import RemoteExecutionError, RemoteExecutor
from stateset_agents.remote.job import JobHandle, JobStatus, RemoteJobSpec

__all__ = ["RunPodApi", "RunPodExecutor", "SshTransport", "package_pin"]

_API_ROOT = "https://rest.runpod.io/v1"
#: Official RunPod PyTorch image — ships a preconfigured sshd that reads
#: PUBLIC_KEY, so no custom start command is needed.
#:
#: torch 2.8 specifically: the older ``runpod/pytorch:2.4.0`` image fails at
#: import with ``cannot import name 'DTensor' from torch.distributed.tensor``
#: because transformers>=4.57.1 requires a torch that exposes DTensor there
#: (2.6+). The pod provisions and the job starts before hitting it, so nothing
#: short of a real run catches it — see the guard in
#: tests/unit/test_remote_runpod_executor.py.
_DEFAULT_IMAGE = "runpod/pytorch:1.1.0-rc.154-cu1290-torch280-ubuntu2204"
_REMOTE_WORKDIR = "/workspace"
_REMOTE_OUTPUT = "/workspace/out"


def package_pin(wheel: Path | None, package_version: str | None) -> str:
    """The pip requirement a pod installs to get this package.

    A locally built ``wheel`` (already uploaded to the pod's workdir) wins —
    that is how an *unreleased* change gets verified on real hardware, since
    the PyPI pin cannot resolve before publish. Otherwise pin the published
    package to ``package_version``, or float when no version is known.
    """
    if wheel:
        return f"{_REMOTE_WORKDIR}/{wheel.name}[training]"
    if package_version:
        return f"stateset-agents[training]=={package_version}"
    return "stateset-agents[training]"


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
            # Without keepalives a dead peer (pod crash/restart mid-job —
            # observed live: RunPod restarted a pod under a running job and
            # its IP changed) leaves the blocking ssh read hung for hours.
            # 12 x 10s: gone-for-2-minutes means gone.
            "-o",
            "ServerAliveInterval=10",
            "-o",
            "ServerAliveCountMax=12",
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
                [
                    "ssh",
                    *self._base_opts(),
                    "-p",
                    str(port),
                    "-o",
                    "ConnectTimeout=10",
                    f"{self.user}@{host}",
                    "true",
                ],
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
        """Copy a remote directory's contents into ``local_dir``.

        Deliberately *not* the ``remote:/path/.`` form: OpenSSH 9 runs scp
        over SFTP, which rejects ``.`` as a filename ("unexpected filename:
        ."). Instead the directory is fetched whole into a staging area and
        its contents moved up, which is portable across scp implementations.
        """
        import shutil
        import tempfile

        remote = remote_dir.rstrip("/")
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as staging:
            self._scp(
                f"{self.user}@{self._host}:{remote}",
                staging,
                recursive=True,
            )
            fetched = Path(staging) / Path(remote).name
            if not fetched.exists():
                return []
            for item in fetched.iterdir():
                destination = local_dir / item.name
                if destination.exists():
                    shutil.rmtree(destination, ignore_errors=True)
                    destination.unlink(missing_ok=True)
                shutil.move(str(item), str(destination))

        return [p for p in local_dir.rglob("*") if p.is_file()]

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
    #: RunPod's own GPU vocabulary. 16 GB is enough for a small-model LoRA
    #: SFT and is among the cheapest widely-available options.
    DEFAULT_GPU = "NVIDIA RTX A4000"

    def __init__(
        self,
        api: Any = None,
        ssh: Any = None,
        *,
        public_key: str | None = None,
        image: str = _DEFAULT_IMAGE,
        wheel: Path | None = None,
        container_disk_gb: int = 40,
        ready_timeout_s: int = 600,
        poll_interval_s: float = 10.0,
    ) -> None:
        self._api = api
        self._ssh = ssh
        self._public_key = public_key
        #: Install this locally built wheel instead of pulling the pinned
        #: version from PyPI. This is how an *unreleased* change gets verified
        #: on real hardware — the PyPI pin cannot resolve before publish.
        self.wheel = Path(wheel) if wheel else None
        self.image = image
        #: Container disk for the pod. The default fits small models; a 30B
        #: BF16 checkpoint alone is ~60GB, so size this at roughly 2.5x the
        #: model download or the job dies mid-download with an opaque
        #: "File reconstruction error" from the HF cache writer (hit for
        #: real on meta-models/Muse-Glimmer-30B with the old fixed 40GB).
        self.container_disk_gb = container_disk_gb
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
        pin = package_pin(self.wheel, spec.package_version)
        args = " ".join(
            [
                "--dataset",
                dataset_remote,
                "--base-model",
                spec.base_model,
                "--output-dir",
                _REMOTE_OUTPUT,
                "--num-epochs",
                str(spec.num_epochs),
                "--lora-r",
                str(spec.lora_r),
                "--lora-alpha",
                str(spec.lora_alpha),
                "--learning-rate",
                str(spec.learning_rate),
                "--max-length",
                str(spec.max_length),
                "--per-device-batch-size",
                str(spec.per_device_batch_size),
                "--gradient-accumulation-steps",
                str(spec.gradient_accumulation_steps),
            ]
        )
        if spec.dry_run:
            args += " --dry-run"
        if spec.eval_prompts:
            # The command travels through ssh + bash, so the JSON blob must
            # survive a real shell — hence shlex.quote, not manual quoting.
            args += f" --eval-prompts-json {shlex.quote(json.dumps(spec.eval_prompts))}"
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
            gpu_type_id=spec.gpu or self.DEFAULT_GPU,
            ports=["22/tcp"],
            env={"PUBLIC_KEY": public_key, "SSH_PUBLIC_KEY": public_key},
            container_disk_gb=spec.container_disk_gb or self.container_disk_gb,
        )
        pod_id = str(pod["id"])
        logs.append(f"created pod {pod_id} ({spec.gpu or self.DEFAULT_GPU})")

        # Everything from here must terminate the pod, whatever happens.
        try:
            host, port = self._wait_for_ssh_endpoint(api, pod_id)
            logs.append(f"pod reachable at {host}:{port}")
            ssh.wait_until_reachable(host, port, self.ready_timeout_s)

            dataset_remote = f"{_REMOTE_WORKDIR}/{spec.dataset.name}"
            ssh.upload(spec.dataset, dataset_remote)
            logs.append(f"uploaded {spec.dataset.name}")

            if self.wheel:
                ssh.upload(self.wheel, f"{_REMOTE_WORKDIR}/{self.wheel.name}")
                logs.append(f"uploaded {self.wheel.name}")

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

            try:
                downloaded = ssh.download_dir(_REMOTE_OUTPUT, spec.output_dir)
            except Exception as exc:
                # Reported, not raised: by this point the job itself has
                # already succeeded, and raising would discard every line of
                # its output — leaving a stack trace and no evidence.
                logs.append(f"failed to download adapter: {exc}")
                self._jobs[job_id] = _RunPodJob(spec, JobStatus.FAILED, logs)
                return handle

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
