"""Run the fine-tune job on a RunPod GPU pod.

RunPod rents a machine, not a function — there is no managed filesystem handed
to you the way Modal's Volumes are. So the transport here is plain SSH: the
pod is created with TCP 22 exposed and the caller's public key injected, the
dataset is copied in with ``scp``, the job runs over ``ssh``, and the adapter
is copied back out. By default no persistent storage is created, so nothing
bills after the pod dies. Passing ``network_volume_id`` on the spec attaches
an *existing* RunPod network volume at ``/workspace`` instead: checkpoints
then survive pod death and the retry path resumes from the newest one rather
than restarting from scratch. The volume is caller-managed (and bills
monthly until the caller deletes it) — this executor never creates or
deletes volumes.

Termination is requested on every exit path after a usable pod ID returns,
including exceptions and the never-becomes-reachable case. An acknowledged
request does not independently verify billing stopped. A leaked pod bills
until someone notices, so uncertain cleanup retains its identity and warning.

Uses ``requests`` (already a dependency) and the system ``ssh``/``scp``
binaries, so there is no extra install beyond ``[training]``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import stat
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stateset_agents.finetune_workspace import read_json
from stateset_agents.remote.executor import (
    RemoteExecutionError,
    RemoteExecutor,
    SubmissionReceiptError,
)
from stateset_agents.remote.job import JobHandle, JobStatus, RemoteJobSpec
from stateset_agents.remote.ledger import (
    BudgetExceeded,
    CostEntry,
    check_budget,
    record_entry,
    remaining_budget_usd,
)
from stateset_agents.remote.runpod_recovery import (
    PodLease,
    RecoveryResult,
    recover_pods,
    training_pod_name,
    valid_pod_id,
)

__all__ = ["RunPodApi", "RunPodExecutor", "SshTransport", "package_pin"]

_MAX_REVIEWED_WHEEL_BYTES = 64 * 1024**2
_REVIEWED_WHEEL_CHECK = """import hashlib, os, stat, sys
descriptor = os.open(sys.argv[1], os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
with os.fdopen(descriptor, 'rb') as source:
    info = os.fstat(source.fileno())
    expected = int(sys.argv[3])
    if not stat.S_ISREG(info.st_mode) or info.st_size != expected:
        sys.exit('Reviewed wheel size or file type differs')
    digest = hashlib.sha256()
    remaining = expected
    while remaining:
        chunk = source.read(min(1024 * 1024, remaining))
        if not chunk:
            sys.exit('Reviewed wheel changed while reading')
        remaining -= len(chunk)
        digest.update(chunk)
    if source.read(1) or digest.hexdigest() != sys.argv[2]:
        sys.exit('Reviewed wheel checksum differs')
"""

_API_ROOT = "https://rest.runpod.io/v1"
GPU_CATALOG_URL = "https://api.runpod.io/graphql"
_HTTP_ATTEMPTS = 5
_HTTP_BACKOFF_S = 2.0
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
#: Where a harvest job's current-generation adapter lands on the pod.
_REMOTE_ADAPTER = "/workspace/current_adapter"
DEFAULT_RUNPOD_LEASE_DIR = (
    Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    / "stateset-agents"
    / "runpod-leases"
)


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


class _PodTimeout(RemoteExecutionError):
    """The admitted pod lifetime has elapsed."""


def _time_left(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _PodTimeout("RunPod job reached its budgeted timeout", provider="runpod")
    return remaining


def pod_hourly_rate(raw: Any) -> float | None:
    """Parse RunPod's total hourly charge for an allocated pod."""
    if raw is None or isinstance(raw, bool):
        return None
    try:
        price = float(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    return price if math.isfinite(price) and price > 0 else None


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

    @staticmethod
    def gpu_catalog() -> list[dict[str, Any]]:
        """Read the public per-GPU, per-cloud prices without account credentials."""
        import requests

        query = "{ gpuTypes { id displayName secureCloud communityCloud securePrice communityPrice } }"
        try:
            response = requests.post(GPU_CATALOG_URL, json={"query": query}, timeout=30)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise RemoteExecutionError(
                "RunPod GPU catalog request failed", provider="runpod"
            ) from exc
        data = (
            payload.get("data")
            if isinstance(payload, dict) and not payload.get("errors")
            else None
        )
        rows = data.get("gpuTypes") if isinstance(data, dict) else None
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise RemoteExecutionError(
                "Invalid RunPod GPU catalog response", provider="runpod"
            )
        return rows

    def get_gpu_rate(self, gpu_type_id: str, cloud_type: str) -> float:
        """Return the on-demand hourly catalog rate for one GPU in the requested tier."""
        matches = [row for row in self.gpu_catalog() if row.get("id") == gpu_type_id]
        tier = {"SECURE": "secure", "COMMUNITY": "community"}.get(cloud_type)
        if tier and len(matches) == 1 and matches[0].get(f"{tier}Cloud") is True:
            rate = matches[0].get(f"{tier}Price")
            if (
                isinstance(rate, (int, float))
                and not isinstance(rate, bool)
                and math.isfinite(rate)
                and rate > 0
            ):
                return float(rate)
        raise RemoteExecutionError(
            f"No usable RunPod catalog price for {gpu_type_id!r} in {cloud_type} cloud",
            provider="runpod",
        )

    @staticmethod
    def _send(
        request: Any,
        attempts: int = _HTTP_ATTEMPTS,
        backoff_s: float = _HTTP_BACKOFF_S,
        deadline: float | None = None,
    ) -> Any:
        """Issue ``request()`` with the caller-selected retry policy.

        Safe reads may retry 429 and 5xx with bounded exponential backoff;
        permanent 4xx and transport errors raise immediately. Non-idempotent
        allocation callers must use ``attempts=1``: even an HTTP 500 or 429
        can leave creation ambiguous, so a repeat may rent another pod.
        """
        import requests

        last: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = request()
                response.raise_for_status()
                return response
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                retryable = status == 429 or status >= 500
                if not retryable or attempt == attempts:
                    raise
                last = exc
                delay = backoff_s * (2 ** (attempt - 1))
                remaining = _time_left(deadline)
                time.sleep(delay if remaining is None else min(delay, remaining))
        assert last is not None  # pragma: no cover - loop always sets it
        raise last

    def create_pod(
        self,
        *,
        name: str,
        image: str,
        gpu_type_id: str,
        gpu_count: int = 1,
        ports: list[str],
        env: dict[str, str],
        container_disk_gb: int = 40,
        cloud_type: str = "SECURE",
        support_public_ip: bool = True,
        global_networking: bool = False,
        network_volume_id: str | None = None,
        volume_mount_path: str | None = None,
        data_center_id: str | None = None,
        docker_entrypoint: list[str] | None = None,
        docker_start_cmd: list[str] | None = None,
    ) -> dict[str, Any]:
        import requests

        payload: dict[str, Any] = {
            "name": name,
            "imageName": image,
            "computeType": "GPU",
            "gpuTypeIds": [gpu_type_id],
            "gpuCount": gpu_count,
            "gpuTypePriority": "availability",
            "cloudType": cloud_type,
            # Without this, a COMMUNITY pod "might not have a public IP
            # address" (RunPod's words) — it starts, reports RUNNING, and
            # never publishes one, which is indistinguishable from a hang.
            "supportPublicIp": support_public_ip,
            "containerDiskInGb": container_disk_gb,
            "ports": ports,
            "env": env,
        }
        if docker_entrypoint is not None:
            payload["dockerEntrypoint"] = docker_entrypoint
        if docker_start_cmd is not None:
            payload["dockerStartCmd"] = docker_start_cmd
        if global_networking:
            payload["globalNetworking"] = True
        if data_center_id:
            payload["dataCenterIds"] = [data_center_id]
        if network_volume_id:
            # Field names verified against the live REST API: the pod payload
            # takes ``networkVolumeId`` + ``volumeMountPath``, and datacenter
            # pinning is ``dataCenterIds`` (a list). Volumes are
            # datacenter-scoped, so the pod must be pinned to the volume's
            # datacenter or provisioning fails with "no capacity".
            payload["networkVolumeId"] = network_volume_id
            payload["volumeMountPath"] = volume_mount_path or _REMOTE_WORKDIR
        try:
            response = self._send(
                lambda: requests.post(
                    f"{self.root}/pods",
                    headers=self._headers(),
                    json=payload,
                    timeout=60,
                ),
                attempts=1,
            )
        except requests.RequestException as exc:
            status = exc.response.status_code if exc.response is not None else None
            detail = f" (HTTP {status})" if status is not None else ""
            request_id = None
            if exc.response is not None:
                request_id = exc.response.headers.get("x-request-id")
            raise RemoteExecutionError(
                "RunPod pod creation failed; allocation may exist, so inspect provider records before retrying"
                + detail,
                provider="runpod",
                attempts=1,
                request_id=request_id,
            ) from exc
        return dict(response.json())

    def get_pod(self, pod_id: str, *, deadline: float | None = None) -> dict[str, Any]:
        import requests

        response = self._send(
            lambda: requests.get(
                f"{self.root}/pods/{pod_id}",
                headers=self._headers(),
                timeout=min(60, _time_left(deadline) or 60),
            ),
            deadline=deadline,
        )
        return dict(response.json())

    def list_pods(self) -> list[dict[str, Any]]:
        """All pods on the account. The REST API returns either a bare list
        or a ``{"pods": [...]}`` envelope depending on version; accept both.
        """
        import requests

        response = self._send(
            lambda: requests.get(
                f"{self.root}/pods", headers=self._headers(), timeout=60
            )
        )
        payload = response.json()
        pods = payload.get("pods", []) if isinstance(payload, dict) else payload
        return [dict(p) for p in pods]

    def list_network_volumes(self) -> list[dict[str, Any]]:
        """All network volumes on the account (id, name, size, dataCenterId).

        Like :meth:`list_pods`, tolerates both a bare list and an
        ``{"networkVolumes": [...]}`` envelope.
        """
        import requests

        response = requests.get(
            f"{self.root}/networkvolumes", headers=self._headers(), timeout=60
        )
        response.raise_for_status()
        payload = response.json()
        volumes = (
            payload.get("networkVolumes", []) if isinstance(payload, dict) else payload
        )
        return [dict(v) for v in volumes]

    def get_network_volume(self, volume_id: str) -> dict[str, Any]:
        import requests

        response = requests.get(
            f"{self.root}/networkvolumes/{volume_id}",
            headers=self._headers(),
            timeout=60,
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
        self._deadline: float | None = None

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
                timeout=_time_left(self._deadline),
            )
            if probe.returncode == 0:
                return
            last_error = (probe.stderr or "").strip()
            time.sleep(min(5, _time_left(self._deadline) or 5))
        raise RemoteExecutionError(
            f"pod ssh at {host}:{port} did not answer within {timeout_s}s: "
            f"{last_error}",
            provider="runpod",
        )

    def upload(self, local: Path, remote: str) -> None:
        self._scp(str(local), f"{self.user}@{self._host}:{remote}")

    def upload_secret(self, secret: str, remote: str) -> None:
        """Stream a secret to a root-only remote file without local storage."""
        command = f"umask 077; cat > {shlex.quote(remote)}"
        result = subprocess.run(
            [
                "ssh",
                *self._base_opts(),
                "-p",
                str(self._port),
                f"{self.user}@{self._host}",
                command,
            ],
            input=secret,
            capture_output=True,
            text=True,
            check=False,
            timeout=_time_left(self._deadline),
        )
        if result.returncode != 0:
            raise RemoteExecutionError(
                f"secret upload failed: {(result.stderr or '').strip()}",
                provider="runpod",
            )

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
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=_time_left(self._deadline),
        )
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
            timeout=_time_left(self._deadline),
        )
        return result.returncode, (result.stdout or "") + (result.stderr or "")


@dataclass
class _PodAttempt:
    duration_s: float | None = None
    hourly_rate: float | None = None
    cleanup_acknowledged: bool = False

    @property
    def cost_usd(self) -> float | None:
        if self.duration_s is None or self.hourly_rate is None:
            return None
        return self.hourly_rate * (self.duration_s / 3600)


@dataclass
class _RunPodJob:
    spec: RemoteJobSpec
    status: JobStatus
    logs: list[str] = field(default_factory=list)
    #: Measured pod lifetime and the resulting spend. None when the provider
    #: reported no price — unknown must never render as free.
    duration_s: float | None = None
    cost_usd: float | None = None


class _PodDiedMidJob(RemoteExecutionError):
    """The pod (or its SSH transport) failed *under a running job*.

    Distinct from a training failure (the job's own non-zero exit) and from a
    pod that never became reachable: this is the interruption case —
    keepalive-detected death, connection reset, a COMMUNITY pod reclaimed —
    where re-provisioning a fresh pod and rerunning is worthwhile.
    """


class RunPodExecutor(RemoteExecutor):
    """Executes the job on a RunPod GPU pod, over SSH."""

    name = "runpod"
    supports_dry_run = True
    supports_cost_ceiling = True
    supports_sft_seed = True
    uses_packaged_training = True
    durable_submission_receipts = True
    supported_job_kinds = frozenset({"sft", "harvest"})
    compute_model = "rented-gpu-machine"
    verification_status = "live-end-to-end"
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
        ledger_path: Path | None = None,
        lease_dir: Path | None = None,
        poll_interval_s: float = 10.0,
        max_provision_attempts: int = 2,
        allow_environment_wheel: bool = True,
        reviewed_wheel_sha256: str | None = None,
        reviewed_wheel_bytes: int | None = None,
    ) -> None:
        self._api = api
        self._ssh = ssh
        self._public_key = public_key
        #: Install this locally built wheel instead of pulling the pinned
        #: version from PyPI. This is how an *unreleased* change gets verified
        #: on real hardware — the PyPI pin cannot resolve before publish.
        #: Settable via STATESET_AGENTS_WHEEL for CLI runs (the constructor
        #: argument wins) — discovered live: the flywheel's first spin died
        #: with "No module named stateset_agents.training.harvest" because
        #: the pod installed the release, which predated the module.
        if wheel is None and allow_environment_wheel:
            env_wheel = os.environ.get("STATESET_AGENTS_WHEEL", "").strip()
            wheel = Path(env_wheel) if env_wheel else None
        self.wheel = Path(wheel) if wheel else None
        self._validate_wheel_binding(
            self.wheel, reviewed_wheel_sha256, reviewed_wheel_bytes
        )
        self.reviewed_wheel_sha256 = reviewed_wheel_sha256
        self.reviewed_wheel_bytes = reviewed_wheel_bytes
        self.image = image
        #: Container disk for the pod. The default fits small models; a 30B
        #: BF16 checkpoint alone is ~60GB, so size this at roughly 2.5x the
        #: model download or the job dies mid-download with an opaque
        #: "File reconstruction error" from the HF cache writer (hit for
        #: real on meta-models/Muse-Glimmer-30B with the old fixed 40GB).
        self.container_disk_gb = container_disk_gb
        self.ready_timeout_s = ready_timeout_s
        #: Override the cost-ledger location (tests, or per-project accounting).
        self.ledger_path = ledger_path
        self.lease_dir = Path(lease_dir or DEFAULT_RUNPOD_LEASE_DIR)
        self.poll_interval_s = poll_interval_s
        #: How many pods to try before giving up when one dies *under a
        #: running job* (keepalive-detected death, connection reset — the
        #: normal failure mode of COMMUNITY/spot pods). Each retry terminates
        #: the dead pod, provisions a fresh one, re-uploads the inputs, and
        #: reruns the job.
        self.max_provision_attempts = max(1, max_provision_attempts)
        self._jobs: dict[str, _RunPodJob] = {}
        self._submission_started = False

    @staticmethod
    def _validate_wheel_binding(
        wheel: Path | None, sha256: str | None, size_bytes: int | None
    ) -> None:
        if sha256 is not None or size_bytes is not None:
            if (
                not isinstance(sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", sha256)
                or type(size_bytes) is not int
                or not 0 < size_bytes <= _MAX_REVIEWED_WHEEL_BYTES
                or wheel is None
            ):
                raise ValueError(
                    "Reviewed wheel requires a wheel, SHA256 and a positive byte count within 64 MiB"
                )

    def bind_reviewed_wheel(
        self, wheel: Path | None, *, sha256: str | None, size_bytes: int | None
    ) -> None:
        """Replace ambient installation choices before the first submission only."""
        if self._submission_started or self._jobs:
            raise ValueError(
                "Reviewed wheel binding cannot change after submission starts"
            )
        selected = Path(wheel) if wheel is not None else None
        if selected is not None and (sha256 is None or size_bytes is None):
            raise ValueError(
                "Reviewed wheel binding requires both SHA256 and byte count"
            )
        self._validate_wheel_binding(selected, sha256, size_bytes)
        self.wheel = selected
        self.reviewed_wheel_sha256 = sha256
        self.reviewed_wheel_bytes = size_bytes

    def _reviewed_commands(self, commands: list[str]) -> list[str]:
        if self.reviewed_wheel_sha256 is None:
            return commands
        assert self.wheel is not None and self.reviewed_wheel_bytes is not None
        return [
            shlex.join(
                [
                    "python",
                    "-c",
                    _REVIEWED_WHEEL_CHECK,
                    f"{_REMOTE_WORKDIR}/{self.wheel.name}",
                    self.reviewed_wheel_sha256,
                    str(self.reviewed_wheel_bytes),
                ]
            ),
            *commands,
        ]

    def _verify_reviewed_wheel(self) -> None:
        if self.reviewed_wheel_sha256 is None:
            return
        try:
            if self.wheel is None or self.wheel.is_symlink():
                raise ValueError("Reviewed wheel is unavailable")
            descriptor = os.open(
                self.wheel,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
            )
            with os.fdopen(descriptor, "rb") as source:
                before = os.fstat(source.fileno())
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_size != self.reviewed_wheel_bytes
                ):
                    raise ValueError("Reviewed wheel size or type differs")
                digest = hashlib.sha256()
                remaining = before.st_size
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("Reviewed wheel changed while reading")
                    remaining -= len(chunk)
                    digest.update(chunk)
                after = os.fstat(source.fileno())
                if (
                    source.read(1)
                    or (before.st_size, before.st_mtime_ns)
                    != (after.st_size, after.st_mtime_ns)
                    or digest.hexdigest() != self.reviewed_wheel_sha256
                ):
                    raise ValueError("Reviewed wheel checksum differs")
        except (OSError, ValueError) as exc:
            raise RemoteExecutionError(
                "Reviewed wheel bytes are unavailable or changed; prepare a fresh review before allocating compute",
                provider=self.name,
                stage="reviewed_wheel",
            ) from exc

    def _observe_pod(
        self,
        job_id: str,
        event: str,
        pod_id: str | None = None,
        *,
        attempt: int,
    ) -> None:
        self._observe_submission(
            {
                "job_id": job_id,
                "event": event,
                "resource_kind": "pod",
                "resource_id": pod_id,
                "resource_name": training_pod_name(job_id, attempt),
                "attempt": attempt,
            }
        )

    def _cleanup_submission_pod(
        self,
        api: Any,
        pod_id: str,
        job_id: str,
        logs: list[str],
        *,
        attempt: int,
        lease: PodLease,
    ) -> bool:
        """Retain cleanup evidence without masking an existing primary failure."""
        primary_error = sys.exc_info()[1]
        try:
            api.terminate_pod(pod_id)
        except Exception as exc:
            logs.append(
                f"could not confirm termination of pod {pod_id}: {exc}; "
                "cleanup lease retained; billing may still be active"
            )
            event = "cleanup_unconfirmed"
        else:
            event = "cleanup_acknowledged"
            try:
                lease.clear()
            except RemoteExecutionError:
                logs.append("Could not remove the acknowledged pod cleanup lease")
        try:
            self._observe_pod(job_id, event, pod_id, attempt=attempt)
        except SubmissionReceiptError:
            if primary_error is None:
                raise
            logs.append(
                "Could not retain pod cleanup evidence; inspect retained resources"
            )
        return event == "cleanup_acknowledged"

    # -- crash-recovery leases -------------------------------------------

    def recover_training(self, *, terminate: bool = False) -> list[RecoveryResult]:
        """Inspect training records, excluding active owners from automatic cleanup."""
        results = recover_pods(
            self.lease_dir / "attempts",
            self._require_api if terminate else None,
            kind="training",
            terminate=terminate,
        )
        # Older training processes and benchmark tools both write unguarded
        # top-level leases. Their presence alone cannot establish owner death.
        for path in sorted(self.lease_dir.glob("*.json")):
            try:
                row = read_json(path)
                if row.get("provider") == self.name:
                    results.append(
                        RecoveryResult(
                            str(row.get("pod_id") or path.name),
                            "legacy",
                            "Owner status unknown. Verify the owner has exited before manual cleanup.",
                            row,
                        )
                    )
            except FileNotFoundError:
                continue  # An older owner removed its lease during the scan.
            except (OSError, ValueError) as exc:
                results.append(RecoveryResult(path.name, "unresolved", str(exc)))
        return results

    def orphaned_leases(self) -> list[dict[str, Any]]:
        """Return local cleanup evidence for rentals without a known active owner."""
        leases = []
        for result in self.recover_training():
            if result.status == "busy":
                continue
            if result.status == "legacy":
                leases.append(dict(result.record or {}, recovery_status="legacy"))
            else:
                record = result.record or {}
                leases.append(
                    {
                        **record.get("metadata", {}),
                        "provider": self.name,
                        "pod_id": record.get("pod_id"),
                        "name": result.name,
                        "recovery_status": result.status,
                    }
                )
        return leases

    def cleanup_orphans(self) -> list[str]:
        """Terminate uniquely matched interrupted rentals and return acknowledged IDs."""
        results = self.recover_training(terminate=True)
        unresolved = [
            f"{row.name}: {row.detail}"
            for row in results
            if row.status in {"unresolved", "legacy"}
        ]
        if unresolved:
            raise RemoteExecutionError(
                "Unresolved RunPod allocations: " + "; ".join(unresolved),
                provider=self.name,
            )
        return [row.detail for row in results if row.status == "termination accepted"]

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

    def _wait_for_ssh_endpoint(
        self, api: Any, pod_id: str, *, pod_deadline: float | None = None
    ) -> tuple[str, int]:
        """Poll until the pod is RUNNING and has published its SSH port."""
        deadline = time.monotonic() + self.ready_timeout_s
        while True:
            _time_left(pod_deadline)
            pod = (
                api.get_pod(pod_id, deadline=pod_deadline)
                if pod_deadline is not None
                else api.get_pod(pod_id)
            )
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
            time.sleep(
                min(
                    self.poll_interval_s,
                    _time_left(pod_deadline) or self.poll_interval_s,
                )
            )

    def _upload_adapter(self, ssh: Any, adapter_dir: Path) -> None:
        """Tar, upload, untar — ``SshTransport.upload`` moves single files."""
        import tarfile
        import tempfile

        tar_remote = f"{_REMOTE_WORKDIR}/current_adapter.tar.gz"
        with tempfile.TemporaryDirectory() as staging:
            tar_path = Path(staging) / "current_adapter.tar.gz"
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(adapter_dir, arcname="current_adapter")
            ssh.upload(tar_path, tar_remote)
        exit_code, output = ssh.run(
            f"tar xzf {tar_remote} -C {_REMOTE_WORKDIR} && rm -f {tar_remote}"
        )
        if exit_code != 0:
            raise RemoteExecutionError(
                f"could not unpack adapter on the pod ({exit_code}): {output}",
                provider=self.name,
            )

    def _remote_commands(
        self,
        spec: RemoteJobSpec,
        dataset_remote: str,
        *,
        force_resume: bool = False,
    ) -> list[str]:
        pin = package_pin(self.wheel, spec.package_version)
        if spec.job_kind == "harvest":
            # The harvest job: same install, different module. The prompts
            # file rode the dataset upload; the adapter (if any) was shipped
            # as a tarball beforehand and lives at _REMOTE_ADAPTER.
            adapter_remote = (
                _REMOTE_ADAPTER if (spec.harvest or {}).get("adapter_dir") else None
            )
            harvest_argv = spec.harvest_cli_args(adapter_dir=adapter_remote)
            # Replace the path values before shell rendering so an identical
            # model/adapter argument is never accidentally rewritten.
            # Generated arguments are option/value pairs plus an optional
            # trailing --dry-run; inspect option positions only.
            for index in range(0, len(harvest_argv) - 1, 2):
                if harvest_argv[index] == "--prompts-file":
                    harvest_argv[index + 1] = dataset_remote
                elif harvest_argv[index] == "--output-dir":
                    harvest_argv[index + 1] = _REMOTE_OUTPUT
            harvest_args = shlex.join(harvest_argv)
            return self._reviewed_commands(
                [
                    f"pip install --quiet {shlex.quote(pin)}",
                    f"python -m stateset_agents.training.harvest {harvest_args}",
                ]
            )
        sft_argv = [
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
        if spec.seed is not None:
            sft_argv += ["--seed", str(spec.seed)]
        if spec.dry_run:
            sft_argv.append("--dry-run")
        if spec.resume or force_resume:
            # Only meaningful when checkpoints already exist remotely: a
            # network-volume retry (``force_resume``), or a caller-passed
            # ``--resume``. A fresh pod without a volume has an empty output
            # dir, so the job logs it and trains from scratch — harmless
            # either way; passed through for parity with the local provider.
            sft_argv.append("--resume")
        if spec.eval_prompts:
            sft_argv += [
                "--eval-prompts-json",
                json.dumps(spec.eval_prompts),
                "--eval-max-new-tokens",
                str(spec.eval_max_new_tokens),
            ]
        commands = [
            f"pip install --quiet {shlex.quote(pin)}",
            f"python -m stateset_agents.training.sft {shlex.join(sft_argv)}",
        ]
        return self._reviewed_commands(commands)

    def submit(self, spec: RemoteJobSpec) -> JobHandle:
        spec = copy(spec)
        self.validate_spec(spec)
        self._submission_observer_failed = False
        self._submission_started = True
        self._verify_reviewed_wheel()
        api = self._require_api()
        public_key = self._require_public_key()
        ssh = self._require_ssh()

        job_id = uuid.uuid4().hex[:12]
        handle = JobHandle(provider=self.name, job_id=job_id)
        logs: list[str] = []

        # A network volume is datacenter-scoped, so the pod must be created
        # in the volume's datacenter. Resolved once, before any pod exists.
        data_center_id: str | None = None
        if spec.network_volume_id:
            volume = api.get_network_volume(spec.network_volume_id)
            data_center_id = volume.get("dataCenterId")
            logs.append(
                f"attaching network volume {spec.network_volume_id} "
                f"({data_center_id}) at {_REMOTE_WORKDIR}"
            )

        total_duration_s = 0.0
        attempt_costs: list[float | None] = []
        for attempt in range(1, self.max_provision_attempts + 1):
            remaining = remaining_budget_usd(spec.max_cost_usd, attempt_costs)
            if remaining is not None:
                if remaining == 0:
                    raise RemoteExecutionError(
                        "RunPod retry has no remaining budget", provider=self.name
                    )
                gpu_rate = api.get_gpu_rate(
                    spec.gpu or self.DEFAULT_GPU, spec.cloud_type
                )
                try:
                    check_budget(
                        gpu_rate, spec.timeout_s, remaining, gpu_count=spec.gpu_count
                    )
                except BudgetExceeded as exc:
                    raise RemoteExecutionError(
                        f"RunPod catalog: {exc}", provider=self.name
                    ) from exc
            observation = _PodAttempt()
            status = JobStatus.FAILED
            try:
                name = training_pod_name(job_id, attempt)
                lease = PodLease(self.lease_dir / "attempts" / f"{name}.json")
                metadata = {
                    "job_id": job_id,
                    "attempt": attempt,
                    "created_at": time.time(),
                    "base_model": spec.base_model,
                    "gpu": spec.gpu or self.DEFAULT_GPU,
                    "gpu_count": spec.gpu_count,
                    "output_dir": str(spec.output_dir),
                }
                with lease.starting(metadata):
                    try:
                        self._observe_pod(job_id, "allocation_intent", attempt=attempt)
                    except BaseException:
                        try:
                            lease.clear()  # No provider request has been sent.
                        except RemoteExecutionError:
                            logs.append(
                                f"Could not retire unused allocation intent {name}"
                            )
                        raise
                    status = self._run_attempt(
                        api,
                        ssh,
                        public_key,
                        spec,
                        job_id,
                        logs,
                        lease=lease,
                        observation=observation,
                        remaining_budget=remaining,
                        data_center_id=data_center_id,
                        # A retry only sees prior checkpoints when they landed on
                        # a network volume; without one they died with the pod.
                        force_resume=bool(spec.network_volume_id) and attempt > 1,
                        attempt=attempt,
                    )
            except _PodDiedMidJob as exc:
                if not observation.cleanup_acknowledged:
                    raise RemoteExecutionError(
                        "RunPod cleanup is unconfirmed; inspect the retained allocation and provider account before starting replacement work",
                        provider=self.name,
                        stage="cleanup_unconfirmed",
                    ) from exc
                if self._submission_observer_failed:
                    raise
                logs.append(str(exc))
                if attempt >= self.max_provision_attempts:
                    raise RemoteExecutionError(
                        f"pod died mid-job on every attempt "
                        f"({self.max_provision_attempts}); giving up: {exc}",
                        provider=self.name,
                    ) from exc
                if spec.network_volume_id:
                    # The volume mounted at /workspace outlived the pod, so
                    # the checkpoint-* directories are still there: the fresh
                    # pod resumes instead of restarting — an interruption
                    # costs at most one epoch, not the whole run.
                    logs.append(
                        f"provisioning a fresh pod and resuming from the "
                        f"newest checkpoint on network volume "
                        f"{spec.network_volume_id} (attempt {attempt + 1}/"
                        f"{self.max_provision_attempts})"
                    )
                else:
                    # Without a network volume the dead pod's checkpoint-*
                    # directories lived on its container disk and died with
                    # it, so cross-pod `--resume` cannot work: the rerun is
                    # from scratch. Pass --network-volume-id to avoid this.
                    logs.append(
                        f"provisioning a fresh pod and restarting training "
                        f"from scratch (attempt {attempt + 1}/"
                        f"{self.max_provision_attempts})"
                    )
                continue
            finally:
                # One row per returned allocation, including attempts that
                # fail before SSH or are followed by a successful replacement.
                attempt_duration_s = observation.duration_s
                attempt_cost_usd = observation.cost_usd
                if attempt_duration_s is not None:
                    total_duration_s += attempt_duration_s
                    attempt_costs.append(attempt_cost_usd)
                    try:
                        self._record_cost(spec, job_id, observation, status)
                    except Exception:
                        # Accounting must not replace the original receipt,
                        # transport, budget or lease diagnosis.
                        logs.append(
                            "Could not retain RunPod attempt cost; inspect provider records"
                        )
            if remaining is not None and not observation.cleanup_acknowledged:
                raise RemoteExecutionError(
                    "RunPod cleanup is unconfirmed; the remaining budget cannot be determined",
                    provider=self.name,
                    stage="cleanup_unconfirmed",
                )
            self._jobs[job_id] = _RunPodJob(
                spec,
                status,
                logs,
                duration_s=total_duration_s,
                cost_usd=(
                    None
                    if None in attempt_costs
                    else math.fsum(c for c in attempt_costs if c is not None)
                ),
            )
            return handle

        raise AssertionError("unreachable")  # pragma: no cover

    def _record_cost(
        self,
        spec: RemoteJobSpec,
        job_id: str,
        observation: _PodAttempt,
        status: JobStatus,
    ) -> None:
        """Retain one allocation's quote-derived observed cost, even on failure."""
        record_entry(
            CostEntry(
                provider=self.name,
                job_id=job_id,
                base_model=spec.base_model,
                gpu=spec.gpu or self.DEFAULT_GPU,
                gpu_count=spec.gpu_count,
                cost_per_hr=observation.hourly_rate,
                duration_s=(
                    round(observation.duration_s, 1)
                    if observation.duration_s is not None
                    else None
                ),
                cost_usd=observation.cost_usd,
                status=status.value,
            ),
            path=self.ledger_path,
        )

    def _run_attempt(
        self,
        api: Any,
        ssh: Any,
        public_key: str,
        spec: RemoteJobSpec,
        job_id: str,
        logs: list[str],
        *,
        lease: PodLease,
        observation: _PodAttempt,
        remaining_budget: float | None,
        data_center_id: str | None = None,
        force_resume: bool = False,
        attempt: int = 1,
    ) -> JobStatus:
        """One full provision → upload → run → download cycle on a new pod.

        Raises :class:`_PodDiedMidJob` when the pod fails *under* the job —
        the caller may then retry on a fresh pod. Every other outcome is
        terminal: a returned :class:`JobStatus`, or a plain
        :class:`RemoteExecutionError`. Cleanup is requested for every known
        allocation, without claiming that an acknowledgement ends billing.
        """
        deadline = (
            time.monotonic() + spec.timeout_s if remaining_budget is not None else None
        )
        pod = api.create_pod(
            name=lease.path.stem,
            image=self.image,
            gpu_type_id=spec.gpu or self.DEFAULT_GPU,
            gpu_count=spec.gpu_count,
            ports=["22/tcp"],
            env={"PUBLIC_KEY": public_key, "SSH_PUBLIC_KEY": public_key},
            container_disk_gb=spec.container_disk_gb or self.container_disk_gb,
            cloud_type=spec.cloud_type,
            network_volume_id=spec.network_volume_id,
            volume_mount_path=_REMOTE_WORKDIR if spec.network_volume_id else None,
            data_center_id=data_center_id,
        )
        pod_id = pod.get("id") if isinstance(pod, dict) else None
        if not valid_pod_id(pod_id):
            raise RemoteExecutionError(
                "RunPod returned no usable pod identity; allocation may exist, so inspect provider records before retrying",
                provider=self.name,
            )
        # Lease timestamps use wall time; elapsed accounting uses monotonic
        # time from the returned identity through the cleanup request. Neither
        # boundary independently verifies the provider's actual billing window.
        pod_started_monotonic = time.monotonic()
        cost_per_hr = pod_hourly_rate(pod.get("costPerHr"))
        observation.hourly_rate = cost_per_hr
        try:
            lease.record_id(pod_id)
            self._observe_pod(job_id, "allocated", pod_id, attempt=attempt)
            logs.append(
                f"created pod {pod_id} "
                f"({spec.gpu_count}x {spec.gpu or self.DEFAULT_GPU}, "
                f"{spec.cloud_type}"
                + (f", ${cost_per_hr}/hr)" if cost_per_hr is not None else ")")
            )

            # Refuse before work, retaining the allocated pod's observed cost.
            # Failed or ambiguous termination can still leave billing active.
            try:
                check_budget(
                    cost_per_hr,
                    spec.timeout_s,
                    remaining_budget,
                    # RunPod quotes the whole Pod, not each GPU separately.
                    gpu_count=1,
                )
            except BudgetExceeded as exc:
                raise RemoteExecutionError(
                    str(exc), provider=self.name, pod_id=pod_id
                ) from exc

            # Everything from here must terminate the pod, whatever happens.
            try:
                if deadline is not None and isinstance(ssh, SshTransport):
                    ssh = copy(ssh)
                    ssh._deadline = deadline
                host, port = self._wait_for_ssh_endpoint(
                    api, pod_id, pod_deadline=deadline
                )
                logs.append(f"pod reachable at {host}:{port}")
                ssh.wait_until_reachable(host, port, self.ready_timeout_s)

                # The job phase: any transport failure past this point means the
                # pod (or its network) died under us — the retryable case.
                try:
                    dataset_remote = f"{_REMOTE_WORKDIR}/{spec.dataset.name}"
                    ssh.upload(spec.dataset, dataset_remote)
                    logs.append(f"uploaded {spec.dataset.name}")

                    adapter_dir = (spec.harvest or {}).get("adapter_dir")
                    if spec.job_kind == "harvest" and adapter_dir:
                        self._upload_adapter(ssh, Path(adapter_dir))
                        logs.append(
                            f"uploaded adapter {adapter_dir} -> {_REMOTE_ADAPTER}"
                        )

                    if self.wheel:
                        ssh.upload(self.wheel, f"{_REMOTE_WORKDIR}/{self.wheel.name}")
                        logs.append(f"uploaded {self.wheel.name}")

                    exit_code = 0
                    commands = self._remote_commands(
                        spec, dataset_remote, force_resume=force_resume
                    )
                    for command in commands:
                        _time_left(deadline)
                        exit_code, output = ssh.run(command)
                        _time_left(deadline)
                        logs.extend(output.splitlines())
                        if exit_code != 0:
                            break

                    if exit_code == 255:
                        # 255 is ssh's OWN exit code (the client failed), not the
                        # remote command's: keepalive-detected pod death lands
                        # here. Observed live — RunPod restarted a pod under a
                        # running job and its IP changed.
                        raise _PodDiedMidJob(
                            f"ssh transport to pod {pod_id} died mid-job (ssh exit 255)",
                            provider=self.name,
                        )
                except (SubmissionReceiptError, _PodDiedMidJob, _PodTimeout):
                    raise
                except Exception as exc:
                    if deadline is not None and isinstance(
                        exc, subprocess.TimeoutExpired
                    ):
                        raise _PodTimeout(
                            "RunPod SSH operation timed out", provider=self.name
                        ) from exc
                    raise _PodDiedMidJob(
                        f"pod {pod_id} failed mid-job: {exc}", provider=self.name
                    ) from exc

                self._observe_pod(job_id, "execution_finished", pod_id, attempt=attempt)
                if exit_code != 0:
                    logs.append(f"remote job exited {exit_code}")
                    # The eval gate fails a job AFTER saving its artifacts, so a
                    # failed run may still hold the adapter + eval_results.json —
                    # and the pod is terminated on the way out, so this is the
                    # last chance to save them. Observed live: wait()'s
                    # fetch-on-failure was defeated by fetch()'s own
                    # success-only guard, and a trained gen-2 adapter died with
                    # its pod. Download best-effort; the failure stands.
                    try:
                        salvaged = ssh.download_dir(_REMOTE_OUTPUT, spec.output_dir)
                        if salvaged:
                            logs.append(
                                f"salvaged {len(salvaged)} artifact file(s) "
                                f"from the failed job to {spec.output_dir}"
                            )
                    except Exception as exc:  # noqa: BLE001 - salvage must not mask
                        logs.append(f"could not salvage artifacts: {exc}")
                    return JobStatus.FAILED

                if spec.dry_run:
                    return JobStatus.SUCCEEDED

                try:
                    downloaded = ssh.download_dir(_REMOTE_OUTPUT, spec.output_dir)
                except Exception as exc:
                    # Reported, not raised (and not retried — a rerun would burn
                    # a full training run to fix a copy): by this point the job
                    # itself has already succeeded, and raising would discard
                    # every line of its output — leaving a stack trace and no
                    # evidence.
                    logs.append(f"failed to download adapter: {exc}")
                    return JobStatus.FAILED

                if not downloaded:
                    logs.append(
                        "job exited cleanly but produced no artifacts — "
                        f"nothing was written to {_REMOTE_OUTPUT}"
                    )
                    return JobStatus.FAILED

                logs.append(
                    f"downloaded {len(downloaded)} file(s) to {spec.output_dir}"
                )
                return JobStatus.SUCCEEDED

            except RemoteExecutionError:
                raise
            except Exception as exc:
                raise RemoteExecutionError.wrap(
                    exc, "RunPod job failed", provider=self.name, pod_id=pod_id
                ) from exc
        finally:
            # Unconditional: an orphaned pod bills by the hour.
            try:
                observation.cleanup_acknowledged = self._cleanup_submission_pod(
                    api, pod_id, job_id, logs, attempt=attempt, lease=lease
                )
            finally:
                # This is the observed interval, not proof that billing ended.
                # Retain it even when cleanup evidence cannot be persisted.
                observation.duration_s = time.monotonic() - pod_started_monotonic
                logs.append(
                    f"pod {pod_id} observed for {observation.duration_s:.0f}s through cleanup request"
                    + (
                        f" (~${observation.cost_usd:.2f})"
                        if observation.cost_usd is not None
                        else " (cost unknown)"
                    )
                )

    # -- executor interface -----------------------------------------------

    def job_cost(self, handle: JobHandle) -> tuple[float | None, float | None]:
        """Observed allocation time and estimated cost across finished attempts.

        Cleanup acknowledgement does not prove billing stopped. Cost remains
        unknown if any allocated attempt had no usable provider quote.
        """
        job = self._job(handle)
        return (job.duration_s, job.cost_usd)

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

    def _is_preview(self, handle: JobHandle) -> bool:
        return self._job(handle).spec.dry_run

    def fetch(self, handle: JobHandle, dest: Path | None = None) -> Path | None:
        job = self._job(handle)
        if not job.status.is_terminal:
            raise RemoteExecutionError(
                f"job {handle.job_id} is not finished; nothing to fetch",
                provider=self.name,
            )
        # A FAILED job may still have salvaged artifacts (the eval gate
        # fails AFTER saving them); point at the output dir either way.
        if job.status is JobStatus.SUCCEEDED and job.spec.dry_run:
            return None
        return dest or job.spec.output_dir

    def cancel(self, handle: JobHandle) -> None:
        job = self._job(handle)
        if not job.status.is_terminal:
            job.status = JobStatus.CANCELLED
