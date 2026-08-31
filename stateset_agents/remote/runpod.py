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
from stateset_agents.remote.ledger import (
    BudgetExceeded,
    CostEntry,
    check_budget,
    estimate_cost_usd,
    record_entry,
)

__all__ = ["RunPodApi", "RunPodExecutor", "SshTransport", "package_pin"]

_API_ROOT = "https://rest.runpod.io/v1"
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
    def _send(
        request: Any,
        attempts: int = _HTTP_ATTEMPTS,
        backoff_s: float = _HTTP_BACKOFF_S,
    ) -> Any:
        """Issue ``request()`` retrying transient HTTP answers.

        RunPod's REST API intermittently answers 500 on /v1/pods (observed
        repeatedly live; one such 500 killed a whole serve attempt during
        provisioning). Retry 429 and 5xx with a bounded exponential backoff;
        permanent 4xx responses still raise immediately. Transport errors are
        deliberately not retried here: after an ambiguous POST disconnect we
        cannot know whether a billable pod was created.
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
                time.sleep(backoff_s * (2 ** (attempt - 1)))
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
        if network_volume_id:
            # Field names verified against the live REST API: the pod payload
            # takes ``networkVolumeId`` + ``volumeMountPath``, and datacenter
            # pinning is ``dataCenterIds`` (a list). Volumes are
            # datacenter-scoped, so the pod must be pinned to the volume's
            # datacenter or provisioning fails with "no capacity".
            payload["networkVolumeId"] = network_volume_id
            payload["volumeMountPath"] = volume_mount_path or _REMOTE_WORKDIR
            if data_center_id:
                payload["dataCenterIds"] = [data_center_id]
        try:
            response = self._send(
                lambda: requests.post(
                    f"{self.root}/pods",
                    headers=self._headers(),
                    json=payload,
                    timeout=60,
                )
            )
        except requests.RequestException as exc:
            status = exc.response.status_code if exc.response is not None else None
            detail = f" (HTTP {status})" if status is not None else ""
            request_id = None
            if exc.response is not None:
                request_id = exc.response.headers.get("x-request-id")
            raise RemoteExecutionError(
                "RunPod pod creation failed after retries" + detail,
                provider="runpod",
                attempts=_HTTP_ATTEMPTS,
                request_id=request_id,
            ) from exc
        return dict(response.json())

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        import requests

        response = self._send(
            lambda: requests.get(
                f"{self.root}/pods/{pod_id}", headers=self._headers(), timeout=60
            )
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
        if wheel is None:
            env_wheel = os.environ.get("STATESET_AGENTS_WHEEL", "").strip()
            wheel = Path(env_wheel) if env_wheel else None
        self.wheel = Path(wheel) if wheel else None
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
        self._last_duration_s: float | None = None
        self._last_cost_usd: float | None = None

    # -- crash-recovery leases -------------------------------------------

    def _lease_path(self, pod_id: str) -> Path:
        safe_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in pod_id)
        return self.lease_dir / f"{safe_id}.json"

    def _write_lease(
        self, pod_id: str, job_id: str, spec: RemoteJobSpec, created_at: float
    ) -> None:
        """Record a billing pod so a later process can clean up after a crash."""
        target = self._lease_path(pod_id)
        temporary = target.with_suffix(".tmp")
        payload = {
            "provider": self.name,
            "pod_id": pod_id,
            "job_id": job_id,
            "created_at": created_at,
            "base_model": spec.base_model,
            "gpu": spec.gpu or self.DEFAULT_GPU,
            "gpu_count": spec.gpu_count,
            "output_dir": str(spec.output_dir),
        }
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(target)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise RemoteExecutionError(
                f"could not record cleanup lease for RunPod pod {pod_id}: {exc}",
                provider=self.name,
                pod_id=pod_id,
            ) from exc

    def orphaned_leases(self) -> list[dict[str, Any]]:
        """Return pods whose process-local cleanup did not clear its lease."""
        if not self.lease_dir.exists():
            return []
        leases: list[dict[str, Any]] = []
        for path in sorted(self.lease_dir.glob("*.json")):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(row, dict) and row.get("provider") == self.name:
                leases.append(row)
        return leases

    def cleanup_orphans(self) -> list[str]:
        """Terminate every locally leased pod, clearing only confirmed leases."""
        api = self._require_api()
        terminated: list[str] = []
        for lease in self.orphaned_leases():
            pod_id = str(lease.get("pod_id", "")).strip()
            if not pod_id:
                continue
            api.terminate_pod(pod_id)
            self._lease_path(pod_id).unlink(missing_ok=True)
            terminated.append(pod_id)
        return terminated

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
            harvest_args = " ".join(
                shlex.quote(a)
                for a in spec.harvest_cli_args(adapter_dir=adapter_remote)
            )
            # The spec's local paths are meaningless on the pod: re-point the
            # prompts file and output dir at the uploaded/remote locations.
            harvest_args = harvest_args.replace(
                shlex.quote(str(spec.dataset)), shlex.quote(dataset_remote), 1
            ).replace(shlex.quote(str(spec.output_dir)), _REMOTE_OUTPUT, 1)
            return [
                f"pip install --quiet '{pin}'",
                f"python -m stateset_agents.training.harvest {harvest_args}",
            ]
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
        if spec.resume or force_resume:
            # Only meaningful when checkpoints already exist remotely: a
            # network-volume retry (``force_resume``), or a caller-passed
            # ``--resume``. A fresh pod without a volume has an empty output
            # dir, so the job logs it and trains from scratch — harmless
            # either way; passed through for parity with the local provider.
            args += " --resume"
        if spec.eval_prompts:
            # The command travels through ssh + bash, so the JSON blob must
            # survive a real shell — hence shlex.quote, not manual quoting.
            args += f" --eval-prompts-json {shlex.quote(json.dumps(spec.eval_prompts))}"
            args += f" --eval-max-new-tokens {spec.eval_max_new_tokens}"
        return [
            f"pip install --quiet '{pin}'",
            f"python -m stateset_agents.training.sft {args}",
        ]

    def submit(self, spec: RemoteJobSpec) -> JobHandle:
        self.validate_spec(spec)
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

        for attempt in range(1, self.max_provision_attempts + 1):
            try:
                status = self._run_attempt(
                    api,
                    ssh,
                    public_key,
                    spec,
                    job_id,
                    logs,
                    data_center_id=data_center_id,
                    # A retry only sees prior checkpoints when they landed on
                    # a network volume; without one they died with the pod.
                    force_resume=bool(spec.network_volume_id) and attempt > 1,
                )
            except _PodDiedMidJob as exc:
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
            self._jobs[job_id] = _RunPodJob(
                spec,
                status,
                logs,
                duration_s=self._last_duration_s,
                cost_usd=self._last_cost_usd,
            )
            record_entry(
                CostEntry(
                    provider=self.name,
                    job_id=job_id,
                    base_model=spec.base_model,
                    gpu=spec.gpu or self.DEFAULT_GPU,
                    gpu_count=spec.gpu_count,
                    cost_per_hr=(
                        round(self._last_cost_usd / (self._last_duration_s / 3600), 4)
                        if self._last_cost_usd and self._last_duration_s
                        else None
                    ),
                    duration_s=(
                        round(self._last_duration_s, 1)
                        if self._last_duration_s is not None
                        else None
                    ),
                    cost_usd=self._last_cost_usd,
                    status=status.value,
                ),
                path=self.ledger_path,
            )
            return handle

        raise AssertionError("unreachable")  # pragma: no cover

    def _run_attempt(
        self,
        api: Any,
        ssh: Any,
        public_key: str,
        spec: RemoteJobSpec,
        job_id: str,
        logs: list[str],
        *,
        data_center_id: str | None = None,
        force_resume: bool = False,
    ) -> JobStatus:
        """One full provision → upload → run → download cycle on a new pod.

        Raises :class:`_PodDiedMidJob` when the pod fails *under* the job —
        the caller may then retry on a fresh pod. Every other outcome is
        terminal: a returned :class:`JobStatus`, or a plain
        :class:`RemoteExecutionError`. The pod is terminated on every path.
        """
        pod = api.create_pod(
            name=f"stateset-sft-{job_id}",
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
        pod_id = str(pod["id"])
        # Billing starts at creation, so the clock does too.
        pod_started_at = time.time()
        try:
            self._write_lease(pod_id, job_id, spec, pod_started_at)
        except Exception:
            # A pod we cannot track after caller death is unsafe to keep.
            try:
                api.terminate_pod(pod_id)
            finally:
                raise
        cost_per_hr = pod.get("costPerHr")
        try:
            cost_per_hr = float(cost_per_hr) if cost_per_hr is not None else None
        except (TypeError, ValueError):
            cost_per_hr = None
        logs.append(
            f"created pod {pod_id} "
            f"({spec.gpu_count}x {spec.gpu or self.DEFAULT_GPU}, "
            f"{spec.cloud_type}"
            + (f", ${cost_per_hr}/hr)" if cost_per_hr is not None else ")")
        )

        # A ceiling is checked before a single second of work: refusing costs
        # only the seconds this pod has existed, which we then terminate.
        try:
            check_budget(
                cost_per_hr,
                spec.timeout_s,
                spec.max_cost_usd,
                # RunPod reports the effective price for the entire Pod,
                # not a per-GPU rate. Confirmed against a live 4x H100 Pod
                # on 2026-08-26 ($13.16/hr total).
                gpu_count=1,
            )
        except BudgetExceeded as exc:
            try:
                api.terminate_pod(pod_id)
            except Exception:  # pragma: no cover - defensive
                pass
            else:
                self._lease_path(pod_id).unlink(missing_ok=True)
            raise RemoteExecutionError(
                str(exc), provider=self.name, pod_id=pod_id
            ) from exc

        # Everything from here must terminate the pod, whatever happens.
        try:
            host, port = self._wait_for_ssh_endpoint(api, pod_id)
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
                    logs.append(f"uploaded adapter {adapter_dir} -> {_REMOTE_ADAPTER}")

                if self.wheel:
                    ssh.upload(self.wheel, f"{_REMOTE_WORKDIR}/{self.wheel.name}")
                    logs.append(f"uploaded {self.wheel.name}")

                exit_code = 0
                commands = self._remote_commands(
                    spec, dataset_remote, force_resume=force_resume
                )
                for command in commands:
                    exit_code, output = ssh.run(command)
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
            except _PodDiedMidJob:
                raise
            except Exception as exc:
                raise _PodDiedMidJob(
                    f"pod {pod_id} failed mid-job: {exc}", provider=self.name
                ) from exc

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

            logs.append(f"downloaded {len(downloaded)} file(s) to {spec.output_dir}")
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
                api.terminate_pod(pod_id)
            except Exception as exc:  # never mask the original failure
                logs.append(
                    f"could not confirm termination of pod {pod_id}: {exc}; "
                    "cleanup lease retained"
                )
            else:
                self._lease_path(pod_id).unlink(missing_ok=True)
            # Bookkeeping last: the money was spent whether or not the job
            # worked, so the ledger records failures too.
            self._last_duration_s = time.time() - pod_started_at
            self._last_cost_usd = estimate_cost_usd(cost_per_hr, self._last_duration_s)
            logs.append(
                f"pod {pod_id} ran {self._last_duration_s:.0f}s"
                + (
                    f" (~${self._last_cost_usd:.2f})"
                    if self._last_cost_usd is not None
                    else " (cost unknown)"
                )
            )

    # -- executor interface -----------------------------------------------

    def job_cost(self, handle: JobHandle) -> tuple[float | None, float | None]:
        """Measured pod lifetime and spend for a finished RunPod job."""
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

    def fetch(self, handle: JobHandle, dest: Path | None = None) -> Path:
        job = self._job(handle)
        if not job.status.is_terminal:
            raise RemoteExecutionError(
                f"job {handle.job_id} is not finished; nothing to fetch",
                provider=self.name,
            )
        # A FAILED job may still have salvaged artifacts (the eval gate
        # fails AFTER saving them); point at the output dir either way.
        return dest or job.spec.output_dir

    def cancel(self, handle: JobHandle) -> None:
        job = self._job(handle)
        if not job.status.is_terminal:
            job.status = JobStatus.CANCELLED
