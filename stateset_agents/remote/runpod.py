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
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import weakref
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

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
    estimate_cost_usd,
    record_entry,
)
from stateset_agents.utils.process_probe import probe_process, process_start_token

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

#: Pod-side deadline: the pod deletes itself at the run deadline even if the
#: submitting worker and its supervisor are both gone (SIGKILL, host crash).
#: It reads RunPod's own pod-scoped ``RUNPOD_POD_ID``/``RUNPOD_API_KEY`` from
#: the container environment; no user credential is copied to the pod. It
#: arms only when those identify the pod this executor created.
_POD_WATCHDOG = r"""import os, sys, time, urllib.error, urllib.request
when, root, expected = sys.argv[1], sys.argv[2], sys.argv[3]
# "@<epoch>" is an absolute wall-clock deadline; otherwise a delay in seconds.
until = float(when[1:]) if when.startswith("@") else time.time() + float(when)
env = dict(os.environ)
if not (env.get("RUNPOD_POD_ID") and env.get("RUNPOD_API_KEY")):
    try:
        with open("/proc/1/environ", "rb") as stream:
            for item in stream.read().split(b"\0"):
                name, _, value = item.decode("utf-8", "replace").partition("=")
                env.setdefault(name, value)
    except OSError:
        pass
pod, key = env.get("RUNPOD_POD_ID"), env.get("RUNPOD_API_KEY")
if not pod or not key or expected not in ("*", pod):
    print("pod deadline unavailable", flush=True)
    sys.exit(0)
if os.fork():
    print("pod deadline armed", flush=True)
    sys.exit(0)
os.setsid()
null = os.open(os.devnull, os.O_RDWR)
for descriptor in (0, 1, 2):
    os.dup2(null, descriptor)
while time.time() < until:
    time.sleep(min(60.0, max(0.0, until - time.time())))
request = urllib.request.Request(
    root + "/pods/" + pod, method="DELETE",
    headers={"Authorization": "Bearer " + key},
)
for _ in range(5):
    try:
        urllib.request.urlopen(request, timeout=30).read()
        break
    except urllib.error.HTTPError as error:
        if error.code == 404:
            break
        time.sleep(30)
    except OSError:
        time.sleep(30)
"""

_API_ROOT = "https://rest.runpod.io/v1"
#: RunPod's own PyTorch images start sshd (from PUBLIC_KEY) via /start.sh.
_START_SCRIPT_IMAGES = ("runpod/pytorch:",)


def _deadline_start_command(deadline_epoch: float) -> str:
    """A pod start command that arms the deadline before sshd starts.

    The watchdog runs from the first moment the container starts, so a pod
    whose submitter dies while it is still provisioning (before SSH is ever
    reachable) still deletes itself at the run deadline. It then hands over
    to the image's own ``/start.sh``.
    """
    watchdog = shlex.join(
        ["python3", "-c", _POD_WATCHDOG, f"@{deadline_epoch:.0f}", _API_ROOT, "*"]
    )
    return f"{watchdog} >/tmp/stateset-deadline.log 2>&1 & exec /start.sh"


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


_TIME_MODULE = time
_REAL_MONOTONIC = time.monotonic
#: Wall-clock lead over monotonic time that counts as a suspend, not jitter.
_SUSPEND_NOISE_S = 1.0
_CLOCK_LOCK = threading.Lock()
_clock_anchor: list[Any] = []


def _billing_now() -> float:
    """Seconds on a clock that keeps counting while the host is suspended.

    A pod bills through a laptop sleep, but ``CLOCK_MONOTONIC`` pauses, so a
    run deadline or a spend estimate measured with it would stretch by the
    suspended time. Linux uses ``CLOCK_BOOTTIME``. Elsewhere this is monotonic
    time plus whatever wall-clock time monotonic missed (a suspend), and it
    never decreases, so a wall clock stepped backwards cannot shorten it.
    """
    clock = time
    if clock is _TIME_MODULE and clock.monotonic is not _REAL_MONOTONIC:
        # A substituted monotonic source (a test double) is trusted as is:
        # comparing it with the real wall clock would invent suspends.
        return float(clock.monotonic())
    boottime = getattr(clock, "CLOCK_BOOTTIME", None)
    gettime = getattr(clock, "clock_gettime", None)
    if boottime is not None and callable(gettime):
        return float(gettime(boottime))
    mono, wall = float(clock.monotonic()), float(clock.time())
    source = clock.monotonic
    key = (id(clock), id(getattr(source, "__func__", source)))
    with _CLOCK_LOCK:
        if not _clock_anchor or _clock_anchor[0] != key:
            _clock_anchor[:] = [key, mono, wall, 0.0, mono]
        _, mono0, wall0, offset, last = _clock_anchor
        missed = (wall - wall0) - (mono - mono0)
        if missed > _SUSPEND_NOISE_S:  # ignore scheduling jitter between reads
            offset = max(offset, missed)
        now = max(last, mono + offset)
        _clock_anchor[3], _clock_anchor[4] = offset, now
        return float(now)


def lease_owner_state(lease: dict[str, Any]) -> str:
    """Whether a lease's submitting process is ``alive``, ``gone`` or ``unknown``.

    ``gone`` needs proof: the PID no longer exists, or it now belongs to a
    different process (start-time token mismatch). Anything uncertain is
    ``unknown`` so that automatic cleanup never ends a live job's pod.
    """
    pid, token = lease.get("owner_pid"), lease.get("owner_token")
    if (
        type(pid) is not int
        or pid < 1
        or lease.get("owner_host") != socket.gethostname()
    ):
        return "unknown"
    current = process_start_token(pid)
    if isinstance(token, str) and current is not None:
        return "alive" if current == token else "gone"
    try:
        probe_process(pid)
    except ProcessLookupError:
        return "gone"
    except (OSError, ValueError, OverflowError):
        return "unknown"
    # The PID answers, but without a comparable token its identity is unproven.
    return "alive" if token is None and current is None else "unknown"


def _pod_already_gone(exc: BaseException) -> bool:
    """A termination refused with 404: the pod no longer exists, so nothing bills."""
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 404


@contextmanager
def _signals_end_submission() -> Iterator[None]:
    """Turn SIGTERM/SIGHUP into ``SystemExit`` while a submission holds pods.

    Their default action kills the process outright, skipping the ``finally``
    that terminates the pod and clears its lease, so a stopped worker could
    leave a pod billing past every ceiling. Only the main thread can install
    handlers, and a handler the application already installed is kept. After
    the first signal, repeats are ignored so cleanup can finish.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous: dict[int, Any] = {}

    def stop(signum: int, _frame: Any) -> None:
        for number in previous:
            signal.signal(number, signal.SIG_IGN)
        raise SystemExit(128 + signum)

    for name in ("SIGTERM", "SIGHUP"):
        number = getattr(signal, name, None)
        if number is None:
            continue
        try:
            if signal.getsignal(number) is signal.SIG_DFL:
                previous[number] = signal.signal(number, stop)
        except (OSError, ValueError):
            continue
    try:
        yield
    finally:
        for number, handler in previous.items():
            try:
                signal.signal(number, handler)
            except (OSError, ValueError):
                pass


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
                time.sleep(backoff_s * (2 ** (attempt - 1)))
        assert last is not None  # pragma: no cover - loop always sets it
        raise last

    def account_balance(self) -> float | None:
        """The account's prepaid credit in USD, or None if it cannot be read.

        RunPod refuses to create pods when the balance is exhausted and reports
        that as a generic HTTP 500, so callers check this first to fail with a
        clear message instead.
        """
        import requests

        try:
            response = requests.post(
                "https://api.runpod.io/graphql",
                json={"query": "query { myself { clientBalance } }"},
                headers=self._headers(),
                timeout=30,
            )
            response.raise_for_status()
            value = response.json()["data"]["myself"]["clientBalance"]
            return float(value)
        except Exception:  # noqa: BLE001 - unknown balance never blocks a run
            return None

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

    def create_cpu_pod(
        self,
        *,
        name: str,
        image: str,
        cpu_flavor_id: str,
        vcpu_count: int,
        ports: list[str],
        env: dict[str, str],
        container_disk_gb: int = 20,
        support_public_ip: bool = True,
        data_center_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a CPU-only pod (``computeType=CPU``) on the REST API.

        CPU pods are Secure Cloud only and are sized by flavor (``cpu3c``,
        ``cpu5g``...) and vCPU count; memory follows the flavor's RAM
        multiplier. Creation is never retried after an ambiguous transport
        error, for the same reason as :meth:`create_pod`.
        """
        import requests

        if isinstance(vcpu_count, bool) or not isinstance(vcpu_count, int):
            raise ValueError("vcpu_count must be an integer")
        if vcpu_count < 1:
            raise ValueError("vcpu_count must be positive")
        payload: dict[str, Any] = {
            "name": name,
            "imageName": image,
            "computeType": "CPU",
            "cpuFlavorIds": [cpu_flavor_id],
            "cpuFlavorPriority": "availability",
            "vcpuCount": vcpu_count,
            "cloudType": "SECURE",
            "supportPublicIp": support_public_ip,
            "containerDiskInGb": container_disk_gb,
            "ports": ports,
            "env": env,
        }
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
            raise RemoteExecutionError(
                "RunPod CPU pod creation failed after retries" + detail,
                provider="runpod",
                attempts=_HTTP_ATTEMPTS,
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
        """Delete a pod; DELETE is idempotent, so 429/5xx are retried with backoff."""
        import requests

        self._send(
            lambda: requests.delete(
                f"{self.root}/pods/{pod_id}", headers=self._headers(), timeout=60
            )
        )


class _SshTimeout(RemoteExecutionError):
    """A bounded SSH operation ended without establishing its remote outcome."""


class SshTransport:
    """SSH/SCP transport with finite client waits and optional job deadlines.

    Client timeouts do not prove a remote command stopped or billing ended.
    Built-in executor attempts use independent transports with shared deadlines;
    injected transports remain responsible for their own blocking operations.
    """

    def __init__(
        self,
        user: str = "root",
        key_path: Path | None = None,
        *,
        operation_timeout_s: float = 3600.0,
    ) -> None:
        self.user = user
        self.key_path = key_path
        self.operation_timeout_s = self._positive_timeout(operation_timeout_s)
        self._deadline: float | None = None
        self._host: str | None = None
        self._port: int | None = None
        self._known_hosts: Path | None = None

    def _known_hosts_file(self) -> Path:
        """This transport's private known_hosts file, created on first use.

        Each transport talks to one freshly created pod, so its host key
        cannot be known in advance. ``StrictHostKeyChecking=accept-new``
        records the key on first contact into this per-transport file (trust
        on first use), and every later connection — reconnects included — is
        then pinned to it: a host key that changes mid-session is rejected.
        The file is never shared between transports, so a new pod that
        happens to reuse an ip:port starts with an empty pin set.
        """
        if self._known_hosts is None:
            directory = Path(tempfile.mkdtemp(prefix="stateset-ssh-"))
            path = directory / "known_hosts"
            path.touch(mode=0o600)
            self._known_hosts = path
            weakref.finalize(self, shutil.rmtree, str(directory), True)
        return self._known_hosts

    @staticmethod
    def _positive_timeout(value: float) -> float:
        if (
            type(value) not in (int, float)
            or value <= 0
            or value > 7 * 24 * 3600
            or not math.isfinite(value)
        ):
            raise ValueError("SSH timeout must be finite and in (0, 604800] seconds")
        return float(value)

    def _for_deadline(self, deadline: float) -> SshTransport:
        """Give an attempt its own endpoint and deadline without mutating a peer."""
        if type(deadline) not in (int, float) or not math.isfinite(deadline):
            raise ValueError("SSH deadline must be finite")
        transport = SshTransport(
            self.user, self.key_path, operation_timeout_s=self.operation_timeout_s
        )
        transport._deadline = deadline
        return transport

    def _timeout(
        self, requested: float | None = None, *, command: bool = False
    ) -> float:
        timeout = (
            self.operation_timeout_s
            if requested is None
            else self._positive_timeout(requested)
        )
        if self._deadline is not None:
            remaining = self._deadline - _billing_now()
            if remaining <= 0:
                raise _SshTimeout(
                    "RunPod transport deadline expired before starting an operation",
                    provider="runpod",
                    stage="ssh_timeout",
                    outcome_unknown=False,
                )
            # The foreground training command may use the remaining job budget;
            # ordinary transfers retain their per-operation limit as well.
            timeout = (
                min(remaining, 7 * 24 * 3600)
                if command and requested is None
                else min(timeout, remaining)
            )
        return timeout

    def _invoke(
        self,
        argv: list[str],
        *,
        operation: str,
        timeout_s: float | None = None,
        command: bool = False,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        timeout = self._timeout(timeout_s, command=command)
        kwargs: dict[str, Any] = {
            "capture_output": True,
            "text": True,
            "check": False,
            "timeout": timeout,
        }
        if input_text is None:
            kwargs["stdin"] = subprocess.DEVNULL
        else:
            kwargs["input"] = input_text
        try:
            return subprocess.run(argv, **kwargs)
        except subprocess.TimeoutExpired:
            # TimeoutExpired can retain command arguments and captured output.
            # Do not expose those (or streamed credentials) in the error chain.
            raise _SshTimeout(
                f"RunPod {operation} exceeded {timeout:g}s; remote outcome is unknown",
                provider="runpod",
                stage="ssh_timeout",
                outcome_unknown=True,
            ) from None

    def _base_opts(self) -> list[str]:
        known_hosts = str(self._known_hosts_file())
        if any(ch.isspace() for ch in known_hosts):
            # ssh splits this option's value on whitespace (it accepts a list).
            known_hosts = f'"{known_hosts}"'
        opts = [
            # The pod is created fresh for this job, so its host key cannot
            # be known in advance: trust it on first use and pin it in a
            # per-transport file for the rest of the session (see
            # ``_known_hosts_file``). System/user known_hosts are ignored so a
            # stale entry for a recycled ip:port cannot block a fresh pod.
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={known_hosts}",
            "-o",
            "GlobalKnownHostsFile=/dev/null",
            "-o",
            "LogLevel=ERROR",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
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

    def wait_until_reachable(self, host: str, port: int, timeout_s: float) -> None:
        """Block until sshd answers, or raise once ``timeout_s`` elapses."""
        self._host, self._port = host, port
        timeout_s = self._positive_timeout(timeout_s)
        deadline = _billing_now() + timeout_s
        if self._deadline is not None:
            deadline = min(deadline, self._deadline)
        last_error = ""
        while _billing_now() < deadline:
            remaining = deadline - _billing_now()
            if remaining <= 0:
                break
            try:
                probe = self._invoke(
                    [
                        "ssh",
                        *self._base_opts(),
                        "-p",
                        str(port),
                        f"{self.user}@{host}",
                        "true",
                    ],
                    operation="SSH readiness probe",
                    timeout_s=min(10.0, remaining),
                )
            except _SshTimeout:
                last_error = "SSH readiness probe timed out"
            else:
                if probe.returncode == 0 and _billing_now() <= deadline:
                    return
                last_error = (probe.stderr or "").strip()
            remaining = deadline - _billing_now()
            if remaining > 0:
                time.sleep(min(5.0, remaining))
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
        result = self._invoke(
            [
                "ssh",
                *self._base_opts(),
                "-p",
                str(self._port),
                f"{self.user}@{self._host}",
                command,
            ],
            operation="secret upload",
            input_text=secret,
        )
        if result.returncode != 0:
            raise RemoteExecutionError(
                f"secret upload failed (ssh exit {result.returncode})",
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
        result = self._invoke(cmd, operation="SCP transfer")
        if result.returncode != 0:
            raise RemoteExecutionError(
                f"scp failed: {(result.stderr or '').strip()}", provider="runpod"
            )

    def run(self, command: str, *, timeout_s: float | None = None) -> tuple[int, str]:
        """Run one remote command, bounding the client wait without retrying it."""
        result = self._invoke(
            [
                "ssh",
                *self._base_opts(),
                "-p",
                str(self._port),
                f"{self.user}@{self._host}",
                command,
            ],
            operation="SSH command",
            timeout_s=timeout_s,
            command=True,
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
        self._last_duration_s: float | None = None
        self._last_cost_usd: float | None = None
        #: Run-level deadline (monotonic), fixed when a submission's first pod
        #: is allocated and shared by every replacement pod.
        self._run_deadline: float | None = None
        #: The most recently allocated pod's quote, for replacement budgeting.
        self._last_cost_per_hr: float | None = None
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
                "resource_name": f"stateset-sft-{job_id}",
                "attempt": attempt,
            }
        )

    def _cleanup_submission_pod(
        self, api: Any, pod_id: str, job_id: str, logs: list[str], *, attempt: int
    ) -> None:
        """Retain cleanup evidence without masking an existing primary failure."""
        primary_error = sys.exc_info()[1]
        try:
            try:
                api.terminate_pod(pod_id)
            except Exception as exc:
                if not _pod_already_gone(exc):
                    raise
                logs.append(f"pod {pod_id} was already gone when cleanup ran")
        except Exception as exc:
            logs.append(
                f"could not confirm termination of pod {pod_id}: {exc}; "
                "cleanup lease retained; billing may still be active"
            )
            event = "cleanup_unconfirmed"
            self._submission_cleanup_unconfirmed = True
        else:
            event = "cleanup_acknowledged"
            try:
                self._lease_path(pod_id).unlink(missing_ok=True)
            except OSError:
                logs.append("Could not remove the acknowledged pod cleanup lease")
        try:
            self._observe_pod(job_id, event, pod_id, attempt=attempt)
        except SubmissionReceiptError:
            if primary_error is None:
                raise
            logs.append(
                "Could not retain pod cleanup evidence; inspect retained resources"
            )

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
        owner = os.getpid()
        payload = {
            "provider": self.name,
            "pod_id": pod_id,
            "job_id": job_id,
            "created_at": created_at,
            "base_model": spec.base_model,
            "gpu": spec.gpu or self.DEFAULT_GPU,
            "gpu_count": spec.gpu_count,
            "output_dir": str(spec.output_dir),
            # Who holds this pod: cleanup tools reclaim only pods whose
            # submitting process is provably gone, never a live job's pod.
            "owner_pid": owner,
            "owner_token": process_start_token(owner),
            "owner_host": socket.gethostname(),
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

    def leases(self) -> list[dict[str, Any]]:
        """Every local cleanup lease, each with its submitting process's state.

        ``owner_state`` is ``"alive"`` while the submitting process runs,
        ``"gone"`` once it has provably exited (or its PID was reused), and
        ``"unknown"`` for leases without an owner record, from another host,
        or whose owner cannot be inspected.
        """
        if not self.lease_dir.exists():
            return []
        leases: list[dict[str, Any]] = []
        for path in sorted(self.lease_dir.glob("*.json")):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(row, dict) and row.get("provider") == self.name:
                leases.append({**row, "owner_state": lease_owner_state(row)})
        return leases

    def orphaned_leases(self) -> list[dict[str, Any]]:
        """Leases whose submitting process is not known to be running.

        A live job's lease is excluded: its process still owns the cleanup.
        """
        return [lease for lease in self.leases() if lease["owner_state"] != "alive"]

    def reclaim_pods(self, pod_ids: Iterable[str]) -> tuple[list[str], dict[str, str]]:
        """Terminate each pod independently, clearing a lease only once it is gone.

        A 404 means the pod no longer exists and counts as reclaimed; the API
        call itself retries 429/5xx. One failure never stops the others.
        Returns the reclaimed pod ids and a credential-free reason per failure.
        """
        api = self._require_api()
        reclaimed: list[str] = []
        failed: dict[str, str] = {}
        for pod_id in dict.fromkeys(str(value).strip() for value in pod_ids):
            if not pod_id:
                continue
            try:
                api.terminate_pod(pod_id)
            except Exception as exc:
                if not _pod_already_gone(exc):
                    status = getattr(
                        getattr(exc, "response", None), "status_code", None
                    )
                    failed[pod_id] = type(exc).__name__ + (
                        f" (HTTP {status})" if status is not None else ""
                    )
                    continue
            reclaimed.append(pod_id)
            try:
                self._lease_path(pod_id).unlink(missing_ok=True)
            except OSError:
                failed[pod_id] = (
                    "terminated, but its cleanup lease could not be removed"
                )
        return reclaimed, failed

    def sweep_abandoned_pods(self) -> tuple[list[str], dict[str, str]]:
        """Reclaim pods whose submitting process on this host has provably exited.

        Runs before new work starts, so a pod left behind by a crashed
        submitter (even mid-provisioning) is ended by the next session instead
        of billing until someone notices. Live and unverifiable owners are
        never touched.
        """
        abandoned = [
            str(lease.get("pod_id", ""))
            for lease in self.leases()
            if lease["owner_state"] == "gone"
        ]
        if not abandoned:
            return [], {}
        return self.reclaim_pods(abandoned)

    def pods_named(self, names: Iterable[str]) -> list[str]:
        """Ids of account pods whose name is exactly one of ``names``."""
        wanted = set(names)
        if not wanted:
            return []
        return [
            str(pod["id"])
            for pod in self._require_api().list_pods()
            if pod.get("name") in wanted and isinstance(pod.get("id"), str)
        ]

    def cleanup_orphans(self, *, force: bool = False) -> list[str]:
        """Terminate leased pods whose submitting process is provably gone.

        A live job's pod, or one whose owner cannot be established, is kept
        unless ``force`` (terminating a live job's pod makes it rent another).
        Every selected lease is attempted; failures are reported together
        afterwards and their leases kept for a later retry.
        """
        reclaimed, failed = self.reclaim_pods(
            str(lease.get("pod_id", ""))
            for lease in self.leases()
            if force or lease["owner_state"] == "gone"
        )
        if failed:
            raise RemoteExecutionError(
                f"could not confirm termination of {len(failed)} leased pod(s): "
                + "; ".join(f"{pod}: {reason}" for pod, reason in failed.items())
                + (f" (terminated {', '.join(reclaimed)})" if reclaimed else "")
                + "; their leases were kept, so retry or inspect the provider console",
                provider=self.name,
                reclaimed=reclaimed,
                failed=sorted(failed),
            )
        return reclaimed

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
        self, api: Any, pod_id: str, run_deadline: float | None = None
    ) -> tuple[str, int]:
        """Poll until the pod is RUNNING and has published its SSH port.

        The wait never outlives ``run_deadline``: readiness time is billed
        against the same run-level ``timeout_s`` as the job itself.
        """
        deadline = _billing_now() + self.ready_timeout_s
        if run_deadline is not None:
            deadline = min(deadline, run_deadline)
        while True:
            pod = api.get_pod(pod_id)
            ip = pod.get("publicIp")
            port = (pod.get("portMappings") or {}).get("22")
            if pod.get("desiredStatus") == "RUNNING" and ip and port:
                return str(ip), int(port)
            now = _billing_now()
            if now >= deadline:
                bound = (
                    "the run's remaining time limit"
                    if run_deadline is not None and deadline >= run_deadline
                    else f"{self.ready_timeout_s}s"
                )
                raise RemoteExecutionError(
                    f"pod {pod_id} never became reachable within "
                    f"{bound} (last status "
                    f"{pod.get('desiredStatus')!r})",
                    provider=self.name,
                )
            time.sleep(max(0.0, min(self.poll_interval_s, deadline - now)))

    def _arm_pod_deadline(
        self, ssh: Any, pod_id: str, run_deadline: float, logs: list[str]
    ) -> None:
        """Best effort: have the pod delete itself at the run deadline.

        Covers the case no local process survives to clean up. The job never
        depends on it; whether it armed is logged.
        """
        delay = max(0, math.ceil(run_deadline - _billing_now()))
        try:
            _, output = ssh.run(
                shlex.join(
                    ["python3", "-c", _POD_WATCHDOG, str(delay), _API_ROOT, pod_id]
                ),
                timeout_s=30,
            )
        except Exception as exc:  # noqa: BLE001 - the deadline is a backstop only
            logs.append(f"could not arm the pod-side deadline: {type(exc).__name__}")
            return
        armed = "pod deadline armed" in output
        logs.append(
            f"pod-side deadline armed: pod {pod_id} deletes itself in {delay}s"
            if armed
            else "pod-side deadline unavailable (no pod-scoped RunPod credentials); "
            "local cleanup and leases still apply"
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
        # A stopped submitter still terminates the pods it holds (see
        # ``_signals_end_submission``).
        with _signals_end_submission():
            return self._submit(spec)

    def _submit(self, spec: RemoteJobSpec) -> JobHandle:
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
        try:
            swept, unswept = self.sweep_abandoned_pods()
        except Exception as exc:  # noqa: BLE001 - never blocks new work
            logs.append(f"could not sweep abandoned pods: {type(exc).__name__}")
        else:
            logs.extend(
                f"terminated pod {pod} left by an exited submitter" for pod in swept
            )
            logs.extend(
                f"could not terminate abandoned pod {pod} ({reason}); lease kept"
                for pod, reason in unswept.items()
            )

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
        total_cost_usd: float | None = 0.0
        # ``timeout_s`` and ``max_cost_usd`` bound the whole run, not each pod:
        # the deadline is fixed when the first pod is allocated and every
        # replacement inherits what is left of it.
        self._run_deadline = None
        self._last_cost_per_hr = None
        pod_death: _PodDiedMidJob | None = None
        for attempt in range(1, self.max_provision_attempts + 1):
            if pod_death is not None:
                refusal = self._replacement_refusal(
                    spec,
                    spent_usd=total_cost_usd,
                    cost_per_hr=self._last_cost_per_hr,
                    start=_billing_now(),
                )
                if refusal is not None:
                    stage, reason = refusal
                    logs.append(f"not provisioning a replacement pod: {reason}")
                    raise RemoteExecutionError(
                        f"pod died mid-job and no replacement pod was "
                        f"provisioned: {reason} ({pod_death})",
                        provider=self.name,
                        stage=stage,
                    ) from pod_death
            self._last_duration_s = None
            self._last_cost_usd = None
            self._submission_cleanup_unconfirmed = False
            status = JobStatus.FAILED
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
                    attempt=attempt,
                    spent_usd=total_cost_usd,
                )
            except _PodDiedMidJob as exc:
                pod_death = exc
                if (
                    getattr(self, "_submission_observer", None) is not None
                    and self._submission_cleanup_unconfirmed
                ):
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
                # The callee fills these nullable measurements. Snapshot the
                # declared types rather than retaining pre-call None narrowing.
                attempt_duration_s = cast(float | None, self._last_duration_s)
                attempt_cost_usd = cast(float | None, self._last_cost_usd)
                if attempt_duration_s is not None:
                    total_duration_s += attempt_duration_s
                    if total_cost_usd is None or attempt_cost_usd is None:
                        total_cost_usd = None
                    else:
                        total_cost_usd += attempt_cost_usd
                    try:
                        self._record_cost(spec, job_id, status)
                    except Exception:
                        # Accounting must not replace the original receipt,
                        # transport, budget or lease diagnosis.
                        logs.append(
                            "Could not retain RunPod attempt cost; inspect provider records"
                        )
            self._jobs[job_id] = _RunPodJob(
                spec,
                status,
                logs,
                duration_s=total_duration_s,
                cost_usd=(
                    round(total_cost_usd, 4) if total_cost_usd is not None else None
                ),
            )
            return handle

        raise AssertionError("unreachable")  # pragma: no cover

    def _replacement_refusal(
        self,
        spec: RemoteJobSpec,
        *,
        spent_usd: float | None,
        cost_per_hr: float | None,
        start: float,
    ) -> tuple[str, str] | None:
        """Why a replacement pod starting at ``start`` must not run, if it must not.

        A replacement only gets what is left of the run deadline, so its worst
        case is its hourly price for that remainder. Together with what the
        earlier pods already cost, that must stay within ``max_cost_usd``.
        Unknown spend or price cannot be checked against a ceiling, so it is a
        refusal, mirroring :func:`check_budget`.
        """
        deadline = self._run_deadline
        remaining_s = (deadline - start) if deadline is not None else spec.timeout_s
        if remaining_s <= 0:
            return (
                "deadline",
                f"the run's {spec.timeout_s}s time limit has already elapsed",
            )
        if spec.max_cost_usd is None:
            return None
        if spent_usd is None or cost_per_hr is None:
            return (
                "budget",
                "the spend so far or the pod price is unknown, so a replacement "
                f"cannot be checked against the --max-cost ceiling of "
                f"${spec.max_cost_usd:.2f}",
            )
        worst_case = spent_usd + cost_per_hr * remaining_s / 3600.0
        if worst_case > spec.max_cost_usd:
            return (
                "budget",
                f"this run could then cost ${worst_case:.2f} "
                f"(${spent_usd:.2f} already spent + ${cost_per_hr}/hr for the "
                f"remaining {remaining_s:.0f}s), above the --max-cost ceiling "
                f"of ${spec.max_cost_usd:.2f}",
            )
        return None

    def _record_cost(self, spec: RemoteJobSpec, job_id: str, status: JobStatus) -> None:
        """Retain one allocation's quote-derived observed cost, even on failure."""
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
        attempt: int = 1,
        spent_usd: float | None = 0.0,
    ) -> JobStatus:
        """One full provision → upload → run → download cycle on a new pod.

        Raises :class:`_PodDiedMidJob` when the pod fails *under* the job —
        the caller may then retry on a fresh pod. Every other outcome is
        terminal: a returned :class:`JobStatus`, or a plain
        :class:`RemoteExecutionError`. Cleanup is requested for every known
        allocation, without claiming that an acknowledgement ends billing.
        """
        self._observe_pod(job_id, "allocation_intent", attempt=attempt)
        start: dict[str, Any] = {}
        if self.image.startswith(_START_SCRIPT_IMAGES):
            # The pod-side backstop is armed at creation: no local process has
            # to survive provisioning for the pod to stop at the run deadline.
            remaining = spec.timeout_s
            if self._run_deadline is not None:
                remaining = max(0.0, self._run_deadline - _billing_now())
            start = {
                "docker_entrypoint": ["/bin/bash", "-c"],
                "docker_start_cmd": [_deadline_start_command(time.time() + remaining)],
            }
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
            **start,
        )
        pod_id = pod.get("id") if isinstance(pod, dict) else None
        if (
            not isinstance(pod_id, str)
            or not pod_id
            or pod_id != pod_id.strip()
            or len(pod_id) > 256
            or any(not character.isprintable() for character in pod_id)
        ):
            raise RemoteExecutionError(
                "RunPod returned no usable pod identity; allocation may exist, so inspect provider records before retrying",
                provider=self.name,
            )
        # Lease timestamps use wall time; elapsed accounting uses monotonic
        # time from the returned identity through the cleanup request. Neither
        # boundary independently verifies the provider's actual billing window.
        pod_started_at = time.time()
        pod_started_clock = _billing_now()
        cost_per_hr = pod.get("costPerHr")
        try:
            cost_per_hr = float(cost_per_hr) if cost_per_hr is not None else None
        except (TypeError, ValueError, OverflowError):
            cost_per_hr = None
        if cost_per_hr is not None and (
            not math.isfinite(cost_per_hr) or cost_per_hr < 0
        ):
            cost_per_hr = None
        self._last_cost_per_hr = cost_per_hr
        if self._run_deadline is None:
            self._run_deadline = pod_started_clock + spec.timeout_s
        run_deadline = self._run_deadline
        try:
            self._write_lease(pod_id, job_id, spec, pod_started_at)
            self._observe_pod(job_id, "allocated", pod_id, attempt=attempt)
            logs.append(
                f"created pod {pod_id} "
                f"({spec.gpu_count}x {spec.gpu or self.DEFAULT_GPU}, "
                f"{spec.cloud_type}"
                + (f", ${cost_per_hr}/hr)" if cost_per_hr is not None else ")")
            )

            if type(ssh) is SshTransport:
                # Keep transport construction inside the allocation's cleanup
                # boundary, after retaining its identity and lease. Every
                # attempt shares the run deadline; a replacement pod gets only
                # what is left of ``timeout_s``, never a fresh window.
                ssh = ssh._for_deadline(run_deadline)

            # Refuse before work, retaining the allocated pod's observed cost.
            # Failed or ambiguous termination can still leave billing active.
            if attempt == 1:
                try:
                    check_budget(
                        cost_per_hr,
                        spec.timeout_s,
                        spec.max_cost_usd,
                        # RunPod quotes the whole Pod, not each GPU separately.
                        gpu_count=1,
                    )
                except BudgetExceeded as exc:
                    raise RemoteExecutionError(
                        str(exc), provider=self.name, pod_id=pod_id
                    ) from exc
            else:
                # The replacement's own quote may differ from the dead pod's.
                refusal = self._replacement_refusal(
                    spec,
                    spent_usd=spent_usd,
                    cost_per_hr=cost_per_hr,
                    start=pod_started_clock,
                )
                if refusal is not None:
                    raise RemoteExecutionError(
                        f"replacement pod {pod_id} refused before work: "
                        f"{refusal[1]}",
                        provider=self.name,
                        pod_id=pod_id,
                        stage=refusal[0],
                    )

            # Everything from here must terminate the pod, whatever happens.
            try:
                host, port = self._wait_for_ssh_endpoint(api, pod_id, run_deadline)
                logs.append(f"pod reachable at {host}:{port}")
                remaining_s = run_deadline - _billing_now()
                if remaining_s <= 0:
                    raise RemoteExecutionError(
                        f"pod {pod_id} became reachable after the run's "
                        f"{spec.timeout_s}s time limit",
                        provider=self.name,
                        pod_id=pod_id,
                        stage="deadline",
                    )
                ssh.wait_until_reachable(
                    host, port, min(self.ready_timeout_s, remaining_s)
                )
                if type(ssh) is SshTransport:
                    self._arm_pod_deadline(ssh, pod_id, run_deadline, logs)

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
                except (SubmissionReceiptError, _PodDiedMidJob, _SshTimeout):
                    raise
                except Exception as exc:
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
                self._cleanup_submission_pod(api, pod_id, job_id, logs, attempt=attempt)
            finally:
                # This is the observed interval, not proof that billing ended.
                # Retain it even when cleanup evidence cannot be persisted.
                self._last_duration_s = _billing_now() - pod_started_clock
                self._last_cost_usd = estimate_cost_usd(
                    cost_per_hr, self._last_duration_s
                )
                logs.append(
                    f"pod {pod_id} observed for {self._last_duration_s:.0f}s through cleanup request"
                    + (
                        f" (~${self._last_cost_usd:.2f})"
                        if self._last_cost_usd is not None
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
