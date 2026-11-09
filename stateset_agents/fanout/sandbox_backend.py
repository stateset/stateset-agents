"""Run each fan-out attempt in its own StateSet Sandbox.

A thin adapter over the ``stateset_sandbox`` SDK, loaded only when a sandbox run
actually happens (``pip install 'stateset-agents[sandbox]'``). The mapping from
platform concepts is direct:

* one attempt is one sandbox, with its own server-enforced ``budget``
* a run is a supervised process, so it outlives any request and any driver, and
  its logs can be resumed by byte offset after a crash
* Spot (``resource_class: shared``) preemption surfaces as a lost sandbox, which
  the orchestrator retries on a fresh one

Snapshot forks are deliberately not used. ``POST /sandbox/:id/fork`` accepts no
``budget`` and only the dedicated class, so a forked fan-out could neither enforce
per-run spend nor use Spot pricing. For fast starts, bake a custom image and use
``package.source: preinstalled``.
"""

from __future__ import annotations

import os
import shlex
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar, cast

from stateset_agents.fanout.backend import (
    Allocation,
    AllocationLostError,
    BackendError,
    CapacityError,
    FatalBackendError,
    FileEntry,
    LogChunk,
    ProcessState,
    RunBackend,
    TransientBackendError,
)
from stateset_agents.fanout.manifest import FanoutManifest, ManifestError, Resources

__all__ = [
    "BLOCKED_ENV_NAMES",
    "BLOCKED_ENV_PREFIXES",
    "DEFAULT_REQUESTS_PER_MINUTE",
    "RequestThrottle",
    "SandboxBackend",
    "boot_command",
    "translate_error",
]

T = TypeVar("T")

WORKER_MODULE = "stateset_agents.fanout.worker"
WORKSPACE_ROOT = "/workspace/fanout"
DEFAULT_BASE_URL = "https://api.sandbox.stateset.app"
#: Without this, ``pip install torch`` pulls gigabytes of CUDA libraries a CPU pod
#: never uses and whose download overflows the default ephemeral disk.
CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"

#: Below the smallest plan's 60 requests a minute and the 100 per-IP default. Raise
#: it on larger plans: Pro allows 600, Team 6000.
DEFAULT_REQUESTS_PER_MINUTE = 50
#: The controller refuses to set these in a sandbox (``sandbox-env.service.ts``).
BLOCKED_ENV_PREFIXES = (
    "KUBERNETES_",
    "AWS_",
    "GOOGLE_",
    "AZURE_",
    "GCP_",
    "EXEC_AGENT_",
)
BLOCKED_ENV_NAMES = (
    "PATH",
    "HOME",
    "USER",
    "SHELL",
    "NODE_OPTIONS",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "USE_GO_EXECD",
)
#: File reads are capped at 50 MB of base64. Artifact uploads go through the same
#: read, so nothing larger can leave a sandbox through the platform's API.
MAX_READ_BYTES = 37 * 1024 * 1024

_DEFAULT_REQUEST_TIMEOUT_MS = 30_000
#: A create returns only once the pod is ready, and a GPU pod always cold-starts.
_DEFAULT_CREATE_TIMEOUT_S = 900
_LOG_READ_BYTES = 1024 * 1024

_CAPACITY_CODES = frozenset(
    {"CONCURRENT_LIMIT_EXCEEDED", "RATE_LIMIT_EXCEEDED", "VCPU_RATE_LIMIT_EXCEEDED"}
)
_LOST_CODES = frozenset(
    {"SANDBOX_NOT_FOUND", "SANDBOX_NOT_RUNNING", "SANDBOX_INVALID_STATE"}
)
_FATAL_CODES = frozenset({"PLAN_LIMIT_EXCEEDED"})
_FATAL_STATUSES = frozenset({400, 401, 402, 403})


class RequestThrottle:
    """Space requests evenly across every thread, to stay under the org's limit."""

    def __init__(
        self,
        per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if type(per_minute) is not int or per_minute < 1:
            raise ValueError("requests per minute must be a positive integer")
        self._interval = 60.0 / per_minute
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = self._clock()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self._interval
        if slot > now:
            self._sleep(slot - now)


def translate_error(error: Exception) -> BackendError:
    """Map an SDK error onto the response it calls for. See ``backend``."""
    from stateset_sandbox.errors import (
        SandboxApiError,
        SandboxNetworkError,
        SandboxTimeoutError,
        SandboxValidationError,
    )

    message = str(error)
    if isinstance(error, SandboxApiError):
        if error.code in _CAPACITY_CODES:
            return CapacityError(message, retry_after=error.retry_after)
        if error.code in _LOST_CODES:
            return AllocationLostError(message)
        if error.code in _FATAL_CODES or error.status_code in _FATAL_STATUSES:
            return FatalBackendError(message)
        if error.status_code == 429:
            # The per-IP limiter's plain body reaches here as UNKNOWN_ERROR.
            return CapacityError(message, retry_after=error.retry_after)
        if error.status_code >= 500:
            return TransientBackendError(message)
        return BackendError(message)
    if isinstance(error, (SandboxNetworkError, SandboxTimeoutError)):
        return TransientBackendError(message)
    if isinstance(error, SandboxValidationError):
        return FatalBackendError(message)
    return BackendError(message)


def boot_command(
    manifest: FanoutManifest, *, root: str, worker_args: list[str]
) -> list[str]:
    """The command that gets the package installed and hands over to the worker.

    ``set -e`` matters: if the install fails the worker never runs and leaves no
    result, which correctly reads as infrastructure rather than a workload failure.
    One line, because the controller refuses control characters in command arguments.
    """
    package = manifest.package
    worker = ["-m", WORKER_MODULE, *worker_args]
    if package.source == "preinstalled":
        return ["python3", *worker]

    venv = f"{root}/venv"
    extras = f"[{','.join(package.extras)}]" if package.extras else ""
    if package.source == "wheel":
        target = f"{root}/package/{Path(package.wheel or '').name}{extras}"
    else:
        target = f"stateset-agents{extras}=={package.version}"
    install = [
        f"{venv}/bin/python",
        "-m",
        "pip",
        "install",
        "--quiet",
        "--disable-pip-version-check",
    ]
    if not package.install_deps:
        install.append("--no-deps")
    if "training" in package.extras and manifest.resources.gpu is None:
        install += ["--extra-index-url", CPU_TORCH_INDEX]
    install.append(target)
    script = "; ".join(
        [
            "set -eu",
            # System packages stay visible, so a GPU image's torch is reused.
            shlex.join(["python3", "-m", "venv", "--system-site-packages", venv]),
            shlex.join(install),
            "exec " + shlex.join([f"{venv}/bin/python", *worker]),
        ]
    )
    return ["sh", "-c", script]


def _granted_lifetime(response: dict[str, Any]) -> float | None:
    """How long the platform agreed to keep a sandbox. Both times come from the
    server, so this is immune to any skew against the local clock."""
    try:
        created = datetime.fromisoformat(response["created_at"].replace("Z", "+00:00"))
        expires = datetime.fromisoformat(response["expires_at"].replace("Z", "+00:00"))
    except (KeyError, AttributeError, TypeError, ValueError):
        return None
    return (expires - created).total_seconds()


class SandboxBackend(RunBackend):
    """Reads ``STATESET_API_KEY`` and ``STATESET_BASE_URL`` unless given them."""

    name = "sandbox"
    metered = True
    max_read_bytes = MAX_READ_BYTES

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        create_timeout_s: int = _DEFAULT_CREATE_TIMEOUT_S,
        requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
    ) -> None:
        self._throttle = RequestThrottle(requests_per_minute)
        if client is not None:
            self._client = client
            self._create_client = client
            return
        try:
            from stateset_sandbox import StateSetSandbox
        except ImportError as exc:
            raise FatalBackendError(
                "The sandbox backend needs the sandbox SDK: "
                "pip install 'stateset-agents[sandbox]'"
            ) from exc
        key = api_key or os.environ.get("STATESET_API_KEY", "")
        if not key.strip():
            raise FatalBackendError(
                "Set STATESET_API_KEY to an API key for the sandbox platform"
            )
        url = base_url or os.environ.get("STATESET_BASE_URL", DEFAULT_BASE_URL)
        self._client = StateSetSandbox(
            base_url=url, auth_token=key, timeout=_DEFAULT_REQUEST_TIMEOUT_MS
        )
        self._create_client = StateSetSandbox(
            base_url=url, auth_token=key, timeout=create_timeout_s * 1000
        )

    def _call(self, function: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        from stateset_sandbox.errors import SandboxError

        self._throttle.acquire()
        try:
            return function(*args, **kwargs)
        except (SandboxError, ValueError) as exc:
            raise translate_error(exc) from exc

    def validate(self, manifest: FanoutManifest) -> None:
        for name in (*manifest.env, *manifest.forward_env):
            upper = name.upper()
            if upper in BLOCKED_ENV_NAMES or upper.startswith(BLOCKED_ENV_PREFIXES):
                raise ManifestError(
                    f"the sandbox platform refuses to set {name} in a sandbox; "
                    "pass it under a different name"
                )
        package = manifest.package
        custom_image = manifest.resources.image is not None
        if package.source == "preinstalled" and not custom_image:
            raise ManifestError(
                "the stock sandbox image has no stateset_agents: set package.source "
                "to wheel or pypi, or resources.image to an image that includes it"
            )
        if manifest.workload["kind"] == "command" or package.source == "preinstalled":
            return
        if not package.install_deps:
            raise ManifestError(
                "package.install_deps false cannot run a learning workload: "
                "it needs torch and transformers"
            )
        if "training" not in package.extras and not custom_image:
            raise ManifestError(
                "this workload needs torch and transformers: add training to "
                "package.extras, or use a resources.image that includes them"
            )

    def allocate(
        self, *, run_id: str, resources: Resources, idempotency_key: str
    ) -> Allocation:
        from stateset_sandbox.types import CreateSandboxOptions, RunBudget

        body = CreateSandboxOptions(
            cpus=resources.cpus,
            memory=resources.memory,
            timeout_seconds=resources.sandbox_lifetime_seconds,
            resource_class=cast(Any, resources.resource_class),
            region=resources.region,
            image=resources.image,
            ephemeral_storage=resources.ephemeral_storage,
            budget=RunBudget(
                max_cost_cents=resources.max_cost_cents,
                max_runtime_seconds=resources.sandbox_lifetime_seconds,
                on_exhausted="stop",
            ),
        ).model_dump(exclude_none=True)
        # The controller accepts this; the SDK's typed options lack the field.
        if resources.gpu is not None:
            body["gpu"] = resources.gpu.to_dict()
        # _request because create() generates its own key on each call, and only a
        # key that outlives the call lets a lost response find its sandbox again.
        response = self._call(
            self._create_client._request,
            "POST",
            "/sandbox/create",
            body,
            idempotency_key=idempotency_key,
        )
        # Only the id, and read directly: once a sandbox exists, a stricter parse
        # failing would orphan it. Without one, asking again with this key is how
        # to learn it.
        sandbox_id = response.get("sandbox_id") if isinstance(response, dict) else None
        if not isinstance(sandbox_id, str) or not sandbox_id:
            raise TransientBackendError("create succeeded without naming a sandbox")
        return Allocation(
            id=sandbox_id,
            root=f"{WORKSPACE_ROOT}/{run_id}",
            lifetime_s=_granted_lifetime(response),
        )

    def write_file(self, allocation_id: str, path: str, content: bytes) -> None:
        self._call(self._client.write_file_bytes, allocation_id, path, content)

    def worker_argv(
        self, manifest: FanoutManifest, *, root: str, worker_args: list[str]
    ) -> list[str]:
        return boot_command(manifest, root=root, worker_args=worker_args)

    def start(
        self,
        allocation_id: str,
        argv: list[str],
        *,
        env: dict[str, str],
        timeout_seconds: int,
        name: str,
    ) -> str:
        info = self._call(
            self._client.start_process,
            allocation_id,
            argv,
            env=env,
            name=name,
            # Retrying is the orchestrator's decision, on a fresh sandbox.
            restart="never",
            timeout_seconds=timeout_seconds,
        )
        return str(info.process_id)

    def process(self, allocation_id: str, process_id: str) -> ProcessState:
        info = self._call(self._client.get_process, allocation_id, process_id)
        if info.status == "lost":
            raise AllocationLostError("the pod restarted and the process was lost")
        return ProcessState(
            status=info.status,
            running=info.running,
            exit_code=info.exit_code,
            detail=info.last_error,
            stdout_bytes=info.log.get("stdout_bytes"),
            stderr_bytes=info.log.get("stderr_bytes"),
        )

    def logs(
        self, allocation_id: str, process_id: str, stream: str, offset: int
    ) -> LogChunk:
        chunk = self._call(
            self._client.process_logs,
            allocation_id,
            process_id,
            stream=stream,
            offset=offset,
            limit=_LOG_READ_BYTES,
        )
        return LogChunk(
            data=chunk.data,
            next_offset=chunk.next_offset,
            eof=chunk.eof,
            truncated=chunk.truncated,
        )

    def read_file(self, allocation_id: str, path: str) -> bytes | None:
        from stateset_sandbox.errors import SandboxApiError, SandboxError

        self._throttle.acquire()
        try:
            content: bytes = self._client.read_file_bytes(allocation_id, path)
            return content
        except SandboxError as exc:
            translated = translate_error(exc)
            if (
                isinstance(exc, SandboxApiError)
                and exc.status_code == 404
                and not isinstance(translated, AllocationLostError)
            ):
                return None
            raise translated from exc
        except ValueError as exc:
            raise BackendError(f"unreadable file {path}: {exc}") from exc

    def list_files(self, allocation_id: str, path: str) -> list[FileEntry]:
        entries = self._call(
            self._client.list_files, allocation_id, path, recursive=True
        )
        return [
            FileEntry(path=entry["path"], size=entry.get("size"))
            for entry in entries
            if entry.get("type") == "file"
        ]

    def _cost(self, allocation_id: str) -> dict[str, Any]:
        result: dict[str, Any] = self._call(
            self._client.get_sandbox_cost, allocation_id
        )
        return result

    def cost_cents(self, allocation_id: str) -> float | None:
        total = self._cost(allocation_id).get("total_cents")
        return float(total) if isinstance(total, (int, float)) else None

    def loss_reason(self, allocation_id: str) -> str | None:
        budget = self._cost(allocation_id).get("budget")
        if isinstance(budget, dict) and budget.get("exhausted_at"):
            return "budget_exhausted"
        return None

    def release(self, allocation_id: str) -> None:
        """Not throttled: a sandbox waiting to stop is a sandbox still billing."""
        from stateset_sandbox.errors import SandboxError

        try:
            self._client.stop(allocation_id)
        except (SandboxError, ValueError) as exc:
            raise translate_error(exc) from exc
