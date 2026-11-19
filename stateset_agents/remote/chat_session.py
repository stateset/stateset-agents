"""An ephemeral chat session against a fine-tuned model on a RunPod GPU.

``stateset-agents chat-remote`` for people without a local GPU: rent a pod,
load base model + LoRA adapter there, chat over SSH, and terminate the pod on
exit. No ports are opened beyond the SSH one RunPod exposes, and nothing
persists — when the session ends the pod dies and billing stops.

**Pod termination is attempted on every exit path.** Same discipline as
:mod:`stateset_agents.remote.runpod`, and for the same reason: a leaked pod
bills by the hour until someone notices. ``close()`` is idempotent, wired
into the context-manager protocol, and registered with ``atexit`` as a
best-effort backstop. A failed termination retains the pod ID and exit hook;
``state`` reports ``cleanup_pending`` so callers can retry ``close()`` or
terminate the pod through the provider. A session owns at most one pod.

The remote side is :mod:`stateset_agents.remote.chat_repl`, launched as one
LONG-RUNNING ``ssh`` subprocess with piped stdio — ``SshTransport.run`` is
one-shot and cannot hold a loaded model between prompts. The wire protocol is
JSON lines: ``{"prompt": ...}`` up, ``{"response": ...}`` / ``{"log": ...}``
/ ``{"error": ...}`` down.
"""

from __future__ import annotations

import atexit
import json
import logging
import shlex
import subprocess
import tarfile
import tempfile
import time
import uuid
from collections.abc import Callable
from contextlib import ExitStack
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import IO, Any, cast

from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.remote.runpod import (
    _DEFAULT_IMAGE,
    _REMOTE_WORKDIR,
    RunPodApi,
    SshTransport,
    package_pin,
    pod_hourly_rate,
)
from stateset_agents.remote.runpod_recovery import (
    PodLease,
    session_lease_dir,
    valid_pod_id,
)

__all__ = ["ChatSessionState", "RemoteChatSession"]

logger = logging.getLogger(__name__)

_REMOTE_ADAPTER_DIR = f"{_REMOTE_WORKDIR}/adapter"
_REMOTE_ADAPTER_TAR = f"{_REMOTE_WORKDIR}/adapter.tar.gz"

#: Builds the persistent ssh process. Injectable so tests never spawn ssh.
PopenFactory = Callable[[list[str]], Any]


class ChatSessionState(str, Enum):
    """Local chat ownership; cleanup remains pending until termination returns."""

    NEW = "new"
    STARTING = "starting"
    READY = "ready"
    CLOSING = "closing"
    CLEANUP_PENDING = "cleanup_pending"
    CLOSED = "closed"


def _default_popen(cmd: list[str]) -> Any:
    # stderr is inherited on purpose: model-download progress and loading
    # logs stream to the user's terminal while stdout stays pure protocol.
    return subprocess.Popen(  # nosec: B603 — argv built here, no shell
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered: the protocol is one JSON object per line
    )


class RemoteChatSession:
    """Rent a pod, chat with a (fine-tuned) model on it, terminate the pod."""

    provider = "runpod"
    DEFAULT_GPU = "NVIDIA H100 80GB HBM3"

    def __init__(
        self,
        api: Any = None,
        ssh: Any = None,
        *,
        public_key: str | None = None,
        image: str = _DEFAULT_IMAGE,
        container_disk_gb: int = 160,
        ready_timeout_s: int = 900,
        wheel: Path | None = None,
        package_version: str | None = None,
        poll_interval_s: float = 10.0,
        popen_factory: PopenFactory | None = None,
        lease_dir: Path | None = None,
    ) -> None:
        self._api = api
        self._ssh = ssh
        self._public_key = public_key
        self.image = image
        self.container_disk_gb = container_disk_gb
        self.ready_timeout_s = ready_timeout_s
        #: Same seam as ``RunPodExecutor.wheel``: install this locally built
        #: wheel instead of the PyPI pin, to chat with an unreleased build.
        self.wheel = Path(wheel) if wheel else None
        self.package_version = package_version
        self.poll_interval_s = poll_interval_s
        self._popen_factory = popen_factory or _default_popen
        self._lease_dir = (
            lease_dir if lease_dir is not None else session_lease_dir("chat")
        )
        self._lease: PodLease | None = None
        self._ownership = ExitStack()

        self._pod_id: str | None = None
        self._pod_started_at: float | None = None
        self._pod_cost_per_hr: float | None = None
        self._base_model: str | None = None
        self._gpu: str | None = None
        self._process: Any = None
        self._state = ChatSessionState.NEW
        # Keep start/ask/close from racing over the same pod and protocol stream.
        # Startup cleanup re-enters close(), so this must be reentrant.
        self._operation_lock = RLock()
        # Client-side mirror of the conversation. The pod keeps the
        # authoritative history for generation, but every prompt and reply
        # passes through ask(), so the transcript can be rebuilt locally —
        # no round-trip to a pod that may already be dead.
        self._messages: list[dict[str, str]] = []
        self._session_meta: dict[str, Any] = {}

    @property
    def state(self) -> ChatSessionState:
        """Return the local lifecycle state without contacting the provider."""
        return self._state

    @property
    def pod_id(self) -> str | None:
        """Return the owned pod ID, retained when cleanup needs another attempt."""
        return self._pod_id

    @property
    def _closed(self) -> bool:
        """Preserve the former closed flag without a second source of truth."""
        return self._state in {
            ChatSessionState.CLOSING,
            ChatSessionState.CLEANUP_PENDING,
            ChatSessionState.CLOSED,
        }

    # -- lazily resolved collaborators (mirrors RunPodExecutor) ------------

    def _require_api(self) -> Any:
        if self._api is not None:
            return self._api
        import os

        key = os.environ.get("RUNPOD_API_KEY", "").strip()
        if not key:
            raise RemoteExecutionError(
                "RUNPOD_API_KEY is not set; create a key at "
                "https://console.runpod.io/user/settings and export it",
                provider=self.provider,
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
            provider=self.provider,
        )

    def _require_ssh(self) -> Any:
        if self._ssh is None:
            self._ssh = SshTransport()
        return self._ssh

    def _wait_for_ssh_endpoint(self, api: Any, pod_id: str) -> tuple[str, int]:
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
                    provider=self.provider,
                )
            time.sleep(self.poll_interval_s)

    # -- lifecycle -----------------------------------------------------------

    def start(
        self, base_model: str, adapter_dir: Path | None = None, gpu: str | None = None
    ) -> None:
        """Rent the pod, install the package, ship the adapter, boot the REPL.

        Blocks until the remote model has loaded and answered with its
        ``{"ready": true}`` line. On failure, cleanup is attempted before the
        exception propagates. Once provisioning begins, use a new session for
        another start; retry pending cleanup on the original session.
        """
        with self._operation_lock:
            if self._state is not ChatSessionState.NEW:
                raise RemoteExecutionError(
                    f"chat session cannot start from {self._state.value}; "
                    "create a new RemoteChatSession for another chat",
                    provider=self.provider,
                )
            self._state = ChatSessionState.STARTING
            try:
                api = self._require_api()
                public_key = self._require_public_key()
                ssh = self._require_ssh()
            except BaseException:
                # No provisioning attempt yet: configuration can be corrected.
                self._state = ChatSessionState.NEW
                raise
            try:
                self._start(api, public_key, ssh, base_model, adapter_dir, gpu)
            except BaseException:
                self.close()
                raise
            self._state = ChatSessionState.READY

    def _start(
        self,
        api: Any,
        public_key: str,
        ssh: Any,
        base_model: str,
        adapter_dir: Path | None,
        gpu: str | None,
    ) -> None:
        """Provision and boot while the caller owns the operation lock."""
        self._session_meta = {
            "source": "chat-remote",
            "base_model": base_model,
            "adapter": str(adapter_dir) if adapter_dir is not None else None,
            "gpu": gpu or self.DEFAULT_GPU,
            "started_at": time.time(),
        }

        name = f"stateset-chat-{uuid.uuid4().hex[:12]}"
        lease = PodLease(self._lease_dir / f"{name}.json")
        self._ownership.enter_context(lease.starting())
        self._lease = lease
        pod = api.create_pod(
            name=name,
            image=self.image,
            gpu_type_id=gpu or self.DEFAULT_GPU,
            ports=["22/tcp"],
            env={"PUBLIC_KEY": public_key, "SSH_PUBLIC_KEY": public_key},
            container_disk_gb=self.container_disk_gb,
        )
        self._base_model = base_model
        self._gpu = gpu or self.DEFAULT_GPU
        pod_id = pod.get("id") if isinstance(pod, dict) else None
        if not valid_pod_id(pod_id):
            raise RemoteExecutionError(
                f"RunPod returned no usable ID for {name}. Inspect runpod-orphans --chat.",
                provider=self.provider,
            )
        self._pod_id = pod_id
        # Billing starts at creation; the ledger needs both ends.
        self._pod_started_at = time.time()
        self._pod_cost_per_hr = pod_hourly_rate(pod.get("costPerHr"))
        # From here on the pod exists and bills; the backstop must be armed
        # before anything that can raise.
        atexit.register(self.close)
        lease.record_id(pod_id)

        host, port = self._wait_for_ssh_endpoint(api, self._pod_id)
        ssh.wait_until_reachable(host, port, self.ready_timeout_s)

        if self.wheel:
            ssh.upload(self.wheel, f"{_REMOTE_WORKDIR}/{self.wheel.name}")
        pin = package_pin(self.wheel, self.package_version)
        self._run_checked(ssh, f"pip install --quiet '{pin}'")

        command = [
            "python",
            "-m",
            "stateset_agents.remote.chat_repl",
            "--base-model",
            shlex.quote(base_model),
        ]
        if adapter_dir is not None:
            self._upload_adapter(ssh, Path(adapter_dir))
            command += ["--adapter", _REMOTE_ADAPTER_DIR]

        self._process = self._popen_factory(
            self._ssh_argv(ssh, host, port, " ".join(command))
        )
        self._await_ready()

    def _run_checked(self, ssh: Any, command: str) -> None:
        exit_code, output = ssh.run(command)
        if exit_code != 0:
            raise RemoteExecutionError(
                f"remote command failed ({exit_code}): {command}\n{output}",
                provider=self.provider,
            )

    def _upload_adapter(self, ssh: Any, adapter_dir: Path) -> None:
        """Ship a local adapter directory to the pod.

        ``SshTransport.upload`` moves single files, so the directory is
        tarred locally, uploaded as one file, and untarred on the pod.
        """
        with tempfile.TemporaryDirectory() as staging:
            tar_path = Path(staging) / "adapter.tar.gz"
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(adapter_dir, arcname="adapter")
            ssh.upload(tar_path, _REMOTE_ADAPTER_TAR)
        self._run_checked(
            ssh,
            f"tar xzf {_REMOTE_ADAPTER_TAR} -C {_REMOTE_WORKDIR} "
            f"&& rm -f {_REMOTE_ADAPTER_TAR}",
        )

    def _ssh_argv(self, ssh: Any, host: str, port: int, command: str) -> list[str]:
        """The persistent-channel ssh invocation, sharing SshTransport's options."""
        base_opts = getattr(ssh, "_base_opts", None)
        opts = base_opts() if callable(base_opts) else SshTransport()._base_opts()
        user = getattr(ssh, "user", "root")
        return ["ssh", *opts, "-p", str(port), f"{user}@{host}", command]

    # -- the wire protocol -----------------------------------------------

    def _stdout(self) -> IO[str]:
        if self._process is None or self._process.stdout is None:
            raise RemoteExecutionError(
                "chat session is not started", provider=self.provider
            )
        return cast(IO[str], self._process.stdout)

    def _read_event(self, timeout_s: float | None) -> dict[str, Any]:
        """Read the next protocol line, raising on EOF, garbage, or timeout."""
        stream = self._stdout()
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            if deadline is not None:
                self._enforce_deadline(stream, deadline)
            line = stream.readline()
            if line == "":
                raise RemoteExecutionError(
                    "remote chat process ended unexpectedly (EOF on its stdout); "
                    "its stderr above may say why",
                    provider=self.provider,
                )
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                raise RemoteExecutionError(
                    f"remote chat process wrote a non-protocol line to stdout: "
                    f"{line!r}",
                    provider=self.provider,
                ) from None
            if isinstance(event, dict):
                return event

    @staticmethod
    def _enforce_deadline(stream: IO[str], deadline: float) -> None:
        """Wait for the stream to become readable, or raise at ``deadline``.

        Only possible when the stream is a real pipe; test fakes built on
        in-memory buffers have no useful ``fileno`` and are always "ready".
        """
        try:
            fd = stream.fileno()
        except (AttributeError, OSError, ValueError):
            return
        import select

        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
            raise RemoteExecutionError(
                "timed out waiting for the remote chat process", provider="runpod"
            )

    def _await_ready(self) -> None:
        """Block until the remote model reports ``{"ready": true}``."""
        while True:
            event = self._read_event(timeout_s=self.ready_timeout_s)
            if event.get("ready"):
                return
            if "error" in event:
                raise RemoteExecutionError(
                    f"remote chat process failed to start: {event['error']}",
                    provider=self.provider,
                )
            # Anything else ({"log": ...} etc.) is startup chatter — skip it.

    def ask(self, prompt: str, timeout_s: int = 120) -> str:
        """Send one prompt, block for the reply.

        Generation history lives on the pod; a client-side mirror feeds
        :attr:`transcript`.
        """
        with self._operation_lock:
            return self._ask(prompt, timeout_s)

    def _ask(self, prompt: str, timeout_s: int) -> str:
        process = self._process
        if process is None or self._state is not ChatSessionState.READY:
            raise RemoteExecutionError(
                "chat session is not open", provider=self.provider
            )
        process.stdin.write(json.dumps({"prompt": prompt}) + "\n")
        process.stdin.flush()
        while True:
            event = self._read_event(timeout_s=timeout_s)
            if "response" in event:
                reply = str(event["response"])
                # Recorded only on success, mirroring chat_repl's rollback:
                # a failed generation never entered the pod's history either.
                self._messages.append({"role": "user", "content": prompt})
                self._messages.append({"role": "assistant", "content": reply})
                return reply
            if "error" in event:
                raise RemoteExecutionError(
                    f"remote generation failed: {event['error']}",
                    provider=self.provider,
                )
            # {"log": ...} and any unknown event: informational, keep reading.

    @property
    def transcript(self) -> dict[str, Any]:
        """The conversation so far, in OpenAI chat format.

        One ``json.dumps`` of this dict is a valid line for
        ``stateset-agents ingest --format openai`` — the shape
        :func:`stateset_agents.data.trajectory_ingest.from_openai_jsonl`
        parses: ``{"messages": [{"role", "content"}, ...], "metadata": {...}}``.
        Only successfully answered turns appear. Safe to read after
        ``close()`` — the history is client-side.
        """
        return {
            "messages": [dict(m) for m in self._messages],
            "metadata": dict(self._session_meta),
        }

    # -- teardown ----------------------------------------------------------

    def _record_cost(self, pod_id: str) -> None:
        """Append this session's pod cost to the ledger.

        A chat pod costs money exactly like a training pod; leaving it out
        made ``stateset-agents costs`` under-report real spend. Never raises:
        bookkeeping must not break teardown.
        """
        from stateset_agents.remote.ledger import (
            CostEntry,
            estimate_cost_usd,
            record_entry,
        )

        try:
            duration = (
                time.time() - self._pod_started_at
                if self._pod_started_at is not None
                else None
            )
            record_entry(
                CostEntry(
                    provider=self.provider,
                    job_id=pod_id,
                    base_model=self._base_model or "?",
                    gpu=self._gpu or "?",
                    cost_per_hr=self._pod_cost_per_hr,
                    duration_s=round(duration, 1) if duration is not None else None,
                    cost_usd=estimate_cost_usd(self._pod_cost_per_hr, duration),
                    status="chat",
                )
            )
        except Exception:  # pragma: no cover - defensive
            pass

    def close(self) -> None:
        """Close the channel and attempt termination; retry if cleanup is pending.

        Provider errors are logged without masking an existing chat/start error.
        Inspect ``state`` and ``pod_id`` when termination cannot be confirmed.
        """
        with self._operation_lock:
            self._close()

    def _close(self) -> None:
        if self._state in {ChatSessionState.CLOSED, ChatSessionState.CLOSING}:
            return
        self._state = ChatSessionState.CLOSING

        process, self._process = self._process, None
        try:
            if process is not None:
                try:
                    if process.stdin is not None:
                        process.stdin.close()  # EOF → chat_repl exits 0
                    process.wait(timeout=10)
                except Exception:
                    try:
                        process.kill()
                    except Exception:
                        pass
        finally:
            self._terminate_pod()

    def _terminate_pod(self) -> None:
        """Release ownership only after the provider accepts termination."""
        pod_id = self._pod_id
        if pod_id is None and self._lease is not None:
            self._state = ChatSessionState.CLEANUP_PENDING
            self._ownership.close()
            logger.warning(
                "Allocation of %s is unconfirmed. Inspect stateset-agents runpod-orphans --chat.",
                self._lease.path.stem,
            )
            return
        if pod_id is not None:
            try:
                self._require_api().terminate_pod(pod_id)
            except BaseException as error:
                self._state = ChatSessionState.CLEANUP_PENDING
                logger.warning(
                    "Chat pod %s termination is unconfirmed; retry close() or "
                    "terminate it through RunPod. Billing may continue.",
                    pod_id,
                )
                if not isinstance(error, Exception):
                    raise
                return
            self._pod_id = None
            try:
                if self._lease is not None:
                    self._lease.clear()
            except RemoteExecutionError as error:
                logger.warning("Chat pod %s deletion was accepted. %s", pod_id, error)
            finally:
                self._lease = None
                self._ownership.close()
        self._state = ChatSessionState.CLOSED
        atexit.unregister(self.close)
        if pod_id is not None:
            self._record_cost(pod_id)

    def __enter__(self) -> RemoteChatSession:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
