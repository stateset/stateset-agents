"""An ephemeral chat session against a fine-tuned model on a RunPod GPU.

``stateset-agents chat-remote`` for people without a local GPU: rent a pod,
load base model + LoRA adapter there, chat over SSH, and terminate the pod on
exit. No ports are opened beyond the SSH one RunPod exposes, and nothing
persists — when the session ends the pod dies and billing stops.

**The pod is terminated on every exit path.** Same discipline as
:mod:`stateset_agents.remote.runpod`, and for the same reason: a leaked pod
bills by the hour until someone notices. ``close()`` is idempotent, wired
into the context-manager protocol, and registered with ``atexit`` as a
best-effort backstop.

The remote side is :mod:`stateset_agents.remote.chat_repl`, launched as one
LONG-RUNNING ``ssh`` subprocess with piped stdio — ``SshTransport.run`` is
one-shot and cannot hold a loaded model between prompts. The wire protocol is
JSON lines: ``{"prompt": ...}`` up, ``{"response": ...}`` / ``{"log": ...}``
/ ``{"error": ...}`` down.
"""

from __future__ import annotations

import atexit
import json
import shlex
import subprocess
import tarfile
import tempfile
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.remote.runpod import (
    _DEFAULT_IMAGE,
    _REMOTE_WORKDIR,
    RunPodApi,
    SshTransport,
    package_pin,
)

__all__ = ["RemoteChatSession"]

_REMOTE_ADAPTER_DIR = f"{_REMOTE_WORKDIR}/adapter"
_REMOTE_ADAPTER_TAR = f"{_REMOTE_WORKDIR}/adapter.tar.gz"

#: Builds the persistent ssh process. Injectable so tests never spawn ssh.
PopenFactory = Callable[[list[str]], Any]


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

        self._pod_id: str | None = None
        self._process: Any = None
        self._closed = False

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
        ``{"ready": true}`` line. On any failure the pod is terminated before
        the exception propagates.
        """
        api = self._require_api()
        public_key = self._require_public_key()
        ssh = self._require_ssh()

        session_id = uuid.uuid4().hex[:12]
        pod = api.create_pod(
            name=f"stateset-chat-{session_id}",
            image=self.image,
            gpu_type_id=gpu or self.DEFAULT_GPU,
            ports=["22/tcp"],
            env={"PUBLIC_KEY": public_key, "SSH_PUBLIC_KEY": public_key},
            container_disk_gb=self.container_disk_gb,
        )
        self._pod_id = str(pod["id"])
        # From here on the pod exists and bills; the backstop must be armed
        # before anything that can raise.
        atexit.register(self.close)

        try:
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
        except BaseException:
            self.close()
            raise

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
        return self._process.stdout

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
        """Send one prompt, block for the reply. History lives on the pod."""
        process = self._process
        if process is None or self._closed:
            raise RemoteExecutionError(
                "chat session is not open", provider=self.provider
            )
        process.stdin.write(json.dumps({"prompt": prompt}) + "\n")
        process.stdin.flush()
        while True:
            event = self._read_event(timeout_s=timeout_s)
            if "response" in event:
                return str(event["response"])
            if "error" in event:
                raise RemoteExecutionError(
                    f"remote generation failed: {event['error']}",
                    provider=self.provider,
                )
            # {"log": ...} and any unknown event: informational, keep reading.

    # -- teardown ----------------------------------------------------------

    def close(self) -> None:
        """End the ssh channel and terminate the pod. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self.close)

        process, self._process = self._process, None
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

        pod_id, self._pod_id = self._pod_id, None
        if pod_id is not None and self._api is not None:
            # Unconditional and last: an orphaned pod bills by the hour.
            try:
                self._api.terminate_pod(pod_id)
            except Exception:
                pass  # best effort — never mask the original failure

    def __enter__(self) -> RemoteChatSession:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
