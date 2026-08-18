"""A persistent vLLM OpenAI-compatible endpoint on a RunPod GPU pod.

``stateset-agents serve-remote`` for people without a local GPU: rent a pod,
install vLLM, load base model (+ optional LoRA adapter), and expose an
OpenAI-compatible ``/v1`` API over the pod's public port mapping. Unlike
:mod:`stateset_agents.remote.chat_session`, the pod deliberately OUTLIVES the
CLI process — that is the whole point of serving — so the cost controls are
different in kind:

1. **A remote self-destruct.** A pod cannot terminate itself without
   credentials, and a local watchdog dies with the laptop. So the RunPod API
   key is copied to the pod (``chmod 600``, root-only container) and a
   ``nohup``-ed script sleeps for ``max_hours`` then calls the RunPod DELETE
   endpoint on its own pod id. **Tradeoff, stated plainly:** the API key
   lives on the rented machine until the pod dies. Anyone with root on the
   pod (you, or RunPod's SECURE-cloud operators) could read it. Use a
   dedicated, revocable key if that matters to you.
2. **Manual controls.** ``serve-remote --stop <name-or-id>`` terminates a
   pod immediately; ``serve-remote --list`` shows every serve pod with its
   age and $/hr so nothing leaks unnoticed.

On any *startup* failure the pod is terminated before the exception
propagates — a half-provisioned pod bills like a healthy one.

The endpoint is authenticated: vLLM is launched with a generated
``--api-key`` token, and every request must carry it as a Bearer token.

VRAM note: vLLM loads the whole model into GPU memory. The default GPU
(16 GB) fits models up to ~7B at fp16; for anything larger pick a bigger
``--gpu`` (e.g. ``"NVIDIA H100 80GB HBM3"``) and raise
``--container-disk-gb`` to ~2.5x the checkpoint size for the download.
"""

from __future__ import annotations

import secrets
import shlex
import tarfile
import tempfile
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.remote.runpod import (
    _API_ROOT,
    _DEFAULT_IMAGE,
    _REMOTE_WORKDIR,
    RunPodApi,
    SshTransport,
)

__all__ = [
    "RemoteServeSession",
    "find_serve_pod",
    "list_serve_pods",
    "self_destruct_script",
]

_POD_PREFIX = "stateset-serve-"
_REMOTE_ADAPTER_DIR = f"{_REMOTE_WORKDIR}/adapter"
_REMOTE_ADAPTER_TAR = f"{_REMOTE_WORKDIR}/adapter.tar.gz"
_REMOTE_KEY_FILE = f"{_REMOTE_WORKDIR}/.runpod_key"
_REMOTE_DESTRUCT_SCRIPT = f"{_REMOTE_WORKDIR}/self_destruct.sh"
_REMOTE_VLLM_LOG = f"{_REMOTE_WORKDIR}/vllm.log"
_VLLM_PORT = 8000
_REMOTE_MERGED_DIR = f"{_REMOTE_WORKDIR}/merged"

#: flashinfer (pulled in by vllm) annotates with ``array.array[int]``, which
#: raises TypeError at import time on the image's Python 3.11 and takes the
#: whole vLLM engine down before it ever listens. The subscript is
#: annotation-only, so stripping it in place is safe. Observed live on the
#: first verified endpoint run (2026-08-17); a no-op once flashinfer fixes it.
_FLASHINFER_PATCH_COMMAND = (
    # find_spec on the submodule would import flashinfer.comm and hit the
    # very crash being patched, so resolve the file from the package root.
    'python -c "'
    "import importlib.util, os; "
    "spec = importlib.util.find_spec('flashinfer'); "
    "root = os.path.dirname(spec.origin) if spec and spec.origin else None; "
    "path = os.path.join(root, 'comm', 'fd_exchange.py') if root else None; "
    "src = open(path).read() if path and os.path.exists(path) else ''; "
    "patched = src.replace('array.array[int]', 'array.array'); "
    "src != patched and (open(path, 'w').write(patched), print('patched flashinfer fd_exchange'))\""
)

#: HTTP GET seam for the readiness poll. Injectable so tests never touch the
#: network. Returns an HTTP status code, raising on connection failure.
HttpGet = Callable[[str, dict[str, str]], int]

#: HTTP POST seam for the adapter-effect probe. Returns the parsed JSON body.
HttpPostJson = Callable[[str, dict[str, str], dict[str, Any]], dict[str, Any]]

#: The effect-probe prompt. Any prompt works — the probe compares greedy
#: completions across models, not their content.
_PROBE_PROMPT = "Reply with one sentence: what can you help me with?"


def _default_http_get(url: str, headers: dict[str, str]) -> int:
    import requests

    return int(requests.get(url, headers=headers, timeout=10).status_code)


def _default_http_post_json(
    url: str, headers: dict[str, str], payload: dict[str, Any]
) -> dict[str, Any]:
    import requests

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    return dict(response.json())


def self_destruct_script(
    pod_id: str, max_hours: float, api_root: str = _API_ROOT
) -> str:
    """The remote self-destruct: sleep ``max_hours``, then DELETE own pod.

    Reads the API key from :data:`_REMOTE_KEY_FILE` at fire time rather than
    embedding it, so the script itself is safe to log. The key file still
    lives on the pod — see the module docstring for the tradeoff.
    """
    seconds = max(1, int(max_hours * 3600))
    return (
        "#!/bin/bash\n"
        f"# stateset-agents serve-remote cost control: terminate this pod\n"
        f"# after {max_hours} hour(s), whatever happens to the laptop that\n"
        "# started it.\n"
        f"sleep {seconds}\n"
        f'curl -s -X DELETE "{api_root}/pods/{pod_id}" '
        f'-H "Authorization: Bearer $(cat {_REMOTE_KEY_FILE})"\n'
    )


def _pod_age(pod: dict[str, Any], *, now: Callable[[], float] = time.time) -> str:
    """Human age of a pod from whichever created-at field the API returned."""
    raw = pod.get("createdAt") or pod.get("created_at")
    if not raw:
        return "?"
    try:
        created = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return "?"
    seconds = max(0.0, now() - created.astimezone(timezone.utc).timestamp())
    hours = seconds / 3600
    return f"{hours:.1f}h"


def list_serve_pods(api: Any) -> list[dict[str, Any]]:
    """Every running serve pod, as rows ready to print.

    Only pods this command created (``stateset-serve-*`` names) are shown —
    training/chat pods manage their own lifecycles.
    """
    rows = []
    for pod in api.list_pods():
        name = str(pod.get("name") or "")
        if not name.startswith(_POD_PREFIX):
            continue
        rows.append(
            {
                "id": str(pod.get("id")),
                "name": name,
                "status": str(pod.get("desiredStatus") or "?"),
                "age": _pod_age(pod),
                "cost_per_hr": pod.get("costPerHr"),
            }
        )
    return rows


def find_serve_pod(api: Any, name_or_id: str) -> dict[str, Any]:
    """Resolve ``--stop``'s argument to one pod, by exact id or name."""
    pods = list(api.list_pods())
    for pod in pods:
        if str(pod.get("id")) == name_or_id or str(pod.get("name")) == name_or_id:
            return dict(pod)
    known = ", ".join(
        sorted(
            str(p.get("name"))
            for p in pods
            if str(p.get("name") or "").startswith(_POD_PREFIX)
        )
    )
    raise RemoteExecutionError(
        f"no pod named or with id {name_or_id!r}"
        + (f"; running serve pods: {known}" if known else "; no serve pods running"),
        provider="runpod",
    )


class _PodNeverNetworked(RemoteExecutionError):
    """The pod reached RUNNING but never published an IP and port mappings.

    Distinct from a slow vLLM boot: this pod cannot serve anything, and the
    fix is a different pod rather than more patience.
    """


class RemoteServeSession:
    """Rent a pod, boot vLLM's OpenAI server on it, hand back URL + token."""

    provider = "runpod"
    #: 16 GB — fits ~7B fp16 models; pick a bigger GPU for anything larger.
    DEFAULT_GPU = "NVIDIA RTX A4000"

    def __init__(
        self,
        api: Any = None,
        ssh: Any = None,
        *,
        public_key: str | None = None,
        image: str = _DEFAULT_IMAGE,
        container_disk_gb: int = 60,
        ready_timeout_s: int = 1800,
        network_timeout_s: int = 300,
        max_provision_attempts: int = 2,
        poll_interval_s: float = 10.0,
        http_get: HttpGet | None = None,
        http_post_json: HttpPostJson | None = None,
        token: str | None = None,
    ) -> None:
        self._api = api
        self._ssh = ssh
        self._public_key = public_key
        self.image = image
        #: vLLM's pip install alone is ~10 GB of wheels on top of the model
        #: download, hence a higher floor than the training default.
        self.container_disk_gb = container_disk_gb
        #: Covers pod provisioning + `pip install vllm` + model download +
        #: weight loading. Generous on purpose: a 30-minute ceiling beats a
        #: false negative that leaks a warming-up pod (start() terminates
        #: the pod on timeout, but only because this bound exists).
        self.ready_timeout_s = ready_timeout_s
        #: Networking either appears within a couple of minutes or never —
        #: observed four times: a pod reaches RUNNING and never publishes an
        #: IP or port mapping. Sharing the (necessarily long) vLLM-load
        #: timeout with this meant burning 30 minutes of billing on a pod
        #: that was never going to serve anything.
        self.network_timeout_s = network_timeout_s
        #: A pod that never gets networking is worth abandoning for a fresh
        #: one; the failure is per-host, not per-account.
        self.max_provision_attempts = max_provision_attempts
        self.poll_interval_s = poll_interval_s
        self._pod_started_at: float | None = None
        self._pod_cost_per_hr: float | None = None
        self._base_model: str | None = None
        self._gpu: str | None = None
        self._http_get = http_get or _default_http_get
        self._http_post_json = http_post_json or _default_http_post_json
        #: Filled by the post-readiness effect probe; the CLI prints these.
        self.effect_warnings: list[str] = []
        #: The Bearer token vLLM will require. Generated unless injected.
        self.token = token or secrets.token_urlsafe(24)

        self.pod_id: str | None = None
        self.pod_name: str | None = None
        self.endpoint_url: str | None = None
        #: (host, ssh_port) once known — lets _run_checked reconnect after
        #: a transport drop.
        self._endpoint: tuple[str, int] | None = None

    # -- lazily resolved collaborators (mirrors RemoteChatSession) ---------

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

    # -- lifecycle ---------------------------------------------------------

    def _wait_for_endpoints(self, api: Any, pod_id: str) -> tuple[str, int, int]:
        """Poll until the pod is RUNNING with both 22 and 8000 mapped.

        Two different mechanisms, learned the expensive way:

        * ``22/tcp`` needs a real TCP mapping — ``publicIp`` plus
          ``portMappings["22"]`` — because ssh cannot go through an HTTP
          proxy.
        * The vLLM port is requested as ``8000/http`` and is reached at
          ``https://<pod-id>-8000.proxy.runpod.net``, RunPod's HTTP proxy,
          which needs no public IP and publishes no mapping.

        Waiting for a TCP mapping on the HTTP port is what made five
        verification attempts hang until timeout: the mapping was never
        going to appear.
        """
        deadline = time.monotonic() + self.network_timeout_s
        while True:
            pod = api.get_pod(pod_id)
            ip = pod.get("publicIp")
            mappings = pod.get("portMappings") or {}
            ssh_port = mappings.get("22")
            if pod.get("desiredStatus") == "RUNNING" and ip and ssh_port:
                return str(ip), int(ssh_port), _VLLM_PORT
            if time.monotonic() >= deadline:
                raise _PodNeverNetworked(
                    f"pod {pod_id} never published an ssh endpoint within "
                    f"{self.network_timeout_s}s (last status "
                    f"{pod.get('desiredStatus')!r}, publicIp {ip!r}, "
                    f"mappings {mappings!r})",
                    provider=self.provider,
                )
            time.sleep(self.poll_interval_s)

    def _proxy_endpoint_url(self, pod_id: str) -> str:
        """Public URL of the vLLM server, via RunPod's HTTP proxy."""
        return f"https://{pod_id}-{_VLLM_PORT}.proxy.runpod.net"

    def _run_checked(self, ssh: Any, command: str) -> None:
        """Run one setup command, absorbing a single ssh-transport death.

        Exit 255 is ssh's OWN code (the client/transport failed), not the
        remote command's — observed live: the pod's sshd dropped the
        connection mid-`pip install vllm` ("Connection closed by remote
        host") and the pod itself came back seconds later. Reconnect and
        retry once; a second 255, or any real remote failure, still raises.
        """
        exit_code, output = ssh.run(command)
        if exit_code == 255 and self._reconnect(ssh):
            exit_code, output = ssh.run(command)
        if exit_code != 0:
            raise RemoteExecutionError(
                f"remote command failed ({exit_code}): {command}\n{output}",
                provider=self.provider,
            )

    def _run_detached(
        self,
        ssh: Any,
        command: str,
        *,
        label: str,
        timeout_s: int,
        poll_s: float | None = None,
    ) -> None:
        """Run a long command detached, polling a marker for its exit code.

        Holding one ssh session open for the length of a multi-minute
        install is fragile — observed live: the transport died partway
        through ``pip install vllm`` and took the whole run with it. Each
        poll is instead its own short connection, so a dropped link costs a
        retry of the poll rather than the install.
        """
        marker = f"/workspace/.{label}.rc"
        log = f"/workspace/.{label}.log"
        launch_code, launch_output = ssh.run(
            f"rm -f {shlex.quote(marker)}; "
            f"nohup bash -c {shlex.quote(f'{command} > {log} 2>&1; echo $? > {marker}')} "
            f"> /dev/null 2>&1 < /dev/null &"
        )
        if launch_code == 255 and self._reconnect(ssh):
            launch_code, launch_output = ssh.run(
                f"rm -f {shlex.quote(marker)}; "
                f"nohup bash -c "
                f"{shlex.quote(f'{command} > {log} 2>&1; echo $? > {marker}')} "
                f"> /dev/null 2>&1 < /dev/null &"
            )
        if launch_code != 0:
            raise RemoteExecutionError(
                f"could not start {label} ({command}) on the pod "
                f"({launch_code}): {launch_output}",
                provider=self.provider,
            )
        interval = poll_s if poll_s is not None else self.poll_interval_s
        deadline = time.monotonic() + timeout_s
        while True:
            exit_code, output = ssh.run(
                f"cat {shlex.quote(marker)} 2>/dev/null || echo PENDING"
            )
            text = (output or "").strip().splitlines()
            state = text[-1] if text else ""
            if exit_code == 0 and state.isdigit():
                if state == "0":
                    return
                _, tail = ssh.run(f"tail -40 {shlex.quote(log)}")
                raise RemoteExecutionError(
                    f"{label} failed on the pod (exit {state}):\n{tail}",
                    provider=self.provider,
                )
            if time.monotonic() >= deadline:
                _, tail = ssh.run(f"tail -40 {shlex.quote(log)}")
                raise RemoteExecutionError(
                    f"{label} did not finish within {timeout_s}s:\n{tail}",
                    provider=self.provider,
                )
            time.sleep(interval)

    def _reconnect(self, ssh: Any) -> bool:
        """Wait for sshd to answer again after a transport drop."""
        if self._endpoint is None:
            return False
        host, ssh_port = self._endpoint
        try:
            ssh.wait_until_reachable(host, ssh_port, self.ready_timeout_s)
        except RemoteExecutionError:
            return False
        return True

    def _upload_adapter(
        self, ssh: Any, adapter_dir: Path, name: str = "adapter"
    ) -> None:
        """Tar, upload, untar — ``SshTransport.upload`` moves single files.

        Each adapter lands at ``/workspace/<name>`` and is served under
        ``name``, so several can ride one endpoint for A/B comparison.
        """
        tar_remote = f"{_REMOTE_WORKDIR}/{name}.tar.gz"
        with tempfile.TemporaryDirectory() as staging:
            tar_path = Path(staging) / f"{name}.tar.gz"
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(adapter_dir, arcname=name)
            ssh.upload(tar_path, tar_remote)
        self._run_checked(
            ssh,
            f"tar xzf {tar_remote} -C {_REMOTE_WORKDIR} " f"&& rm -f {tar_remote}",
        )

    def _arm_self_destruct(self, ssh: Any, api: Any, max_hours: float) -> None:
        """Install and start the remote self-destruct (see module docstring)."""
        with tempfile.TemporaryDirectory() as staging:
            key_file = Path(staging) / "runpod_key"
            key_file.write_text(str(api.api_key))
            ssh.upload(key_file, _REMOTE_KEY_FILE)
            script_file = Path(staging) / "self_destruct.sh"
            script_file.write_text(
                self_destruct_script(str(self.pod_id), max_hours, api.root)
            )
            ssh.upload(script_file, _REMOTE_DESTRUCT_SCRIPT)
        self._run_checked(
            ssh,
            # The parenthesised subshell is load-bearing. In
            # `chmod && nohup script > log & echo`, the `&` backgrounds the
            # WHOLE `chmod && nohup` chain, whose subshell then runs the
            # hour-long script in its foreground while holding the ssh
            # session's stdout/stderr — so sshd keeps the channel open until
            # the self-destruct fires and the client blocks on the arm
            # command for the pod's whole lifetime. Observed live
            # (2026-08-17): the CLI hung 28 minutes on `echo armed` while
            # the pod sat idle. The subshell scopes the `&` to nohup alone;
            # < /dev/null keeps the script off the session's stdin.
            f"chmod 600 {_REMOTE_KEY_FILE} && "
            f"(nohup bash {_REMOTE_DESTRUCT_SCRIPT} "
            f"> {_REMOTE_WORKDIR}/self_destruct.log 2>&1 < /dev/null &) "
            f"&& echo armed",
        )

    def _merge_adapter_remotely(self, ssh: Any, base_model: str, name: str) -> None:
        """Fold the uploaded adapter into full weights at ``/workspace/merged``.

        Exists because vLLM loads hybrid-Qwen3.5 LoRA adapters without error
        and silently serves the base weights — the hybrid ``linear_attn``
        target modules never match its LoRA mapping (proven by byte-identical
        greedy completions; ``docs/PROOFS.md`` 2026-08-18). peft knows the
        modules it trained, so merging applies every delta, and vLLM then
        serves an ordinary full checkpoint with no ``--enable-lora`` at all.

        Runs detached: a 30B merge is a model download plus a full-weight
        save, and a dropped ssh link must cost a poll, not the run.
        """
        import os

        from stateset_agents import __version__

        # Same seam as the training executor: STATESET_AGENTS_WHEEL ships an
        # unreleased build to the pod, because the PyPI pin cannot contain a
        # module that has not been released yet (the flywheel's first live
        # spin died exactly this way).
        env_wheel = os.environ.get("STATESET_AGENTS_WHEEL", "").strip()
        if env_wheel:
            wheel = Path(env_wheel)
            wheel_remote = f"{_REMOTE_WORKDIR}/{wheel.name}"
            ssh.upload(wheel, wheel_remote)
            requirement = f"{wheel_remote}[training]"
        else:
            requirement = f"stateset-agents[training]=={__version__}"
        # The merge runs in its OWN venv (system-site-packages for torch):
        # installing the training stack into vLLM's environment downgraded
        # its transformers and crashed the engine at boot — observed live,
        # a 30-minute readiness timeout with the root cause off the tail.
        venv_python = f"{_REMOTE_WORKDIR}/.merge-venv/bin/python"
        self._run_detached(
            ssh,
            f"python -m venv --system-site-packages "
            f"{_REMOTE_WORKDIR}/.merge-venv && "
            f"{venv_python} -m pip install --quiet '{requirement}'",
            label="merge-deps",
            timeout_s=self.ready_timeout_s,
        )
        self._run_detached(
            ssh,
            f"{venv_python} -m stateset_agents.training.merge_adapter "
            f"--base-model {shlex.quote(base_model)} "
            f"--adapter {_REMOTE_WORKDIR}/{shlex.quote(name)} "
            f"--output-dir {_REMOTE_MERGED_DIR}",
            label="merge",
            timeout_s=self.ready_timeout_s,
        )

    def _vllm_command(
        self,
        base_model: str,
        adapter_names: list[str],
        model_path: str | None = None,
    ) -> str:
        """The launch line: vLLM's built-in OpenAI-compatible server.

        ``vllm serve`` binds 0.0.0.0:8000 with ``/v1`` chat/completions
        endpoints, requiring ``Authorization: Bearer <token>`` because of
        ``--api-key``. With adapters, ``--enable-lora`` registers each as a
        served model under its own name — request any of them (or the base)
        via the ``model`` field, which is how A/B comparison works.
        """
        parts = [
            "nohup",
            "vllm",
            "serve",
            shlex.quote(model_path or base_model),
            "--host 0.0.0.0",
            f"--port {_VLLM_PORT}",
            f"--api-key {shlex.quote(self.token)}",
        ]
        if model_path is not None:
            # A merged checkpoint serves under the API name "adapter" so
            # callers address it the same way with and without --merge.
            parts += ["--served-model-name adapter"]
        if adapter_names:
            modules = " ".join(
                f"{shlex.quote(n)}={_REMOTE_WORKDIR}/{shlex.quote(n)}"
                for n in adapter_names
            )
            parts += ["--enable-lora", f"--lora-modules {modules}"]
        parts += [f"> {_REMOTE_VLLM_LOG} 2>&1 < /dev/null & echo launched"]
        return " ".join(parts)

    def _await_server_ready(self, ssh: Any) -> None:
        """Poll ``/v1/models`` (with the token) until vLLM answers 200."""
        assert self.endpoint_url is not None
        url = f"{self.endpoint_url}/v1/models"
        headers = {"Authorization": f"Bearer {self.token}"}
        deadline = time.monotonic() + self.ready_timeout_s
        while True:
            try:
                status = self._http_get(url, headers)
            except Exception:
                status = -1  # connection refused: server still booting
            if status == 200:
                return
            if status == 401:
                # Answering but rejecting OUR token — misconfiguration, and
                # more polling will not fix it.
                raise RemoteExecutionError(
                    "vLLM is up but rejected the generated API token; "
                    "refusing to serve unauthenticated",
                    provider=self.provider,
                )
            if time.monotonic() >= deadline:
                # The tail alone showed only the APIServer wrapper once; the
                # engine's root cause scrolls off it. Collect ERROR lines
                # from the whole log too.
                # The root exception sits at the END of the engine's
                # ERROR traceback — tail it (a head once cut the evidence
                # off exactly at the interesting frame).
                _, log_tail = ssh.run(
                    f"grep -E 'ERROR|Error' {_REMOTE_VLLM_LOG} | tail -n 20; "
                    f"echo '--- tail ---'; tail -n 10 {_REMOTE_VLLM_LOG}"
                )
                raise RemoteExecutionError(
                    f"vLLM did not become ready within {self.ready_timeout_s}s "
                    f"(last HTTP status {status}). Errors and tail of its "
                    f"log:\n{log_tail}",
                    provider=self.provider,
                )
            time.sleep(self.poll_interval_s)

    def _greedy_probe(self, model: str) -> str:
        """One greedy completion from ``model`` through the live endpoint."""
        assert self.endpoint_url is not None
        body = self._http_post_json(
            f"{self.endpoint_url}/v1/chat/completions",
            {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            {
                "model": model,
                "temperature": 0,
                "max_tokens": 48,
                "messages": [{"role": "user", "content": _PROBE_PROMPT}],
            },
        )
        return str(body["choices"][0]["message"]["content"])

    def _verify_adapter_effect(
        self, base_model: str, adapter_names: list[str], *, strict: bool
    ) -> None:
        """Greedy base-vs-adapter probe: identical output means no effect.

        Exists because vLLM loaded hybrid-Qwen3.5 LoRA adapters without
        error and silently served the base weights — a "successful"
        verification only caught it later by exactly this probe
        (docs/PROOFS.md, 2026-08-18). Now every adapter-serving start
        checks its own effect: identical greedy completions raise with
        ``strict``, and are reported as loud warnings otherwise. The probe
        is best-effort — a transport hiccup must not kill a healthy serve —
        but a *completed* comparison that finds no effect is never ignored.
        """
        try:
            base_completion = self._greedy_probe(base_model)
        except Exception as exc:  # noqa: BLE001 - probe transport is best-effort
            self.effect_warnings.append(
                f"adapter-effect probe skipped (base probe failed: {exc})"
            )
            return
        for name in adapter_names:
            try:
                adapter_completion = self._greedy_probe(name)
            except Exception as exc:  # noqa: BLE001
                self.effect_warnings.append(
                    f"adapter-effect probe skipped for {name!r}: {exc}"
                )
                continue
            if adapter_completion == base_completion:
                message = (
                    f"adapter {name!r} has NO EFFECT: its greedy completion "
                    "is byte-identical to the base model's. vLLM loads some "
                    "adapters (hybrid Qwen3.5 families) without error and "
                    "silently serves base weights — use --merge for those. "
                    "See docs/PROOFS.md."
                )
                if strict:
                    raise RemoteExecutionError(message, provider=self.provider)
                self.effect_warnings.append(message)

    def start(
        self,
        base_model: str,
        adapter_dir: Path | None = None,
        gpu: str | None = None,
        max_hours: float = 1.0,
        adapters: dict[str, Path] | None = None,
        merge: bool = False,
        strict_effect: bool = False,
    ) -> None:
        """Provision, arm the self-destruct, boot vLLM, block until ready.

        On success the pod KEEPS RUNNING (that is the product); on any
        failure it is terminated before the exception propagates.
        """
        if max_hours <= 0:
            raise RemoteExecutionError(
                "--max-hours must be positive; the self-destruct is the "
                "backstop that keeps a forgotten pod from billing forever",
                provider=self.provider,
            )
        # One spelling internally: ``adapter_dir`` is sugar for a single
        # adapter served under the name "adapter".
        all_adapters: dict[str, Path] = dict(adapters or {})
        if adapter_dir is not None:
            all_adapters.setdefault("adapter", Path(adapter_dir))
        if merge and len(all_adapters) != 1:
            raise RemoteExecutionError(
                "--merge folds ONE adapter into the base weights; got "
                f"{len(all_adapters)}. Serve multiple adapters without "
                "--merge, or pick one.",
                provider=self.provider,
            )
        api = self._require_api()
        public_key = self._require_public_key()
        ssh = self._require_ssh()

        for attempt in range(1, self.max_provision_attempts + 1):
            self.pod_name = f"{_POD_PREFIX}{uuid.uuid4().hex[:12]}"
            pod = api.create_pod(
                name=self.pod_name,
                image=self.image,
                gpu_type_id=gpu or self.DEFAULT_GPU,
                ports=["22/tcp", f"{_VLLM_PORT}/http"],
                env={"PUBLIC_KEY": public_key, "SSH_PUBLIC_KEY": public_key},
                container_disk_gb=self.container_disk_gb,
            )
            self.pod_id = str(pod["id"])
            self._pod_started_at = time.time()
            self._base_model = base_model
            self._gpu = gpu or self.DEFAULT_GPU
            try:
                raw_price = pod.get("costPerHr")
                self._pod_cost_per_hr = (
                    float(raw_price) if raw_price is not None else None
                )
            except (TypeError, ValueError):
                self._pod_cost_per_hr = None
            try:
                endpoints = self._wait_for_endpoints(api, self.pod_id)
                break
            except _PodNeverNetworked:
                # This host will not serve; take a different one. The dead
                # pod goes first so a retry never doubles the bill.
                self.terminate()
                if attempt >= self.max_provision_attempts:
                    raise
        else:  # pragma: no cover - loop always breaks or raises
            raise AssertionError("unreachable")

        try:
            host, ssh_port, vllm_port = endpoints
            self.endpoint_url = self._proxy_endpoint_url(self.pod_id)
            self._endpoint = (host, ssh_port)
            ssh.wait_until_reachable(host, ssh_port, self.ready_timeout_s)

            # Armed FIRST: from here even a botched vllm install cannot
            # leak a forever-billing pod past max_hours.
            self._arm_self_destruct(ssh, api, max_hours)

            for name, directory in all_adapters.items():
                self._upload_adapter(ssh, Path(directory), name)

            # The runpod/pytorch image ships torch but NOT vllm.
            self._run_detached(
                ssh,
                "pip install --quiet vllm",
                label="vllm-install",
                timeout_s=self.ready_timeout_s,
            )
            # flashinfer (pulled in by vllm) annotates with
            # ``array.array[int]``, which raises TypeError at import on the
            # image's Python 3.11 and takes the whole engine down. Strip the
            # subscript in place; it is annotation-only. Observed live on the
            # first verified endpoint run (2026-08-17).
            self._run_checked(ssh, _FLASHINFER_PATCH_COMMAND)
            if merge:
                name = next(iter(all_adapters))
                self._merge_adapter_remotely(ssh, base_model, name)
                launch = self._vllm_command(
                    base_model, [], model_path=_REMOTE_MERGED_DIR
                )
            else:
                launch = self._vllm_command(base_model, list(all_adapters))
            self._run_checked(ssh, launch)
            self._await_server_ready(ssh)
            if all_adapters and not merge:
                self._verify_adapter_effect(
                    base_model, list(all_adapters), strict=strict_effect
                )
        except BaseException:
            self.terminate()
            raise

    def _record_cost(self, pod_id: str) -> None:
        """Append this pod's cost to the ledger; never raises."""
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
                    status="serve",
                )
            )
        except Exception:  # pragma: no cover - defensive
            pass

    def terminate(self) -> None:
        """Terminate the pod now. Safe to call twice; best-effort.

        Records the pod's cost on the way out. A serve pod outlives the
        process that started it, so a pod reaped by its own ``--max-hours``
        self-destruct records nothing — the ledger reflects what this machine
        observed, and `serve-remote --list` is the live view.
        """
        pod_id, self.pod_id = self.pod_id, None
        if pod_id is not None:
            self._record_cost(pod_id)
        if pod_id is not None and self._api is not None:
            try:
                self._api.terminate_pod(pod_id)
            except Exception:
                pass  # never mask the original failure
