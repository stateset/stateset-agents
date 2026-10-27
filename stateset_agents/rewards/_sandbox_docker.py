"""Docker isolation backend for :mod:`stateset_agents.rewards.code_verifier`.

The verifier's harness (``_sandbox_harness.py``) runs in one fresh container
per verification (``docker run --rm``). Nothing is reused between candidates,
so no state can carry over from one run to the next. The container gets:

* ``--network none`` (loopback only) and ``--ipc none`` (no ``/dev/shm``);
* a read-only root (``--read-only``) and one small ``--tmpfs`` at ``/sandbox``
  (``nosuid,nodev,noexec``, owned by the sandbox user), which holds the empty
  working directory and a program's stdin files;
* a non-root numeric ``--user``, ``--cap-drop ALL``,
  ``--security-opt no-new-privileges`` and the daemon's default seccomp and
  AppArmor profiles (never overridden);
* ``--pids-limit``, ``--memory`` = ``--memory-swap`` (no swap) and ``--cpus``
  derived from the verifier's :class:`SandboxLimits`, on top of the rlimits
  the harness still applies to itself (so ``cpu_seconds`` decides a score
  exactly as in the rlimit backend);
* no host mounts and no environment: the harness source is a command-line
  argument, the payload (test *inputs*, never expected outputs) arrives on
  stdin, and ``/usr/bin/env -i`` starts the interpreter with an empty
  environment (the image's ``ENV`` and any proxy variables the CLI injects
  are dropped);
* ``--pull never`` (``--pull missing`` only with ``allow_pull=True``), so a
  missing image is a clear :class:`RewardUnavailableError`, never a silent
  download during training; ``--log-driver none``, so candidate output is
  never written to the daemon's disk.

The harness runs under a tiny supervisor that is PID 1 in the container: PID
1 ignores ``SIGXCPU`` (the kernel does not deliver default-action signals to
a namespace's init), so the supervisor runs the harness as its child and
reports how it ended (``128 + signal``), keeping ``status="cpu_limit"``
identical across backends. It also kills the child at its own wall-clock
deadline, so a container cannot outlive a trainer that died without killing
it.

The wall-clock backstop (``timeout_seconds``) starts when the harness reports
that it is running (its first record), not when ``docker run`` starts: a slow
or loaded daemon delays a run but never fails it. Container start-up has its
own budget (``startup_timeout_seconds``). A container that never starts, an
unreachable daemon or a missing image raise :class:`RewardUnavailableError`
(no candidate code has run, so it is an outage, not a score); once the
harness has started, everything is a score, so a candidate cannot forge an
outage.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
import re
import secrets
import shutil
import subprocess  # nosec: B404
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, NamedTuple

from stateset_agents.core.errors import ConfigurationError
from stateset_agents.core.reward_context import RewardUnavailableError

__all__ = [
    "DEFAULT_DOCKER_IMAGE",
    "DockerRunSpec",
    "DockerSandboxConfig",
    "DockerStatus",
    "HarnessOutput",
    "docker_run_command",
    "docker_sandbox_status",
    "run_harness_in_docker",
]

#: ``python:3.12-slim`` (Debian), pinned by its multi-architecture index
#: digest. Pre-pull it before training (``docker pull <this>``).
DEFAULT_DOCKER_IMAGE = (
    "python:3.12-slim@sha256:"
    "5072b08ad74609c5329ab4085a96dfa873de565fb4751a4cfcd7dcc427661df0"
)
#: The container's only writable mount.
SANDBOX_ROOT = "/sandbox"
#: The candidate's (empty) working directory, created by the harness.
WORKDIR = f"{SANDBOX_ROOT}/work"
#: A program's per-test stdin files, outside the working directory.
STDIN_DIR = f"{SANDBOX_ROOT}/stdin"
#: Every container the verifier starts carries this label.
CONTAINER_LABEL = "stateset-agents.sandbox"
_CONTAINER_PREFIX = "stateset-verifier-"
_PINNED = re.compile(r"(?:.+@sha256:[0-9a-f]{64}|sha256:[0-9a-f]{64})")
_USER = re.compile(r"([0-9]+):([0-9]+)")
_LABEL_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")
_PROBE_TIMEOUT = 20.0
_CLEANUP_TIMEOUT = 30.0


@dataclass(frozen=True)
class DockerSandboxConfig:
    """How the docker backend runs a candidate.

    ``image`` must be pinned by digest (``name@sha256:...`` or an image ID
    ``sha256:...``) unless ``allow_unpinned_image=True``; a tag can be moved
    to other content. It needs ``/usr/bin/env`` and the interpreter at
    ``python`` (the official ``python`` images have both). ``user`` is a
    numeric non-root ``uid:gid``. With ``allow_pull=False`` (the default) a
    missing image raises :class:`RewardUnavailableError` instead of being
    downloaded during training. The container's memory limit is the
    verifier's ``memory_bytes`` plus the tmpfs plus ``memory_headroom_bytes``,
    so ``RLIMIT_AS`` (a ``MemoryError`` the candidate sees, as in the rlimit
    backend) is reached before the cgroup's OOM killer. ``labels`` are added to
    every container (e.g. to find and remove test containers).
    """

    image: str = DEFAULT_DOCKER_IMAGE
    docker: str = "docker"
    python: str = "/usr/local/bin/python"
    user: str = "65534:65534"
    allow_pull: bool = False
    allow_unpinned_image: bool = False
    cpus: float = 1.0
    tmpfs_bytes: int = 64 * 1024 * 1024
    memory_headroom_bytes: int = 64 * 1024 * 1024
    startup_timeout_seconds: float = 60.0
    labels: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("image", "docker", "python"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(f"DockerSandboxConfig.{name} must be set")
        if not self.allow_unpinned_image and not _PINNED.fullmatch(self.image):
            raise ConfigurationError(
                "DockerSandboxConfig.image must be pinned by digest "
                "(name@sha256:<digest> or sha256:<image id>); pass "
                "allow_unpinned_image=True to run a tag"
            )
        user = _USER.fullmatch(self.user) if isinstance(self.user, str) else None
        if user is None or int(user.group(1)) == 0 or int(user.group(2)) == 0:
            raise ConfigurationError(
                "DockerSandboxConfig.user must be a numeric non-root 'uid:gid'"
            )
        _check_number("cpus", self.cpus, minimum=0.01)
        _check_number("tmpfs_bytes", self.tmpfs_bytes, minimum=1024 * 1024)
        _check_number("memory_headroom_bytes", self.memory_headroom_bytes, minimum=0)
        _check_number("startup_timeout_seconds", self.startup_timeout_seconds, 0.1)
        labels = dict(self.labels)
        for key, value in labels.items():
            if not isinstance(key, str) or not _LABEL_KEY.fullmatch(key):
                raise ConfigurationError(f"invalid docker label key {key!r}")
            if not isinstance(value, str):
                raise ConfigurationError(f"docker label {key!r} needs a string value")
        object.__setattr__(self, "labels", MappingProxyType(labels))

    @property
    def uid_gid(self) -> tuple[int, int]:
        match = _USER.fullmatch(self.user)
        assert match is not None  # validated in __post_init__
        return int(match.group(1)), int(match.group(2))


def _check_number(name: str, value: object, minimum: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
    ):
        raise ConfigurationError(f"DockerSandboxConfig.{name} must be >= {minimum}")


@dataclass(frozen=True)
class DockerStatus:
    """Whether the docker backend can run here, and why not."""

    usable: bool
    reason: str
    server_version: str | None = None
    security_options: tuple[str, ...] = ()


@dataclass(frozen=True)
class DockerRunSpec:
    """One verification's container: what to run and its resource limits."""

    source: str
    payload: bytes
    python_flags: tuple[str, ...]
    memory_bytes: int
    max_processes: int
    timeout_seconds: float
    max_stdout_bytes: int
    max_stderr_bytes: int
    #: Added to the tmpfs for a program's stdin files.
    extra_tmpfs_bytes: int = 0


class HarnessOutput(NamedTuple):
    """What a harness run produced, whichever backend ran it."""

    stdout: bytes
    stderr: bytes
    #: Negative for a signal, as for a local subprocess.
    returncode: int | None
    #: ``None`` when the run ended by itself; else the CPU seconds it had
    #: used when the wall-clock backstop killed it (``-1.0`` when unknown).
    stalled_cpu: float | None


# --- is docker usable here? ----------------------------------------------------

_STATUS_LOCK = threading.Lock()
_STATUS: dict[tuple[str, str, bool], DockerStatus] = {}


def _status_key(config: DockerSandboxConfig) -> tuple[str, str, bool]:
    return (config.docker, config.image, config.allow_pull)


def cached_docker_status(config: DockerSandboxConfig) -> DockerStatus | None:
    """The status already probed for ``config``, without probing."""
    return _STATUS.get(_status_key(config))


def docker_sandbox_status(
    config: DockerSandboxConfig | None = None, *, refresh: bool = False
) -> DockerStatus:
    """Probe (once per process, unless ``refresh``) whether the docker backend
    can run: the CLI is on ``PATH``, a Linux daemon answers, and the image is
    present locally (or ``allow_pull=True``). Blocks for up to a few seconds
    on the first call; never pulls."""
    config = config or DockerSandboxConfig()
    key = _status_key(config)
    with _STATUS_LOCK:
        if refresh or key not in _STATUS:
            _STATUS[key] = _probe(config)
        return _STATUS[key]


def _cli(argv: Sequence[str]) -> tuple[int, str, str]:
    """Run a docker CLI query: ``(returncode, stdout, stderr)``; -1 on failure.

    This module's only use of ``subprocess``: the docker CLI with a fixed
    argument list and no shell (``# nosec`` B404/B603 for that reason).
    """
    try:
        # A fixed docker CLI argument list, no shell.
        done = subprocess.run(  # nosec: B603
            list(argv),
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return -1, "", str(error)
    return done.returncode, done.stdout.strip(), done.stderr.strip()


def _first_line(text: str) -> str:
    return text.splitlines()[0] if text else "no output"


def _probe(config: DockerSandboxConfig) -> DockerStatus:
    executable = shutil.which(config.docker)
    if executable is None:
        return DockerStatus(False, f"the docker CLI ({config.docker!r}) is not on PATH")
    code, out, err = _cli(
        [
            executable,
            "info",
            "--format",
            '{{.ServerVersion}}|{{.OSType}}|{{join .SecurityOptions ";"}}',
        ]
    )
    version, _, rest = out.partition("|")
    os_type, _, options = rest.partition("|")
    if code != 0 or not version:
        return DockerStatus(
            False, f"the docker daemon is not reachable ({_first_line(err)})"
        )
    if os_type != "linux":
        return DockerStatus(False, f"the docker daemon runs {os_type!r} containers")
    security = tuple(item for item in options.split(";") if item)
    code, _, err = _cli(
        [executable, "image", "inspect", "--format", "{{.Id}}", config.image]
    )
    if code != 0 and not config.allow_pull:
        return DockerStatus(
            False,
            f"the sandbox image {config.image} is not present locally; pre-pull it "
            f"with `docker pull {config.image}` (or pass allow_pull=True)",
            version,
            security,
        )
    present = "present" if code == 0 else "pulled on first use"
    return DockerStatus(True, f"docker {version}, image {present}", version, security)


# --- one container per verification --------------------------------------------


def docker_run_command(
    config: DockerSandboxConfig, spec: DockerRunSpec, *, name: str
) -> list[str]:
    """The ``docker run`` argv for one verification (pure; see module docs)."""
    uid, gid = config.uid_gid
    tmpfs = config.tmpfs_bytes + max(0, spec.extra_tmpfs_bytes)
    memory = spec.memory_bytes + tmpfs + config.memory_headroom_bytes
    supervise = spec.timeout_seconds + config.startup_timeout_seconds
    argv = [config.docker, "run", "--rm", "--interactive", "--name", name]
    argv += ["--label", f"{CONTAINER_LABEL}=code-verifier"]
    for key, value in sorted(config.labels.items()):
        argv += ["--label", f"{key}={value}"]
    argv += [
        "--pull",
        "missing" if config.allow_pull else "never",
        "--network",
        "none",
        "--ipc",
        "none",
        "--read-only",
        "--tmpfs",
        f"{SANDBOX_ROOT}:rw,nosuid,nodev,noexec,size={tmpfs},mode=0700,"
        f"uid={uid},gid={gid}",
        "--workdir",
        SANDBOX_ROOT,
        "--user",
        f"{uid}:{gid}",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        # The supervisor and the harness, plus any processes the limits allow.
        "--pids-limit",
        str(spec.max_processes + 2),
        "--memory",
        str(memory),
        "--memory-swap",
        str(memory),
        "--cpus",
        f"{config.cpus:g}",
        "--log-driver",
        "none",
        "--entrypoint",
        "/usr/bin/env",
        config.image,
        "-i",
        config.python,
        *spec.python_flags,
        "-c",
        spec.source,
        "--supervise",
        f"{supervise:g}",
    ]
    return argv


async def run_harness_in_docker(
    config: DockerSandboxConfig, spec: DockerRunSpec
) -> HarnessOutput:
    """Run the harness in a fresh container and collect what it reported.

    Raises :class:`RewardUnavailableError` when the docker CLI cannot start
    or the container does not start within ``startup_timeout_seconds``. The
    container is removed on every path, including cancellation.
    """
    name = _CONTAINER_PREFIX + secrets.token_hex(8)
    argv = docker_run_command(config, spec, name=name)
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as error:
        raise RewardUnavailableError(f"cannot run the docker CLI: {error}") from error
    clean = False
    try:
        output, clean = await _exchange(config, spec, process, name)
    finally:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
        if not clean:
            await _shielded_remove(config, name)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=_CLEANUP_TIMEOUT)
    return output


async def _read(
    stream: asyncio.StreamReader | None,
    sink: bytearray,
    cap: int,
    started: asyncio.Event | None,
) -> None:
    """Read ``stream`` to EOF, keeping at most ``cap`` bytes; set ``started``
    at the first byte (the harness's first record)."""
    if stream is None:
        return
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            return
        if started is not None:
            started.set()
        room = cap - len(sink)
        if room > 0:
            sink.extend(chunk[:room])


async def _feed(process: asyncio.subprocess.Process, payload: bytes) -> None:
    assert process.stdin is not None
    try:
        process.stdin.write(payload)
        await process.stdin.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        process.stdin.close()


async def _exchange(
    config: DockerSandboxConfig,
    spec: DockerRunSpec,
    process: asyncio.subprocess.Process,
    name: str,
) -> tuple[HarnessOutput, bool]:
    """Feed the payload, wait for the start, then for the end under the
    wall-clock backstop. Returns the output and whether the container ended
    by itself (and so removed itself)."""
    out, err = bytearray(), bytearray()
    started = asyncio.Event()
    tasks: set[asyncio.Future[Any]] = {
        asyncio.ensure_future(
            _read(process.stdout, out, spec.max_stdout_bytes, started)
        ),
        asyncio.ensure_future(_read(process.stderr, err, spec.max_stderr_bytes, None)),
        asyncio.ensure_future(_feed(process, payload=spec.payload)),
    }
    exited = asyncio.ensure_future(process.wait())
    start = asyncio.ensure_future(started.wait())
    tasks.add(exited)
    stalled: float | None = None
    try:
        await asyncio.wait(
            {start, exited},
            timeout=config.startup_timeout_seconds,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not started.is_set() and not exited.done():
            raise RewardUnavailableError(
                f"the docker sandbox did not start within "
                f"{config.startup_timeout_seconds:g} s ({_first_line(err.decode())})"
            )
        # The candidate's clock: from the harness's first record.
        _, pending = await asyncio.wait(tasks, timeout=spec.timeout_seconds)
        if pending:
            used = await _container_cpu_seconds(config, name)
            await _docker_cli(config, ["kill", name])
            _, pending = await asyncio.wait(pending, timeout=_CLEANUP_TIMEOUT)
            stalled = -1.0 if used is None else used
    finally:
        for task in (*tasks, start):
            task.cancel()
    returncode = _signal_returncode(process.returncode)
    # A run that never started may leave a created container behind.
    clean = stalled is None and not pending and started.is_set()
    return HarnessOutput(bytes(out), bytes(err), returncode, stalled), clean


def _signal_returncode(code: int | None) -> int | None:
    """``128 + N`` (the supervisor's report of signal ``N``) as ``-N``."""
    if code is not None and 128 < code <= 128 + 64:
        return -(code - 128)
    return code


async def _docker_cli(
    config: DockerSandboxConfig, args: Iterable[str]
) -> tuple[int, str]:
    """Run ``docker <args>`` (bounded); ``(returncode, stdout)``, -1 on failure."""
    try:
        process = await asyncio.create_subprocess_exec(
            config.docker,
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except OSError:
        return -1, ""
    try:
        out, _ = await asyncio.wait_for(process.communicate(), _CLEANUP_TIMEOUT)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        await process.wait()
        return -1, ""
    return int(process.returncode or 0), out.decode("utf-8", "replace").strip()


async def _shielded_remove(config: DockerSandboxConfig, name: str) -> None:
    """``docker rm -f`` the container. The removal finishes even if the caller
    is cancelled meanwhile (the cancellation still propagates)."""
    removal = asyncio.ensure_future(_docker_cli(config, ["rm", "--force", name]))
    await asyncio.shield(removal)


async def _container_cpu_seconds(
    config: DockerSandboxConfig, name: str
) -> float | None:
    """CPU seconds the container's processes have used, read from ``/proc``
    (a local Linux daemon); ``None`` when that is not possible."""
    code, out = await _docker_cli(
        config, ["inspect", "--format", "{{.State.Pid}} {{.Id}}", name]
    )
    pid_text, _, container_id = out.partition(" ")
    if code != 0 or not pid_text.isdigit() or not container_id:
        return None
    return _process_tree_cpu(int(pid_text), container_id)


def _process_tree_cpu(pid: int, container_id: str) -> float | None:
    try:
        with open(f"/proc/{pid}/cgroup", encoding="utf-8") as handle:
            if container_id not in handle.read():
                return None  # not this host's pid namespace (remote or rootless)
        pids = [pid]
        for task in os.listdir(f"/proc/{pid}/task"):
            with open(f"/proc/{pid}/task/{task}/children", encoding="utf-8") as handle:
                pids += [int(child) for child in handle.read().split()]
        ticks = 0
        for member in pids:
            with open(f"/proc/{member}/stat", "rb") as handle:
                stat = handle.read().decode("ascii", "replace")
            fields = stat[stat.rindex(")") + 2 :].split()
            ticks += int(fields[11]) + int(fields[12])
        return ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None
