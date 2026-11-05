"""Sandboxed code verifier: run candidate code against hidden tests.

:class:`SandboxedCodeVerifier` extracts the code from the reply (the last
fenced block, else the whole reply), runs it against test cases in a fresh
subprocess and scores the pass rate in ``[0, 1]``. Tests are either function
tests ``{"args": [...], "kwargs": {...}, "expected": value}`` against an
``entry_point``, or program tests ``{"stdin": "...", "expected_stdout": "..."}``.
They come from the reward context (``tests``/``entry_point``, i.e. the
scenario) or the constructor, and should be *hidden*: different from any
examples shown in the prompt.

Two isolation backends run the same harness with the same limits, framing and
scoring (``isolation=``; ``"auto"`` is the default):

* ``"docker"``: one fresh container per verification, with no network, a
  read-only root and a small tmpfs, a non-root user, all capabilities
  dropped, ``no-new-privileges``, the default seccomp and AppArmor profiles,
  pids/memory/CPU cgroup limits, no host mounts and an empty environment
  (:class:`DockerSandboxConfig`, ``_sandbox_docker``). The image is pinned by
  digest and must be pre-pulled (``docker pull`` it; a missing image raises
  ``RewardUnavailableError`` unless ``allow_pull=True``).
* ``"rlimit"``: a local ``python -I -S`` subprocess (POSIX only) with an
  empty environment, a fresh, empty temporary working directory, its own
  session (the whole process group is killed afterwards) and, on Linux 5.13+,
  Landlock filesystem confinement (it reads only the standard library and
  system libraries and writes only its working directory; no ``/proc`` of
  other processes, home directory or dataset files). It does not isolate the
  network or signals to this user's processes. As root it needs an explicit
  opt-in (``allow_root_rlimit=True`` or ``STATESET_SANDBOX_ALLOW_ROOT=1``),
  because root ignores ``RLIMIT_NPROC``.
* ``"auto"`` uses docker when a daemon is reachable and the image is present,
  else rlimit with a one-time warning naming what is not isolated. It decides
  once per process and never falls back after choosing docker.

In both, the harness applies rlimits to itself before any candidate code
runs: CPU seconds, address space, file size, process count (``RLIMIT_NPROC``
0: no forks or threads), open files, no core dumps; stdout and stderr are
read with caps (the rest is drained and discarded).

The time limit that decides a score is CPU time
(``SandboxLimits.cpu_seconds``, ``RLIMIT_CPU``; a run killed by it reports
``status="cpu_limit"``), not the wall clock, so waiting for a CPU no longer
fails a run. CPU time itself still varies with contention (about 2x between
an idle machine and a fully loaded one with hyperthreads and frequency
scaling), so keep ``cpu_seconds`` several times what a correct solution
needs. The wall clock (``timeout_seconds``) is only a generous backstop that
guarantees a stalled run (a sleep, a blocked read) cannot hang the trainer;
a run it kills reports ``"wall_timeout"`` and, on Linux, the CPU seconds it
had used, so a stall is told apart from a CPU-limit kill. In a container the
backstop starts when the harness is running, so container start-up never
counts against it. At most :func:`sandbox_concurrency` sandboxes (default:
the CPUs this process may use; ``STATESET_SANDBOX_CONCURRENCY`` or
:func:`set_sandbox_concurrency` to change it) run at once across every event
loop and thread of the process, whichever backend runs them; the others
wait for a slot before their clocks start.

How a verdict resists the obvious exploits:

* expected outputs never enter the sandbox: the harness receives test
  *inputs* only and reports each return value as strict JSON (or a program's
  captured stdout); this process compares them. Printing the expected output
  without computing it needs the hidden tests, the working directory holds no
  tests file, and an object whose ``__eq__`` always returns True cannot be
  encoded;
* the harness frames its records with a per-run random nonce written to a
  private copy of stdout, and points fd 1 at ``/dev/null``: stray output is
  never parsed as a result, and a test with two records (a forged one) fails;
* only tests with a record count: ``sys.exit``/``os._exit`` before or during
  the tests, a crash, a limit kill or a timeout (an infinite loop) leaves the
  remaining tests failed. A timeout scores 0 for those tests; it is not an
  outage.

What each backend does *not* isolate: the candidate always sees the test
inputs (they are in its own process); under ``rlimit`` it runs as the
trainer's user, can open network connections, can signal (kill) that user's
processes and, without Landlock (macOS, Linux before 5.13), reads anything
that user can (``/proc/<pid>/environ``, credential files, a dataset holding
the expected outputs) and writes outside its directory. A container shares
the host kernel, so a kernel exploit escapes it. **For untrusted or
adversarial workloads use the docker backend, or a stronger runtime (gVisor
via ``--runtime=runsc``, a Firecracker microVM) around the trainer.**
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import logging
import math
import os
import platform
import re
import secrets
import signal
import sys
import sysconfig
import tempfile
import threading
from collections import deque
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stateset_agents.core.environments.json_action import strict_json_loads
from stateset_agents.core.errors import ConfigurationError
from stateset_agents.core.reward_base import RewardFunction, RewardResult, RewardType
from stateset_agents.core.reward_context import (
    RewardUnavailableError,
    assistant_turns_in_scope,
    unanswered_termination,
)
from stateset_agents.core.tool_calls import strip_tool_calls

from ._sandbox_docker import (
    DEFAULT_DOCKER_IMAGE,
    STDIN_DIR,
    WORKDIR,
    DockerRunSpec,
    DockerSandboxConfig,
    DockerStatus,
    HarnessOutput,
    cached_docker_status,
    docker_sandbox_status,
    run_harness_in_docker,
)

__all__ = [
    "DEFAULT_DOCKER_IMAGE",
    "ISOLATION_MODES",
    "DockerSandboxConfig",
    "DockerStatus",
    "SandboxLimits",
    "SandboxRun",
    "SandboxedCodeVerifier",
    "docker_sandbox_status",
    "extract_code",
    "landlock_available",
    "normalize_tests",
    "resolve_isolation",
    "run_sandboxed",
    "sandbox_available",
    "sandbox_concurrency",
    "score_outcomes",
    "set_sandbox_concurrency",
]

logger = logging.getLogger(__name__)

_HARNESS = Path(__file__).with_name("_sandbox_harness.py")
_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.S)
_CODE_LANGUAGES = {"", "python", "py", "python3"}
#: Environment variable that sets how many sandboxes may run at once.
CONCURRENCY_ENV = "STATESET_SANDBOX_CONCURRENCY"
#: Environment variable that opts in to the rlimit backend as root.
ALLOW_ROOT_ENV = "STATESET_SANDBOX_ALLOW_ROOT"
#: The ``isolation=`` values: pick a backend, or let ``"auto"`` pick.
ISOLATION_MODES = ("auto", "docker", "rlimit")


@dataclass(frozen=True)
class SandboxLimits:
    """Per-run limits.

    ``cpu_seconds`` is the time limit that decides scores: CPU time for the
    whole run (``RLIMIT_CPU``), which waiting for a CPU does not consume
    (contention still slows each CPU second, about 2x under full load, so
    leave headroom). ``timeout_seconds`` is a wall-clock backstop for runs
    that stall without using CPU (sleeping, blocked); keep it several times
    ``cpu_seconds`` so that load never reaches it.
    """

    timeout_seconds: float = 30.0
    cpu_seconds: int = 5
    memory_bytes: int = 512 * 1024 * 1024
    file_size_bytes: int = 1024 * 1024
    max_processes: int = 0
    max_open_files: int = 64
    max_output_bytes: int = 1024 * 1024
    max_stderr_bytes: int = 8192
    max_result_chars: int = 65536
    max_code_chars: int = 65536

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigurationError(f"SandboxLimits.{name} must be a number")
            if not math.isfinite(value) or value < 0:
                raise ConfigurationError(f"SandboxLimits.{name} must be >= 0")
        if self.timeout_seconds <= 0 or self.cpu_seconds <= 0:
            raise ConfigurationError("SandboxLimits timeouts must be positive")

    def rlimits(self) -> dict[str, tuple[int, int]]:
        cpu = int(math.ceil(self.cpu_seconds))
        return {
            "RLIMIT_CPU": (cpu, cpu + 1),
            "RLIMIT_AS": (int(self.memory_bytes), int(self.memory_bytes)),
            "RLIMIT_FSIZE": (int(self.file_size_bytes), int(self.file_size_bytes)),
            "RLIMIT_NPROC": (int(self.max_processes), int(self.max_processes)),
            "RLIMIT_NOFILE": (int(self.max_open_files), int(self.max_open_files)),
            "RLIMIT_CORE": (0, 0),
        }


@dataclass(frozen=True)
class SandboxRun:
    """What one sandboxed run reported: a record (or ``None``) per test."""

    outcomes: tuple[dict[str, Any] | None, ...]
    status: str
    returncode: int | None = None
    stderr: str = ""
    limits_applied: tuple[str, ...] = ()
    duplicate_records: int = 0
    load_error: str | None = None
    timed_out: bool = False
    #: CPU seconds the run used: reported by the harness when it finished, or
    #: read when the wall-clock backstop killed it (``None`` when unknown).
    cpu_seconds: float | None = None
    #: An error raised by code after the entry point was defined (an example
    #: block); the tests still ran.
    load_warning: str | None = None
    #: The backend that ran it: ``"docker"`` or ``"rlimit"``.
    isolation: str = "rlimit"


def sandbox_available() -> bool:
    """Whether this platform can run the rlimit backend (POSIX with
    ``resource``). The docker backend needs only a reachable daemon."""
    if os.name != "posix" or not sys.executable:
        return False
    try:
        import resource  # noqa: F401
    except ImportError:  # pragma: no cover - non-POSIX
        return False
    return True


def extract_code(reply: str) -> str:
    """The last fenced Python block of ``reply``, else the reply itself.

    Tool-call markup is removed first, so a tool call is never run as code.
    """
    text = strip_tool_calls(reply)
    blocks = [
        body
        for language, body in _FENCE.findall(text)
        if language.lower() in _CODE_LANGUAGES
    ]
    if blocks:
        return str(blocks[-1])
    return "" if "```" in text else text


def normalize_tests(tests: Any) -> tuple[str, list[dict[str, Any]]]:
    """``("function" | "program", tests)`` with expected values JSON-normalized.

    Raises ``ConfigurationError`` for missing, mixed or malformed tests.
    """
    if not isinstance(tests, Sequence) or isinstance(tests, (str, bytes)) or not tests:
        raise ConfigurationError("the code verifier needs a non-empty list of tests")
    modes = set()
    normalized = []
    for index, test in enumerate(tests):
        if not isinstance(test, Mapping):
            raise ConfigurationError(f"test {index} must be a mapping")
        if "expected_stdout" in test:
            modes.add("program")
            normalized.append(
                {
                    "stdin": str(test.get("stdin", "")),
                    "expected_stdout": str(test["expected_stdout"]),
                }
            )
        elif "expected" in test:
            modes.add("function")
            normalized.append(_function_test(index, test))
        else:
            raise ConfigurationError(
                f"test {index} needs 'expected' (function) or 'expected_stdout'"
            )
    if len(modes) != 1:
        raise ConfigurationError("tests must be all function tests or all programs")
    return modes.pop(), normalized


def _function_test(index: int, test: Mapping[str, Any]) -> dict[str, Any]:
    args, kwargs = test.get("args", []), test.get("kwargs", {})
    if not isinstance(args, (list, tuple)) or not isinstance(kwargs, Mapping):
        raise ConfigurationError(f"test {index}: args must be a list, kwargs a mapping")
    try:
        # Round-trip: tuples become lists, as the candidate's results will.
        roundtrip = json.loads(
            json.dumps(
                {
                    "args": list(args),
                    "kwargs": dict(kwargs),
                    "expected": test["expected"],
                },
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"test {index} is not JSON data: {error}") from error
    return dict(roundtrip)


def _json_equal(actual: Any, expected: Any, tolerance: float) -> bool:
    """Type-strict JSON equality (``True`` is not ``1``) with float tolerance."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if tolerance > 0 and (isinstance(expected, float) or isinstance(actual, float)):
            return math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance)
        return actual == expected
    if isinstance(expected, list) and isinstance(actual, list):
        return len(actual) == len(expected) and all(
            _json_equal(a, e, tolerance) for a, e in zip(actual, expected, strict=True)
        )
    if isinstance(expected, dict) and isinstance(actual, dict):
        return set(actual) == set(expected) and all(
            _json_equal(actual[key], expected[key], tolerance) for key in expected
        )
    return type(actual) is type(expected) and actual == expected


def _stdout_lines(text: str) -> list[str]:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return lines


def _passes(
    outcome: Mapping[str, Any] | None, test: Mapping[str, Any], tol: float
) -> bool:
    if outcome is None or "error" in outcome:
        return False
    if "expected_stdout" in test:
        stdout = outcome.get("stdout")
        return (
            outcome.get("status") == "ok"
            and isinstance(stdout, str)
            and _stdout_lines(stdout) == _stdout_lines(test["expected_stdout"])
        )
    encoded = outcome.get("value")
    if not isinstance(encoded, str):
        return False
    try:
        actual = strict_json_loads(encoded)
    except ValueError:
        return False
    return _json_equal(actual, test["expected"], tol)


def score_outcomes(
    run: SandboxRun, tests: Sequence[Mapping[str, Any]], tolerance: float = 0.0
) -> int:
    """How many ``tests`` the run passed (compared here, outside the sandbox)."""
    return sum(
        _passes(outcome, test, tolerance)
        for outcome, test in zip(run.outcomes, tests, strict=True)
    )


# --- how many sandboxes run at once ---------------------------------------------


def _default_concurrency() -> int:
    """``STATESET_SANDBOX_CONCURRENCY``, else the CPUs this process may use."""
    configured = os.environ.get(CONCURRENCY_ENV, "").strip()
    if configured:
        try:
            value = int(configured)
        except ValueError as error:
            raise ConfigurationError(
                f"{CONCURRENCY_ENV} must be a positive integer, got {configured!r}"
            ) from error
        if value < 1:
            raise ConfigurationError(f"{CONCURRENCY_ENV} must be at least 1")
        return value
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):  # not Linux
        return max(1, os.cpu_count() or 1)


class _SandboxSlots:
    """A process-wide bound on simultaneous sandboxes.

    One counter shared by every event loop and thread: an ``asyncio``
    semaphore belongs to a single loop, and trainers may score from several.
    A released slot passes straight to the oldest waiter (first come, first
    served); a waiter cancelled while its slot was being handed over passes
    it on.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._limit: int | None = None
        self._active = 0
        self._waiters: deque[tuple[asyncio.AbstractEventLoop, asyncio.Future[None]]] = (
            deque()
        )

    def limit(self) -> int:
        with self._lock:
            return self._limit_locked()

    def _limit_locked(self) -> int:
        if self._limit is None:
            self._limit = _default_concurrency()
        return self._limit

    def set_limit(self, limit: int | None) -> None:
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
        ):
            raise ConfigurationError("sandbox concurrency must be an integer >= 1")
        with self._lock:
            self._limit = limit
            while self._waiters and self._active < self._limit_locked():
                self._active += 1
                self._hand_over_locked()

    def active(self) -> int:
        with self._lock:
            return self._active

    async def acquire(self) -> None:
        loop = asyncio.get_running_loop()
        with self._lock:
            if not self._waiters and self._active < self._limit_locked():
                self._active += 1
                return
            waiter: asyncio.Future[None] = loop.create_future()
            self._waiters.append((loop, waiter))
        try:
            await waiter
        except asyncio.CancelledError:
            with self._lock:
                if (loop, waiter) in self._waiters:
                    self._waiters.remove((loop, waiter))
                    raise
            # The slot was handed over: give it back. (When the hand-over has
            # not run yet, it finds the waiter cancelled and passes it on.)
            if waiter.done() and not waiter.cancelled():
                self.release()
            raise

    def release(self) -> None:
        with self._lock:
            self._hand_over_locked()

    def _hand_over_locked(self) -> None:
        """Pass one held slot to the oldest live waiter, or free it (also
        when a lowered limit means fewer slots should be held)."""
        while self._waiters and self._active <= self._limit_locked():
            loop, waiter = self._waiters.popleft()
            try:
                loop.call_soon_threadsafe(self._grant, waiter)
            except RuntimeError:  # that loop is closed: try the next waiter
                continue
            return
        self._active -= 1

    def _grant(self, waiter: asyncio.Future[None]) -> None:
        if waiter.done():  # cancelled meanwhile: pass the slot on
            self.release()
        else:
            waiter.set_result(None)

    @contextlib.asynccontextmanager
    async def hold(self) -> AsyncIterator[None]:
        await self.acquire()
        try:
            yield
        finally:
            self.release()


_SLOTS = _SandboxSlots()


def sandbox_concurrency() -> int:
    """How many sandboxes may run at once in this process."""
    return _SLOTS.limit()


def set_sandbox_concurrency(limit: int | None) -> None:
    """Bound the sandboxes that run at once (``None``: back to the default,
    ``STATESET_SANDBOX_CONCURRENCY`` or the CPUs this process may use).

    Waiting runs start when a slot frees; their clocks start only then.
    """
    _SLOTS.set_limit(limit)


# --- which backend runs a candidate ---------------------------------------------

_DEFAULT_DOCKER = DockerSandboxConfig()
#: Machines whose Landlock syscalls are 444-446 (the shared syscall table).
_LANDLOCK_MACHINES = frozenset(
    {
        "x86_64",
        "amd64",
        "i386",
        "i686",
        "aarch64",
        "arm64",
        "armv7l",
        "armv8l",
        "riscv64",
        "ppc64le",
        "s390x",
        "loongarch64",
    }
)
#: Read-only for a Landlock-confined candidate, besides the standard library:
#: shared libraries, the linker cache, the time zone and random devices.
_SYSTEM_READ_PATHS = (
    "/usr",
    "/lib",
    "/lib32",
    "/lib64",
    "/etc/ld.so.cache",
    "/etc/localtime",
    "/dev/zero",
    "/dev/random",
    "/dev/urandom",
)
_WARNED: set[str] = set()
_WARNED_LOCK = threading.Lock()


@functools.lru_cache(maxsize=1)
def landlock_available() -> int:
    """The Landlock ABI version the rlimit backend can use (0: none).

    Landlock (Linux 5.13+, enabled in the kernel's LSM list) lets the
    unprivileged harness confine its own filesystem access; it needs no
    namespaces and changes nothing in this process (this only queries it).
    """
    if not sys.platform.startswith("linux"):
        return 0
    if platform.machine().lower() not in _LANDLOCK_MACHINES:
        return 0
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        syscall = libc.syscall
        syscall.restype = ctypes.c_long
        word = ctypes.c_long
        # landlock_create_ruleset(NULL, 0, LANDLOCK_CREATE_RULESET_VERSION)
        abi = syscall(word(444), None, word(0), word(1))
    except (ImportError, OSError, AttributeError):
        return 0
    return max(0, int(abi))


@functools.lru_cache(maxsize=2)
def _landlock_read_paths(site_packages: bool) -> tuple[str, ...]:
    """What a Landlock-confined candidate may read (no user data)."""
    names = ["stdlib", "platstdlib"]
    if site_packages:
        names += ["purelib", "platlib"]
    paths = {sysconfig.get_path(name) for name in names}
    paths.add(os.path.join(sys.base_prefix, "lib"))
    paths.update(_SYSTEM_READ_PATHS)
    return tuple(sorted(path for path in paths if path and os.path.exists(path)))


def _warn_once(key: str, message: str) -> None:
    with _WARNED_LOCK:
        if key in _WARNED:
            return
        _WARNED.add(key)
    logger.warning(message)


def _check_isolation(isolation: Any) -> str:
    if isolation not in ISOLATION_MODES:
        raise ConfigurationError(
            f"isolation must be one of {', '.join(ISOLATION_MODES)}; got {isolation!r}"
        )
    return str(isolation)


def _check_root(allow_root_rlimit: bool) -> None:
    """Refuse the rlimit backend as root unless explicitly allowed."""
    if allow_root_rlimit or not (hasattr(os, "geteuid") and os.geteuid() == 0):
        return
    if os.environ.get(ALLOW_ROOT_ENV, "").strip().lower() in {"1", "true", "yes"}:
        return
    raise ConfigurationError(
        "the rlimit sandbox backend would run candidate code as root, where "
        "RLIMIT_NPROC does not hold (a candidate can fork without limit); use "
        "isolation='docker', run the trainer as a non-root user, or opt in with "
        f"allow_root_rlimit=True or {ALLOW_ROOT_ENV}=1"
    )


def _rlimit_gaps() -> str:
    """What the rlimit backend does not isolate here, in words."""
    gaps = [
        "the network (it can open sockets and connect out)",
        "signals to this user's processes (it can kill the trainer)",
        "the test inputs (they are in its own process)",
    ]
    if not landlock_available():
        gaps.insert(
            0,
            "the filesystem and /proc (without Landlock it runs as this user: it "
            "reads /proc/<pid>/environ of this user's processes, credential files "
            "and any dataset holding expected outputs, and writes outside its "
            "directory)",
        )
    return "; ".join(gaps)


def resolve_isolation(
    isolation: str = "auto",
    *,
    docker: DockerSandboxConfig | None = None,
    allow_root_rlimit: bool = False,
) -> str:
    """The backend that runs candidates for these settings: ``"docker"`` or
    ``"rlimit"``.

    ``"auto"`` picks docker when :func:`docker_sandbox_status` finds a usable
    daemon and image (probed once per process), else rlimit with a one-time
    warning naming what is not isolated. Raises ``RewardUnavailableError``
    when the requested backend cannot run here and ``ConfigurationError`` for
    the rlimit backend as root without the opt-in. May block on the first call
    while Docker is probed.
    """
    isolation = _check_isolation(isolation)
    config = docker or _DEFAULT_DOCKER
    status: DockerStatus | None = None
    if isolation != "rlimit":
        status = docker_sandbox_status(config)
        if isolation == "docker" and not status.usable:
            status = docker_sandbox_status(config, refresh=True)
        if status.usable:
            if not any(o.startswith("name=seccomp") for o in status.security_options):
                _warn_once(
                    "no-seccomp",
                    "the docker daemon reports no seccomp profile: sandbox "
                    "containers run without a system-call filter",
                )
            return "docker"
        if isolation == "docker":
            raise RewardUnavailableError(f"isolation='docker' but {status.reason}")
    if not sandbox_available():
        why = f" (Docker: {status.reason})" if status is not None else ""
        raise RewardUnavailableError(
            "the sandboxed code verifier needs Docker or a POSIX platform with the "
            f"resource module (rlimits){why}; run it on Linux or macOS, or with Docker"
        )
    _check_root(allow_root_rlimit)
    if status is not None:
        _warn_once(
            "auto-rlimit",
            f"SandboxedCodeVerifier(isolation='auto'): Docker is not usable "
            f"({status.reason}), so candidate code runs in the rlimit backend "
            f"(a local subprocess as this user). Not isolated: {_rlimit_gaps()}. "
            "For untrusted "
            f"code start Docker and pre-pull {config.image}, or pass "
            "isolation='rlimit' to choose this backend knowingly.",
        )
    elif sys.platform.startswith("linux") and not landlock_available():
        _warn_once(
            "no-landlock",
            "the rlimit sandbox backend runs without Landlock (it needs Linux "
            "5.13+ with landlock in the LSM list); not isolated: "
            f"{_rlimit_gaps()}",
        )
    return "rlimit"


async def _resolve(
    isolation: str, docker: DockerSandboxConfig | None, allow_root_rlimit: bool
) -> str:
    """:func:`resolve_isolation`, off the event loop when Docker needs probing
    (``"auto"`` reuses its first probe; ``"docker"`` re-probes a failure)."""
    cached = cached_docker_status(docker or _DEFAULT_DOCKER)
    settled = cached is not None and (cached.usable or isolation == "auto")
    if isolation == "rlimit" or settled:
        return resolve_isolation(
            isolation, docker=docker, allow_root_rlimit=allow_root_rlimit
        )
    return await asyncio.to_thread(
        functools.partial(
            resolve_isolation,
            isolation,
            docker=docker,
            allow_root_rlimit=allow_root_rlimit,
        )
    )


# --- running the sandbox -----------------------------------------------------


async def _drain(
    stream: asyncio.StreamReader | None, sink: bytearray, cap: int
) -> None:
    """Read ``stream`` to EOF, keeping at most ``cap`` bytes."""
    if stream is None:
        return
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            return
        room = cap - len(sink)
        if room > 0:
            sink.extend(chunk[:room])


def _kill_group(pid: int) -> None:
    """SIGKILL the run's whole process group (it is a session leader)."""
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _process_cpu_seconds(pid: int) -> float | None:
    """CPU seconds a live process has used (Linux ``/proc``), else ``None``."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            stat = handle.read().decode("ascii", "replace")
        # Fields after the parenthesized name start at field 3 (state);
        # utime and stime are fields 14 and 15, in clock ticks.
        fields = stat[stat.rindex(")") + 2 :].split()
        ticks = int(fields[11]) + int(fields[12])
        return ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None


async def _exchange(
    process: asyncio.subprocess.Process,
    payload: bytes,
    limits: SandboxLimits,
    out: bytearray,
    err: bytearray,
) -> float | None:
    """Feed the payload and collect output.

    Returns ``None`` when the run ended by itself, else the CPU seconds it had
    used when the wall-clock backstop killed it (``-1.0`` when unknown).
    """

    async def feed() -> None:
        assert process.stdin is not None
        try:
            process.stdin.write(payload)
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            process.stdin.close()

    def readers() -> Any:
        return asyncio.gather(
            _drain(process.stdout, out, limits.max_output_bytes),
            _drain(process.stderr, err, limits.max_stderr_bytes),
        )

    try:
        await asyncio.wait_for(
            asyncio.gather(feed(), readers(), process.wait()),
            timeout=limits.timeout_seconds,
        )
    except asyncio.TimeoutError:
        # Measure, kill, then read what the run reported before it stopped.
        used = _process_cpu_seconds(process.pid)
        _kill_group(process.pid)
        try:
            await asyncio.wait_for(readers(), timeout=2.0)
        except asyncio.TimeoutError:
            pass
        return -1.0 if used is None else used
    return None


def _records(raw: bytes, nonce: str, count: int) -> tuple[list[Any], dict[str, Any]]:
    """Per-test records (``None`` when missing or forged twice) and run facts."""
    outcomes: list[Any] = [None] * count
    seen = [0] * count
    facts: dict[str, Any] = {
        "started": False,
        "done": False,
        "limits": [],
        "load_error": None,
        "load_warning": None,
        "cpu_seconds": None,
    }
    prefix = nonce + " "
    for line in raw.decode("utf-8", "replace").split("\n"):
        if not line.startswith(prefix):
            continue
        try:
            record = strict_json_loads(line[len(prefix) :])
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        kind = record.get("kind")
        index = record.get("index")
        if kind == "result" and type(index) is int and 0 <= index < count:
            seen[index] += 1
            outcomes[index] = record
        else:
            _run_fact(kind, record, facts)
    facts["duplicates"] = sum(1 for n in seen if n > 1)
    return [o if n == 1 else None for o, n in zip(outcomes, seen, strict=True)], facts


def _run_fact(kind: Any, record: Mapping[str, Any], facts: dict[str, Any]) -> None:
    if kind == "start" and isinstance(record.get("limits"), list):
        facts["started"] = True
        facts["limits"] = [str(name) for name in record["limits"]]
    elif kind in ("load_error", "load_warning"):
        facts[kind] = str(record.get("error"))
    elif kind == "done":
        facts["done"] = True
        cpu = record.get("cpu_seconds")
        if isinstance(cpu, (int, float)) and not isinstance(cpu, bool):
            facts["cpu_seconds"] = float(cpu)


def _status(timed_out: bool, facts: Mapping[str, Any], returncode: int | None) -> str:
    if timed_out:
        return "wall_timeout"  # the backstop: the run stalled (little CPU)
    if facts["load_error"]:
        return "load_error"
    if facts["done"]:
        return "ok"
    if returncode is not None and returncode == -getattr(signal, "SIGXCPU", 0):
        return "cpu_limit"
    if returncode is not None and returncode < 0:
        return "killed"  # another signal (the hard CPU limit's SIGKILL...)
    return "exited_early" if returncode == 0 else "crashed"


def _payload(
    code: str,
    tests: Sequence[Mapping[str, Any]],
    mode: str,
    entry_point: str | None,
    limits: SandboxLimits,
    extra: Mapping[str, Any],
) -> tuple[bytes, str]:
    """The harness's input (test *inputs* only) and the run's nonce.

    ``extra`` says where a program's stdin files are (``stdin_dir``, outside
    the working directory; the harness puts one on fd 0 per test) and, per
    backend, the stdin texts to write there, the working directory or the
    Landlock rules.
    """
    nonce = secrets.token_hex(16)
    if mode == "program":
        inputs: list[dict[str, Any]] = [{} for _ in tests]
    else:
        inputs = [{"args": t["args"], "kwargs": t["kwargs"]} for t in tests]
    payload = json.dumps(
        {
            "nonce": nonce,
            "mode": mode,
            "entry_point": entry_point,
            "code": code,
            "inputs": inputs,
            "limits": limits.rlimits(),
            "max_result_chars": limits.max_result_chars,
            **extra,
        }
    ).encode()
    return payload, nonce


def _sandbox_run(
    output: HarnessOutput, nonce: str, count: int, isolation: str
) -> SandboxRun:
    """What the harness reported, as a :class:`SandboxRun`."""
    outcomes, facts = _records(output.stdout, nonce, count)
    timed_out = output.stalled_cpu is not None
    cpu = facts["cpu_seconds"]
    if output.stalled_cpu is not None:
        cpu = None if output.stalled_cpu < 0 else output.stalled_cpu
    return SandboxRun(
        outcomes=tuple(outcomes),
        status=_status(timed_out, facts, output.returncode),
        returncode=output.returncode,
        stderr=output.stderr.decode("utf-8", "replace")[-2000:],
        limits_applied=tuple(facts["limits"]),
        duplicate_records=int(facts["duplicates"]),
        load_error=facts["load_error"],
        timed_out=timed_out,
        cpu_seconds=cpu,
        load_warning=facts["load_warning"],
        isolation=isolation,
    )


@functools.lru_cache(maxsize=1)
def _harness_source() -> str:
    return _HARNESS.read_text(encoding="utf-8")


def _python_flags(site_packages: bool) -> tuple[str, ...]:
    return ("-I",) if site_packages else ("-I", "-S")


async def run_sandboxed(
    code: str,
    tests: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    entry_point: str | None = None,
    limits: SandboxLimits | None = None,
    site_packages: bool = False,
    isolation: str = "auto",
    docker: DockerSandboxConfig | None = None,
    allow_root_rlimit: bool = False,
) -> SandboxRun:
    """Run ``code`` on the inputs of ``tests`` in a fresh sandbox.

    ``isolation`` picks the backend (:func:`resolve_isolation`). Returns the
    per-test records; it never compares expected outputs (see
    :func:`score_outcomes`). Waits for one of the :func:`sandbox_concurrency`
    slots first. Raises ``RewardUnavailableError`` when no backend can run
    here, the docker image is missing, or the sandbox does not start (no
    candidate code has run then); once the harness runs, every outcome is a
    record, never an outage.
    """
    limits = limits or SandboxLimits()
    backend = await _resolve(_check_isolation(isolation), docker, allow_root_rlimit)
    async with _SLOTS.hold():
        if backend == "docker":
            return await _run_in_docker(
                code, tests, mode, entry_point, limits, site_packages, docker
            )
        return await _run_with_rlimits(
            code, tests, mode, entry_point, limits, site_packages
        )


async def _run_in_docker(
    code: str,
    tests: Sequence[Mapping[str, Any]],
    mode: str,
    entry_point: str | None,
    limits: SandboxLimits,
    site_packages: bool,
    docker: DockerSandboxConfig | None,
) -> SandboxRun:
    config = docker or _DEFAULT_DOCKER
    extra: dict[str, Any] = {"stdin_dir": STDIN_DIR, "workdir": WORKDIR}
    stdin_bytes = 0
    if mode == "program":
        extra["stdins"] = [str(test["stdin"]) for test in tests]
        stdin_bytes = sum(len(text.encode()) + 4096 for text in extra["stdins"])
    payload, nonce = _payload(code, tests, mode, entry_point, limits, extra)
    spec = DockerRunSpec(
        source=_harness_source(),
        payload=payload,
        python_flags=_python_flags(site_packages),
        memory_bytes=int(limits.memory_bytes),
        max_processes=int(limits.max_processes),
        timeout_seconds=float(limits.timeout_seconds),
        max_stdout_bytes=int(limits.max_output_bytes),
        max_stderr_bytes=int(limits.max_stderr_bytes),
        extra_tmpfs_bytes=stdin_bytes,
    )
    output = await run_harness_in_docker(config, spec)
    if not _records(output.stdout, nonce, len(tests))[1]["started"]:
        # No candidate code ran: the container did not start. An outage.
        detail = " ".join(output.stderr.decode("utf-8", "replace").split())[-500:]
        hint = ""
        if "No such image" in detail:
            hint = f"; pre-pull it with `docker pull {config.image}`"
        raise RewardUnavailableError(
            f"the docker sandbox did not start (exit status {output.returncode}): "
            f"{detail or 'no output'}{hint}"
        )
    return _sandbox_run(output, nonce, len(tests), "docker")


async def _run_with_rlimits(
    code: str,
    tests: Sequence[Mapping[str, Any]],
    mode: str,
    entry_point: str | None,
    limits: SandboxLimits,
    site_packages: bool,
) -> SandboxRun:
    with tempfile.TemporaryDirectory(prefix="stateset-sandbox-") as root:
        workdir = Path(root) / "work"
        workdir.mkdir(mode=0o700)
        stdin_dir = Path(root) / "stdin"
        if mode == "program":
            stdin_dir.mkdir(mode=0o700)
            for index, test in enumerate(tests):
                (stdin_dir / str(index)).write_text(
                    str(test["stdin"]), encoding="utf-8"
                )
        extra: dict[str, Any] = {"stdin_dir": str(stdin_dir)}
        if landlock_available():
            extra["landlock"] = {
                "read": [*_landlock_read_paths(site_packages), str(stdin_dir)],
                "write": [str(workdir), os.devnull],
            }
        payload, nonce = _payload(code, tests, mode, entry_point, limits, extra)
        output = await _run_harness(payload, workdir, limits, site_packages)
    run = _sandbox_run(output, nonce, len(tests), "rlimit")
    if (
        "landlock" in extra
        and run.limits_applied
        and "LANDLOCK" not in run.limits_applied
    ):
        _warn_once(
            "landlock-failed",
            "the kernel offers Landlock but the rlimit sandbox could not apply it: "
            "candidates can read this user's files and /proc/<pid>/environ and "
            "write outside their directory",
        )
    return run


async def _run_harness(
    payload: bytes,
    workdir: Path,
    limits: SandboxLimits,
    site_packages: bool,
) -> HarnessOutput:
    out, err = bytearray(), bytearray()
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            *_python_flags(site_packages),
            "-c",
            _harness_source(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
            env={},
            start_new_session=True,
        )
    except OSError as error:
        raise RewardUnavailableError(f"cannot start the sandbox: {error}") from error
    try:
        stalled_cpu = await _exchange(process, payload, limits, out, err)
    finally:
        _kill_group(process.pid)
        await process.wait()
    return HarnessOutput(bytes(out), bytes(err), process.returncode, stalled_cpu)


def _compiles(code: str) -> bool:
    """Whether ``code`` compiles (checked here so prose never starts a process)."""
    try:
        compile(code, "<candidate>", "exec", dont_inherit=True)
    except (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError):
        return False
    return True


class SandboxedCodeVerifier(RewardFunction):
    """Pass rate of the reply's code on hidden tests, run in a sandbox.

    A SPARSE reward: applied once to the finished conversation, it scores the
    newest assistant turn's code (``turns[-1]`` for a single reply). Tests and
    ``entry_point`` come from the context (the scenario's ``tests`` and
    ``entry_point``) or the constructor. Missing or malformed tests raise
    ``ConfigurationError``; a platform where the chosen backend cannot run
    (no usable Docker for ``isolation="docker"``, a missing image, neither
    Docker nor POSIX rlimits) raises ``RewardUnavailableError``. Code that
    fails to compile, exits early, crashes, exceeds a limit or times out fails
    the tests it did not finish; that is a score, not an outage.

    ``isolation`` (``"auto"``, ``"docker"`` or ``"rlimit"``; see the module
    docs) picks the backend, ``docker`` configures the container
    (:class:`DockerSandboxConfig`), and ``allow_root_rlimit`` opts in to the
    rlimit backend as root. ``metadata["isolation"]`` records the backend
    that ran.

    An episode whose context records that it ended without a final answer
    (``termination`` other than ``"final_answer"``, e.g. a spent tool budget)
    scores 0 without running anything.

    Only the docker backend isolates the filesystem, network and processes;
    see the module docs before running untrusted code.
    """

    score_range = (0.0, 1.0)

    def __init__(
        self,
        weight: float = 1.0,
        *,
        tests: Sequence[Mapping[str, Any]] | None = None,
        entry_point: str | None = None,
        limits: SandboxLimits | None = None,
        float_tolerance: float = 0.0,
        site_packages: bool = False,
        isolation: str = "auto",
        docker: DockerSandboxConfig | None = None,
        allow_root_rlimit: bool = False,
        name: str = "SandboxedCodeVerifier",
    ) -> None:
        super().__init__(weight, RewardType.SPARSE, name)
        self.tests = list(tests) if tests is not None else None
        self.entry_point = entry_point
        self.limits = limits or SandboxLimits()
        self.float_tolerance = float(float_tolerance)
        self.site_packages = bool(site_packages)
        self.isolation = _check_isolation(isolation)
        if docker is not None and not isinstance(docker, DockerSandboxConfig):
            raise ConfigurationError("docker must be a DockerSandboxConfig")
        self.docker = docker
        self.allow_root_rlimit = bool(allow_root_rlimit)
        if self.isolation == "rlimit":
            _check_root(self.allow_root_rlimit)
        if self.tests is not None:
            normalize_tests(self.tests)

    def _test_spec(
        self, context: Mapping[str, Any] | None
    ) -> tuple[str, list[Any], Any]:
        context = context or {}
        raw = context.get("tests", self.tests)
        if raw is None:
            raise ConfigurationError(
                "SandboxedCodeVerifier needs tests (scenario 'tests' or tests=...)"
            )
        mode, tests = normalize_tests(raw)
        entry_point = context.get("entry_point", self.entry_point)
        if mode == "function" and not (
            isinstance(entry_point, str) and entry_point.isidentifier()
        ):
            raise ConfigurationError("function tests need an identifier entry_point")
        return mode, tests, entry_point

    async def compute_reward(
        self, turns: list[Any], context: dict[str, Any] | None = None
    ) -> RewardResult:
        mode, tests, entry_point = self._test_spec(context)
        termination = unanswered_termination(context)
        if termination is not None:
            return RewardResult(
                score=0.0,
                breakdown={"pass_rate": 0.0},
                metadata={
                    "passed": 0,
                    "total": len(tests),
                    "status": "no_final_answer",
                    "termination": termination,
                },
            )
        replies = assistant_turns_in_scope(turns, context)
        reply = replies[-1].content if replies else ""
        code = extract_code(reply if isinstance(reply, str) else "")
        status = "no_code" if not code.strip() else None
        if status is None and len(code) > self.limits.max_code_chars:
            status = "too_long"
        if status is None and not _compiles(code):
            status = "syntax_error"
        passed, details = 0, {}
        if status is None:
            run = await run_sandboxed(
                code,
                tests,
                mode=mode,
                entry_point=entry_point,
                limits=self.limits,
                site_packages=self.site_packages,
                isolation=self.isolation,
                docker=self.docker,
                allow_root_rlimit=self.allow_root_rlimit,
            )
            passed = score_outcomes(run, tests, self.float_tolerance)
            status = run.status
            details = {
                "isolation": run.isolation,
                "limits_applied": list(run.limits_applied),
                "duplicate_records": run.duplicate_records,
                "load_error": run.load_error,
                "load_warning": run.load_warning,
                "cpu_seconds": run.cpu_seconds,
            }
        score = passed / len(tests)
        return RewardResult(
            score=score,
            breakdown={"pass_rate": score},
            metadata={
                "passed": passed,
                "total": len(tests),
                "status": status,
                **details,
            },
        )
