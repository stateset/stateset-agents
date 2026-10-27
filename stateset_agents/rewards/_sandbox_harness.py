"""Test harness that runs inside the code sandbox (stdlib only; never imported
by the package at runtime).

:mod:`stateset_agents.rewards.code_verifier` starts it as
``python -I -S -c <this source>`` in an empty environment, either as its own
child (the ``rlimit`` backend, in a fresh temp directory) or in a fresh
container (the ``docker`` backend, :mod:`._sandbox_docker`), and writes one
JSON payload to its stdin: a per-run random ``nonce``, resource ``limits``,
the candidate ``code``, the ``mode`` (``"function"`` or ``"program"``), the
``entry_point`` and the test *inputs* only (for programs, the directory
holding one stdin file per test, outside the working directory, or the
stdin texts themselves for the harness to write there). Expected outputs
never enter this process; the parent compares results.

The harness:

1. reads the whole payload, creates the working directory and stdin files
   when the payload names them (the container), then applies the rlimits to
   itself (soft = hard, so the candidate cannot raise them) and, when the
   payload asks for it (the rlimit backend on Linux), Landlock filesystem
   rules, all before compiling any candidate code;
2. keeps a private duplicate of stdout for its records and points fd 1 at
   ``/dev/null``, so stray candidate output never mixes with them;
3. for each test, calls the entry point (function mode) or runs the program
   with the test's stdin (program mode), and writes one record line
   ``<nonce> <json>`` per test as soon as it finishes: the JSON-encoded return
   value (strict JSON: an object with a custom ``__eq__`` cannot pass), or the
   captured stdout, or the exception type;
4. writes a final ``done`` record (with the CPU seconds the run used) and
   leaves with ``os._exit`` (no atexit or finalizer code of the candidate runs
   after it).

Landlock (Linux 5.13+, unprivileged, no namespaces) confines the candidate's
filesystem: it may read the interpreter's standard library and the system
library directories, read its stdin files and use its working directory, and
nothing else (no ``/proc``, home directory, dataset or credential files, no
writes elsewhere). A Landlock-confined process also cannot ptrace or read
the ``/proc`` entries (``environ``, ``mem``) of processes outside its
domain, such as the trainer. It does not restrict the network or signals.

In a container, ``--supervise <seconds>`` makes this process (PID 1, which
the kernel shields from default-action signals such as ``SIGXCPU``) a
supervisor: it forks the harness, waits at most ``<seconds>`` (then kills
it), and exits with ``128 + signal`` or the harness's exit code, so the
parent sees a CPU-limit kill exactly as it does for a local subprocess.

Ordinary scripts run as they would under ``python``: a program's stdin is a
real file on fd 0 (``open(0).read()``, ``os.read(0, n)`` and
``sys.stdin.fileno()`` work), and ``exit()``/``quit()`` exist although
``-S`` skips the ``site`` module that defines them. In function mode, code
after the entry point that fails at load (an "example usage" block reading
``input()`` or naming an undefined variable) does not hide the function: the
tests still call it, and the error is reported as a ``load_warning``.

A run that exits early (``sys.exit``/``os._exit`` at load before the entry
point exists, or in a test), crashes, or is killed for a limit leaves tests
without records, and the parent scores those tests 0.
"""

from __future__ import annotations

import builtins
import io
import json
import os
import signal
import struct
import sys
from typing import Any, NoReturn

#: Bound before any candidate code runs, so patching ``json.dumps`` does
#: not change how results are encoded.
_DUMPS = json.dumps
_LIMITS = (
    "RLIMIT_CPU",
    "RLIMIT_AS",
    "RLIMIT_FSIZE",
    "RLIMIT_NPROC",
    "RLIMIT_NOFILE",
    "RLIMIT_CORE",
)

# Landlock (include/uapi/linux/landlock.h); the syscall numbers are shared by
# the architectures the parent enables it on.
_SYS_LANDLOCK_CREATE_RULESET = 444
_SYS_LANDLOCK_ADD_RULE = 445
_SYS_LANDLOCK_RESTRICT_SELF = 446
_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38
_FS_EXECUTE = 1 << 0
_FS_WRITE_FILE = 1 << 1
_FS_READ_FILE = 1 << 2
_FS_READ_DIR = 1 << 3
_FS_TRUNCATE = 1 << 14
_FS_IOCTL_DEV = 1 << 15
#: Rights that apply to a file (the others only to directories).
_FS_FILE_RIGHTS = (
    _FS_EXECUTE | _FS_WRITE_FILE | _FS_READ_FILE | _FS_TRUNCATE | _FS_IOCTL_DEV
)


def supervise(seconds: float) -> None:  # pragma: no cover - container PID 1
    """Fork; the child returns and runs the harness, this process waits for
    it (killing it after ``seconds``) and exits as it did: ``128 + signal``
    when killed (as a shell reports it), else its exit code (below 128)."""
    import select

    pid = os.fork()
    if pid == 0:
        return
    try:
        pidfd = os.pidfd_open(pid)
    except (AttributeError, OSError):
        pidfd = None
    if pidfd is not None and not select.select([pidfd], [], [], seconds)[0]:
        os.kill(pid, signal.SIGKILL)
    _, status = os.waitpid(pid, 0)
    if os.WIFSIGNALED(status):
        os._exit(128 + os.WTERMSIG(status))
    code = os.WEXITSTATUS(status)
    os._exit(code if code < 128 else 1)


def prepare_files(payload: dict[str, Any]) -> None:  # pragma: no cover - child
    """Create the working directory and the stdin files the payload carries."""
    workdir = payload.get("workdir")
    if isinstance(workdir, str):
        os.makedirs(workdir, mode=0o700, exist_ok=True)
        os.chdir(workdir)
    stdins = payload.get("stdins")
    if isinstance(stdins, list):
        stdin_dir = str(payload["stdin_dir"])
        os.makedirs(stdin_dir, mode=0o700, exist_ok=True)
        for index, text in enumerate(stdins):
            with open(os.path.join(stdin_dir, str(index)), "w", encoding="utf-8") as f:
                f.write(str(text))


def _landlock_rights(abi: int) -> int:
    """Every filesystem right Landlock ``abi`` knows (so all are denied
    unless a rule allows them)."""
    if abi >= 5:
        return (1 << 16) - 1  # ... IOCTL_DEV
    if abi >= 3:
        return (1 << 15) - 1  # ... TRUNCATE
    if abi >= 2:
        return (1 << 14) - 1  # ... REFER
    return (1 << 13) - 1


def apply_landlock(rules: dict[str, Any]) -> list[str]:  # pragma: no cover - child
    """Allow reading beneath ``rules["read"]`` and everything but executing
    beneath ``rules["write"]``; deny the rest of the filesystem. Returns
    ``["LANDLOCK"]`` when applied, ``[]`` when the kernel cannot (nothing is
    restricted then, and the parent reports it)."""
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
    except (ImportError, OSError):
        return []
    syscall = libc.syscall
    syscall.restype = ctypes.c_long
    word = ctypes.c_long
    abi = syscall(
        word(_SYS_LANDLOCK_CREATE_RULESET),
        None,
        word(0),
        word(_LANDLOCK_CREATE_RULESET_VERSION),
    )
    if abi < 1:
        return []
    handled = _landlock_rights(abi)
    attr = ctypes.c_uint64(handled)  # struct landlock_ruleset_attr, ABI 1 part
    ruleset = syscall(
        word(_SYS_LANDLOCK_CREATE_RULESET),
        ctypes.byref(attr),
        word(ctypes.sizeof(attr)),
        word(0),
    )
    if ruleset < 0:
        return []
    try:
        grants = [(path, _FS_READ_FILE | _FS_READ_DIR) for path in rules["read"]]
        grants += [(path, handled & ~_FS_EXECUTE) for path in rules["write"]]
        for path, access in grants:
            if not _landlock_allow(syscall, word, int(ruleset), str(path), access):
                return []
        if libc.prctl(word(_PR_SET_NO_NEW_PRIVS), word(1), word(0), word(0), word(0)):
            return []
        if syscall(word(_SYS_LANDLOCK_RESTRICT_SELF), word(ruleset), word(0)) != 0:
            return []
    finally:
        os.close(int(ruleset))
    return ["LANDLOCK"]


def _landlock_allow(
    syscall: Any, word: Any, ruleset: int, path: str, access: int
) -> bool:  # pragma: no cover - child
    """Add one path-beneath rule; a path that does not exist is skipped."""
    import ctypes
    import stat

    try:
        fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
    except OSError:
        return True
    try:
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            access &= _FS_FILE_RIGHTS
        # struct landlock_path_beneath_attr: packed {__u64 allowed; __s32 fd}.
        rule = ctypes.create_string_buffer(struct.pack("=Qi", access, fd), 12)
        added = syscall(
            word(_SYS_LANDLOCK_ADD_RULE),
            word(ruleset),
            word(_LANDLOCK_RULE_PATH_BENEATH),
            rule,
            word(0),
        )
    finally:
        os.close(fd)
    return bool(added == 0)


def apply_limits(limits: dict[str, Any]) -> list[str]:  # pragma: no cover - child
    """Set each ``RLIMIT_*`` the platform supports to ``(soft, hard)``."""
    import resource

    applied = []
    for name in _LIMITS:
        if name not in limits or not hasattr(resource, name):
            continue
        soft, hard = limits[name]
        try:
            resource.setrlimit(getattr(resource, name), (int(soft), int(hard)))
        except (ValueError, OSError):
            continue
        applied.append(name)
    return applied


class _Records:  # pragma: no cover - child
    """Writes nonce-framed JSON records to a private copy of stdout."""

    def __init__(self, nonce: str) -> None:
        self.nonce = nonce
        self.fd = os.dup(1)
        self._write = os.write
        self._dumps = _DUMPS
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.close(devnull)

    def emit(self, record: dict[str, Any]) -> None:
        line = f"{self.nonce} {self._dumps(record)}\n".encode()
        view = memoryview(line)
        while view:
            written = self._write(self.fd, view)
            view = view[written:]


def _exit(code: Any = None) -> NoReturn:  # pragma: no cover - child
    """``exit``/``quit`` as the ``site`` module defines them (``-S`` skips it)."""
    raise SystemExit(code)


def _cpu_seconds() -> float:  # pragma: no cover - child
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


def _encode(value: Any, limit: int) -> dict[str, Any]:  # pragma: no cover - child
    try:
        encoded = _DUMPS(value, allow_nan=False)
    except (TypeError, ValueError, RecursionError, OverflowError):
        return {"error": "UnserializableResult"}
    if len(encoded) > limit:
        return {"error": "ResultTooLarge"}
    return {"value": encoded}


def run_function_tests(
    code: Any,
    entry_point: str,
    inputs: list[dict[str, Any]],
    records: _Records,
    limit: int,
) -> None:  # pragma: no cover - child
    namespace: dict[str, Any] = {"__name__": "__candidate__"}
    try:
        # Running the candidate is this child's purpose: it executes only in
        # the rlimited ``python -I -S`` subprocess code_verifier starts.
        exec(code, namespace)  # nosec: B102
    except BaseException as error:  # noqa: BLE001 - SystemExit included
        if not callable(namespace.get(entry_point)):
            records.emit({"kind": "load_error", "error": type(error).__name__})
            return
        # An example block after the solution failed: the function stands.
        records.emit({"kind": "load_warning", "error": type(error).__name__})
    function = namespace.get(entry_point)
    if not callable(function):
        records.emit({"kind": "load_error", "error": "NameError"})
        return
    for index, test in enumerate(inputs):
        try:
            value = function(*test.get("args", []), **test.get("kwargs", {}))
            outcome = _encode(value, limit)
        except BaseException as error:  # noqa: BLE001 - SystemExit included
            outcome = {"error": type(error).__name__}
        records.emit({"kind": "result", "index": index, **outcome})


def _attach_stdin(path: str) -> Any:  # pragma: no cover - child
    """Make the file at ``path`` the process's stdin (fd 0 and ``sys.stdin``)."""
    fd = os.open(path, os.O_RDONLY)
    if fd != 0:
        os.dup2(fd, 0)
        os.close(fd)
    return open(0, encoding="utf-8", closefd=False)


def run_program_tests(
    code: Any, stdin_dir: str, count: int, records: _Records, limit: int
) -> None:  # pragma: no cover - child
    for index in range(count):
        raw = io.BytesIO()
        sink = io.TextIOWrapper(raw, "utf-8", write_through=True)
        status = "ok"
        try:
            sys.stdin = _attach_stdin(os.path.join(stdin_dir, str(index)))
            sys.stdout = sink
            exec(code, {"__name__": "__main__"})  # nosec: B102 - sandbox child
        except SystemExit as stop:
            if stop.code not in (None, 0):
                status = "exit_nonzero"
        except BaseException as error:  # noqa: BLE001 - KeyboardInterrupt too
            status = type(error).__name__
        try:
            sink.flush()
            output = raw.getvalue().decode("utf-8", "replace")
        except (ValueError, AttributeError):
            output, status = "", "OutputUnavailable"
        record: dict[str, Any] = {"kind": "result", "index": index, "status": status}
        if len(output) > limit:
            record["error"] = "ResultTooLarge"
        else:
            record["stdout"] = output
        records.emit(record)


def main() -> None:  # pragma: no cover - child
    if len(sys.argv) == 3 and sys.argv[1] == "--supervise":
        supervise(float(sys.argv[2]))  # returns in the child only
    payload = json.loads(sys.stdin.buffer.read())
    prepare_files(payload)
    applied = apply_limits(payload["limits"])
    if hasattr(signal, "SIGXFSZ"):
        # Writing past RLIMIT_FSIZE raises OSError instead of killing us.
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    records = _Records(str(payload["nonce"]))
    if isinstance(payload.get("landlock"), dict):
        applied += apply_landlock(payload["landlock"])
    records.emit({"kind": "start", "limits": applied})
    for name in ("exit", "quit"):  # what ``site`` would define
        setattr(builtins, name, _exit)
    limit = int(payload["max_result_chars"])
    try:
        code = compile(payload["code"], "<candidate>", "exec", dont_inherit=True)
    except (
        SyntaxError,
        ValueError,
        RecursionError,
        MemoryError,
        OverflowError,
    ) as error:
        records.emit({"kind": "load_error", "error": type(error).__name__})
        code = None
    if code is not None and payload["mode"] == "function":
        run_function_tests(
            code, payload["entry_point"], payload["inputs"], records, limit
        )
    elif code is not None:
        run_program_tests(
            code, str(payload["stdin_dir"]), len(payload["inputs"]), records, limit
        )
    records.emit({"kind": "done", "cpu_seconds": _cpu_seconds()})
    os._exit(0)


if __name__ == "__main__":  # pragma: no cover - child
    main()
