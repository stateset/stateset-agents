"""Test harness that runs inside the code sandbox (stdlib only; never imported
by the package at runtime).

:mod:`stateset_agents.rewards.code_verifier` starts it as
``python -I -S -c <this source>`` in an empty environment and a fresh temp
directory, and writes one JSON payload to its stdin: a per-run random
``nonce``, resource ``limits``, the candidate ``code``, the ``mode``
(``"function"`` or ``"program"``), the ``entry_point`` and the test
*inputs* only. Expected outputs never enter this process; the parent
compares results.

The harness:

1. reads the whole payload, then applies the rlimits to itself (soft = hard,
   so the candidate cannot raise them) before compiling any candidate code;
2. keeps a private duplicate of stdout for its records and points fd 1 at
   ``/dev/null``, so stray candidate output never mixes with them;
3. for each test, calls the entry point (function mode) or runs the program
   with the test's stdin (program mode), and writes one record line
   ``<nonce> <json>`` per test as soon as it finishes: the JSON-encoded return
   value (strict JSON: an object with a custom ``__eq__`` cannot pass), or the
   captured stdout, or the exception type;
4. writes a final ``done`` record and leaves with ``os._exit`` (no atexit or
   finalizer code of the candidate runs after it).

A run that exits early (``sys.exit``/``os._exit`` at load or in a test),
crashes, or is killed for a limit leaves tests without records, and the
parent scores those tests 0.
"""

from __future__ import annotations

import io
import json
import os
import signal
import sys
from typing import Any

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
        function = namespace.get(entry_point)
        if not callable(function):
            raise NameError(f"no callable named {entry_point}")
    except BaseException as error:  # noqa: BLE001 - SystemExit included
        records.emit({"kind": "load_error", "error": type(error).__name__})
        return
    for index, test in enumerate(inputs):
        try:
            value = function(*test.get("args", []), **test.get("kwargs", {}))
            outcome = _encode(value, limit)
        except BaseException as error:  # noqa: BLE001 - SystemExit included
            outcome = {"error": type(error).__name__}
        records.emit({"kind": "result", "index": index, **outcome})


def run_program_tests(
    code: Any, inputs: list[dict[str, Any]], records: _Records, limit: int
) -> None:  # pragma: no cover - child
    for index, test in enumerate(inputs):
        stdin = io.TextIOWrapper(io.BytesIO(str(test["stdin"]).encode()), "utf-8")
        raw = io.BytesIO()
        sink = io.TextIOWrapper(raw, "utf-8", write_through=True)
        sys.stdin, sys.stdout = stdin, sink
        status = "ok"
        try:
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
    payload = json.loads(sys.stdin.buffer.read())
    applied = apply_limits(payload["limits"])
    if hasattr(signal, "SIGXFSZ"):
        # Writing past RLIMIT_FSIZE raises OSError instead of killing us.
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    records = _Records(str(payload["nonce"]))
    records.emit({"kind": "start", "limits": applied})
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
        run_program_tests(code, payload["inputs"], records, limit)
    records.emit({"kind": "done"})
    os._exit(0)


if __name__ == "__main__":  # pragma: no cover - child
    main()
