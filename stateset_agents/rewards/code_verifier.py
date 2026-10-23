"""Sandboxed code verifier: run candidate code against hidden tests.

:class:`SandboxedCodeVerifier` extracts the code from the reply (the last
fenced block, else the whole reply), runs it against test cases in a fresh
subprocess and scores the pass rate in ``[0, 1]``. Tests are either function
tests ``{"args": [...], "kwargs": {...}, "expected": value}`` against an
``entry_point``, or program tests ``{"stdin": "...", "expected_stdout": "..."}``.
They come from the reward context (``tests``/``entry_point``, i.e. the
scenario) or the constructor, and should be *hidden*: different from any
examples shown in the prompt.

How a run is isolated (POSIX only; elsewhere scoring raises
``RewardUnavailableError``):

* ``python -I -S`` (isolated mode, no site-packages) with an empty
  environment, a fresh temporary directory as the working directory, and its
  own session, so the whole process group is killed afterwards;
* a wall-clock timeout, plus rlimits the harness applies to itself before any
  candidate code runs: CPU seconds, address space, file size, process count
  (``RLIMIT_NPROC`` 0: no forks or threads), open files, no core dumps;
* capped reads of stdout and stderr (the rest is drained and discarded).

How a verdict resists the obvious exploits:

* expected outputs never enter the sandbox: the harness receives test
  *inputs* only and reports each return value as strict JSON (or a program's
  captured stdout); this process compares them. Printing the expected output
  without computing it needs the hidden tests, reading a tests file finds
  none, and an object whose ``__eq__`` always returns True cannot be encoded;
* the harness frames its records with a per-run random nonce written to a
  private copy of stdout, and points fd 1 at ``/dev/null``: stray output is
  never parsed as a result, and a test with two records (a forged one) fails;
* only tests with a record count: ``sys.exit``/``os._exit`` before or during
  the tests, a crash, a limit kill or a timeout (an infinite loop) leaves the
  remaining tests failed. A timeout scores 0 for those tests; it is not an
  outage.

**This is resource isolation, not a security boundary.** The candidate runs
as the trainer's user with normal filesystem and network access; it can read
anything that user can and, as root, ignores ``RLIMIT_NPROC``. Run untrusted
or adversarial code in a container or VM sandbox (gVisor, Firecracker, a
locked-down Docker container without network) and point this verifier at
trusted policy output only, or run the trainer itself inside such a sandbox.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import secrets
import signal
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stateset_agents.core.environments.json_action import strict_json_loads
from stateset_agents.core.errors import ConfigurationError
from stateset_agents.core.reward_base import RewardFunction, RewardResult, RewardType
from stateset_agents.core.reward_context import (
    RewardUnavailableError,
    assistant_turns_in_scope,
)
from stateset_agents.core.tool_calls import strip_tool_calls

__all__ = [
    "SandboxLimits",
    "SandboxRun",
    "SandboxedCodeVerifier",
    "extract_code",
    "normalize_tests",
    "run_sandboxed",
    "sandbox_available",
    "score_outcomes",
]

_HARNESS = Path(__file__).with_name("_sandbox_harness.py")
_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.S)
_CODE_LANGUAGES = {"", "python", "py", "python3"}


@dataclass(frozen=True)
class SandboxLimits:
    """Per-run limits. ``timeout_seconds`` is wall clock for the whole run."""

    timeout_seconds: float = 10.0
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


def sandbox_available() -> bool:
    """Whether this platform can run the sandbox (POSIX with ``resource``)."""
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


async def _exchange(
    process: asyncio.subprocess.Process,
    payload: bytes,
    limits: SandboxLimits,
    out: bytearray,
    err: bytearray,
) -> bool:
    """Feed the payload and collect output; True when the run timed out."""

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
        # Kill, then read what the run reported before it was stopped.
        _kill_group(process.pid)
        try:
            await asyncio.wait_for(readers(), timeout=2.0)
        except asyncio.TimeoutError:
            pass
        return True
    return False


def _records(raw: bytes, nonce: str, count: int) -> tuple[list[Any], dict[str, Any]]:
    """Per-test records (``None`` when missing or forged twice) and run facts."""
    outcomes: list[Any] = [None] * count
    seen = [0] * count
    facts: dict[str, Any] = {"done": False, "limits": [], "load_error": None}
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
        elif kind == "start" and isinstance(record.get("limits"), list):
            facts["limits"] = [str(name) for name in record["limits"]]
        elif kind == "load_error":
            facts["load_error"] = str(record.get("error"))
        elif kind == "done":
            facts["done"] = True
    facts["duplicates"] = sum(1 for n in seen if n > 1)
    return [o if n == 1 else None for o, n in zip(outcomes, seen, strict=True)], facts


def _status(timed_out: bool, facts: Mapping[str, Any], returncode: int | None) -> str:
    if timed_out:
        return "timeout"
    if facts["load_error"]:
        return "load_error"
    if facts["done"]:
        return "ok"
    if returncode is not None and returncode < 0:
        return "killed"  # a limit's signal (SIGXCPU, SIGKILL...)
    return "exited_early" if returncode == 0 else "crashed"


async def run_sandboxed(
    code: str,
    tests: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    entry_point: str | None = None,
    limits: SandboxLimits | None = None,
    site_packages: bool = False,
) -> SandboxRun:
    """Run ``code`` on the inputs of ``tests`` in a fresh sandboxed process.

    Returns the per-test records; it never compares expected outputs (see
    :func:`score_outcomes`). Raises ``RewardUnavailableError`` when the
    platform cannot sandbox (not POSIX) or the interpreter cannot start.
    """
    if not sandbox_available():
        raise RewardUnavailableError(
            "the sandboxed code verifier needs a POSIX platform with the "
            "resource module (rlimits); run it on Linux or macOS"
        )
    limits = limits or SandboxLimits()
    nonce = secrets.token_hex(16)
    inputs = [
        (
            {"stdin": t["stdin"]}
            if mode == "program"
            else {"args": t["args"], "kwargs": t["kwargs"]}
        )
        for t in tests
    ]
    payload = json.dumps(
        {
            "nonce": nonce,
            "mode": mode,
            "entry_point": entry_point,
            "code": code,
            "inputs": inputs,
            "limits": limits.rlimits(),
            "max_result_chars": limits.max_result_chars,
        }
    ).encode()
    flags = ["-I"] if site_packages else ["-I", "-S"]
    out, err = bytearray(), bytearray()
    with tempfile.TemporaryDirectory(prefix="stateset-sandbox-") as workdir:
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                *flags,
                "-c",
                _HARNESS.read_text(encoding="utf-8"),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
                env={},
                start_new_session=True,
            )
        except OSError as error:
            raise RewardUnavailableError(
                f"cannot start the sandbox: {error}"
            ) from error
        try:
            timed_out = await _exchange(process, payload, limits, out, err)
        finally:
            _kill_group(process.pid)
            await process.wait()
    outcomes, facts = _records(bytes(out), nonce, len(tests))
    return SandboxRun(
        outcomes=tuple(outcomes),
        status=_status(timed_out, facts, process.returncode),
        returncode=process.returncode,
        stderr=err.decode("utf-8", "replace")[-2000:],
        limits_applied=tuple(facts["limits"]),
        duplicate_records=int(facts["duplicates"]),
        load_error=facts["load_error"],
        timed_out=timed_out,
    )


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
    ``ConfigurationError``; a platform without the sandbox raises
    ``RewardUnavailableError``. Code that fails to compile, exits early,
    crashes, exceeds a limit or times out fails the tests it did not finish;
    that is a score, not an outage.

    Resource isolation only: see the module docs before running untrusted
    code, and prefer a container sandbox for it.
    """

    def __init__(
        self,
        weight: float = 1.0,
        *,
        tests: Sequence[Mapping[str, Any]] | None = None,
        entry_point: str | None = None,
        limits: SandboxLimits | None = None,
        float_tolerance: float = 0.0,
        site_packages: bool = False,
        name: str = "SandboxedCodeVerifier",
    ) -> None:
        super().__init__(weight, RewardType.SPARSE, name)
        self.tests = list(tests) if tests is not None else None
        self.entry_point = entry_point
        self.limits = limits or SandboxLimits()
        self.float_tolerance = float(float_tolerance)
        self.site_packages = bool(site_packages)
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
            )
            passed = score_outcomes(run, tests, self.float_tolerance)
            status = run.status
            details = {
                "limits_applied": list(run.limits_applied),
                "duplicate_records": run.duplicate_records,
                "load_error": run.load_error,
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
