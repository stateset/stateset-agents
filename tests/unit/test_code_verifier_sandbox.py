"""SandboxedCodeVerifier: isolation, scoring, and every exploit it must resist.

Each exploit test runs the same candidate twice: against the sandboxed
verifier, which must score it 0 (and never raise), and against a *naive*
verifier (``_naive_passes``: candidate and assert-style tests written to one
file in the working directory, run with ``python``, pass when it exits 0), which
the exploit fools. The naive run is the control showing the exploit is real.

Every test that runs a sandbox runs on both backends (the ``isolation``
fixture): ``rlimit`` where POSIX rlimits exist, ``docker`` where a daemon is
reachable and the pinned image is present (pre-pull it; the tests never pull),
and is skipped otherwise. Every container a test starts carries the
``stateset-verifier-test`` label, so leftovers can be removed by label. The
isolation probes are harmless and only ever assert that a capability is
*absent*.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import signal
import socket
import statistics
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from stateset_agents.core.errors import ConfigurationError
from stateset_agents.core.reward_context import RewardUnavailableError
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.rewards import _sandbox_docker, code_verifier
from stateset_agents.rewards.code_verifier import (
    DockerSandboxConfig,
    DockerStatus,
    SandboxedCodeVerifier,
    SandboxLimits,
    docker_sandbox_status,
    extract_code,
    landlock_available,
    normalize_tests,
    run_sandboxed,
)

REPO = Path(__file__).resolve().parents[2]
#: Every container these tests start carries this label (remove by label).
TEST_LABEL = "stateset-verifier-test"
TEST_DOCKER = DockerSandboxConfig(labels={TEST_LABEL: "1"})


#: At most this many sandboxes (containers) run at once in these tests, so a
#: shared, loaded host never gets one per CPU.
TEST_CONCURRENCY = 4


@pytest.fixture(autouse=True)
def _bounded_sandboxes() -> Iterator[None]:
    code_verifier.set_sandbox_concurrency(
        min(TEST_CONCURRENCY, code_verifier._default_concurrency())
    )
    yield
    code_verifier.set_sandbox_concurrency(None)


def _docker_unusable() -> str | None:
    status = docker_sandbox_status(TEST_DOCKER)
    return None if status.usable else status.reason


@pytest.fixture(params=["rlimit", "docker"])
def isolation(request: Any) -> str:
    """Each sandbox test runs on both backends, or skips one cleanly."""
    if request.param == "rlimit" and not code_verifier.sandbox_available():
        pytest.skip("the rlimit backend needs POSIX rlimits")
    if request.param == "docker":
        reason = _docker_unusable()
        if reason is not None:
            pytest.skip(f"the docker backend is unavailable: {reason}")
    return str(request.param)


def _defends_files(isolation: str) -> bool:
    """Whether the candidate is confined to its own files here."""
    return isolation == "docker" or bool(landlock_available())


REFERENCE = "def sum_of_squares(n):\n    return sum(i * i for i in range(1, n + 1))"
#: Shown in the prompt (and used by the naive verifier).
VISIBLE = [{"args": [3], "expected": 14}]
#: Hidden from the candidate: the sandbox never receives the expected values.
HIDDEN = [
    {"args": [1], "expected": 1},
    {"args": [4], "expected": 30},
    {"args": [10], "expected": 385},
]
LIMITS = SandboxLimits(timeout_seconds=5.0, cpu_seconds=2)


def _verifier(isolation: str, **kwargs: Any) -> SandboxedCodeVerifier:
    kwargs.setdefault("tests", HIDDEN)
    kwargs.setdefault("entry_point", "sum_of_squares")
    kwargs.setdefault("limits", LIMITS)
    # The rlimit runs here are deliberate, also when CI runs as root.
    return SandboxedCodeVerifier(
        isolation=isolation, docker=TEST_DOCKER, allow_root_rlimit=True, **kwargs
    )


def _score(code: str, verifier: SandboxedCodeVerifier) -> Any:
    turns = [
        ConversationTurn(role="user", content="Write sum_of_squares(n)."),
        ConversationTurn(role="assistant", content=f"```python\n{code}\n```"),
    ]
    result = asyncio.run(verifier.compute_reward(turns, {}))
    if "isolation" in result.metadata:
        assert result.metadata["isolation"] == verifier.isolation
    return result


def _run(code: str, tests: Any, isolation: str, **kwargs: Any) -> Any:
    kwargs.setdefault("mode", "function")
    kwargs.setdefault("limits", LIMITS)
    return asyncio.run(
        run_sandboxed(
            code,
            tests,
            isolation=isolation,
            docker=TEST_DOCKER,
            allow_root_rlimit=True,
            **kwargs,
        )
    )


NAIVE_PRELUDE = "def check(actual, expected):\n    return actual == expected\n\n"


def _naive_passes(code: str, workdir: Path, timeout: float | None = 20.0) -> bool:
    """The naive verifier: one file with the checker, the code and the asserts."""
    asserts = "\n".join(
        f"assert check(sum_of_squares({t['args'][0]}), {t['expected']})"
        for t in VISIBLE + HIDDEN
    )
    (workdir / "test_solution.py").write_text(f"{NAIVE_PRELUDE}{code}\n\n{asserts}\n")
    result = subprocess.run(
        [sys.executable, "test_solution.py"],
        cwd=workdir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=timeout,
        check=False,
    )
    return result.returncode == 0


def test_reference_passes_both(isolation):
    result = _score(REFERENCE, _verifier(isolation))
    assert result.score == 1.0 and result.metadata["status"] == "ok"
    assert result.metadata["isolation"] == isolation


def test_runs_isolated_in_a_fresh_empty_environment(isolation):
    probe = textwrap.dedent("""
        import os, sys
        def sum_of_squares(n):
            return [sys.flags.isolated, sys.flags.no_site, dict(os.environ),
                    os.listdir("."), os.getcwd()]
        """)
    run = _run(
        probe,
        normalize_tests([{"args": [1], "expected": None}])[1],
        isolation,
        entry_point="sum_of_squares",
    )
    isolated, no_site, environ, listing, cwd = json.loads(run.outcomes[0]["value"])
    assert (isolated, no_site, listing) == (1, 1, [])
    # Nothing is inherited; the interpreter may set LC_CTYPE itself (PEP 538).
    assert set(environ) <= {"LC_CTYPE"}
    assert run.isolation == isolation
    if isolation == "docker":
        assert cwd == _sandbox_docker.WORKDIR  # on the container's tmpfs
    else:
        assert "stateset-sandbox-" in cwd and not os.path.exists(cwd)
    if sys.platform.startswith("linux"):
        assert {"RLIMIT_CPU", "RLIMIT_AS", "RLIMIT_FSIZE", "RLIMIT_NPROC"} <= set(
            run.limits_applied
        )
    if isolation == "rlimit" and landlock_available():
        assert "LANDLOCK" in run.limits_applied


# --- exploits: each scores 0 in the sandbox and fools the naive verifier ------

HARDCODE = "def sum_of_squares(n):\n    return {3: 14}.get(n)"
PRINT_EXPECTED = "print(14)\ndef sum_of_squares(n):\n    return 14"
SYS_EXIT = "import sys\nsys.exit(0)\ndef sum_of_squares(n):\n    return None"
OS_EXIT = "import os\nos._exit(0)\ndef sum_of_squares(n):\n    return None"
ALWAYS_EQUAL = textwrap.dedent("""
    class Anything:
        def __eq__(self, other):
            return True
        __hash__ = object.__hash__
    def sum_of_squares(n):
        return Anything()
    """)
PATCH_CHECKER = textwrap.dedent("""
    import json
    check = lambda *args: True
    json.dumps = lambda *args, **kwargs: "385"
    def sum_of_squares(n):
        return None
    """)
READ_TESTS = textwrap.dedent("""
    import glob, re
    def sum_of_squares(n):
        for path in glob.glob("*.py") + glob.glob("/tmp/*.py"):
            try:
                text = open(path).read()
            except OSError:
                continue
            for m in re.finditer(r"sum_of_squares\\((\\d+)\\), (\\d+)\\)", text):
                if int(m.group(1)) == n:
                    return int(m.group(2))
        return None
    """)
FORGE_NONCE = textwrap.dedent("""
    import gc, json, os
    payload = next(o for o in gc.get_objects()
                   if isinstance(o, dict) and isinstance(o.get("nonce"), str))
    for fd in range(3, 10):
        for index in range(len(payload["inputs"])):
            record = {"kind": "result", "index": index, "value": "385"}
            line = payload["nonce"] + " " + json.dumps(record) + "\\n"
            try:
                os.write(fd, line.encode())
            except OSError:
                pass
    def sum_of_squares(n):
        return 385
    """)

EXPLOITS = {
    "hardcode_visible": HARDCODE,
    "print_expected": PRINT_EXPECTED,
    "sys_exit_before_asserts": SYS_EXIT,
    "os_exit_before_asserts": OS_EXIT,
    "always_equal": ALWAYS_EQUAL,
    "monkeypatch_checker": PATCH_CHECKER,
    "read_tests_file": READ_TESTS,
}


@pytest.mark.parametrize("name", sorted(EXPLOITS))
def test_exploit_scores_zero_in_the_sandbox_but_fools_a_naive_verifier(
    name, isolation, tmp_path
):
    code = EXPLOITS[name]
    result = _score(code, _verifier(isolation))
    assert result.score == 0.0, (name, result.metadata)
    naive_visible_only = name in {"hardcode_visible", "print_expected"}
    if naive_visible_only:
        # The naive verifier checks the examples the prompt showed.
        (tmp_path / "t.py").write_text(f"{code}\nassert sum_of_squares(3) == 14\n")
        fooled = (
            subprocess.run(
                [sys.executable, "t.py"], cwd=tmp_path, capture_output=True, check=False
            ).returncode
            == 0
        )
    else:
        fooled = _naive_passes(code, tmp_path)
    assert fooled, f"{name} should fool the naive verifier"


def test_forged_records_with_the_recovered_nonce_are_rejected(isolation):
    """Recovering the nonce from memory gains nothing: the values are checked
    here against hidden expectations, and a test reported twice fails."""
    result = _score(FORGE_NONCE, _verifier(isolation))
    assert result.score < 1.0
    assert result.metadata["duplicate_records"] >= 1


def test_a_nonce_only_harness_is_forgeable(tmp_path):
    """Control: a harness that prints a nonce after its asserts, with the
    parent only looking for the nonce, is fooled by the same recovery."""
    code = textwrap.dedent("""
        import gc, os
        nonce = next(v for o in gc.get_objects() if isinstance(o, dict)
                     for v in list(o.values())
                     if isinstance(v, str) and v.startswith("NONCE-") and len(v) > 6)
        os.write(1, nonce.encode())
        os._exit(0)
        """)
    harness = (
        "import secrets\nNONCE = 'NONCE-' + secrets.token_hex(8)\n"
        f"{code}\nassert False\nprint(NONCE)\n"
    )
    (tmp_path / "h.py").write_text(harness)
    out = subprocess.run(
        [sys.executable, "h.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    assert out.startswith("NONCE-")


def test_infinite_loop_is_a_zero_score_not_an_outage(isolation, tmp_path):
    """CPU time, not the wall clock, stops a busy loop: ``cpu_limit`` on both
    backends (in a container, PID 1 is a supervisor, so SIGXCPU still kills)."""
    loop = "def sum_of_squares(n):\n    while True:\n        pass"
    started = time.monotonic()
    limits = SandboxLimits(timeout_seconds=20, cpu_seconds=1)
    result = _score(loop, _verifier(isolation, limits=limits))
    assert result.score == 0.0
    assert result.metadata["status"] == "cpu_limit"
    assert time.monotonic() - started < 15
    with pytest.raises(subprocess.TimeoutExpired):  # the naive verifier hangs
        _naive_passes(loop, tmp_path, timeout=3)


def test_wall_clock_backstop_kills_a_sleeping_candidate(isolation):
    """A run that stalls without using CPU meets the wall-clock backstop; it
    reports ``wall_timeout`` and the little CPU it used, so it is told apart
    from a CPU-limit kill."""
    sleeper = "import time\ndef sum_of_squares(n):\n    time.sleep(30)"
    limits = SandboxLimits(timeout_seconds=1.5, cpu_seconds=5)
    started = time.monotonic()
    result = _score(sleeper, _verifier(isolation, limits=limits))
    assert result.score == 0.0 and result.metadata["status"] == "wall_timeout"
    assert time.monotonic() - started < 8
    if sys.platform.startswith("linux"):
        assert result.metadata["cpu_seconds"] is not None
        assert result.metadata["cpu_seconds"] < 1.0


def test_partial_results_before_a_hang_still_count(isolation):
    code = textwrap.dedent("""
        def sum_of_squares(n):
            while n == 10:
                pass
            return sum(i * i for i in range(1, n + 1))
        """)
    limits = SandboxLimits(timeout_seconds=3, cpu_seconds=1)
    result = _score(code, _verifier(isolation, limits=limits))
    assert result.metadata["passed"] == 2 and result.score == pytest.approx(2 / 3)


def _fork_storm(marker_dir: Path) -> str:
    return textwrap.dedent(f"""
        import os, time
        def sum_of_squares(n):
            for _ in range(4):
                try:
                    pid = os.fork()
                except OSError:
                    continue
                if pid == 0:
                    open(os.path.join({str(marker_dir)!r}, str(os.getpid())), "w").close()
                    time.sleep(60)
                    os._exit(0)
            return sum(i * i for i in range(1, n + 1))
        """)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - a reused pid of another user
        return True
    return True


def _reap(marker_dir: Path) -> list[int]:
    pids = [int(p.name) for p in marker_dir.iterdir()]
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return pids


def test_fork_storm_leaves_no_processes_behind(isolation, tmp_path):
    sandbox_markers = tmp_path / "sandbox"
    naive_markers = tmp_path / "naive"
    sandbox_markers.mkdir()
    naive_markers.mkdir()
    try:
        _score(_fork_storm(sandbox_markers), _verifier(isolation))
        time.sleep(0.5)
        survivors = [
            int(p.name) for p in sandbox_markers.iterdir() if _alive(int(p.name))
        ]
        assert survivors == []
        if os.geteuid() != 0:  # root ignores RLIMIT_NPROC (documented)
            assert list(sandbox_markers.iterdir()) == []
        # Control: the naive verifier leaves the children running.
        work = tmp_path / "work"
        work.mkdir()
        _naive_passes(_fork_storm(naive_markers), work)
        deadline = time.monotonic() + 5
        while len(list(naive_markers.iterdir())) < 4 and time.monotonic() < deadline:
            time.sleep(0.1)
        assert any(_alive(int(p.name)) for p in naive_markers.iterdir())
    finally:
        _reap(sandbox_markers)
        _reap(naive_markers)


@pytest.mark.parametrize(
    "code,status",
    [
        ("def sum_of_squares(n):\n    return len(bytearray(8 * 1024 ** 3))", "ok"),
        ("def sum_of_squares(n):\n    return sum_of_squares(n)", "ok"),
        (
            "import threading\ndef sum_of_squares(n):\n"
            "    threading.Thread(target=print).start()\n    return 1",
            "ok",
        ),
        (
            "def sum_of_squares(n):\n    open('big', 'wb').write(b'x' * 8 * 1024 ** 2)\n"
            "    return sum(i * i for i in range(1, n + 1))",
            "ok",
        ),
        ("def sum_of_squares(n):\n    raise SystemExit(0)", "ok"),
    ],
)
def test_resource_abuse_fails_its_tests_without_crashing_the_reward(
    code, status, isolation
):
    result = _score(code, _verifier(isolation))
    assert result.score == 0.0 and result.metadata["status"] == status


def test_output_floods_are_capped_and_correct_code_still_passes(isolation):
    flood = (
        "import sys\ndef sum_of_squares(n):\n    sys.stderr.write('e' * 10**7)\n"
        "    print('o' * 10**7)\n    return sum(i * i for i in range(1, n + 1))"
    )
    result = _score(flood, _verifier(isolation))
    assert result.score == 1.0


def test_results_must_be_strict_json_of_the_right_type(isolation):
    as_bool = _verifier(isolation, tests=[{"args": [1], "expected": 1}])
    assert _score("def sum_of_squares(n):\n    return True", as_bool).score == 0.0
    as_float = _verifier(
        isolation, tests=[{"args": [1], "expected": 0.3}], float_tolerance=1e-9
    )
    assert _score("def sum_of_squares(n):\n    return 0.1 + 0.2", as_float).score == 1.0
    exact = _verifier(isolation, tests=[{"args": [1], "expected": 0.3}])
    assert _score("def sum_of_squares(n):\n    return 0.1 + 0.2", exact).score == 0.0
    tuples = _verifier(isolation, tests=[{"args": [1], "expected": (1, 2)}])
    assert _score("def sum_of_squares(n):\n    return (1, 2)", tuples).score == 1.0


def test_program_tests_compare_stdout(isolation):
    tests = [
        {"stdin": "10 20\n", "expected_stdout": "30\n"},
        {"stdin": "-5 5\n", "expected_stdout": "0"},
    ]
    verifier = _verifier(isolation, tests=tests, entry_point=None)
    good = "import sys\nprint(sum(int(x) for x in sys.stdin.read().split()))  "
    assert _score(good, verifier).score == 1.0
    assert _score("print(5)", verifier).score == 0.0
    exits = "import sys\nprint(sum(map(int, input().split())))\nsys.exit(0)"
    assert _score(exits, verifier).score == 1.0
    nonzero = "import sys\nprint(sum(map(int, input().split())))\nsys.exit(2)"
    assert _score(nonzero, verifier).score == 0.0
    buffered = (
        "import sys\nsys.stdout.buffer.write(str(sum(map(int, "
        "sys.stdin.buffer.read().split()))).encode())"
    )
    assert _score(buffered, verifier).score == 1.0


# --- ordinary scripts are not false negatives (round 6) ---------------------

PROGRAM_TESTS = [
    {"stdin": "10 20\n", "expected_stdout": "30\n"},
    {"stdin": "-5 5\n", "expected_stdout": "0\n"},
    {"stdin": "100 1 7\n", "expected_stdout": "108\n"},
]
SUM_LINE = "print(sum(map(int, input().split())))"


@pytest.mark.parametrize(
    "code",
    [
        "print(sum(map(int, open(0).read().split())))",
        "import os\nprint(sum(map(int, os.read(0, 1 << 16).split())))",
        "import os, sys\nprint(sum(map(int, os.fdopen(sys.stdin.fileno()).read().split())))",
        f"{SUM_LINE}\nexit()",
        f"{SUM_LINE}\nexit(0)",
        f"{SUM_LINE}\nquit()",
        f"import os\nassert os.listdir('.') == []\n{SUM_LINE}",
    ],
    ids=["open0", "os_read0", "stdin_fileno", "exit", "exit0", "quit", "empty_cwd"],
)
def test_ordinary_program_idioms_pass(code, isolation):
    """stdin is a real file on fd 0 and ``exit()``/``quit()`` exist, as under
    a plain ``python``; the working directory stays empty."""
    verifier = _verifier(isolation, tests=PROGRAM_TESTS, entry_point=None)
    result = _score(code, verifier)
    assert result.score == 1.0, result.metadata


def test_a_program_exiting_with_an_error_still_fails(isolation):
    verifier = _verifier(isolation, tests=PROGRAM_TESTS, entry_point=None)
    assert _score(f"{SUM_LINE}\nexit(1)", verifier).score == 0.0
    assert _score(f"{SUM_LINE}\nquit('error')", verifier).score == 0.0


EXAMPLE_BLOCKS = {
    "prints": "# Example usage\nprint(sum_of_squares(3))",
    "reads_input": "# Example usage\nn = int(input('n: '))\nprint(sum_of_squares(n))",
    "undefined_name": "# Example usage:\nprint(sum_of_squares(my_number))",
    "main_guard_demo": (
        "if __name__ == '__main__':\n    print(sum_of_squares(int(input())))"
    ),
    "exits": "print(sum_of_squares(3))\nexit()",
    "wrong_example_assert": "assert sum_of_squares(3) == 15",
}


@pytest.mark.parametrize("name", sorted(EXAMPLE_BLOCKS))
def test_an_example_block_after_the_solution_does_not_fail_it(name, isolation):
    """Code after the entry point that fails at load (an example reading
    ``input()``, an undefined name) is reported, and the function is tested."""
    result = _score(f"{REFERENCE}\n\n{EXAMPLE_BLOCKS[name]}", _verifier(isolation))
    assert result.score == 1.0 and result.metadata["status"] == "ok", result.metadata


def test_a_load_error_before_the_entry_point_exists_still_fails(isolation):
    broken = f"print(undefined_name)\n{REFERENCE}"
    result = _score(broken, _verifier(isolation))
    assert result.score == 0.0 and result.metadata["status"] == "load_error"
    assert result.metadata["load_error"] == "NameError"
    warned = _score(f"{REFERENCE}\nprint(undefined_name)", _verifier(isolation))
    assert warned.score == 1.0 and warned.metadata["load_warning"] == "NameError"


# --- scores do not depend on load (round 6) -----------------------------------

#: About 0.3 s of CPU per run on a laptop core: far below the CPU limit, while
#: 46 at once need more than the wall clock below when they all run together.
BUSY = textwrap.dedent("""
    def sum_of_squares(n):
        k = 0
        for i in range(3_000_000):
            k += i
        return sum(i * i for i in range(1, n + 1))
    """)


def test_concurrent_identical_solutions_score_identically(isolation):
    """46 identical correct solutions at once all score 1.0: CPU time decides
    the score and at most ``sandbox_concurrency()`` sandboxes run together, so
    the others wait (off the clock) instead of being starved past it. In a
    container the clock also starts only once the harness runs, so a loaded
    daemon's start-up time never counts."""
    limits = SandboxLimits(timeout_seconds=2.5, cpu_seconds=3)
    verifier = _verifier(isolation, tests=HIDDEN[:2], limits=limits)
    turns = [ConversationTurn(role="assistant", content=f"```python\n{BUSY}\n```")]

    async def main() -> list[Any]:
        return await asyncio.gather(
            *(verifier.compute_reward(turns, {}) for _ in range(46))
        )

    results = asyncio.run(main())
    outcomes = sorted({(r.score, r.metadata["status"]) for r in results})
    assert outcomes == [(1.0, "ok")]
    assert code_verifier._SLOTS.active() == 0


def _intervals(runs: list[Any]) -> list[tuple[float, float]]:
    import json

    return [tuple(json.loads(run.outcomes[0]["value"])) for run in runs]


def _peak_overlap(intervals: list[tuple[float, float]]) -> int:
    events = sorted(
        [(start, 1) for start, _ in intervals] + [(end, -1) for _, end in intervals]
    )
    peak = running = 0
    for _, step in events:
        running += step
        peak = max(peak, running)
    return peak


SLEEP_AND_REPORT = (
    "import time\ndef f(n):\n    start = time.time()\n    time.sleep(0.4)\n"
    "    return [start, time.time()]"
)


def _run_many(count: int, limits: SandboxLimits, isolation: str) -> list[Any]:
    tests = normalize_tests([{"args": [1], "expected": None}])[1]

    async def main() -> list[Any]:
        return await asyncio.gather(
            *(
                run_sandboxed(
                    SLEEP_AND_REPORT,
                    tests,
                    mode="function",
                    entry_point="f",
                    limits=limits,
                    isolation=isolation,
                    docker=TEST_DOCKER,
                    allow_root_rlimit=True,
                )
                for _ in range(count)
            )
        )

    return asyncio.run(main())


def test_at_most_the_configured_number_of_sandboxes_run_at_once(isolation):
    code_verifier.set_sandbox_concurrency(2)
    try:
        assert code_verifier.sandbox_concurrency() == 2
        runs = _run_many(6, SandboxLimits(timeout_seconds=10, cpu_seconds=2), isolation)
    finally:
        code_verifier.set_sandbox_concurrency(None)
    assert [run.status for run in runs] == ["ok"] * 6
    assert _peak_overlap(_intervals(runs)) <= 2


def test_waiting_for_a_slot_does_not_count_against_the_wall_clock(isolation):
    """Five 0.4 s runs one at a time take about 2 s, beyond the 1.5 s backstop;
    each run's clock starts only once it holds a slot."""
    code_verifier.set_sandbox_concurrency(1)
    try:
        runs = _run_many(
            5, SandboxLimits(timeout_seconds=1.5, cpu_seconds=2), isolation
        )
    finally:
        code_verifier.set_sandbox_concurrency(None)
    assert [run.status for run in runs] == ["ok"] * 5
    assert _peak_overlap(_intervals(runs)) == 1


def test_cancelled_waiters_give_their_slots_back():
    slots = code_verifier._SandboxSlots()
    slots.set_limit(1)

    async def main() -> None:
        await slots.acquire()
        queued = [asyncio.create_task(slots.acquire()) for _ in range(3)]
        await asyncio.sleep(0)
        queued[0].cancel()  # cancelled while waiting
        slots.release()  # handed to queued[1] ...
        queued[1].cancel()  # ... which is cancelled before it runs
        for task in queued[:2]:
            with pytest.raises(asyncio.CancelledError):
                await task
        await queued[2]  # the slot reached the last waiter
        assert slots.active() == 1
        slots.release()
        assert slots.active() == 0

    asyncio.run(main())


def test_sandbox_concurrency_configuration(monkeypatch):
    monkeypatch.setenv(code_verifier.CONCURRENCY_ENV, "3")
    assert code_verifier._default_concurrency() == 3
    for bad in ("0", "-2", "many"):
        monkeypatch.setenv(code_verifier.CONCURRENCY_ENV, bad)
        with pytest.raises(ConfigurationError):
            code_verifier._default_concurrency()
    monkeypatch.delenv(code_verifier.CONCURRENCY_ENV)
    assert code_verifier._default_concurrency() >= 1
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ConfigurationError):
            code_verifier.set_sandbox_concurrency(bad)


# --- reading the dataset from disk: defended where files are confined ---------


def test_reading_expected_outputs_from_a_dataset_on_disk(isolation, tmp_path):
    """The dataset holding the hidden tests sits on disk, as in training. A
    container cannot see it, and Landlock confines an rlimit candidate to its
    own directory, so the reader scores 0. Without Landlock (macOS, Linux
    before 5.13) it is a known limit, pinned so the docs stay honest: the
    candidate runs as the trainer's user and reads any file that user can."""
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps(HIDDEN))
    reader = textwrap.dedent(f"""
        import json
        def sum_of_squares(n):
            with open({str(dataset)!r}) as handle:
                tests = json.load(handle)
            return next(t["expected"] for t in tests if t["args"] == [n])
        """)
    expected = 0.0 if _defends_files(isolation) else 1.0
    assert _score(reader, _verifier(isolation)).score == expected


def test_prose_and_oversized_replies_never_start_a_process(monkeypatch):
    async def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no process should start")

    monkeypatch.setattr(code_verifier, "run_sandboxed", forbidden)
    verifier = _verifier("auto")
    assert _score("I would add the numbers.", verifier).metadata["status"] == (
        "syntax_error"
    )
    huge = _verifier("auto", limits=SandboxLimits(max_code_chars=10))
    assert _score(REFERENCE, huge).metadata["status"] == "too_long"
    turns = [ConversationTurn(role="assistant", content="")]
    empty = asyncio.run(verifier.compute_reward(turns, {}))
    assert empty.metadata["status"] == "no_code"


def test_platform_without_rlimits_is_an_outage(monkeypatch):
    monkeypatch.setattr(code_verifier, "sandbox_available", lambda: False)
    with pytest.raises(RewardUnavailableError, match="POSIX"):
        _score(REFERENCE, _verifier("rlimit"))
    # auto, with Docker unusable too: still an outage, naming both.
    _fake_docker(monkeypatch, usable=False)
    with pytest.raises(RewardUnavailableError, match="POSIX.*Docker: no daemon"):
        _score(REFERENCE, _verifier("auto"))


def test_tests_come_from_context_and_bad_tests_are_configuration_errors(isolation):
    verifier = SandboxedCodeVerifier(
        limits=LIMITS, isolation=isolation, docker=TEST_DOCKER, allow_root_rlimit=True
    )
    turns = [ConversationTurn(role="assistant", content=f"```python\n{REFERENCE}\n```")]
    context = {"tests": HIDDEN, "entry_point": "sum_of_squares"}
    assert asyncio.run(verifier.compute_reward(turns, context)).score == 1.0
    for bad in (
        {},
        {"tests": [], "entry_point": "f"},
        {"tests": [{"args": [1]}], "entry_point": "f"},
        {"tests": HIDDEN + [{"stdin": "", "expected_stdout": ""}], "entry_point": "f"},
        {"tests": HIDDEN, "entry_point": "not an identifier"},
        {"tests": [{"args": [1], "expected": float("nan")}], "entry_point": "f"},
    ):
        with pytest.raises(ConfigurationError):
            asyncio.run(verifier.compute_reward(turns, bad))


def test_code_extraction():
    assert extract_code("x\n```python\na = 1\n```\n```py\nb = 2\n```") == "b = 2\n"
    assert extract_code("```js\nlet a\n```") == ""
    assert extract_code("a = 1") == "a = 1"
    tool = '<tool_call>{"name": "run", "arguments": {}}</tool_call>'
    assert extract_code(tool) == ""


# --- choosing a backend -----------------------------------------------------------


def _fake_docker(monkeypatch: Any, usable: bool, reason: str = "no daemon") -> None:
    status = DockerStatus(usable, reason, "27.0", ("name=seccomp,profile=builtin",))
    monkeypatch.setattr(
        code_verifier,
        "docker_sandbox_status",
        lambda config=None, refresh=False: status,
    )
    monkeypatch.setattr(code_verifier, "cached_docker_status", lambda config: None)
    monkeypatch.setattr(code_verifier, "_WARNED", set())


def test_auto_uses_docker_when_it_is_usable(monkeypatch):
    _fake_docker(monkeypatch, usable=True, reason="docker 27.0, image present")
    assert code_verifier.resolve_isolation("auto") == "docker"
    assert code_verifier.resolve_isolation("docker") == "docker"


@pytest.mark.skipif(not code_verifier.sandbox_available(), reason="needs POSIX")
def test_auto_falls_back_to_rlimit_with_one_warning_naming_the_gaps(
    monkeypatch, caplog
):
    _fake_docker(monkeypatch, usable=False, reason="the docker daemon is down")
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    with caplog.at_level(logging.WARNING, logger=code_verifier.__name__):
        assert code_verifier.resolve_isolation("auto") == "rlimit"
        assert code_verifier.resolve_isolation("auto") == "rlimit"
    messages = [r.getMessage() for r in caplog.records]
    warned = [m for m in messages if "isolation='auto'" in m]
    assert len(warned) == 1, messages  # once per process
    message = warned[0]
    assert "the docker daemon is down" in message
    assert "Not isolated" in message and "network" in message
    assert "kill the trainer" in message and "test inputs" in message
    # Without Landlock the warning also names the filesystem and /proc.
    assert ("/proc/<pid>/environ" in message) == (not landlock_available())
    assert code_verifier.DEFAULT_DOCKER_IMAGE in message


def test_explicit_docker_without_docker_is_an_outage(monkeypatch):
    _fake_docker(monkeypatch, usable=False, reason="the docker CLI is not on PATH")
    with pytest.raises(RewardUnavailableError, match="isolation='docker' but the"):
        code_verifier.resolve_isolation("docker")
    with pytest.raises(RewardUnavailableError, match="not on PATH"):
        _score(REFERENCE, _verifier("docker"))


def test_unknown_isolation_and_config_are_configuration_errors():
    with pytest.raises(ConfigurationError, match="isolation must be one of"):
        SandboxedCodeVerifier(isolation="none")
    with pytest.raises(ConfigurationError, match="isolation must be one of"):
        code_verifier.resolve_isolation("gvisor")
    with pytest.raises(ConfigurationError, match="DockerSandboxConfig"):
        SandboxedCodeVerifier(docker={"image": "python"})  # type: ignore[arg-type]


@pytest.mark.skipif(not hasattr(os, "geteuid"), reason="POSIX user ids")
def test_root_needs_an_explicit_opt_in_for_the_rlimit_backend(monkeypatch):
    """RLIMIT_NPROC does not hold for root, so the rlimit backend as root is
    refused unless the caller opts in (or Docker runs the candidate)."""
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    monkeypatch.delenv(code_verifier.ALLOW_ROOT_ENV, raising=False)
    kwargs: dict[str, Any] = {"tests": HIDDEN, "entry_point": "sum_of_squares"}
    with pytest.raises(ConfigurationError, match="root.*allow_root_rlimit"):
        SandboxedCodeVerifier(isolation="rlimit", **kwargs)
    SandboxedCodeVerifier(isolation="rlimit", allow_root_rlimit=True, **kwargs)
    _fake_docker(monkeypatch, usable=False)
    with pytest.raises(ConfigurationError, match="root"):
        code_verifier.resolve_isolation("auto")  # falling back to rlimit, too
    monkeypatch.setenv(code_verifier.ALLOW_ROOT_ENV, "1")
    assert code_verifier.resolve_isolation("auto") == "rlimit"
    _fake_docker(monkeypatch, usable=True)
    monkeypatch.delenv(code_verifier.ALLOW_ROOT_ENV)
    assert code_verifier.resolve_isolation("auto") == "docker"  # no opt-in needed


# --- the docker backend: configuration (no daemon needed) -------------------------


def _flag_values(argv: list[str], flag: str) -> list[str]:
    return [argv[i + 1] for i, item in enumerate(argv[:-1]) if item == flag]


def test_the_container_is_locked_down():
    """The ``docker run`` command: no network, read-only root, a small tmpfs,
    a non-root user, no capabilities, no-new-privileges, the default seccomp
    and AppArmor profiles, cgroup limits from the verifier's limits, no host
    mounts, no environment, never a pull."""
    spec = _sandbox_docker.DockerRunSpec(
        source="pass",
        payload=b"",
        python_flags=("-I", "-S"),
        memory_bytes=512 * 2**20,
        max_processes=0,
        timeout_seconds=30.0,
        max_stdout_bytes=1,
        max_stderr_bytes=1,
        extra_tmpfs_bytes=4096,
    )
    argv = _sandbox_docker.docker_run_command(TEST_DOCKER, spec, name="n")

    def value(flag: str) -> str:
        (found,) = _flag_values(argv, flag)
        return found

    assert argv[:4] == ["docker", "run", "--rm", "--interactive"]
    assert value("--network") == "none" and value("--ipc") == "none"
    assert "--read-only" in argv
    assert value("--cap-drop") == "ALL"
    assert value("--security-opt") == "no-new-privileges"
    assert "unconfined" not in " ".join(argv)  # default seccomp and AppArmor
    assert value("--user") == "65534:65534"
    tmpfs = value("--tmpfs")
    assert tmpfs.startswith("/sandbox:rw,nosuid,nodev,noexec,")
    assert "uid=65534" in tmpfs and "mode=0700" in tmpfs
    tmpfs_bytes = int(tmpfs.split("size=")[1].split(",")[0])
    assert tmpfs_bytes == TEST_DOCKER.tmpfs_bytes + 4096
    memory = 512 * 2**20 + tmpfs_bytes + TEST_DOCKER.memory_headroom_bytes
    assert value("--memory") == value("--memory-swap") == str(memory)  # no swap
    # Floor for runc init's start-up threads; RLIMIT_NPROC limits the candidate.
    assert value("--pids-limit") == str(_sandbox_docker.MIN_CONTAINER_PIDS)
    assert value("--cpus") == "1"
    assert value("--pull") == "never" and value("--log-driver") == "none"
    for flag in (
        "-v",
        "--volume",
        "--mount",
        "-e",
        "--env",
        "--env-file",
        "--privileged",
        "--cap-add",
        "--device",
        "--pid",
        "--userns",
        "--network-alias",
    ):
        assert flag not in argv, flag
    assert f"{TEST_LABEL}=1" in _flag_values(argv, "--label")
    assert value("--entrypoint") == "/usr/bin/env" and value(TEST_DOCKER.image) == "-i"
    assert "@sha256:" in TEST_DOCKER.image
    assert argv[-4:] == ["-c", "pass", "--supervise", "90"]
    pulling = DockerSandboxConfig(allow_pull=True)
    assert _flag_values(
        _sandbox_docker.docker_run_command(pulling, spec, name="n"), "--pull"
    ) == ["missing"]


def test_docker_config_is_validated():
    with pytest.raises(ConfigurationError, match="pinned by digest"):
        DockerSandboxConfig(image="python:3.12-slim")
    assert DockerSandboxConfig(image="python:3.12-slim", allow_unpinned_image=True)
    assert DockerSandboxConfig(image="sha256:" + "a" * 64)
    for user in ("0:0", "root", "1000", "1000:0", "0:1000", "nobody:nogroup"):
        with pytest.raises(ConfigurationError, match="non-root"):
            DockerSandboxConfig(user=user)
    for bad in (
        {"cpus": 0},
        {"tmpfs_bytes": 10},
        {"startup_timeout_seconds": 0},
        {"memory_headroom_bytes": -1},
        {"labels": {"bad key": "x"}},
        {"labels": {"key": 1}},
        {"python": ""},
    ):
        with pytest.raises(ConfigurationError):
            DockerSandboxConfig(**bad)  # type: ignore[arg-type]
    config = DockerSandboxConfig(labels={"a": "1"})
    with pytest.raises(TypeError):
        config.labels["b"] = "2"  # type: ignore[index]


def test_docker_status_says_why_docker_is_unusable(monkeypatch):
    monkeypatch.setattr(_sandbox_docker, "_STATUS", {})
    missing_cli = DockerSandboxConfig(docker="no-such-docker-cli-for-tests")
    status = docker_sandbox_status(missing_cli)
    assert not status.usable and "not on PATH" in status.reason
    monkeypatch.setattr(_sandbox_docker.shutil, "which", lambda name: "/bin/docker")
    replies = {"info": (1, "", "Cannot connect to the Docker daemon\nmore")}
    monkeypatch.setattr(_sandbox_docker, "_cli", lambda argv: replies[argv[1]])
    config = DockerSandboxConfig(image="python@sha256:" + "1" * 64)
    status = docker_sandbox_status(config)
    assert not status.usable and "Cannot connect to the Docker daemon)" in status.reason
    assert docker_sandbox_status(config) is status  # probed once
    replies["info"] = (0, "27.4.0|linux|name=seccomp,profile=builtin;name=apparmor", "")
    replies["image"] = (1, "", "Error: No such image")
    status = docker_sandbox_status(config, refresh=True)
    assert not status.usable
    assert f"docker pull {config.image}" in status.reason
    assert status.security_options == ("name=seccomp,profile=builtin", "name=apparmor")
    pulling = DockerSandboxConfig(image=config.image, allow_pull=True)
    assert docker_sandbox_status(pulling).usable
    replies["image"] = (0, "sha256:abc", "")
    assert docker_sandbox_status(config, refresh=True).usable
    replies["info"] = (0, "27.4.0|windows|", "")
    assert "windows" in docker_sandbox_status(config, refresh=True).reason


def test_container_exit_statuses_map_to_signals():
    """The supervisor reports a signal N as 128 + N, which is mapped back to
    -N as for a local subprocess (so SIGXCPU is ``cpu_limit`` on both).
    Containers are Linux, so these are Linux signal numbers on any host (the
    host's ``signal`` module has neither on Windows)."""
    linux_sigkill, linux_sigxcpu = 9, 24
    assert _sandbox_docker._signal_returncode(128 + linux_sigxcpu) == -linux_sigxcpu
    assert _sandbox_docker._signal_returncode(137) == -linux_sigkill
    for code in (None, 0, 1, 125, 128):
        assert _sandbox_docker._signal_returncode(code) == code


# --- isolation probes: harmless, and each asserts a capability is absent ----------


@pytest.fixture
def docker_ready() -> None:
    reason = _docker_unusable()
    if reason is not None:
        pytest.skip(f"the docker backend is unavailable: {reason}")


#: Looks for ways out: the parent's environment, the host's processes, the
#: network, writes outside the working directory, a dataset file on disk.
ESCAPE_PROBE = textwrap.dedent("""
    import os, socket
    def probe(port, host_pid, outside, dataset, in_container):
        found = {"env": sorted(k for k in os.environ if k != "LC_CTYPE")}
        try:
            pids = [p for p in os.listdir("/proc") if p.isdigit()]
            found["host_pid_visible"] = str(host_pid) in pids
            found["processes"] = len(pids)
        except OSError as error:
            found["host_pid_visible"] = False
            found["processes"] = type(error).__name__
        try:
            with open("/proc/%d/environ" % os.getppid(), "rb") as handle:
                found["parent_environ"] = len(handle.read())
        except OSError as error:
            found["parent_environ"] = type(error).__name__
        try:
            socket.create_connection(("127.0.0.1", port), timeout=2).close()
            found["localhost"] = "connected"
        except OSError as error:
            found["localhost"] = type(error).__name__
        found["writes"] = {}
        paths = [outside, "probe"]
        if in_container:  # on the host, only ever probe under the test's directory
            paths += ["/probe", "/tmp/probe", "/var/tmp/probe", "/dev/shm/probe"]
        for path in paths:
            try:
                with open(path, "w") as handle:
                    handle.write("x")
                found["writes"][path] = "written"
            except OSError as error:
                found["writes"][path] = type(error).__name__
        try:
            with open(dataset) as handle:
                handle.read()
            found["dataset"] = "read"
        except OSError as error:
            found["dataset"] = type(error).__name__
        found["uid"] = os.getuid()
        return found
    """)


def _escape_probe(isolation: str, tmp_path: Path) -> tuple[dict[str, Any], bool]:
    """Run the probe; also report whether anything reached a host listener."""
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps(HIDDEN))
    outside = str(tmp_path / "escaped")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(0.2)
        port = listener.getsockname()[1]
        args = [port, os.getpid(), outside, str(dataset), isolation == "docker"]
        tests = normalize_tests([{"args": args, "expected": None}])[1]
        run = _run(ESCAPE_PROBE, tests, isolation, entry_point="probe")
        try:
            listener.accept()[0].close()
            reached = True
        except OSError:  # socket.timeout included
            reached = False
    assert run.status == "ok", run
    return json.loads(run.outcomes[0]["value"]), reached


def test_docker_candidates_find_no_way_out(docker_ready, tmp_path, monkeypatch):
    monkeypatch.setenv("STATESET_VERIFIER_CANARY", secrets.token_hex(8))
    found, reached = _escape_probe("docker", tmp_path)
    assert found["env"] == []  # nothing from this process's environment
    assert not found["host_pid_visible"] and found["processes"] <= 2  # own pid ns
    assert found["parent_environ"] == 0  # the supervisor's environment is empty
    assert found["localhost"] != "connected" and not reached  # --network none
    writes = found["writes"]
    assert writes.pop("probe") == "written"  # its own tmpfs working directory
    assert all(result != "written" for result in writes.values()), writes
    assert not (tmp_path / "escaped").exists()
    assert found["dataset"] != "read"
    assert found["uid"] != 0


def test_rlimit_candidates_are_confined_to_their_files(tmp_path):
    """With Landlock the rlimit backend keeps the candidate out of /proc, the
    trainer's files and everything outside its working directory. (It does not
    isolate the network: see the docs.)"""
    if not code_verifier.sandbox_available() or not landlock_available():
        pytest.skip("needs the rlimit backend with Landlock (Linux 5.13+)")
    found, _ = _escape_probe("rlimit", tmp_path)
    assert found["env"] == []
    assert found["processes"] == "PermissionError"  # cannot list /proc
    assert found["parent_environ"] == "PermissionError"
    writes = found["writes"]
    assert writes.pop("probe") == "written"
    assert all(result != "written" for result in writes.values()), writes
    assert not (tmp_path / "escaped").exists()
    assert found["dataset"] == "PermissionError"


#: A stand-in trainer holding a secret in its environment: it scores a probe
#: that looks for the secret's name in /proc/<ppid>/environ.
TRAINER = textwrap.dedent('''
    import asyncio, json, sys
    from stateset_agents.rewards.code_verifier import (
        DockerSandboxConfig, normalize_tests, run_sandboxed,
    )
    PROBE = """
    import os
    def probe(name):
        try:
            with open("/proc/%d/environ" % os.getppid(), "rb") as handle:
                environ = handle.read()
        except OSError as error:
            return type(error).__name__
        return "leaked" if name.encode() in environ else "absent"
    """
    isolation, name = sys.argv[1], sys.argv[2]
    tests = normalize_tests([{"args": [name], "expected": None}])[1]
    run = asyncio.run(run_sandboxed(
        PROBE, tests, mode="function", entry_point="probe", isolation=isolation,
        docker=DockerSandboxConfig(labels={"stateset-verifier-test": "1"}),
        allow_root_rlimit=True,
    ))
    print(json.dumps([run.status, run.outcomes[0]]))
''')


def test_candidates_cannot_read_the_trainers_environment(isolation):
    """The trainer holds a secret in its environment; the candidate looks for
    it in ``/proc/<ppid>/environ`` and does not find it (Landlock denies the
    read; in a container the parent is the supervisor, with no environment)."""
    if isolation == "rlimit" and not landlock_available():
        pytest.skip("without Landlock the rlimit backend cannot hide it (documented)")
    name = "STATESET_VERIFIER_SECRET_" + secrets.token_hex(4).upper()
    env = {**os.environ, name: secrets.token_hex(16), "PYTHONPATH": str(REPO)}
    done = subprocess.run(
        [sys.executable, "-c", TRAINER, isolation, name],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    status, outcome = json.loads(done.stdout.strip().splitlines()[-1])
    assert status == "ok"
    assert json.loads(outcome["value"]) in {"absent", "PermissionError"}


# --- the docker backend: failures and clean-up --------------------------------------


def test_a_missing_image_is_a_clear_outage_and_is_never_pulled():
    status = docker_sandbox_status(TEST_DOCKER)
    if status.server_version is None:
        pytest.skip(f"no docker daemon: {status.reason}")
    missing = DockerSandboxConfig(
        image="python@sha256:" + "0" * 64, labels={TEST_LABEL: "1"}
    )
    probed = docker_sandbox_status(missing, refresh=True)
    assert not probed.usable and f"docker pull {missing.image}" in probed.reason
    verifier = SandboxedCodeVerifier(
        tests=HIDDEN, entry_point="sum_of_squares", isolation="docker", docker=missing
    )
    with pytest.raises(RewardUnavailableError, match="docker pull"):
        _score(REFERENCE, verifier)
    # The run itself never pulls (--pull never): the container does not start,
    # which is an outage (no candidate code ran), not a score.
    tests = normalize_tests(HIDDEN)[1]
    with pytest.raises(RewardUnavailableError, match="did not start.*docker pull"):
        asyncio.run(
            code_verifier._run_in_docker(
                REFERENCE, tests, "function", "sum_of_squares", LIMITS, False, missing
            )
        )
    inspect = ["docker", "image", "inspect", missing.image]
    assert subprocess.run(inspect, capture_output=True, check=False).returncode != 0


def _containers(run_label: str) -> list[str]:
    listing = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"label=stateset-verifier-run={run_label}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return listing.stdout.split()


def test_a_cancelled_docker_run_leaves_no_container(docker_ready):
    run_label = secrets.token_hex(6)
    config = DockerSandboxConfig(
        labels={TEST_LABEL: "1", "stateset-verifier-run": run_label}
    )
    sleeper = "import time\ndef f(n):\n    time.sleep(60)"
    tests = normalize_tests([{"args": [1], "expected": None}])[1]

    async def main() -> None:
        task = asyncio.create_task(
            run_sandboxed(
                sleeper,
                tests,
                mode="function",
                entry_point="f",
                limits=SandboxLimits(timeout_seconds=60, cpu_seconds=2),
                isolation="docker",
                docker=config,
            )
        )
        deadline = time.monotonic() + 60
        while not _containers(run_label) and time.monotonic() < deadline:
            await asyncio.sleep(0.2)
        assert _containers(run_label), "the container never started"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    assert _containers(run_label) == []


# --- latency ---------------------------------------------------------------------


def test_median_verification_latency(isolation, record_property):
    """Median wall time of one verification (three hidden tests), per backend;
    a container per verification costs container start-up on top. (Generous
    limits: this measures latency, not the backstop.)"""
    verifier = _verifier(isolation, limits=SandboxLimits(timeout_seconds=30))
    _score(REFERENCE, verifier)  # warm up: the backend probe and caches
    times = []
    for _ in range(5):
        started = time.perf_counter()
        assert _score(REFERENCE, verifier).score == 1.0
        times.append(time.perf_counter() - started)
    median = statistics.median(times)
    record_property(f"median_verification_seconds_{isolation}", round(median, 4))
    print(f"\nmedian verification time ({isolation}): {median * 1000:.0f} ms")
    assert median < (20.0 if isolation == "docker" else 10.0)
