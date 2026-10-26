"""SandboxedCodeVerifier: isolation, scoring, and every exploit it must resist.

Each exploit test runs the same candidate twice: against the sandboxed
verifier, which must score it 0 (and never raise), and against a *naive*
verifier (``_naive_passes``: candidate and assert-style tests written to one
file in the working directory, run with ``python``, pass when it exits 0), which
the exploit fools. The naive run is the control showing the exploit is real.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

from stateset_agents.core.errors import ConfigurationError
from stateset_agents.core.reward_context import RewardUnavailableError
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.rewards import code_verifier
from stateset_agents.rewards.code_verifier import (
    SandboxedCodeVerifier,
    SandboxLimits,
    extract_code,
    normalize_tests,
    run_sandboxed,
)

pytestmark = pytest.mark.skipif(
    not code_verifier.sandbox_available(), reason="the sandbox needs POSIX rlimits"
)

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


def _verifier(**kwargs: Any) -> SandboxedCodeVerifier:
    kwargs.setdefault("tests", HIDDEN)
    kwargs.setdefault("entry_point", "sum_of_squares")
    kwargs.setdefault("limits", LIMITS)
    return SandboxedCodeVerifier(**kwargs)


def _score(code: str, verifier: SandboxedCodeVerifier | None = None) -> Any:
    turns = [
        ConversationTurn(role="user", content="Write sum_of_squares(n)."),
        ConversationTurn(role="assistant", content=f"```python\n{code}\n```"),
    ]
    return asyncio.run((verifier or _verifier()).compute_reward(turns, {}))


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


def test_reference_passes_both():
    result = _score(REFERENCE)
    assert result.score == 1.0 and result.metadata["status"] == "ok"


def test_runs_isolated_in_a_fresh_empty_environment(tmp_path):
    probe = textwrap.dedent("""
        import os, sys
        def sum_of_squares(n):
            return [sys.flags.isolated, sys.flags.no_site, dict(os.environ),
                    os.listdir("."), os.getcwd()]
        """)
    run = asyncio.run(
        run_sandboxed(
            probe,
            normalize_tests([{"args": [1], "expected": None}])[1],
            mode="function",
            entry_point="sum_of_squares",
            limits=LIMITS,
        )
    )
    import json

    isolated, no_site, environ, listing, cwd = json.loads(run.outcomes[0]["value"])
    assert (isolated, no_site, listing) == (1, 1, [])
    # Nothing is inherited; the interpreter may set LC_CTYPE itself (PEP 538).
    assert set(environ) <= {"LC_CTYPE"}
    assert "stateset-sandbox-" in cwd and not os.path.exists(cwd)
    if sys.platform.startswith("linux"):
        assert {"RLIMIT_CPU", "RLIMIT_AS", "RLIMIT_FSIZE", "RLIMIT_NPROC"} <= set(
            run.limits_applied
        )


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
def test_exploit_scores_zero_in_the_sandbox_but_fools_a_naive_verifier(name, tmp_path):
    code = EXPLOITS[name]
    result = _score(code)
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


def test_forged_records_with_the_recovered_nonce_are_rejected():
    """Recovering the nonce from memory gains nothing: the values are checked
    here against hidden expectations, and a test reported twice fails."""
    result = _score(FORGE_NONCE)
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


def test_infinite_loop_is_a_zero_score_not_an_outage(tmp_path):
    """CPU time, not the wall clock, stops a busy loop: ``cpu_limit``."""
    loop = "def sum_of_squares(n):\n    while True:\n        pass"
    started = time.monotonic()
    result = _score(
        loop, _verifier(limits=SandboxLimits(timeout_seconds=20, cpu_seconds=1))
    )
    assert result.score == 0.0
    assert result.metadata["status"] == "cpu_limit"
    assert time.monotonic() - started < 15
    with pytest.raises(subprocess.TimeoutExpired):  # the naive verifier hangs
        _naive_passes(loop, tmp_path, timeout=3)


def test_wall_clock_backstop_kills_a_sleeping_candidate():
    """A run that stalls without using CPU meets the wall-clock backstop; it
    reports ``wall_timeout`` and the little CPU it used, so it is told apart
    from a CPU-limit kill."""
    sleeper = "import time\ndef sum_of_squares(n):\n    time.sleep(30)"
    limits = SandboxLimits(timeout_seconds=1.5, cpu_seconds=5)
    started = time.monotonic()
    result = _score(sleeper, _verifier(limits=limits))
    assert result.score == 0.0 and result.metadata["status"] == "wall_timeout"
    assert time.monotonic() - started < 8
    if sys.platform.startswith("linux"):
        assert result.metadata["cpu_seconds"] is not None
        assert result.metadata["cpu_seconds"] < 1.0


def test_partial_results_before_a_hang_still_count():
    code = textwrap.dedent("""
        def sum_of_squares(n):
            while n == 10:
                pass
            return sum(i * i for i in range(1, n + 1))
        """)
    limits = SandboxLimits(timeout_seconds=3, cpu_seconds=1)
    result = _score(code, _verifier(limits=limits))
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


def test_fork_storm_leaves_no_processes_behind(tmp_path):
    sandbox_markers = tmp_path / "sandbox"
    naive_markers = tmp_path / "naive"
    sandbox_markers.mkdir()
    naive_markers.mkdir()
    try:
        _score(_fork_storm(sandbox_markers))
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
def test_resource_abuse_fails_its_tests_without_crashing_the_reward(code, status):
    result = _score(code)
    assert result.score == 0.0 and result.metadata["status"] == status


def test_output_floods_are_capped_and_correct_code_still_passes():
    flood = (
        "import sys\ndef sum_of_squares(n):\n    sys.stderr.write('e' * 10**7)\n"
        "    print('o' * 10**7)\n    return sum(i * i for i in range(1, n + 1))"
    )
    result = _score(flood)
    assert result.score == 1.0


def test_results_must_be_strict_json_of_the_right_type():
    as_bool = _verifier(tests=[{"args": [1], "expected": 1}])
    assert _score("def sum_of_squares(n):\n    return True", as_bool).score == 0.0
    as_float = _verifier(tests=[{"args": [1], "expected": 0.3}], float_tolerance=1e-9)
    assert _score("def sum_of_squares(n):\n    return 0.1 + 0.2", as_float).score == 1.0
    exact = _verifier(tests=[{"args": [1], "expected": 0.3}])
    assert _score("def sum_of_squares(n):\n    return 0.1 + 0.2", exact).score == 0.0
    tuples = _verifier(tests=[{"args": [1], "expected": (1, 2)}])
    assert _score("def sum_of_squares(n):\n    return (1, 2)", tuples).score == 1.0


def test_program_tests_compare_stdout():
    tests = [
        {"stdin": "10 20\n", "expected_stdout": "30\n"},
        {"stdin": "-5 5\n", "expected_stdout": "0"},
    ]
    verifier = _verifier(tests=tests, entry_point=None)
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
def test_ordinary_program_idioms_pass(code):
    """stdin is a real file on fd 0 and ``exit()``/``quit()`` exist, as under
    a plain ``python``; the working directory stays empty."""
    verifier = _verifier(tests=PROGRAM_TESTS, entry_point=None)
    result = _score(code, verifier)
    assert result.score == 1.0, result.metadata


def test_a_program_exiting_with_an_error_still_fails():
    verifier = _verifier(tests=PROGRAM_TESTS, entry_point=None)
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
def test_an_example_block_after_the_solution_does_not_fail_it(name):
    """Code after the entry point that fails at load (an example reading
    ``input()``, an undefined name) is reported, and the function is tested."""
    result = _score(f"{REFERENCE}\n\n{EXAMPLE_BLOCKS[name]}")
    assert result.score == 1.0 and result.metadata["status"] == "ok", result.metadata


def test_a_load_error_before_the_entry_point_exists_still_fails():
    broken = f"print(undefined_name)\n{REFERENCE}"
    result = _score(broken)
    assert result.score == 0.0 and result.metadata["status"] == "load_error"
    assert result.metadata["load_error"] == "NameError"
    warned = _score(f"{REFERENCE}\nprint(undefined_name)")
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


def test_concurrent_identical_solutions_score_identically():
    """46 identical correct solutions at once all score 1.0: CPU time decides
    the score and at most ``sandbox_concurrency()`` sandboxes run together, so
    the others wait (off the clock) instead of being starved past it."""
    limits = SandboxLimits(timeout_seconds=2.5, cpu_seconds=3)
    verifier = _verifier(tests=HIDDEN[:2], limits=limits)
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


def _run_many(count: int, limits: SandboxLimits) -> list[Any]:
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
                )
                for _ in range(count)
            )
        )

    return asyncio.run(main())


def test_at_most_the_configured_number_of_sandboxes_run_at_once():
    code_verifier.set_sandbox_concurrency(2)
    try:
        assert code_verifier.sandbox_concurrency() == 2
        runs = _run_many(6, SandboxLimits(timeout_seconds=10, cpu_seconds=2))
    finally:
        code_verifier.set_sandbox_concurrency(None)
    assert [run.status for run in runs] == ["ok"] * 6
    assert _peak_overlap(_intervals(runs)) <= 2


def test_waiting_for_a_slot_does_not_count_against_the_wall_clock():
    """Five 0.4 s runs one at a time take about 2 s, beyond the 1.5 s backstop;
    each run's clock starts only once it holds a slot."""
    code_verifier.set_sandbox_concurrency(1)
    try:
        runs = _run_many(5, SandboxLimits(timeout_seconds=1.5, cpu_seconds=2))
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


# --- known limit: the candidate can read files on disk (round 6) --------------


def test_reading_expected_outputs_from_a_dataset_on_disk_is_a_known_limit(tmp_path):
    """Not defended, pinned so the docs stay honest: only the working directory
    is empty. The candidate runs as the trainer's user and reads any file that
    user can, such as the dataset holding the hidden tests."""
    import json

    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps(HIDDEN))
    reader = textwrap.dedent(f"""
        import json
        def sum_of_squares(n):
            with open({str(dataset)!r}) as handle:
                tests = json.load(handle)
            return next(t["expected"] for t in tests if t["args"] == [n])
        """)
    assert _score(reader).score == 1.0


def test_prose_and_oversized_replies_never_start_a_process(monkeypatch):
    async def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no process should start")

    monkeypatch.setattr(code_verifier, "run_sandboxed", forbidden)
    assert _score("I would add the numbers.").metadata["status"] == "syntax_error"
    huge = _verifier(limits=SandboxLimits(max_code_chars=10))
    assert _score(REFERENCE, huge).metadata["status"] == "too_long"
    turns = [ConversationTurn(role="assistant", content="")]
    empty = asyncio.run(_verifier().compute_reward(turns, {}))
    assert empty.metadata["status"] == "no_code"


def test_platform_without_rlimits_is_an_outage(monkeypatch):
    monkeypatch.setattr(code_verifier, "sandbox_available", lambda: False)
    with pytest.raises(RewardUnavailableError, match="POSIX"):
        _score(REFERENCE)


def test_tests_come_from_context_and_bad_tests_are_configuration_errors():
    verifier = SandboxedCodeVerifier(limits=LIMITS)
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
