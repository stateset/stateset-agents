"""Regression guard: pytest must exit without a logging traceback.

`tests/conftest.py` installs an atexit hook that keeps litellm's own
interpreter-exit cleanup from writing a DEBUG record into a logging handler
that is still bound to pytest's already-closed captured stdout (which surfaces
as a "--- Logging error ---" traceback after the test summary).

The failure only happens at *interpreter exit*, after the summary is printed,
so it cannot be observed from inside a test. This guard therefore runs a real
pytest in a subprocess and inspects its output.

The probe file is written into a dot-prefixed directory under `tests/` for two
reasons: pytest walks up from the argument to collect `conftest.py` files, so
it must live inside `tests/` for the fix under test to be loaded at all, and a
leading dot keeps the outer run (including xdist workers) from collecting it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = REPO_ROOT / "tests"

# Reproduces the real-world conditions: an API test leaves the root logger at
# DEBUG with a StreamHandler bound to pytest's captured stdout, and an async
# test leaves no usable event loop behind. litellm's atexit hook then falls
# back to asyncio.new_event_loop(), whose DEBUG record hits the closed stream.
PROBE_SOURCE = """
import asyncio
import logging
import sys

import pytest

pytest.importorskip("litellm")
try:
    import wandb  # noqa: F401
except Exception:
    pass


def test_leaves_hostile_shutdown_state():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(logging.StreamHandler(sys.stdout))
    asyncio.set_event_loop(None)
"""


def _have_litellm_or_wandb() -> bool:
    from importlib.util import find_spec

    for name in ("litellm", "wandb"):
        try:
            if find_spec(name) is not None:
                return True
        except Exception:  # pragma: no cover - broken/partial install
            continue
    return False


@pytest.mark.skipif(
    not _have_litellm_or_wandb(),
    reason="neither litellm nor wandb is installed; nothing to trigger the traceback",
)
def test_pytest_session_exits_without_logging_error(tmp_path: Path) -> None:
    probe_dir = TESTS_DIR / f".shutdown_probe_{os.getpid()}"
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe = probe_dir / "test_clean_exit_probe.py"
    probe.write_text(PROBE_SOURCE)

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "-n0",
                "-c",
                str(REPO_ROOT / "pytest.ini"),
                "--rootdir",
                str(REPO_ROOT),
                str(probe),
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
            # Keep the child out of the parent's coverage/tmp bookkeeping.
            env={**os.environ, "COV_CORE_SOURCE": ""},
        )
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)

    output = result.stdout + result.stderr
    assert result.returncode == 0, f"probe session failed:\n{output}"
    assert "Logging error" not in output, (
        "pytest exited with a logging traceback — the atexit cleanup in "
        f"tests/conftest.py regressed:\n{output}"
    )
    assert "I/O operation on closed file" not in output, output
