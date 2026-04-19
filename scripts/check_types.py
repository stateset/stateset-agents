#!/usr/bin/env python3
"""Type checking utility for the StateSet Agents repo.

We intentionally gate a small, stable mypy surface in `mypy.ini` so type
checking is meaningful and CI stays fast while the rest of the codebase is
incrementally typed.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _mypy_base_cmd() -> list[str]:
    """Return a stable mypy invocation for local and CI type checks.

    ``mypy`` cache writes have intermittently crashed on this repo with
    ``AssertionError: unresolved placeholder type None`` during cache
    serialization. Running without incremental mode and with an explicit
    throwaway cache dir keeps the type gate deterministic.
    """
    return [
        "mypy",
        "--config-file",
        "mypy.ini",
        "--no-incremental",
        "--cache-dir=/dev/null",
    ]


def _run(cmd: list[str]) -> int:
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run mypy type checks")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run mypy on the full `stateset_agents/` package (may fail).",
    )
    args = parser.parse_args()

    if args.all:
        return _run([*_mypy_base_cmd(), "stateset_agents"])

    if not Path("mypy.ini").exists():
        print("mypy.ini not found; refusing to guess config.")
        return 2

    return _run(_mypy_base_cmd())


if __name__ == "__main__":
    raise SystemExit(main())
