"""Guardrail: the mypy typed-surface allowlist only ever grows.

mypy.ini gates CI on an explicit `files =` allowlist while the rest of the
repo is incrementally typed. This ratchet stops the allowlist from silently
shrinking: removing a file from the gate must be a deliberate act that also
lowers the floor here, in the same change, where a reviewer can see it.
"""

from __future__ import annotations

import configparser
from pathlib import Path

MYPY_INI = Path(__file__).resolve().parents[2] / "mypy.ini"

# Floor = number of files in mypy.ini's `files =` list as of 2026-08-11.
# Raise this whenever files are added to the gate; never lower it without
# a written justification in the commit that does so.
ALLOWLIST_FLOOR = 35


def _allowlisted_files() -> list[str]:
    parser = configparser.ConfigParser()
    parser.read(MYPY_INI)
    raw = parser.get("mypy", "files")
    return [entry.strip() for entry in raw.split(",") if entry.strip()]


def test_mypy_allowlist_never_shrinks() -> None:
    files = _allowlisted_files()
    assert len(files) >= ALLOWLIST_FLOOR, (
        f"mypy.ini's typed-surface allowlist shrank to {len(files)} files "
        f"(floor: {ALLOWLIST_FLOOR}). Removing files from the type gate is a "
        "quality regression — restore them, or lower ALLOWLIST_FLOOR in this "
        "test with a justification in the same commit."
    )


def test_mypy_allowlist_files_exist() -> None:
    repo_root = MYPY_INI.parent
    missing = [f for f in _allowlisted_files() if not (repo_root / f).exists()]
    assert not missing, (
        f"mypy.ini lists files that do not exist: {missing}. The gate "
        "silently checks nothing for these paths."
    )


def test_ratchet_floor_matches_reality() -> None:
    files = _allowlisted_files()
    assert len(files) == ALLOWLIST_FLOOR or len(files) > ALLOWLIST_FLOOR, "unreachable"
    if len(files) > ALLOWLIST_FLOOR:
        raise AssertionError(
            f"mypy.ini now gates {len(files)} files but ALLOWLIST_FLOOR is "
            f"{ALLOWLIST_FLOOR}. Raise the floor to {len(files)} to lock in "
            "the gain."
        )
