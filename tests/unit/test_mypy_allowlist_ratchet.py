"""Guardrail: the blocking mypy source surface only ever grows.

The `files =` setting may name individual files or package directories. This
ratchet expands directories before counting, so switching to a package-wide
gate cannot accidentally look like one checked file.
"""

from __future__ import annotations

import configparser
from pathlib import Path

MYPY_INI = Path(__file__).resolve().parents[2] / "mypy.ini"

# Floor = all 329 packaged Python files at v0.56.0.
ALLOWLIST_FLOOR = 329


def _allowlisted_files() -> list[str]:
    parser = configparser.ConfigParser()
    parser.read(MYPY_INI)
    raw = parser.get("mypy", "files")
    repo_root = MYPY_INI.parent
    files: set[str] = set()
    for entry in (part.strip() for part in raw.split(",")):
        if not entry:
            continue
        path = repo_root / entry
        if path.is_dir():
            files.update(
                child.relative_to(repo_root).as_posix()
                for child in path.rglob("*.py")
                if child.is_file()
            )
        else:
            files.add(entry)
    return sorted(files)


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
