"""examples/README.md must list every non-archived, non-package example.

Prevents the copy-paste sprawl this repo just cleaned up from silently
regrowing an undocumented example file.
"""

from __future__ import annotations

from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
README_PATH = EXAMPLES_DIR / "README.md"

# Files that intentionally don't need a README mention: package marker.
_EXCLUDED_NAMES = {"__init__.py"}


def _top_level_example_files() -> list[Path]:
    return sorted(
        p
        for p in EXAMPLES_DIR.glob("*.py")
        if p.name not in _EXCLUDED_NAMES
    )


def test_readme_exists() -> None:
    assert README_PATH.is_file(), "examples/README.md must exist"


def test_every_top_level_example_is_documented_in_readme() -> None:
    readme_text = README_PATH.read_text()
    files = _top_level_example_files()
    assert files, "expected at least one example script under examples/"

    missing = [p.name for p in files if p.name not in readme_text]
    assert not missing, (
        "examples/README.md is missing an entry for these top-level "
        f"examples: {missing}. Add a one-line description (and, if "
        "applicable, a usage snippet) for each, or move the file to "
        "examples/archive/ if it is no longer canonical."
    )


def test_archive_directory_is_excluded_from_the_walk() -> None:
    archive_dir = EXAMPLES_DIR / "archive"
    assert archive_dir.is_dir(), "examples/archive/ should exist"
    assert archive_dir not in {p.parent for p in _top_level_example_files()}
