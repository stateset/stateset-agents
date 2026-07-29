"""Structure tests for the docs consolidation (surface-consolidation Task 3).

Asserts that superseded comparison/dev-journal docs are archived rather than
left at their old top-level paths, and that the merged docs/COMPARISONS.md
contains all three comparison sections.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ARCHIVED_DOC_FILES = [
    "docs/COMPARISON_TRL.md",
    "docs/COMPARISON_LLM_FRAMEWORKS.md",
    "docs/COMPARISON_TRADITIONAL_RL.md",
    "docs/ENHANCEMENTS_SUMMARY.md",
    "docs/FRAMEWORK_ENHANCEMENT_SUMMARY.md",
    "GYM_INTEGRATION_COMPLETE.md",
]


def test_superseded_docs_absent_at_old_paths() -> None:
    for rel_path in ARCHIVED_DOC_FILES:
        assert not (
            REPO_ROOT / rel_path
        ).exists(), f"{rel_path} should have been moved to docs/archive/"


def test_archived_docs_present_in_archive() -> None:
    for rel_path in ARCHIVED_DOC_FILES:
        archived_path = REPO_ROOT / "docs" / "archive" / Path(rel_path).name
        assert archived_path.exists(), f"expected {archived_path} to exist"


def test_comparisons_doc_has_all_three_sections() -> None:
    contents = (REPO_ROOT / "docs" / "COMPARISONS.md").read_text(encoding="utf-8")

    assert "## StateSet Agents vs Hugging Face TRL" in contents
    assert "## StateSet Agents vs Traditional RL Frameworks" in contents
    assert "## StateSet Agents vs LLM Orchestration Frameworks" in contents


def test_readme_marks_exactly_one_latest_release() -> None:
    """Guard against the version-bump sed mangling What's-new headers.

    Release bumps rewrite version strings across the README; twice now that
    rewrote a historical ``**vX.Y.Z:**`` heading into a second block claiming
    to be the latest release. Exactly one block may carry the marker, and its
    version must match the packaged version.
    """
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    markers = re.findall(r"\*\*v([0-9.]+) \(latest release", readme)

    assert len(markers) == 1, (
        f"README must mark exactly one release as latest; found {len(markers)}: "
        f"{markers}"
    )

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    version_match = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE)
    assert version_match is not None, "could not read version from pyproject.toml"
    assert markers[0] == version_match.group(1), (
        f"README's latest-release block says v{markers[0]} but pyproject.toml "
        f"says {version_match.group(1)}"
    )
