"""Structure tests for the docs consolidation (surface-consolidation Task 3).

Asserts that superseded comparison/dev-journal docs are archived rather than
left at their old top-level paths, and that the merged docs/COMPARISONS.md
contains all three comparison sections.
"""

from __future__ import annotations

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
