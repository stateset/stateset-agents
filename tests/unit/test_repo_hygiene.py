"""Unit tests for repo hygiene checks."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from stateset_agents.utils.repo_hygiene import (
    find_repo_hygiene_issues,
    get_tracked_git_paths,
    normalize_repo_path,
    render_repo_hygiene_report,
)


def test_normalize_repo_path_handles_prefixes_and_windows_separators() -> None:
    assert normalize_repo_path("./stateset_agents\\core\\agent.py") == (
        "stateset_agents/core/agent.py"
    )


def test_find_repo_hygiene_issues_returns_empty_for_source_files() -> None:
    issues = find_repo_hygiene_issues(
        [
            "stateset_agents/core/agent.py",
            "tests/unit/test_agent.py",
            "docs/ARCHITECTURE.md",
        ]
    )

    assert issues == []


def test_find_repo_hygiene_issues_flags_generated_directories() -> None:
    issues = find_repo_hygiene_issues(
        [
            "htmlcov/index.html",
            "outputs/checkpoint-1/model.safetensors",
            "dashboard/dist/assets/index.js",
        ]
    )

    assert "tracked generated artifact: htmlcov/index.html" in issues
    assert "tracked generated artifact: outputs/checkpoint-1/model.safetensors" in issues
    assert "tracked generated artifact: dashboard/dist/assets/index.js" in issues


def test_find_repo_hygiene_issues_flags_reports_and_backup_files() -> None:
    issues = find_repo_hygiene_issues(
        [
            ".coverage",
            "coverage.xml",
            "pyproject.toml.backup",
            "notes.tmp",
        ]
    )

    assert "tracked generated report: .coverage" in issues
    assert "tracked generated report: coverage.xml" in issues
    assert "tracked backup/temp file: pyproject.toml.backup" in issues
    assert "tracked backup/temp file: notes.tmp" in issues


def test_render_repo_hygiene_report_formats_failures() -> None:
    report = render_repo_hygiene_report(
        [
            "tracked generated artifact: htmlcov/index.html",
            "tracked backup/temp file: pyproject.toml.backup",
        ]
    )

    assert report.startswith("Repo hygiene check failed.")
    assert "- tracked generated artifact: htmlcov/index.html" in report


def test_get_tracked_git_paths_parses_null_delimited_output(tmp_path: Path) -> None:
    completed = Mock(returncode=0, stdout=b"a.py\0b.py\0", stderr=b"")
    with patch("stateset_agents.utils.repo_hygiene.subprocess.run", return_value=completed):
        assert get_tracked_git_paths(tmp_path) == ["a.py", "b.py"]


def test_get_tracked_git_paths_raises_on_git_failure(tmp_path: Path) -> None:
    completed = Mock(returncode=1, stdout=b"", stderr=b"fatal: not a git repository")
    with patch("stateset_agents.utils.repo_hygiene.subprocess.run", return_value=completed):
        with pytest.raises(RuntimeError, match="fatal: not a git repository"):
            get_tracked_git_paths(tmp_path)
