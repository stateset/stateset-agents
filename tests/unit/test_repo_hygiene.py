"""Unit tests for repo hygiene checks."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from stateset_agents.utils.repo_hygiene import (
    extract_dunder_version,
    extract_project_version,
    find_repo_hygiene_issues,
    find_version_hygiene_issues,
    get_tracked_git_paths,
    normalize_repo_path,
    render_repo_hygiene_report,
    uses_package_version_alias,
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


def test_extract_project_version_reads_pyproject_metadata(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "stateset-agents"\nversion = "1.2.3"\n')

    assert extract_project_version(pyproject) == "1.2.3"


def test_extract_dunder_version_reads_python_module_version(tmp_path: Path) -> None:
    module = tmp_path / "__init__.py"
    module.write_text('__version__ = "1.2.3"\n')

    assert extract_dunder_version(module) == "1.2.3"


def test_uses_package_version_alias_detects_single_source_pattern(tmp_path: Path) -> None:
    module = tmp_path / "__init__.py"
    module.write_text("__version__ = _PACKAGE_VERSION\n")

    assert uses_package_version_alias(module) is True


def test_find_version_hygiene_issues_flags_mismatched_internal_versions(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path
    (repo_root / "stateset_agents" / "api").mkdir(parents=True)
    (repo_root / "stateset_agents" / "core" / "enhanced").mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text(
        '[project]\nname = "stateset-agents"\nversion = "1.2.3"\n'
    )
    (repo_root / "stateset_agents" / "__init__.py").write_text('__version__ = "1.2.3"\n')
    (repo_root / "stateset_agents" / "api" / "__init__.py").write_text(
        '__version__ = "1.2.3"\n'
    )
    (repo_root / "stateset_agents" / "core" / "enhanced" / "__init__.py").write_text(
        '__version__ = "0.5.0"\n'
    )

    issues = find_version_hygiene_issues(repo_root)

    assert issues == [
        "version mismatch: stateset_agents/core/enhanced/__init__.py -> 0.5.0 "
        "(expected 1.2.3)"
    ]
