"""Repo hygiene checks used by local workflows and CI."""

from __future__ import annotations

import re
from pathlib import Path
import subprocess

TRACKED_GENERATED_PATH_PREFIXES = (
    "build/",
    "dist/",
    "htmlcov/",
    "outputs/",
    "checkpoints/",
    "stateset_agents.egg-info/",
    "dashboard/dist/",
    "docs/_build/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".hypothesis/",
    ".codex/",
    ".autoresearch/",
)
TRACKED_GENERATED_BASENAMES = {
    ".coverage",
    "coverage.xml",
    "junit.xml",
    "pytest.xml",
}
TRACKED_TEMP_SUFFIXES = (".backup", ".bak", ".orig", ".tmp")
VERSION_SURFACE_PATHS = (
    Path("pyproject.toml"),
    Path("stateset_agents/__init__.py"),
    Path("stateset_agents/api/__init__.py"),
    Path("stateset_agents/core/enhanced/__init__.py"),
)


def normalize_repo_path(path: str) -> str:
    """Normalize a repo-relative path for comparisons."""
    normalized = path.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.strip("/")


def _matches_prefix(path: str, prefix: str) -> bool:
    stripped = prefix.rstrip("/")
    return path == stripped or path.startswith(prefix)


def find_repo_hygiene_issues(paths: list[str]) -> list[str]:
    """Return hygiene issues for a set of tracked repo-relative paths."""
    issues: list[str] = []
    seen: set[str] = set()

    for raw_path in paths:
        path = normalize_repo_path(raw_path)
        if not path or path in seen:
            continue
        seen.add(path)

        if any(
            _matches_prefix(path, prefix) for prefix in TRACKED_GENERATED_PATH_PREFIXES
        ):
            issues.append(f"tracked generated artifact: {path}")
            continue

        basename = Path(path).name
        if basename in TRACKED_GENERATED_BASENAMES:
            issues.append(f"tracked generated report: {path}")
            continue

        if basename.endswith(TRACKED_TEMP_SUFFIXES):
            issues.append(f"tracked backup/temp file: {path}")

    return sorted(issues)


def extract_project_version(pyproject_path: str | Path) -> str:
    """Return the package version declared in ``pyproject.toml``."""
    match = re.search(
        r"(?ms)^\[project\].*?^version\s*=\s*[\"']([^\"']+)[\"']\s*$",
        Path(pyproject_path).read_text(encoding="utf-8"),
    )
    if match is None:
        raise ValueError(f"Could not find [project].version in {pyproject_path}")
    return match.group(1)


def extract_dunder_version(module_path: str | Path) -> str:
    """Return the assigned ``__version__`` string from a Python module."""
    match = re.search(
        r'(?m)^__version__\s*=\s*[\"\']([^\"\']+)[\"\']\s*$',
        Path(module_path).read_text(encoding="utf-8"),
    )
    if match is None:
        raise ValueError(f"Could not find __version__ in {module_path}")
    return match.group(1)


def uses_package_version_alias(module_path: str | Path) -> bool:
    """Return whether ``__version__`` delegates to ``_PACKAGE_VERSION``."""
    return bool(
        re.search(
            r"(?m)^__version__\s*=\s*_PACKAGE_VERSION\s*$",
            Path(module_path).read_text(encoding="utf-8"),
        )
    )


def find_version_hygiene_issues(
    repo_root: str | Path,
    *,
    package_version: str | None = None,
) -> list[str]:
    """Return internal version mismatches for known public version surfaces."""
    root = Path(repo_root)
    expected_version = package_version or extract_project_version(root / "pyproject.toml")

    issues: list[str] = []
    for relative_path in VERSION_SURFACE_PATHS:
        path = root / relative_path
        if not path.exists():
            issues.append(f"missing version surface: {normalize_repo_path(str(relative_path))}")
            continue

        actual_version = (
            extract_project_version(path)
            if path.name == "pyproject.toml"
            else (
                expected_version
                if uses_package_version_alias(path)
                else extract_dunder_version(path)
            )
        )
        if actual_version != expected_version:
            issues.append(
                "version mismatch: "
                f"{normalize_repo_path(str(relative_path))} -> "
                f"{actual_version} (expected {expected_version})"
            )

    return sorted(issues)


def get_tracked_git_paths(cwd: str | Path | None = None) -> list[str]:
    """Return tracked paths from git."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(stderr or "git ls-files failed")

    payload = result.stdout.decode("utf-8", errors="replace")
    return [item for item in payload.split("\0") if item]


def render_repo_hygiene_report(issues: list[str]) -> str:
    """Render a stable human-readable report."""
    if not issues:
        return "Repo hygiene check passed."

    lines = [
        "Repo hygiene check failed.",
        "Remove generated, backup, or local-tool artifacts from tracked files:",
    ]
    lines.extend(f"- {issue}" for issue in issues)
    return "\n".join(lines)


__all__ = [
    "TRACKED_GENERATED_BASENAMES",
    "TRACKED_GENERATED_PATH_PREFIXES",
    "TRACKED_TEMP_SUFFIXES",
    "VERSION_SURFACE_PATHS",
    "extract_dunder_version",
    "extract_project_version",
    "find_repo_hygiene_issues",
    "find_version_hygiene_issues",
    "get_tracked_git_paths",
    "normalize_repo_path",
    "render_repo_hygiene_report",
    "uses_package_version_alias",
]
