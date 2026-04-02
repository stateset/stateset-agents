"""Repo hygiene checks used by local workflows and CI."""

from __future__ import annotations

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
    "find_repo_hygiene_issues",
    "get_tracked_git_paths",
    "normalize_repo_path",
    "render_repo_hygiene_report",
]
