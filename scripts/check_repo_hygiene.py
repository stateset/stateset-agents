#!/usr/bin/env python3
"""Fail when generated or backup artifacts are tracked in git."""

from __future__ import annotations

from pathlib import Path

from stateset_agents.utils.repo_hygiene import (
    find_repo_hygiene_issues,
    get_tracked_git_paths,
    render_repo_hygiene_report,
)


def main() -> int:
    try:
        tracked_paths = [
            path for path in get_tracked_git_paths() if Path(path).exists()
        ]
    except RuntimeError as exc:
        print(f"Repo hygiene check could not inspect tracked files: {exc}")
        return 2

    issues = find_repo_hygiene_issues(tracked_paths)
    print(render_repo_hygiene_report(issues))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
