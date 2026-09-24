"""Shared provenance checks for measured benchmark runners."""

from __future__ import annotations

import subprocess
from pathlib import Path


class BenchmarkProvenanceError(RuntimeError):
    """Raised when a measured run cannot prove its source checkout."""


def resolve_harness_commit(root: Path, claimed: str | None = None) -> str:
    """Return ``HEAD`` only for a clean checkout matching any claimed commit."""
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode != 0:
        raise BenchmarkProvenanceError("could not inspect benchmark harness worktree")
    if status.stdout.strip():
        raise BenchmarkProvenanceError("benchmark harness worktree must be clean")

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    commit = revision.stdout.strip()
    if revision.returncode != 0 or len(commit) != 40:
        raise BenchmarkProvenanceError(
            "could not resolve a full benchmark harness commit"
        )
    if claimed is not None and claimed != commit:
        raise BenchmarkProvenanceError(
            f"claimed harness commit {claimed!r} does not match checkout {commit}"
        )
    return commit
