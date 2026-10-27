"""Supply-chain guardrails for GitHub Actions workflows.

Every third-party action must be pinned to an immutable 40-hex commit SHA (with
a trailing version comment so Dependabot can keep it current), and every
workflow must declare a top-level ``permissions:`` block so the default
``GITHUB_TOKEN`` is least-privilege.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = sorted((ROOT / ".github/workflows").glob("*.y*ml"))
USES_RE = re.compile(r"^\s*(?:-\s*)?uses:\s*['\"]?([^'\"\s#]+)['\"]?\s*(#.*)?$")
PINNED_RE = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")


def _uses_lines(path: Path) -> list[tuple[int, str, str]]:
    found = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = USES_RE.match(line)
        if match:
            found.append((lineno, match.group(1), match.group(2) or ""))
    return found


def test_workflows_exist() -> None:
    assert WORKFLOWS, "expected .github/workflows/*.yml"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_third_party_actions_are_sha_pinned(path: Path) -> None:
    unpinned = []
    for lineno, ref, comment in _uses_lines(path):
        if ref.startswith("./") or ref.startswith("docker://"):
            continue
        if not PINNED_RE.match(ref) or not comment.strip("# ").strip():
            unpinned.append(f"{path.name}:{lineno}: {ref} {comment}".rstrip())
    assert (
        not unpinned
    ), "Pin actions to a full commit SHA with a `# vX.Y.Z` comment:\n" + "\n".join(
        unpinned
    )


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_declares_top_level_permissions(path: Path) -> None:
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "permissions" in workflow, f"{path.name} lacks top-level permissions"


def test_benchmark_gh_pages_writes_only_on_push() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/benchmark.yml").read_text(encoding="utf-8")
    )
    assert workflow["permissions"] == {"contents": "read"}
    for name, job in workflow["jobs"].items():
        steps = job.get("steps", [])
        pushes = [
            s for s in steps if s.get("with", {}).get("auto-push") in (True, "true")
        ]
        writes = job.get("permissions", {}).get("contents") == "write"
        if pushes or writes:
            assert "github.event_name == 'push'" in job.get("if", ""), name
            assert "refs/heads/master" in job["if"], name


def test_dependabot_tracks_github_actions() -> None:
    config = yaml.safe_load((ROOT / ".github/dependabot.yml").read_text())
    ecosystems = {u["package-ecosystem"] for u in config["updates"]}
    assert "github-actions" in ecosystems


@pytest.mark.parametrize("name", ["ci.yml", "security.yml"])
def test_pull_requests_into_integration_branches_run_ci(name: str) -> None:
    """Release-candidate work lands through PRs into integration/* branches;
    those PRs must get the test and security gates without a manual dispatch."""
    path = ROOT / ".github/workflows" / name
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    triggers = workflow.get("on", workflow.get(True))
    branches = triggers["pull_request"]["branches"]
    assert "integration/**" in branches
    assert {"main", "master"} <= set(branches)
