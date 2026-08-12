"""Unit tests for scripts/release.py — pure functions against fixture files.

No git operations, no network: build_plan runs against a tmp fixture repo,
and the git preflight is tested with git_output monkeypatched.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_release_module():
    spec = importlib.util.spec_from_file_location(
        "release_script", REPO_ROOT / "scripts" / "release.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["release_script"] = module
    spec.loader.exec_module(module)
    return module


release = _load_release_module()

CHANGELOG_FIXTURE = """# Changelog

Intro text.

## [Unreleased]

### Added

- A new thing.

## [0.25.0] - 2026-08-12 — old release

### Added

- Old thing.
"""

README_FIXTURE = """# StateSet Agents

## What's new

**v0.25.0 (latest release — [live on PyPI](https://pypi.org/project/stateset-agents/)):**

- Old latest bullet one.
- Old latest bullet two.

**v0.24.0:**

- Older bullet.

## Installation

```bash
pip install stateset-agents          # latest release (v0.23.0)
```

- [`CHANGELOG.md`](CHANGELOG.md) — what changed (latest release `v0.23.0`).
"""


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Path:
    """A minimal repo layout carrying every file build_plan touches."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "stateset-agents"\nversion = "0.25.0"\n'
        'dependencies = ["pytest-asyncio>=0.21.0"]\n',
        encoding="utf-8",
    )
    pkg = tmp_path / "stateset_agents"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('__version__ = "0.25.0"\n', encoding="utf-8")

    helm = tmp_path / "deployment" / "helm" / "stateset-agents"
    helm.mkdir(parents=True)
    (helm / "Chart.yaml").write_text(
        'apiVersion: v2\nversion: 0.1.0\nappVersion: "0.25.0"\n', encoding="utf-8"
    )
    (helm / "README.md").write_text(
        "image tag 0.25.0 twice: 0.25.0\n", encoding="utf-8"
    )
    (helm / "values.yaml").write_text('tag: "0.25.0"\n', encoding="utf-8")

    k8s = tmp_path / "deployment" / "kubernetes"
    k8s.mkdir(parents=True)
    for name in (
        "glm5-1-training-job.yaml",
        "glm5-2-training-job.yaml",
        "kimi-k25-training-job.yaml",
        "qwen3-5-27b-training-job.yaml",
        "production-deployment.yaml",
    ):
        (k8s / name).write_text("image: trainer:0.25.0\n", encoding="utf-8")
    # deployment.yaml deliberately has NO version — exercises the 0-count warning
    (k8s / "deployment.yaml").write_text("image: api:latest\n", encoding="utf-8")

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "ARCHITECTURE.md").write_text("Source version: 0.25.0\n", encoding="utf-8")
    (docs / "KIMI_K25_GKE_AUTOPILOT.md").write_text(
        "trainer:0.25.0 build\n", encoding="utf-8"
    )

    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG_FIXTURE, encoding="utf-8")
    (tmp_path / "README.md").write_text(README_FIXTURE, encoding="utf-8")
    return tmp_path


# --- semver preflight -------------------------------------------------------


def test_parse_semver_ok():
    assert release.parse_semver("1.2.30") == (1, 2, 30)


@pytest.mark.parametrize("bad", ["1.2", "v1.2.3", "1.2.3rc1", "abc", "1.2.3.4"])
def test_parse_semver_rejects(bad):
    with pytest.raises(SystemExit):
        release.parse_semver(bad)


def test_version_must_be_newer():
    with pytest.raises(SystemExit):
        release.check_version_newer("0.25.0", "0.25.0")
    with pytest.raises(SystemExit):
        release.check_version_newer("0.24.9", "0.25.0")
    release.check_version_newer("0.26.0", "0.25.0")  # no raise


# --- changelog preflight + insertion ---------------------------------------


def test_empty_unreleased_rejected():
    empty = "# Changelog\n\n## [Unreleased]\n\n## [0.25.0] - 2026-08-12 — x\n"
    with pytest.raises(SystemExit):
        release.check_unreleased_nonempty(empty)


def test_missing_unreleased_rejected():
    with pytest.raises(SystemExit):
        release.check_unreleased_nonempty("# Changelog\n\n## [0.25.0] - d — t\n")


def test_nonempty_unreleased_accepted():
    release.check_unreleased_nonempty(CHANGELOG_FIXTURE)


def test_changelog_insertion():
    out = release.insert_changelog_heading(
        CHANGELOG_FIXTURE, "0.26.0", "2026-08-12", "great title"
    )
    lines = out.splitlines()
    idx = lines.index("## [Unreleased]")
    assert lines[idx + 1] == ""
    assert lines[idx + 2] == "## [0.26.0] - 2026-08-12 — great title"
    # previous unreleased body now sits under the new heading
    assert "- A new thing." in out
    assert out.count("## [Unreleased]") == 1


# --- anchored bumps ---------------------------------------------------------


def test_anchored_bump_replaces_exactly_one():
    text = 'version = "0.25.0"\npytest-asyncio>=0.21.0\n'
    out = release.anchored_bump(
        text, 'version = "0.25.0"', 'version = "0.26.0"', "pyproject.toml"
    )
    assert 'version = "0.26.0"' in out
    assert "pytest-asyncio>=0.21.0" in out  # never sweeps dependency pins


def test_anchored_bump_rejects_zero_and_multiple():
    with pytest.raises(SystemExit):
        release.anchored_bump("nothing here", "anchor", "new", "f")
    with pytest.raises(SystemExit):
        release.anchored_bump("anchor anchor", "anchor", "new", "f")


def test_plain_bump_counts():
    out, count = release.plain_bump("a 0.25.0 b 0.25.0", "0.25.0", "0.26.0")
    assert count == 2 and out == "a 0.26.0 b 0.26.0"
    _, zero = release.plain_bump("no version", "0.25.0", "0.26.0")
    assert zero == 0


# --- README rewrite ---------------------------------------------------------


def test_readme_rewrite_promotes_and_demotes():
    out = release.rewrite_readme(README_FIXTURE, "0.25.0", "0.26.0", "- New bullet.\n")
    assert (
        "**v0.26.0 (latest release — [live on PyPI](https://pypi.org/project/stateset-agents/)):**"
        in out
    )
    assert "- New bullet.\n\n**v0.25.0:**" in out
    assert "- Old latest bullet one." in out  # old block content preserved
    assert "# latest release (v0.26.0)" in out
    assert "latest release `v0.26.0`" in out
    assert "v0.23.0" not in out


def test_readme_rewrite_todo_when_no_notes():
    out = release.rewrite_readme(README_FIXTURE, "0.25.0", "0.26.0", None)
    assert "- TODO: describe this release." in out


def test_readme_rewrite_requires_latest_heading():
    with pytest.raises(SystemExit):
        release.rewrite_readme("no heading here", "0.25.0", "0.26.0", None)


# --- build_plan over the fixture repo ---------------------------------------


def test_build_plan_full(fixture_repo: Path):
    plan = release.build_plan(
        fixture_repo, "0.26.0", "the title", "- Bullet.\n", "2026-08-12"
    )
    assert 'version = "0.26.0"' in plan.changes["pyproject.toml"]
    assert "pytest-asyncio>=0.21.0" in plan.changes["pyproject.toml"]
    assert '__version__ = "0.26.0"' in plan.changes["stateset_agents/__init__.py"]
    assert (
        'appVersion: "0.26.0"'
        in plan.changes["deployment/helm/stateset-agents/Chart.yaml"]
    )
    assert "0.26.0" in plan.changes["deployment/helm/stateset-agents/values.yaml"]
    assert "## [0.26.0] - 2026-08-12 — the title" in plan.changes["CHANGELOG.md"]
    assert "**v0.26.0 (latest release" in plan.changes["README.md"]
    # zero-occurrence file warned about, not rewritten
    assert any("deployment/kubernetes/deployment.yaml" in w for w in plan.warnings)
    assert "deployment/kubernetes/deployment.yaml" not in plan.changes


def test_build_plan_rejects_stale_version(fixture_repo: Path):
    with pytest.raises(SystemExit):
        release.build_plan(fixture_repo, "0.25.0", "t", None, "2026-08-12")


def test_build_plan_rejects_empty_unreleased(fixture_repo: Path):
    (fixture_repo / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n## [0.25.0] - d — t\n", encoding="utf-8"
    )
    with pytest.raises(SystemExit):
        release.build_plan(fixture_repo, "0.26.0", "t", None, "2026-08-12")


# --- git preflight (simulated) ----------------------------------------------


def test_preflight_rejects_dirty_tree(monkeypatch):
    def fake_git_output(*args):
        if args[0] == "status":
            return " M some_file.py"
        return "master"

    monkeypatch.setattr(release, "git_output", fake_git_output)
    with pytest.raises(SystemExit, match="not clean"):
        release.preflight_git()


def test_preflight_rejects_non_master(monkeypatch):
    def fake_git_output(*args):
        if args[0] == "status":
            return ""
        return "feature/foo"

    monkeypatch.setattr(release, "git_output", fake_git_output)
    with pytest.raises(SystemExit, match="master"):
        release.preflight_git()


def test_preflight_passes_clean_master(monkeypatch):
    def fake_git_output(*args):
        if args[0] == "status":
            return ""
        return "master"

    monkeypatch.setattr(release, "git_output", fake_git_output)
    release.preflight_git()  # no raise
