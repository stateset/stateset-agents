#!/usr/bin/env python3
"""Codified release procedure for stateset-agents.

Automates the exact manual procedure used for v0.21.0 through v0.25.0:

1. Preflight: clean tree, on master, semver VERSION > current, non-empty
   ``## [Unreleased]`` section in CHANGELOG.md.
2. Anchored version bumps (count must be exactly 1 per anchor — never a blind
   sweep, which once corrupted ``pytest-asyncio>=0.21.0``) in pyproject.toml,
   stateset_agents/__init__.py, and the helm Chart.yaml; plain replacement in
   the deployment/docs files listed in PLAIN_FILES (warn when a file contains
   zero occurrences).
3. CHANGELOG: insert ``## [<new>] - <today> — <TITLE>`` under Unreleased.
4. README: promote the new version into the "What's new" latest-release
   heading, demote the previous block, and refresh the two other
   "latest release" mentions.
5. Guard tests (pytest on the README/pyproject guards).
6. ``git add -u`` + commit + annotated tag.
7. Optional ``--push`` (master + tag) and ``--publish`` (build, twine check,
   twine upload). Publishing reads the PyPI token from the environment
   variable ``STATESET_PYPI_TOKEN`` when set; otherwise it falls back to the
   FIRST LINE of the file ``/home/dom/pypi`` (a local, git-ignored token
   file kept on this machine only).

Default run does neither push nor publish, so it is dry-safe for the remote
world; ``--dry-run`` additionally touches nothing locally and just prints
every change it would make.

Usage:
    python scripts/release.py --version 0.26.0 --title "short release title" \
        [--notes-file NOTES.md] [--push] [--publish] [--dry-run]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPI_TOKEN_FALLBACK_FILE = Path("/home/dom/pypi")

#: Files that get an exact-anchor bump: (relative path, anchor template).
#: ``{v}`` is the current version; each anchor must occur exactly once.
ANCHORED_FILES: list[tuple[str, str]] = [
    ("pyproject.toml", 'version = "{v}"'),
    ("stateset_agents/__init__.py", '__version__ = "{v}"'),
    ("deployment/helm/stateset-agents/Chart.yaml", 'appVersion: "{v}"'),
]

#: Files where every occurrence of the current version becomes the new one.
PLAIN_FILES: list[str] = [
    "deployment/helm/stateset-agents/README.md",
    "deployment/helm/stateset-agents/values.yaml",
    "deployment/kubernetes/glm5-1-training-job.yaml",
    "deployment/kubernetes/glm5-2-training-job.yaml",
    "deployment/kubernetes/kimi-k25-training-job.yaml",
    "deployment/kubernetes/qwen3-5-27b-training-job.yaml",
    "deployment/kubernetes/production-deployment.yaml",
    "deployment/kubernetes/deployment.yaml",
    "docs/ARCHITECTURE.md",
    "docs/KIMI_K25_GKE_AUTOPILOT.md",
]

GUARD_TESTS = [
    "tests/unit/test_readme_onboarding.py",
    "tests/unit/test_pyproject_extras.py",
]

SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


class ReleaseError(SystemExit):
    """Abort with a clear message and non-zero status."""

    def __init__(self, message: str) -> None:
        super().__init__(f"release aborted: {message}")


# ---------------------------------------------------------------------------
# Pure functions (unit-tested against fixture files — no git, no network)
# ---------------------------------------------------------------------------


def parse_semver(version: str) -> tuple[int, int, int]:
    """Parse ``X.Y.Z`` or raise ReleaseError."""
    match = SEMVER_RE.match(version.strip())
    if not match:
        raise ReleaseError(f"VERSION {version!r} is not plain semver (expected X.Y.Z)")
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def check_version_newer(new: str, current: str) -> None:
    """Require ``new`` to be strictly greater than ``current``."""
    if parse_semver(new) <= parse_semver(current):
        raise ReleaseError(
            f"VERSION {new} must be greater than current version {current}"
        )


def check_unreleased_nonempty(changelog_text: str) -> None:
    """Require the ``## [Unreleased]`` section to contain at least one entry line."""
    match = re.search(
        r"^## \[Unreleased\]\s*\n(.*?)(?=^## \[|\Z)",
        changelog_text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise ReleaseError("CHANGELOG.md has no '## [Unreleased]' section")
    body = match.group(1)
    if not any(line.strip() for line in body.splitlines()):
        raise ReleaseError(
            "the '## [Unreleased]' section of CHANGELOG.md is empty — "
            "write the release notes there first"
        )


def anchored_bump(text: str, anchor_old: str, anchor_new: str, label: str) -> str:
    """Replace an exact anchor that must occur exactly once (never a blind sweep)."""
    count = text.count(anchor_old)
    if count != 1:
        raise ReleaseError(
            f"anchor {anchor_old!r} found {count} times in {label} (expected exactly 1)"
        )
    return text.replace(anchor_old, anchor_new)


def plain_bump(text: str, current: str, new: str) -> tuple[str, int]:
    """Replace every occurrence of ``current`` with ``new``; return (text, count)."""
    count = text.count(current)
    return text.replace(current, new), count


def insert_changelog_heading(
    changelog_text: str, new_version: str, date: str, title: str
) -> str:
    """Insert ``## [<new>] - <date> — <title>`` directly under ``## [Unreleased]``."""
    marker = "## [Unreleased]"
    if marker not in changelog_text:
        raise ReleaseError("CHANGELOG.md has no '## [Unreleased]' section")
    heading = f"## [{new_version}] - {date} — {title}"
    return changelog_text.replace(marker, f"{marker}\n\n{heading}", 1)


def rewrite_readme(readme_text: str, current: str, new: str, notes: str | None) -> str:
    """Update README's What's-new section and other 'latest release' mentions.

    - The ``**v<cur> (latest release — [live on PyPI]...)**`` heading becomes
      a ``**v<new> (latest release ...)**`` heading whose bullets are ``notes``
      (or a TODO line), and the old block is demoted to ``**v<cur>:**``.
    - ``# latest release (vX.Y.Z)`` and ``latest release `vX.Y.Z``` mentions
      are refreshed to v<new> whatever version they currently carry.
    """
    latest_re = re.compile(
        r"^\*\*v" + re.escape(current) + r" \(latest release[^\n]*\):\*\*$",
        flags=re.MULTILINE,
    )
    match = latest_re.search(readme_text)
    if match is None:
        raise ReleaseError(
            f"README.md has no '**v{current} (latest release ...):**' heading"
        )
    old_heading = match.group(0)
    new_heading = old_heading.replace(f"v{current}", f"v{new}", 1)
    body = notes.strip() if notes else "- TODO: describe this release."
    replacement = f"{new_heading}\n\n{body}\n\n**v{current}:**"
    readme_text = latest_re.sub(lambda _m: replacement, readme_text, count=1)

    # The two other "latest release" mentions (which may lag behind).
    readme_text = re.sub(
        r"# latest release \(v\d+\.\d+\.\d+\)",
        f"# latest release (v{new})",
        readme_text,
    )
    readme_text = re.sub(
        r"latest release `v\d+\.\d+\.\d+`",
        f"latest release `v{new}`",
        readme_text,
    )
    return readme_text


def read_current_version(pyproject_text: str) -> str:
    """Read the current version from pyproject.toml's [project] table."""
    match = re.search(
        r'^version = "(\d+\.\d+\.\d+)"$', pyproject_text, flags=re.MULTILINE
    )
    if match is None:
        raise ReleaseError("could not find 'version = \"X.Y.Z\"' in pyproject.toml")
    return match.group(1)


# ---------------------------------------------------------------------------
# Plan construction (pure given file contents) + IO shell
# ---------------------------------------------------------------------------


@dataclass
class ReleasePlan:
    """Every file change the release will make, computed before touching disk."""

    new_version: str
    title: str
    changes: dict[str, str] = field(default_factory=dict)  # relpath -> new content
    warnings: list[str] = field(default_factory=list)


def build_plan(
    root: Path,
    new_version: str,
    title: str,
    notes: str | None,
    today: str,
) -> ReleasePlan:
    """Compute all file rewrites. Raises ReleaseError on any anchor violation."""
    pyproject_text = (root / "pyproject.toml").read_text(encoding="utf-8")
    current = read_current_version(pyproject_text)
    check_version_newer(new_version, current)

    changelog_text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    check_unreleased_nonempty(changelog_text)

    plan = ReleasePlan(new_version=new_version, title=title)

    for rel, anchor_tpl in ANCHORED_FILES:
        path = root / rel
        text = path.read_text(encoding="utf-8")
        plan.changes[rel] = anchored_bump(
            text, anchor_tpl.format(v=current), anchor_tpl.format(v=new_version), rel
        )

    for rel in PLAIN_FILES:
        path = root / rel
        if not path.exists():
            plan.warnings.append(f"{rel}: file missing, skipped")
            continue
        text = path.read_text(encoding="utf-8")
        new_text, count = plain_bump(text, current, new_version)
        if count == 0:
            plan.warnings.append(f"{rel}: contained 0 occurrences of {current}")
        else:
            plan.changes[rel] = new_text

    plan.changes["CHANGELOG.md"] = insert_changelog_heading(
        changelog_text, new_version, today, title
    )
    plan.changes["README.md"] = rewrite_readme(
        (root / "README.md").read_text(encoding="utf-8"), current, new_version, notes
    )
    return plan


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=REPO_ROOT, check=True, **kwargs)


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def preflight_git() -> None:
    if git_output("status", "--porcelain"):
        raise ReleaseError("working tree is not clean — commit or stash first")
    branch = git_output("rev-parse", "--abbrev-ref", "HEAD")
    if branch != "master":
        raise ReleaseError(f"must be on master (currently on {branch!r})")


def pypi_token() -> str:
    """PyPI token: env var STATESET_PYPI_TOKEN, else first line of /home/dom/pypi."""
    token = os.environ.get("STATESET_PYPI_TOKEN", "").strip()
    if token:
        return token
    if PYPI_TOKEN_FALLBACK_FILE.exists():
        first_line = (
            PYPI_TOKEN_FALLBACK_FILE.read_text(encoding="utf-8").splitlines()[0].strip()
        )
        if first_line:
            print(f"using PyPI token from {PYPI_TOKEN_FALLBACK_FILE} (fallback)")
            return first_line
    raise ReleaseError(
        "no PyPI token: set STATESET_PYPI_TOKEN or put the token on the "
        f"first line of {PYPI_TOKEN_FALLBACK_FILE}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--version", required=True, help="new semver version, e.g. 0.26.0"
    )
    parser.add_argument(
        "--title", required=True, help="short release title for CHANGELOG/commit"
    )
    parser.add_argument(
        "--notes-file", help="markdown bullets for the README What's-new block"
    )
    parser.add_argument("--push", action="store_true", help="push master and the tag")
    parser.add_argument(
        "--publish", action="store_true", help="build + twine check + upload to PyPI"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print planned changes, touch nothing"
    )
    args = parser.parse_args(argv)

    parse_semver(args.version)
    notes = None
    if args.notes_file:
        notes = Path(args.notes_file).read_text(encoding="utf-8")

    if not args.dry_run:
        preflight_git()

    today = _dt.date.today().isoformat()
    plan = build_plan(REPO_ROOT, args.version, args.title, notes, today)

    for warning in plan.warnings:
        print(f"WARNING: {warning}")

    if args.dry_run:
        print(f"\nDRY RUN — v{plan.new_version} — {plan.title}")
        for rel in plan.changes:
            print(f"would rewrite: {rel}")
        print("would run guard tests:", " ".join(GUARD_TESTS))
        print(
            f"would commit 'chore(release): v{plan.new_version} — {plan.title}' and tag v{plan.new_version}"
        )
        print(f"would push: {args.push}; would publish: {args.publish}")
        return 0

    for rel, content in plan.changes.items():
        (REPO_ROOT / rel).write_text(content, encoding="utf-8")
        print(f"rewrote {rel}")

    run([sys.executable, "-m", "pytest", *GUARD_TESTS, "-q"])

    message = (
        f"chore(release): v{plan.new_version} — {plan.title}\n\n"
        "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
    )
    run(["git", "add", "-u"])
    run(["git", "commit", "-m", message])
    run(
        [
            "git",
            "tag",
            "-a",
            f"v{plan.new_version}",
            "-m",
            f"v{plan.new_version} — {plan.title}",
        ]
    )

    if args.push:
        run(["git", "push", "origin", "master"])
        run(["git", "push", "origin", f"v{plan.new_version}"])
    else:
        print("not pushing (pass --push to push master + tag)")

    if args.publish:
        run([sys.executable, "-m", "build"])
        run([sys.executable, "-m", "twine", "check", "dist/*"])
        env = dict(os.environ, TWINE_USERNAME="__token__", TWINE_PASSWORD=pypi_token())
        print("$ python -m twine upload dist/* (token redacted)")
        subprocess.run(
            [sys.executable, "-m", "twine", "upload", "--skip-existing", "dist/*"],
            cwd=REPO_ROOT,
            check=True,
            env=env,
        )
    else:
        print("not publishing (pass --publish to build + upload to PyPI)")

    print(f"release v{plan.new_version} complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
