"""Every wheel-filename version mention under docs/ and README.md must match
the current package version.

A stale ``stateset_agents-<old-ver>-py3-none-any.whl`` in a doc example is a
silent lie: it looks copy-pasteable but names a wheel that was never built
for this release. ``scripts/release.py`` rewrites ``PLAIN_FILES`` on every
release; this test is the tripwire for anything that drifts anyway (a new
doc that mentions a wheel filename but isn't wired into PLAIN_FILES, or a
one-off mention that predates the file's addition to that list).
"""

from __future__ import annotations

import pathlib
import re

import stateset_agents

ROOT = pathlib.Path(__file__).resolve().parents[2]
WHEEL_RE = re.compile(r"stateset_agents-(\d+\.\d+\.\d+(?:[.\w-]*)?)-py3-none-any\.whl")


def _doc_files() -> list[pathlib.Path]:
    files = [ROOT / "README.md"]
    files.extend(sorted((ROOT / "docs").rglob("*.md")))
    return [f for f in files if f.is_file()]


def test_wheel_version_mentions_match_current_version():
    current = stateset_agents.__version__
    stale: list[str] = []
    for path in _doc_files():
        text = path.read_text(encoding="utf-8")
        for match in WHEEL_RE.finditer(text):
            version = match.group(1)
            if version != current:
                lineno = text.count("\n", 0, match.start()) + 1
                stale.append(f"{path.relative_to(ROOT)}:{lineno}: wheel version {version} != {current}")
    assert not stale, "stale wheel version mentions found:\n" + "\n".join(stale)
