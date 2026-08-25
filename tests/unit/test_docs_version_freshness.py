"""Every wheel-filename version mention under docs/ and README.md must match
the current package version.

A stale ``stateset_agents-<old-ver>-py3-none-any.whl`` in a doc example is a
silent lie: it looks copy-pasteable but names a wheel that was never built
for this release. ``scripts/release.py`` rewrites ``PLAIN_FILES`` on every
release; this test is the tripwire for anything that drifts anyway (a new
doc that mentions a wheel filename but isn't wired into PLAIN_FILES, or a
one-off mention that predates the file's addition to that list).

Historical run logs are the exception: they record the wheel that a past
experiment actually ran on, so bumping them would make the record false.
Mark them with the HTML comment ``<!-- historical: do not bump -->`` --
on its own line directly above the mention to exempt that one line, or as
the very first line of the file to exempt the whole file -- and keep the
file out of ``release.py``'s ``PLAIN_FILES``.
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


HISTORICAL_MARKER = "<!-- historical: do not bump -->"


def _stale_mentions(text: str, current: str, label: str) -> list[str]:
    """Wheel mentions in ``text`` naming something other than ``current``.

    Lines exempted by ``HISTORICAL_MARKER`` (and every line of a file whose
    first line is the marker) are skipped.
    """
    lines = text.splitlines()
    if lines and lines[0].strip() == HISTORICAL_MARKER:
        return []
    exempt: set[int] = set()
    for index, line in enumerate(lines):
        if line.strip() != HISTORICAL_MARKER:
            continue
        for following in range(index + 1, len(lines)):
            if lines[following].strip():
                exempt.add(following + 1)
                break

    stale: list[str] = []
    for match in WHEEL_RE.finditer(text):
        version = match.group(1)
        if version == current:
            continue
        lineno = text.count("\n", 0, match.start()) + 1
        if lineno in exempt:
            continue
        stale.append(f"{label}:{lineno}: wheel version {version} != {current}")
    return stale


def test_wheel_version_mentions_match_current_version():
    current = stateset_agents.__version__
    stale: list[str] = []
    for path in _doc_files():
        stale.extend(
            _stale_mentions(
                path.read_text(encoding="utf-8"), current, str(path.relative_to(ROOT))
            )
        )
    assert not stale, "stale wheel version mentions found:\n" + "\n".join(stale)


def test_marker_exempts_the_next_line():
    text = (
        "intro\n"
        f"{HISTORICAL_MARKER}\n"
        "`wheel=dist/stateset_agents-0.26.0-py3-none-any.whl`\n"
    )
    assert _stale_mentions(text, "9.9.9", "run-log.md") == []


def test_marker_exempts_only_the_line_it_marks():
    text = (
        "intro\n"
        f"{HISTORICAL_MARKER}\n"
        "stateset_agents-0.26.0-py3-none-any.whl\n"
        "stateset_agents-0.27.0-py3-none-any.whl\n"
    )
    stale = _stale_mentions(text, "9.9.9", "run-log.md")
    assert len(stale) == 1 and "0.27.0" in stale[0]


def test_marker_on_the_first_line_exempts_the_whole_file():
    text = (
        f"{HISTORICAL_MARKER}\n"
        "# A historical run log\n"
        "stateset_agents-0.26.0-py3-none-any.whl\n"
        "stateset_agents-0.27.0-py3-none-any.whl\n"
    )
    assert _stale_mentions(text, "9.9.9", "run-log.md") == []


def test_unmarked_stale_mentions_are_still_reported():
    text = "stateset_agents-0.26.0-py3-none-any.whl\n"
    assert len(_stale_mentions(text, "9.9.9", "doc.md")) == 1


def test_flywheel_experiment_is_marked_historical():
    """The run log records the wheel that actually ran; do not bump it."""
    text = (ROOT / "docs" / "FLYWHEEL_EXPERIMENT.md").read_text(encoding="utf-8")
    assert HISTORICAL_MARKER in text
    assert "stateset_agents-0.26.0-py3-none-any.whl" in text
