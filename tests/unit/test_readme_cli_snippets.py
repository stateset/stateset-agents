"""Every ``stateset-agents ...`` snippet in the docs must be a real command.

The docs are the first thing a new user runs. A snippet that names a
subcommand or a flag the CLI does not have is a broken promise, so this test
executes ``--help`` for the deepest subcommand path each snippet names and
asserts every long flag it uses appears in that help text.
"""

from __future__ import annotations

import os
import pathlib
import re
import shlex
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCS = ("README.md", "QUICKSTART.md")
SNIPPET = re.compile(r"^stateset-agents\s+(.+)$")


def _logical_lines(text: str) -> list[tuple[int, str]]:
    """Join backslash continuations into single logical lines."""
    out: list[tuple[int, str]] = []
    buf: list[str] = []
    start = 0
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        if not buf:
            start = lineno
        if line.endswith("\\"):
            buf.append(line[:-1].strip())
            continue
        if buf:
            buf.append(line.strip())
            out.append((start, " ".join(p for p in buf if p)))
            buf = []
        else:
            out.append((lineno, line))
    if buf:
        out.append((start, " ".join(p for p in buf if p)))
    return out


def _snippets() -> list[tuple[str, int, str]]:
    found: list[tuple[str, int, str]] = []
    for name in DOCS:
        for lineno, line in _logical_lines((ROOT / name).read_text()):
            m = SNIPPET.match(line)
            if not m:
                continue
            body = m.group(1).split(" #")[0].strip()
            if body:
                found.append((name, lineno, body))
    return found


SNIPPETS = _snippets()


def _is_subcommand(word: str) -> bool:
    if word.startswith("-"):
        return False
    if word.startswith(("/", "./", "~")) or "/" in word or "." in word:
        return False
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", word))


def _run_help(path: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "stateset_agents", *path, "--help"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**os.environ, "COLUMNS": "200", "NO_COLOR": "1"},
    )


_HELP_CACHE: dict[tuple[str, ...], subprocess.CompletedProcess[str]] = {}


def _help(path: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    if path not in _HELP_CACHE:
        _HELP_CACHE[path] = _run_help(path)
    return _HELP_CACHE[path]


def _shlex(line: str) -> list[str]:
    try:
        return shlex.split(line)
    except ValueError:  # pragma: no cover - unbalanced quotes in docs
        return line.split()


@pytest.mark.parametrize(
    "src,lineno,line",
    SNIPPETS,
    ids=[f"{s}:{n}" for s, n, _ in SNIPPETS],
)
def test_readme_command_flags_exist(src: str, lineno: int, line: str) -> None:
    words = _shlex(line)
    flags = {w.split("=")[0] for w in words if w.startswith("--")}

    # Descend at most two levels of subcommand, keeping the deepest that works.
    path: tuple[str, ...] = ()
    for word in words[:2]:
        if not _is_subcommand(word):
            break
        candidate = path + (word,)
        if _help(candidate).returncode != 0:
            break
        path = candidate

    result = _help(path)
    where = f"{src}:{lineno}: `{line}`"
    assert result.returncode == 0, f"{where} — {result.stderr[-400:]}"
    assert path, f"{where} — no valid subcommand found"

    help_text = result.stdout
    for flag in sorted(flags):
        assert flag in help_text, (
            f"{where} — unknown flag {flag} for `{' '.join(path)}`"
        )

    # Boolean flags (rendered as `--flag  --no-flag`) take no value: a snippet
    # writing `--dry-run false` would silently pass `false` as a positional.
    for idx, word in enumerate(words):
        if not word.startswith("--") or "=" in word:
            continue
        negated = "--no-" + word[2:]
        if negated not in help_text:
            continue
        nxt = words[idx + 1] if idx + 1 < len(words) else None
        assert nxt is None or nxt.startswith("-"), (
            f"{where} — {word} is a boolean flag; it takes no value "
            f"(got {nxt!r}, use {negated} instead)"
        )


def test_expected_snippet_count() -> None:
    """Guard against the extractor silently matching nothing."""
    per_doc = {name: sum(1 for s, _, _ in SNIPPETS if s == name) for name in DOCS}
    assert per_doc == {"README.md": 21, "QUICKSTART.md": 6}, per_doc
