"""Every ``stateset-agents ...`` snippet in the docs must be a real command.

The docs are the first thing a new user runs. A snippet that names a
subcommand or a flag the CLI does not have is a broken promise, so this test
executes ``--help`` for the deepest subcommand path each snippet names and
asserts every long flag it uses appears in that help text.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
from functools import cache

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
        for lineno, line in _logical_lines((ROOT / name).read_text(encoding="utf-8")):
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


# Rich styles help text token-by-token when colour is forced, so a flag can
# come back as ``\x1b[1m--\x1b[0m\x1b[1mformat\x1b[0m`` — no longer a plain
# substring. GitHub Actions exports ``FORCE_COLOR=1`` for every step, which is
# why this test passed locally and failed on every snippet in CI. Belt and
# braces: drop the colour-forcing variables from the child environment *and*
# strip any escape sequences that survive.
_COLOR_FORCING_VARS = ("FORCE_COLOR", "CLICOLOR_FORCE", "CLICOLOR")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _run_help(path: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in _COLOR_FORCING_VARS}
    env.update({"COLUMNS": "200", "NO_COLOR": "1", "TERM": "dumb"})
    proc = subprocess.run(
        [sys.executable, "-m", "stateset_agents", *path, "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=ROOT,
        env=env,
    )
    return subprocess.CompletedProcess(
        proc.args,
        proc.returncode,
        _ANSI.sub("", proc.stdout),
        _ANSI.sub("", proc.stderr),
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


@cache
def _command_options(path: tuple[str, ...]) -> dict[str, dict[str, object]]:
    """Read exact Click metadata in the same fresh process as the real CLI."""
    probe = r"""
import json
import sys

from typer.main import get_command

from stateset_agents.cli import app

command = get_command(app)
for word in json.loads(sys.argv[1]):
    if not hasattr(command, "get_command"):
        break
    command = command.get_command(None, word)
    if command is None:
        raise SystemExit(f"command not found: {word}")

options = {}
for parameter in command.params:
    if hasattr(parameter, "is_flag"):
        metadata = {
            "is_flag": parameter.is_flag,
            "secondary_opts": list(getattr(parameter, "secondary_opts", ())),
        }
        for option in (*parameter.opts, *metadata["secondary_opts"]):
            options[option] = metadata
print(json.dumps(options, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, json.dumps(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


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

    options = _command_options(path)
    for flag in sorted(flags):
        assert flag in options, f"{where} — unknown flag {flag} for `{' '.join(path)}`"

    # Boolean flags (rendered as `--flag  --no-flag`) take no value: a snippet
    # writing `--dry-run false` would silently pass `false` as a positional.
    for idx, word in enumerate(words):
        if not word.startswith("--") or "=" in word:
            continue
        option = options.get(word)
        if option is None or not option["is_flag"]:
            continue
        nxt = words[idx + 1] if idx + 1 < len(words) else None
        alternative = next(
            iter(option["secondary_opts"]), f"omit the value after {word}"
        )
        assert nxt is None or nxt.startswith("-"), (
            f"{where} — {word} is a boolean flag; it takes no value "
            f"(got {nxt!r}; use {alternative})"
        )


def test_expected_snippet_count() -> None:
    """Guard against the extractor silently matching nothing."""
    per_doc = {name: sum(1 for s, _, _ in SNIPPETS if s == name) for name in DOCS}
    # QUICKSTART was simplified in the "golden path" update; it now contains
    # fewer direct CLI snippets. Keep this guard aligned with the current doc.
    assert per_doc == {"README.md": 25, "QUICKSTART.md": 2}, per_doc


def test_help_stays_plain_when_the_environment_forces_color(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CI exports FORCE_COLOR=1; the help probe must still be plain text.

    Without this, rich splits every flag across ANSI escapes and each snippet
    assertion fails with a bogus "unknown flag" message.
    """
    monkeypatch.setenv("FORCE_COLOR", "1")
    result = _run_help(("ingest",))
    assert result.returncode == 0, result.stderr
    assert "\x1b[" not in result.stdout
    assert "--format" in result.stdout


def test_long_options_are_read_from_click_metadata_without_help_truncation() -> None:
    """Rich may abbreviate long flags on narrow Windows consoles."""
    options = _command_options(("train-remote",))

    assert "--container-disk-gb" in options
    assert "--gradient-accumulation-steps" in options
