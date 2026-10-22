"""Every CLI command a user-facing hint names must exist."""

from __future__ import annotations

import re
from pathlib import Path

import typer

import stateset_agents
from stateset_agents.cli import app

PACKAGE = Path(stateset_agents.__file__).resolve().parent
HINT = re.compile(r"`stateset-agents ([^`]+)`")
WORD = re.compile(r"[a-z][a-z0-9-]*")


def _hints() -> list[tuple[str, str]]:
    found = []
    for path in sorted(PACKAGE.rglob("*.py")):
        for match in HINT.finditer(path.read_text(encoding="utf-8")):
            found.append((f"{path.relative_to(PACKAGE.parent)}", match.group(1)))
    return found


def _resolve(root: object, words: list[str]) -> str | None:
    """Return why the command path does not exist, or ``None``."""
    command = root
    for word in words:
        if not WORD.fullmatch(word):
            break  # an option, placeholder or argument ends the command path
        children = getattr(command, "commands", None)
        if not isinstance(children, dict):
            break  # a leaf command's positional argument
        child = children.get(word)
        if child is None:
            return f"no command {word!r} under {getattr(command, 'name', None)!r}"
        command = child
    return None if command is not root else "names no command"


def test_cli_hints_in_the_package_name_real_commands():
    root = typer.main.get_command(app)
    hints = _hints()
    assert hints, "expected backtick-quoted CLI hints in the package"
    broken = [
        f"{where}: `stateset-agents {hint}` ({reason})"
        for where, hint in hints
        # A template such as `stateset-agents {command_name}` is filled at run
        # time from registered commands and cannot be checked statically.
        if WORD.fullmatch(hint.split()[0])
        and (reason := _resolve(root, hint.split())) is not None
    ]
    assert broken == []


def test_resolver_rejects_a_misplaced_command():
    root = typer.main.get_command(app)
    assert _resolve(root, ["remote", "runpod-orphans"]) is not None
    assert _resolve(root, ["runpod-orphans", "--terminate"]) is None
