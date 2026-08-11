"""Guardrail: docs/CLI_REFERENCE.md must document every registered CLI command.

Same spirit as test_examples_readme_complete.py — the reference is maintained
by hand, so this test is what keeps it from drifting when commands are added,
renamed, or removed.
"""

from __future__ import annotations

import re
from pathlib import Path

import typer

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_REFERENCE = REPO_ROOT / "docs" / "CLI_REFERENCE.md"

# Commands that register only when optional extras are installed. They must
# stay documented even when absent from the runtime app in a minimal env.
LAZILY_REGISTERED = {"advanced"}


def _registered_commands() -> set[str]:
    from stateset_agents.cli import app

    return set(typer.main.get_command(app).commands)


def _documented_commands() -> set[str]:
    text = CLI_REFERENCE.read_text(encoding="utf-8")
    return set(re.findall(r"^### `stateset-agents ([a-z0-9-]+)`", text, re.MULTILINE))


def test_every_cli_command_is_documented() -> None:
    missing = _registered_commands() - _documented_commands()
    assert not missing, (
        f"docs/CLI_REFERENCE.md is missing a section for: {sorted(missing)}. "
        "Add a '### `stateset-agents <command>`' section describing the "
        "command and its options."
    )


def test_no_stale_commands_documented() -> None:
    stale = _documented_commands() - _registered_commands() - LAZILY_REGISTERED
    assert not stale, (
        f"docs/CLI_REFERENCE.md documents commands that no longer exist: "
        f"{sorted(stale)}. Remove their sections (or, if the command is "
        "lazily registered behind an extra, add it to LAZILY_REGISTERED in "
        "this test)."
    )
