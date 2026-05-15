"""Unit tests for the ``stateset-agents recipe`` cookbook accessor."""

from __future__ import annotations

import subprocess
import sys


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "stateset_agents.cli", "recipe", *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )


class TestRecipeListing:
    def test_list_default_with_no_args(self) -> None:
        result = _run("list")
        assert result.returncode == 0
        assert "Available cookbook recipes:" in result.stdout
        # All 8 recipes should appear (Recipe 5 — batch eval — added in v0.12.0)
        for keyword in ("first-fine-tune", "iterate-from", "reproduce", "tool-using", "batch", "debug", "hand-off", "demos"):
            assert keyword in result.stdout, f"missing recipe with keyword {keyword!r}"

    def test_default_arg_is_list(self) -> None:
        # No argument defaults to "list" (the Argument default).
        result = _run()
        assert result.returncode == 0
        assert "Available cookbook recipes:" in result.stdout


class TestRecipeLookup:
    def test_by_number(self) -> None:
        result = _run("1")
        assert result.returncode == 0
        assert "Recipe 1" in result.stdout
        assert "first fine-tune" in result.stdout.lower()

    def test_by_full_slug(self) -> None:
        result = _run("your-first-fine-tune-in-4-hours")
        assert result.returncode == 0
        assert "Recipe 1" in result.stdout

    def test_by_substring(self) -> None:
        result = _run("iterate")
        assert result.returncode == 0
        assert "Recipe 2" in result.stdout

    def test_by_short_substring(self) -> None:
        # After 0.12.0 added "Run a batch evaluation" as Recipe 5,
        # debug-a-stuck-reward is now Recipe 6.
        result = _run("debug")
        assert result.returncode == 0
        assert "Recipe 6" in result.stdout

    def test_unknown_recipe_exits_2(self) -> None:
        result = _run("does-not-exist")
        assert result.returncode == 2
        assert "No recipe matches" in result.stderr

    def test_recipe_includes_workflow_commands(self) -> None:
        """A recipe should contain at least one bash code block."""
        result = _run("1")
        assert result.returncode == 0
        assert "```bash" in result.stdout
        assert "stateset-agents" in result.stdout
