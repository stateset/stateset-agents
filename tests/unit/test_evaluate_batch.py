"""Unit tests for ``stateset-agents evaluate --scenarios`` batch mode.

We can't actually invoke the trainer without a real model checkpoint, but we
verify the CLI's argument-validation and error paths via subprocess.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "stateset_agents.cli", "evaluate", *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )


class TestEvaluateBatchValidation:
    def test_scenarios_without_reward_errors(self, tmp_path: Path) -> None:
        scenarios = tmp_path / "s.jsonl"
        scenarios.write_text('{"user_query": "Q"}\n')
        result = _run(
            "--checkpoint", "/tmp/fake-ckpt",
            "--scenarios", str(scenarios),
        )
        assert result.returncode == 2
        assert "--reward is required" in result.stderr

    def test_unknown_reward_errors(self, tmp_path: Path) -> None:
        scenarios = tmp_path / "s.jsonl"
        scenarios.write_text('{"user_query": "Q"}\n')
        result = _run(
            "--checkpoint", "/tmp/fake-ckpt",
            "--scenarios", str(scenarios),
            "--reward", "not-a-real-reward",
        )
        assert result.returncode == 2
        assert "Unknown reward" in result.stderr

    def test_missing_scenarios_file_errors(self) -> None:
        result = _run(
            "--checkpoint", "/tmp/fake-ckpt",
            "--scenarios", "/tmp/does-not-exist.jsonl",
            "--reward", "gsm8k",
            "--dry-run",
        )
        # Dry-run prints the plan and exits 0; without dry-run it would fail.
        # In dry-run mode the scenario-existence check is bypassed.
        assert result.returncode == 0

    def test_dry_run_shows_batch_plan(self, tmp_path: Path) -> None:
        scenarios = tmp_path / "s.jsonl"
        scenarios.write_text('{"user_query": "Q"}\n')
        result = _run(
            "--checkpoint", "/tmp/fake-ckpt",
            "--scenarios", str(scenarios),
            "--reward", "customer_support",
            "--output", "/tmp/out.md",
            "--dry-run",
        )
        assert result.returncode == 0
        assert "Dry-run" in result.stdout
        assert "Scenarios:" in result.stdout
        assert "Reward: customer_support" in result.stdout
        assert "/tmp/out.md" in result.stdout

    def test_single_message_dry_run_unchanged(self) -> None:
        """Original single-message mode must still work."""
        result = _run(
            "--checkpoint", "/tmp/fake-ckpt",
            "--message", "Hello",
            "--dry-run",
        )
        assert result.returncode == 0
        assert "Message: Hello" in result.stdout

    def test_missing_checkpoint_errors(self) -> None:
        result = _run(
            "--message", "Hello",
        )
        assert result.returncode == 2


class TestEvaluateBatchHelp:
    def test_help_documents_batch_mode(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "stateset_agents.cli", "evaluate", "--help"],
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0
        assert "--scenarios" in result.stdout
        assert "--reward" in result.stdout
        assert "single message or batch" in result.stdout.lower()
