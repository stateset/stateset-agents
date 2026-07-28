"""Unit tests for ``stateset-agents chat`` — the interactive REPL CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run_chat(input_text: str, *args: str) -> subprocess.CompletedProcess:
    """Run the chat command with stdin = input_text. Caller checks the result."""
    return subprocess.run(
        [sys.executable, "-m", "stateset_agents.cli", "chat", *args],
        input=input_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=30,
    )


class TestChatStubMode:
    def test_stub_agent_quits_cleanly(self) -> None:
        result = _run_chat("/quit\n")
        assert result.returncode == 0
        assert "Initializing agent" in result.stdout
        assert "Interactive Chat" in result.stdout
        assert "Bye." in result.stdout

    def test_eof_exits_cleanly(self) -> None:
        # Ctrl+D / empty stdin should still exit 0.
        result = _run_chat("")
        assert result.returncode == 0
        assert "Bye." in result.stdout

    def test_message_generates_response(self) -> None:
        result = _run_chat("hello\n/quit\n")
        assert result.returncode == 0
        # Stub model produces some response text after "agent>".
        assert "agent>" in result.stdout

    def test_reset_clears_history(self) -> None:
        result = _run_chat("first message\n/reset\nsecond message\n/quit\n")
        assert result.returncode == 0
        assert "(conversation reset)" in result.stdout

    def test_system_prompt_accepted(self) -> None:
        result = _run_chat(
            "/quit\n",
            "--system",
            "You are a helpful customer support agent.",
        )
        assert result.returncode == 0
        assert "Initializing agent" in result.stdout

    def test_help_shows_examples(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "stateset_agents.cli", "chat", "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=10,
        )
        assert result.returncode == 0
        assert "/reset" in result.stdout
        assert "/quit" in result.stdout


class TestChatHistoryAndReplay:
    def test_history_appends_each_turn(self, tmp_path: Path) -> None:
        hist = tmp_path / "convo.jsonl"
        result = _run_chat(
            "hello there\n/quit\n",
            "--history",
            str(hist),
        )
        assert result.returncode == 0
        assert hist.exists()
        lines = [
            __import__("json").loads(line)
            for line in hist.read_text().splitlines()
            if line.strip()
        ]
        # One user + one assistant turn.
        assert len(lines) == 2
        roles = [line["role"] for line in lines]
        assert "user" in roles
        assert "assistant" in roles

    def test_history_path_created_if_missing(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested" / "dir" / "convo.jsonl"
        result = _run_chat(
            "ping\n/quit\n",
            "--history",
            str(nested),
        )
        assert result.returncode == 0
        assert nested.exists()

    def test_replay_preloads_conversation(self, tmp_path: Path) -> None:
        # Pre-seed a transcript.
        hist = tmp_path / "seed.jsonl"
        hist.write_text(
            '{"role": "user", "content": "previous question"}\n'
            '{"role": "assistant", "content": "previous answer"}\n'
        )
        result = _run_chat(
            "/quit\n",
            "--replay",
            str(hist),
        )
        assert result.returncode == 0
        assert "Loaded 2 turn(s)" in result.stdout

    def test_replay_missing_file_exits_2(self, tmp_path: Path) -> None:
        result = _run_chat(
            "/quit\n",
            "--replay",
            "/tmp/does/not/exist.jsonl",
        )
        assert result.returncode == 2
        assert "Replay path not found" in result.stderr


class TestChatLiveGrading:
    def test_unknown_grade_exits_2(self) -> None:
        result = _run_chat("/quit\n", "--grade", "not-a-real-reward")
        assert result.returncode == 2
        assert "Unknown --grade reward" in result.stderr

    def test_customer_support_grade_runs_and_prints_score(self) -> None:
        result = _run_chat("hello\n/quit\n", "--grade", "customer_support")
        assert result.returncode == 0
        assert "Live grading enabled: customer_support" in result.stdout
        assert "reward[customer_support]" in result.stdout

    def test_gsm8k_grade_runs(self) -> None:
        result = _run_chat("what is 2+2?\n/quit\n", "--grade", "gsm8k")
        assert result.returncode == 0
        assert "reward[gsm8k]" in result.stdout

    def test_tool_calling_grade_runs(self) -> None:
        result = _run_chat("calculate 5*5\n/quit\n", "--grade", "tool_calling")
        assert result.returncode == 0
        assert "reward[tool_calling]" in result.stdout

    def test_grade_marker_changes_with_score(self) -> None:
        """The output uses ✅/⚠️ /❌ markers based on score thresholds."""
        # GSM8K reward on a stub response with no number → score = 0.0 → ❌
        result = _run_chat("hello there\n/quit\n", "--grade", "gsm8k")
        assert result.returncode == 0
        # Stub response won't contain a number, so gsm8k should mark ❌.
        assert "❌" in result.stdout


class TestChatCheckpointValidation:
    def test_missing_checkpoint_path_exits_2(self) -> None:
        result = _run_chat(
            "/quit\n",
            "--checkpoint",
            "/tmp/this/definitely/does/not/exist",
        )
        assert result.returncode == 2
        assert "Checkpoint path not found" in result.stderr

    def test_existing_checkpoint_path_accepted(self, tmp_path: Path) -> None:
        # Stub mode skips the actual LoRA load, but the path existence check
        # should still pass.
        ckpt = tmp_path / "adapter"
        ckpt.mkdir()
        result = _run_chat(
            "/quit\n",
            "--checkpoint",
            str(ckpt),
        )
        assert result.returncode == 0
