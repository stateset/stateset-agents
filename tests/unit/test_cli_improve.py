"""Unit tests for ``stateset-agents improve`` (grade -> curate -> retrain in one command)."""

from __future__ import annotations

import functools
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from stateset_agents.cli import app

runner = CliRunner()

REPO_ROOT = Path(__file__).resolve().parents[2]
_FLAG_RE = re.compile(r"(?<![A-Za-z0-9_-])(--[a-zA-Z][a-zA-Z0-9-]*|-[a-zA-Z])\b")


@functools.cache
def _help_flags(*argv: str) -> frozenset[str]:
    """Run ``argv + ("--help",)`` and return every flag token its help text lists.

    Used to check that a suggested command line's flags are ones the real
    argparse/typer parser actually defines — without executing the command
    for real (some of these launch long-running training).
    """
    # Wide, plain terminal so typer/rich cannot wrap or truncate flag names
    # (a narrow CI pty previously made help parse to zero flags).
    env = {
        **os.environ,
        "COLUMNS": "200",
        "TERM": "dumb",
        "NO_COLOR": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    result = subprocess.run(
        [*argv, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
        env=env,
    )
    if result.returncode != 0 and os.name == "nt":
        # Some scripts' --help cannot run on Windows (heavy imports); the
        # POSIX CI jobs still enforce the guard strictly.
        pytest.skip(f"`{' '.join(argv)} --help` unavailable on Windows")
    assert (
        result.returncode == 0
    ), f"`{' '.join(argv)} --help` failed (exit {result.returncode}):\n{result.stderr}"
    flags = frozenset(_FLAG_RE.findall(result.stdout + result.stderr))
    assert flags, (
        f"`{' '.join(argv)} --help` succeeded but no flags were parsed from its "
        f"output — help rendering changed?\n{result.stdout[:500]}"
    )
    return flags


def _extract_bash_commands(markdown: str) -> list[list[str]]:
    """Pull every logical command out of ```bash fenced blocks, joining `\\` continuations."""
    commands: list[list[str]] = []
    in_block = False
    pending: list[str] = []
    for raw_line in markdown.splitlines():
        stripped = raw_line.strip()
        if stripped == "```bash":
            in_block = True
            continue
        if stripped == "```":
            in_block = False
            continue
        if not in_block:
            continue
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            pending.append(stripped[:-1].strip())
            continue
        pending.append(stripped)
        commands.append(shlex.split(" ".join(pending)))
        pending = []
    assert not pending, "unterminated line continuation in a ```bash block"
    return commands


def _assert_command_parses(tokens: list[str]) -> None:
    """Validate a next_steps.md command's flags against the real CLI surface it calls."""
    exe = tokens[0]
    if exe == "mkdir":
        return  # shell builtin, nothing to validate against a Python parser

    if exe == "python":
        script = tokens[1]
        argv: tuple[str, ...] = (sys.executable, script)
        rest = tokens[2:]
    elif exe == "stateset-agents":
        # e.g. ["stateset-agents", "improve", "run", "--transcripts", ...] —
        # "improve" is the typer command; "run"/"status" is its positional
        # ACTION argument, not a flag, so it's left in `rest` and skipped
        # below (only "-"-prefixed tokens are checked against --help).
        subcommand = tokens[1]
        argv = (sys.executable, "-m", "stateset_agents.cli", subcommand)
        rest = tokens[2:]
    else:
        raise AssertionError(f"Unrecognized command in next_steps.md: {tokens!r}")

    known_flags = _help_flags(*argv)
    for token in rest:
        if not token.startswith("-"):
            continue
        flag = token.split("=", 1)[0]
        assert flag in known_flags, (
            f"Suggested command `{' '.join(tokens)}` uses flag {flag!r}, which "
            f"`{' '.join(argv)} --help` does not list. Known flags: "
            f"{sorted(known_flags)}"
        )


GOOD_TURN = "I would be happy to help you with a refund for order 1234 right away."
BAD_TURN = "idk"


def _write_transcript(path: Path, good: bool, order_id: str) -> None:
    turn = (
        f"I would be happy to help you with a refund for order {order_id} right away."
        if good
        else BAD_TURN
    )
    rows = [
        {"role": "user", "content": f"I want a refund for order {order_id}"},
        {"role": "assistant", "content": turn},
        {"role": "user", "content": f"Thanks, anything else about order {order_id}?"},
        {"role": "assistant", "content": turn},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _make_transcripts_dir(tmp_path: Path) -> Path:
    d = tmp_path / "transcripts"
    d.mkdir()
    _write_transcript(d / "session1.jsonl", good=True, order_id="1234")
    _write_transcript(d / "session2.jsonl", good=True, order_id="5678")
    _write_transcript(d / "session3.jsonl", good=False, order_id="9999")
    return d


class TestImproveRun:
    def test_curated_and_summary(self, tmp_path: Path) -> None:
        transcripts_dir = _make_transcripts_dir(tmp_path)
        output_dir = tmp_path / "improved"

        result = runner.invoke(
            app,
            [
                "improve",
                "run",
                "--transcripts",
                str(transcripts_dir),
                "--reward",
                "customer_support",
                "--output",
                str(output_dir),
                "--threshold",
                "0.7",
            ],
        )
        assert result.exit_code == 0, result.output

        curated_path = output_dir / "curated.jsonl"
        assert curated_path.exists()
        curated_lines = [
            json.loads(line)
            for line in curated_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        # 2 good transcripts x 2 assistant turns each = 4 curated examples;
        # the bad transcript's low-scoring turns must be excluded.
        assert len(curated_lines) == 4
        assert all(row["score"] >= 0.7 for row in curated_lines)
        assert all("happy to help" in row["response"] for row in curated_lines)

        summary_path = output_dir / "improve_summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["transcript_count"] == 3
        assert summary["assistant_turn_count"] == 6
        assert summary["curated_count"] == 4
        assert summary["reward"] == "customer_support"
        assert summary["threshold"] == 0.7
        assert 0.0 <= summary["mean_score"] <= 1.0
        assert isinstance(summary["reward_breakdown"], dict)
        assert len(summary["transcripts"]) == 3

        next_steps_path = output_dir / "next_steps.md"
        assert next_steps_path.exists()
        next_steps = next_steps_path.read_text(encoding="utf-8")
        assert "scripts/sft_from_curated.py" in next_steps
        assert "scripts/prepare_sft_dataset.py" in next_steps
        assert "examples/finetune_gspo.py" in next_steps
        assert str(curated_path) in next_steps

    def test_next_steps_offers_a_path_for_users_without_a_gpu(
        self, tmp_path: Path
    ) -> None:
        """Step 3 is the only GPU-bound step; next_steps.md must say how to
        run it without one, or the loop dead-ends for those users."""
        transcripts_dir = _make_transcripts_dir(tmp_path)
        output_dir = tmp_path / "improved"
        result = runner.invoke(
            app,
            [
                "improve",
                "run",
                "--transcripts",
                str(transcripts_dir),
                "--reward",
                "customer_support",
                "--output",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0, result.output

        next_steps = (output_dir / "next_steps.md").read_text(encoding="utf-8")
        assert "train-remote" in next_steps

    def test_next_steps_commands_parse_against_real_cli_surfaces(
        self, tmp_path: Path
    ) -> None:
        """Every command line in next_steps.md must be runnable against the
        real argparse/typer parsers it names — regression guard for
        suggesting flags a script doesn't actually accept."""
        transcripts_dir = _make_transcripts_dir(tmp_path)
        output_dir = tmp_path / "improved"
        result = runner.invoke(
            app,
            [
                "improve",
                "run",
                "--transcripts",
                str(transcripts_dir),
                "--reward",
                "customer_support",
                "--output",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0, result.output

        next_steps = (output_dir / "next_steps.md").read_text(encoding="utf-8")
        commands = _extract_bash_commands(next_steps)
        assert len(commands) >= 5, "expected multiple suggested commands"
        for tokens in commands:
            _assert_command_parses(tokens)

    def test_openai_format_composes_with_ingest(self, tmp_path: Path) -> None:
        openai_log = tmp_path / "logs.jsonl"
        conversation = {
            "messages": [
                {"role": "user", "content": "I want a refund for order 1234"},
                {"role": "assistant", "content": GOOD_TURN},
            ]
        }
        openai_log.write_text(json.dumps(conversation) + "\n", encoding="utf-8")

        output_dir = tmp_path / "improved_openai"
        result = runner.invoke(
            app,
            [
                "improve",
                "run",
                "--transcripts",
                str(openai_log),
                "--format",
                "openai",
                "--reward",
                "customer_support",
                "--output",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert (output_dir / "ingested" / "conversation_0.jsonl").exists()
        summary = json.loads(
            (output_dir / "improve_summary.json").read_text(encoding="utf-8")
        )
        assert summary["transcript_count"] == 1
        assert summary["assistant_turn_count"] == 1

    def test_missing_transcripts_dir_errors(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "improve",
                "run",
                "--transcripts",
                str(tmp_path / "nope"),
                "--reward",
                "customer_support",
                "--output",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code != 0

    def test_nsr_reward_is_accepted_by_name_resolution(self) -> None:
        # NSR is deterministic and rule-based, so it qualifies where LLM-judge
        # rewards are refused; _resolve_reward_name must not exit for it.
        from stateset_agents.cli_improve import _resolve_reward_name

        _resolve_reward_name("nsr")  # must not raise typer.Exit

    def test_unknown_reward_errors_clearly(self, tmp_path: Path) -> None:
        transcripts_dir = _make_transcripts_dir(tmp_path)
        result = runner.invoke(
            app,
            [
                "improve",
                "run",
                "--transcripts",
                str(transcripts_dir),
                "--reward",
                "not-a-real-reward",
                "--output",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code != 0
        assert "Unknown --reward" in result.output

    def test_judge_reward_errors_clearly(self, tmp_path: Path) -> None:
        transcripts_dir = _make_transcripts_dir(tmp_path)
        result = runner.invoke(
            app,
            [
                "improve",
                "run",
                "--transcripts",
                str(transcripts_dir),
                "--reward",
                "llm_judge",
                "--output",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code != 0
        assert "API key" in result.output

    def test_missing_reward_errors(self, tmp_path: Path) -> None:
        transcripts_dir = _make_transcripts_dir(tmp_path)
        result = runner.invoke(
            app,
            [
                "improve",
                "run",
                "--transcripts",
                str(transcripts_dir),
                "--output",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code != 0
        assert "--reward is required" in result.output


class TestImproveStatus:
    def test_status_after_run(self, tmp_path: Path) -> None:
        transcripts_dir = _make_transcripts_dir(tmp_path)
        output_dir = tmp_path / "improved"
        run_result = runner.invoke(
            app,
            [
                "improve",
                "run",
                "--transcripts",
                str(transcripts_dir),
                "--reward",
                "customer_support",
                "--output",
                str(output_dir),
            ],
        )
        assert run_result.exit_code == 0, run_result.output

        status_result = runner.invoke(
            app, ["improve", "status", "--output", str(output_dir)]
        )
        assert status_result.exit_code == 0, status_result.output
        payload = json.loads(status_result.output)
        assert payload["reward"] == "customer_support"
        assert payload["curated_count"] == 4

    def test_status_without_prior_run_errors(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["improve", "status", "--output", str(tmp_path / "never-ran")]
        )
        assert result.exit_code != 0
