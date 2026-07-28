"""Unit tests for ``stateset-agents improve`` (grade -> curate -> retrain in one command)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from stateset_agents.cli import app

runner = CliRunner()

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
