"""Unit tests for ``stateset-agents train-remote``."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from stateset_agents.cli import app

runner = CliRunner()


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "curated.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            }
        )
        + "\n"
    )
    return path


class TestCommandRegistration:
    def test_command_is_registered(self):
        result = runner.invoke(app, ["train-remote", "--help"])

        assert result.exit_code == 0
        assert "--provider" in result.output

    def test_help_lists_the_available_providers(self):
        result = runner.invoke(app, ["train-remote", "--help"])

        assert "local" in result.output
        assert "modal" in result.output


class TestSuccessfulRun:
    def test_local_dry_run_exits_zero(self, dataset, tmp_path):
        result = runner.invoke(
            app,
            [
                "train-remote",
                "--provider",
                "local",
                "--dataset",
                str(dataset),
                "--base-model",
                "Qwen/Qwen3.5-0.8B",
                "--output-dir",
                str(tmp_path / "out"),
                "--dry-run",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Qwen/Qwen3.5-0.8B" in result.output


class TestFailurePaths:
    def test_unknown_provider_exits_nonzero_and_names_valid_options(
        self, dataset, tmp_path
    ):
        result = runner.invoke(
            app,
            [
                "train-remote",
                "--provider",
                "aws-batch",
                "--dataset",
                str(dataset),
                "--base-model",
                "Qwen/Qwen3.5-0.8B",
            ],
        )

        assert result.exit_code != 0
        assert "local" in result.output

    def test_missing_dataset_exits_nonzero(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "train-remote",
                "--provider",
                "local",
                "--dataset",
                str(tmp_path / "absent.jsonl"),
                "--base-model",
                "Qwen/Qwen3.5-0.8B",
            ],
        )

        assert result.exit_code != 0
        assert "does not exist" in result.output

    def test_failed_job_exits_nonzero(self, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")

        result = runner.invoke(
            app,
            [
                "train-remote",
                "--provider",
                "local",
                "--dataset",
                str(empty),
                "--base-model",
                "Qwen/Qwen3.5-0.8B",
                "--dry-run",
            ],
        )

        assert result.exit_code != 0
