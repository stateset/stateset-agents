"""`--objective` and `--list-objectives` on the training commands."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from stateset_agents.cli import app

runner = CliRunner()


def test_train_help_mentions_objective():
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "--objective" in result.output
    assert "--list-objectives" in result.output


def test_train_list_objectives_prints_every_preset():
    from stateset_agents.training.objectives import OBJECTIVES

    result = runner.invoke(app, ["train", "--list-objectives"])
    assert result.exit_code == 0, result.output
    for name in OBJECTIVES:
        assert name in result.output


def test_train_rejects_unknown_objective():
    result = runner.invoke(app, ["train", "--objective", "bogus", "--dry-run"])
    assert result.exit_code == 2
    assert "bogus" in result.output and "grpo" in result.output


def test_train_dry_run_reports_selected_objective():
    result = runner.invoke(app, ["train", "--objective", "cispo", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "cispo" in result.output


def test_model_command_writes_objective_into_config(tmp_path):
    pytest.importorskip("transformers")
    out = tmp_path / "cfg.json"
    result = runner.invoke(
        app,
        ["qwen3-5-0-8b", "--objective", "cispo", "--write-config", str(out)],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text())
    assert payload["objective"] == "cispo"


def test_model_command_rejects_unknown_objective():
    pytest.importorskip("transformers")
    result = runner.invoke(app, ["qwen3-5-0-8b", "--objective", "bogus"])
    assert result.exit_code == 2
    assert "bogus" in result.output


def test_model_command_forwards_objective_to_gspo_config():
    pytest.importorskip("transformers")
    from stateset_agents.training import starter_common
    from stateset_agents.training.qwen3_5_starter import get_qwen3_5_config

    cfg = get_qwen3_5_config(objective="gspo_token")
    overrides = starter_common.build_gspo_overrides(cfg)
    assert overrides["objective"] == "gspo_token"
