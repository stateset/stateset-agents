"""`--objective` and `--list-objectives` on the training commands."""

from __future__ import annotations

import json
import re

import pytest
from typer.testing import CliRunner

from stateset_agents.cli import app

# CI sets FORCE_COLOR=1 and a narrow terminal, so rich wraps flags across
# lines and escape sequences; ask for plain wide output and strip the rest.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_ENV = {"COLUMNS": "200", "NO_COLOR": "1", "TERM": "dumb", "FORCE_COLOR": "0"}
runner = CliRunner(env=_ENV)


def _invoke(args: list[str]) -> tuple[int, str]:
    result = runner.invoke(app, args)
    plain = " ".join(_ANSI.sub("", result.output).split())
    return result.exit_code, plain


def test_train_help_mentions_objective():
    code, out = _invoke(["train", "--help"])
    assert code == 0, out
    assert "--objective" in out
    assert "--list-objectives" in out


def test_train_list_objectives_prints_every_preset():
    from stateset_agents.training.objectives import OBJECTIVES

    code, out = _invoke(["train", "--list-objectives"])
    assert code == 0, out
    for name in OBJECTIVES:
        assert name in out


def test_train_rejects_unknown_objective():
    code, out = _invoke(["train", "--objective", "bogus", "--dry-run"])
    assert code == 2
    assert "bogus" in out and "grpo" in out


def test_train_dry_run_reports_selected_objective():
    code, out = _invoke(["train", "--objective", "cispo", "--dry-run"])
    assert code == 0, out
    assert "cispo" in out


def test_model_command_writes_objective_into_config(tmp_path):
    pytest.importorskip("transformers")
    out_path = tmp_path / "cfg.json"
    code, out = _invoke(
        ["qwen3-5-0-8b", "--objective", "cispo", "--write-config", str(out_path)]
    )
    assert code == 0, out
    payload = json.loads(out_path.read_text())
    assert payload["objective"] == "cispo"


def test_model_command_rejects_unknown_objective():
    pytest.importorskip("transformers")
    code, out = _invoke(["qwen3-5-0-8b", "--objective", "bogus"])
    assert code == 2
    assert "bogus" in out


def test_model_command_forwards_objective_to_gspo_config():
    pytest.importorskip("transformers")
    from stateset_agents.training import starter_common
    from stateset_agents.training.qwen3_5_starter import get_qwen3_5_config

    cfg = get_qwen3_5_config(objective="gspo_token")
    overrides = starter_common.build_gspo_overrides(cfg)
    assert overrides["objective"] == "gspo_token"
