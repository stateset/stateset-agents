"""Dated DeepSeek identity, operator discovery and explicit loader limits."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from stateset_agents.core.frontier_models import get_frontier_model
from stateset_agents.core.model_presets import get_preset
from stateset_agents.remote.model_catalog import plan_runpod_resources
from stateset_agents.training.frontier_sft import prepare_frontier_tokenizer
from stateset_agents.tui import model_choices

KEY = "deepseek-v4.1-flash-1017"
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"


def test_dated_checkpoint_has_its_own_pinned_preset_and_operator_choice():
    dated = get_frontier_model(KEY)
    original = get_frontier_model("deepseek-v4.1-flash")
    assert get_frontier_model(MODEL.upper()) is dated
    assert dated is not original and dated.revision != original.revision
    assert dated.revision == REVISION and dated.audit_date == "2026-10-17"
    preset = get_preset(KEY)
    assert preset.model_id == preset.tokenizer_id == MODEL
    assert preset.model_revision == REVISION
    assert not preset.trust_remote_code
    assert not preset.use_4bit and not preset.use_8bit
    assert all(reason in preset.notes for reason in dated.blockers)
    assert preset.chat_template_override is None and preset.chat_template_kwargs == ()
    choices = dict(model_choices())
    assert choices[MODEL] == "configured" and original.model_id in choices


def test_agent_model_discovery_lists_the_exact_release_without_starting_work():
    from stateset_agents.cli_finetune import app

    result = CliRunner().invoke(app, ["models"])
    assert result.exit_code == 0, result.output
    response = json.loads(result.stdout)
    assert response["ok"] and response["command"] == "models"
    report = response["result"]
    assert report["backend_fit_verified"] is False
    assert {"model": MODEL, "maturity": "configured"} in report["models"]


@pytest.mark.parametrize("name", [KEY, MODEL])
def test_frontier_cli_previews_and_saved_config_keep_exact_identity(tmp_path, name):
    from stateset_agents.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["frontier-sft", "--model", name, "--dry-run"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["model_id"] == MODEL and report["revision"] == REVISION
    assert report["audit_date"] == "2026-10-17"
    assert not report["training_path_available"]
    assert not report["full_size_training_verified"]
    assert any("differentiable" in reason for reason in report["blockers"])
    assert any("chat_template" in reason for reason in report["blockers"])
    config = tmp_path / "deepseek-1017.json"
    written = runner.invoke(
        app, ["frontier-sft", "--model", name, "--write-config", str(config)]
    )
    assert written.exit_code == 0, written.output
    assert json.loads(config.read_text())["model"] == name
    replayed = runner.invoke(app, ["frontier-sft", "--config", str(config)])
    assert replayed.exit_code == 0, replayed.output
    assert json.loads(replayed.stdout)["revision"] == REVISION
    refused = runner.invoke(
        app, ["frontier-sft", "--config", str(config), "--no-dry-run"]
    )
    assert refused.exit_code != 0 and "training is blocked" in refused.output
    assert not (tmp_path / "outputs").exists()


def test_missing_native_template_is_never_replaced_with_parent_encoding():
    tokenizer = SimpleNamespace(chat_template=None, eos_token_id=1, pad_token_id=1)
    with pytest.raises(ValueError, match="native chat template"):
        prepare_frontier_tokenizer(tokenizer, get_frontier_model(KEY))
    assert tokenizer.chat_template is None


def test_unknown_inference_architecture_cannot_start_generic_weight_loading(
    monkeypatch,
):
    from stateset_agents.core import transformers_compat

    monkeypatch.setattr(
        transformers_compat,
        "importlib",
        SimpleNamespace(import_module=lambda name: SimpleNamespace()),
    )
    loader = MagicMock()
    with pytest.raises(RuntimeError, match="compatible Transformers runtime"):
        transformers_compat.load_generation_model(loader, MODEL, {})
    loader.from_pretrained.assert_not_called()


def test_runpod_plan_requires_review_without_promising_capacity_or_launch():
    plan = plan_runpod_resources(MODEL)
    assert plan["model"] == MODEL and plan["certification"] == "configured"
    assert plan["recommendation_evidence"] == "unknown"
    assert plan["manual_review_required"] and not plan["provisions_hardware"]
