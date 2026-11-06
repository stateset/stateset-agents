"""Kimi K3.1 Pro is independently pinned and never reuses the K3 loader."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from typer.testing import CliRunner

from stateset_agents.cli import app
from stateset_agents.core.frontier_models import get_frontier_model
from stateset_agents.core.model_presets import get_preset
from stateset_agents.remote.model_catalog import get_model_support
from stateset_agents.remote.model_evidence import model_provider_evidence


def test_kimi31_uses_its_own_checkpoint_architecture_and_preset():
    spec = get_frontier_model("kimi-k3.1-pro")
    previous = get_frontier_model("kimi-k3")
    preset = get_preset(spec.key)
    assert spec.model_id == "moonshotai/Kimi-K3.1-Pro"
    assert spec.revision != previous.revision
    assert spec.model_type == "kimi_k31"
    assert spec.architecture == "KimiK31ForConditionalGeneration"
    assert preset.model_id == spec.model_id
    assert preset.model_revision == spec.revision
    assert preset.lora_target_modules == spec.attention_targets
    assert preset.use_4bit is False and preset.use_8bit is False
    assert spec.blockers
    assert "wq_b" in spec.attention_targets
    assert re.fullmatch(
        spec.lora_regex, "language_model.model.layers.3.self_attn.indexer.wq_b"
    )
    assert not re.fullmatch(spec.lora_regex, "vision_tower.blocks.0.attn.q_proj")
    support = get_model_support(spec.model_id)
    assert support.certification == "configured"
    assert support.tier == "frontier-preview" and support.runpod is None
    evidence = [r for r in model_provider_evidence() if r["model"] == spec.model_id]
    assert len(evidence) == 1
    assert evidence[0]["level"] == "unverified"
    assert evidence[0]["outcome"] == "blocked"


def test_pinned_kimi31_config_matches_retained_metadata():
    root = (
        Path(__file__).parents[2]
        / "benchmark_results/training_contracts/20261106/kimi-k3.1-pro"
    )
    raw = (root / "config.json").read_bytes()
    config = json.loads(raw)
    row = json.loads((root / "metadata.json").read_text())[0]
    spec = get_frontier_model(row["key"])
    assert hashlib.sha256(raw).hexdigest() == row["config_sha256"]
    assert config["architectures"] == [spec.architecture]
    assert config["model_type"] == spec.model_type
    text = config["text_config"]
    assert text["max_position_embeddings"] == 1048576
    assert text["num_hidden_layers"] == 93
    assert len(text["linear_attn_config"]["kda_layers"]) == 69
    assert len(text["dsa_config"]["dsa_layers"]) == 24
    assert text["quantization_config"]["format"] == spec.checkpoint_format
    assert spec.checkpoint_bytes == row["index_metadata"]["total_size"]
    assert row["license_metadata"]["license_name"] == spec.license


def test_cli_kimi31_preview_explains_local_training_blockers():
    runner = CliRunner()
    result = runner.invoke(app, ["frontier-models", "--model", "kimi-k3.1-pro"])
    assert result.exit_code == 0, result.output
    row = json.loads(result.output)[0]
    assert row["training_path_available"] is False
    assert row["full_size_training_verified"] is False
    preview = runner.invoke(app, ["frontier-sft", "--model", "kimi-k3.1-pro"])
    assert preview.exit_code == 0, preview.output
    assert json.loads(preview.output)["training_path_available"] is False
    rejected = runner.invoke(
        app, ["frontier-sft", "--model", "kimi-k3.1-pro", "--no-dry-run"]
    )
    assert rejected.exit_code != 0
    assert "training is blocked" in rejected.output
