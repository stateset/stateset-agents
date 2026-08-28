"""Tests for explicit model/provider verification claims."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from stateset_agents.cli import app
from stateset_agents.remote.model_evidence import model_provider_evidence

runner = CliRunner()


def test_new_models_have_framework_evidence() -> None:
    rows = model_provider_evidence()

    framework_models = {
        row["model"]
        for row in rows
        if row["provider"] == "framework" and row["outcome"] == "pass"
    }

    assert "Qwen/Qwen3.8-Flash-Next" in framework_models
    assert "zai-org/GLM-5.3-Flash" in framework_models


def test_no_failed_run_claims_inference_verification() -> None:
    for row in model_provider_evidence():
        if row["outcome"] != "pass":
            assert row["level"] != "inference-verified"


def test_model_support_json_is_machine_readable() -> None:
    result = runner.invoke(app, ["model-support", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == 1
    assert payload["evidence"] == model_provider_evidence()
