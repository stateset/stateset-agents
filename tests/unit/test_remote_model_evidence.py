"""Tests for explicit model/provider verification claims."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from stateset_agents.cli import app
from stateset_agents.remote.model_catalog import (
    get_model_support,
    model_catalog,
    plan_runpod_resources,
)
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
    assert payload["schema_version"] == 2
    assert payload["models"] == model_catalog()
    assert payload["evidence"] == model_provider_evidence()


def test_default_and_frontier_tiers_are_explicit() -> None:
    default = get_model_support("qwen/qwen3.5-0.8b")
    frontier = get_model_support("Qwen/Qwen3.8-Flash-Next-FP8")

    assert default is not None and default.tier == "default"
    assert default.certification == "training-verified"
    assert frontier is not None and frontier.tier == "frontier-preview"
    assert frontier.certification == "smoke-tested"


def test_catalog_ids_and_aliases_are_unique() -> None:
    names = []
    for row in model_catalog():
        names.append(str(row["model"]).casefold())
        names.extend(str(alias).casefold() for alias in row["aliases"])

    assert len(names) == len(set(names))


def test_runpod_plan_distinguishes_measured_and_estimated_resources() -> None:
    measured = plan_runpod_resources("Qwen/Qwen3.8-27B")
    estimated = plan_runpod_resources("zai-org/GLM-5.3-Flash")

    assert measured["gpu"] == "NVIDIA H100 80GB HBM3"
    assert measured["gpu_count"] == 1
    assert measured["manual_review_required"] is False
    assert estimated["gpu_count"] == 8
    assert estimated["manual_review_required"] is True
    assert estimated["provisions_hardware"] is False


def test_explicit_runpod_resource_overrides_are_preserved() -> None:
    plan = plan_runpod_resources(
        "Qwen/Qwen3.8-Flash-Next", gpu="custom", gpu_count=2, container_disk_gb=99
    )

    assert (plan["gpu"], plan["gpu_count"], plan["container_disk_gb"]) == (
        "custom",
        2,
        99,
    )
    assert all(plan["explicit_overrides"].values())
