"""Regression tests for package/deployment version alignment."""

from __future__ import annotations

from pathlib import Path

from stateset_agents import __version__
from stateset_agents.utils.repo_hygiene import find_version_hygiene_issues


def test_makefile_uses_package_version_for_docker_tags() -> None:
    makefile_path = Path(__file__).resolve().parents[2] / "Makefile"
    contents = makefile_path.read_text()

    assert "PACKAGE_VERSION := $(shell $(PYTHON_BIN) -c " in contents
    assert "stateset/stateset-agents-api:$(PACKAGE_VERSION)" in contents
    assert "stateset/stateset-agents-trainer:$(PACKAGE_VERSION)" in contents


def test_helm_values_use_current_package_version() -> None:
    values_path = (
        Path(__file__).resolve().parents[2]
        / "deployment"
        / "helm"
        / "stateset-agents"
        / "values.yaml"
    )
    chart_path = (
        Path(__file__).resolve().parents[2]
        / "deployment"
        / "helm"
        / "stateset-agents"
        / "Chart.yaml"
    )
    values_contents = values_path.read_text()
    chart_contents = chart_path.read_text()

    assert f'tag: "{__version__}"' in values_contents
    assert f'appVersion: "{__version__}"' in chart_contents


def test_selected_kubernetes_and_docs_refs_use_current_package_version() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    targets = [
        repo_root / "deployment" / "kubernetes" / "production-deployment.yaml",
        repo_root / "deployment" / "kubernetes" / "kimi-k25-training-job.yaml",
        repo_root / "deployment" / "kubernetes" / "glm5-1-training-job.yaml",
        repo_root / "deployment" / "kubernetes" / "qwen3-5-27b-training-job.yaml",
        repo_root / "deployment" / "helm" / "stateset-agents" / "README.md",
        repo_root / "docs" / "KIMI_K25_GKE_AUTOPILOT.md",
    ]

    for target in targets:
        contents = target.read_text()
        assert "0.7.1" not in contents
        assert __version__ in contents


def test_public_deployment_examples_require_auth_by_default() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    targets = [
        repo_root
        / "deployment"
        / "helm"
        / "stateset-agents"
        / "values-glm5-1-fp8.yaml",
        repo_root
        / "deployment"
        / "helm"
        / "stateset-agents"
        / "values-qwen3-5-27b-minimal.yaml",
        repo_root
        / "deployment"
        / "helm"
        / "stateset-agents"
        / "values-gke-example-staging.yaml",
    ]

    for target in targets:
        contents = target.read_text()
        assert 'API_REQUIRE_AUTH: "true"' in contents
        assert 'API_REQUIRE_AUTH: "false"' not in contents


def test_public_docs_and_examples_do_not_embed_internal_identifiers() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    targets = [
        repo_root / "docs" / "GLM5_1_HOSTING_PLAN.md",
        repo_root / "docs" / "glm5_1_starter.rst",
        repo_root / "examples" / "README.md",
        repo_root / "examples" / "finetune_glm5_1_gspo.py",
        repo_root / "scripts" / "gke" / "publish_model_to_gcs.sh",
        repo_root / "deployment" / "helm" / "stateset-agents" / "values-glm5-1-fp8.yaml",
        repo_root / "deployment" / "kubernetes" / "glm5-1-vllm-fp8.yaml",
    ]

    forbidden_markers = [
        "gs://stateset-models",
        "gs://stateset-models-prod",
        "gs://stateset-models-dev",
        "zai-org/GLM-5.1-FP8",
    ]

    for target in targets:
        contents = target.read_text()
        for marker in forbidden_markers:
            assert marker not in contents


def test_internal_version_surfaces_match_package_version() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    assert find_version_hygiene_issues(repo_root, package_version=__version__) == []
