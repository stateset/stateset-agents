"""Regression tests for package/deployment version alignment."""

from __future__ import annotations

from pathlib import Path

from stateset_agents import __version__


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
