"""Regression tests for live release-evidence workflow semantics."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_codeql_merge_gate_uses_stable_query_suite() -> None:
    config = yaml.safe_load((ROOT / ".github/codeql-config.yml").read_text())
    query_suites = {entry["uses"] for entry in config["queries"]}

    assert "security-and-quality" in query_suites
    assert "security-experimental" not in query_suites


def test_gpu_verification_cannot_pass_by_skipping_for_missing_key() -> None:
    workflow = (ROOT / ".github/workflows/gpu-verify.yml").read_text(encoding="utf-8")

    assert workflow.count("RUNPOD_API_KEY secret not set") == 2
    assert workflow.count("exit 1") >= 2
    assert "skipping GPU verification" not in workflow
    assert "skipping RL GPU verification" not in workflow
    assert "needs: sft-live-smoke" in workflow


def test_provider_canaries_run_for_release_tags() -> None:
    workflow = (ROOT / ".github/workflows/provider-canary.yml").read_text(
        encoding="utf-8"
    )

    assert "push:" in workflow
    assert "tags:" in workflow
    assert "- 'v*'" in workflow
    for provider in ("river", "runpod", "fireworks"):
        assert f"--provider {provider} --strict" in workflow


def test_publish_requires_readiness_before_build_or_upload() -> None:
    workflow = (ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8")

    readiness = workflow.index("Run mandatory publish readiness gate")
    build = workflow.index("Build distribution")
    test_pypi = workflow.index("Publish to TestPyPI")
    assert readiness < build < test_pypi
    assert "continue-on-error: true" not in workflow
    assert "run_readiness_gate" not in workflow
    assert "publish-readiness-${{ github.sha }}" in workflow
