"""Regression tests for live release-evidence workflow semantics."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_gpu_verification_cannot_pass_by_skipping_for_missing_key() -> None:
    workflow = (ROOT / ".github/workflows/gpu-verify.yml").read_text(encoding="utf-8")

    assert workflow.count("RUNPOD_API_KEY secret not set") == 2
    assert workflow.count("exit 1") >= 2
    assert "skipping GPU verification" not in workflow
    assert "skipping RL GPU verification" not in workflow


def test_provider_canaries_run_for_release_tags() -> None:
    workflow = (ROOT / ".github/workflows/provider-canary.yml").read_text(
        encoding="utf-8"
    )

    assert "push:" in workflow
    assert "tags:" in workflow
    assert "- 'v*'" in workflow
    for provider in ("river", "runpod", "fireworks"):
        assert f"--provider {provider} --strict" in workflow
