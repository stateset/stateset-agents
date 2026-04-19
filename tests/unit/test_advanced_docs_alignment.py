"""Regression tests for advanced guide bootstrap examples."""

from __future__ import annotations

from pathlib import Path


def test_selected_guides_no_longer_default_to_gpt2_bootstrap() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    targets = [
        repo_root / "docs" / "GSPO_GUIDE.md",
        repo_root / "docs" / "QUICKSTART_5MIN.md",
        repo_root / "docs" / "AUTO_RESEARCH_GUIDE.md",
        repo_root / "docs" / "SINGLE_TURN_TRAINING.md",
        repo_root / "docs" / "RL_FRAMEWORK_GUIDE.md",
        repo_root / "docs" / "MIGRATION_GRPO_TO_GSPO.md",
        repo_root / "docs" / "GRPO_IMPLEMENTATION.md",
    ]

    for target in targets:
        contents = target.read_text()
        assert 'model_name="gpt2"' not in contents


def test_selected_guides_use_current_repo_identity() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    targets = [
        repo_root / "docs" / "COMPREHENSIVE_USAGE_GUIDE.md",
        repo_root / "docs" / "GRPO_FRAMEWORK_ENHANCEMENTS_v4.md",
    ]

    for target in targets:
        contents = target.read_text()
        assert "grpo-framework.readthedocs.io" not in contents
        assert "github.com/stateset/grpo-agent-framework" not in contents
