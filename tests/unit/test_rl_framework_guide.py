"""Regression tests for the canonical usage guide (docs/RL_FRAMEWORK_GUIDE.md)."""

from __future__ import annotations

from pathlib import Path

GUIDE = Path(__file__).resolve().parents[2] / "docs" / "RL_FRAMEWORK_GUIDE.md"


def test_canonical_guide_has_stub_quickstart_flow() -> None:
    contents = GUIDE.read_text()

    assert "pip install stateset-agents" in contents
    assert 'model_name="stub://quickstart"' in contents
    assert "use_stub_model=True" in contents


def test_canonical_guide_drops_old_package_name() -> None:
    contents = GUIDE.read_text()

    assert "pip install grpo-agent-framework" not in contents
    assert "image: grpo-agent-framework:latest" not in contents
    assert "github.com/yourusername/grpo-agent-framework" not in contents
