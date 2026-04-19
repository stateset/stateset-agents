"""Regression tests for the main usage guide."""

from __future__ import annotations

from pathlib import Path


def test_usage_guide_quick_start_matches_packaged_stub_flow() -> None:
    guide_path = Path(__file__).resolve().parents[2] / "docs" / "USAGE_GUIDE.md"
    contents = guide_path.read_text()

    assert "pip install stateset-agents" in contents
    assert 'model_name="stub://quickstart"' in contents
    assert "use_stub_model=True" in contents
    assert 'training_mode="single_turn"' in contents
    assert "python examples/quick_start.py" in contents


def test_usage_guide_drops_old_package_name() -> None:
    guide_path = Path(__file__).resolve().parents[2] / "docs" / "USAGE_GUIDE.md"
    contents = guide_path.read_text()

    assert "pip install grpo-agent-framework" not in contents
    assert "github.com/yourusername/grpo-agent-framework/issues" not in contents
