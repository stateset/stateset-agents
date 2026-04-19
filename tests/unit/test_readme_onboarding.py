"""Regression tests for README onboarding snippets."""

from __future__ import annotations

from pathlib import Path


def test_readme_points_users_to_stub_backed_quickstart() -> None:
    readme_path = Path(__file__).resolve().parents[2] / "README.md"
    contents = readme_path.read_text()

    assert "python examples/quick_start.py" in contents
    assert 'model_name="stub://quickstart"' in contents
    assert "use_stub_model=True" in contents
    assert 'training_mode="single_turn"' in contents


def test_readme_real_model_example_uses_placeholder_model_id() -> None:
    readme_path = Path(__file__).resolve().parents[2] / "README.md"
    contents = readme_path.read_text()

    assert 'model_name="your-real-model-id"' in contents
