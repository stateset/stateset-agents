"""Regression tests for public onboarding docs."""

from __future__ import annotations

from pathlib import Path


def test_index_quickstart_matches_stub_backed_flow() -> None:
    index_path = Path(__file__).resolve().parents[2] / "docs" / "index.rst"
    contents = index_path.read_text()

    assert 'model_name="stub://quickstart"' in contents
    assert "use_stub_model=True" in contents
    assert 'training_mode="single_turn"' in contents
    assert "asyncio.run(main())" in contents
    assert "python examples/quick_start.py" in contents


def test_quickstart_doc_avoids_stale_gpt2_bootstrap() -> None:
    quickstart_path = Path(__file__).resolve().parents[2] / "docs" / "quickstart.rst"
    contents = quickstart_path.read_text()

    assert 'model_name="stub://quickstart"' in contents
    assert "use_stub_model=True" in contents
    assert 'training_mode="single_turn"' in contents
    assert "python examples/quick_start.py" in contents
    assert 'model_name="gpt2"' not in contents
