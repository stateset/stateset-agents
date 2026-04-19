"""Regression tests for publish readiness automation."""

from __future__ import annotations

from pathlib import Path


def test_publish_readiness_uses_stable_type_check_wrapper() -> None:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "publish_readiness.sh"
    contents = script_path.read_text()

    assert "python scripts/check_types.py --all" in contents
    assert "mypy --config-file mypy.ini" not in contents
