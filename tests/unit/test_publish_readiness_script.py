"""Regression tests for publish readiness automation."""

from __future__ import annotations

from pathlib import Path


def test_publish_readiness_uses_stable_type_check_wrapper() -> None:
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "publish_readiness.sh"
    )
    contents = script_path.read_text(encoding="utf-8")

    assert "python scripts/check_types.py --all" in contents
    assert "mypy --config-file mypy.ini" not in contents


def test_publish_readiness_enforces_stable_api_contract() -> None:
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "publish_readiness.sh"
    )
    contents = script_path.read_text(encoding="utf-8")

    assert 'CURRENT_STEP="api_compatibility"' in contents
    assert '"$PYTHON_BIN" scripts/check_api_compatibility.py' in contents


def test_publish_readiness_enforces_release_governance() -> None:
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "publish_readiness.sh"
    )
    contents = script_path.read_text(encoding="utf-8")

    assert 'CURRENT_STEP="release_governance"' in contents
    assert '"$PYTHON_BIN" scripts/check_release_governance.py' in contents


def test_publish_readiness_enforces_agent_quality_contract() -> None:
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "publish_readiness.sh"
    )
    contents = script_path.read_text(encoding="utf-8")

    assert 'CURRENT_STEP="agent_quality_contract"' in contents
    assert "benchmarks/run_agent_quality_matrix.py" in contents
    assert "benchmarks/agent_quality_manifest.example.json" in contents
    assert "--dry-run" in contents


def test_publish_readiness_normalizes_safety_input() -> None:
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "publish_readiness.sh"
    )
    contents = script_path.read_text(encoding="utf-8")

    assert "grep -v '^cuda-toolkit\\[' requirements-dev-lock.txt" in contents
    assert 'safety check -r "$SAFETY_INPUT_PATH"' in contents
    assert 'rm -f "$SAFETY_INPUT_PATH"' in contents
