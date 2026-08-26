"""Guard publication-facing benchmarks against synthetic evidence regressions."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_comparison_entrypoints_cannot_simulate_results() -> None:
    for relative in (
        "benchmarks/algorithm_comparison.py",
        "benchmarks/framework_comparison.py",
    ):
        contents = (ROOT / relative).read_text(encoding="utf-8")
        forbidden = ("asyncio.sleep", "time.sleep", "np.random", "random.uniform")
        assert not any(token in contents for token in forbidden), relative
        assert "measured must be true" in (
            (ROOT / "benchmarks/framework_comparison.py").read_text(encoding="utf-8")
        )


def test_benchmark_docs_state_unproven_claims_explicitly() -> None:
    contents = (ROOT / "docs/BENCHMARKS.md").read_text(encoding="utf-8")
    assert "does **not** currently claim" in contents
    assert "faster training or lower memory than TRL" in contents
    assert "multi-node or 2/4/8-GPU scaling efficiency" in contents
    assert "Synthetic and stub outputs must never" in contents


def test_framework_schema_requires_measured_matched_runs() -> None:
    contents = (ROOT / "benchmark_results/framework_comparison/SCHEMA.md").read_text(
        encoding="utf-8"
    )
    assert '"measured": true' in contents
    assert '"algorithm_revision"' in contents
    assert "at least three unique seeds" in contents
