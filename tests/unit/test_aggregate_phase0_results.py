"""Unit tests for the Phase 0 results aggregator."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import aggregate_phase0_results as agg  # noqa: E402


def _make_run(
    trainer: str,
    seed: int,
    pass_at_1: float,
    baseline: float = 0.32,
    wall: float = 2700.0,
    commit: str = "a" * 40,
    model: str = "Qwen/Qwen3.5-0.8B",
    evidence_class: str = "measured",
) -> dict:
    return {
        "trainer": trainer,
        "model": model,
        "model_revision": "b" * 40,
        "seed": seed,
        "commit": commit,
        "evidence_class": evidence_class,
        "timestamp": f"2026-05-13T11:{seed:02d}:00Z",
        "config": {},
        "metrics": {
            "eval_pass_at_1": pass_at_1,
            "eval_pass_at_1_baseline": baseline,
            "wall_clock_seconds": wall,
            "peak_vram_mb": 24000,
            "status": "trained",
            "max_grad_norm_ratio": 1.2,
        },
        "hardware": {"gpu": "NVIDIA A100-SXM4-80GB"},
    }


class TestLoadRuns:
    def test_loads_valid_files(self, tmp_path: Path) -> None:
        for i, seed in enumerate([42, 1337]):
            run = _make_run("gspo", seed, 0.4 + i * 0.01)
            (tmp_path / f"gspo_seed{seed}.json").write_text(json.dumps(run))
        loaded = agg.load_runs(tmp_path)
        assert len(loaded) == 2
        assert {r["seed"] for r in loaded} == {42, 1337}

    def test_skips_malformed_json(self, tmp_path: Path) -> None:
        (tmp_path / "valid.json").write_text(json.dumps(_make_run("gspo", 42, 0.4)))
        (tmp_path / "bad.json").write_text("{not valid json")
        loaded = agg.load_runs(tmp_path)
        assert len(loaded) == 1

    def test_skips_missing_required_fields(self, tmp_path: Path) -> None:
        (tmp_path / "valid.json").write_text(json.dumps(_make_run("gspo", 42, 0.4)))
        (tmp_path / "missing_trainer.json").write_text(
            json.dumps(
                {
                    "model": "Qwen",
                    "seed": 1,
                    "commit": "abc",
                    "timestamp": "x",
                    "metrics": {"eval_pass_at_1": 0.5},
                }
            )
        )
        loaded = agg.load_runs(tmp_path)
        assert len(loaded) == 1

    def test_skips_missing_metrics(self, tmp_path: Path) -> None:
        run = _make_run("gspo", 42, 0.4)
        del run["metrics"]["eval_pass_at_1"]
        (tmp_path / "missing_metric.json").write_text(json.dumps(run))
        loaded = agg.load_runs(tmp_path)
        assert len(loaded) == 0

    def test_skips_invalid_identity_types_and_nonfinite_scores(
        self, tmp_path: Path
    ) -> None:
        invalid_revision = _make_run("gspo", 42, 0.4)
        invalid_revision["model_revision"] = None
        (tmp_path / "invalid-revision.json").write_text(json.dumps(invalid_revision))
        invalid_score = _make_run("gspo", 1337, 0.4)
        invalid_score["metrics"]["eval_pass_at_1"] = float("nan")
        (tmp_path / "invalid-score.json").write_text(json.dumps(invalid_score))
        assert agg.load_runs(tmp_path) == []

    def test_empty_dir(self, tmp_path: Path) -> None:
        assert agg.load_runs(tmp_path) == []

    def test_synthetic_evidence_requires_explicit_preview_flag(
        self, tmp_path: Path
    ) -> None:
        run = _make_run("gspo", 42, 0.4, evidence_class="synthetic")
        (tmp_path / "synthetic.json").write_text(json.dumps(run))
        assert agg.load_runs(tmp_path) == []
        assert len(agg.load_runs(tmp_path, allow_synthetic=True)) == 1


class TestSummarize:
    def test_three_seeds(self) -> None:
        runs = [
            _make_run("gspo", 42, 0.41, baseline=0.32),
            _make_run("gspo", 1337, 0.43, baseline=0.31),
            _make_run("gspo", 2026, 0.39, baseline=0.33),
        ]
        s = agg.summarize_group(runs)
        assert s["pass_at_1"]["n"] == 3
        assert s["pass_at_1"]["mean"] == pytest.approx(0.41, abs=1e-9)
        assert s["pass_at_1"]["std"] > 0
        assert s["improvement"] == pytest.approx(0.41 - 0.32, abs=1e-9)

    def test_single_seed(self) -> None:
        runs = [_make_run("grpo", 42, 0.5, baseline=0.4)]
        s = agg.summarize_group(runs)
        assert s["pass_at_1"]["n"] == 1
        assert s["pass_at_1"]["std"] == 0.0
        assert s["improvement"] == pytest.approx(0.1, abs=1e-9)

    def test_missing_baseline(self) -> None:
        run = _make_run("grpo", 42, 0.5)
        del run["metrics"]["eval_pass_at_1_baseline"]
        s = agg.summarize_group([run])
        assert math.isnan(s["improvement"])


class TestGates:
    def test_all_pass(self) -> None:
        runs = [
            _make_run("gspo", 42, 0.41, baseline=0.32),
            _make_run("gspo", 1337, 0.43, baseline=0.31),
            _make_run("gspo", 2026, 0.39, baseline=0.33),
        ]
        summary = agg.summarize_group(runs)
        passed, failures = agg.check_gates(summary)
        assert passed, f"Should have passed: {failures}"

    def test_fail_too_few_seeds(self) -> None:
        runs = [_make_run("gspo", 42, 0.50, baseline=0.30)]
        passed, failures = agg.check_gates(agg.summarize_group(runs))
        assert not passed
        assert any("seeds" in f for f in failures)

    def test_fail_insufficient_improvement(self) -> None:
        runs = [
            _make_run("gspo", 42, 0.32, baseline=0.32),
            _make_run("gspo", 1337, 0.33, baseline=0.31),
            _make_run("gspo", 2026, 0.31, baseline=0.33),
        ]
        passed, failures = agg.check_gates(agg.summarize_group(runs))
        assert not passed
        assert any("improvement" in f for f in failures)

    def test_fail_high_std(self) -> None:
        runs = [
            _make_run("gspo", 42, 0.20, baseline=0.30),
            _make_run("gspo", 1337, 0.60, baseline=0.30),
            _make_run("gspo", 2026, 0.40, baseline=0.30),
        ]
        passed, failures = agg.check_gates(agg.summarize_group(runs))
        assert not passed
        assert any("std" in f for f in failures)

    def test_fail_mixed_commits(self) -> None:
        runs = [
            _make_run("gspo", 42, 0.41, baseline=0.32, commit="a" * 40),
            _make_run("gspo", 1337, 0.43, baseline=0.31, commit="b" * 40),
            _make_run("gspo", 2026, 0.39, baseline=0.33, commit="c" * 40),
        ]
        passed, failures = agg.check_gates(agg.summarize_group(runs))
        assert not passed
        assert any("commit" in f for f in failures)

    def test_synthetic_results_never_pass(self) -> None:
        runs = [
            _make_run("gspo", seed, 0.42, evidence_class="synthetic")
            for seed in (42, 1337, 2026)
        ]
        passed, failures = agg.check_gates(agg.summarize_group(runs))
        assert not passed
        assert any("non-measured" in failure for failure in failures)


class TestRenderMarkdown:
    def test_includes_table_header(self) -> None:
        runs = [_make_run("gspo", 42, 0.5, baseline=0.4)]
        grouped = {("gspo", "Qwen/Qwen3.5-0.8B"): agg.summarize_group(runs)}
        gates = {("gspo", "Qwen/Qwen3.5-0.8B"): (False, ["only 1 seed"])}
        md = agg.render_markdown(grouped, gates)
        assert "Trainer" in md
        assert "Final pass@1" in md
        assert "GSPO" in md
        assert "Qwen/Qwen3.5-0.8B" in md

    def test_empty_results(self) -> None:
        md = agg.render_markdown({}, {})
        assert "no results" in md.lower()
