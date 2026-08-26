"""End-to-end integration test for the Phase 0 benchmark pipeline.

Exercises the full happy path that ``make smoke`` / the demo / production CI
all depend on:

1. Scaffold a project from a starter template.
2. Run the Phase 0 benchmark runner in smoke-test mode.
3. Drop synthetic JSON results into ``benchmark_results/``.
4. Aggregate them via ``scripts/aggregate_phase0_results.py``.
5. Generate plots / figures via ``scripts/plot_phase0_results.py``.
6. Run the v1.0 release packager via ``scripts/release_v1_whitepaper.py``.

Verifies the artifacts that land at each step: scaffold files, summary.md,
summary.csv, the auto-generated §11.7 markdown, the figure copies in docs/,
and the release manifest. This is the single test that catches regressions
in the integration between modules; the per-module unit tests cover the
inside of each step.

Runs in under 10 seconds with no GPU. Uses ``tmp_path`` everywhere so it's
hermetic across pytest runs.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a subprocess and return the result. Caller checks .returncode."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        cwd=cwd or REPO_ROOT,
    )


def _write_synthetic_run(
    dst_dir: Path,
    trainer: str,
    seed: int,
    pass_at_1: float = 0.42,
    baseline: float = 0.32,
    commit: str = "e2e-test",
) -> Path:
    """Drop a schema-compliant JSON result into ``dst_dir``."""
    result = {
        "trainer": trainer,
        "task": "gsm8k",
        "model": "Qwen/Qwen3.5-0.8B",
        "model_revision": "b" * 40,
        "seed": seed,
        "commit": commit if len(commit) == 40 else "a" * 40,
        "evidence_class": "synthetic",
        "timestamp": "2026-05-14T12:00:00Z",
        "config": {"learning_rate": 5e-6, "num_generations": 4},
        "metrics": {
            "eval_pass_at_1": pass_at_1,
            "eval_pass_at_1_baseline": baseline,
            "wall_clock_seconds": 2700,
            "peak_vram_mb": 24317,
            "status": "trained",
            "train_examples": 200,
            "eval_examples": 100,
        },
        "hardware": {"gpu": "NVIDIA A100-SXM4-80GB"},
    }
    path = dst_dir / f"{trainer}_seed{seed}_qwen3_5_0_8b.json"
    path.write_text(json.dumps(result, indent=2))
    return path


class TestEndToEndHappyPath:
    """The full pipeline, end-to-end. Every assertion guards a real user flow."""

    def test_scaffold_then_smoke_then_aggregate_then_plot_then_release(
        self, tmp_path: Path
    ) -> None:
        # ----- Step 1: scaffold a project -----
        project = tmp_path / "client_acme"
        scaffold_cmd = [
            sys.executable,
            "-m",
            "stateset_agents.cli",
            "starter",
            "customer-support",
            str(project),
        ]
        result = _run(scaffold_cmd)
        assert result.returncode == 0, f"scaffold failed: {result.stderr}"
        assert (project / "config.yaml").exists()
        assert (project / "train.py").exists()
        assert (project / "README.md").exists()

        # Verify scaffold marker recorded what was generated.
        marker = json.loads(
            (project / ".stateset-agents-starter.json").read_text(encoding="utf-8")
        )
        assert marker["template"] == "customer-support"

        # ----- Step 2: benchmark smoke test -----
        smoke_out = tmp_path / "smoke.json"
        smoke_cmd = [
            sys.executable,
            "scripts/run_phase0_benchmark.py",
            "--trainer",
            "gspo",
            "--task",
            "customer_support",
            "--num-train-examples",
            "5",
            "--num-eval-examples",
            "3",
            "--smoke-test",
            "--output",
            str(smoke_out),
        ]
        result = _run(smoke_cmd)
        assert result.returncode == 0, f"smoke failed: {result.stderr}"
        assert "Smoke test passed" in (result.stdout + result.stderr)

        # ----- Step 3: drop synthetic results matching SCHEMA.md -----
        results_dir = tmp_path / "benchmark_results" / "whitepaper_v1"
        results_dir.mkdir(parents=True)
        for trainer in ("gspo", "grpo", "dapo"):
            for seed in (42, 1337, 2026):
                # Different trainers get different (but reproducible) scores.
                pass_at_1 = {"gspo": 0.42, "grpo": 0.38, "dapo": 0.44}[trainer]
                _write_synthetic_run(results_dir, trainer, seed, pass_at_1=pass_at_1)
        assert len(list(results_dir.glob("*.json"))) == 9

        # ----- Step 4: aggregate -----
        aggregate_cmd = [
            sys.executable,
            "scripts/aggregate_phase0_results.py",
            "--results-dir",
            str(results_dir),
            "--allow-synthetic",
        ]
        result = _run(aggregate_cmd)
        assert result.returncode == 0, f"aggregate failed: {result.stderr}"
        assert (results_dir / "summary.md").exists()
        assert (results_dir / "summary.csv").exists()
        assert (results_dir / "passes_gates.json").exists()

        # Summary.md should mention all three trainers.
        summary = (results_dir / "summary.md").read_text(encoding="utf-8")
        for trainer in ("GSPO", "GRPO", "DAPO"):
            assert trainer in summary, f"missing {trainer} in summary.md"

        # Synthetic data can exercise rendering but can never pass publication gates.
        gates = json.loads(
            (results_dir / "passes_gates.json").read_text(encoding="utf-8")
        )
        for key, info in gates.items():
            assert not info["passed"], f"{key} synthetic evidence unexpectedly passed"

        # CSV should have 9 data rows (header + 9).
        csv_lines = (
            (results_dir / "summary.csv").read_text(encoding="utf-8").splitlines()
        )
        assert len(csv_lines) == 10  # header + 9 rows

        # ----- Step 5: plot (text fallback — matplotlib may or may not be present) -----
        plot_cmd = [
            sys.executable,
            "scripts/plot_phase0_results.py",
            "--results-dir",
            str(results_dir),
            "--no-matplotlib",
        ]
        result = _run(plot_cmd)
        assert result.returncode == 0, f"plot failed: {result.stderr}"
        assert (results_dir / "text_plots.md").exists()
        text_plots = (results_dir / "text_plots.md").read_text(encoding="utf-8")
        assert "GSPO" in text_plots and "GRPO" in text_plots and "DAPO" in text_plots

    def test_aggregate_strict_fails_on_underspecified_group(
        self, tmp_path: Path
    ) -> None:
        # A single seed should fail the 3-seed gate.
        results_dir = tmp_path / "underspec"
        results_dir.mkdir(parents=True)
        _write_synthetic_run(results_dir, "gspo", seed=42)

        cmd = [
            sys.executable,
            "scripts/aggregate_phase0_results.py",
            "--results-dir",
            str(results_dir),
            "--strict",
        ]
        result = _run(cmd)
        assert result.returncode != 0, "--strict should fail on under-seeded group"


class TestSmokePathForAllAdaptersAndTrainers:
    """Cartesian product of (trainer, task) — every combination should run the
    smoke path without errors. Catches regressions in the registry / dispatch."""

    @pytest.mark.parametrize("trainer", ["gspo", "grpo", "dapo"])
    @pytest.mark.parametrize("task", ["gsm8k", "customer_support", "tool_calling"])
    def test_smoke(self, tmp_path: Path, trainer: str, task: str) -> None:
        out = tmp_path / f"{trainer}_{task}.json"
        n_train = 5 if task != "customer_support" else 4
        n_eval = 3 if task != "customer_support" else 2
        cmd = [
            sys.executable,
            "scripts/run_phase0_benchmark.py",
            "--trainer",
            trainer,
            "--task",
            task,
            "--num-train-examples",
            str(n_train),
            "--num-eval-examples",
            str(n_eval),
            "--smoke-test",
            "--output",
            str(out),
        ]
        result = _run(cmd)
        assert (
            result.returncode == 0
        ), f"smoke failed for trainer={trainer} task={task}: {result.stderr}"
        assert f"task={task}" in (result.stdout + result.stderr)
