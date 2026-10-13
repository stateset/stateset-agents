"""End-to-end: the matrix runner drives the local driver through all scenarios.

Two node process groups (controller + 2 workers, 2 workers) over loopback TCP
run every scenario for one seed with the diagnostic manifest's short
durations (soak 60 s, faults 30 s). Each run must pass the per-run evidence
validator with zero lost or duplicate work, bounded lag, and an observed
recovery, and the publication gate must still refuse the local provider.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.distributed_async_evidence import (
    REQUIRED_SCENARIOS,
    DistributedAsyncEvidenceError,
    load_runs,
    validate_matrix,
)
from benchmarks.run_distributed_async_matrix import run_scenario

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "benchmarks" / "distributed_async_local_manifest.json"

pytest.importorskip("torch")
pytest.importorskip("uvicorn")


def _head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    commit = result.stdout.strip()
    return commit if len(commit) == 40 else "a" * 40


@pytest.mark.slow
@pytest.mark.timeout(1500)
def test_local_matrix_all_scenarios_one_seed(tmp_path: Path) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["command"][0] = sys.executable
    # Diagnostic floors: the scenario durations pinned in the config.
    manifest["minimum_duration_seconds"] = dict(
        manifest["config"]["scenario_duration_seconds"]
    )
    seed = manifest["seeds"][0]
    commit = _head()
    for scenario in REQUIRED_SCENARIOS:
        run_scenario(
            manifest,
            scenario,
            seed,
            output_dir=tmp_path,
            root=ROOT,
            harness_commit=commit,
            timeout_seconds=600,
        )
    runs = load_runs([tmp_path / "evidence"])
    assert {run["scenario"] for run in runs} == set(REQUIRED_SCENARIOS)
    max_lag = manifest["config"]["max_policy_lag"]
    for run in runs:
        metrics = run["metrics"]
        assert metrics["lost_optimizer_updates"] == 0
        assert metrics["duplicate_optimizer_updates"] == 0
        assert metrics["duplicate_trained_rollouts"] == 0
        assert metrics["lost_admitted_rollouts"] == 0
        assert metrics["phantom_trained_rollouts"] == 0
        assert metrics["artifact_digest_mismatches"] == 0
        assert metrics["max_observed_policy_lag"] <= max_lag
        assert metrics["optimizer_updates"] >= 5
        assert run["recovery"]["recovered"] is True
        assert run["recovery"]["resources_remaining"] == 0
        assert run["topology"]["diagnostic"] is True
        if run["scenario"] == "controller_restart":
            assert metrics["controller_epochs"] == 2
    with pytest.raises(DistributedAsyncEvidenceError, match="not publishable"):
        validate_matrix(runs, min_seeds=1, min_soak_seconds=1)
