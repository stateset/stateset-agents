"""Contract tests for the distributed scaling collector."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
SPEC = importlib.util.spec_from_file_location(
    "run_scaling_matrix", BENCHMARKS / "run_scaling_matrix.py"
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)
import distributed_scaling_workload as workload  # noqa: E402


def test_collector_default_matches_a_plus_efficiency_gate() -> None:
    args = runner.parse_args([])
    assert args.min_efficiency == pytest.approx(0.70)
    assert args.gpu_counts == [1, 2, 4, 8]
    assert args.seeds == [42, 1337, 2026]


def _launcher_manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "stateset-scaling-launcher-manifest",
        "topologies": {
            "1": {"node_count": 1, "nproc_per_node": 1},
            "2": {"node_count": 1, "nproc_per_node": 2},
            "4": {"node_count": 1, "nproc_per_node": 4},
            "8": {"node_count": 2, "nproc_per_node": 4},
        },
        "command": [
            "driver",
            "{gpu_count}",
            "{node_count}",
            "{nproc_per_node}",
            "{seed}",
            "{harness_commit}",
            "{config_json}",
            "{command_label}",
            "{output}",
            "{workload}",
        ],
    }


def test_multi_node_launcher_manifest_is_exact_and_shell_free(tmp_path: Path) -> None:
    path = tmp_path / "launcher.json"
    path.write_text(json.dumps(_launcher_manifest()), encoding="utf-8")
    loaded = runner.load_launcher_manifest(path, [1, 2, 4, 8])
    assert loaded["topologies"]["8"]["node_count"] == 2

    invalid = _launcher_manifest()
    invalid["topologies"]["8"] = {"node_count": 1, "nproc_per_node": 8}
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(runner.ScalingRunnerError, match="at least two nodes"):
        runner.load_launcher_manifest(path, [1, 2, 4, 8])


def test_launcher_rejects_missing_binding_placeholder(tmp_path: Path) -> None:
    manifest = _launcher_manifest()
    manifest["command"].remove("{harness_commit}")
    path = tmp_path / "launcher.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(runner.ScalingRunnerError, match="harness_commit"):
        runner.load_launcher_manifest(path, [1, 2, 4, 8])


def test_observed_topology_must_match_launcher(tmp_path: Path) -> None:
    output = tmp_path / "run.json"
    output.write_text(
        json.dumps(
            {
                "hardware": {
                    "node_count": 1,
                    "ranks_per_node": {"node-a": 8},
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(runner.ScalingRunnerError, match="node_count"):
        runner._validate_launched_topology(
            output, {"node_count": 2, "nproc_per_node": 4}
        )


def test_node_identity_prefers_host_level_dmi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dmi = tmp_path / "product_uuid"
    dmi.write_text("host-uuid\n", encoding="utf-8")
    monkeypatch.setattr(workload, "NODE_ID_SOURCES", ((dmi, "dmi-product-uuid"),))
    node_id, source = workload._node_identity()
    assert len(node_id) == 16
    assert source == "dmi-product-uuid"


def test_node_identity_labels_hostname_as_diagnostic_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workload, "NODE_ID_SOURCES", ())
    monkeypatch.setattr(workload.socket, "gethostname", lambda: "container-a")
    _, source = workload._node_identity()
    assert source == "hostname-fallback"
