"""The native shootout leg accepts the checked-in v3 and v4 protocol configs.

The v4 RunPod matrix failed on the pod because the native script rejected the
v4 manifest's objective keys; this pins every checked-in manifest's config
against the script's validator so that cannot recur unseen.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _script():
    spec = importlib.util.spec_from_file_location(
        "run_phase0_benchmark", ROOT / "scripts" / "run_phase0_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "manifest", ["shootout_manifest_v3.json", "shootout_manifest_v4.json"]
)
def test_checked_in_manifest_config_is_accepted(manifest):
    data = json.loads((ROOT / "benchmarks" / manifest).read_text())
    assert _script().validate_shootout_config(data["algorithm"], data["config"]) is None


def test_v4_objective_mismatch_is_rejected():
    data = json.loads((ROOT / "benchmarks" / "shootout_manifest_v4.json").read_text())
    config = dict(data["config"], epsilon=0.2)
    error = _script().validate_shootout_config("gspo", config)
    assert error is not None and "epsilon" in error


def test_v4_keys_require_the_gspo_trainer():
    data = json.loads((ROOT / "benchmarks" / "shootout_manifest_v4.json").read_text())
    assert _script().validate_shootout_config("grpo", data["config"]) is not None


def test_unknown_keys_are_rejected():
    data = json.loads((ROOT / "benchmarks" / "shootout_manifest_v4.json").read_text())
    assert (
        _script().validate_shootout_config("gspo", dict(data["config"], extra=1))
        is not None
    )
