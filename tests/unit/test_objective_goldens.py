"""Trainers must reproduce the loss goldens captured before migration.

Regenerate deliberately with ``scripts/capture_objective_goldens.py --only X``
and explain the change in CHANGELOG.md; never regenerate to make a red test
green without understanding why the number moved.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

GOLDENS = json.loads(
    (Path(__file__).parent / "goldens" / "objective_goldens.json").read_text()
)
_spec = importlib.util.spec_from_file_location(
    "capture_objective_goldens",
    Path(__file__).resolve().parents[2] / "scripts" / "capture_objective_goldens.py",
)
assert _spec is not None and _spec.loader is not None
capture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(capture)


def _assert_same(got, want, path="", atol=1e-6):
    if isinstance(want, dict):
        assert set(got) == set(want), path
        for k in want:
            _assert_same(got[k], want[k], f"{path}/{k}", atol)
    elif isinstance(want, list):
        assert len(got) == len(want), path
        for i, (g, w) in enumerate(zip(got, want, strict=True)):
            _assert_same(g, w, f"{path}[{i}]", atol)
    else:
        assert got == pytest.approx(want, abs=atol), path


@pytest.mark.parametrize("name", sorted(capture.CAPTURES))
def test_trainer_reproduces_golden(name):
    torch.use_deterministic_algorithms(True)
    _assert_same(capture.CAPTURES[name](), GOLDENS[name], name)
