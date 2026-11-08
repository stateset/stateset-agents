"""Edits must keep their provenance, remain finite and roll back without data loss."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.core.errors import ConfigurationError  # noqa: E402
from stateset_agents.experimental.weight_space import (  # noqa: E402
    LowRankDelta,
    WeightSpaceSession,
)


def session(dtype=torch.float32):
    model = torch.nn.Sequential(torch.nn.Linear(4, 3, bias=False, dtype=dtype))
    return WeightSpaceSession(model, lambda _: [1], budget=3)


def delta(value=0.1):
    return LowRankDelta(B=torch.full((3, 1), value), A=torch.ones(1, 4))


def test_caller_factor_and_snapshot_mutation_cannot_corrupt_edit_or_rollback():
    s = session()
    original = s.weight("0")
    d = delta()
    edit_id = s.apply("0", d)
    d.B.zero_()
    d.A.fill_(42)
    snapshot = s.edits[0]
    snapshot.delta.B.fill_(9)
    snapshot.previous.fill_(99)
    torch.testing.assert_close(s.edits[0].delta.dense(), torch.full((3, 4), 0.1))
    assert s.active_edit_ids == [edit_id]
    s.revert()
    assert torch.equal(s.weight("0"), original)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 3e38])
def test_failed_edit_does_not_modify_model_or_history(value):
    s = session()
    original = s.weight("0")
    d = LowRankDelta(B=torch.full((3, 1), value), A=torch.full((1, 4), 2.0))
    with pytest.raises(ConfigurationError, match="finite"):
        s.apply("0", d)
    assert torch.equal(s.weight("0"), original)
    assert s.active_edit_ids == []


@pytest.mark.parametrize("operation", ["apply", "revert", "export"])
def test_external_weight_changes_are_detected_without_losing_undo_history(
    operation, tmp_path
):
    if operation == "export":
        pytest.importorskip("peft")
    s = session()
    edit_id = s.apply("0", delta())
    with torch.no_grad():
        s.model[0].weight.add_(1)
    current = s.weight("0")
    with pytest.raises(ConfigurationError, match="outside the session"):
        if operation == "apply":
            s.apply("0", delta())
        elif operation == "revert":
            s.revert()
        else:
            s.export(tmp_path / "adapter")
    assert torch.equal(s.weight("0"), current)
    assert s.active_edit_ids == [edit_id]
    assert not (tmp_path / "adapter").exists()


def test_shared_storage_cannot_be_exported_as_a_module_local_adapter():
    s = session()
    s.model.add_module("1", torch.nn.Linear(4, 3, bias=False))
    s.model[1].weight = s.model[0].weight
    with pytest.raises(ConfigurationError, match="shared weight"):
        s.apply("0", delta())
    assert not s.active_edit_ids


def test_export_records_quantization_discrepancy_and_owns_factors(tmp_path):
    pytest.importorskip("peft")
    s = session(torch.bfloat16)
    with torch.no_grad():
        s.model[0].weight.fill_(1)
    d = delta(0.003)
    s.apply("0", d)
    s.apply("0", d)
    assert torch.equal(s.weight("0"), torch.ones(3, 4))
    d.B.fill_(100)
    out = s.export(tmp_path / "adapter")
    manifest = json.loads((out / "weight_space_manifest.json").read_text())
    assert manifest["merge_weight_max_errors"]["0"] == 0.0078125
    assert "evaluate adapter outputs separately" in manifest["merge_validation"]


def test_constrained_session_edit_reverts_exactly_and_failure_is_atomic():
    s = session()
    original = s.weight("0")
    keys = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    protect = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    targets = torch.tensor([[2.0, 3.0, 4.0]])
    edit_id, result = s.apply_constrained("0", keys, targets, protect=protect)
    torch.testing.assert_close(s.model(keys), targets, atol=1e-5, rtol=1e-5)
    assert s.active_edit_ids == [edit_id]
    assert result.diagnostics.target_max_error < 1e-5
    assert s.budget.used == 0
    after = s.weight("0")
    with pytest.raises(ConfigurationError, match="infeasible"):
        s.apply_constrained("0", keys, targets + 1, protect=keys)
    assert s.active_edit_ids == [edit_id]
    assert torch.equal(s.weight("0"), after)
    s.revert()
    assert torch.equal(s.weight("0"), original)


def test_delta_materialization_ignores_ambient_autocast():
    d = delta(0.12345)
    expected = d.dense()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = d.dense()
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)
