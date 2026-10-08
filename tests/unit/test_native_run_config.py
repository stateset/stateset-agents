"""Native configuration and dry-run semantics must agree at every entry point."""

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from stateset_agents.training import river_refund as native


def inputs(path, *, mode="train", dry_run=False):
    args = SimpleNamespace(
        output=path,
        base_model="base",
        benchmark="refund-policy-v2",
        seed=42,
        steps=2,
        concurrency=8,
        max_staleness=0,
        learning_rate=1e-5,
        checkpoint=None,
        evaluate_only=mode == "evaluate",
        collect_only=mode == "collect",
        dry_run=dry_run,
    )
    names = (
        ("test",)
        if mode == "evaluate"
        else ("train",) if mode == "collect" else ("train", "validation", "test")
    )
    return args, {
        name: native.BENCHMARKS[args.benchmark][1](name, 1, args.seed) for name in names
    }


INVALID = [
    (field, value)
    for field in ("steps", "concurrency", "seed", "max_staleness")
    for value in (-1, True, 1.5, "2", None)
] + [
    ("steps", 0),
    ("concurrency", 0),
    *[
        ("learning_rate", value)
        for value in (False, 0, -1, float("nan"), float("inf"), "1e-5", None, 10**400)
    ],
    *[("base_model", value) for value in (None, "", " ", 123)],
    *[
        (field, value)
        for field in ("dry_run", "collect_only", "evaluate_only")
        for value in (0, "false", None)
    ],
    *[("rollout_token_budget", value) for value in (True, 0, 1023, 1024.0, "1024")],
    *[("zero_update_patience", value) for value in (-1, True, 1.5, "2", None)],
]


@pytest.mark.parametrize("field,value", INVALID)
def test_invalid_settings_fail_before_artifacts_or_sdk(
    tmp_path, monkeypatch, field, value
):
    args, splits = inputs(tmp_path)
    setattr(args, field, value)
    monkeypatch.setitem(sys.modules, "river_client", None)
    monkeypatch.setattr(
        native, "require_native_runtime", lambda: pytest.fail("SDK probe")
    )
    with pytest.raises(ValueError, match=field):
        native.prepare_run(args, splits)
    with pytest.raises(ValueError, match=field):
        native.execute_run(args, splits)
    with pytest.raises(ValueError, match=field):
        asyncio.run(native.campaign(object(), object(), object(), args, splits))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("mode", ["train", "evaluate", "collect"])
def test_all_native_dry_run_modes_avoid_sdk_and_preserve_prepared_evidence(
    tmp_path, monkeypatch, mode
):
    args, splits = inputs(tmp_path, mode=mode, dry_run=True)
    args.rollout_token_budget = 1024
    native.prepare_run(args, splits)
    # A dry inspection may read prepared inputs even after a previous attempt.
    (tmp_path / "test_attempt.json").write_text("{}")
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    monkeypatch.setitem(sys.modules, "river_client", None)
    monkeypatch.setattr(
        native, "require_native_runtime", lambda: pytest.fail("SDK probe")
    )

    class ForbiddenModel:
        def save_weights(self, *args, **kwargs):
            pytest.fail("Dry run must not save provider weights")

    native.execute_run(args, splits)
    asyncio.run(native.campaign(ForbiddenModel(), object(), object(), args, splits))
    assert {p: p.read_bytes() for p in tmp_path.iterdir()} == before
    ledger = json.loads((tmp_path / "rollout_budget.json").read_text())
    assert ledger["admitted_trajectories"] == 0


@pytest.mark.parametrize("patience", [0, 2, 5])
def test_zero_update_threshold_cannot_change_in_a_prepared_run(
    tmp_path, monkeypatch, patience
):
    args, splits = inputs(tmp_path)
    args.zero_update_patience = patience
    native.prepare_run(args, splits)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["training_settings"]["zero_update_patience"] == patience
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    args.zero_update_patience = patience + 1
    monkeypatch.setattr(
        native, "require_native_runtime", lambda: pytest.fail("SDK probe")
    )
    with pytest.raises(ValueError):
        native.prepare_run(args, splits)
    with pytest.raises(ValueError):
        native.execute_run(args, splits)
    with pytest.raises(ValueError):
        asyncio.run(native.campaign(object(), object(), object(), args, splits))
    assert {p: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize("mode", ["train", "evaluate", "collect"])
def test_dry_runs_still_validate_required_cases(tmp_path, monkeypatch, mode):
    args, _ = inputs(tmp_path, mode=mode, dry_run=True)
    monkeypatch.setitem(sys.modules, "river_client", None)
    with pytest.raises(ValueError, match="required cases"):
        native.execute_run(args, {})
    with pytest.raises(ValueError, match="required cases"):
        asyncio.run(native.campaign(object(), object(), object(), args, {}))


@pytest.mark.parametrize(
    "option,value,message",
    [
        ("--steps", "0", "steps"),
        ("--concurrency", "0", "concurrency"),
        ("--seed", "-1", "seed"),
        ("--max-staleness", "-1", "max_staleness"),
        ("--learning-rate", "nan", "learning_rate"),
        ("--learning-rate", "inf", "learning_rate"),
        ("--base-model", " ", "base_model"),
        ("--rollout-token-budget", "1023", "rollout_token_budget"),
        ("--zero-update-patience", "-1", "zero_update_patience"),
    ],
)
def test_cli_uses_shared_configuration_rules_before_preparation(
    tmp_path, monkeypatch, capsys, option, value, message
):
    output = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        ["river-refund", "--dry-run", "--output", str(output), option, value],
    )
    with pytest.raises(SystemExit) as exc:
        native.main()
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    assert not output.exists()
