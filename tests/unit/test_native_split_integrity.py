"""Native experiments must validate cases and holdout boundaries before use."""

import asyncio
import copy
import json
import sys
from types import SimpleNamespace

import pytest

from stateset_agents.training import river_refund as native


def inputs(path, benchmark="refund-v1", mode="train"):
    args = SimpleNamespace(
        output=path,
        benchmark=benchmark,
        base_model="base",
        seed=42,
        steps=2,
        concurrency=8,
        max_staleness=0,
        learning_rate=1e-5,
        checkpoint=None,
        evaluate_only=mode == "evaluate",
        collect_only=mode == "collect",
        dry_run=False,
    )
    splits = {
        name: native.BENCHMARKS[benchmark][1](name, 2, 42)
        for name in ("train", "validation", "test")
    }
    return args, splits


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_train",
        "empty_validation",
        "empty_test",
        "not_list",
        "unknown_split",
        "path_split",
        "bad_row",
        "missing_id",
        "blank_id",
        "boolean_money",
        "invalid_eligibility",
        "duplicate",
        "train_validation_overlap",
        "train_test_overlap",
        "validation_test_overlap",
        "both_modes",
        "integer_mode",
        "v1_collection",
    ],
)
def test_invalid_inputs_never_publish_a_run(tmp_path, mutation):
    args, splits = inputs(tmp_path)
    if mutation == "missing_train":
        del splits["train"]
    elif mutation.startswith("empty_"):
        splits[mutation.removeprefix("empty_")] = []
    elif mutation == "not_list":
        splits["train"] = tuple(splits["train"])
    elif mutation in ("unknown_split", "path_split"):
        splits["other" if mutation == "unknown_split" else "../outside"] = []
    elif mutation == "bad_row":
        splits["train"][0] = "not an order"
    elif mutation == "missing_id":
        del splits["train"][0]["order_id"]
    elif mutation == "blank_id":
        splits["train"][0]["order_id"] = "  "
    elif mutation == "boolean_money":
        splits["train"][0]["amount_cents"] = True
    elif mutation == "invalid_eligibility":
        splits["train"][0]["eligible"] = 1
    elif mutation == "duplicate":
        splits["train"].append(copy.deepcopy(splits["train"][0]))
    elif mutation.endswith("_overlap"):
        left, right, _ = mutation.split("_")
        # Different order facts cannot disguise reuse of a held-out identity.
        splits[right][0]["order_id"] = splits[left][0]["order_id"]
    elif mutation == "both_modes":
        args.evaluate_only = args.collect_only = True
    elif mutation == "integer_mode":
        args.evaluate_only = 0
    else:
        args.collect_only = True
    with pytest.raises(ValueError):
        native.prepare_run(args, splits)
    assert not list(tmp_path.iterdir())
    assert not (tmp_path.parent / "outside.json").exists()


@pytest.mark.parametrize("mode", ["train", "evaluate", "collect"])
def test_policy_family_is_required_before_artifacts_or_provider_work(tmp_path, mode):
    args, splits = inputs(tmp_path, "refund-policy-v2", mode)
    del splits["test"][0]["family"]
    with pytest.raises(ValueError, match="policy family"):
        native.prepare_run(args, splits)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("mode", ["evaluate", "collect"])
def test_unused_splits_may_be_absent_or_empty(tmp_path, mode):
    args, splits = inputs(tmp_path, "refund-policy-v2", mode)
    required = "test" if mode == "evaluate" else "train"
    splits = {name: rows if name == required else [] for name, rows in splits.items()}
    del splits["validation"]
    native.prepare_run(args, splits)
    assert json.loads((tmp_path / "run_manifest.json").read_text())["split_hashes"]


def test_matching_manifest_cannot_authorize_overlapping_splits(tmp_path, monkeypatch):
    args, splits = inputs(tmp_path)
    splits["test"][0]["order_id"] = splits["train"][0]["order_id"]
    # Simulate a legacy directory whose internally matching files were never
    # checked for leakage. Execution must validate beyond matching hashes.
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(native.run_manifest(args, splits))
    )
    for name, rows in splits.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(rows))
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    monkeypatch.setitem(sys.modules, "river_client", None)
    monkeypatch.setattr(
        native, "require_native_runtime", lambda: pytest.fail("SDK probe")
    )
    with pytest.raises(ValueError, match="splits must be disjoint"):
        native.execute_run(args, splits)
    with pytest.raises(ValueError, match="splits must be disjoint"):
        asyncio.run(native.campaign(object(), object(), object(), args, splits))
    assert {p: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize("benchmark_name", list(native.BENCHMARKS))
@pytest.mark.parametrize("row", [None, [], "order", {}, {"order_id": " "}])
def test_preflight_and_environment_reset_share_scenario_rules(benchmark_name, row):
    environment = native.BENCHMARKS[benchmark_name][0]()
    with pytest.raises(ValueError):
        environment.validate_scenario(row)
    with pytest.raises(ValueError):
        asyncio.run(environment.reset(row))
