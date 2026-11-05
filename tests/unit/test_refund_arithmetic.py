"""Borrow contrasts preserve provenance and keep arithmetic holdouts isolated."""

import copy
import hashlib
import json
from collections import Counter

import pytest

from stateset_agents.core.environments.refund_policy_environment import (
    RefundPolicyEnvironment,
    refund_policy_benchmark,
)
from stateset_agents.data.refund_arithmetic_stress import refund_arithmetic_stress_cases
from stateset_agents.data.refund_curriculum import (
    ARITHMETIC_POLICY,
    _arithmetic_cases,
    _parameter_cases,
    audit_refund_curriculum,
    prepare_refund_curriculum,
)
from stateset_agents.data.refund_demonstrations import (
    prepare_refund_data,
    reference_refund_demonstration,
)


def read(path):
    return json.loads(path.read_text())


@pytest.mark.asyncio
async def test_borrow_groups_keep_existing_curriculum_and_replay_exact_amounts(
    tmp_path,
):
    source, output = tmp_path / "source", tmp_path / "borrow"
    await prepare_refund_data(source, train_count=24, validation_count=8, test_count=8)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    splits = {
        name: read(source / f"{name}.json") for name in ("train", "validation", "test")
    }
    parents = {r["order_id"]: r for r in splits["train"]}
    base, _, _ = _parameter_cases(splits)
    manifest = await prepare_refund_curriculum(source, output, policy=ARITHMETIC_POLICY)
    cases = {r["order_id"]: r for r in read(output / "train.json")}
    assert all(cases[r["order_id"]] == r for r in base)
    groups = [
        g
        for g in read(output / "contrast_groups.json")
        if g.get("kind") == "arithmetic"
    ]
    assert len(groups) == manifest["arithmetic_contrast_groups"] > 0
    examples = {
        r["metadata"]["case_id"]: r
        for r in map(json.loads, (output / "train.jsonl").read_text().splitlines())
    }
    for group in groups:
        parent = parents[group["source_case_id"]]
        assert 0 < parent["refunded_cents"] < parent["paid_cents"]
        assert len(group["cases"]) == 9
        assert {(m["paid_offset"], m["refunded_offset"]) for m in group["cases"]} == {
            (p, r) for p in (-1, 0, 1) for r in (-1, 0, 1)
        }
        for member in group["cases"]:
            case = cases[member["case_id"]]
            assert 0 < case["refunded_cents"] < case["paid_cents"]
            assert case["paid_cents"] == group["paid_anchor"] + member["paid_offset"]
            assert (
                case["refunded_cents"]
                == group["refunded_anchor"] + member["refunded_offset"]
            )
            assert (
                case["days_since_delivery"]
                == case["return_window_days"]
                == parent["return_window_days"]
            )
            for key in ("status", "chargeback_open", "customer_note"):
                assert case[key] == parent[key]
            example = examples[case["order_id"]]
            decision = json.loads(example["messages"][3]["content"])
            assert decision == {
                "tool": "refund",
                "args": {
                    "order_id": case["order_id"],
                    "amount_cents": case["paid_cents"] - case["refunded_cents"],
                },
            }
            assert json.loads(example["messages"][-1]["content"]) == {
                "tool": "finish",
                "args": {},
            }
            assert example["metadata"]["source_case_id"] == parent["order_id"]
    fingerprint = RefundPolicyEnvironment.case_fingerprint
    assert len({fingerprint(r) for r in cases.values()}) == len(cases)
    assert (await audit_refund_curriculum(source, output, policy=ARITHMETIC_POLICY))[
        "passed"
    ]
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    duplicate = tmp_path / "duplicate"
    await prepare_refund_curriculum(source, duplicate, policy=ARITHMETIC_POLICY)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == {
        p.name: p.read_bytes() for p in duplicate.iterdir()
    }


@pytest.mark.asyncio
async def test_arithmetic_lineage_cannot_be_rewritten_with_updated_artifact_hash(
    tmp_path,
):
    source, output = tmp_path / "source", tmp_path / "borrow"
    await prepare_refund_data(source, train_count=8, validation_count=8, test_count=8)
    await prepare_refund_curriculum(source, output, policy=ARITHMETIC_POLICY)
    path = output / "contrast_groups.json"
    groups = read(path)
    group = next(g for g in groups if g.get("kind") == "arithmetic")
    group["cases"][0]["paid_offset"] = 1
    path.write_text(json.dumps(groups))
    manifest = read(output / "data_manifest.json")
    manifest["artifacts"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "data_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="reconstruction"):
        await audit_refund_curriculum(source, output, policy=ARITHMETIC_POLICY)


def test_small_orders_remain_partial_and_aliased_heldout_facts_are_rejected():
    splits = {
        s: refund_policy_benchmark(s, 8, 8201) for s in ("train", "validation", "test")
    }
    parent = next(r for r in splits["train"] if r["family"] == "partial_refund")
    parent.update(paid_cents=201, refunded_cents=200)
    before = copy.deepcopy(splits)
    rows, _, groups = _arithmetic_cases(splits)
    group = next(
        g
        for g in groups
        if g.get("kind") == "arithmetic" and g["source_case_id"] == parent["order_id"]
    )
    selected = {m["case_id"] for m in group["cases"]}
    assert all(
        0 < r["refunded_cents"] < r["paid_cents"]
        for r in rows
        if r["order_id"] in selected
    )
    assert before == splits
    leak = copy.deepcopy(next(r for r in rows if r["order_id"] in selected))
    leak["order_id"] = "aliased-exposed-case"
    splits["test"].append(leak)
    with pytest.raises(ValueError, match="overlaps"):
        _arithmetic_cases(splits)


@pytest.mark.asyncio
async def test_arithmetic_grid_covers_each_cell_and_keeps_metadata_private():
    cases = refund_arithmetic_stress_cases(
        "validation", seed=8202, windows=(5, 18, 40), quantum=10
    )
    assert len(cases) == 54
    assert len({r["stress"]["source_case_hash"] for r in cases}) == 54
    assert Counter(r["return_window_days"] for r in cases) == {5: 18, 18: 18, 40: 18}
    for case in cases:
        assert 0 < case["refunded_cents"] < case["paid_cents"]
        assert case["paid_cents"] % 10 == case["arithmetic_stress"]["paid_offset"] % 10
        demo = await reference_refund_demonstration(case)
        observation = json.loads(demo["messages"][2]["content"])
        assert not {"stress", "arithmetic_stress", "family"} & set(observation)
    test = refund_arithmetic_stress_cases(
        "test",
        seed=8202,
        windows=(8, 25, 75),
        quantum=100,
        repetitions=2,
        excluded=cases,
    )
    assert len(test) == 108
    fingerprint = RefundPolicyEnvironment.case_fingerprint
    assert len({fingerprint(r) for r in cases + test}) == 162
    alias = copy.deepcopy(cases[0])
    alias["order_id"] = "another-id"
    with pytest.raises(ValueError, match="overlap"):
        refund_arithmetic_stress_cases(
            "validation", seed=8202, windows=(5, 18, 40), quantum=10, excluded=[alias]
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"split": "train"},
        {"windows": ()},
        {"windows": (5, 5)},
        {"windows": (True,)},
        {"quantum": True},
        {"quantum": 1},
        {"quantum": 15},
        {"repetitions": 0},
    ],
)
def test_invalid_arithmetic_stress_inputs_fail(kwargs):
    arguments = {
        "split": "validation",
        "seed": 8203,
        "windows": (5,),
        "quantum": 10,
        **kwargs,
    }
    with pytest.raises(ValueError):
        refund_arithmetic_stress_cases(**arguments)
