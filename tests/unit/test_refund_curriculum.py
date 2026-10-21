"""Boundary contrasts preserve source isolation and reject rewritten evidence."""

import copy
import hashlib
import json

import pytest
from typer.testing import CliRunner

from stateset_agents.cli import app
from stateset_agents.core.environments.refund_policy_environment import (
    RefundPolicyEnvironment,
)
from stateset_agents.data.refund_curriculum import (
    ARITHMETIC_POLICY,
    CURRICULUM_POLICY,
    PARAMETER_POLICY,
    PARAMETER_WINDOWS,
    _contrast_cases,
    _parameter_cases,
    audit_refund_curriculum,
    prepare_refund_curriculum,
)
from stateset_agents.data.refund_demonstrations import (
    prepare_refund_data,
    replay_refund_candidate,
)
from stateset_agents.evaluation.agent_runs import content_hash


def read(path):
    return json.loads(path.read_text())


@pytest.mark.asyncio
async def test_adjacent_triplets_replay_boundary_arithmetic_and_all_original_cases(
    tmp_path,
):
    source, output = tmp_path / "source", tmp_path / "curriculum"
    await prepare_refund_data(source, train_count=24, validation_count=8, test_count=8)
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    manifest = await prepare_refund_curriculum(source, output)
    cases = read(output / "train.json")
    by_id = {row["order_id"]: row for row in cases}
    originals = read(source / "train.json")
    parents = {row["order_id"]: row for row in originals}
    assert all(by_id[row["order_id"]] == row for row in originals)
    assert manifest["original_cases"] == 24
    assert manifest["derived_cases"] > 0
    assert manifest["contrast_groups"] == 9
    fingerprint = RefundPolicyEnvironment.case_fingerprint
    assert len({fingerprint(row) for row in cases}) == len(cases)
    held_out = {
        fingerprint(row)
        for name in ("validation", "test")
        for row in read(source / f"{name}.json")
    }
    assert not held_out & {fingerprint(row) for row in cases}
    chats = {
        row["metadata"]["case_id"]: row
        for row in map(json.loads, (output / "train.jsonl").read_text().splitlines())
    }
    for row in cases:
        item = chats[row["order_id"]]
        assert item["metadata"]["split"] == "train"
        assert item["metadata"]["source_case_id"] in parents
        await replay_refund_candidate(
            row,
            {
                "case_id": row["order_id"],
                "case_hash": content_hash(row),
                "truncated": None,
                "messages": item["messages"],
            },
        )
    for group in read(output / "contrast_groups.json"):
        parent = parents[group["source_case_id"]]
        assert [case["boundary_offset"] for case in group["cases"]] == [-1, 0, 1]
        for member in group["cases"]:
            row = by_id[member["case_id"]]
            assert (
                row["days_since_delivery"]
                == parent["return_window_days"] + member["boundary_offset"]
            )
            assert row["return_window_days"] == parent["return_window_days"]
            action = json.loads(chats[row["order_id"]]["messages"][3]["content"])
            if member["boundary_offset"] <= 0:
                assert action["tool"] == "refund"
                assert action["args"]["amount_cents"] == (
                    parent["paid_cents"] - parent["refunded_cents"]
                )
            else:
                assert action["tool"] == "deny"
                assert action["args"]["reason"] == "outside_window"
    audit = await audit_refund_curriculum(source, output)
    assert audit["replayed_demonstrations"] == manifest["selected"] == len(cases)
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
    assert not (output / "validation.json").exists()
    assert not (output / "test.json").exists()
    duplicate = tmp_path / "duplicate"
    assert await prepare_refund_curriculum(source, duplicate) == manifest
    assert {p.name: p.read_bytes() for p in output.iterdir()} == {
        p.name: p.read_bytes() for p in duplicate.iterdir()
    }
    with pytest.raises(ValueError, match="not empty"):
        await prepare_refund_curriculum(source, output)
    with pytest.raises(ValueError, match="prepared"):
        await prepare_refund_curriculum(output, tmp_path / "recursive")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "artifact", ["train.json", "train.jsonl", "lineage.json", "contrast_groups.json"]
)
@pytest.mark.parametrize(
    "policy", [CURRICULUM_POLICY, PARAMETER_POLICY, ARITHMETIC_POLICY]
)
async def test_audit_rejects_rehashed_facts_transcripts_and_lineage(
    tmp_path, artifact, policy
):
    source, output = tmp_path / "source", tmp_path / "curriculum"
    await prepare_refund_data(source, train_count=8, validation_count=8, test_count=8)
    await prepare_refund_curriculum(source, output, policy=policy)
    path = output / artifact
    if artifact == "train.jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["messages"][3]["content"] = '{"tool":"finish","args":{}}'
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    else:
        rows = read(path)
        if artifact == "train.json":
            rows[0]["days_since_delivery"] += 1
        elif artifact == "lineage.json":
            rows[0]["source_split"] = "validation"
        else:
            rows[0]["cases"][0]["boundary_offset"] = 0
        path.write_text(json.dumps(rows))
    manifest = read(output / "data_manifest.json")
    manifest["artifacts"][artifact] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "data_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="reconstruction"):
        await audit_refund_curriculum(source, output, policy=policy)


@pytest.mark.asyncio
async def test_parameter_contrasts_preserve_priority_notes_and_exact_completion(
    tmp_path,
):
    source, output = tmp_path / "source", tmp_path / "parameters"
    await prepare_refund_data(source, train_count=80, validation_count=8, test_count=8)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    manifest = await prepare_refund_curriculum(source, output, policy=PARAMETER_POLICY)
    assert manifest["selected"] == 320
    assert manifest["contrast_groups"] == 80
    assert manifest["parameter_windows"] == list(PARAMETER_WINDOWS)
    parents = {r["order_id"]: r for r in read(source / "train.json")}
    cases = {r["order_id"]: r for r in read(output / "train.json")}
    assert all(cases[key] == row for key, row in parents.items())
    examples = {
        r["metadata"]["case_id"]: r
        for r in map(json.loads, (output / "train.jsonl").read_text().splitlines())
    }
    coverage = {}
    for group in read(output / "contrast_groups.json"):
        parent = parents[group["source_case_id"]]
        coverage.setdefault(parent["family"], set()).add(group["return_window_days"])
        for member in group["cases"]:
            row = cases[member["case_id"]]
            assert row["days_since_delivery"] == (
                row["return_window_days"] + member["boundary_offset"]
            )
            for key in (
                "paid_cents",
                "refunded_cents",
                "status",
                "chargeback_open",
                "customer_note",
            ):
                assert row[key] == parent[key]
            example = examples[row["order_id"]]
            messages = example["messages"]
            assert json.loads(messages[-1]["content"]) == {"tool": "finish", "args": {}}
            decision = json.loads(messages[3]["content"])
            assert decision["args"]["order_id"] == row["order_id"]
            if row["chargeback_open"]:
                assert decision["tool"] == "escalate"
                assert decision["args"]["reason"] == "chargeback"
            elif row["refunded_cents"] == row["paid_cents"]:
                assert decision["tool"] == "deny"
                assert decision["args"]["reason"] == "already_refunded"
            elif row["status"] != "delivered":
                assert decision["tool"] == "deny"
                assert decision["args"]["reason"] == "not_delivered"
            elif member["boundary_offset"] > 0:
                assert decision["tool"] == "deny"
                assert decision["args"]["reason"] == "outside_window"
            else:
                assert decision["tool"] == "refund"
                assert decision["args"]["amount_cents"] == (
                    row["paid_cents"] - row["refunded_cents"]
                )
            assert example["metadata"]["source_case_id"] == parent["order_id"]
            await replay_refund_candidate(
                row,
                {
                    "case_id": row["order_id"],
                    "case_hash": content_hash(row),
                    "truncated": None,
                    "messages": messages,
                },
            )
    assert all(windows == set(PARAMETER_WINDOWS) for windows in coverage.values())
    fingerprint = RefundPolicyEnvironment.case_fingerprint
    assert len({fingerprint(r) for r in cases.values()}) == len(cases)
    held_out = {
        fingerprint(r)
        for split in ("validation", "test")
        for r in read(source / f"{split}.json")
    }
    assert not held_out & {fingerprint(r) for r in cases.values()}
    assert (await audit_refund_curriculum(source, output, policy=PARAMETER_POLICY))[
        "passed"
    ]
    with pytest.raises(ValueError, match="provenance"):
        await audit_refund_curriculum(source, output)
    duplicate = tmp_path / "duplicate"
    await prepare_refund_curriculum(source, duplicate, policy=PARAMETER_POLICY)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == {
        p.name: p.read_bytes() for p in duplicate.iterdir()
    }
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before


@pytest.mark.asyncio
async def test_parameter_contrasts_reject_aliased_heldout_facts(tmp_path):
    source = tmp_path / "source"
    await prepare_refund_data(source, train_count=8, validation_count=8, test_count=8)
    splits = {
        name: read(source / f"{name}.json") for name in ("train", "validation", "test")
    }
    rows, _, _ = _parameter_cases(splits)
    derived = copy.deepcopy(
        next(r for r in rows if r["order_id"].startswith("curriculum-"))
    )
    derived["order_id"] = "different-id-same-policy-facts"
    splits["test"].append(derived)
    with pytest.raises(ValueError, match="overlaps"):
        _parameter_cases(splits)


@pytest.mark.asyncio
async def test_unknown_curriculum_policy_fails_before_publication(tmp_path):
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Unsupported"):
        await prepare_refund_curriculum(tmp_path / "missing", output, policy="typo")
    assert not output.exists()


@pytest.mark.asyncio
async def test_rejects_source_case_relabelled_as_training_even_with_updated_hash(
    tmp_path,
):
    source = tmp_path / "source"
    await prepare_refund_data(source, train_count=8, validation_count=8, test_count=8)
    path = source / "train.json"
    rows = read(path)
    rows[0] = read(source / "validation.json")[0]
    path.write_text(json.dumps(rows))
    manifest = read(source / "data_manifest.json")
    manifest["artifacts"]["train.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (source / "data_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="canonical"):
        await prepare_refund_curriculum(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.asyncio
async def test_rejects_fact_overlap_even_when_held_out_id_differs(tmp_path):
    source = tmp_path / "source"
    await prepare_refund_data(source, train_count=8, validation_count=8, test_count=8)
    splits = {
        name: read(source / f"{name}.json") for name in ("train", "validation", "test")
    }
    rows, _, groups = _contrast_cases(splits)
    derived = next(r for r in rows if r["order_id"].startswith("curriculum-"))
    assert groups
    leaked = copy.deepcopy(derived)
    leaked["order_id"] = "unrelated-held-out-id"
    splits["validation"].append(leaked)
    with pytest.raises(ValueError, match="overlaps"):
        _contrast_cases(splits)


@pytest.mark.parametrize(
    "policy", [CURRICULUM_POLICY, PARAMETER_POLICY, ARITHMETIC_POLICY]
)
def test_curriculum_cli_round_trip_and_failure(tmp_path, policy):
    runner = CliRunner()
    source, output = tmp_path / "source", tmp_path / "curriculum"
    result = runner.invoke(
        app,
        [
            "benchmark",
            "prepare-refund-data",
            "--output",
            str(source),
            "--train-count",
            "8",
            "--validation-count",
            "8",
            "--test-count",
            "8",
        ],
    )
    assert result.exit_code == 0, result.output
    arguments = ["--data-dir", str(source), "--output", str(output), "--policy", policy]
    result = runner.invoke(app, ["benchmark", "prepare-refund-curriculum", *arguments])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["benchmark", "audit-refund-curriculum", *arguments])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output
    (output / "train.jsonl").write_text("[]\n")
    result = runner.invoke(app, ["benchmark", "audit-refund-curriculum", *arguments])
    assert result.exit_code == 2, result.output
