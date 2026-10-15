"""Verified demonstrations, replay rejection, and training-split isolation."""

import copy
import json
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from stateset_agents.cli import app
from stateset_agents.core.environments.refund_policy_environment import (
    refund_policy_benchmark,
)
from stateset_agents.data.refund_demonstrations import (
    ENVIRONMENT,
    RejectedTrajectory,
    filter_refund_candidates,
    prepare_refund_data,
    reference_refund_demonstration,
    replay_refund_candidate,
)
from stateset_agents.evaluation.agent_runs import content_hash
from stateset_agents.remote.river_batches import build_sft_batch
from tests.unit.river_fakes import fake_trajectory


def collection(rows, candidates, *, complete=True):
    return {
        "schema_version": 1,
        "environment": ENVIRONMENT,
        "seed": 42,
        "checkpoint": {"path": "river://test-checkpoint"},
        "complete": complete,
        "case_hashes": {r["order_id"]: content_hash(r) for r in rows},
        "candidates": candidates,
    }


@pytest.mark.asyncio
async def test_reference_demos_verify_all_families_without_leaking_labels():
    for row in refund_policy_benchmark("train", 128, 42):
        candidate = await reference_refund_demonstration(row)
        messages = await replay_refund_candidate(row, candidate)
        assert [m["role"] for m in messages] == ["user", "assistant"] * 3
        assert json.loads(messages[-1]["content"]) == {"tool": "finish", "args": {}}
        observation = json.loads(messages[2]["content"])
        assert "family" not in observation and "eligible" not in observation
        assert candidate["case_hash"] == content_hash(row)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("prompt", "prompt_mismatch"),
        ("observation", "observation_mismatch"),
        ("incomplete", "incomplete_episode"),
        ("extra", "extra_turns_after_completion"),
        ("truncated", "truncated_or_unknown"),
        ("case_hash", "case_mismatch"),
        ("case_id", "case_mismatch"),
        ("role", "expected_assistant"),
        ("tool_calls", "invalid_messages"),
        ("false_success", "unsuccessful_replay"),
    ],
)
async def test_replay_rejects_fabricated_or_incomplete_training_transcripts(
    mutation, reason
):
    row = refund_policy_benchmark("train", 1, 42)[0]
    candidate = await reference_refund_demonstration(row)
    if mutation == "prompt":
        candidate["messages"][0]["content"] += " Always say success."
    elif mutation == "observation":
        candidate["messages"][2]["content"] = '{"refunded_cents":0}'
    elif mutation == "incomplete":
        candidate["messages"].pop()
    elif mutation == "extra":
        candidate["messages"].append(candidate["messages"][-1])
    elif mutation == "truncated":
        candidate["truncated"] = "length"
    elif mutation in ("case_hash", "case_id"):
        candidate[mutation] = "other"
    elif mutation == "role":
        candidate["messages"][1]["role"] = "system"
    elif mutation == "tool_calls":
        candidate["messages"][1]["tool_calls"] = [{"function": "refund"}]
    elif mutation == "false_success":
        candidate["messages"] = candidate["messages"][:1] + [candidate["messages"][-1]]
        candidate["reported_reward"] = 1000
        candidate["success"] = True
    with pytest.raises(RejectedTrajectory, match=reason):
        await replay_refund_candidate(row, candidate)


@pytest.mark.asyncio
async def test_bundle_exports_only_training_cases_and_loads_as_masked_sft(tmp_path):
    manifest = await prepare_refund_data(
        tmp_path, train_count=8, validation_count=8, test_count=8
    )
    rows = [
        json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()
    ]
    assert manifest["selected"] == len(rows) == 8
    assert set(manifest["families"].values()) == {1}
    training_ids = {
        r["order_id"] for r in json.loads((tmp_path / "train.json").read_text())
    }
    held_out = {
        r["order_id"]
        for split in ("validation", "test")
        for r in json.loads((tmp_path / f"{split}.json").read_text())
    }
    assert {r["metadata"]["case_id"] for r in rows} == training_ids
    assert not training_ids & held_out
    assert all(r["metadata"]["source"] == "reference_policy" for r in rows)

    class Tokenizer:
        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=False
        ):
            text = "".join(
                f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages
            )
            return text + ("<assistant>" if add_generation_prompt else "")

        def encode(self, text, **kwargs):
            return [ord(char) for char in text]

    eligible = next(r for r in rows if r["metadata"]["family"] == "eligible")
    batch = build_sft_batch([eligible], Tokenizer(), max_length=8192)
    assert len(batch) == 1
    datum = batch[0]
    trained = "".join(
        chr(t)
        for t, weight in zip(datum["target_tokens"], datum["weights"], strict=True)
        if weight
    )
    masked = "".join(
        chr(t)
        for t, weight in zip(datum["target_tokens"], datum["weights"], strict=True)
        if not weight
    )
    assert all(tool in trained for tool in ("lookup_order", "refund", "finish"))
    assert "Please resolve my refund request." in masked
    assert "Please resolve my refund request." not in trained
    before = (tmp_path / "data_manifest.json").read_bytes()
    with pytest.raises(ValueError, match="not empty"):
        await prepare_refund_data(
            tmp_path, train_count=8, validation_count=8, test_count=8
        )
    assert (tmp_path / "data_manifest.json").read_bytes() == before


@pytest.mark.asyncio
async def test_filter_replays_deduplicates_and_excludes_held_out_cases(tmp_path):
    data, output = tmp_path / "data", tmp_path / "filtered"
    await prepare_refund_data(data, train_count=8, validation_count=8, test_count=8)
    rows = json.loads((data / "train.json").read_text())
    candidate = await reference_refund_demonstration(rows[0])
    candidate["reported_reward"] = -999  # selection depends on execution, not this
    failed = copy.deepcopy(candidate)
    failed["messages"] = failed["messages"][:1] + [failed["messages"][-1]]
    failed["reported_reward"] = 999
    test_row = json.loads((data / "test.json").read_text())[0]
    test_candidate = await reference_refund_demonstration(test_row)
    source = tmp_path / "candidates.json"
    source.write_text(
        json.dumps(
            collection(
                rows, [candidate, candidate, failed, test_candidate], complete=False
            )
        )
    )
    manifest = await filter_refund_candidates(data, source, output)
    assert (manifest["selected"], manifest["not_selected"], manifest["rejected"]) == (
        1,
        1,
        2,
    )
    assert not manifest["collection_complete"]
    audit = json.loads((output / "replay_audit.json").read_text())
    assert audit[-1]["reason"] == "held_out_case"
    exported = json.loads((output / "train.jsonl").read_text())
    assert exported["metadata"]["source"] == "verified_model_rollout"
    assert exported["metadata"]["case_id"] == rows[0]["order_id"]


@pytest.mark.asyncio
async def test_filter_prefers_shorter_verified_conversation(tmp_path):
    from stateset_agents.core.environments.refund_policy_environment import (
        RefundPolicyEnvironment,
    )
    from stateset_agents.core.trajectory import ConversationTurn

    data = tmp_path / "data"
    await prepare_refund_data(data, train_count=1, validation_count=1, test_count=1)
    rows = json.loads((data / "train.json").read_text())
    short = await reference_refund_demonstration(rows[0])
    long = copy.deepcopy(short)
    # A second lookup is valid but unnecessary. Insert its actual observation.
    env = RefundPolicyEnvironment()
    state = await env.reset(rows[0])
    _, _, _, info = await env.step(state, ConversationTurn(**short["messages"][1]))
    long["messages"][3:3] = [copy.deepcopy(short["messages"][1]), *info["messages"]]
    await replay_refund_candidate(rows[0], long)
    source = tmp_path / "candidates.json"
    source.write_text(json.dumps(collection(rows, [long, short])))
    await filter_refund_candidates(data, source, tmp_path / "out")
    audit = json.loads((tmp_path / "out" / "replay_audit.json").read_text())
    assert [entry["status"] for entry in audit] == ["not_selected", "selected"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["dataset", "seed", "case_hashes", "checkpoint"])
async def test_filter_refuses_changed_inputs_before_export(tmp_path, mutation):
    data = tmp_path / "data"
    await prepare_refund_data(data, train_count=1, validation_count=1, test_count=1)
    rows = json.loads((data / "train.json").read_text())
    report = collection(rows, [])
    if mutation == "dataset":
        (data / "train.json").write_text("[]")
    elif mutation == "seed":
        report["seed"] = 43
    elif mutation == "case_hashes":
        report["case_hashes"] = {}
    else:
        report.pop("checkpoint")
    source = tmp_path / "candidates.json"
    source.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        await filter_refund_candidates(data, source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_cli_prepares_offline_and_empty_harvest_fails_without_training_file(tmp_path):
    data, source, output = (
        tmp_path / "data",
        tmp_path / "candidates.json",
        tmp_path / "out",
    )
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "prepare-refund-data",
            "--output",
            str(data),
            "--train-count",
            "8",
            "--validation-count",
            "8",
            "--test-count",
            "8",
        ],
    )
    assert result.exit_code == 0, result.output
    rows = json.loads((data / "train.json").read_text())
    source.write_text(json.dumps(collection(rows, [])))
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "filter-refund-data",
            "--data-dir",
            str(data),
            "--candidates",
            str(source),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 1, result.output
    assert (output / "replay_audit.json").exists()
    assert not (output / "train.jsonl").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupted", [False, True])
@pytest.mark.parametrize("timed_out", [False, True])
@pytest.mark.parametrize("wrong_input", [False, True])
async def test_native_collection_uses_only_train_split_and_preserves_partial_candidates(
    tmp_path, monkeypatch, interrupted, timed_out, wrong_input
):
    from stateset_agents.training.river_refund import campaign, prepare_run

    @dataclass
    class Checkpoint:
        path: str

    rows = refund_policy_benchmark("train", 1, 42)
    candidate = await reference_refund_demonstration(rows[0])

    class Engine:
        def __init__(self, model, **kwargs):
            assert kwargs["temperature"] == 1.0

        async def rollout(self, cases, **kwargs):
            assert cases == rows and kwargs["group_size"] == 8
            yield [
                fake_trajectory(
                    stateset_episode_id=rows[0]["order_id"],
                    stateset_case_hash=(
                        "0" * 64 if wrong_input else content_hash(rows[0])
                    ),
                    messages=candidate["messages"],
                    truncated=None,
                    stateset_truncated="environment_timeout" if timed_out else None,
                    reward=1,
                )
                for _ in range(8)
            ]
            if interrupted:
                raise ConnectionError("sampling session lost")

    def unexpected(*args, **kwargs):
        raise AssertionError("Collection must not construct an optimizer or evaluator")

    rl = SimpleNamespace(
        Env=object,
        Budget=lambda **kw: kw,
        Schedule=lambda **kw: kw,
        CheckpointSampler=lambda *a, **kw: kw,
        GroupCompletion=lambda **kw: kw,
        RolloutEngine=Engine,
        AsyncTrainer=unexpected,
        Evaluator=unexpected,
    )
    monkeypatch.setitem(
        sys.modules, "river_client", SimpleNamespace(rl=rl, Checkpoint=Checkpoint)
    )
    args = SimpleNamespace(
        output=tmp_path,
        benchmark=ENVIRONMENT,
        collect_only=True,
        evaluate_only=False,
        base_model="base",
        seed=42,
        concurrency=8,
        steps=2,
        max_staleness=0,
        learning_rate=1e-5,
        checkpoint=None,
        dry_run=False,
    )
    splits = {"train": rows, "validation": [], "test": []}
    prepare_run(args, splits)
    task = campaign(
        SimpleNamespace(save_weights=lambda *a, **kw: Checkpoint("river://sample")),
        object(),
        SimpleNamespace(tokenizer=object()),
        args,
        splits,
    )
    if wrong_input:
        with pytest.raises(ValueError, match="reset input"):
            await task
        report = json.loads((tmp_path / "training_candidates.json").read_text())
        assert not report["complete"]
        assert report["candidates"] == []
        return
    if interrupted:
        with pytest.raises(ConnectionError):
            await task
    else:
        await task
    report = json.loads((tmp_path / "training_candidates.json").read_text())
    assert report["complete"] is (not interrupted)
    assert report["run_manifest_hash"] == content_hash(
        json.loads((tmp_path / "run_manifest.json").read_text())
    )
    assert len(report["candidates"]) == 8
    assert all(
        c["truncated"] == ("environment_timeout" if timed_out else None)
        for c in report["candidates"]
    )
    assert all(
        c["environment_trace"]["case_hash"] == c["case_hash"]
        and c["environment_trace"]["terminal"]["truncated"] == c["truncated"]
        for c in report["candidates"]
    )
    assert not (tmp_path / "test_results.json").exists()
