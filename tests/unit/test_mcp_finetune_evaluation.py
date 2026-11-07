"""MCP evaluation and recovery use real snapshots with no model inference."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from stateset_agents import finetune_evaluation as evaluation
from stateset_agents import finetune_workspace as ws
from stateset_agents.cli_mcp_finetune import FinetuneTools
from tests.unit.test_finetune_evaluation import make_evaluation, run_pipeline
from tests.unit.test_mcp_finetune import ROOT, result


@pytest.fixture
def completed(tmp_path):
    run, config = make_evaluation(tmp_path)
    cases = run.parent / "held-out.json"
    cases.write_bytes(Path(config.cases).read_bytes())
    values = {**asdict(config), "cases": cases.name}
    return SimpleNamespace(
        run=run,
        workspace=run.parent,
        config=values,
        cases=cases,
        service=FinetuneTools(run.parent),
    )


@pytest.fixture
def submissions(monkeypatch):
    calls = []

    def start(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(pid=12345)

    monkeypatch.setattr(evaluation.subprocess, "Popen", start)
    monkeypatch.setattr(evaluation, "queue_executable", lambda: "/test-buildr")
    return calls


async def prepare(completed):
    service, run_id, config = completed.service, completed.run.name, completed.config
    review = result(await service.finetune_evaluation_plan(run_id, config))
    prepared = result(
        await service.finetune_evaluation_prepare(run_id, config, review["plan_sha256"])
    )
    return prepared


def arguments(completed, prepared):
    return completed.run.name, prepared["evaluation_id"], prepared["plan_sha256"]


async def test_review_preparation_and_reconnect_preserve_private_inputs(
    completed, submissions
):
    service = completed.service
    before = {p: p.read_bytes() for p in completed.run.rglob("*") if p.is_file()}
    targets = result(await service.finetune_evaluation_targets(completed.run.name))
    assert targets["targets"][0]["checkpoint"] == ""
    assert targets["bytes_verified"] is False
    review = result(
        await service.finetune_evaluation_plan(completed.run.name, completed.config)
    )
    assert not review["prepared"] and not review["launched"]
    assert "PRIVATE" not in json.dumps(review)
    assert before == {
        p: p.read_bytes() for p in completed.run.rglob("*") if p.is_file()
    }
    prepared = await prepare(completed)
    assert prepared["state"]["status"] == "prepared" and not prepared["launched"]
    path = completed.run / "evaluations" / prepared["evaluation_id"]
    assert ws.read_json(path / "cases.json") == ws.read_json(completed.cases)
    completed.cases.unlink()
    assert evaluation.read_evaluation(path).sha256 == prepared["plan_sha256"]
    restarted = FinetuneTools(completed.workspace)
    status = result(
        await restarted.finetune_evaluation_status(*arguments(completed, prepared)[:2])
    )
    assert status["plan_sha256"] == review["plan_sha256"]
    assert "PRIVATE" not in json.dumps(status)
    rows = result(await restarted.finetune_evaluations(completed.run.name))[
        "evaluations"
    ]
    assert rows[0]["evaluation_id"] == prepared["evaluation_id"]
    assert not submissions and not (completed.run / "launch.lock").exists()


@pytest.mark.parametrize("mutation", ["cases", "artifact", "checksum"])
async def test_stale_review_cannot_prepare(completed, mutation):
    review = result(
        await completed.service.finetune_evaluation_plan(
            completed.run.name, completed.config
        )
    )
    digest = review["plan_sha256"]
    if mutation == "cases":
        document = ws.read_json(completed.cases)
        document["cases"][0]["expected_resolution"] = "A different reviewed outcome"
        ws.write_json(completed.cases, document)
    elif mutation == "artifact":
        (
            completed.run / "artifacts/checkpoint-1/adapter_model.safetensors"
        ).write_bytes(b"changed")
    else:
        digest = "0" * 64
    response = await completed.service.finetune_evaluation_prepare(
        completed.run.name, completed.config, digest
    )
    assert response.isError and "checksum" in response.content[0].text
    assert not (completed.run / "evaluations").exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"unknown": "PRIVATE"},
        {"seed": True},
        {"judge_revision": "0" * 40},
        {"judge_model": "../../secret"},
        {"checkpoint": "checkpoint-1"},
    ],
)
async def test_invalid_evaluation_settings_do_not_prepare(completed, changes):
    response = await completed.service.finetune_evaluation_plan(
        completed.run.name, {**completed.config, **changes}
    )
    assert response.isError and "PRIVATE" not in response.model_dump_json()
    assert not (completed.run / "evaluations").exists()


async def test_provenance_and_training_overlap_errors_are_preserved(completed):
    document = ws.read_json(completed.cases)
    document["provenance"]["test_is_fresh"] = False
    ws.write_json(completed.cases, document)
    denied = await completed.service.finetune_evaluation_plan(
        completed.run.name, completed.config
    )
    assert denied.isError
    assert denied.structuredContent["error"]["code"] == "evaluation_provenance"
    assert (
        denied.structuredContent["error"]["diagnostic"]["expected_partition"] == "test"
    )
    document["provenance"]["test_is_fresh"] = True
    document["cases"][0]["user_turns"] = ["PRIVATE CUSTOMER"]
    ws.write_json(completed.cases, document)
    denied = await completed.service.finetune_evaluation_plan(
        completed.run.name, completed.config
    )
    assert denied.isError and "overlap" in denied.content[0].text
    assert "PRIVATE" not in denied.model_dump_json()


async def test_published_checkpoint_uses_reviewed_validation_partition(completed):
    document = ws.read_json(completed.cases)
    document["split"] = "validation"
    document["provenance"].pop("test_is_fresh")
    document["provenance"]["validation_is_fresh"] = True
    ws.write_json(completed.cases, document)
    completed.config.update(split="validation", checkpoint="checkpoint-1")
    prepared = await prepare(completed)
    assert prepared["split"] == "validation"
    assert prepared["checkpoint"] == "checkpoint-1"
    assert not prepared["quality_gain_verified"]


async def test_case_paths_and_evaluation_ids_stay_in_workspace(completed):
    for cases in (
        str(completed.workspace.parent / "held-out.json"),
        "../held-out.json",
    ):
        response = await completed.service.finetune_evaluation_plan(
            completed.run.name, {**completed.config, "cases": cases}
        )
        assert response.isError
    linked = completed.workspace / "linked.json"
    linked.symlink_to(completed.cases)
    assert (
        await completed.service.finetune_evaluation_plan(
            completed.run.name, {**completed.config, "cases": linked.name}
        )
    ).isError
    prepared = await prepare(completed)
    for evaluation_id in ("../outside", "/tmp", "bad", "a/b"):
        assert (
            await completed.service.finetune_evaluation_status(
                completed.run.name, evaluation_id
            )
        ).isError
    path = completed.run / "evaluations" / prepared["evaluation_id"]
    saved = path.with_name("saved")
    path.rename(saved)
    path.symlink_to(saved, target_is_directory=True)
    assert (
        await completed.service.finetune_evaluation_status(
            completed.run.name, prepared["evaluation_id"]
        )
    ).isError
    assert (
        result(await completed.service.finetune_evaluations(completed.run.name))[
            "evaluations"
        ]
        == []
    )


async def test_execution_requires_startup_capability_queue_and_review(
    completed, submissions, monkeypatch
):
    prepared = await prepare(completed)
    args = arguments(completed, prepared)
    assert (await completed.service.finetune_evaluation_launch(*args)).isError
    assert (await completed.service.finetune_evaluation_stop(*args)).isError
    enabled = FinetuneTools(completed.workspace, allow_training=True)
    assert (await enabled.finetune_evaluation_launch(*args[:2], "0" * 64)).isError
    assert (await enabled.finetune_evaluation_stop(*args[:2], "0" * 64)).isError
    monkeypatch.setattr(evaluation, "queue_executable", lambda: None)
    denied = await enabled.finetune_evaluation_launch(*args)
    assert denied.isError and "Buildr" in denied.content[0].text
    assert not submissions
    assert not (
        completed.run / "evaluations" / prepared["evaluation_id"] / "launch.lock"
    ).exists()


async def test_launch_once_retry_after_reconnect_and_separate_stop(
    completed, submissions
):
    prepared = await prepare(completed)
    args = arguments(completed, prepared)
    enabled = FinetuneTools(completed.workspace, allow_training=True)
    replies = await asyncio.gather(
        enabled.finetune_evaluation_launch(*args),
        enabled.finetune_evaluation_launch(*args),
    )
    assert [result(reply)["launched"] for reply in replies] == [True, False]
    restarted = FinetuneTools(completed.workspace, allow_training=True)
    assert result(await restarted.finetune_evaluation_launch(*args))[
        "already_submitted"
    ]
    assert len(submissions) == 1
    argv, options = submissions[0]
    assert "stateset_agents.finetune_evaluation_worker" in argv
    assert "--supervise" in argv and options.get("start_new_session")
    stopped = result(await restarted.finetune_evaluation_stop(*args))
    assert stopped["stop_requested"] and stopped["state"]["status"] == "cancelled"
    assert not (completed.run / "stop.json").exists()
    assert ws.run_status(completed.run)["status"] == "succeeded"


async def test_failed_submission_receipt_prevents_automatic_retry(
    completed, submissions, monkeypatch
):
    prepared = await prepare(completed)
    args = arguments(completed, prepared)
    attempts = []

    def fail(*args, **kwargs):
        attempts.append(True)
        raise OSError("supervisor submission failed")

    monkeypatch.setattr(evaluation.subprocess, "Popen", fail)
    enabled = FinetuneTools(completed.workspace, allow_training=True)
    assert (await enabled.finetune_evaluation_launch(*args)).isError
    restarted = FinetuneTools(completed.workspace, allow_training=True)
    retained = result(await restarted.finetune_evaluation_launch(*args))
    assert retained["already_submitted"] and not retained["launched"]
    assert retained["state"]["status"] == "failed"
    assert attempts == [True]


@pytest.mark.parametrize("mutation", ["snapshot", "artifact", "receipt"])
async def test_mutation_blocks_submission(completed, submissions, mutation):
    prepared = await prepare(completed)
    path = completed.run / "evaluations" / prepared["evaluation_id"]
    if mutation == "snapshot":
        document = ws.read_json(path / "cases.json")
        document["cases"][0]["expected_resolution"] = "Changed"
        ws.write_json(path / "cases.json", document)
    elif mutation == "artifact":
        (
            completed.run / "artifacts/checkpoint-1/adapter_model.safetensors"
        ).write_bytes(b"changed")
    else:
        ws.write_json(path / "launch.lock", {"plan_sha256": "0" * 64})
    enabled = FinetuneTools(completed.workspace, allow_training=True)
    assert (
        await enabled.finetune_evaluation_launch(*arguments(completed, prepared))
    ).isError
    assert not submissions


async def test_verified_evidence_replays_and_rejects_tampering(completed, monkeypatch):
    prepared = await prepare(completed)
    args = arguments(completed, prepared)[:2]
    assert (
        await completed.service.finetune_evaluation_status(*args, verify=True)
    ).isError
    path = completed.run / "evaluations" / prepared["evaluation_id"]
    # The shared pipeline fixture uses deterministic fake generators. Its seed
    # call needs no training runtime in the minimal MCP installation either.
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(set_seed=lambda _: None)
    )
    run_pipeline(path, monkeypatch)
    status = result(
        await completed.service.finetune_evaluation_status(*args, verify=True)
    )
    assert status["state"]["status"] == "succeeded"
    verified = status["verification"]
    assert verified["judge_replay_verified"] and not verified["quality_gain_verified"]
    assert verified["summary"]["regressions_total"] == 2
    assert "PRIVATE" not in json.dumps(status)
    (path / "judge-recordings/batch-000000/response-000.json").write_text("tampered")
    assert (
        await completed.service.finetune_evaluation_status(*args, verify=True)
    ).isError


async def test_lost_evaluation_owner_stays_unconfirmed_after_stop(
    completed, submissions
):
    prepared = await prepare(completed)
    args = arguments(completed, prepared)
    enabled = FinetuneTools(completed.workspace, allow_training=True)
    result(await enabled.finetune_evaluation_launch(*args))
    path = completed.run / "evaluations" / prepared["evaluation_id"]
    ws.write_json(path / "state.json", {"status": "running"})
    ws.write_json(
        path / "execution.json", {"pid": 999999999, "process_token": "missing"}
    )
    stopped = result(await enabled.finetune_evaluation_stop(*args))
    assert stopped["stop_requested"] and stopped["state"]["execution_unconfirmed"]
    assert stopped["state"]["status"] == "interrupted"


async def test_recovery_prepares_independent_run_and_separate_launch(
    completed, submissions
):
    choices = result(await completed.service.finetune_checkpoints(completed.run.name))
    choice = choices["checkpoints"][0]
    prepared = result(
        await completed.service.finetune_prepare_resume(
            completed.run.name, choice["checkpoint"], choice["recovery_sha256"]
        )
    )
    run = prepared["run"]
    child = completed.workspace / run["run_id"]
    assert prepared["parent_run_id"] == completed.run.name
    assert not prepared["launched"] and not submissions
    assert child != completed.run and run["state"]["status"] == "prepared"
    assert (child / "dataset").read_bytes() == (completed.run / "dataset").read_bytes()
    assert run["plan"]["resume"]["step"] == 1
    assert run["plan"]["config"] == ws.read_run_plan(completed.run)[0]["config"]
    assert (
        await completed.service.finetune_launch(run["run_id"], run["plan_sha256"])
    ).isError
    enabled = FinetuneTools(completed.workspace, allow_training=True)
    assert result(await enabled.finetune_launch(run["run_id"], run["plan_sha256"]))[
        "launched"
    ]
    assert len(submissions) == 1


@pytest.mark.parametrize("mutation", ["hash", "weights", "missing_optimizer"])
async def test_invalid_recovery_creates_no_child(completed, mutation):
    choice = result(await completed.service.finetune_checkpoints(completed.run.name))[
        "checkpoints"
    ][0]
    digest = choice["recovery_sha256"]
    artifact = completed.run / "artifacts/checkpoint-1"
    if mutation == "hash":
        digest = "0" * 64
    elif mutation == "weights":
        (artifact / "adapter_model.safetensors").write_bytes(b"changed")
    else:
        (artifact / "optimizer.pt").unlink()
    denied = await completed.service.finetune_prepare_resume(
        completed.run.name, choice["checkpoint"], digest
    )
    assert denied.isError
    assert list(completed.workspace.glob("*/plan.json")) == [
        completed.run / "plan.json"
    ]


@pytest.mark.slow
async def test_stdio_evaluation_review_resources_and_reconnect(completed):
    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "stateset_agents.cli",
            "mcp",
            "--finetune-workspace",
            str(completed.workspace),
        ],
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        },
        cwd=str(ROOT),
    )
    saved = None
    for _ in range(2):
        async with stdio_client(params, errlog=sys.__stderr__) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), 30)
                tools = {tool.name for tool in (await session.list_tools()).tools}
                assert "finetune_prepare_resume" in tools
                assert "finetune_evaluation_launch" not in tools
                arguments = {"run_id": completed.run.name, "config": completed.config}
                if saved is None:
                    review = result(
                        await session.call_tool("finetune_evaluation_plan", arguments)
                    )
                    saved = result(
                        await session.call_tool(
                            "finetune_evaluation_prepare",
                            {
                                **arguments,
                                "expected_plan_sha256": review["plan_sha256"],
                            },
                        )
                    )
                resource = await session.read_resource(saved["resource_uri"])
                status = json.loads(resource.contents[0].text)
                assert status["result"]["state"]["status"] == "prepared"
                resource = await session.read_resource(
                    f"finetune://runs/{completed.run.name}/evaluations"
                )
                assert (
                    json.loads(resource.contents[0].text)["result"]["evaluations"][0][
                        "evaluation_id"
                    ]
                    == saved["evaluation_id"]
                )
                failed = await session.call_tool(
                    "finetune_evaluation_status",
                    {
                        "run_id": completed.run.name,
                        "evaluation_id": saved["evaluation_id"],
                        "verify": True,
                    },
                )
                assert failed.isError and failed.structuredContent["ok"] is False
