"""MCP checkpoint selection with real evidence replay and mocked model/queue I/O."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from stateset_agents import finetune_validation as comparison
from stateset_agents import finetune_workspace as ws
from stateset_agents.cli_mcp_finetune import FinetuneTools
from stateset_agents.mcp_server import create_server
from tests.unit.test_finetune_validation_comparison import cohort, queued
from tests.unit.test_mcp_finetune import ROOT, result


@pytest.fixture
def validated(tmp_path, monkeypatch):
    # The cohort uses deterministic fake generators; seed setup needs no ML
    # dependencies. Recorded judge evidence and its verification remain real.
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(set_seed=lambda _: None)
    )
    run, config, paths = cohort(tmp_path, monkeypatch)
    return SimpleNamespace(
        run=run,
        paths=paths,
        config=config,
        workspace=run.parent,
        service=FinetuneTools(run.parent, allow_training=True),
    )


async def compared(validated, monkeypatch):
    queued(monkeypatch, execute=True)
    return result(
        await validated.service.finetune_validation_compare(
            validated.run.name, validated.paths[-1].name
        )
    )["comparison"]


async def nominated(validated, monkeypatch):
    saved = await compared(validated, monkeypatch)
    reviewed = result(
        await validated.service.finetune_nomination_plan(
            validated.run.name, saved["comparison_id"]
        )
    )
    return result(
        await validated.service.finetune_nomination_prepare(
            validated.run.name, saved["comparison_id"], reviewed["plan_sha256"]
        )
    )


def fresh_cases(validated, *, overlap=None):
    document = ws.read_json(validated.paths[0] / "cases.json")
    document["split"] = "test"
    document["provenance"] = {
        "source": "reviewed",
        "reviewer": "Synthetic fixture",
        "test_is_fresh": True,
    }
    for case in document["cases"]:
        if overlap != "id":
            case["id"] = "fresh-" + case["id"]
        if overlap != "turn":
            case["user_turns"] = [
                "Fresh assessment: " + turn for turn in case["user_turns"]
            ]
    path = validated.workspace / "fresh.json"
    ws.write_json(path, document)
    return path.name


async def restored_config(validated, receipt):
    return result(
        await validated.service.finetune_nomination_config(
            validated.run.name, receipt["nomination_id"], receipt["plan_sha256"]
        )
    )["config"]


async def test_comparison_requires_execution_and_queue(validated, monkeypatch):
    calls = queued(monkeypatch)
    reader = FinetuneTools(validated.workspace)
    assert (
        await reader.finetune_validation_compare(
            validated.run.name, validated.paths[-1].name
        )
    ).isError
    monkeypatch.setattr(comparison, "queue_executable", lambda: None)
    denied = await validated.service.finetune_validation_compare(
        validated.run.name, validated.paths[-1].name
    )
    assert denied.isError
    assert denied.structuredContent["error"]["code"] == "invalid_validation_comparison"
    assert "Buildr" in denied.content[0].text
    assert not calls and not (validated.run / comparison.DIRECTORY).exists()
    assert (
        result(await reader.finetune_validation_comparison_status(validated.run.name))[
            "comparison"
        ]
        is None
    )


async def test_comparison_replays_evidence_and_recommends_earliest_safe_checkpoint(
    validated, monkeypatch
):
    before = {
        name: (validated.run / name).read_bytes()
        for name in ("plan.json", "state.json")
    }
    saved = await compared(validated, monkeypatch)
    assert saved["state"]["status"] == "succeeded"
    assert set(saved["evaluation_ids"]) == {p.name for p in validated.paths}
    report = saved["result"]
    assert report["comparison_complete"]
    assert report["recommendation"]["checkpoint"] == "checkpoint-1"
    assert [row["checkpoint"] for row in report["rows"]] == [
        "checkpoint-1",
        "checkpoint-3",
        "checkpoint-2",
    ]
    assert not report["rows"][-1]["eligible"]
    assert not report["quality_gain_verified"] and "PRIVATE" not in json.dumps(saved)
    assert not saved["training_started"] and not saved["inference_started"]
    assert {name: (validated.run / name).read_bytes() for name in before} == before
    assert not (validated.run / "quality.json").exists()


async def test_active_comparison_reconnect_and_explicit_resubmission(
    validated, monkeypatch
):
    calls = queued(monkeypatch)
    anchor = validated.paths[-1].name
    first = result(
        await validated.service.finetune_validation_compare(validated.run.name, anchor)
    )["comparison"]
    reader = FinetuneTools(validated.workspace)
    reopened = result(
        await reader.finetune_validation_comparison_status(validated.run.name)
    )["comparison"]
    assert reopened["comparison_id"] == first["comparison_id"] and len(calls) == 1
    # The backend coalesces active work for the same anchor. Expose what that
    # request actually captured, even if this caller supplied a different subset.
    reused = result(
        await validated.service.finetune_validation_compare(
            validated.run.name, anchor, [anchor]
        )
    )["comparison"]
    assert reused["comparison_id"] == first["comparison_id"]
    assert set(reused["evaluation_ids"]) == {p.name for p in validated.paths}
    assert len(calls) == 1
    resubmitted = result(
        await validated.service.finetune_validation_compare(
            validated.run.name, anchor, [anchor], resubmit=True
        )
    )["comparison"]
    assert resubmitted["comparison_id"] != first["comparison_id"]
    assert resubmitted["evaluation_ids"] == [anchor] and len(calls) == 2
    assert not (
        await reader.finetune_validation_comparison_status(
            validated.run.name, first["comparison_id"]
        )
    ).isError


async def test_uncertain_submission_is_reported_without_automatic_retry(
    validated, monkeypatch
):
    calls = queued(
        monkeypatch,
        response=SimpleNamespace(
            returncode=1, stdout="PRIVATE queue details", stderr="private"
        ),
    )
    saved = result(
        await validated.service.finetune_validation_compare(
            validated.run.name, validated.paths[-1].name
        )
    )["comparison"]
    assert saved["state"]["status"] == "submission_uncertain"
    assert "PRIVATE" not in json.dumps(saved)
    result(
        await validated.service.finetune_validation_comparison_status(
            validated.run.name, saved["comparison_id"]
        )
    )
    assert len(calls) == 1
    assert (
        await validated.service.finetune_nomination_plan(
            validated.run.name, saved["comparison_id"]
        )
    ).isError


async def test_explicit_comparison_subset_is_bounded_by_tool_schema(validated):
    server = create_server(finetune_workspace=validated.workspace, allow_training=True)
    tool = next(
        tool
        for tool in await server.list_tools()
        if tool.name == "finetune_validation_compare"
    )
    branches = tool.inputSchema["properties"]["evaluation_ids"]["anyOf"]
    assert (
        next(branch for branch in branches if branch.get("type") == "array")["maxItems"]
        == 100
    )
    denied = await validated.service.finetune_validation_compare(
        validated.run.name, validated.paths[-1].name, ["../outside"]
    )
    assert denied.isError
    assert not (validated.run / comparison.DIRECTORY).exists()


async def test_nomination_review_preparation_and_blank_fresh_test_config(
    validated, monkeypatch
):
    saved = await compared(validated, monkeypatch)
    reader = FinetuneTools(validated.workspace)
    review = result(
        await reader.finetune_nomination_plan(
            validated.run.name, saved["comparison_id"]
        )
    )
    assert not review["prepared"] and not review["inference_started"]
    assert not (validated.run / "nominations").exists()
    assert (
        await reader.finetune_nomination_prepare(
            validated.run.name, saved["comparison_id"], "0" * 64
        )
    ).isError
    assert not (validated.run / "nominations").exists()
    receipt = result(
        await reader.finetune_nomination_prepare(
            validated.run.name, saved["comparison_id"], review["plan_sha256"]
        )
    )
    status = result(
        await reader.finetune_nomination_status(
            validated.run.name, receipt["nomination_id"]
        )
    )
    assert status["checkpoint"] == "checkpoint-1" and not status["bytes_verified"]
    assert not receipt["training_started"] and not receipt["inference_started"]
    assert (
        result(await reader.finetune_nominations(validated.run.name))["nominations"][0][
            "nomination_id"
        ]
        == receipt["nomination_id"]
    )
    config = await restored_config(validated, receipt)
    assert config["cases"] == "" and config["split"] == "test"
    assert (
        config["checkpoint"] == "checkpoint-1"
        and config["nomination_id"] == receipt["nomination_id"]
    )
    assert config["judge_revision"] == validated.config.judge_revision
    assert (
        await reader.finetune_nomination_config(
            validated.run.name, receipt["nomination_id"], "0" * 64
        )
    ).isError
    assert len(list((validated.run / "evaluations").iterdir())) == 3


async def test_nomination_prepares_fresh_test_for_selected_checkpoint(
    validated, monkeypatch
):
    receipt = await nominated(validated, monkeypatch)
    config = await restored_config(validated, receipt)
    config["cases"] = fresh_cases(validated)
    review = result(
        await validated.service.finetune_evaluation_plan(validated.run.name, config)
    )
    binding = review["plan"]["binding"]
    assert binding["checkpoint"] == "checkpoint-1"
    assert binding["nomination"]["nomination_id"] == receipt["nomination_id"]
    prepared = result(
        await validated.service.finetune_evaluation_prepare(
            validated.run.name, config, review["plan_sha256"]
        )
    )
    assert prepared["split"] == "test" and prepared["checkpoint"] == "checkpoint-1"
    assert prepared["state"]["status"] == "prepared" and not prepared["launched"]
    assert "PRIVATE" not in json.dumps(prepared)


@pytest.mark.parametrize("overlap", ["id", "turn"])
async def test_nomination_rejects_validation_reuse_as_fresh_test(
    validated, monkeypatch, overlap
):
    receipt = await nominated(validated, monkeypatch)
    config = await restored_config(validated, receipt)
    config["cases"] = fresh_cases(validated, overlap=overlap)
    denied = await validated.service.finetune_evaluation_plan(
        validated.run.name, config
    )
    assert denied.isError and "overlap retained validation" in denied.content[0].text
    assert "PRIVATE" not in denied.model_dump_json()
    assert len(list((validated.run / "evaluations").iterdir())) == 3


@pytest.mark.parametrize("change", ["receipt", "comparison", "weights"])
async def test_changed_selection_cannot_authorize_test_preparation(
    validated, monkeypatch, change
):
    receipt = await nominated(validated, monkeypatch)
    config = await restored_config(validated, receipt)
    config["cases"] = fresh_cases(validated)
    if change == "receipt":
        path = validated.run / "nominations" / receipt["nomination_id"] / "receipt.json"
        value = ws.read_json(path)
        value["plan"]["checkpoint"] = "checkpoint-3"
        ws.write_json(path, value)
    elif change == "comparison":
        path = (
            comparison.comparison_path(validated.run, receipt["comparison_id"])
            / "result.json"
        )
        value = ws.read_json(path)
        value["verified_at"] += 1
        ws.write_json(path, value)
    else:
        (
            validated.run / "artifacts/checkpoint-1/adapter_model.safetensors"
        ).write_bytes(b"changed")
    assert (
        await validated.service.finetune_evaluation_plan(validated.run.name, config)
    ).isError
    assert len(list((validated.run / "evaluations").iterdir())) == 3
    if change != "weights":
        history = result(
            await validated.service.finetune_nominations(validated.run.name)
        )
        assert history["nominations"][0]["state"] == "unavailable"


async def test_incomplete_comparison_cannot_nominate(validated, monkeypatch):
    queued(monkeypatch)
    saved = result(
        await validated.service.finetune_validation_compare(
            validated.run.name, validated.paths[-1].name
        )
    )["comparison"]
    assert (
        await validated.service.finetune_nomination_plan(
            validated.run.name, saved["comparison_id"]
        )
    ).isError
    assert not (validated.run / "nominations").exists()


async def test_record_ids_and_symlinks_cannot_escape_workspace(validated, monkeypatch):
    receipt = await nominated(validated, monkeypatch)
    for identifier in ("../outside", "/tmp", "bad/id"):
        assert (
            await validated.service.finetune_validation_comparison_status(
                validated.run.name, identifier
            )
        ).isError
        assert (
            await validated.service.finetune_nomination_plan(
                validated.run.name, identifier
            )
        ).isError
        assert (
            await validated.service.finetune_nomination_status(
                validated.run.name, identifier
            )
        ).isError
    path = validated.run / "nominations" / receipt["nomination_id"]
    saved = path.with_name("saved")
    path.rename(saved)
    path.symlink_to(saved, target_is_directory=True)
    assert (
        await validated.service.finetune_nomination_status(
            validated.run.name, receipt["nomination_id"]
        )
    ).isError
    path = comparison.comparison_path(validated.run, receipt["comparison_id"])
    saved = path.with_name("saved")
    path.rename(saved)
    path.symlink_to(saved, target_is_directory=True)
    assert (
        await validated.service.finetune_validation_comparison_status(
            validated.run.name, receipt["comparison_id"]
        )
    ).isError


@pytest.mark.slow
async def test_stdio_selection_resources_and_fresh_test_handoff(validated, monkeypatch):
    saved = await compared(validated, monkeypatch)
    case_path = fresh_cases(validated)
    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "stateset_agents.cli",
            "mcp",
            "--finetune-workspace",
            str(validated.workspace),
        ],
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        },
        cwd=str(ROOT),
    )
    receipt = None
    for _ in range(2):
        async with stdio_client(params, errlog=sys.__stderr__) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), 30)
                tools = {tool.name for tool in (await session.list_tools()).tools}
                assert (
                    "finetune_nomination_prepare" in tools
                    and "finetune_validation_compare" not in tools
                )
                resource = await session.read_resource(saved["resource_uri"])
                assert (
                    json.loads(resource.contents[0].text)["result"]["comparison"][
                        "comparison_id"
                    ]
                    == saved["comparison_id"]
                )
                resource = await session.read_resource(
                    f"finetune://runs/{validated.run.name}/comparisons"
                )
                assert (
                    json.loads(resource.contents[0].text)["result"]["comparison"][
                        "comparison_id"
                    ]
                    == saved["comparison_id"]
                )
                if receipt is None:
                    args = {
                        "run_id": validated.run.name,
                        "comparison_id": saved["comparison_id"],
                    }
                    reviewed = result(
                        await session.call_tool("finetune_nomination_plan", args)
                    )
                    receipt = result(
                        await session.call_tool(
                            "finetune_nomination_prepare",
                            {**args, "expected_plan_sha256": reviewed["plan_sha256"]},
                        )
                    )
                resource = await session.read_resource(receipt["resource_uri"])
                assert (
                    json.loads(resource.contents[0].text)["result"]["checkpoint"]
                    == "checkpoint-1"
                )
                resource = await session.read_resource(
                    f"finetune://runs/{validated.run.name}/nominations"
                )
                assert (
                    json.loads(resource.contents[0].text)["result"]["nominations"][0][
                        "nomination_id"
                    ]
                    == receipt["nomination_id"]
                )
                restored = result(
                    await session.call_tool(
                        "finetune_nomination_config",
                        {
                            "run_id": validated.run.name,
                            "nomination_id": receipt["nomination_id"],
                            "expected_plan_sha256": receipt["plan_sha256"],
                        },
                    )
                )
                assert restored["config"]["cases"] == ""
                restored["config"]["cases"] = case_path
                review = result(
                    await session.call_tool(
                        "finetune_evaluation_plan",
                        {"run_id": validated.run.name, "config": restored["config"]},
                    )
                )
                assert review["plan"]["binding"]["checkpoint"] == "checkpoint-1"
