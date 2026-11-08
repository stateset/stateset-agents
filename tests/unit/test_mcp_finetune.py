"""MCP fine-tuning orchestration without downloads, GPUs or paid submissions."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from stateset_agents import finetune_workspace as ws
from stateset_agents.cli import app
from stateset_agents.cli_mcp_finetune import FinetuneTools
from stateset_agents.mcp_server import create_server

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "workspace with spaces"
    root.mkdir()
    (root / "train.jsonl").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "PRIVATE CUSTOMER QUESTION"},
                    {"role": "assistant", "content": "PRIVATE SUPPORT ANSWER"},
                ]
            }
        )
        + "\n"
    )
    return root


def result(response):
    assert not response.isError, response
    assert response.structuredContent["ok"]
    assert json.loads(response.content[0].text) == response.structuredContent
    return response.structuredContent["result"]


async def prepare(service, **overrides):
    return result(
        await service.finetune_prepare({"dataset": "train.jsonl", **overrides})
    )["run"]


def submission(run):
    return run["run_id"], run["plan_sha256"]


@pytest.fixture
def starts(monkeypatch):
    calls = []

    def start(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(pid=12345)

    monkeypatch.setattr(ws.subprocess, "Popen", start)
    return calls


async def test_default_and_workspace_tool_capabilities(workspace):
    default = await create_server().list_tools()
    assert len(default) == 7
    for enabled, count in [(False, 18), (True, 23)]:
        server = create_server(finetune_workspace=workspace, allow_training=enabled)
        tools = {tool.name: tool for tool in await server.list_tools()}
        assert len(tools) == 7 + count
        assert ("finetune_launch" in tools) is enabled
        assert ("finetune_stop" in tools) is enabled
        assert ("finetune_evaluation_launch" in tools) is enabled
        assert ("finetune_evaluation_stop" in tools) is enabled
        assert ("finetune_validation_compare" in tools) is enabled
        assert tools["finetune_prepare"].annotations.idempotentHint is False
        assert tools["finetune_prepare_resume"].annotations.idempotentHint is False
        assert tools["finetune_evaluation_prepare"].annotations.readOnlyHint is False
        assert tools["finetune_nomination_prepare"].annotations.idempotentHint is False
        assert tools["finetune_nomination_status"].annotations.readOnlyHint is True
        if enabled:
            assert (
                tools["finetune_validation_compare"].annotations.idempotentHint is False
            )
        assert tools["finetune_logs"].annotations.readOnlyHint is True
        capabilities = result(await server.call_tool("finetune_capabilities", {}))
        assert capabilities["launch_enabled"] is enabled
        assert capabilities["paid_launch_enabled"] is False
        assert capabilities["evaluation_launch_enabled"] is enabled
        assert capabilities["validation_comparison_enabled"] is enabled
        assert (
            capabilities["checkpoint_validation_comparison"]["model_inference"] is False
        )
        assert capabilities["checkpoint_nomination"]["automatic_promotion"] is False
        assert capabilities["local_evaluation"]["config_schema"]["required"] == [
            "cases",
            "judge_model",
            "judge_revision",
        ]
        assert capabilities["backends"]["local"]["methods"] == [
            "sft",
            "grpo",
            "gspo",
            "dapo",
        ]
        assert capabilities["backends"]["river"]["methods"] == ["sft"]


@pytest.mark.parametrize("flags", [["--allow-training"], ["--allow-paid"]])
def test_incomplete_startup_permissions_fail(flags):
    response = CliRunner().invoke(app, ["mcp", *flags])
    assert response.exit_code == 2
    assert "requires" in response.output


def test_cli_passes_workspace_and_execution_flags(workspace, monkeypatch):
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(run=lambda **kwargs: calls.append(kwargs))

    monkeypatch.setattr("stateset_agents.mcp_server.create_server", factory)
    response = CliRunner().invoke(
        app,
        [
            "mcp",
            "--finetune-workspace",
            str(workspace),
            "--allow-training",
            "--allow-paid",
        ],
    )
    assert response.exit_code == 0, response.output
    assert calls == [
        {
            "weight_space": False,
            "finetune_workspace": workspace,
            "allow_training": True,
            "allow_paid": True,
        },
        {"transport": "stdio"},
    ]


def test_missing_sdk_prints_install_hint(monkeypatch):
    def missing():
        raise ImportError("Install stateset-agents[mcp]")

    monkeypatch.setattr("stateset_agents.mcp_server._require_mcp", missing)
    response = CliRunner().invoke(app, ["mcp"])
    assert response.exit_code == 1
    assert "stateset-agents[mcp]" in response.output


async def test_prepare_snapshots_data_and_survives_new_server(workspace, starts):
    service = FinetuneTools(workspace)
    run = await prepare(service)
    path = workspace / run["run_id"]
    assert run["state"]["status"] == "prepared"
    assert (path / "dataset").read_bytes() == (workspace / "train.jsonl").read_bytes()
    assert "PRIVATE CUSTOMER" not in json.dumps(run)
    (workspace / "train.jsonl").write_text("changed source")
    restarted = FinetuneTools(workspace)
    status = result(await restarted.finetune_status(run["run_id"]))
    assert status["plan_sha256"] == run["plan_sha256"]
    assert (
        result(await restarted.finetune_list_runs())["runs"][0]["run_id"]
        == run["run_id"]
    )
    assert not starts and not (path / "launch.lock").exists()


@pytest.mark.parametrize("method", ["grpo", "gspo", "dapo"])
async def test_prepare_native_rl_scenarios(workspace, method):
    (workspace / "scenarios.json").write_text(
        json.dumps(
            [
                {
                    "id": "refund",
                    "context": "Customer requests a refund",
                    "policy_facts": ["Refund defective items"],
                    "expected_resolution": "Refund",
                    "user_responses": ["It is defective"],
                }
            ]
        )
    )
    run = await prepare(
        FinetuneTools(workspace), method=method, dataset="scenarios.json"
    )
    assert run["plan"]["config"]["method"] == method
    assert run["plan"]["examples"] == 1


@pytest.mark.parametrize(
    "config", [{"epochs": True}, {"extra": 1}, {"dataset": "../outside.jsonl"}]
)
async def test_invalid_prepare_has_protocol_error_and_no_run(workspace, config):
    response = await FinetuneTools(workspace).finetune_prepare(
        {"dataset": "train.jsonl", **config}
    )
    assert response.isError and not response.structuredContent["ok"]
    assert not list(workspace.glob("*/plan.json"))


@pytest.mark.parametrize("run_id", ["..", "../elsewhere", "/tmp", "a/b", "a\\b"])
async def test_run_ids_cannot_escape_workspace(workspace, run_id):
    assert (await FinetuneTools(workspace).finetune_status(run_id)).isError


async def test_symlink_dataset_and_run_are_rejected(workspace):
    service = FinetuneTools(workspace)
    (workspace / "linked.jsonl").symlink_to(workspace / "train.jsonl")
    assert (await service.finetune_prepare({"dataset": "linked.jsonl"})).isError
    run = await prepare(service)
    (workspace / "linked-run").symlink_to(
        workspace / run["run_id"], target_is_directory=True
    )
    assert (await service.finetune_status("linked-run")).isError


async def test_disabled_controls_and_checksum_mismatch_do_not_submit(workspace, starts):
    reader = FinetuneTools(workspace)
    run = await prepare(reader)
    assert (await reader.finetune_launch(*submission(run))).isError
    assert (await reader.finetune_stop(*submission(run))).isError
    writer = FinetuneTools(workspace, allow_training=True)
    assert (await writer.finetune_launch(run["run_id"], "0" * 64)).isError
    assert (await writer.finetune_stop(run["run_id"], "0" * 64)).isError
    assert not starts


async def test_launch_once_retry_after_reconnect_and_stop(workspace, starts):
    service = FinetuneTools(workspace, allow_training=True)
    run = await prepare(service)
    replies = await asyncio.gather(
        service.finetune_launch(*submission(run)),
        service.finetune_launch(*submission(run)),
    )
    assert [result(reply)["launched"] for reply in replies] == [True, False]
    restarted = FinetuneTools(workspace, allow_training=True)
    assert result(await restarted.finetune_launch(*submission(run)))[
        "already_submitted"
    ]
    assert len(starts) == 1
    argv, options = starts[0]
    assert "stateset_agents.finetune_worker" in argv and "--supervise" in argv
    assert options["cwd"] == str(workspace)
    assert options.get("start_new_session") or options.get("creationflags")
    stopped = result(await restarted.finetune_stop(*submission(run)))
    assert stopped["stop_requested"] and stopped["status"] == "cancelled"
    assert (workspace / run["run_id"] / "stop.json").is_file()


async def test_mutated_snapshot_and_forged_launch_receipt_are_rejected(
    workspace, starts
):
    service = FinetuneTools(workspace, allow_training=True)
    run = await prepare(service)
    path = workspace / run["run_id"]
    (path / "dataset").write_text("tampered")
    assert (await service.finetune_launch(*submission(run))).isError
    ws.write_json(path / "launch.lock", {"schema_version": 1, "plan_sha256": "0" * 64})
    assert (await service.finetune_launch(*submission(run))).isError
    assert not starts


async def test_paid_launch_requires_startup_capability_and_credentials(
    workspace, starts, monkeypatch
):
    service = FinetuneTools(workspace, allow_training=True)
    run = await prepare(service, provider="river", cost_policy="provider_billed")
    denied = await service.finetune_launch(*submission(run))
    assert denied.isError and "--allow-paid" in denied.content[0].text
    paid = FinetuneTools(workspace, allow_training=True, allow_paid=True)
    monkeypatch.delenv("RIVER_API_KEY", raising=False)
    assert (await paid.finetune_launch(*submission(run))).isError
    assert not starts
    monkeypatch.setenv("RIVER_API_KEY", "test-secret-value")
    assert result(await paid.finetune_launch(*submission(run)))["launched"]
    assert len(starts) == 1


async def test_logs_are_bounded_redacted_and_reject_symlinks(workspace, monkeypatch):
    service = FinetuneTools(workspace)
    run = await prepare(service)
    path = workspace / run["run_id"] / "supervisor.log"
    monkeypatch.setenv("HF_TOKEN", "secret-token-value")
    path.write_text("first\n\x1b[31msecret-token-value\x1b[0m\nlast\n")
    response = await service.finetune_logs(
        run["run_id"], lines=1, offset=1, source="supervisor"
    )
    assert result(response)["lines"] == ["[REDACTED]"]
    for args in ({"lines": 201}, {"offset": -1}, {"source": "../../other"}):
        assert (await service.finetune_logs(run["run_id"], **args)).isError
    path.unlink()
    path.symlink_to(workspace / "train.jsonl")
    assert "PRIVATE" not in json.dumps(
        result(await service.finetune_logs(run["run_id"], source="supervisor"))
    )


async def test_status_returns_recent_metrics_and_redacts_errors(workspace, monkeypatch):
    service = FinetuneTools(workspace)
    run = await prepare(service)
    path = workspace / run["run_id"]
    (path / "metrics.jsonl").write_text(
        json.dumps({"step": 1, "metrics": {"loss": 0.5}}) + "\n"
    )
    status = result(await service.finetune_status(run["run_id"]))
    assert status["metrics"] == [
        {"step": 1, "metrics": {"loss": 0.5}, "updated_at": None}
    ]
    assert status["quality_gain_verified"] is False
    monkeypatch.setenv("HF_TOKEN", "secret-token-value")

    def fail(*args):
        raise ValueError("Failure with secret-token-value")

    monkeypatch.setattr("stateset_agents.cli_mcp_finetune.run_status", fail)
    response = await service.finetune_status(run["run_id"])
    assert response.isError
    assert "secret-token-value" not in response.model_dump_json()
    assert "[REDACTED]" in response.content[0].text


@pytest.mark.skipif(sys.platform != "linux", reason="Recovery requires Linux ownership")
async def test_checkpoint_inspection_requires_terminal_run(workspace):
    service = FinetuneTools(workspace, allow_training=True)
    run = await prepare(service, revision="a" * 40)
    assert (await service.finetune_checkpoints(run["run_id"])).isError
    result(await service.finetune_stop(*submission(run)))
    checkpoints = result(await service.finetune_checkpoints(run["run_id"]))
    assert checkpoints["parent_run_id"] == run["run_id"]
    assert checkpoints["checkpoints"] == []


async def test_stop_does_not_claim_unconfirmed_worker_exit(workspace, starts):
    service = FinetuneTools(workspace, allow_training=True)
    run = await prepare(service)
    result(await service.finetune_launch(*submission(run)))
    path = workspace / run["run_id"]
    ws.write_json(path / "state.json", {"status": "running"})
    ws.write_json(
        path / "execution.json", {"pid": 999999999, "process_token": "missing"}
    )
    stopped = result(await service.finetune_stop(*submission(run)))
    assert stopped["stop_requested"] and stopped["execution_unconfirmed"]
    assert stopped["terminal"] is None


async def test_cancelled_request_keeps_submission_owned_until_finished(workspace):
    service = FinetuneTools(workspace)
    entered, release = threading.Event(), threading.Event()

    def action():
        entered.set()
        assert release.wait(5)
        return {"finished": True}

    task = asyncio.create_task(service._call("test", action))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert service._gate.locked() and not task.done()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not service._gate.locked()


@pytest.mark.slow
async def test_real_stdio_prepare_resources_errors_and_restart(workspace):
    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "stateset_agents.cli",
            "mcp",
            "--finetune-workspace",
            str(workspace),
        ],
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        },
        cwd=str(ROOT),
    )
    run_id = None
    for _ in range(2):
        async with stdio_client(params, errlog=sys.__stderr__) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), 30)
                names = {tool.name for tool in (await session.list_tools()).tools}
                assert "finetune_prepare" in names and "finetune_launch" not in names
                if run_id is None:
                    run = result(
                        await session.call_tool(
                            "finetune_prepare", {"config": {"dataset": "train.jsonl"}}
                        )
                    )["run"]
                    run_id = run["run_id"]
                status = result(
                    await session.call_tool("finetune_status", {"run_id": run_id})
                )
                assert status["state"]["status"] == "prepared"
                resources = {
                    str(resource.uri)
                    for resource in (await session.list_resources()).resources
                }
                assert {"finetune://capabilities", "finetune://runs"} <= resources
                resource = await session.read_resource(f"finetune://runs/{run_id}")
                assert (
                    json.loads(resource.contents[0].text)["result"]["run_id"] == run_id
                )
                error = await session.call_tool(
                    "finetune_status", {"run_id": "../outside"}
                )
                assert error.isError and error.structuredContent["ok"] is False
                assert (
                    await session.call_tool("finetune_list_runs", {"limit": True})
                ).isError
