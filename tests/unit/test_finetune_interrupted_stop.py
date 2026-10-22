"""Supervisor loss must not prevent cooperative stop of detached local work."""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from stateset_agents import cli_finetune as cli
from stateset_agents import finetune_worker as worker
from stateset_agents import finetune_workspace as workspace
from stateset_agents.tui import Studio


@pytest.fixture
def interrupted_run(tmp_path, monkeypatch):
    source = tmp_path / "chat.jsonl"
    source.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "PRIVATE-CUSTOMER training text"},
                    {"role": "assistant", "content": "A private fixture reply"},
                ]
            }
        )
        + "\n"
    )
    config = workspace.FinetuneConfig(dataset=str(source))
    run = workspace.create_run(tmp_path / "workspace", workspace.plan_run(config))
    _, digest = workspace.read_run_plan(run)
    workspace.write_json(
        run / "launch.lock", {"schema_version": 1, "plan_sha256": digest}
    )
    workspace.write_json(run / "state.json", {"status": "queued"})
    workspace.write_json(
        run / "supervisor.json", {"pid": 100001, "process_token": "original"}
    )
    # Control provenance deterministically; never inspect or signal a real PID.
    monkeypatch.setattr(
        workspace, "sys", SimpleNamespace(platform="linux", executable=sys.executable)
    )
    monkeypatch.setattr(workspace, "process_token", lambda _pid: None)
    monkeypatch.setattr(
        workspace.os, "kill", lambda *_: pytest.fail("Stop must not signal processes")
    )
    monkeypatch.setattr(
        workspace.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("Stop must not start a process or queue"),
    )
    return SimpleNamespace(run=run, digest=digest, config=config)


@pytest.mark.parametrize("saved_status", ["queued", "running", "interrupted"])
@pytest.mark.parametrize("supervisor_token", [None, "reused-pid"])
def test_interrupted_work_can_be_stopped_without_inferring_worker_exit(
    interrupted_run, monkeypatch, saved_status, supervisor_token
):
    fixture = interrupted_run
    workspace.write_json(fixture.run / "state.json", {"status": saved_status})
    workspace.write_json(
        fixture.run / "execution.json", {"pid": 200001, "process_token": "worker"}
    )
    monkeypatch.setattr(
        workspace,
        "process_token",
        lambda pid: "worker" if pid == 200001 else supervisor_token,
    )
    before = (fixture.run / "state.json").read_bytes()
    status = workspace.run_status(fixture.run)
    assert status["status"] == "interrupted"
    assert status["execution_unconfirmed"] is True
    assert workspace.run_terminal_status(status) is None
    assert (fixture.run / "state.json").read_bytes() == before

    response = workspace.request_stop(fixture.run, expected_plan_sha256=fixture.digest)
    assert response["status"] == "interrupted"
    assert response["terminal"] is None
    assert response["execution_unconfirmed"] is True
    assert response["stop_requested"] is True
    assert workspace.stop_requested(fixture.run)
    assert (fixture.run / "state.json").read_bytes() == before
    raw = (fixture.run / "stop.json").read_bytes()
    repeated = workspace.request_stop(fixture.run, expected_plan_sha256=fixture.digest)
    assert repeated["stop_requested"] is True
    assert (fixture.run / "stop.json").read_bytes() == raw
    if os.name != "nt":  # Windows st_mode carries no POSIX group/other bits.
        assert (fixture.run / "stop.json").stat().st_mode & 0o077 == 0
    # An already running learner consumes the same receipt at its next boundary.
    with pytest.raises(workspace.FinetuneFailure) as stopped:
        worker._check_stop(fixture.run)
    assert stopped.value.code == "cancelled"


def test_detached_queued_worker_honors_stop_before_model_or_queue_access(
    interrupted_run, monkeypatch
):
    fixture = interrupted_run
    token = "a" * 64
    workspace.write_json(
        fixture.run / "admission.json",
        {
            "schema_version": 1,
            "mode": "supervised",
            "pid": 100001,
            "process_token": "original",
            "supervisor_token": token,
        },
    )
    monkeypatch.setattr(worker, "process_token", lambda _pid: "queued-child")
    workspace.request_stop(fixture.run, expected_plan_sha256=fixture.digest)
    monkeypatch.setattr(worker.os, "umask", lambda _value: 0)
    monkeypatch.setattr(
        worker, "queue_executable", lambda: pytest.fail("No queue access")
    )
    monkeypatch.setattr(
        worker, "_train_sft", lambda *_: pytest.fail("No model loading or training")
    )
    assert worker.execute(fixture.run, supervisor_token=token) == 1
    status = workspace.run_status(fixture.run)
    assert status["status"] == "cancelled"
    assert workspace.run_terminal_status(status) is True
    assert status["execution_unconfirmed"] is False
    assert not (fixture.run / "completion.json").exists()


@pytest.mark.parametrize("invalid", ["checksum", "package", "remote"])
def test_interrupted_stop_keeps_reviewed_plan_code_and_local_scope_guards(
    interrupted_run, invalid
):
    fixture = interrupted_run
    digest = fixture.digest
    if invalid == "checksum":
        digest = "0" * 64
    else:
        plan, _ = workspace.read_run_plan(fixture.run)
        if invalid == "package":
            plan["package_implementation_sha256"] = "0" * 64
        else:
            plan["config"]["provider"] = "river"
        workspace.write_json(fixture.run / "plan.json", plan)
        _, digest = workspace.read_run_plan(fixture.run)
    before = (fixture.run / "state.json").read_bytes()
    with pytest.raises(ValueError):
        workspace.request_stop(fixture.run, expected_plan_sha256=digest)
    assert not (fixture.run / "stop.json").exists()
    assert (fixture.run / "state.json").read_bytes() == before


@pytest.mark.parametrize("status", ["succeeded", "failed", "cancelled"])
def test_confirmed_terminal_runs_stay_terminal_and_out_of_active_filter(
    interrupted_run, status
):
    fixture = interrupted_run
    workspace.write_json(fixture.run / "state.json", {"status": status})
    result = workspace.request_stop(fixture.run, expected_plan_sha256=fixture.digest)
    assert result["status"] == status
    assert result["terminal"] is True
    assert result["execution_unconfirmed"] is False
    assert result["stop_requested"] is False
    assert not (fixture.run / "stop.json").exists()
    studio = Studio(fixture.run.parent)
    studio.run_filter = "active"
    studio.refresh()
    assert studio.runs == []


@pytest.mark.parametrize("status", ["prepared", "starting", "queued", "running"])
def test_known_active_statuses_have_definitive_nonterminal_metadata(status):
    assert workspace.run_terminal_status({"status": status}) is False


@pytest.mark.parametrize("status", ["interrupted", "unreadable", "unknown", None, []])
def test_unknown_execution_never_claims_terminal_completion(status):
    assert workspace.run_terminal_status({"status": status}) is None


def test_tui_keeps_interrupted_work_visible_and_records_explicit_stop(interrupted_run):
    fixture = interrupted_run
    studio = Studio(fixture.run.parent)
    studio.tab, studio.run_filter = "runs", "active"
    studio.refresh()
    draft = dict(studio.values)
    assert [row["id"] for row in studio.runs] == [fixture.run.name]
    assert "Execution: unconfirmed" in "\n".join(studio._run_details())
    studio.handle("k")
    assert studio.overlay == "stop"
    studio.handle("ESC")
    assert not (fixture.run / "stop.json").exists()
    studio.handle("k")
    studio.handle("ENTER")
    assert workspace.stop_requested(fixture.run)
    assert "worker exit remains unconfirmed" in studio.message
    assert "already terminal" not in studio.message
    assert studio.values == draft
    assert studio.runs[0]["state"]["status"] == "interrupted"


def test_cli_status_and_stop_report_unconfirmed_execution(interrupted_run):
    fixture = interrupted_run
    runner = CliRunner()
    argv = [fixture.run.name, "--workspace", str(fixture.run.parent)]
    response = runner.invoke(cli.app, ["status", *argv])
    assert response.exit_code == 0, response.stdout
    run = json.loads(response.stdout)["result"]["run"]
    assert run["terminal"] is None
    assert run["state"]["execution_unconfirmed"] is True
    assert "PRIVATE-CUSTOMER" not in response.stdout
    response = runner.invoke(
        cli.app,
        ["stop", *argv, "--expected-plan-sha256", run["plan_sha256"]],
    )
    assert response.exit_code == 0, response.stdout
    stop = json.loads(response.stdout)["result"]
    assert stop["stop_requested"] is True
    assert stop["terminal"] is None and stop["execution_unconfirmed"] is True


@pytest.mark.asyncio
async def test_http_stop_and_metadata_keep_supervisor_loss_explicit(
    interrupted_run, monkeypatch
):
    fastapi = pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")
    from stateset_agents.api.dependencies import get_strict_current_user
    from stateset_agents.api.errors import setup_exception_handlers
    from stateset_agents.api.routers import finetune

    fixture = interrupted_run
    monkeypatch.setenv("STATESET_FINETUNE_WORKSPACE", str(fixture.run.parent))
    app = fastapi.FastAPI()
    setup_exception_handlers(app)
    app.include_router(finetune.router)
    app.dependency_overrides[get_strict_current_user] = lambda: SimpleNamespace(
        user_id="operator", roles=["admin"]
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        endpoint = f"/finetune/runs/{fixture.run.name}"
        response = await client.get(endpoint)
        assert response.status_code == 200
        review = response.json()["result"]
        assert review["status"] == "interrupted"
        assert review["terminal"] is None and review["execution_unconfirmed"] is True
        assert "PRIVATE-CUSTOMER" not in response.text
        invalid = await client.post(
            endpoint + "/stop", json={"expected_plan_sha256": "0" * 64}
        )
        assert invalid.status_code == 400 and not (fixture.run / "stop.json").exists()
        response = await client.post(
            endpoint + "/stop", json={"expected_plan_sha256": review["plan_sha256"]}
        )
        assert response.status_code == 200
        stop = response.json()["result"]
        assert stop["stop_requested"] is True
        assert stop["terminal"] is None and stop["execution_unconfirmed"] is True
        raw = (fixture.run / "stop.json").read_bytes()
        repeated = await client.post(
            endpoint + "/stop", json={"expected_plan_sha256": review["plan_sha256"]}
        )
        assert repeated.status_code == 200
        assert (fixture.run / "stop.json").read_bytes() == raw
        rows = (await client.get("/finetune/runs")).json()["result"]["runs"]
        assert rows[0]["terminal"] is None
        assert rows[0]["execution_unconfirmed"] is True
        assert rows[0]["stop_requested"] is True
