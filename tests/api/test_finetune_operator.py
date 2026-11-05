"""Authenticated, bounded HTTP access to the operator's fine-tuning workspace."""

import asyncio
import os
import threading
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from stateset_agents import finetune_workspace as workspace
from stateset_agents.api.dependencies import get_strict_current_user
from stateset_agents.api.errors import setup_exception_handlers
from stateset_agents.api.routers import finetune
from stateset_agents.finetune_workspace import read_run_plan, write_json
from tests.api.asgi_client import SyncASGIClient
from tests.unit.test_finetune_quality import evidence_for, make_completed


@pytest.fixture
def completed(tmp_path):
    return make_completed(tmp_path)


@pytest.fixture
def app(completed, monkeypatch):
    run, _ = completed
    monkeypatch.setenv("STATESET_FINETUNE_WORKSPACE", str(run.parent))
    application = FastAPI()
    setup_exception_handlers(application)
    application.state.config = SimpleNamespace(
        security=SimpleNamespace(require_auth=True)
    )
    application.include_router(finetune.router)
    application.dependency_overrides[get_strict_current_user] = lambda: SimpleNamespace(
        user_id="operator", roles=["admin"]
    )
    return application


def test_capabilities_and_validation_do_not_read_workspace_or_start_training(
    app, monkeypatch
):
    monkeypatch.delenv("STATESET_FINETUNE_WORKSPACE")
    with SyncASGIClient(app) as client:
        response = client.get("/finetune/capabilities")
        assert (
            response.status_code == 200
            and not response.json()["result"]["launch_supported"]
        )
        response = client.post(
            "/finetune/validate", json={"epochs": 0, "dataset": "PRIVATE"}
        )
        assert response.status_code == 200
        assert not response.json()["result"]["valid"] and "PRIVATE" not in response.text
        assert client.get("/finetune/runs").status_code == 503


@pytest.mark.parametrize("mode", ["anonymous", "trainer", "auth_disabled"])
def test_workspace_routes_fail_closed_without_authenticated_admin_access(
    app, mode, monkeypatch
):
    if mode == "anonymous":

        def reject():
            raise HTTPException(status_code=401, detail="Authentication required")

        app.dependency_overrides[get_strict_current_user] = reject
    elif mode == "trainer":
        app.dependency_overrides[get_strict_current_user] = lambda: SimpleNamespace(
            roles=["trainer"]
        )
    else:
        app.dependency_overrides.pop(get_strict_current_user)
        config = SimpleNamespace(security=SimpleNamespace(require_auth=False))
        monkeypatch.setattr("stateset_agents.api.config.get_config", lambda: config)
        monkeypatch.setattr("stateset_agents.api.auth.get_config", lambda: config)
    with SyncASGIClient(app) as client:
        response = client.get("/finetune/capabilities")
        assert (
            response.status_code
            == {"anonymous": 401, "trainer": 403, "auth_disabled": 401}[mode]
        )
        assert client.get("/finetune/runs/example").status_code == response.status_code


def test_api_quality_upload_rechecks_artifact_and_never_echoes_case_ids(app, completed):
    run, config = completed
    with SyncASGIClient(app) as client:
        response = client.post(
            f"/finetune/runs/{run.name}/quality",
            json=evidence_for(run, config, regression=True),
        )
        assert response.status_code == 200
        assert (
            response.json()["result"]["regressions_total"] == 32
            and "PRIVATE-ID" not in response.text
        )
        if os.name != "nt":  # Windows st_mode carries no POSIX group/other bits.
            assert (run / "quality.json").stat().st_mode & 0o077 == 0
        assert client.get(f"/finetune/runs/{run.name}/quality").status_code == 200
        (run / "artifacts/adapter_model.bin").write_bytes(b"changed")
        assert client.get(f"/finetune/runs/{run.name}/quality").status_code == 400


def test_retained_workflow_and_listing_hide_private_source_contents(app, completed):
    run, _ = completed
    with SyncASGIClient(app) as client:
        response = client.get("/finetune/runs")
        assert (
            response.status_code == 200
            and response.json()["result"]["runs"][0]["run_id"] == run.name
        )
        assert (
            "training question" not in response.text and "dataset" not in response.text
        )
        response = client.get(f"/finetune/runs/{run.name}/workflow")
        assert (
            response.status_code == 200
            and not response.json()["result"]["training_started"]
        )
        assert "training question" not in response.text


def test_http_stop_requires_exact_review_and_is_idempotent(app, completed):
    run, _ = completed
    write_json(run / "state.json", {"status": "prepared"})
    path = f"/finetune/runs/{run.name}/stop"
    with SyncASGIClient(app) as client:
        assert (
            client.post(path, json={"expected_plan_sha256": "0" * 64}).status_code
            == 400
        )
        assert not (run / "stop.json").exists()
        review = client.get(f"/finetune/runs/{run.name}")
        assert review.status_code == 200
        metadata = review.json()["result"]
        assert metadata["run_id"] == run.name and metadata["status"] == "prepared"
        assert set(metadata["config"]) == {"model", "provider", "method"}
        assert "dataset" not in review.text and "training question" not in review.text
        request = {"expected_plan_sha256": metadata["plan_sha256"]}
        response = client.post(path, json=request)
        assert (
            response.status_code == 200
            and response.json()["result"]["status"] == "cancelled"
        )
        raw = (run / "stop.json").read_bytes()
        assert client.post(path, json=request).status_code == 200
        assert (run / "stop.json").read_bytes() == raw


@pytest.mark.parametrize("status", ["queued", "running"])
def test_active_http_run_review_requires_no_quality_or_weight_inspection(
    app, completed, monkeypatch, status
):
    run, _ = completed
    write_json(run / "state.json", {"status": status})
    if status == "running":
        owner = {"pid": 12345, "process_token": "synthetic-active-owner"}
        write_json(run / "execution.json", owner)

        def confirmed_owner(path):
            assert path == run / "execution.json"
            return workspace.read_json(path, max_bytes=4096) == owner

        monkeypatch.setattr(workspace, "_owner_confirmed", confirmed_owner)
    app.state.finetune_inspection_gate = asyncio.Semaphore(0)
    for name in ("inspect_quality", "inspect_checkpoints", "inspect_workflow"):
        monkeypatch.setattr(
            finetune,
            name,
            lambda *_: pytest.fail("Metadata must not inspect artifacts"),
        )
    with SyncASGIClient(app) as client:
        review = client.get(f"/finetune/runs/{run.name}")
        assert review.status_code == 200
        metadata = review.json()["result"]
        assert metadata["status"] == status and not metadata["stop_requested"]
        assert not (run / "stop.json").exists()
        response = client.post(
            f"/finetune/runs/{run.name}/stop",
            json={"expected_plan_sha256": metadata["plan_sha256"]},
        )
        assert (
            response.status_code == 200 and response.json()["result"]["stop_requested"]
        )
        assert response.json()["result"]["status"] == status
        assert client.get(f"/finetune/runs/{run.name}").json()["result"][
            "stop_requested"
        ]


def test_http_run_review_is_stale_after_plan_change(app, completed):
    run, _ = completed
    write_json(run / "state.json", {"status": "prepared"})
    with SyncASGIClient(app) as client:
        endpoint = f"/finetune/runs/{run.name}"
        reviewed = client.get(endpoint).json()["result"]["plan_sha256"]
        plan, _ = read_run_plan(run)
        plan["config"]["model"] = "changed-model"
        write_json(run / "plan.json", plan)
        response = client.post(
            endpoint + "/stop", json={"expected_plan_sha256": reviewed}
        )
        assert response.status_code == 400 and not (run / "stop.json").exists()
        fresh = client.get(endpoint).json()["result"]
        assert fresh["plan_sha256"] != reviewed
        assert fresh["config"]["model"] == "changed-model"


def test_remote_stop_does_not_claim_provider_cleanup(app, completed):
    run, _ = completed
    plan, _ = read_run_plan(run)
    plan["config"]["provider"] = "river"
    write_json(run / "plan.json", plan)
    with SyncASGIClient(app) as client:
        response = client.post(
            f"/finetune/runs/{run.name}/stop",
            json={"expected_plan_sha256": read_run_plan(run)[1]},
        )
        assert response.status_code == 400 and not (run / "stop.json").exists()


@pytest.mark.parametrize(
    "payload", [b"{", b"[]", b'{"secret":NaN}', b'{"secret":"\xff"}']
)
def test_malformed_http_json_uses_controlled_errors(app, payload):
    with SyncASGIClient(app) as client:
        response = client.post("/finetune/validate", content=payload)
        assert response.status_code == 400 and "secret" not in response.text


def test_upload_limits_apply_before_quality_validation(app, completed, monkeypatch):
    run, _ = completed
    monkeypatch.setattr(finetune, "MAX_EVIDENCE_BYTES", 16)
    monkeypatch.setattr(
        finetune,
        "record_quality_document",
        lambda *_: pytest.fail("Oversized data must not be parsed"),
    )
    with SyncASGIClient(app) as client:
        response = client.post(f"/finetune/runs/{run.name}/quality", content=b"x" * 17)
        assert response.status_code == 413 and not (run / "quality.json").exists()
        assert response.json()["error"]["details"][0]["code"] == "request_too_large"


def test_symlinked_runs_and_dataset_snapshots_are_rejected(app, completed):
    run, _ = completed
    alias = run.parent / "alias"
    alias.symlink_to(run, target_is_directory=True)
    with SyncASGIClient(app) as client:
        assert client.get("/finetune/runs/alias/quality").status_code == 404
        (run / "dataset").unlink()
        (run / "dataset").symlink_to(run / "plan.json")
        assert client.get(f"/finetune/runs/{run.name}/workflow").status_code == 400


@pytest.mark.asyncio
async def test_busy_inspection_does_not_block_event_loop_or_detach_on_cancellation(
    app, completed, monkeypatch
):
    run, _ = completed
    entered, release = threading.Event(), threading.Event()

    def inspect(_run):
        entered.set()
        assert release.wait(5)
        return {"done": True}

    monkeypatch.setattr(finetune, "inspect_quality", inspect)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = asyncio.create_task(client.get(f"/finetune/runs/{run.name}/quality"))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            response = await client.get("/finetune/capabilities")
            assert response.status_code == 200
            second = await client.get(f"/finetune/runs/{run.name}/quality")
            assert second.status_code == 429
            assert (await client.get("/finetune/runs")).status_code == 200
            stop = await client.post(
                f"/finetune/runs/{run.name}/stop",
                json={"expected_plan_sha256": read_run_plan(run)[1]},
            )
            assert (
                stop.status_code == 200
            )  # Lifecycle controls do not queue behind hashing.
            first.cancel()
            await asyncio.sleep(0.02)
            assert app.state.finetune_inspection_gate.locked() and not first.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert not app.state.finetune_inspection_gate.locked()


def test_known_credentials_are_redacted_from_operator_metadata(
    app, completed, monkeypatch
):
    run, _ = completed
    secret = "fixture-secret-value"
    monkeypatch.setenv("RIVER_API_KEY", secret)
    plan, _ = read_run_plan(run)
    plan["config"]["model"] = secret
    write_json(run / "plan.json", plan)
    with SyncASGIClient(app) as client:
        response = client.get("/finetune/runs")
        assert secret not in response.text and "[REDACTED]" in response.text
        response = client.get(f"/finetune/runs/{run.name}")
        assert secret not in response.text and "[REDACTED]" in response.text


def test_openapi_and_capabilities_expose_shared_config_and_payload_contracts(app):
    from stateset_agents.finetune_workspace import finetune_schema

    document = app.openapi()
    for path in [
        "/finetune/validate",
        "/finetune/runs/{run_id}/quality",
        "/finetune/runs/{run_id}/stop",
    ]:
        assert (
            "application/json"
            in document["paths"][path]["post"]["requestBody"]["content"]
        )
    with SyncASGIClient(app) as client:
        assert (
            client.get("/finetune/capabilities").json()["result"]["config_schema"]
            == finetune_schema()["config_schema"]
        )


@pytest.mark.parametrize("kind", ["fifo", "oversize", "symlink"])
def test_bad_workspace_metadata_cannot_hang_or_escape_listing(app, completed, kind):
    run, _ = completed
    state = run / "state.json"
    state.unlink()
    if kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO unavailable")
        os.mkfifo(state)
    elif kind == "oversize":
        state.write_bytes(b"x" * (128 * 1024 + 1))
    else:
        state.symlink_to(run / "dataset")
    with SyncASGIClient(app) as client:
        result = client.get("/finetune/runs")
        assert result.status_code == 200
        rows = result.json()["result"]["runs"]
        assert len(rows) == 1 and rows[0]["run_id"] == run.name
        assert rows[0]["status"] == "failed"
        assert "training question" not in result.text
        detail = client.get(f"/finetune/runs/{run.name}")
        assert detail.status_code == 400 and "training question" not in detail.text


def test_operator_recovery_inspection_is_bound_and_does_not_prepare_or_launch(
    app, tmp_path, monkeypatch
):
    from tests.unit.test_finetune_recovery import make_parent

    run, _ = make_parent(tmp_path / "recovery")
    monkeypatch.setenv("STATESET_FINETUNE_WORKSPACE", str(run.parent))
    with SyncASGIClient(app) as client:
        capabilities = client.get("/finetune/capabilities").json()["result"]
        assert capabilities["checkpoint_resume_supported"]
        assert not capabilities["http_recovery_prepare_supported"]
        response = client.get(f"/finetune/runs/{run.name}/checkpoints")
        assert response.status_code == 200
        result = response.json()["result"]
        assert result["checkpoints"][0]["available"]
        assert not result["prepared"] and not result["training_started"]
        assert "PRIVATE" not in response.text
        assert len(list(run.parent.glob("*/plan.json"))) == 1
        (run / "artifacts/checkpoint-1/optimizer.pt").write_bytes(b"tampered")
        response = client.get(f"/finetune/runs/{run.name}/checkpoints")
        assert response.status_code == 200
        assert not response.json()["result"]["checkpoints"][0]["available"]
        assert (
            client.post(f"/finetune/runs/{run.name}/checkpoints", json={}).status_code
            == 405
        )
