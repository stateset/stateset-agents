"""Admin-only local provider receipt inspection without provider authentication."""

import asyncio
import json
import os
import threading
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException

from stateset_agents import finetune_provider_submission as submission
from stateset_agents.api.dependencies import get_strict_current_user
from stateset_agents.api.errors import setup_exception_handlers
from stateset_agents.api.routers import finetune
from stateset_agents.remote import registry
from tests.api.asgi_client import SyncASGIClient
from tests.unit.test_finetune_provider_details import (
    JOB_ID,
    POD_ID,
    damage_receipt,
    event,
    make_provider_run,
)
from tests.unit.test_finetune_provider_submission import guarded_record_open


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.delenv("STATESET_AGENTS_WHEEL", raising=False)
    run = make_provider_run(tmp_path)
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
    monkeypatch.setattr(
        registry,
        "get_executor",
        lambda *_: pytest.fail("Inspection must not contact providers"),
    )
    return application, run


def test_provider_route_reports_retained_identity_and_uncertainty_without_artifact_work(
    setup, monkeypatch
):
    app, run = setup
    journal = submission.SubmissionJournal(run, "runpod", durable_supported=True)
    journal.publish(event("allocation_intent"))
    journal.publish(event("allocated", POD_ID))
    journal.publish(event("cleanup_acknowledged", POD_ID))
    app.state.finetune_inspection_gate = asyncio.Semaphore(0)
    for name in ("inspect_quality", "inspect_checkpoints", "inspect_workflow"):
        monkeypatch.setattr(
            finetune,
            name,
            lambda *_: pytest.fail("Receipt view must not inspect artifacts"),
        )
    main_thread = threading.get_ident()
    inspect = finetune.inspect_submission

    def in_worker(path):
        assert threading.get_ident() != main_thread
        return inspect(path)

    monkeypatch.setattr(finetune, "inspect_submission", in_worker)
    before = (run / "provider-submission.json").read_bytes()
    with SyncASGIClient(app) as client:
        response = client.get(f"/finetune/runs/{run.name}/provider")
    assert response.status_code == 200
    result = response.json()["result"]
    assert response.json()["command"] == "provider"
    assert result["available"] and result["job_id"] == JOB_ID
    assert result["resources"][0]["id"] == POD_ID
    assert result["resources"][0]["state"] == "cleanup_acknowledged"
    assert result["cleanup_required_or_unconfirmed"]
    assert result["live_status_checked"] is result["provider_cleanup_verified"] is False
    assert result["automatic_restart_supported"] is False
    assert "PRIVATE" not in response.text and "dataset" not in response.text
    assert (run / "provider-submission.json").read_bytes() == before


def test_missing_provider_receipt_is_available_false_without_writes(setup):
    app, run = setup
    with SyncASGIClient(app) as client:
        response = client.get(f"/finetune/runs/{run.name}/provider")
    assert response.status_code == 200
    assert response.json()["result"] == {
        "available": False,
        "live_status_checked": False,
    }
    assert not (run / "provider-submission.json").exists()


@pytest.mark.parametrize(
    "kind", ["malformed", "plan", "symlink", "fifo", "oversize", "public"]
)
def test_invalid_receipt_returns_safe_error_without_reading_special_files(
    setup, tmp_path, monkeypatch, kind
):
    app, run = setup
    submission.SubmissionJournal(run, "runpod", durable_supported=True)
    damage_receipt(run, kind, tmp_path)
    guarded_record_open(monkeypatch, run / "provider-submission.json")
    with SyncASGIClient(app) as client:
        response = client.get(f"/finetune/runs/{run.name}/provider")
    assert response.status_code == 400
    assert response.json()["error"]["details"][0]["code"] == "invalid_provider_receipt"
    assert "PRIVATE" not in response.text and str(tmp_path) not in response.text
    assert not (run / "admission.json").exists()


@pytest.mark.parametrize("mode", ["anonymous", "trainer", "auth_disabled"])
def test_provider_receipts_require_authenticated_admin_before_inspection(
    setup, monkeypatch, mode
):
    app, run = setup

    def forbidden(*_args):
        pytest.fail("Unauthorized request reached a receipt")

    monkeypatch.setattr(finetune, "inspect_submission", forbidden)
    if mode == "anonymous":

        def denied():
            raise HTTPException(401, "Authentication required")

        app.dependency_overrides[get_strict_current_user] = denied
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
        response = client.get(f"/finetune/runs/{run.name}/provider")
    assert (
        response.status_code
        == {"anonymous": 401, "trainer": 403, "auth_disabled": 401}[mode]
    )


def test_provider_capability_documents_readonly_effect_and_preserves_no_remote_stop(
    setup,
):
    app, _run = setup
    with SyncASGIClient(app) as client:
        result = client.get("/finetune/capabilities").json()["result"]
    assert result["effects"]["provider_get"] == "reads_retained_provider_submission"
    assert result["launch_supported"] is False
    assert result["remote_stop_supported"] is (os.name != "nt")
    assert result["remote_stop_providers"] == (["runpod"] if os.name != "nt" else [])


def test_provider_response_redacts_known_credentials_and_omits_unrelated_record_fields(
    setup, monkeypatch
):
    app, run = setup
    secret = "synthetic-secret-credential"
    journal = submission.SubmissionJournal(run, "runpod", durable_supported=True)
    journal.publish(event("allocated", secret))
    path = run / "provider-submission.json"
    record = json.loads(path.read_text())
    record["unrelated_private_field"] = "PRIVATE prompt or SDK exception"
    path.write_text(json.dumps(record))
    monkeypatch.setenv("RUNPOD_API_KEY", secret)
    with SyncASGIClient(app) as client:
        response = client.get(f"/finetune/runs/{run.name}/provider")
    assert response.status_code == 200
    assert secret not in response.text and "[REDACTED]" in response.text
    assert (
        "PRIVATE" not in response.text
        and "unrelated_private_field" not in response.text
    )


def test_provider_route_rejects_symlinked_run(setup):
    app, run = setup
    alias = run.parent / "alias"
    try:
        alias.symlink_to(run, target_is_directory=True)
    except OSError:
        pytest.skip("Symlinks unavailable")
    with SyncASGIClient(app) as client:
        response = client.get("/finetune/runs/alias/provider")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_cancelled_provider_inspection_drains_its_owned_reader(
    setup, monkeypatch
):
    _app, run = setup
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def blocked(_run):
        started.set()
        assert release.wait(3), "Receipt reader was not released"
        finished.set()
        return {"available": False, "live_status_checked": False}

    monkeypatch.setattr(finetune, "inspect_submission", blocked)
    task = asyncio.create_task(finetune.provider_submission(run.name))
    try:
        assert await asyncio.to_thread(started.wait, 3)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and not finished.is_set()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 3)
    assert finished.is_set()
