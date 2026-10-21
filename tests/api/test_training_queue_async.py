"""Durable HTTP operations stay responsive and own accepted submissions."""

import asyncio
import threading
from datetime import datetime

import httpx
import pytest
from fastapi import FastAPI

from stateset_agents.api.auth import AuthenticatedUser
from stateset_agents.api.dependencies import get_current_user, get_training_service
from stateset_agents.api.errors import ServiceUnavailableError, setup_exception_handlers
from stateset_agents.api.routers.training import router
from stateset_agents.api.schemas import TrainingRequest
from stateset_agents.api.services.training_service import TrainingService


@pytest.fixture
def service(tmp_path):
    return TrainingService(queue_path=tmp_path / "jobs.sqlite3")


@pytest.fixture
def request_data():
    return TrainingRequest(
        agent_config={"model_name": "test"},
        environment_scenarios=[{"id": "case", "context": "accepted"}],
        reward_config={},
        training_config_overrides={"learning_rate": 0.001},
    )


@pytest.fixture
async def client(service):
    app = FastAPI()
    app.include_router(router)
    setup_exception_handlers(app)
    app.dependency_overrides[get_training_service] = lambda: service
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        user_id="owner", roles=["trainer"]
    )

    @app.get("/ping")
    async def ping():
        return {"ready": True}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as value:
        yield value
    await service.aclose()


@pytest.mark.parametrize("operation", ["submit", "get", "page", "cancel"])
async def test_database_wait_does_not_block_unrelated_http_requests(
    service, client, request_data, monkeypatch, operation
):
    job_id = await service.start_training(request_data, "owner")
    entered, release = threading.Event(), threading.Event()

    def block(function):
        def call(*args, **kwargs):
            entered.set()
            assert release.wait(5), "Database operation blocked the request loop"
            return function(*args, **kwargs)

        return call

    monkeypatch.setattr(
        service.queue, operation, block(getattr(service.queue, operation))
    )
    if operation == "page":
        monkeypatch.setattr(service.queue, "jobs", block(service.queue.jobs))
    if operation == "submit":
        action = client.post("/training", json=request_data.model_dump())
    elif operation == "get":
        action = client.get(f"/training/{job_id}")
    elif operation == "page":
        action = client.get("/training")
    else:
        action = client.delete(f"/training/{job_id}")
    task = asyncio.create_task(action)
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        assert (await asyncio.wait_for(client.get("/ping"), 2)).status_code == 200
        assert not task.done(), "The blocked request ran on the event loop"
    finally:
        release.set()
        results = await asyncio.gather(task, return_exceptions=True)
    assert not isinstance(results[0], BaseException)
    assert results[0].status_code in (200, 202)


async def test_cancelled_submission_and_shutdown_drain_commit_and_freeze_inputs(
    service, request_data, monkeypatch
):
    entered, release = threading.Event(), threading.Event()
    original = service.queue.submit

    def delayed_submit(*args, **kwargs):
        entered.set()
        assert release.wait(15)
        return original(*args, **kwargs)

    monkeypatch.setattr(service.queue, "submit", delayed_submit)
    submission = asyncio.create_task(
        service.start_training(request_data, "owner", idempotency_key="accepted")
    )
    closing = None
    try:
        assert await asyncio.to_thread(entered.wait, 20)
        request_data.environment_scenarios[0]["context"] = "caller mutation"
        request_data.training_config_overrides["learning_rate"] = 0.25
        submission.cancel()
        await asyncio.sleep(0)
        closing = asyncio.create_task(service.aclose())
        await asyncio.sleep(0)
        assert service.is_closing
        with pytest.raises(ServiceUnavailableError):
            await service.start_training(request_data, "owner")
        closing.cancel()
        submission.cancel()
        for _ in range(3):
            await asyncio.sleep(0)
        assert not submission.done() and not closing.done()
    finally:
        release.set()
        results = await asyncio.gather(
            *([submission, closing] if closing is not None else [submission]),
            return_exceptions=True,
        )
    assert all(isinstance(result, asyncio.CancelledError) for result in results)
    job = next(iter(service.jobs.values()))
    assert job["status"] == "queued"
    assert job["config"]["environment_scenarios"][0]["context"] == "accepted"
    assert job["config"]["training_config_overrides"]["learning_rate"] == 0.001
    restarted = TrainingService(queue_path=service.queue.path)
    accepted = TrainingRequest.model_validate(job["config"])
    retried = await restarted.start_training(
        accepted, "owner", idempotency_key="accepted"
    )
    assert restarted.get_training_status(retried) == job
    assert len(restarted.jobs) == 1
    await restarted.aclose()


async def test_http_idempotency_conflict_returns_409_and_preserves_accepted_request(
    client, service, request_data
):
    headers = {"Idempotency-Key": "one"}
    accepted = await client.post(
        "/training", json=request_data.model_dump(), headers=headers
    )
    request_data.num_episodes += 1
    conflict = await client.post(
        "/training", json=request_data.model_dump(), headers=headers
    )
    assert accepted.status_code == 202
    assert conflict.status_code == 409, conflict.text
    assert conflict.json()["error"]["code"] == "CONFLICT"
    assert len(service.jobs) == 1


async def test_http_full_durable_queue_returns_429(
    client, service, request_data, monkeypatch
):
    from stateset_agents.api.training_queue import TrainingQueueFullError

    def full(*args, **kwargs):
        raise TrainingQueueFullError("Training queue is full")

    monkeypatch.setattr(service.queue, "submit", full)
    response = await client.post("/training", json=request_data.model_dump())
    assert response.status_code == 429, response.text
    assert response.json()["error"]["code"] == "RESOURCE_EXHAUSTED"
    assert "training_queue" in response.text and not service.jobs


async def test_http_listing_never_loads_all_jobs(
    service, client, request_data, monkeypatch
):
    owned = await service.start_training(request_data, "owner")
    await service.start_training(request_data, "other")
    public = await service.start_training(request_data)

    def forbidden():
        raise AssertionError("Paginated HTTP listing loaded every job")

    monkeypatch.setattr(service.queue, "jobs", forbidden)
    response = await client.get("/training", params={"page_size": 1})
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2 and len(data["items"]) == 1 and data["has_next"]
    assert data["items"][0]["training_id"] in (owned, public)


@pytest.mark.parametrize("durable", [False, True])
async def test_python_pagination_has_matching_owner_status_order_and_copy_semantics(
    tmp_path, durable
):
    service = TrainingService(queue_path=tmp_path / "jobs.sqlite3" if durable else None)
    records = {
        "b": ("owner", "queued"),
        "a": ("owner", "queued"),
        "c": (None, "failed"),
        "z": ("someone-else", "queued"),
    }
    for job_id, (owner, status) in records.items():
        snapshot = {
            "user_id": owner,
            "status": status,
            "created_at": datetime(2026, 1, 1),
            "metrics": {"nested": {"value": "original"}},
        }
        if durable:
            with service.queue._transaction() as connection:
                connection.execute(
                    "INSERT INTO stateset_training_jobs(id,owner,fingerprint,status,snapshot,created) VALUES(?,?,?,?,?,?)",
                    (
                        job_id,
                        owner,
                        job_id,
                        status,
                        service.queue._encode(snapshot),
                        100,
                    ),
                )
        else:
            service.training_jobs[job_id] = snapshot
    first, total = await service.list_training_jobs(
        page_size=1, user_id="owner", status="queued"
    )
    assert total == 2 and first[0][0] == "b"
    second, total = await service.list_training_jobs(
        page=2, page_size=1, user_id="owner", status="queued"
    )
    assert total == 2 and second[0][0] == "a"
    assert await service.list_training_jobs(
        page=10**100, user_id="owner", status="queued"
    ) == ([], 2)
    visible, total = await service.list_training_jobs(user_id="owner")
    assert total == 3 and [job_id for job_id, _ in visible] == ["c", "b", "a"]
    admin, total = await service.list_training_jobs()
    assert total == 4 and len(admin) == 4
    first[0][1]["metrics"]["nested"]["value"] = "caller mutation"
    assert (await service.aget_training_status("b", "owner"))["metrics"]["nested"][
        "value"
    ] == "original"
    assert await service.aget_training_status("z", "owner") is None
    assert not await service.acancel_training("z", "owner")
    await service.aclose()
