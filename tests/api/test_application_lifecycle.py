"""Application generations retain requests, resources, and failed cleanup."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from fastapi import WebSocket, WebSocketDisconnect
from starlette.responses import StreamingResponse

from stateset_agents.api import config as config_module
from stateset_agents.api import distributed_cache, main, persistence
from stateset_agents.api.application_lifecycle import ApplicationLifecycle
from stateset_agents.api.config import APIConfig
from stateset_agents.api.resource_lifecycle import ResourceOwnershipError
from stateset_agents.api.schemas import TrainingRequest
from stateset_agents.api.services import training_service


@pytest.fixture
def application(monkeypatch, tmp_path):
    config = APIConfig(v1_conversation_db_path=str(tmp_path / "conversations.db"))
    config.security.require_auth = False
    config.rate_limit.enabled = False
    monkeypatch.setattr(config_module, "_config", config)
    monkeypatch.setattr(main, "get_config", lambda: config)
    for name, value in {
        "CACHE_BACKEND": "memory",
        "DB_BACKEND": "memory",
        "INFERENCE_BACKEND": "stub",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("STATESET_DEFAULT_CHECKPOINT", raising=False)
    monkeypatch.delenv("STATESET_DEFAULT_BASE_MODEL", raising=False)
    monkeypatch.setattr(distributed_cache, "_cache_instance", None)
    monkeypatch.setattr(
        distributed_cache, "_cache_registry", distributed_cache._new_registry()
    )
    monkeypatch.setattr(persistence, "_database", None)
    monkeypatch.setattr(persistence, "_database_registry", persistence._new_registry())
    return main.create_app()


def client(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


async def wait(event):
    await asyncio.wait_for(event.wait(), 3)


@pytest.mark.asyncio
async def test_native_restart_preserves_conversations_agents_and_training_history(
    application,
):
    app = application
    app.state.health_checker.add_check("custom", lambda: True)
    async with client(app) as http:
        async with main.lifespan(app):
            assert (await http.get("/ready")).status_code == 200
            response = await http.post(
                "/api/v1/conversations",
                json={
                    "message": "remember me",
                    "conversation_id": "session",
                    "model": "stub://test",
                },
            )
            assert response.status_code == 200, response.text
            old = {
                name: getattr(app.state, name)
                for name in (
                    "health_checker",
                    "inference_service",
                    "training_service",
                    "v1_conversation_store",
                )
            }
            app.state.training_service.training_jobs["finished"] = {
                "status": "completed",
                "metadata": {"value": 1},
            }
            agents = app.state.agent_service
            agents.agents["preserved"] = object()
            model = agents.agents["preserved"]
        assert app.state.lifecycle.get_status()["cleanup_complete"]
        assert (await http.get("/ready")).status_code == 503
        assert (await http.get("/live")).status_code == 200
        assert (await http.get("/healthz")).status_code == 200
        async with main.lifespan(app):
            assert app.state.lifecycle.get_status()["generation"] == 2
            assert all(
                getattr(app.state, name) is not previous
                for name, previous in old.items()
            )
            assert (
                app.state.agent_service is agents
                and agents.agents["preserved"] is model
            )
            assert (
                "custom" in app.state.health_checker.get_status()["registered_checks"]
            )
            assert (await http.get("/ready")).status_code == 200
            history = await http.get("/api/v1/conversations/session")
            assert history.status_code == 200 and "remember me" in history.text
            assert app.state.training_service.jobs["finished"]["status"] == "completed"
            app.state.training_service.jobs["finished"]["metadata"]["value"] = 2
            assert old["training_service"].jobs["finished"]["metadata"]["value"] == 1
            assert not app.state.training_service.is_closing
            assert old["training_service"].is_closing


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["starting", "serving", "stopping"])
async def test_overlapping_lifespan_rejected_before_it_can_touch_active_resources(
    application, monkeypatch, stage
):
    app = application
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    acquire = distributed_cache.acquire_cache
    close_health = app.state.health_checker.aclose

    async def acquire_cache(config):
        calls.append("acquire")
        if stage == "starting":
            entered.set()
            await release.wait()
        return await acquire(config)

    async def close():
        calls.append("close")
        if stage == "stopping":
            entered.set()
            await release.wait()
        await close_health()

    monkeypatch.setattr(distributed_cache, "acquire_cache", acquire_cache)
    monkeypatch.setattr(app.state.health_checker, "aclose", close)
    scope = main.lifespan(app)
    opening = asyncio.create_task(scope.__aenter__())
    closing = None
    if stage == "starting":
        await wait(entered)
    else:
        await opening
        if stage == "stopping":
            closing = asyncio.create_task(scope.__aexit__(None, None, None))
            await wait(entered)
    before = list(calls)
    with pytest.raises(ResourceOwnershipError):
        async with main.lifespan(app):
            pytest.fail("A second lifespan entered")
    assert calls == before
    release.set()
    await opening
    if closing is not None:
        await closing
    else:
        await scope.__aexit__(None, None, None)
    assert calls == ["acquire", "close"]


@pytest.mark.asyncio
async def test_startup_and_shutdown_admission_keep_passive_status_available(
    application, monkeypatch
):
    app = application
    entered, release = asyncio.Event(), asyncio.Event()
    acquire = distributed_cache.acquire_cache

    async def blocked(config):
        entered.set()
        await release.wait()
        return await acquire(config)

    monkeypatch.setattr(distributed_cache, "acquire_cache", blocked)
    scope = main.lifespan(app)
    opening = asyncio.create_task(scope.__aenter__())
    await wait(entered)
    async with client(app) as http:
        assert (await http.get("/")).status_code == 503
        assert (await http.get("/ready")).json()["status"] == "not_ready"
        status = (await http.get("/lifecycle")).json()
        assert status["state"] == "starting" and status["phase"] == "cache"
        assert (await http.get("/live")).status_code == 200
        release.set()
        await opening
        assert (await http.get("/")).status_code == 200
        await scope.__aexit__(None, None, None)
        assert (await http.get("/")).status_code == 503
        assert (await http.get("/lifecycle")).json()["state"] == "closed"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_shutdown_drains_accepted_responses_before_dependencies(
    application, stream
):
    app = application
    entered, release = asyncio.Event(), asyncio.Event()

    @app.get("/held")
    async def held():
        if not stream:
            entered.set()
            await release.wait()
            assert not app.state.training_service.is_closing
            return {"answer": "complete"}

        async def body():
            yield b"first"
            entered.set()
            await release.wait()
            assert not app.state.training_service.is_closing
            yield b"last"

        return StreamingResponse(body())

    scope = main.lifespan(app)
    await scope.__aenter__()
    async with client(app) as http:
        request = asyncio.create_task(http.get("/held"))
        await wait(entered)
        closing = asyncio.create_task(scope.__aexit__(None, None, None))
        await asyncio.sleep(0)
        assert app.state.lifecycle.get_status()["phase"] == "requests"
        assert not closing.done()
        assert (await http.get("/held")).status_code == 503
        release.set()
        response = await request
        assert response.status_code == 200
        assert (
            "last" in response.text
            if stream
            else response.json() == {"answer": "complete"}
        )
        await closing
        assert app.state.lifecycle.get_status()["active_requests"] == 0


@pytest.mark.asyncio
async def test_request_cancellation_cleanup_survives_repeated_cancellation_and_shutdown(
    application,
):
    app = application
    entered, cleaning, release = (asyncio.Event() for _ in range(3))

    @app.get("/held")
    async def held():
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cleaning.set()
            await release.wait()
            assert not app.state.training_service.is_closing

    scope = main.lifespan(app)
    await scope.__aenter__()
    async with client(app) as http:
        request = asyncio.create_task(http.get("/held"))
        await wait(entered)
        request.cancel()
        await wait(cleaning)
        closing = asyncio.create_task(scope.__aexit__(None, None, None))
        await asyncio.sleep(0)
        for _ in range(3):
            request.cancel()
            closing.cancel()
            await asyncio.sleep(0)
            assert not request.done() and not closing.done()
        release.set()
        outcomes = await asyncio.gather(request, closing, return_exceptions=True)
        assert all(isinstance(outcome, asyncio.CancelledError) for outcome in outcomes)
    assert app.state.lifecycle.get_status()["state"] == "closed"


@pytest.mark.asyncio
async def test_request_callback_cannot_close_its_own_application(application):
    app = application

    @app.get("/self-close")
    async def callback():
        with pytest.raises(ResourceOwnershipError, match="own owner"):
            await asyncio.create_task(app.state.lifecycle.aclose())
        return app.state.lifecycle.get_status()["state"]

    async with main.lifespan(app), client(app) as http:
        assert (await http.get("/self-close")).json() == "serving"


@pytest.mark.asyncio
async def test_old_lifespan_exit_cannot_close_a_successor_generation(application):
    app = application
    first = main.lifespan(app)
    await first.__aenter__()
    await app.state.lifecycle.aclose()
    second = main.lifespan(app)
    await second.__aenter__()
    await first.__aexit__(None, None, None)
    assert app.state.lifecycle.get_status()["generation"] == 2
    assert app.state.lifecycle.get_status()["state"] == "serving"
    assert not app.state.training_service.is_closing
    await second.__aexit__(None, None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", list(ApplicationLifecycle._ORDER))
async def test_cleanup_failure_retains_outcome_and_dependency_order_without_replay(
    failed,
):
    owner = ApplicationLifecycle()
    generation = owner.begin()
    calls = []
    failure = ValueError("secret failure details")

    def closer(name):
        async def close():
            calls.append(name)
            if name == failed:
                raise failure

        return close

    critical = {"health_checker", "v1_conversation_store", "training_service"}
    for name in owner._ORDER:
        owner.bind(generation, name, object(), closer(name), critical=name in critical)
    owner.serving(generation)
    for _ in range(2):
        with pytest.raises(ValueError) as caught:
            await owner.aclose()
        assert caught.value is failure
    expected = list(owner._ORDER)
    if failed in critical:
        expected = expected[: expected.index(failed) + 1]
    assert calls == expected
    status = owner.get_status()
    assert status["state"] == "failed" and not status["cleanup_complete"]
    assert status["cleanup_errors"] == [
        {"component": failed, "error_type": "ValueError"}
    ]
    assert "secret" not in str(status)
    status["cleanup_errors"].clear()
    assert owner.get_status()["cleanup_errors"]
    with pytest.raises(ResourceOwnershipError):
        owner.begin()


@pytest.mark.asyncio
async def test_primary_failure_survives_critical_cleanup_failure(
    application, monkeypatch
):
    app = application
    failure = LookupError("primary")
    calls = []

    async def close():
        calls.append("health")
        raise ValueError("secondary")

    monkeypatch.setattr(app.state.health_checker, "aclose", close)
    with pytest.raises(LookupError) as caught:
        async with main.lifespan(app):
            raise failure
    assert caught.value is failure and calls == ["health"]
    status = app.state.lifecycle.get_status()
    assert status["serving_error_type"] == "LookupError"
    assert status["state"] == "failed"
    with pytest.raises(ResourceOwnershipError):
        async with main.lifespan(app):
            pytest.fail("Unconfirmed cleanup permitted restart")


@pytest.mark.asyncio
async def test_clean_startup_rollback_allows_a_new_native_generation(
    application, monkeypatch
):
    app = application
    previous = app.state.health_checker

    def fail(*args):
        raise ValueError("registration failed")

    monkeypatch.setattr(previous, "add_check", fail)
    with pytest.raises(ValueError):
        async with main.lifespan(app):
            pytest.fail("Failed startup yielded")
    status = app.state.lifecycle.get_status()
    assert status["state"] == "closed" and status["startup_error_type"] == "ValueError"
    async with main.lifespan(app):
        assert app.state.health_checker is not previous
        assert app.state.lifecycle.get_status()["generation"] == 2


@pytest.mark.asyncio
async def test_custom_closed_dependency_requires_fresh_instance_or_factory(application):
    app = application
    closed = []

    def create():
        resource = SimpleNamespace(check_health=lambda: True)

        async def close():
            closed.append(resource)

        resource.aclose = close
        return resource

    original = create()
    app.state.inference_service = original
    async with main.lifespan(app):
        pass
    with pytest.raises(ResourceOwnershipError, match="lifecycle_factories"):
        async with main.lifespan(app):
            pytest.fail("Closed custom resource reused")
    assert closed == [original]
    app.state.lifecycle_factories = {"inference_service": create}
    async with main.lifespan(app):
        fresh = app.state.inference_service
        assert fresh is not original
    assert closed == [original, fresh]


@pytest.mark.asyncio
async def test_restart_keeps_a_successfully_registered_default_model(
    application, monkeypatch, tmp_path
):
    app = application
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    monkeypatch.setenv("STATESET_DEFAULT_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("STATESET_DEFAULT_BASE_MODEL", "stub://restart-test")
    async with main.lifespan(app):
        model = app.state.agent_service.agents["default"]
        model.metadata["retained"] = True
    async with main.lifespan(app):
        assert app.state.agent_service.agents["default"] is model
        assert model.metadata["retained"]


def test_same_app_can_restart_on_a_new_event_loop_after_confirmed_close(application):
    app = application

    async def serve():
        async with main.lifespan(app), client(app) as http:
            assert (await http.get("/ready")).status_code == 200

    asyncio.run(serve())
    asyncio.run(serve())
    assert app.state.lifecycle.get_status()["generation"] == 2


@pytest.mark.asyncio
async def test_training_admission_reopens_after_restart_and_retains_settled_jobs(
    application, monkeypatch
):
    from tests.api.test_training_job_lifecycle import set_trainer

    app = application
    monkeypatch.setattr(training_service, "MultiTurnAgent", lambda config: object())

    async def train(**kwargs):
        return object()

    set_trainer(monkeypatch, train)
    request = TrainingRequest(
        agent_config={"model_name": "synthetic-test"},
        environment_scenarios=[],
        reward_config={},
        num_episodes=1,
    )
    async with main.lifespan(app):
        old = app.state.training_service
        first = await old.start_training(request, user_id="owner")
        await old._tasks[first]
    async with main.lifespan(app):
        current = app.state.training_service
        assert current.jobs[first]["status"] == "completed"
        second = await current.start_training(request, user_id="owner")
        await current._tasks[second]
        assert second != first and current.jobs[second]["status"] == "completed"
        assert second not in old.jobs


@pytest.mark.asyncio
async def test_websocket_drain_and_rejected_new_handshakes(application):
    app = application
    entered = asyncio.Event()

    @app.websocket("/socket")
    async def socket(ws: WebSocket):
        await ws.accept()
        entered.set()
        try:
            await ws.receive_text()
        except WebSocketDisconnect:
            pass

    incoming = asyncio.Queue()
    incoming.put_nowait({"type": "websocket.connect"})
    messages = []

    async def send(message):
        messages.append(message)

    scope = {
        "type": "websocket",
        "asgi": {"version": "3.0"},
        "scheme": "ws",
        "path": "/socket",
        "raw_path": b"/socket",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 123),
        "server": ("test", 80),
        "subprotocols": [],
    }
    lifespan = main.lifespan(app)
    await lifespan.__aenter__()
    request = asyncio.create_task(app(dict(scope), incoming.get, send))
    await wait(entered)
    closing = asyncio.create_task(lifespan.__aexit__(None, None, None))
    await asyncio.sleep(0)
    assert not closing.done()
    await app(dict(scope), incoming.get, send)
    assert messages[-1] == {"type": "websocket.close", "code": 1013}
    incoming.put_nowait({"type": "websocket.disconnect", "code": 1000})
    await request
    await closing
    assert app.state.lifecycle.get_status()["active_requests"] == 0


@pytest.mark.asyncio
async def test_cancelled_startup_drains_rollback_then_permits_a_fresh_generation(
    application, monkeypatch
):
    app = application
    entered, cleaning, release = (asyncio.Event() for _ in range(3))
    original = persistence.acquire_database

    async def acquire(config):
        entered.set()
        await asyncio.Future()

    async def close_cache():
        cleaning.set()
        await release.wait()

    async def cache(config):
        return SimpleNamespace(aclose=close_cache)

    monkeypatch.setattr(distributed_cache, "acquire_cache", cache)
    monkeypatch.setattr(persistence, "acquire_database", acquire)
    scope = main.lifespan(app)
    opening = asyncio.create_task(scope.__aenter__())
    await wait(entered)
    opening.cancel()
    await wait(cleaning)
    for _ in range(3):
        opening.cancel()
        await asyncio.sleep(0)
        assert not opening.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await opening
    status = app.state.lifecycle.get_status()
    assert (
        status["state"] == "closed" and status["startup_error_type"] == "CancelledError"
    )
    monkeypatch.setattr(persistence, "acquire_database", original)
    async with main.lifespan(app):
        assert app.state.lifecycle.get_status()["generation"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["startup", "cleanup"])
async def test_inherited_dependency_callback_cannot_close_or_restart_its_application(
    application, monkeypatch, stage
):
    app = application
    observed = []

    async def rejected():
        with pytest.raises(ResourceOwnershipError):
            await asyncio.create_task(app.state.lifecycle.aclose())
        with pytest.raises(ResourceOwnershipError):
            async with main.lifespan(app):
                pytest.fail("Reentrant lifespan entered")
        observed.append(stage)

    if stage == "startup":
        original = distributed_cache.acquire_cache

        async def acquire(config):
            await rejected()
            return await original(config)

        monkeypatch.setattr(distributed_cache, "acquire_cache", acquire)
    else:
        original_close = app.state.health_checker.aclose

        async def close():
            await rejected()
            await original_close()

        monkeypatch.setattr(app.state.health_checker, "aclose", close)
    async with main.lifespan(app):
        pass
    assert observed == [stage]


@pytest.mark.asyncio
async def test_concurrent_close_observers_share_failed_cleanup_after_cancellation():
    owner = ApplicationLifecycle()
    generation = owner.begin()
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    failure = ValueError("cleanup failed")

    async def close():
        calls.append("close")
        entered.set()
        await release.wait()
        raise failure

    owner.bind(generation, "health_checker", object(), close)
    owner.serving(generation)
    first = asyncio.create_task(owner.aclose())
    await wait(entered)
    second = asyncio.create_task(owner.aclose())
    for _ in range(3):
        first.cancel()
        await asyncio.sleep(0)
        assert not first.done() and not second.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    for observer in (second, owner.aclose()):
        with pytest.raises(ValueError) as caught:
            await observer
        assert caught.value is failure
    assert calls == ["close"]
    assert owner.get_status()["state"] == "failed"


@pytest.mark.asyncio
async def test_cleanup_task_construction_failure_is_retained(monkeypatch):
    owner = ApplicationLifecycle()
    generation = owner.begin()
    owner.serving(generation)
    failure = RuntimeError("task factory unavailable")
    calls = []

    def fail(operation):
        calls.append("create")
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(asyncio, "create_task", fail)
        with pytest.raises(RuntimeError) as caught:
            await owner.aclose()
        assert caught.value is failure
    with pytest.raises(RuntimeError) as caught:
        await owner.aclose()
    assert caught.value is failure and calls == ["create"]
    assert owner.get_status()["state"] == "failed"
    with pytest.raises(ResourceOwnershipError):
        owner.begin()


@pytest.mark.asyncio
async def test_generation_handle_cannot_close_another_application():
    first, second = ApplicationLifecycle(), ApplicationLifecycle()
    generation = first.begin()
    first.serving(generation)
    with pytest.raises(ResourceOwnershipError, match="another owner"):
        await second.aclose(generation)
    assert first.get_status()["state"] == "serving"
    await first.aclose()


@pytest.mark.asyncio
async def test_factory_constructs_absent_and_retired_dependencies(application):
    app = application
    del app.state.inference_service
    created, closed = [], []

    def create():
        resource = SimpleNamespace(check_health=lambda: True)

        async def close():
            closed.append(resource)

        resource.aclose = close
        created.append(resource)
        return resource

    app.state.lifecycle_factories = {"inference_service": create}
    for _ in range(2):
        async with main.lifespan(app):
            assert app.state.inference_service is created[-1]
    assert len(created) == 2 and created[0] is not created[1]
    assert closed == created


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["same", "none", "async"])
async def test_invalid_factory_cannot_publish_a_retired_or_missing_resource(
    application, invalid
):
    app = application
    async with main.lifespan(app):
        old = app.state.inference_service

    async def asynchronous_factory():
        pytest.fail("An async factory must not execute")

    app.state.lifecycle_factories = {
        "inference_service": (
            asynchronous_factory
            if invalid == "async"
            else lambda: old if invalid == "same" else None
        )
    }
    error_type = TypeError if invalid == "async" else ResourceOwnershipError
    with pytest.raises(error_type):
        async with main.lifespan(app):
            pytest.fail("An invalid factory reached serving")
    assert app.state.inference_service is old
    status = app.state.lifecycle.get_status()
    assert status["state"] == "closed"
    assert status["startup_error_type"] == error_type.__name__
    assert status["completed_cleanup"] == ["health_checker"]


@pytest.mark.asyncio
async def test_restart_reopens_durable_training_queue_without_resubmission(
    application, tmp_path
):
    app = application
    app.state.config.training_queue_path = str(tmp_path / "training.db")
    request = TrainingRequest(
        agent_config={"model_name": "stub://queued"},
        environment_scenarios=[],
        reward_config={},
        num_episodes=1,
    )
    async with main.lifespan(app):
        old = app.state.training_service
        first = await old.start_training(
            request, user_id="owner", idempotency_key="first"
        )
        assert old.jobs[first]["status"] == "queued"
    async with main.lifespan(app):
        current = app.state.training_service
        assert current is not old and current.queue is not old.queue
        assert current.queue.path == old.queue.path
        assert list(current.jobs) == [first]
        assert current.jobs[first]["status"] == "queued"
        assert (
            await current.start_training(
                request, user_id="owner", idempotency_key="first"
            )
            == first
        )
        second = await current.start_training(
            request, user_id="owner", idempotency_key="second"
        )
        assert second != first and len(current.jobs) == 2
        assert not current._tasks
