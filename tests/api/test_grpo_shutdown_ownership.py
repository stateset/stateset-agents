"""Shutdown retains GRPO engine ownership until workers and cleanup return."""

import asyncio
import gc
import threading
import weakref
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from stateset_agents.api.grpo import handlers, service
from stateset_agents.api.grpo.metrics import GRPOMetrics
from stateset_agents.api.grpo.models import GRPOTrainingRequest
from stateset_agents.api.grpo.state import StateManager
from stateset_agents.api.shutdown import GracefulShutdownManager, ShutdownPhase
from tests.api.test_grpo_handlers import _grpo_service_test_app


class Engine:
    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.in_use = False
        self.closed = False
        self.calls = 0
        self.cleanups = 0

    async def train_iteration(self, prompts):
        assert not self.closed
        self.calls += 1
        self.in_use = True
        self.entered.set()
        try:
            await self.release.wait()
            assert not self.closed
            return {"trajectories_generated": 1}
        finally:
            self.in_use = False

    def cleanup(self):
        self.cleanups += 1
        assert not self.in_use, "Engine closed during an owned iteration"
        self.closed = True


@pytest.fixture
def setup_service(monkeypatch):
    from stateset_agents.core import multiturn_agent
    from stateset_agents.utils import cache, monitoring

    engine = Engine()
    state = StateManager()
    metrics = GRPOMetrics()
    manager = GracefulShutdownManager()
    monkeypatch.setattr(handlers, "get_state_manager", lambda: state)
    monkeypatch.setattr(handlers, "get_grpo_metrics", lambda: metrics)
    handler = handlers.TrainingHandler({"demo_engine": engine})
    monkeypatch.setattr(service, "get_shutdown_manager", lambda: manager)
    monkeypatch.setattr(service, "get_state_manager", lambda: state)
    monkeypatch.setattr(
        service,
        "get_grpo_config",
        lambda: SimpleNamespace(
            validate=lambda: [], enable_prometheus=False, use_lightweight_engine=True
        ),
    )
    monkeypatch.setattr(service, "_services", {})
    monkeypatch.setattr(service, "LightweightDemoEngine", lambda: engine)
    monkeypatch.setattr(service, "_get_training_handler", lambda: handler)
    monkeypatch.setattr(service, "_get_conversation_handler", lambda: None)
    monkeypatch.setattr(service, "_get_websocket_handler", lambda: None)
    monkeypatch.setattr(cache, "CacheService", lambda: object())
    monkeypatch.setattr(monitoring, "MonitoringService", lambda **kwargs: object())
    monkeypatch.setattr(multiturn_agent, "MultiTurnAgent", lambda *a, **kw: object())
    return SimpleNamespace(engine=engine, state=state, handler=handler, manager=manager)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["normal", "serving_error", "cancel_shutdown"])
async def test_lifespan_keeps_engine_until_iteration_returns(setup_service, mode):
    env = setup_service
    shutting_down = asyncio.Event()
    worker = None
    job = env.state.create_job("job", "computational", "owner", "request")

    async def observe_shutdown():
        shutting_down.set()

    env.manager.register_task("observe", ShutdownPhase.STOP_ACCEPTING, observe_shutdown)

    async def serve():
        nonlocal worker
        async with service.lifespan(SimpleNamespace()):
            worker = asyncio.create_task(
                env.handler.run_computational_training(job.job_id, ["prompt"], 2)
            )
            await env.engine.entered.wait()
            if mode == "serving_error":
                raise ValueError("serving failed")

    server = asyncio.create_task(serve())
    try:
        await asyncio.wait_for(shutting_down.wait(), timeout=0.3)
        await asyncio.sleep(0)
        assert not server.done()
        assert not env.engine.closed
        if mode == "cancel_shutdown":
            server.cancel()
            await asyncio.sleep(0)
            server.cancel()
            await asyncio.sleep(0)
            assert not server.done()
            assert not env.engine.closed
        env.engine.release.set()
        outcome = (await asyncio.gather(server, return_exceptions=True))[0]
        if mode == "serving_error":
            assert isinstance(outcome, ValueError) and str(outcome) == "serving failed"
        elif mode == "cancel_shutdown":
            assert isinstance(outcome, asyncio.CancelledError)
        else:
            assert outcome is None
        assert env.engine.closed
        assert env.engine.cleanups == 1
        assert env.engine.calls == 1
        assert job.status == "cancelled"
    finally:
        env.engine.release.set()
        await asyncio.gather(server, return_exceptions=True)
        if worker is not None:
            await asyncio.gather(worker, return_exceptions=True)


@pytest.mark.asyncio
async def test_lifespan_closes_monitoring_after_engine_cleanup(
    setup_service, monkeypatch
):
    from stateset_agents.utils import monitoring

    env = setup_service
    calls = []

    async def close():
        assert env.engine.closed
        calls.append("closed")

    monitor = SimpleNamespace(aclose=close)
    monkeypatch.setattr(monitoring, "MonitoringService", lambda **kwargs: monitor)
    async with service.lifespan(SimpleNamespace()):
        assert service._services["monitoring"] is monitor
    assert calls == ["closed"]
    assert "monitoring" not in service._services


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_startup_failure_drains_or_retains_monitoring_owner(
    setup_service, monkeypatch, cleanup_fails
):
    from stateset_agents.utils import monitoring

    entered, release = asyncio.Event(), asyncio.Event()
    failure = ValueError("startup failed")
    calls = []

    def fail():
        raise failure

    async def close():
        calls.append("close")
        entered.set()
        await release.wait()
        if cleanup_fails:
            raise RuntimeError("cleanup unconfirmed")

    monitor = SimpleNamespace(aclose=close)
    monkeypatch.setattr(monitoring, "MonitoringService", lambda **kwargs: monitor)
    monkeypatch.setattr(service, "_get_conversation_handler", fail)

    async def serve():
        async with service.lifespan(SimpleNamespace()):
            pytest.fail("Startup should fail")

    task = asyncio.create_task(serve())
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(ValueError) as caught:
            await task
        assert caught.value is failure
        assert calls == ["close"]
        if cleanup_fails:
            assert service._services["monitoring"] is monitor
        else:
            assert "monitoring" not in service._services
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_concurrent_shutdown_callers_join_one_owned_sequence():
    manager = GracefulShutdownManager()
    assert manager.get_status()["outcome"] == "not_started"
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def cleanup():
        calls.append("cleanup")
        entered.set()
        await release.wait()
        calls.append("finished")

    manager.register_task("cleanup", ShutdownPhase.CLEANUP_JOBS, cleanup)
    first = asyncio.create_task(manager.execute_shutdown())
    second = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        assert manager.get_status()["outcome"] == "running"
        with pytest.raises(RuntimeError, match="after shutdown starts"):
            manager.register_task("late", ShutdownPhase.FINAL, cleanup)
        second = asyncio.create_task(manager.execute_shutdown())
        await asyncio.sleep(0)
        assert not second.done()
        first.cancel()
        await asyncio.sleep(0)
        first.cancel()
        await asyncio.sleep(0)
        assert not first.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        await second
        await manager.execute_shutdown()
        assert calls == ["cleanup", "finished"]
        assert manager.get_status()["completed_tasks"] == ["cleanup"]
        assert manager.get_status()["outcome"] == "completed"
    finally:
        release.set()
        await asyncio.gather(
            first, *([second] if second is not None else []), return_exceptions=True
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_waiter", [False, True])
async def test_async_engine_cleanup_is_awaited_once_for_all_aliases(
    setup_service, monkeypatch, cancel_waiter
):
    env = setup_service
    entered, release = asyncio.Event(), asyncio.Event()

    async def cleanup():
        env.engine.cleanups += 1
        entered.set()
        await release.wait()
        env.engine.closed = True

    monkeypatch.setattr(env.engine, "cleanup", cleanup)
    env.state.register_engine("one", env.engine)
    env.state.register_engine("two", env.engine)

    async def serve():
        async with service.lifespan(SimpleNamespace()):
            pass

    server = asyncio.create_task(serve())
    try:
        await asyncio.wait_for(entered.wait(), timeout=0.3)
        assert not server.done()
        assert env.state.get_engine("one") is env.engine
        if cancel_waiter:
            server.cancel()
            await asyncio.sleep(0)
            server.cancel()
            await asyncio.sleep(0)
            assert not server.done()
            assert not env.engine.closed
        release.set()
        if cancel_waiter:
            with pytest.raises(asyncio.CancelledError):
                await server
        else:
            await server
        assert env.engine.cleanups == 1
        assert env.engine.closed
        assert not env.state.engines
    finally:
        release.set()
        await asyncio.gather(server, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("remove_record", [False, True])
async def test_handler_close_tracks_workers_independently_of_job_retention(
    setup_service, remove_record
):
    env = setup_service
    response = await env.handler.start_training(
        GRPOTrainingRequest(prompts=["prompt"]), "owner", "request"
    )
    worker = asyncio.create_task(
        env.handler.run_computational_training(response.job_id, ["prompt"], 2)
    )
    close = None
    try:
        await asyncio.wait_for(env.engine.entered.wait(), timeout=1)
        if remove_record:
            env.state.jobs.pop(response.job_id)
        env.handler.close_admission()
        close = asyncio.create_task(env.handler.aclose())
        await asyncio.sleep(0)
        assert env.handler.is_closing
        assert not env.handler.is_closed
        assert env.handler.active_workers == 1
        assert not close.done()
        with pytest.raises(RuntimeError, match="shutting down"):
            await env.handler.start_training(
                GRPOTrainingRequest(prompts=["late"]), "owner", "late"
            )
        env.engine.release.set()
        await asyncio.wait_for(asyncio.gather(worker, close), timeout=1)
        await env.handler.aclose()
        assert env.handler.is_closed
        assert env.handler.active_workers == 0
        assert env.engine.calls == 1
    finally:
        env.engine.release.set()
        await worker
        if close is not None:
            await close


@pytest.mark.asyncio
async def test_shutdown_cancels_callbacks_that_never_entered_worker(setup_service):
    env = setup_service
    response = await env.handler.start_training(
        GRPOTrainingRequest(prompts=["prompt"]), "owner", "request"
    )
    never_started = asyncio.create_task(
        env.handler.run_computational_training(response.job_id, ["prompt"], 1)
    )
    never_started.cancel()
    with pytest.raises(asyncio.CancelledError):
        await never_started
    async with service.lifespan(SimpleNamespace()):
        pass
    assert env.state.get_job(response.job_id).status == "cancelled"
    await env.handler.run_distributed_training(response.job_id, ["prompt"], 1, {})
    assert env.handler.is_closed
    assert env.engine.calls == 0
    assert env.engine.cleanups == 1


@pytest.mark.asyncio
async def test_cancellation_before_queued_iteration_starts_skips_engine(
    setup_service, monkeypatch
):
    env = setup_service
    scheduled = asyncio.Event()
    original = env.handler._run_iteration

    async def observed(*args):
        scheduled.set()
        return await original(*args)

    monkeypatch.setattr(env.handler, "_run_iteration", observed)
    job = env.state.create_job("job", "computational", "owner", "request")
    worker = asyncio.create_task(
        env.handler.run_computational_training(job.job_id, ["prompt"], 1)
    )
    try:
        await asyncio.wait_for(scheduled.wait(), timeout=1)
        assert env.handler.cancel_job(job.job_id, "owner")
        env.engine.release.set()
        await worker
        assert env.engine.calls == 0
        assert job.status == "cancelled"
    finally:
        env.engine.release.set()
        await asyncio.gather(worker, return_exceptions=True)


@pytest.mark.asyncio
async def test_repeated_worker_cancellation_drains_engine_unwind(
    setup_service, monkeypatch
):
    env = setup_service
    unwinding, finish_unwind = asyncio.Event(), asyncio.Event()

    async def train_iteration(prompts):
        env.engine.in_use = True
        env.engine.entered.set()
        try:
            await asyncio.Future()
        finally:
            unwinding.set()
            await finish_unwind.wait()
            env.engine.in_use = False

    monkeypatch.setattr(env.engine, "train_iteration", train_iteration)
    job = env.state.create_job("job", "computational", "owner", "request")
    worker = asyncio.create_task(
        env.handler.run_computational_training(job.job_id, ["prompt"], 1)
    )
    close = None
    try:
        await asyncio.wait_for(env.engine.entered.wait(), timeout=1)
        worker.cancel()
        await asyncio.wait_for(unwinding.wait(), timeout=1)
        worker.cancel()
        await asyncio.sleep(0)
        worker.cancel()
        close = asyncio.create_task(env.handler.aclose())
        await asyncio.sleep(0)
        assert not worker.done() and not close.done()
        assert env.handler.active_workers == 1
        finish_unwind.set()
        with pytest.raises(asyncio.CancelledError):
            await worker
        await close
        assert not env.engine.in_use
        assert env.handler.is_closed
        assert env.handler.active_workers == 0
    finally:
        finish_unwind.set()
        await asyncio.gather(worker, return_exceptions=True)
        if close is not None:
            await close


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["/api", "/v1"])
@pytest.mark.parametrize("batch", [False, True])
async def test_training_admission_returns_503_during_shutdown(prefix, batch):
    with _grpo_service_test_app() as fixture:
        handler = fixture["training_handler"]
        handler.close_admission()
        before = handler.metrics.training_jobs_started
        payload = {"prompts": ["prompt"], "num_iterations": 1}
        if batch:
            payload = {"items": [payload]}
        path = f"{prefix}/batch/train" if batch else f"{prefix}/train"
        async with AsyncClient(
            transport=ASGITransport(app=fixture["app"]), base_url="http://testserver"
        ) as client:
            response = await client.post(
                path, json=payload, headers={"x-user-id": "owner"}
            )
        assert response.status_code == 503
        assert handler.metrics.training_jobs_started == before
        assert not handler.state.list_jobs()


@pytest.mark.asyncio
async def test_engine_cleanup_failure_retains_handle_and_failure_on_repeat(
    setup_service, monkeypatch
):
    env = setup_service
    error = OSError("cleanup failed")

    async def cleanup():
        env.engine.cleanups += 1
        raise error

    monkeypatch.setattr(env.engine, "cleanup", cleanup)
    env.state.register_engine("owned", env.engine)
    with pytest.raises(OSError) as caught:
        async with service.lifespan(SimpleNamespace()):
            pass
    assert caught.value is error
    assert env.state.get_engine("owned") is env.engine
    assert env.manager.get_status()["outcome"] == "failed"
    assert env.manager.get_status()["failed_tasks"] == ["cleanup_engines"]
    with pytest.raises(OSError) as repeated:
        await env.manager.execute_shutdown()
    assert repeated.value is error
    assert env.engine.cleanups == 1


@pytest.mark.asyncio
async def test_critical_timeout_waits_for_unwind_and_never_starts_engine_cleanup():
    manager = GracefulShutdownManager()
    timed_out, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def drain_jobs():
        try:
            await asyncio.Future()
        finally:
            timed_out.set()
            await release.wait()
            calls.append("drained")

    async def cleanup_engines():
        calls.append("engines")

    manager.register_task(
        "jobs", ShutdownPhase.CLEANUP_JOBS, drain_jobs, 0.01, critical=True
    )
    manager.register_task("engines", ShutdownPhase.CLEANUP_ENGINES, cleanup_engines)
    shutdown = asyncio.create_task(manager.execute_shutdown())
    try:
        await asyncio.wait_for(timed_out.wait(), timeout=1)
        assert not shutdown.done()
        assert not calls
        release.set()
        with pytest.raises(RuntimeError, match="Critical shutdown task jobs timed out"):
            await shutdown
        assert calls == ["drained"]
        assert manager.get_status()["outcome"] == "failed"
        assert manager.get_status()["failed_tasks"] == ["jobs"]
    finally:
        release.set()
        await asyncio.gather(shutdown, return_exceptions=True)


@pytest.mark.asyncio
async def test_worker_drain_failure_blocks_all_engine_cleanup(
    setup_service, monkeypatch
):
    env = setup_service
    error = OSError("worker drain failed")

    async def fail_close():
        raise error

    monkeypatch.setattr(env.handler, "aclose", fail_close)
    env.state.register_engine("owned", env.engine)
    with pytest.raises(OSError) as caught:
        async with service.lifespan(SimpleNamespace()):
            pass
    assert caught.value is error
    assert env.engine.cleanups == 0
    assert env.state.get_engine("owned") is env.engine
    assert env.manager.get_status()["failed_tasks"] == ["cleanup_jobs"]
    assert env.manager.get_status()["outcome"] == "failed"


@pytest.mark.asyncio
async def test_sync_cleanup_keeps_loop_responsive_and_drains_before_cancellation(
    setup_service, monkeypatch
):
    env = setup_service
    loop = asyncio.get_running_loop()
    cleaning = asyncio.Event()
    release = threading.Event()

    def cleanup():
        env.engine.cleanups += 1
        loop.call_soon_threadsafe(cleaning.set)
        assert release.wait(timeout=2), "Cleanup blocked the event loop"
        env.engine.closed = True

    monkeypatch.setattr(env.engine, "cleanup", cleanup)

    async def serve():
        async with service.lifespan(SimpleNamespace()):
            pass

    server = asyncio.create_task(serve())
    try:
        await asyncio.wait_for(cleaning.wait(), timeout=1)
        server.cancel()
        await asyncio.sleep(0)
        server.cancel()
        await asyncio.sleep(0)
        assert not server.done()
        assert not env.engine.closed
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(server, timeout=1)
        assert env.engine.closed
        assert env.engine.cleanups == 1
    finally:
        release.set()
        await asyncio.gather(server, return_exceptions=True)


@pytest.mark.asyncio
async def test_request_drain_timeout_is_visible_in_shutdown_outcome():
    manager = GracefulShutdownManager(drain_timeout=0)
    manager.increment_in_flight()
    await manager.execute_shutdown()
    status = manager.get_status()
    assert status["outcome"] == "completed_with_errors"
    assert status["failed_tasks"] == ["drain_requests"]
    assert status["in_flight_requests"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [ValueError("expected"), KeyError("unexpected")])
async def test_callback_failures_preserve_classification_and_are_not_replayed(failure):
    manager = GracefulShutdownManager()
    calls = []

    async def failing():
        calls.append("failure")
        raise failure

    async def final():
        calls.append("final")

    manager.register_task("failure", ShutdownPhase.CLEANUP_STATE, failing)
    manager.register_task("final", ShutdownPhase.FINAL, final)
    if isinstance(failure, KeyError):
        for _ in range(2):
            with pytest.raises(KeyError) as caught:
                await manager.execute_shutdown()
            assert caught.value is failure
        assert calls == ["failure"]
        assert manager.get_status()["outcome"] == "failed"
    else:
        await manager.execute_shutdown()
        await manager.execute_shutdown()
        assert calls == ["failure", "final"]
        assert manager.get_status()["outcome"] == "completed_with_errors"
    assert manager.get_status()["failed_tasks"] == ["failure"]


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["handler", "manager"])
async def test_closed_lifecycle_is_rejected_before_allocating_engines(
    setup_service, monkeypatch, owner
):
    env = setup_service
    if owner == "handler":
        await env.handler.aclose()
    else:
        await env.manager.execute_shutdown()

    def forbidden_engine():
        pytest.fail("A terminal lifecycle must not allocate another engine")

    monkeypatch.setattr(service, "LightweightDemoEngine", forbidden_engine)
    with pytest.raises(RuntimeError, match="GRPO startup requires"):
        async with service.lifespan(SimpleNamespace()):
            pytest.fail("A terminal lifecycle entered serving")


@pytest.mark.asyncio
async def test_handler_retains_worker_even_when_caller_drops_task(
    setup_service, monkeypatch
):
    env = setup_service
    entered = asyncio.Event()
    references = {}

    async def train_iteration(prompts):
        release = asyncio.get_running_loop().create_future()
        references["release"] = weakref.ref(release)
        entered.set()
        await release
        return {"trajectories_generated": 1}

    monkeypatch.setattr(env.engine, "train_iteration", train_iteration)
    job = env.state.create_job("job", "computational", "owner", "request")
    worker = asyncio.create_task(
        env.handler.run_computational_training(job.job_id, ["prompt"], 1)
    )
    reference = weakref.ref(worker)
    del worker
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        gc.collect()
        assert reference() is not None, "The handler lost its running task"
        release = references["release"]()
        assert release is not None
        release.set_result(None)
        await asyncio.wait_for(reference(), timeout=1)
        assert env.handler.active_workers == 0
        assert job.status == "completed"
    finally:
        release = references.get("release", lambda: None)()
        if release is not None and not release.done():
            release.set_result(None)
        if reference() is not None:
            await asyncio.gather(reference(), return_exceptions=True)
        await env.handler.aclose()
