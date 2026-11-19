"""Health observations belong to their registration, probe, and detached view."""

import asyncio
import inspect
import threading
from types import SimpleNamespace

import httpx
import pytest

from stateset_agents.api.resilience import (
    HealthChecker,
    HealthCheckResult,
    HealthStatus,
)


async def wait_for_probes(checker, count):
    async def wait():
        while checker.get_status()["active_probes"] != count:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=2)


@pytest.mark.asyncio
async def test_removed_probe_cannot_republish_its_result():
    checker = HealthChecker()
    entered, release = asyncio.Event(), asyncio.Event()

    async def operation():
        entered.set()
        await release.wait()
        return True

    checker.add_check("service", operation)
    task = asyncio.create_task(checker.check("service"))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        checker.remove_check("service")
        release.set()
        assert (await task).status is HealthStatus.HEALTHY
        assert await checker.check_all() == {}
        assert checker.overall_status is HealthStatus.UNHEALTHY
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("same_callable", [False, True])
async def test_replacement_fences_old_probe_even_with_the_same_callable(same_callable):
    checker = HealthChecker()
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def operation():
        calls.append("probe")
        if len(calls) == 1:
            entered.set()
            await release.wait()
            return False
        return True

    async def replacement():
        return True

    checker.add_check("service", operation)
    old = asyncio.create_task(checker.check("service"))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        checker.add_check("service", operation if same_callable else replacement)
        assert (await checker.check("service")).status is HealthStatus.HEALTHY
        release.set()
        assert (await old).status is HealthStatus.UNHEALTHY
        assert checker.overall_status is HealthStatus.HEALTHY
    finally:
        release.set()
        await asyncio.gather(old, return_exceptions=True)


@pytest.mark.asyncio
async def test_probe_results_do_not_mutate_provider_or_internal_state():
    checker = HealthChecker()
    supplied = HealthCheckResult(
        name="service",
        status=HealthStatus.HEALTHY,
        details={"workers": [{"ready": True}]},
    )
    checker.add_check("service", lambda: supplied)
    first = await checker.check("service")
    assert supplied.latency_ms == 0
    first.status = HealthStatus.UNHEALTHY
    first.details["workers"][0]["ready"] = False
    assert supplied.details["workers"][0]["ready"] is True
    assert checker.overall_status is HealthStatus.HEALTHY


@pytest.mark.asyncio
async def test_batch_results_are_detached_from_membership_and_nested_values():
    checker = HealthChecker()
    supplied = HealthCheckResult(
        name="service",
        status=HealthStatus.HEALTHY,
        details={"items": [1]},
    )
    checker.add_check("service", lambda: supplied)
    result = await checker.check_all()
    result["service"].status = HealthStatus.UNHEALTHY
    result["service"].details["items"].append(2)
    result.clear()
    assert checker.overall_status is HealthStatus.HEALTHY
    assert supplied.details == {"items": [1]}


@pytest.mark.asyncio
async def test_new_or_replaced_registration_is_unknown_until_observed():
    checker = HealthChecker()
    checker.add_check("known", lambda: True)
    await checker.check("known")
    assert checker.overall_status is HealthStatus.HEALTHY
    checker.add_check("new", lambda: True)
    assert checker.overall_status is HealthStatus.UNHEALTHY
    await checker.check("new")
    assert checker.overall_status is HealthStatus.HEALTHY
    checker.add_check("known", lambda: False)
    assert checker.overall_status is HealthStatus.UNHEALTHY


@pytest.mark.asyncio
async def test_concurrent_callers_share_one_probe_and_receive_separate_results():
    checker = HealthChecker()
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def operation():
        calls.append("probe")
        entered.set()
        await release.wait()
        return HealthCheckResult(
            "service", HealthStatus.HEALTHY, details={"items": [1]}
        )

    checker.add_check("service", operation)
    first = asyncio.create_task(checker.check("service"))
    second = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        second = asyncio.create_task(checker.check("service"))
        await asyncio.sleep(0)
        assert calls == ["probe"]
        release.set()
        left, right = await asyncio.gather(first, second)
        left.details["items"].append(2)
        assert right.details == {"items": [1]}
    finally:
        release.set()
        await asyncio.gather(
            first, *([second] if second else []), return_exceptions=True
        )


@pytest.mark.asyncio
async def test_cancelling_one_observer_does_not_cancel_a_shared_probe():
    checker = HealthChecker()
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def operation():
        calls.append("probe")
        entered.set()
        await release.wait()
        return True

    checker.add_check("service", operation)
    first = asyncio.create_task(checker.check("service"))
    second = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        second = asyncio.create_task(checker.check("service"))
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert checker.get_status()["cancelling_probes"] == 0
        assert not second.done()
        release.set()
        assert (await second).status is HealthStatus.HEALTHY
        assert calls == ["probe"]
    finally:
        release.set()
        await asyncio.gather(
            first, *([second] if second else []), return_exceptions=True
        )
        await checker.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("suppress_cancellation", [False, True])
async def test_last_observer_cancellation_retains_cleanup_without_reprobe(
    suppress_cancellation,
):
    checker = HealthChecker()
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = []

    async def operation():
        calls.append("probe")
        if len(calls) > 1:
            return True
        entered.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cleaning.set()
            await release.wait()
            calls.append("cleaned")
            if suppress_cancellation:
                return True
            raise

    checker.add_check("service", operation)
    task = asyncio.create_task(checker.check("service"))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        await asyncio.wait_for(cleaning.wait(), timeout=2)
        for _ in range(3):
            assert (await checker.check("service")).status is HealthStatus.UNHEALTHY
        assert checker.get_status()["active_probes"] == 1
        assert checker.get_status()["cancelling_probes"] == 1
        assert calls == ["probe"]
        release.set()
        await wait_for_probes(checker, 0)
        assert checker.overall_status is HealthStatus.UNHEALTHY
        assert (await checker.check("service")).status is HealthStatus.HEALTHY
        assert calls == ["probe", "cleaned", "probe"]
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await checker.aclose()


@pytest.mark.asyncio
async def test_cancelled_sync_probe_is_owned_until_its_thread_finishes():
    checker = HealthChecker()
    entered, release = threading.Event(), threading.Event()
    calls = []

    def operation():
        calls.append("probe")
        entered.set()
        assert release.wait(timeout=3)
        calls.append("finished")
        return True

    checker.add_check("service", operation)
    task = asyncio.create_task(checker.check("service"))
    closing = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        assert (await checker.check("service")).status is HealthStatus.UNHEALTHY
        closing = asyncio.create_task(checker.aclose())
        await asyncio.sleep(0)
        assert not closing.done()
        assert checker.get_status()["active_probes"] == 1
        assert calls == ["probe"]
        release.set()
        await asyncio.wait_for(closing, timeout=2)
        assert calls == ["probe", "finished"]
        assert checker.get_status()["state"] == "closed"
        assert checker.get_status()["active_probes"] == 0
    finally:
        release.set()
        await asyncio.gather(
            task, *([closing] if closing else []), return_exceptions=True
        )
        await checker.aclose()


@pytest.mark.asyncio
async def test_cancelled_sync_factory_closes_its_unstarted_coroutine():
    checker = HealthChecker()
    entered, release = threading.Event(), threading.Event()
    returned, calls = [], []

    async def result():
        calls.append("unexpected effect")
        return True

    def factory():
        entered.set()
        assert release.wait(timeout=3)
        pending = result()
        returned.append(pending)
        return pending

    checker.add_check("factory", factory)
    task = asyncio.create_task(checker.check("factory"))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        await checker.aclose()
        assert not calls
        assert inspect.getcoroutinestate(returned[0]) == inspect.CORO_CLOSED
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await checker.aclose()


@pytest.mark.asyncio
async def test_cancelled_sync_factory_drains_the_running_task_it_returns():
    checker = HealthChecker()
    entered, return_task = threading.Event(), threading.Event()
    running, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    effects = []

    async def operation():
        running.set()
        try:
            await asyncio.Future()
        finally:
            cleaning.set()
            await release.wait()
            effects.append("cleaned")

    upstream = asyncio.create_task(operation())

    def factory():
        entered.set()
        assert return_task.wait(timeout=3)
        return upstream

    checker.add_check("factory", factory)
    observer = asyncio.create_task(checker.check("factory"))
    closing = None
    try:
        await asyncio.wait_for(running.wait(), timeout=2)
        assert await asyncio.to_thread(entered.wait, 2)
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        return_task.set()
        await asyncio.wait_for(cleaning.wait(), timeout=2)
        closing = asyncio.create_task(checker.aclose())
        await asyncio.sleep(0)
        assert not closing.done()
        assert checker.get_status()["active_probes"] == 1
        assert not effects
        release.set()
        await asyncio.wait_for(closing, timeout=2)
        assert effects == ["cleaned"]
        assert upstream.cancelled()
    finally:
        return_task.set()
        release.set()
        await checker.aclose()
        if not upstream.done():
            upstream.cancel()
        await asyncio.gather(
            observer, upstream, *([closing] if closing else []), return_exceptions=True
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("replace", [False, True])
async def test_close_owns_old_probes_after_registration_changes(replace):
    checker = HealthChecker()
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    effects = []

    async def operation():
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cleaning.set()
            await release.wait()
            effects.append("cleaned")

    checker.add_check("service", operation)
    observer = asyncio.create_task(checker.check("service"))
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        if replace:
            checker.add_check("service", lambda: True)
            assert (await checker.check("service")).status is HealthStatus.HEALTHY
        else:
            checker.remove_check("service")
        closing = asyncio.create_task(checker.aclose())
        await asyncio.wait_for(cleaning.wait(), timeout=2)
        assert not closing.done()
        assert not effects
        assert checker.get_status()["active_probes"] == 1
        release.set()
        await asyncio.wait_for(closing, timeout=2)
        assert effects == ["cleaned"]
        assert (await observer).status is HealthStatus.UNHEALTHY
        assert checker.state.value == "closed"
    finally:
        release.set()
        await checker.aclose()
        await asyncio.gather(
            observer, *([closing] if closing else []), return_exceptions=True
        )


@pytest.mark.asyncio
async def test_batch_includes_new_unobserved_checks_and_excludes_removed_checks():
    checker = HealthChecker()
    entered, release = asyncio.Event(), asyncio.Event()

    async def operation():
        entered.set()
        await release.wait()
        return True

    checker.add_check("removed", operation)
    task = asyncio.create_task(checker.check_all())
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        checker.remove_check("removed")
        checker.add_check("new", lambda: True)
        release.set()
        snapshot = await task
        assert set(snapshot) == {"new"}
        assert snapshot["new"].status is HealthStatus.UNHEALTHY
        assert (await checker.check("new")).status is HealthStatus.HEALTHY
        assert snapshot["new"].status is HealthStatus.UNHEALTHY
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await checker.aclose()


@pytest.mark.asyncio
async def test_failed_snapshot_cannot_preserve_a_prior_healthy_observation():
    checker = HealthChecker()
    failure = KeyError("metadata cannot be copied")
    supplied = HealthCheckResult("service", HealthStatus.HEALTHY)

    class Uncopyable:
        def __deepcopy__(self, memo):
            raise failure

    checker.add_check("service", lambda: supplied)
    await checker.check("service")
    supplied.details["resource"] = Uncopyable()
    with pytest.raises(KeyError) as caught:
        await checker.check("service")
    assert caught.value is failure
    assert checker.overall_status is HealthStatus.UNHEALTHY
    assert checker.get_results()["service"].status is HealthStatus.UNHEALTHY
    await checker.aclose()


@pytest.mark.asyncio
async def test_batch_failure_retires_sibling_observers_but_owns_probe_cleanup():
    checker = HealthChecker()
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def slow():
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cleaning.set()
            await release.wait()

    async def broken():
        await entered.wait()
        raise KeyError("probe bug")

    checker.add_check("slow", slow)
    checker.add_check("broken", broken)
    task = asyncio.create_task(checker.check_all())
    try:
        with pytest.raises(KeyError, match="probe bug"):
            await asyncio.wait_for(task, timeout=2)
        await asyncio.wait_for(cleaning.wait(), timeout=2)
        assert checker.get_status()["active_probes"] == 1
        assert checker.get_status()["cancelling_probes"] == 1
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await checker.aclose()


@pytest.mark.asyncio
async def test_concurrent_close_survives_cancelled_waiters_and_rejects_new_work():
    checker = HealthChecker()
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = []

    async def operation():
        calls.append("probe")
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cleaning.set()
            await release.wait()
            calls.append("cleaned")

    checker.add_check("service", operation)
    task = asyncio.create_task(checker.check("service"))
    first = second = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        first = asyncio.create_task(checker.aclose())
        await asyncio.wait_for(cleaning.wait(), timeout=2)
        second = asyncio.create_task(checker.aclose())
        first.cancel()
        await asyncio.sleep(0)
        first.cancel()
        await asyncio.sleep(0)
        assert not first.done() and not second.done()
        assert (await checker.check("service")).status is HealthStatus.UNHEALTHY
        with pytest.raises(RuntimeError, match="closing"):
            checker.add_check("new", lambda: True)
        assert calls == ["probe"]
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        await second
        await checker.aclose()
        assert checker.get_status()["state"] == "closed"
        assert calls == ["probe", "cleaned"]
    finally:
        release.set()
        await asyncio.gather(
            task, *[value for value in (first, second) if value], return_exceptions=True
        )
        await checker.aclose()


@pytest.mark.asyncio
async def test_probe_cannot_wait_for_its_own_shutdown():
    checker = HealthChecker()
    checker.add_check("self", checker.aclose)
    result = await checker.check("self")
    assert result.status is HealthStatus.UNHEALTHY
    assert "own checker" in result.message
    assert checker.state.value == "ready"
    await checker.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("healthy", [False, True])
async def test_health_endpoint_summarizes_the_returned_snapshot(healthy):
    from fastapi import FastAPI

    from stateset_agents.api.routers import metrics

    async def check_all():
        return {
            "service": HealthCheckResult(
                "service", HealthStatus.HEALTHY if healthy else HealthStatus.UNHEALTHY
            )
        }

    app = FastAPI()
    app.state.health_checker = SimpleNamespace(
        check_all=check_all,
        overall_status=HealthStatus.UNHEALTHY if healthy else HealthStatus.HEALTHY,
    )
    app.include_router(metrics.router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")
    assert response.status_code == (200 if healthy else 503)
    body = response.json()
    assert body["status"] == body["components"]["service"]


@pytest.mark.asyncio
async def test_readiness_uses_current_application_checker_and_bounds_wait(monkeypatch):
    from stateset_agents.api import main
    from stateset_agents.api.routers import metrics

    app = main.create_app()
    checker = HealthChecker()
    checker.add_check("service", lambda: True)
    app.state.health_checker = checker
    entered, release = threading.Event(), threading.Event()

    def blocked():
        entered.set()
        assert release.wait(timeout=3)
        return True

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.get("/ready")).status_code == 200
            checker.add_check("service", blocked)
            monkeypatch.setattr(metrics, "HEALTH_TIMEOUT_SECONDS", 0.01)
            response = await asyncio.wait_for(client.get("/ready"), timeout=2)
            assert response.status_code == 503
            assert checker.get_status()["active_probes"] == 1
    finally:
        release.set()
        await checker.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("recursive", ["check", "check_all", "aclose"])
async def test_self_wait_is_rejected_even_with_an_eager_task_factory(eager, recursive):
    if eager and not hasattr(asyncio, "eager_task_factory"):
        pytest.skip("Eager task factory requires Python 3.12")
    loop = asyncio.get_running_loop()
    old_factory = loop.get_task_factory()
    checker = HealthChecker()

    async def operation():
        if recursive == "check":
            return await checker.check("self")
        return await getattr(checker, recursive)()

    checker.add_check("self", operation)
    try:
        if eager:
            loop.set_task_factory(asyncio.eager_task_factory)
        result = await asyncio.wait_for(checker.check("self"), timeout=2)
        assert result.status is HealthStatus.UNHEALTHY
        assert checker.state.value == "ready"
    finally:
        loop.set_task_factory(old_factory)
        await checker.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_shutdown", [False, True])
async def test_lifespan_drains_probes_before_closing_their_dependencies(
    monkeypatch, cancel_shutdown
):
    from fastapi import FastAPI

    from stateset_agents.api import distributed_cache, main, persistence

    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    events = []
    checker = HealthChecker()
    observers = []

    async def probe():
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cleaning.set()
            await release.wait()
            events.append("probe")

    async def acquire_cache(config):
        return SimpleNamespace(aclose=lambda: close("cache"))

    async def acquire_database(config):
        return SimpleNamespace(aclose=lambda: close("database"))

    async def close(name):
        events.append(name)

    monkeypatch.setattr(distributed_cache, "acquire_cache", acquire_cache)
    monkeypatch.setattr(persistence, "acquire_database", acquire_database)
    monkeypatch.delenv("STATESET_DEFAULT_CHECKPOINT", raising=False)
    app = FastAPI()
    app.state.config = SimpleNamespace(api_version="test", is_production=lambda: False)
    app.state.health_checker = checker
    app.state.agent_service = object()
    app.state.training_service = SimpleNamespace(aclose=lambda: close("training"))
    app.state.inference_service = SimpleNamespace(
        aclose=lambda: close("inference"), check_health=lambda: True
    )
    checker.add_check("resource", probe)

    async def serve():
        async with main.lifespan(app):
            observer = asyncio.create_task(checker.check("resource"))
            observers.append(observer)
            await entered.wait()

    task = asyncio.create_task(serve())
    try:
        await asyncio.wait_for(cleaning.wait(), timeout=2)
        assert not events
        assert checker.state.value == "closing"
        if cancel_shutdown:
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
        assert not task.done()
        release.set()
        if cancel_shutdown:
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await task
        assert events == ["probe", "training", "cache", "database", "inference"]
        assert checker.state.value == "closed"
    finally:
        release.set()
        await asyncio.gather(task, *observers, return_exceptions=True)
        await checker.aclose()


@pytest.mark.asyncio
async def test_probe_self_cancellation_is_unhealthy_without_cancelling_its_observer():
    checker = HealthChecker()

    async def operation():
        raise asyncio.CancelledError

    checker.add_check("cancelled", operation)
    result = await checker.check("cancelled")
    assert result.status is HealthStatus.UNHEALTHY
    assert "cancelled" in result.message
    assert checker.overall_status is HealthStatus.UNHEALTHY
    assert (await checker.check_all())["cancelled"].status is HealthStatus.UNHEALTHY
    await checker.aclose()


@pytest.mark.asyncio
async def test_batch_keeps_its_completed_measurement_when_a_new_probe_starts():
    checker = HealthChecker()
    next_entered, release = asyncio.Event(), asyncio.Event()
    calls, successors = [], []

    def start_next():
        successors.append(asyncio.create_task(checker.check("service")))

    async def operation():
        calls.append("probe")
        if len(calls) == 1:
            asyncio.get_running_loop().call_soon(start_next)
            return True
        next_entered.set()
        await release.wait()
        return True

    checker.add_check("service", operation)
    batch = asyncio.create_task(checker.check_all())
    try:
        await asyncio.wait_for(next_entered.wait(), timeout=2)
        snapshot = await batch
        assert snapshot["service"].status is HealthStatus.HEALTHY
        assert checker.overall_status is HealthStatus.UNHEALTHY
        release.set()
        await asyncio.gather(*successors)
    finally:
        release.set()
        await asyncio.gather(batch, *successors, return_exceptions=True)
        await checker.aclose()
