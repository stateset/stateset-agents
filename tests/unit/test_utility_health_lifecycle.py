"""Utility health observations retain probes through deadlines and shutdown."""

import asyncio
import inspect
import threading
from dataclasses import asdict

import pytest

from stateset_agents.utils import monitoring


async def wait_until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=2)


@pytest.mark.asyncio
async def test_simultaneous_observers_share_one_probe():
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def check():
        calls.append("check")
        entered.set()
        await release.wait()
        return True

    owner = monitoring.HealthCheck("service", check)
    first = asyncio.create_task(owner.run())
    second = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        assert "_probe" not in asdict(owner)
        second = asyncio.create_task(owner.run())
        await asyncio.sleep(0)
        release.set()
        results = await asyncio.gather(first, second)
        assert calls == ["check"]
        assert all(result["status"] == "healthy" for result in results)
        assert results[0] is not results[1]
    finally:
        release.set()
        await asyncio.gather(
            first, *([second] if second else []), return_exceptions=True
        )
        if hasattr(owner, "aclose"):
            await owner.aclose()


@pytest.mark.asyncio
async def test_timed_out_sync_probe_cannot_overlap_its_replacement():
    entered, release = threading.Event(), threading.Event()
    calls = []

    def check():
        calls.append("check")
        entered.set()
        assert release.wait(timeout=3)
        return True

    owner = monitoring.HealthCheck("service", check, timeout=0.01)
    try:
        result = await owner.run()
        assert entered.is_set()
        assert result["status"] == "timeout"
        retry = await owner.run()
        assert retry["status"] == "unavailable"
        assert calls == ["check"]
    finally:
        release.set()
        if hasattr(owner, "aclose"):
            await owner.aclose()


@pytest.mark.asyncio
async def test_upstream_timeout_is_a_probe_error():
    async def check():
        raise TimeoutError("dependency rejected request")

    owner = monitoring.HealthCheck("service", check)
    try:
        result = await owner.run()
        assert result["status"] == "error"
        assert result["error"] == "dependency rejected request"
    finally:
        if hasattr(owner, "aclose"):
            await owner.aclose()


@pytest.mark.asyncio
async def test_unconfigured_dependencies_do_not_claim_health(monkeypatch):
    monkeypatch.setattr(monitoring, "HAS_PSUTIL", False)
    owner = monitoring.MonitoringService(enable_prometheus=False)
    try:
        result = await owner.get_health_status()
        checks = {check["name"]: check for check in result["checks"]}
        assert checks["database"]["status"] == "unavailable"
        assert checks["cache"]["status"] == "unavailable"
        assert checks["system_resources"]["status"] == "unavailable"
        assert result["overall_status"] != "healthy"
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_cancelled_observer_does_not_cancel_another_observers_probe():
    entered, release = asyncio.Event(), asyncio.Event()

    async def check():
        entered.set()
        await release.wait()
        return True

    owner = monitoring.HealthCheck("service", check)
    first = asyncio.create_task(owner.run())
    second = asyncio.create_task(owner.run())
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        await wait_until(lambda: owner.get_status()["observers"] == 2)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert owner.get_status()["cancelling_probes"] == 0
        release.set()
        assert (await second)["status"] == "healthy"
    finally:
        release.set()
        await asyncio.gather(first, second, return_exceptions=True)
        await owner.aclose()


@pytest.mark.asyncio
async def test_timeout_returns_before_async_cleanup_and_close_retains_it():
    cleaning, release = asyncio.Event(), asyncio.Event()

    async def check():
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()

    owner = monitoring.HealthCheck("service", check, timeout=0.01)
    closing = other = None
    try:
        assert (await asyncio.wait_for(owner.run(), timeout=1))["status"] == "timeout"
        await asyncio.wait_for(cleaning.wait(), timeout=2)
        assert (await owner.run())["status"] == "unavailable"
        closing = asyncio.create_task(owner.aclose())
        other = asyncio.create_task(owner.aclose())
        await wait_until(lambda: owner.state.value == "closing")
        for _ in range(2):
            closing.cancel()
            await asyncio.sleep(0)
        assert not closing.done() and not other.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await closing
        await other
        assert owner.get_status()["state"] == "closed"
        assert owner.get_status()["active_probes"] == 0
        assert (await owner.run())["status"] == "unavailable"
    finally:
        release.set()
        await asyncio.gather(
            *[task for task in (closing, other) if task], return_exceptions=True
        )
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_kind", ["check", "checker", "service"])
async def test_close_drains_sync_probe_before_releasing_dependencies(
    monkeypatch, owner_kind
):
    monkeypatch.setattr(monitoring, "HAS_PSUTIL", False)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def operation():
        entered.set()
        assert release.wait(timeout=3)
        calls.append("probe finished")
        return True

    check = monitoring.HealthCheck("service", operation, timeout=0.01)
    if owner_kind == "check":
        owner = check
    elif owner_kind == "checker":
        owner = monitoring.HealthChecker()
        owner.add_health_check(check)
    else:
        owner = monitoring.MonitoringService(enable_prometheus=False)
        owner.health_checker.add_health_check(check)
        original_close = owner.metrics_collector.close

        def close_collector():
            assert calls == ["probe finished"]
            calls.append("collector closed")
            original_close()

        monkeypatch.setattr(owner.metrics_collector, "close", close_collector)
    closing = None
    try:
        assert (await check.run())["status"] == "timeout"
        assert entered.is_set()
        with pytest.raises(RuntimeError, match="aclose"):
            owner.close()
        assert owner.state.value in ("ready", "running")
        closing = asyncio.create_task(owner.aclose())
        await wait_until(lambda: owner.state.value == "closing")
        closing.cancel()
        closing.cancel()
        await asyncio.sleep(0)
        assert not closing.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert owner.state.value == "closed"
        assert calls[0] == "probe finished"
    finally:
        release.set()
        if closing:
            await asyncio.gather(closing, return_exceptions=True)
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["async_object", "sync_awaitable"])
async def test_awaitable_results_are_awaited_instead_of_treated_as_true(kind):
    async def operation():
        await asyncio.sleep(0)
        return False

    class AsyncCheck:
        async def __call__(self):
            return await operation()

    owner = monitoring.HealthCheck(
        "service", AsyncCheck() if kind == "async_object" else lambda: operation()
    )
    try:
        assert (await owner.run())["status"] == "unhealthy"
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_late_awaitable_is_closed_after_sync_probe_timeout():
    release = threading.Event()
    created = []

    async def late():
        pytest.fail("Retired probe must not start its returned coroutine")

    def operation():
        assert release.wait(timeout=3)
        result = late()
        created.append(result)
        return result

    owner = monitoring.HealthCheck("service", operation, timeout=0.01)
    try:
        assert (await owner.run())["status"] == "timeout"
    finally:
        release.set()
        await owner.aclose()
    assert len(created) == 1
    assert inspect.getcoroutinestate(created[0]) == inspect.CORO_CLOSED


@pytest.mark.asyncio
async def test_registration_list_changes_do_not_abandon_started_probes():
    entered, release = asyncio.Event(), asyncio.Event()

    async def operation():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            await release.wait()

    check = monitoring.HealthCheck("retired", operation)
    owner = monitoring.HealthChecker()
    owner.add_health_check(check)
    observing = asyncio.create_task(owner.run_all_checks())
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        owner.health_checks.clear()
        closing = asyncio.create_task(owner.aclose())
        await wait_until(lambda: check.state.value == "closing")
        assert not closing.done()
        release.set()
        await closing
        result = await observing
        assert result["checks"][0]["status"] == "cancelled"
        assert result["overall_status"] != "healthy"
        with pytest.raises(RuntimeError, match="closing|closed"):
            owner.add_health_check(monitoring.HealthCheck("late", lambda: True))
    finally:
        release.set()
        await asyncio.gather(
            observing, *([closing] if closing else []), return_exceptions=True
        )
        await owner.aclose()


@pytest.mark.asyncio
async def test_metadata_is_captured_for_each_probe():
    entered, release = asyncio.Event(), asyncio.Event()

    async def operation():
        entered.set()
        await release.wait()
        return True

    owner = monitoring.HealthCheck("original", operation, critical=True)
    observing = asyncio.create_task(owner.run())
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        owner.name = "replacement"
        owner.critical = False
        owner.check_func = lambda: False
        release.set()
        result = await observing
        assert result["name"] == "original" and result["critical"]
        assert result["status"] == "healthy"
        next_result = await owner.run()
        assert next_result["name"] == "replacement" and not next_result["critical"]
        assert next_result["status"] == "unhealthy"
    finally:
        release.set()
        await asyncio.gather(observing, return_exceptions=True)
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["sync", "async"])
async def test_probe_cannot_close_its_parent_service(monkeypatch, kind):
    monkeypatch.setattr(monitoring, "HAS_PSUTIL", False)
    owner = monitoring.MonitoringService(enable_prometheus=False)

    async def async_check():
        await owner.aclose()
        return True

    def sync_check():
        owner.close()
        return True

    check = monitoring.HealthCheck(
        "self", sync_check if kind == "sync" else async_check
    )
    owner.health_checker.add_health_check(check)
    try:
        result = await check.run()
        assert result["status"] == "error"
        assert "own owner" in result["error"]
        assert owner.state.value == "running"
    finally:
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["sync", "async"])
async def test_probe_child_task_cannot_create_a_parent_cleanup_cycle(monkeypatch, kind):
    monkeypatch.setattr(monitoring, "HAS_PSUTIL", False)
    owner = monitoring.MonitoringService(enable_prometheus=False)
    guards = []

    async def child():
        with pytest.raises(RuntimeError, match="own owner"):
            await owner.aclose()
        guards.append("rejected")

    async def asynchronous():
        await asyncio.create_task(child())
        return True

    def synchronous():
        asyncio.run(child())
        return True

    check = monitoring.HealthCheck(
        "nested", synchronous if kind == "sync" else asynchronous
    )
    owner.health_checker.add_health_check(check)
    try:
        assert (await check.run())["status"] == "healthy"
        assert guards == ["rejected"]
        assert owner.state.value == "running"
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_empty_checker_and_closed_checks_cannot_establish_health():
    owner = monitoring.HealthChecker()
    assert (await owner.run_all_checks())["overall_status"] == "unavailable"
    check = monitoring.HealthCheck("configured", lambda: True)
    owner.add_health_check(check)
    assert (await owner.run_all_checks())["overall_status"] == "healthy"
    await owner.aclose()
    assert (await owner.run_all_checks())["overall_status"] != "healthy"


@pytest.mark.asyncio
async def test_cancellation_during_executor_shutdown_retains_returned_awaitable(
    monkeypatch,
):
    entered, release = threading.Event(), threading.Event()
    created = []
    original_executor = monitoring.ThreadPoolExecutor

    class SlowShutdown(original_executor):
        def shutdown(self, *args, **kwargs):
            entered.set()
            assert release.wait(timeout=3)
            super().shutdown(*args, **kwargs)

    async def late():
        pytest.fail("Cancelled probe must not start a returned coroutine")

    def check():
        result = late()
        created.append(result)
        return result

    monkeypatch.setattr(monitoring, "ThreadPoolExecutor", SlowShutdown)
    owner = monitoring.HealthCheck("service", check)
    observing = asyncio.create_task(owner.run())
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        for _ in range(2):
            owner._probe.task.cancel()
            await asyncio.sleep(0)
        assert not observing.done()
        assert owner.get_status()["executor_owned"]
        release.set()
        assert (await observing)["status"] == "cancelled"
        await owner.aclose()
        assert owner.get_status()["state"] == "closed"
        assert not owner.get_status()["executor_owned"]
        assert inspect.getcoroutinestate(created[0]) == inspect.CORO_CLOSED
    finally:
        release.set()
        await asyncio.gather(observing, return_exceptions=True)
        await owner.aclose()


@pytest.mark.asyncio
async def test_executor_cleanup_failure_retains_owner_and_dependencies(monkeypatch):
    monkeypatch.setattr(monitoring, "HAS_PSUTIL", False)
    original_executor = monitoring.ThreadPoolExecutor
    failure = RuntimeError("executor shutdown unconfirmed")
    attempts = []

    class FailingShutdown(original_executor):
        def shutdown(self, *args, **kwargs):
            attempts.append("shutdown")
            super().shutdown(*args, **kwargs)
            raise failure

    monkeypatch.setattr(monitoring, "ThreadPoolExecutor", FailingShutdown)
    owner = monitoring.MonitoringService(enable_prometheus=False)
    check = monitoring.HealthCheck("service", lambda: True)
    owner.health_checker.add_health_check(check)
    try:
        assert (await check.run())["status"] == "error"
        assert check.get_status()["executor_owned"]
        assert (await check.run())["status"] == "unavailable"
        for _ in range(2):
            with pytest.raises(RuntimeError) as caught:
                await owner.aclose()
            assert caught.value is failure
        assert attempts == ["shutdown"]
        assert owner.get_status()["state"] == "failed"
        assert owner.metrics_collector.state.value == "running"
    finally:
        # The failure path intentionally retains dependencies. This test owns
        # their manual teardown after its fake executor actually joined.
        owner.metrics_collector.close()
        owner._close_listener()


@pytest.mark.asyncio
async def test_closed_checker_rejects_direct_list_insertion():
    owner = monitoring.HealthChecker()
    await owner.aclose()
    calls = []
    check = monitoring.HealthCheck("late", lambda: calls.append("called") or True)
    owner.health_checks.append(check)
    result = await owner.run_all_checks()
    assert result["checks"][0]["status"] == "unavailable"
    assert not calls
    await check.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [-1.0, float("nan"), float("inf"), True, "5"])
async def test_invalid_timeout_fails_before_invoking_dependency(timeout):
    calls = []
    owner = monitoring.HealthCheck(
        "service", lambda: calls.append("called"), timeout=timeout
    )
    try:
        with pytest.raises((TypeError, ValueError), match="timeout"):
            await owner.run()
        assert not calls
        assert owner.get_status()["active_probes"] == 0
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_task_creation_failure_releases_probe_admission(monkeypatch):
    original = asyncio.create_task
    failure = RuntimeError("task creation failed")
    owner = monitoring.HealthCheck("service", lambda: True)
    operations = []

    def fail(operation):
        operations.append(operation)
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(asyncio, "create_task", fail)
        with pytest.raises(RuntimeError) as caught:
            await owner.run()
        assert caught.value is failure
        assert inspect.getcoroutinestate(operations[0]) == inspect.CORO_CLOSED
        assert owner.get_status()["active_probes"] == 0
    assert asyncio.create_task is original
    try:
        assert (await owner.run())["status"] == "healthy"
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_eager_task_factory_preserves_shared_admission_and_cleanup():
    if not hasattr(asyncio, "eager_task_factory"):
        pytest.skip("Eager task factories require Python 3.12")
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    calls = []

    async def operation():
        calls.append("check")
        return True

    owner = monitoring.HealthCheck("service", operation)
    try:
        loop.set_task_factory(asyncio.eager_task_factory)
        first, second = await asyncio.gather(owner.run(), owner.run())
        assert first["status"] == second["status"] == "healthy"
        assert calls == ["check"]
        await owner.aclose()
        assert owner.state.value == "closed"
    finally:
        loop.set_task_factory(previous)
        await owner.aclose()
