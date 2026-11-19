"""Shared shutdown deadlines preserve cleanup ownership and terminal outcomes."""

import asyncio

import pytest

from stateset_agents.api import shutdown as shutdown_module
from stateset_agents.api.shutdown import GracefulShutdownManager, ShutdownPhase


@pytest.mark.asyncio
@pytest.mark.parametrize("same_phase", [False, True])
async def test_total_budget_is_shared_across_callbacks(monkeypatch, same_phase):
    now = 100.0
    monkeypatch.setattr(shutdown_module, "monotonic", lambda: now)
    manager = GracefulShutdownManager(total_timeout=10)
    calls = []

    async def first():
        nonlocal now
        calls.append("first")
        now += 6

    async def second():
        nonlocal now
        calls.append("second")
        now += 4

    async def forbidden():
        calls.append("forbidden")

    manager.register_task("first", ShutdownPhase.STOP_ACCEPTING, first)
    phase = ShutdownPhase.STOP_ACCEPTING if same_phase else ShutdownPhase.CLEANUP_JOBS
    manager.register_task("second", phase, second)
    manager.register_task("forbidden", ShutdownPhase.FINAL, forbidden)

    with pytest.raises(
        asyncio.TimeoutError, match="Total shutdown timeout"
    ) as first_error:
        await manager.execute_shutdown()
    with pytest.raises(asyncio.TimeoutError) as repeated:
        await manager.execute_shutdown()
    assert repeated.value is first_error.value
    assert calls == ["first", "second"]
    status = manager.get_status()
    assert status["outcome"] == "failed"
    assert status["deadline_exceeded"] is True
    assert status["completed_tasks"] == calls
    assert status["failed_tasks"] == []


@pytest.mark.asyncio
async def test_queued_callback_checks_deadline_before_starting_effects(monkeypatch):
    now = 100.0
    monkeypatch.setattr(shutdown_module, "monotonic", lambda: now)
    manager = GracefulShutdownManager(total_timeout=10)
    calls = []
    original_wait_for = asyncio.wait_for

    async def delayed_wait_for(operation, timeout):
        nonlocal now
        now += 10
        return await original_wait_for(operation, timeout)

    monkeypatch.setattr(shutdown_module.asyncio, "wait_for", delayed_wait_for)

    async def callback():
        calls.append("effect")

    manager.register_task("callback", ShutdownPhase.STOP_ACCEPTING, callback)
    with pytest.raises(asyncio.TimeoutError, match="Total shutdown timeout"):
        await manager.execute_shutdown()
    assert not calls
    assert manager.get_status()["deadline_exceeded"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("critical", [False, True])
@pytest.mark.parametrize("suppress_cancellation", [False, True])
@pytest.mark.parametrize("cancel_waiter", [False, True])
async def test_total_timeout_drains_active_callback_before_failure(
    critical, suppress_cancellation, cancel_waiter
):
    manager = GracefulShutdownManager(total_timeout=0.05)
    unwinding, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def jobs():
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            unwinding.set()
            await release.wait()
            calls.append("drained")
            if not suppress_cancellation:
                raise

    async def engines():
        calls.append("engines")

    manager.register_task("jobs", ShutdownPhase.CLEANUP_JOBS, jobs, critical=critical)
    manager.register_task("engines", ShutdownPhase.CLEANUP_ENGINES, engines)
    waiter = asyncio.create_task(manager.execute_shutdown())
    second = None
    try:
        await asyncio.wait_for(unwinding.wait(), timeout=2)
        assert not waiter.done()
        assert not calls
        status = manager.get_status()
        assert status["outcome"] == "running"
        assert status["deadline_exceeded"] is True
        second = asyncio.create_task(manager.execute_shutdown())
        if cancel_waiter:
            waiter.cancel()
            await asyncio.sleep(0)
            waiter.cancel()
            await asyncio.sleep(0)
            assert not waiter.done()
        release.set()
        expected_error = (
            asyncio.CancelledError if cancel_waiter else asyncio.TimeoutError
        )
        with pytest.raises(expected_error):
            await waiter
        with pytest.raises(
            asyncio.TimeoutError, match="Total shutdown timeout"
        ) as error:
            await second
        with pytest.raises(asyncio.TimeoutError) as repeated:
            await manager.execute_shutdown()
        assert repeated.value is error.value
        assert calls == ["drained"]
        status = manager.get_status()
        assert status["outcome"] == "failed"
        assert status["deadline_exceeded"] is True
        assert status["completed_tasks"] == (["jobs"] if suppress_cancellation else [])
        assert status["failed_tasks"] == ([] if suppress_cancellation else ["jobs"])
    finally:
        release.set()
        await asyncio.gather(
            waiter, *([second] if second else []), return_exceptions=True
        )


@pytest.mark.asyncio
async def test_total_deadline_bounds_request_drain_and_blocks_cleanup():
    manager = GracefulShutdownManager(drain_timeout=30, total_timeout=0.01)
    manager.increment_in_flight()
    calls = []

    async def cleanup():
        calls.append("cleanup")

    manager.register_task("cleanup", ShutdownPhase.CLEANUP_ENGINES, cleanup)
    with pytest.raises(asyncio.TimeoutError, match="Total shutdown timeout"):
        await asyncio.wait_for(manager.execute_shutdown(), timeout=2)
    status = manager.get_status()
    assert status["outcome"] == "failed"
    assert status["deadline_exceeded"] is True
    assert status["failed_tasks"] == ["drain_requests"]
    assert status["in_flight_requests"] == 1
    assert not calls


@pytest.mark.asyncio
async def test_shorter_noncritical_task_timeout_keeps_remaining_budget():
    manager = GracefulShutdownManager(total_timeout=10)
    calls = []

    async def optional_cleanup():
        try:
            await asyncio.Future()
        finally:
            calls.append("unwound")

    async def final():
        calls.append("final")

    manager.register_task(
        "optional", ShutdownPhase.CLEANUP_STATE, optional_cleanup, 0.01
    )
    manager.register_task("final", ShutdownPhase.FINAL, final)
    await manager.execute_shutdown()
    assert calls == ["unwound", "final"]
    status = manager.get_status()
    assert status["outcome"] == "completed_with_errors"
    assert status["deadline_exceeded"] is False
    assert status["failed_tasks"] == ["optional"]


@pytest.mark.asyncio
async def test_local_timeout_unwind_can_exhaust_total_budget_without_second_cancel(
    monkeypatch,
):
    now = 100.0
    monkeypatch.setattr(shutdown_module, "monotonic", lambda: now)
    manager = GracefulShutdownManager(total_timeout=10)
    unwinding, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def cleanup():
        nonlocal now
        try:
            await asyncio.Future()
        finally:
            now += 10
            unwinding.set()
            await release.wait()
            calls.append("unwound")

    async def final():
        calls.append("final")

    manager.register_task("cleanup", ShutdownPhase.CLEANUP_STATE, cleanup, 0.01)
    manager.register_task("final", ShutdownPhase.FINAL, final)
    waiter = asyncio.create_task(manager.execute_shutdown())
    try:
        await asyncio.wait_for(unwinding.wait(), timeout=2)
        assert not waiter.done()
        assert manager.get_status()["deadline_exceeded"] is True
        release.set()
        with pytest.raises(asyncio.TimeoutError, match="Total shutdown timeout"):
            await waiter
        assert calls == ["unwound"]
        assert manager.get_status()["failed_tasks"] == ["cleanup"]
    finally:
        release.set()
        await asyncio.gather(waiter, return_exceptions=True)


@pytest.mark.asyncio
async def test_success_does_not_become_timed_out_after_completion(monkeypatch):
    now = 100.0
    monkeypatch.setattr(shutdown_module, "monotonic", lambda: now)
    manager = GracefulShutdownManager(total_timeout=10)
    await manager.execute_shutdown()
    now += 100
    await manager.execute_shutdown()
    assert manager.get_status()["outcome"] == "completed"
    assert manager.get_status()["deadline_exceeded"] is False


@pytest.mark.asyncio
async def test_zero_budget_admits_no_callbacks():
    manager = GracefulShutdownManager(total_timeout=0)
    calls = []

    async def callback():
        calls.append("callback")

    manager.register_task("callback", ShutdownPhase.STOP_ACCEPTING, callback)
    with pytest.raises(asyncio.TimeoutError, match="Total shutdown timeout"):
        await manager.execute_shutdown()
    assert manager.is_shutting_down
    assert not calls


@pytest.mark.parametrize("timeout", [-1, float("-inf"), float("nan")])
def test_invalid_total_budget_is_rejected(timeout):
    with pytest.raises(ValueError, match="total_timeout"):
        GracefulShutdownManager(total_timeout=timeout)


@pytest.mark.asyncio
async def test_infinite_budget_preserves_phase_order():
    manager = GracefulShutdownManager(total_timeout=float("inf"))
    calls = []

    async def first():
        calls.append("first")

    async def final():
        calls.append("final")

    manager.register_task("final", ShutdownPhase.FINAL, final)
    manager.register_task("first", ShutdownPhase.STOP_ACCEPTING, first)
    await manager.execute_shutdown()
    assert calls == ["first", "final"]
    assert manager.get_status()["outcome"] == "completed"
