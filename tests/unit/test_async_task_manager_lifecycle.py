"""Admission, cancellation, and resource ownership for the async task manager."""

import asyncio
import gc
import inspect
import weakref

import pytest

from stateset_agents.core import async_pool
from stateset_agents.core.async_pool import AsyncTaskManager, TaskManagerState


@pytest.mark.asyncio
async def test_start_is_idempotent_and_closed_manager_cannot_restart():
    manager = AsyncTaskManager()
    await manager.start()
    cleanup = manager._cleanup_task
    try:
        await manager.start()
        assert manager._cleanup_task is cleanup
        await manager.shutdown()
        assert cleanup.done()
        with pytest.raises(RuntimeError, match="shut"):
            await manager.start()
    finally:
        cleanup.cancel()
        await asyncio.gather(cleanup, return_exceptions=True)
        await manager.shutdown()


@pytest.mark.asyncio
async def test_shutdown_rejects_new_coroutine_without_running_or_leaking_it():
    manager = AsyncTaskManager()
    await manager.shutdown()
    calls = []

    async def operation():
        calls.append("effect")

    coroutine = operation()
    try:
        with pytest.raises(RuntimeError, match="shut"):
            await manager.submit_task(coroutine)
        assert not calls
        assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED
    finally:
        coroutine.close()


@pytest.mark.asyncio
async def test_shutdown_wakes_waiters_and_does_not_admit_them():
    manager = AsyncTaskManager(max_concurrent_tasks=1)
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def active():
        entered.set()
        await release.wait()
        return "active result"

    async def queued():
        calls.append("queued effect")

    first = asyncio.create_task(manager.submit_task(active()))
    coroutine = queued()
    waiting = closing = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        waiting = asyncio.create_task(manager.submit_task(coroutine))
        await asyncio.sleep(0)
        closing = asyncio.create_task(manager.shutdown())
        with pytest.raises(RuntimeError, match="shut"):
            await asyncio.wait_for(asyncio.shield(waiting), timeout=0.2)
        assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED
        assert not calls
        assert not closing.done()
        release.set()
        assert await first == "active result"
        await closing
        assert manager.get_status()["active_tasks"] == 0
    finally:
        release.set()
        if waiting is not None:
            waiting.cancel()
        await asyncio.gather(
            first,
            *([waiting] if waiting else []),
            *([closing] if closing else []),
            return_exceptions=True,
        )
        coroutine.close()
        await manager.shutdown()


@pytest.mark.asyncio
async def test_cancelled_queue_waiter_closes_unstarted_coroutine():
    manager = AsyncTaskManager(max_concurrent_tasks=1)
    entered, release = asyncio.Event(), asyncio.Event()

    async def active():
        entered.set()
        await release.wait()

    async def queued():
        pytest.fail("Cancelled queued work ran")

    first = asyncio.create_task(manager.submit_task(active()))
    coroutine = queued()
    waiting = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        waiting = asyncio.create_task(manager.submit_task(coroutine))
        await asyncio.sleep(0)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED
    finally:
        release.set()
        await asyncio.gather(
            first, *([waiting] if waiting else []), return_exceptions=True
        )
        coroutine.close()
        await manager.shutdown()


@pytest.mark.asyncio
async def test_repeated_submitter_cancellation_keeps_slot_until_cleanup_returns():
    manager = AsyncTaskManager(max_concurrent_tasks=1)
    entered, unwinding, release = [asyncio.Event() for _ in range(3)]
    calls = []

    async def operation():
        try:
            entered.set()
            await asyncio.Future()
        finally:
            unwinding.set()
            await release.wait()
            calls.append("cleaned")

    async def next_operation():
        calls.append("next")

    first = asyncio.create_task(manager.submit_task(operation()))
    next_task = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        first.cancel()
        await asyncio.wait_for(unwinding.wait(), timeout=2)
        first.cancel()
        await asyncio.sleep(0)
        first.cancel()
        next_task = asyncio.create_task(manager.submit_task(next_operation()))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not first.done()
        assert not calls
        assert manager.get_status()["active_tasks"] == 1
        assert manager.get_status()["available_slots"] == 0
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        await next_task
        assert calls == ["cleaned", "next"]
    finally:
        release.set()
        await asyncio.gather(
            first, *([next_task] if next_task else []), return_exceptions=True
        )
        await manager.shutdown()


@pytest.mark.asyncio
async def test_zero_timeout_does_not_start_coroutine():
    manager = AsyncTaskManager()
    calls = []

    async def operation():
        calls.append("effect")

    coroutine = operation()
    try:
        with pytest.raises(asyncio.TimeoutError):
            await manager.submit_task(coroutine, timeout=0)
        assert not calls
        assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED
    finally:
        coroutine.close()
        await manager.shutdown()


@pytest.mark.asyncio
async def test_concurrent_shutdown_waiters_join_without_cancelling_active_work():
    manager = AsyncTaskManager()
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def operation():
        entered.set()
        await release.wait()
        calls.append("finished")
        return 42

    submitted = asyncio.create_task(manager.submit_task(operation()))
    first = second = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        first = asyncio.create_task(manager.shutdown())
        await asyncio.sleep(0)
        second = asyncio.create_task(manager.shutdown())
        await asyncio.sleep(0)
        first.cancel()
        await asyncio.sleep(0)
        first.cancel()
        await asyncio.sleep(0)
        assert not first.done()
        assert not second.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        await second
        assert await submitted == 42
        await manager.shutdown()
        assert calls == ["finished"]
    finally:
        release.set()
        await asyncio.gather(
            submitted,
            *([first] if first else []),
            *([second] if second else []),
            return_exceptions=True,
        )
        await manager.shutdown()


@pytest.mark.asyncio
async def test_global_getter_replaces_closed_owner(monkeypatch):
    manager = AsyncTaskManager()
    await manager.shutdown()
    monkeypatch.setattr(async_pool, "_task_manager", manager)
    replacement = await async_pool.get_task_manager()
    try:
        assert replacement is not manager
        assert await replacement.submit_task(asyncio.sleep(0, result=42)) == 42
    finally:
        await replacement.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("initiator", ["timeout", "caller", "shutdown"])
async def test_cancellation_sources_share_one_unwind(initiator):
    manager = AsyncTaskManager(shutdown_timeout=0)
    entered, unwinding, release = [asyncio.Event() for _ in range(3)]
    calls = []

    async def operation():
        try:
            entered.set()
            await asyncio.Future()
        finally:
            unwinding.set()
            await release.wait()
            calls.append("cleaned")

    submitted = asyncio.create_task(
        manager.submit_task(operation(), timeout=0.01 if initiator == "timeout" else 30)
    )
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        if initiator == "caller":
            submitted.cancel()
        elif initiator == "shutdown":
            closing = asyncio.create_task(manager.shutdown())
        await asyncio.wait_for(unwinding.wait(), timeout=2)
        if closing is None:
            closing = asyncio.create_task(manager.shutdown())
        submitted.cancel()
        await asyncio.sleep(0)
        submitted.cancel()
        await asyncio.sleep(0)
        assert not submitted.done()
        assert not closing.done()
        assert manager.state is TaskManagerState.DRAINING
        assert manager.get_status()["active_tasks"] == 1
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await submitted
        await closing
        assert calls == ["cleaned"]
        assert manager.get_status() == {
            "state": "closed",
            "active_tasks": 0,
            "waiting_tasks": 0,
            "available_slots": 0,
            "max_concurrent_tasks": 100,
        }
    finally:
        release.set()
        await asyncio.gather(
            submitted, *([closing] if closing else []), return_exceptions=True
        )
        await manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("suppress_timeout", [False, True])
async def test_timeout_retains_ownership_and_preserves_wait_for_result(
    suppress_timeout,
):
    manager = AsyncTaskManager()
    unwinding, release = asyncio.Event(), asyncio.Event()

    async def operation():
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            unwinding.set()
            await release.wait()
            if suppress_timeout:
                return "recovered"
            raise

    submitted = asyncio.create_task(manager.submit_task(operation(), timeout=0.01))
    try:
        await asyncio.wait_for(unwinding.wait(), timeout=2)
        assert not submitted.done()
        assert manager.get_status()["active_tasks"] == 1
        release.set()
        if suppress_timeout:
            assert await submitted == "recovered"
        else:
            with pytest.raises(asyncio.TimeoutError):
                await submitted
        assert manager.get_status()["active_tasks"] == 0
        assert manager.get_status()["available_slots"] == manager.max_concurrent_tasks
    finally:
        release.set()
        await asyncio.gather(submitted, return_exceptions=True)
        await manager.shutdown()


@pytest.mark.asyncio
async def test_cancel_before_operation_entry_has_no_effects(monkeypatch):
    manager = AsyncTaskManager()
    entered, release = asyncio.Event(), asyncio.Event()
    run_operation = manager._run_operation
    calls = []

    async def gated_operation(coro, timeout, cancellation):
        entered.set()
        await release.wait()
        return await run_operation(coro, timeout, cancellation)

    monkeypatch.setattr(manager, "_run_operation", gated_operation)

    async def operation():
        calls.append("effect")

    coroutine = operation()
    submitted = asyncio.create_task(manager.submit_task(coroutine))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        submitted.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await submitted
        assert not calls
        assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED
    finally:
        release.set()
        await asyncio.gather(submitted, return_exceptions=True)
        await manager.shutdown()


@pytest.mark.asyncio
async def test_rejected_future_remains_owned_by_caller():
    manager = AsyncTaskManager()
    await manager.shutdown()
    future = asyncio.get_running_loop().create_future()
    with pytest.raises(RuntimeError, match="shut"):
        await manager.submit_task(future)
    assert not future.done()
    future.set_result(42)
    assert await future == 42


@pytest.mark.asyncio
async def test_admitted_tasks_futures_and_custom_awaitables_keep_results():
    manager = AsyncTaskManager()

    class CustomAwaitable:
        def __await__(self):
            return asyncio.sleep(0, result="custom").__await__()

    future = asyncio.get_running_loop().create_future()
    future.set_result("future")
    task = asyncio.create_task(asyncio.sleep(0, result="task"))
    try:
        assert manager.state is TaskManagerState.NEW
        assert await manager.submit_batch([future, task, CustomAwaitable()]) == [
            "future",
            "task",
            "custom",
        ]
        completed = asyncio.get_running_loop().create_future()
        completed.set_result("already done")
        assert await manager.submit_task(completed, timeout=0) == "already done"
        assert manager.state is TaskManagerState.RUNNING
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_batch", [False, True])
async def test_batch_failure_or_cancellation_drains_siblings(cancel_batch):
    manager = AsyncTaskManager(max_concurrent_tasks=2)
    entered, fail, unwinding, release = [asyncio.Event() for _ in range(4)]
    calls = []
    error = ValueError("batch failed")

    async def failing():
        await fail.wait()
        raise error

    async def sibling():
        try:
            entered.set()
            await asyncio.Future()
        finally:
            unwinding.set()
            await release.wait()
            calls.append("cleaned")

    batch = asyncio.create_task(manager.submit_batch([failing(), sibling()]))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        if cancel_batch:
            batch.cancel()
        else:
            fail.set()
        await asyncio.wait_for(unwinding.wait(), timeout=2)
        assert not batch.done()
        if cancel_batch:
            batch.cancel()
            await asyncio.sleep(0)
            batch.cancel()
        assert not calls
        release.set()
        expected = asyncio.CancelledError if cancel_batch else ValueError
        with pytest.raises(expected) as caught:
            await batch
        if not cancel_batch:
            assert caught.value is error
        assert calls == ["cleaned"]
        assert manager.get_status()["active_tasks"] == 0
    finally:
        fail.set()
        release.set()
        await asyncio.gather(batch, return_exceptions=True)
        await manager.shutdown()


@pytest.mark.asyncio
async def test_batch_cancelled_before_submitters_enter_closes_coroutines():
    manager = AsyncTaskManager()

    async def operation():
        pytest.fail("Cancelled batch performed an effect")

    coroutines = [operation(), operation()]
    batch = asyncio.create_task(manager.submit_batch(coroutines))
    await asyncio.sleep(0)
    batch.cancel()
    with pytest.raises(asyncio.CancelledError):
        await batch
    assert all(
        inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for coro in coroutines
    )
    await manager.shutdown()


@pytest.mark.asyncio
async def test_batch_exception_results_preserve_order_and_successful_siblings():
    manager = AsyncTaskManager(max_concurrent_tasks=1)
    error = KeyError("expected")

    async def failing():
        raise error

    try:
        results = await manager.submit_batch(
            [asyncio.sleep(0, result=1), failing(), asyncio.sleep(0, result=3)],
            return_exceptions=True,
        )
        assert results == [1, error, 3]
        assert manager.get_status()["active_tasks"] == 0
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_managed_operation_cannot_deadlock_by_shutting_down_its_own_manager():
    manager = AsyncTaskManager()
    try:
        with pytest.raises(RuntimeError, match="outside a managed task"):
            await manager.submit_task(manager.shutdown())
        assert manager.state is TaskManagerState.RUNNING
        assert await manager.submit_task(asyncio.sleep(0, result=42)) == 42
    finally:
        await manager.shutdown()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_concurrent_tasks": 0},
        {"max_concurrent_tasks": -1},
        {"cleanup_interval": 0},
        {"cleanup_interval": float("nan")},
        {"task_timeout": float("nan")},
        {"shutdown_timeout": -1},
        {"shutdown_timeout": float("nan")},
    ],
)
def test_invalid_capacity_or_timing_fails_before_start(kwargs):
    with pytest.raises(ValueError):
        AsyncTaskManager(**kwargs)


@pytest.mark.asyncio
async def test_capacity_and_waiter_observations_match_actual_running_operations():
    manager = AsyncTaskManager(max_concurrent_tasks=3)
    saturated, release = asyncio.Event(), asyncio.Event()
    running = peak = 0

    async def operation(value):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        if running == 3:
            saturated.set()
        try:
            await release.wait()
            return value
        finally:
            running -= 1

    batch = asyncio.create_task(manager.submit_batch([operation(i) for i in range(20)]))
    try:
        await asyncio.wait_for(saturated.wait(), timeout=2)
        status = manager.get_status()
        assert status["active_tasks"] == 3
        assert status["waiting_tasks"] == 17
        assert status["available_slots"] == 0
        release.set()
        assert await batch == list(range(20))
        assert peak == 3
        assert running == 0
        assert manager.get_status()["waiting_tasks"] == 0
    finally:
        release.set()
        await asyncio.gather(batch, return_exceptions=True)
        await manager.shutdown()


@pytest.mark.asyncio
async def test_dropped_submitter_reference_cannot_abandon_owned_operation():
    manager = AsyncTaskManager()
    entered = asyncio.Event()
    references = {}

    async def operation():
        release = asyncio.get_running_loop().create_future()
        references["release"] = weakref.ref(release)
        entered.set()
        await release
        return "finished"

    submitted = asyncio.create_task(manager.submit_task(operation()))
    reference = weakref.ref(submitted)
    del submitted
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        gc.collect()
        assert reference() is not None
        release = references["release"]()
        assert release is not None
        release.set_result(None)
        assert await reference() == "finished"
        assert manager.get_status()["active_tasks"] == 0
    finally:
        release = references.get("release", lambda: None)()
        if release is not None and not release.done():
            release.set_result(None)
        if reference() is not None:
            await asyncio.gather(reference(), return_exceptions=True)
        await manager.shutdown()


@pytest.mark.asyncio
async def test_global_getter_does_not_replace_a_draining_owner(monkeypatch):
    manager = AsyncTaskManager()
    monkeypatch.setattr(async_pool, "_task_manager", manager)
    entered, release = asyncio.Event(), asyncio.Event()

    async def operation():
        entered.set()
        await release.wait()

    submitted = asyncio.create_task(manager.submit_task(operation()))
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        closing = asyncio.create_task(manager.shutdown())
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="shut"):
            await async_pool.get_task_manager()
        assert async_pool._task_manager is manager
    finally:
        release.set()
        await asyncio.gather(
            submitted, *([closing] if closing else []), return_exceptions=True
        )
        await manager.shutdown()


@pytest.mark.asyncio
async def test_queued_admission_preserves_order_when_head_waiter_is_cancelled():
    manager = AsyncTaskManager(max_concurrent_tasks=1)
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def active():
        entered.set()
        await release.wait()

    async def queued(value):
        calls.append(value)

    first = asyncio.create_task(manager.submit_task(active()))
    waiters = []
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        for value in range(4):
            waiters.append(asyncio.create_task(manager.submit_task(queued(value))))
            await asyncio.sleep(0)
        waiters[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiters[0]
        assert manager.get_status()["waiting_tasks"] == 3
        release.set()
        late = asyncio.create_task(manager.submit_task(queued(4)))
        waiters.append(late)
        await first
        await asyncio.gather(*waiters[1:])
        assert calls == [1, 2, 3, 4]
    finally:
        release.set()
        await asyncio.gather(first, *waiters, return_exceptions=True)
        await manager.shutdown()
