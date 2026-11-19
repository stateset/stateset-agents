"""Timeouts distinguish deadlines from upstream errors and retain cleanup."""

import asyncio
import inspect
from types import SimpleNamespace

import pytest

from stateset_agents.api import resilience
from stateset_agents.api.resilience import Bulkhead, BulkheadConfig
from stateset_agents.api.resilience import TimeoutError as OperationTimeoutError
from stateset_agents.api.resilience import timeout


@pytest.mark.asyncio
async def test_upstream_timeout_keeps_its_type_and_identity():
    failure = asyncio.TimeoutError("upstream connect timeout")

    @timeout(30)
    async def operation():
        raise failure

    with pytest.raises(asyncio.TimeoutError) as caught:
        await operation()
    assert caught.value is failure


@pytest.mark.asyncio
async def test_deadline_expiry_is_not_success_when_upstream_suppresses_cancellation():
    calls = []

    @timeout(0.01)
    async def operation():
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            calls.append("unwound")
            return "late result"

    with pytest.raises(OperationTimeoutError, match="timed out"):
        await operation()
    assert calls == ["unwound"]


@pytest.mark.asyncio
async def test_repeated_caller_cancellation_waits_for_upstream_cleanup():
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    workers, calls = [], []

    @timeout(30)
    async def operation():
        workers.append(asyncio.current_task())
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cleaning.set()
            await release.wait()
            calls.append("cleaned")

    task = asyncio.create_task(operation())
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        await asyncio.wait_for(cleaning.wait(), timeout=2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert calls == ["cleaned"]
        assert all(worker.done() for worker in workers)
    finally:
        release.set()
        await asyncio.gather(task, *workers, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_caller", [False, True])
@pytest.mark.parametrize("cleanup_outcome", ["cancelled", "returned", "failed"])
async def test_deadline_drains_cleanup_and_keeps_error_precedence(
    cancel_caller, cleanup_outcome
):
    cleaning, release = asyncio.Event(), asyncio.Event()
    calls = []
    failure = ValueError("cleanup failed")

    @timeout(0.02)
    async def operation():
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cleaning.set()
            await release.wait()
            calls.append("cleaned")
            if cleanup_outcome == "failed":
                raise failure from None
            if cleanup_outcome == "returned":
                return "late result"
            raise

    task = asyncio.create_task(operation())
    try:
        await asyncio.wait_for(cleaning.wait(), timeout=2)
        assert not task.done()
        if cancel_caller:
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
        assert not task.done()
        assert not calls
        release.set()
        expected = (
            asyncio.CancelledError
            if cancel_caller
            else ValueError if cleanup_outcome == "failed" else OperationTimeoutError
        )
        with pytest.raises(expected) as caught:
            await task
        if not cancel_caller and cleanup_outcome == "failed":
            assert caught.value is failure
        assert calls == ["cleaned"]
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_expired_before_worker_starts_cannot_invoke_user_work(monkeypatch):
    now = 100.0
    calls = []
    monkeypatch.setattr(resilience, "time", SimpleNamespace(monotonic=lambda: now))

    def advance_clock():
        nonlocal now
        now += 30

    @timeout(10)
    async def operation():
        calls.append("effect")

    # Advance the monotonic clock after the wrapper starts but before its worker.
    asyncio.get_running_loop().call_soon(advance_clock)
    with pytest.raises(OperationTimeoutError):
        await operation()
    assert not calls


@pytest.mark.asyncio
@pytest.mark.parametrize("late_completion", [False, True])
async def test_deadline_uses_completion_time_not_observer_scheduling(
    monkeypatch, late_completion
):
    now = 100.0
    monkeypatch.setattr(resilience, "time", SimpleNamespace(monotonic=lambda: now))

    def advance_clock():
        nonlocal now
        now += 30

    @timeout(10)
    async def operation():
        if late_completion:
            # Simulate upstream blocking the loop past the deadline before returning.
            advance_clock()
        else:
            # The operation finishes on time; only its observer is scheduled late.
            asyncio.get_running_loop().call_soon(advance_clock)
        return "result"

    if late_completion:
        with pytest.raises(OperationTimeoutError):
            await operation()
    else:
        assert await operation() == "result"


@pytest.mark.asyncio
@pytest.mark.parametrize("seconds", [0, -1])
async def test_nonpositive_deadline_rejects_before_invocation(seconds):
    calls = []

    @timeout(seconds)
    async def operation():
        calls.append("effect")

    with pytest.raises(OperationTimeoutError):
        await operation()
    assert not calls


@pytest.mark.asyncio
@pytest.mark.parametrize("outer_timeout", [False, True])
async def test_timeout_and_bulkhead_composition_keeps_capacity_through_cleanup(
    outer_timeout,
):
    bulkhead = Bulkhead("composition", BulkheadConfig(max_concurrent=1, max_waiting=1))
    cleaning, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def operation():
        try:
            await asyncio.Future()
        finally:
            cleaning.set()
            await release.wait()
            calls.append("cleaned")

    async def successor():
        calls.append("next")

    wrapped = (
        timeout(0.02)(bulkhead(operation))
        if outer_timeout
        else bulkhead(timeout(0.02)(operation))
    )
    task = asyncio.create_task(wrapped())
    next_task = None
    try:
        await asyncio.wait_for(cleaning.wait(), timeout=2)
        assert not task.done()
        assert bulkhead.stats["active"] == 1
        next_task = asyncio.create_task(bulkhead.execute(successor))
        await asyncio.sleep(0)
        assert bulkhead.stats["waiting"] == 1
        assert not calls
        release.set()
        with pytest.raises(OperationTimeoutError):
            await task
        await next_task
        assert calls == ["cleaned", "next"]
        assert bulkhead.stats["active"] == bulkhead.stats["waiting"] == 0
    finally:
        release.set()
        await asyncio.gather(
            task, *([next_task] if next_task else []), return_exceptions=True
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("seconds", [None, float("inf"), 30])
async def test_success_keeps_result_identity_and_decorator_metadata(seconds):
    result = object()

    async def operation(value: str) -> object:
        """One operation."""
        return result

    wrapped = timeout(seconds)(operation)
    assert await wrapped("value") is result
    assert wrapped.__name__ == operation.__name__
    assert wrapped.__doc__ == operation.__doc__
    assert inspect.signature(wrapped) == inspect.signature(operation)
    assert inspect.iscoroutinefunction(wrapped)


def test_nan_timeout_is_rejected_before_work():
    with pytest.raises(ValueError, match="NaN"):
        timeout(float("nan"))
