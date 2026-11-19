"""Pool ownership regressions using local resources and explicit event barriers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from stateset_agents.core.async_pool import (
    AsyncResourceFactory,
    AsyncResourcePool,
    PooledResource,
    PoolState,
    ResourceState,
)
from stateset_agents.core.error_handling import ResourceException


@dataclass(eq=False)
class Resource:
    number: int
    closed: bool = False
    valid: bool = True


class Factory(AsyncResourceFactory[Resource]):
    def __init__(self) -> None:
        self.created: list[Resource] = []
        self.cleaned: list[Resource] = []

    async def create_resource(self, **kwargs) -> Resource:
        resource = Resource(len(self.created))
        self.created.append(resource)
        return resource

    async def validate_resource(self, resource: Resource) -> bool:
        assert not resource.closed, "A closed resource reached validation"
        return resource.valid

    async def cleanup_resource(self, resource: Resource) -> None:
        self.cleaned.append(resource)
        assert not resource.closed, "Resource cleanup ran twice"
        resource.closed = True


def make_pool(factory: Factory, **kwargs) -> AsyncResourcePool[Resource]:
    return AsyncResourcePool(
        factory, health_check_interval=0, acquire_timeout=0.01, **kwargs
    )


def counts(pool: AsyncResourcePool[Resource]) -> tuple[int, int, int]:
    stats = pool.get_stats()
    return stats.total_connections, stats.active_connections, stats.idle_connections


@pytest.mark.asyncio
async def test_failed_wrapper_cannot_be_released_back_to_idle():
    pool = make_pool(Factory())
    resource = PooledResource(Resource(0), "one", pool)
    await resource.acquire()
    await resource.mark_failed(RuntimeError("broken connection"))
    await resource.release()
    assert resource.info.state is ResourceState.FAILED
    with pytest.raises(ValueError, match="not idle"):
        await resource.acquire()


@pytest.mark.asyncio
async def test_failed_borrow_is_retired_once_and_replaced():
    factory = Factory()
    pool = make_pool(factory, max_size=1)
    await pool.initialize()
    try:
        with pytest.raises(RuntimeError, match="request failed"):
            async with pool.acquire() as failed:
                raise RuntimeError("request failed")
        assert factory.cleaned == [failed]
        assert counts(pool) == (0, 0, 0)
        assert pool.get_stats().failed_requests == 1
        async with pool.acquire() as replacement:
            assert replacement is not failed
            assert counts(pool) == (1, 1, 0)
        assert counts(pool) == (1, 0, 1)
    finally:
        await pool.close()
    assert factory.cleaned == factory.created


@pytest.mark.asyncio
async def test_scale_down_completes_and_removed_resources_cannot_be_borrowed():
    factory = Factory()
    pool = make_pool(factory, min_size=1, max_size=3)
    await pool.initialize()
    try:
        await pool.scale(3)
        assert counts(pool) == (3, 0, 3)
        await asyncio.wait_for(pool.scale(1), timeout=1)
        assert counts(pool) == (1, 0, 1)
        assert len(factory.cleaned) == 2
        for _ in range(3):
            async with pool.acquire() as resource:
                assert resource not in factory.cleaned
    finally:
        await pool.close()
    assert len(factory.cleaned) == 3


@pytest.mark.asyncio
async def test_health_check_replaces_invalid_idle_resources_without_stale_queue_entries():
    factory = Factory()
    pool = make_pool(factory, max_size=1)
    await pool.initialize()
    try:
        original = factory.created[0]
        original.valid = False
        await pool._perform_health_check()
        assert factory.cleaned == [original]
        async with pool.acquire() as resource:
            assert resource is not original
        assert counts(pool) == (1, 0, 1)
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_health_check_never_closes_an_aged_borrowed_resource():
    factory = Factory()
    pool = make_pool(factory, max_size=1)
    await pool.initialize()
    try:
        async with pool.acquire() as resource:
            pool._pool[0].info.created_at -= pool.max_age + 1
            await pool._perform_health_check()
            assert not resource.closed
            assert counts(pool) == (1, 1, 0)
        assert resource.closed
        assert counts(pool) == (0, 0, 0)
    finally:
        await pool.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ValueError("invalid"), KeyError("invalid")])
async def test_validation_error_retires_claimed_resource(error):
    class InvalidFactory(Factory):
        async def validate_resource(self, resource):
            raise error

    factory = InvalidFactory()
    pool = make_pool(factory)
    await pool.initialize()
    try:
        with pytest.raises(type(error)):
            async with pool.acquire():
                pytest.fail("Invalid resource was yielded")
        assert factory.cleaned == factory.created
        assert counts(pool) == (0, 0, 0)
        assert pool.get_stats().failed_requests == 1
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_cancelled_validation_retires_resource():
    validating = asyncio.Event()

    class BlockingFactory(Factory):
        async def validate_resource(self, resource):
            validating.set()
            await asyncio.Future()

    factory = BlockingFactory()
    pool = make_pool(factory)
    await pool.initialize()

    async def borrow():
        async with pool.acquire():
            pytest.fail("Cancelled validation was yielded")

    task = asyncio.create_task(borrow())
    try:
        await asyncio.wait_for(validating.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert factory.cleaned == factory.created
        assert counts(pool) == (0, 0, 0)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await pool.close()


@pytest.mark.asyncio
async def test_partial_initialization_failure_releases_all_membership():
    class FailingFactory(Factory):
        async def create_resource(self, **kwargs):
            if self.created:
                raise OSError("creation failed")
            return await super().create_resource(**kwargs)

    factory = FailingFactory()
    pool = make_pool(factory, min_size=2)
    with pytest.raises(ResourceException):
        await pool.initialize()
    await pool.close()
    assert factory.cleaned == factory.created
    assert counts(pool) == (0, 0, 0)


@pytest.mark.asyncio
async def test_close_clears_stats_and_is_terminal():
    factory = Factory()
    pool = make_pool(factory)
    await pool.initialize()
    await pool.close()
    await pool.close()
    assert counts(pool) == (0, 0, 0)
    assert pool._state is PoolState.CLOSED
    assert factory.cleaned == factory.created
    with pytest.raises(RuntimeError):
        await pool.scale(1)
    with pytest.raises(RuntimeError):
        await pool.initialize()
    async with pool._lock:
        assert pool._available.empty()


@pytest.mark.asyncio
async def test_concurrent_initialization_has_one_resource_set_and_health_task():
    factory = Factory()
    pool = AsyncResourcePool(factory, min_size=2, health_check_interval=100)
    await asyncio.gather(pool.initialize(), pool.initialize())
    try:
        assert len(factory.created) == 2
        assert pool.state is PoolState.READY
        assert counts(pool) == (2, 0, 2)
        health_task = pool._health_check_task
        await pool.initialize()
        assert pool._health_check_task is health_task
    finally:
        await pool.close()
    assert health_task.done()
    assert factory.cleaned == factory.created


@pytest.mark.asyncio
async def test_drain_rejects_waiters_without_waiting_for_acquire_timeout(monkeypatch):
    factory = Factory()
    pool = make_pool(factory, max_size=1)
    await pool.initialize()
    waiting = asyncio.Event()
    wait = pool._condition.wait

    async def observed_wait():
        waiting.set()
        return await wait()

    monkeypatch.setattr(pool._condition, "wait", observed_wait)

    async def borrow():
        async with pool.acquire(timeout=100):
            pytest.fail("Waiting borrower was admitted after draining")

    async with pool.acquire() as first:
        waiter = asyncio.create_task(borrow())
        await asyncio.wait_for(waiting.wait(), timeout=1)
        drain = asyncio.create_task(pool.drain())
        try:
            with pytest.raises(RuntimeError, match="not ready"):
                await asyncio.wait_for(waiter, timeout=1)
            assert pool.state is PoolState.DRAINING
            assert not drain.done()
            assert not first.closed
        finally:
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
    await asyncio.wait_for(drain, timeout=1)
    await pool.close()
    assert counts(pool) == (0, 0, 0)


@pytest.mark.asyncio
async def test_validation_is_borrowed_ownership_during_health_scale_and_close():
    validating, proceed, using, finish = (asyncio.Event() for _ in range(4))

    class BlockingFactory(Factory):
        async def validate_resource(self, resource):
            if resource.number == 0:
                validating.set()
                await proceed.wait()
            return await super().validate_resource(resource)

    factory = BlockingFactory()
    pool = make_pool(factory, min_size=1, max_size=2)
    await pool.initialize()
    await pool.scale(2)

    async def borrow():
        async with pool.acquire() as resource:
            using.set()
            await finish.wait()
            assert not resource.closed

    task = asyncio.create_task(borrow())
    close = None
    try:
        await asyncio.wait_for(validating.wait(), timeout=1)
        assert counts(pool) == (2, 1, 1)
        await asyncio.wait_for(pool._perform_health_check(), timeout=1)
        await asyncio.wait_for(pool.scale(1), timeout=1)
        assert factory.cleaned == [factory.created[1]]
        close = asyncio.create_task(pool.close())
        async with pool._condition:
            await asyncio.wait_for(
                pool._condition.wait_for(lambda: pool.state is PoolState.DRAINING),
                timeout=1,
            )
        assert not close.done()
        assert not factory.created[0].closed
        proceed.set()
        await asyncio.wait_for(using.wait(), timeout=1)
        assert not close.done()
        finish.set()
        await asyncio.wait_for(task, timeout=1)
        await asyncio.wait_for(close, timeout=1)
        assert counts(pool) == (0, 0, 0)
    finally:
        proceed.set()
        finish.set()
        await task
        if close is not None:
            await close
        await pool.close()


@pytest.mark.asyncio
async def test_cancelled_creation_is_drained_and_rolled_back_during_initialization():
    created, proceed = asyncio.Event(), asyncio.Event()

    class BlockingFactory(Factory):
        async def create_resource(self, **kwargs):
            resource = await super().create_resource(**kwargs)
            created.set()
            await proceed.wait()
            return resource

    factory = BlockingFactory()
    pool = make_pool(factory)
    initialize = asyncio.create_task(pool.initialize())
    try:
        await asyncio.wait_for(created.wait(), timeout=1)
        initialize.cancel()
        await asyncio.sleep(0)
        initialize.cancel()
        assert not initialize.done()
        proceed.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(initialize, timeout=1)
        assert pool.state is PoolState.CLOSED
        assert factory.cleaned == factory.created
        assert counts(pool) == (0, 0, 0)
    finally:
        proceed.set()
        await asyncio.gather(initialize, return_exceptions=True)
        await pool.close()


@pytest.mark.asyncio
async def test_cancelled_growth_keeps_created_resource_owned_and_reusable():
    created, proceed = asyncio.Event(), asyncio.Event()

    class BlockingFactory(Factory):
        async def create_resource(self, **kwargs):
            resource = await super().create_resource(**kwargs)
            created.set()
            await proceed.wait()
            return resource

    factory = BlockingFactory()
    pool = make_pool(factory, min_size=0, max_size=1)
    await pool.initialize()

    async def borrow():
        async with pool.acquire():
            pytest.fail("Cancelled borrower must not receive a resource")

    borrower = asyncio.create_task(borrow())
    try:
        await asyncio.wait_for(created.wait(), timeout=1)
        borrower.cancel()
        await asyncio.sleep(0)
        proceed.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(borrower, timeout=1)
        assert counts(pool) == (1, 0, 1)
        async with pool.acquire() as resource:
            assert resource is factory.created[0]
    finally:
        proceed.set()
        await asyncio.gather(borrower, return_exceptions=True)
        await pool.close()
    assert factory.cleaned == factory.created


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["close", "scale", "borrow"])
async def test_repeated_cancellation_waits_for_one_cleanup_attempt(operation):
    cleaning, proceed = asyncio.Event(), asyncio.Event()

    class BlockingFactory(Factory):
        async def cleanup_resource(self, resource):
            cleaning.set()
            await proceed.wait()
            await super().cleanup_resource(resource)

    factory = BlockingFactory()
    pool = make_pool(factory, min_size=0, max_size=1)
    await pool.initialize()
    await pool.scale(1)

    async def fail_borrow():
        async with pool.acquire():
            raise ValueError("failed request")

    work = {"close": pool.close, "scale": lambda: pool.scale(0), "borrow": fail_borrow}
    task = asyncio.create_task(work[operation]())
    closer = None
    try:
        await asyncio.wait_for(cleaning.wait(), timeout=1)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        closer = asyncio.create_task(pool.close())
        await asyncio.sleep(0)
        assert not closer.done()
        proceed.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
        await asyncio.wait_for(closer, timeout=1)
        assert factory.cleaned == factory.created
        assert pool.state is PoolState.CLOSED
    finally:
        proceed.set()
        await asyncio.gather(task, return_exceptions=True)
        if closer is not None:
            await closer
        await pool.close()


@pytest.mark.asyncio
async def test_inflight_cleanup_still_counts_against_creation_capacity():
    cleaning, proceed = asyncio.Event(), asyncio.Event()

    class BlockingFactory(Factory):
        async def cleanup_resource(self, resource):
            cleaning.set()
            await proceed.wait()
            await super().cleanup_resource(resource)

    factory = BlockingFactory()
    pool = make_pool(factory, min_size=0, max_size=1)
    await pool.initialize()
    await pool.scale(1)
    retiring = asyncio.create_task(pool.scale(0))
    try:
        await asyncio.wait_for(cleaning.wait(), timeout=1)
        with pytest.raises(RuntimeError, match="exhausted"):
            async with pool.acquire():
                pytest.fail("Closing resource's capacity was reused too early")
        assert len(factory.created) == 1
        proceed.set()
        await asyncio.wait_for(retiring, timeout=1)
        async with pool.acquire() as resource:
            assert resource is not factory.created[0]
    finally:
        proceed.set()
        await retiring
        await pool.close()


@pytest.mark.asyncio
async def test_cleanup_error_preserves_borrow_error_and_never_reuses_resource(caplog):
    class FailingFactory(Factory):
        async def cleanup_resource(self, resource):
            self.cleaned.append(resource)
            raise OSError("cleanup failed")

    factory = FailingFactory()
    pool = make_pool(factory, max_size=1)
    await pool.initialize()
    try:
        with pytest.raises(ValueError, match="original error"):
            async with pool.acquire() as failed:
                raise ValueError("original error")
        assert counts(pool) == (0, 0, 0)
        assert "cleanup failed" in caplog.text
        async with pool.acquire() as replacement:
            assert replacement is not failed
    finally:
        await pool.close()
    assert factory.cleaned == factory.created


@pytest.mark.parametrize("min_size,max_size", [(-1, 1), (2, 1), (0, -1)])
def test_invalid_bounds_fail_before_resource_creation(min_size, max_size):
    factory = Factory()
    with pytest.raises(ValueError, match="Pool sizes"):
        make_pool(factory, min_size=min_size, max_size=max_size)
    assert not factory.created


@pytest.mark.asyncio
async def test_close_interrupts_background_health_validation():
    validating = asyncio.Event()

    class BlockingFactory(Factory):
        async def validate_resource(self, resource):
            validating.set()
            await asyncio.Future()

    factory = BlockingFactory()
    pool = AsyncResourcePool(factory, health_check_interval=0.001)
    await pool.initialize()
    await asyncio.wait_for(validating.wait(), timeout=1)
    await asyncio.wait_for(pool.close(), timeout=1)
    assert factory.cleaned == factory.created
    assert pool.state is PoolState.CLOSED


@pytest.mark.asyncio
async def test_health_check_cleanup_does_not_replenish_a_closing_pool():
    cleaning, proceed = asyncio.Event(), asyncio.Event()

    class BlockingFactory(Factory):
        async def cleanup_resource(self, resource):
            cleaning.set()
            await proceed.wait()
            await super().cleanup_resource(resource)

    factory = BlockingFactory()
    pool = make_pool(factory, max_size=1)
    await pool.initialize()
    factory.created[0].valid = False
    health = asyncio.create_task(pool._perform_health_check())
    close = None
    try:
        await asyncio.wait_for(cleaning.wait(), timeout=1)
        assert len(factory.created) == 1
        close = asyncio.create_task(pool.close())
        async with pool._condition:
            await asyncio.wait_for(
                pool._condition.wait_for(lambda: pool.state is PoolState.DRAINING),
                timeout=1,
            )
        assert not close.done()
        proceed.set()
        await asyncio.wait_for(asyncio.gather(health, close), timeout=1)
        assert len(factory.created) == 1
        assert factory.cleaned == factory.created
        assert pool.state is PoolState.CLOSED
    finally:
        proceed.set()
        await health
        if close is not None:
            await close
        await pool.close()


@pytest.mark.asyncio
async def test_cancelled_borrow_retires_resource_without_counting_request_failure():
    using = asyncio.Event()
    factory = Factory()
    pool = make_pool(factory)
    await pool.initialize()

    async def borrow():
        async with pool.acquire():
            using.set()
            await asyncio.Future()

    task = asyncio.create_task(borrow())
    try:
        await asyncio.wait_for(using.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
        assert factory.cleaned == factory.created
        assert counts(pool) == (0, 0, 0)
        assert pool.get_stats().failed_requests == 0
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await pool.close()


@pytest.mark.asyncio
async def test_shared_http_connector_survives_retirement_until_last_session(
    monkeypatch,
):
    pytest.importorskip("aiohttp")
    from stateset_agents.core import async_pool

    monkeypatch.setattr(async_pool, "_http_pool", None)
    pool = await async_pool.get_http_pool()
    try:
        async with pool.acquire() as survivor:
            connector = survivor.connector
            assert connector.limit == 100
            with pytest.raises(RuntimeError, match="request failed"):
                async with pool.acquire() as failed:
                    assert failed.connector is connector
                    raise RuntimeError("request failed")
            assert failed.closed
            assert not survivor.closed
            assert not connector.closed
            await pool._perform_health_check()
            async with pool.acquire() as replacement:
                assert replacement.connector is connector
                assert not replacement.closed
        with pytest.raises(ValueError, match="both requests failed"):
            async with pool.acquire(), pool.acquire():
                raise ValueError("both requests failed")
        assert connector.closed
        assert counts(pool) == (0, 0, 0)
        await pool._perform_health_check()
        async with pool.acquire() as recovered:
            assert recovered.connector is not connector
            connector = recovered.connector
            assert not connector.closed
    finally:
        await pool.close()
    assert connector.closed
    # Explicit closure is terminal for that pool; the global accessor creates
    # a new owner rather than reviving its closed connector or lifecycle.
    replacement_pool = await async_pool.get_http_pool()
    try:
        assert replacement_pool is not pool
        async with replacement_pool.acquire() as session:
            assert session.connector is not connector
            assert not session.closed
    finally:
        await replacement_pool.close()


@pytest.mark.asyncio
async def test_global_http_callers_wait_for_initialization(monkeypatch):
    pytest.importorskip("aiohttp")
    from stateset_agents.core import async_pool

    created, proceed, joined = (asyncio.Event() for _ in range(3))

    class BlockingFactory(Factory):
        async def create_resource(self, **kwargs):
            resource = await super().create_resource(**kwargs)
            created.set()
            await proceed.wait()
            return resource

    factory = BlockingFactory()
    monkeypatch.setattr(async_pool, "_http_pool", None)
    monkeypatch.setattr(async_pool, "_SharedHTTPSessionFactory", lambda **kw: factory)
    first = asyncio.create_task(async_pool.get_http_pool())
    second = None
    try:
        await asyncio.wait_for(created.wait(), timeout=1)

        async def join():
            joined.set()
            return await async_pool.get_http_pool()

        second = asyncio.create_task(join())
        await asyncio.wait_for(joined.wait(), timeout=1)
        assert not second.done()
        proceed.set()
        first_pool, second_pool = await asyncio.wait_for(
            asyncio.gather(first, second), timeout=1
        )
        assert first_pool is second_pool
        assert first_pool.state is PoolState.READY
        assert len(factory.created) == 2
    finally:
        proceed.set()
        await asyncio.gather(
            first, *([second] if second is not None else []), return_exceptions=True
        )
        if async_pool._http_pool is not None:
            await async_pool._http_pool.close()


@pytest.mark.asyncio
async def test_global_http_creation_failure_can_be_retried_with_a_new_pool(monkeypatch):
    pytest.importorskip("aiohttp")
    from stateset_agents.core import async_pool

    class FailingFactory(Factory):
        async def create_resource(self, **kwargs):
            raise OSError("creation failed")

    monkeypatch.setattr(async_pool, "_http_pool", None)
    monkeypatch.setattr(
        async_pool, "_SharedHTTPSessionFactory", lambda **kw: FailingFactory()
    )
    with pytest.raises(ResourceException):
        await async_pool.get_http_pool()
    assert async_pool._http_pool is None
    factory = Factory()
    monkeypatch.setattr(async_pool, "_SharedHTTPSessionFactory", lambda **kw: factory)
    pool = await async_pool.get_http_pool()
    try:
        assert pool.state is PoolState.READY
        assert counts(pool) == (2, 0, 2)
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_contending_borrowers_never_share_or_reuse_retired_resources():
    factory = Factory()
    pool = make_pool(factory, min_size=3, max_size=3)
    await pool.initialize()
    leased: set[Resource] = set()

    async def request(index):
        try:
            async with pool.acquire() as resource:
                assert resource not in leased
                assert not resource.closed
                leased.add(resource)
                try:
                    await asyncio.sleep(0)
                    assert not resource.closed
                    assert sum(not item.closed for item in factory.created) <= 3
                    if index % 3 == 0:
                        raise ValueError("request failed")
                finally:
                    leased.remove(resource)
        except ValueError as exc:
            assert str(exc) == "request failed"
            return "failed"
        except RuntimeError as exc:
            # Exhaustion is legitimate when deadlines coincide while all
            # capacity belongs to other borrowers or pending cleanup.
            assert "exhausted" in str(exc)
            return "exhausted"
        return "success"

    try:
        outcomes = await asyncio.wait_for(
            asyncio.gather(*(request(index) for index in range(30))), timeout=2
        )
        assert not leased
        stats = pool.get_stats()
        assert stats.total_requests == 30 - outcomes.count("exhausted")
        assert stats.successful_requests == outcomes.count("success")
        assert stats.failed_requests == outcomes.count("failed") + outcomes.count(
            "exhausted"
        )
        assert stats.peak_connections == 3
        assert counts(pool)[1] == 0
    finally:
        await pool.close()
    assert len(factory.cleaned) == len(set(factory.cleaned)) == len(factory.created)
    assert counts(pool) == (0, 0, 0)
