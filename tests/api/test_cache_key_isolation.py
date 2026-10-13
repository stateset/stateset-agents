"""Regression tests for cache keys that previously mixed unrelated results."""

import pytest

from stateset_agents.api.cache import cached as local_cached
from stateset_agents.api.cache import cached_sync
from stateset_agents.api.cache import get_cache as get_local_cache
from stateset_agents.api.distributed_cache import CacheConfig
from stateset_agents.api.distributed_cache import cached as distributed_cached
from stateset_agents.api.distributed_cache import close_cache, init_cache


@pytest.mark.asyncio
async def test_local_async_functions_with_same_name_do_not_share_values():
    """Methods with the same short name retain their own cached results."""
    cache = get_local_cache()
    cache.clear()

    class First:
        @staticmethod
        @local_cached(ttl_seconds=60)
        async def value():
            return "first"

    class Second:
        @staticmethod
        @local_cached(ttl_seconds=60)
        async def value():
            return "second"

    try:
        assert await First.value() == "first"
        assert await Second.value() == "second"
    finally:
        cache.clear()


def test_local_sync_arguments_with_same_short_digest_do_not_share_values():
    """A known collision in the old 32-bit digest remains distinct."""
    cache = get_local_cache()
    cache.clear()

    @cached_sync(ttl_seconds=60)
    def value(number: int) -> int:
        return number

    try:
        assert value(123657) == 123657
        assert value(162286) == 162286
    finally:
        cache.clear()


@pytest.mark.asyncio
async def test_distributed_cache_isolates_functions_and_colliding_arguments():
    """The distributed decorator isolates both functions and arguments."""
    await init_cache(CacheConfig())

    class First:
        @staticmethod
        @distributed_cached(ttl_seconds=60)
        async def value(number: int) -> str:
            return f"first:{number}"

    class Second:
        @staticmethod
        @distributed_cached(ttl_seconds=60)
        async def value(number: int) -> str:
            return f"second:{number}"

    try:
        assert await First.value(123657) == "first:123657"
        assert await First.value(162286) == "first:162286"
        assert await Second.value(123657) == "second:123657"
    finally:
        await close_cache()
