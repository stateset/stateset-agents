import asyncio

import pytest

from utils.monitoring import HealthCheck


@pytest.mark.asyncio
async def test_health_check_supports_async_callable():
    async def _check() -> bool:
        await asyncio.sleep(0)
        return True

    result = await HealthCheck(name="async", check_func=_check).run()
    assert result["status"] == "healthy"


@pytest.mark.asyncio
async def test_health_check_supports_sync_callable():
    result = await HealthCheck(name="sync", check_func=lambda: True).run()
    assert result["status"] == "healthy"


@pytest.mark.asyncio
async def test_health_check_handles_exceptions():
    def boom() -> bool:
        raise RuntimeError("fail")

    result = await HealthCheck(name="boom", check_func=boom).run()
    assert result["status"] == "error"
    assert "fail" in result.get("error", "")
