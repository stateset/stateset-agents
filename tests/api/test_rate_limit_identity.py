"""Tests for identity-keyed rate limiting and the optional Redis backend."""

import asyncio
import hashlib
import time

import httpx
import pytest
from fastapi import FastAPI

from stateset_agents.api import config as api_config
from stateset_agents.api.config import APIConfig, RateLimitConfig, SecurityConfig
from stateset_agents.api.grpo.rate_limiter import UnifiedRateLimiter
from stateset_agents.api.middleware import (
    RateLimitMiddleware,
    RedisSlidingWindowLimiter,
)


@pytest.fixture
def preserve_api_config():
    prev = api_config._config
    yield
    api_config._config = prev


def _build_app(
    rate_limit: RateLimitConfig, security: SecurityConfig | None = None
) -> FastAPI:
    config = APIConfig(rate_limit=rate_limit, security=security or SecurityConfig())
    api_config._config = config

    app = FastAPI()

    @app.get("/probe")
    async def probe():
        return {"ok": True}

    app.add_middleware(RateLimitMiddleware)
    return app


def _client(app: FastAPI, client_ip: str = "1.2.3.4") -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app, client=(client_ip, 12345))
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


def _expected_key(api_key: str) -> str:
    return f"key:{hashlib.sha256(api_key.encode('utf-8')).hexdigest()[:16]}"


@pytest.mark.asyncio
async def test_different_api_keys_do_not_share_a_bucket(preserve_api_config):
    app = _build_app(
        RateLimitConfig(requests_per_minute=1, enabled=True),
        security=SecurityConfig(
            api_keys={"key-a": ["user"], "key-b": ["user"]}, require_auth=False
        ),
    )

    async with _client(app) as client:
        r1 = await client.get("/probe", headers={"X-API-Key": "key-a"})
        r2 = await client.get("/probe", headers={"X-API-Key": "key-b"})

    assert r1.status_code == 200
    assert r2.status_code == 200


@pytest.mark.asyncio
async def test_same_api_key_from_two_ips_shares_one_bucket(preserve_api_config):
    app = _build_app(
        RateLimitConfig(requests_per_minute=1, enabled=True),
        security=SecurityConfig(api_keys={"shared-key": ["user"]}, require_auth=False),
    )

    async with _client(app, client_ip="1.1.1.1") as client:
        r1 = await client.get("/probe", headers={"X-API-Key": "shared-key"})

    async with _client(app, client_ip="2.2.2.2") as client:
        r2 = await client.get("/probe", headers={"X-API-Key": "shared-key"})

    assert r1.status_code == 200
    assert r2.status_code == 429


@pytest.mark.asyncio
async def test_two_ips_without_credentials_do_not_share_a_bucket(preserve_api_config):
    app = _build_app(RateLimitConfig(requests_per_minute=1, enabled=True))

    async with _client(app, client_ip="1.1.1.1") as client:
        r1 = await client.get("/probe")

    async with _client(app, client_ip="2.2.2.2") as client:
        r2 = await client.get("/probe")

    assert r1.status_code == 200
    assert r2.status_code == 200


@pytest.mark.asyncio
async def test_xff_ignored_when_trust_flag_off(preserve_api_config):
    app = _build_app(
        RateLimitConfig(requests_per_minute=1, enabled=True, trust_proxy_headers=False)
    )

    async with _client(app, client_ip="9.9.9.9") as client:
        r1 = await client.get("/probe", headers={"X-Forwarded-For": "5.5.5.5"})
        r2 = await client.get("/probe", headers={"X-Forwarded-For": "6.6.6.6"})

    # Same real client IP -> same bucket regardless of XFF -> second is limited.
    assert r1.status_code == 200
    assert r2.status_code == 429


@pytest.mark.asyncio
async def test_xff_honored_when_trust_flag_on(preserve_api_config):
    app = _build_app(
        RateLimitConfig(requests_per_minute=1, enabled=True, trust_proxy_headers=True)
    )

    async with _client(app, client_ip="9.9.9.9") as client:
        r1 = await client.get("/probe", headers={"X-Forwarded-For": "5.5.5.5"})
        r2 = await client.get("/probe", headers={"X-Forwarded-For": "6.6.6.6"})

    # Different first-hop XFF -> different buckets -> both allowed.
    assert r1.status_code == 200
    assert r2.status_code == 200


@pytest.mark.asyncio
async def test_redis_backend_falls_back_to_memory_when_unavailable(
    preserve_api_config,
):
    app = _build_app(
        RateLimitConfig(
            requests_per_minute=5,
            enabled=True,
            backend="redis",
            # Numeric loopback avoids platform-dependent DNS resolver hangs
            # while still exercising a guaranteed unavailable Redis endpoint.
            redis_url="redis://127.0.0.1:1",
        )
    )

    async with _client(app) as client:
        response = await client.get("/probe")

    # Must not crash; falls back to the in-memory limiter.
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_redis_limiter_bounds_pipeline_and_releases_failed_client():
    limiter = RedisSlidingWindowLimiter(
        "redis://placeholder:6379", operation_timeout_seconds=0.01
    )

    class SlowPipeline:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def incr(self, key):
            return None

        def expire(self, key, seconds, nx=False):
            return None

        async def execute(self):
            await asyncio.sleep(10)

    class FakeClient:
        closed = False

        def pipeline(self, transaction=True):
            return SlowPipeline()

        async def aclose(self):
            self.closed = True

    client = FakeClient()
    limiter._client = client

    with pytest.raises(asyncio.TimeoutError):
        await limiter.is_allowed("key", 1)

    await limiter.close()
    assert client.closed is True


def test_redis_limiter_uses_lazy_import(preserve_api_config):
    limiter = RedisSlidingWindowLimiter("redis://localhost:6379", window_seconds=60)
    assert limiter._client is None


@pytest.mark.asyncio
async def test_garbage_api_key_falls_back_to_ip_bucket(preserve_api_config):
    """Unvalidated credentials must not mint their own bucket.

    Without validation, presenting a fresh garbage `X-API-Key` on every
    request would give each request its own unlimited bucket, bypassing
    IP-based limits entirely. Garbage keys must share the IP bucket.
    """
    app = _build_app(
        RateLimitConfig(requests_per_minute=1, enabled=True),
        security=SecurityConfig(api_keys={}, require_auth=False),
    )

    async with _client(app, client_ip="7.7.7.7") as client:
        r1 = await client.get("/probe", headers={"X-API-Key": "garbage-one"})
        r2 = await client.get("/probe", headers={"X-API-Key": "garbage-two"})

    assert r1.status_code == 200
    # Second garbage key shares the same IP bucket as the first -> limited.
    assert r2.status_code == 429


@pytest.mark.asyncio
async def test_valid_api_key_still_gets_credential_bucket_alongside_garbage(
    preserve_api_config,
):
    """A validated key gets its own bucket even when garbage keys share IP."""
    app = _build_app(
        RateLimitConfig(requests_per_minute=1, enabled=True),
        security=SecurityConfig(api_keys={"real-key": ["user"]}, require_auth=False),
    )

    async with _client(app, client_ip="8.8.8.8") as client:
        garbage = await client.get("/probe", headers={"X-API-Key": "not-real"})
        real = await client.get("/probe", headers={"X-API-Key": "real-key"})

    assert garbage.status_code == 200
    # Different bucket (validated credential) -> not limited by the IP hit.
    assert real.status_code == 200


def test_bucket_dict_capped_under_unique_key_flood():
    """The in-memory limiter must not grow unboundedly between cleanups."""
    limiter = UnifiedRateLimiter(window_seconds=60, max_buckets=100)

    for i in range(20_000):
        limiter.is_allowed(f"flood-key-{i}", limit=5)

    assert len(limiter.windows) <= 100


@pytest.mark.asyncio
async def test_redis_limiter_retries_after_cooldown(preserve_api_config, monkeypatch):
    """After a Redis failure, the middleware retries Redis after cooldown
    instead of permanently self-disabling."""
    app_config = APIConfig(
        rate_limit=RateLimitConfig(
            requests_per_minute=5,
            enabled=True,
            backend="redis",
            redis_url="redis://placeholder:6379",
        )
    )
    api_config._config = app_config

    # Build the middleware directly so we can control/inspect its state,
    # then monkeypatch its Redis limiter to fail then succeed.
    dummy_app = FastAPI()
    middleware = RateLimitMiddleware(dummy_app)
    assert middleware._redis_limiter is not None

    call_state = {"calls": 0}

    async def flaky_is_allowed(key, limit):
        call_state["calls"] += 1
        raise RuntimeError("redis unreachable")

    monkeypatch.setattr(middleware._redis_limiter, "is_allowed", flaky_is_allowed)

    fake_time = {"now": 1000.0}
    monkeypatch.setattr(time, "monotonic", lambda: fake_time["now"])

    # First failure -> disabled until now + cooldown, in-memory used.
    allowed, _ = await middleware._check_rate_limit("some-key", 5)
    assert allowed is True
    assert call_state["calls"] == 1
    assert middleware._redis_currently_disabled is True

    # Still within cooldown -> Redis not retried.
    fake_time["now"] += 10
    allowed, _ = await middleware._check_rate_limit("some-key", 5)
    assert call_state["calls"] == 1  # unchanged: Redis skipped

    # Past cooldown -> Redis retried.
    async def recovered_is_allowed(key, limit):
        call_state["calls"] += 1
        return True, 4

    fake_time["now"] += 60
    monkeypatch.setattr(middleware._redis_limiter, "is_allowed", recovered_is_allowed)
    allowed, remaining = await middleware._check_rate_limit("some-key", 5)
    assert call_state["calls"] == 2
    assert allowed is True
    assert remaining == 4
    assert middleware._redis_currently_disabled is False
