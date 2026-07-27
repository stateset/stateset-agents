"""Tests for identity-keyed rate limiting and the optional Redis backend."""

import hashlib

import httpx
import pytest
from fastapi import FastAPI

from stateset_agents.api import config as api_config
from stateset_agents.api.config import APIConfig, RateLimitConfig
from stateset_agents.api.middleware import (
    RateLimitMiddleware,
    RedisSlidingWindowLimiter,
)


@pytest.fixture
def preserve_api_config():
    prev = api_config._config
    yield
    api_config._config = prev


def _build_app(rate_limit: RateLimitConfig) -> FastAPI:
    config = APIConfig(rate_limit=rate_limit)
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
    app = _build_app(RateLimitConfig(requests_per_minute=1, enabled=True))

    async with _client(app) as client:
        r1 = await client.get("/probe", headers={"X-API-Key": "key-a"})
        r2 = await client.get("/probe", headers={"X-API-Key": "key-b"})

    assert r1.status_code == 200
    assert r2.status_code == 200


@pytest.mark.asyncio
async def test_same_api_key_from_two_ips_shares_one_bucket(preserve_api_config):
    app = _build_app(RateLimitConfig(requests_per_minute=1, enabled=True))

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
        RateLimitConfig(
            requests_per_minute=1, enabled=True, trust_proxy_headers=False
        )
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
            redis_url="redis://this-host-does-not-exist.invalid:6379",
        )
    )

    async with _client(app) as client:
        response = await client.get("/probe")

    # Must not crash; falls back to the in-memory limiter.
    assert response.status_code == 200


def test_redis_limiter_uses_lazy_import(preserve_api_config):
    limiter = RedisSlidingWindowLimiter("redis://localhost:6379", window_seconds=60)
    assert limiter._client is None
