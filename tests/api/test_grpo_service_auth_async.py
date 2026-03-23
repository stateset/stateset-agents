"""Async transport coverage for GRPO service auth and ownership routing."""

import hashlib

import pytest
from fastapi import Depends, FastAPI
from httpx import ASGITransport, AsyncClient

from stateset_agents.api.grpo.config import reset_config
from stateset_agents.api.grpo.service import create_app, verify_request


@pytest.fixture(autouse=True)
def _grpo_api_key_auth(monkeypatch):
    monkeypatch.setenv("GRPO_API_KEYS", "unit-key:user,admin-key:admin")
    monkeypatch.setenv("GRPO_ALLOW_ANONYMOUS", "false")
    monkeypatch.setenv("GRPO_RATE_LIMIT_PER_MIN", "1000")
    reset_config()
    try:
        yield
    finally:
        reset_config()


def _expected_api_user_id(api_key: str) -> str:
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
    return f"api_key:{digest}"


@pytest.mark.asyncio
async def test_verify_request_ignores_spoofed_user_id_with_async_transport():
    app = FastAPI()

    @app.get("/probe")
    async def probe(ctx=Depends(verify_request)):
        return {"user_id": ctx.user_id, "roles": ctx.roles}

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.get(
            "/probe",
            headers={
                "x-api-key": "unit-key",
                "x-user-id": "intruder-user",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == _expected_api_user_id("unit-key")
    assert payload["roles"] == ["user"]


@pytest.mark.asyncio
async def test_grpo_service_job_access_is_scoped_to_api_key_identity():
    app = create_app()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        start_response = await client.post(
            "/api/train",
            headers={"x-api-key": "unit-key"},
            json={"prompts": ["stabilize identity"], "strategy": "computational"},
        )

        assert start_response.status_code == 200
        job_id = start_response.json()["job_id"]

        owner_status = await client.get(
            f"/api/status/{job_id}",
            headers={"x-api-key": "unit-key", "x-user-id": "spoofed-owner"},
        )
        intruder_status = await client.get(
            f"/api/status/{job_id}",
            headers={"x-api-key": "admin-key"},
        )

    assert owner_status.status_code == 200
    assert intruder_status.status_code == 404
