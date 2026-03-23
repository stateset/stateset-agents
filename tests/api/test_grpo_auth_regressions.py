"""Regression tests for GRPO service authentication identity handling."""

import hashlib

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
import pytest

from stateset_agents.api.grpo.config import reset_config
from stateset_agents.api.grpo.service import create_app, verify_request


@pytest.fixture(autouse=True)
def _grpo_api_key_auth(monkeypatch):
    """
    Configure GRPO auth for GRPO API verification tests.
    """
    monkeypatch.setenv("GRPO_API_KEYS", "unit-key:user,admin-key:admin")
    monkeypatch.setenv("GRPO_ALLOW_ANONYMOUS", "false")
    monkeypatch.setenv("GRPO_RATE_LIMIT_PER_MIN", "1000")
    reset_config()
    try:
        yield
    finally:
        reset_config()


def _build_probe_app() -> FastAPI:
    app = FastAPI()

    @app.get("/probe")
    async def probe(ctx=Depends(verify_request)):
        return {"user_id": ctx.user_id, "roles": ctx.roles}

    return app


def _expected_api_user_id(api_key: str) -> str:
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
    return f"api_key:{digest}"


def test_verify_request_ignores_spoofed_user_id_header():
    app = _build_probe_app()
    client = TestClient(app)

    response = client.get(
        "/probe",
        headers={
            "x-api-key": "unit-key",
            "x-user-id": "intruder-user",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] != "intruder-user"
    assert payload["user_id"] == _expected_api_user_id("unit-key")


def test_verify_request_uses_stable_hashed_identity_for_api_key():
    app = _build_probe_app()
    client = TestClient(app)

    response_a = client.get("/probe", headers={"x-api-key": "unit-key"})
    response_b = client.get("/probe", headers={"x-api-key": "unit-key"})

    assert response_a.status_code == 200
    assert response_b.status_code == 200
    assert response_a.json()["user_id"] == response_b.json()["user_id"] == _expected_api_user_id(
        "unit-key"
    )
    assert response_a.json()["roles"] == ["user"]


def test_verify_request_rejects_unknown_api_key():
    app = _build_probe_app()
    client = TestClient(app)

    response = client.get("/probe", headers={"x-api-key": "bad-key"})

    assert response.status_code == 401


def _build_grpo_app() -> FastAPI:
    return create_app()


def test_api_endpoints_derive_identity_from_api_key_for_owner_checks():
    app = _build_grpo_app()
    client = TestClient(app)

    start_response = client.post(
        "/api/train",
        headers={"x-api-key": "unit-key"},
        json={"prompts": ["stabilize identity"], "strategy": "computational"},
    )
    assert start_response.status_code == 200
    job_id = start_response.json()["job_id"]

    status_as_owner = client.get(
        f"/api/status/{job_id}",
        headers={"x-api-key": "unit-key", "x-user-id": "owner-user"},
    )
    assert status_as_owner.status_code == 200

    status_as_spoofed = client.get(
        f"/api/status/{job_id}",
        headers={"x-api-key": "unit-key", "x-user-id": "intruder-user"},
    )
    assert status_as_spoofed.status_code == 200


def test_api_job_lookup_isolated_by_api_key_identity():
    app = _build_grpo_app()
    client = TestClient(app)

    start_response = client.post(
        "/api/train",
        headers={"x-api-key": "unit-key"},
        json={"prompts": ["another train"], "strategy": "computational"},
    )
    assert start_response.status_code == 200
    job_id = start_response.json()["job_id"]

    # Same job should not be accessible with a different API key.
    blocked = client.get(f"/api/status/{job_id}", headers={"x-api-key": "admin-key"})
    assert blocked.status_code == 404


def test_verify_request_ignores_spoofed_user_id_in_anonymous_mode(monkeypatch):
    app = _build_probe_app()
    client = TestClient(app)
    monkeypatch.setenv("GRPO_API_KEYS", "")
    monkeypatch.setenv("GRPO_ALLOW_ANONYMOUS", "true")
    monkeypatch.setenv("GRPO_RATE_LIMIT_PER_MIN", "1000")
    reset_config()

    response = client.get(
        "/probe",
        headers={"x-user-id": "intruder-anon"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["roles"] == ["anonymous"]
    assert payload["user_id"] != "intruder-anon"
    assert payload["user_id"].startswith("anonymous:")

    repeat_response = client.get("/probe", headers={"x-user-id": "intruder-anon-two"})
    assert repeat_response.status_code == 200
    assert repeat_response.json()["user_id"] == payload["user_id"]
