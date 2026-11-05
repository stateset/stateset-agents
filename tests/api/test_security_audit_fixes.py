"""Regression tests for the v0.64.0 API security review.

Each test reproduces a verified finding (with ``API_REQUIRE_AUTH=true``):

1. training overrides could point ``output_dir`` / ``resume_from_checkpoint``
   anywhere on the server's filesystem;
2. ``/agents`` had no ownership, so any user could read and delete any agent;
3. any user could make the server load an arbitrary Hugging Face repo or
   local path (outbound downloads, disk/memory exhaustion, path probing);
4. the training lab had no tenant isolation and its clone/patch endpoints
   skipped config validation;
5. a valid key cleared brute-force lockouts for every credential from the IP;
7. the lab WebSocket accepted credentials in the query string, and the
   Prometheus endpoint was unauthenticated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from stateset_agents.api import config as api_config

ALICE = "alice-key-aaaaaaaaaaaa"
BOB = "bob-key-bbbbbbbbbbbbbb"
TRAINER = "trainer-key-cccccccccc"
TRAINER2 = "trainer-key-dddddddddd"
ADMIN = "admin-key-eeeeeeeeeeee"
KEYS = f"{ALICE}:user,{BOB}:user,{TRAINER}:trainer,{TRAINER2}:trainer,{ADMIN}:admin"


def _h(key: str) -> dict[str, str]:
    return {"X-API-Key": key}


@pytest.fixture
def secure_env(monkeypatch, tmp_path):
    """Authenticated API in development mode with a temp training root."""
    prev = api_config._config
    monkeypatch.setenv("API_REQUIRE_AUTH", "true")
    monkeypatch.setenv("API_KEYS", KEYS)
    monkeypatch.setenv("API_JWT_SECRET", "x" * 40)
    monkeypatch.setenv("API_ENVIRONMENT", "development")
    monkeypatch.setenv("API_ENABLE_TRAINING_LAB", "true")
    monkeypatch.setenv("API_RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("API_TRAINING_OUTPUT_ROOT", str(tmp_path / "training-root"))
    for name in (
        "API_ALLOW_EXTERNAL_MODELS",
        "API_MODEL_ALLOWLIST",
        "API_PUBLIC_METRICS",
        "API_LAB_WS_QUERY_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    api_config.reload_config()

    from stateset_agents.api.routers import training_lab
    from stateset_agents.api.security import get_api_security_monitor

    tracker = get_api_security_monitor().auth_tracker
    tracker.failures.clear()
    tracker.lockouts.clear()
    for state in (
        training_lab._experiments,
        training_lab._episodes,
        training_lab._logs,
        getattr(training_lab, "_experiment_owners", {}),
        training_lab._playground_sessions,
        getattr(training_lab, "_playground_owners", {}),
    ):
        state.clear()
    yield tmp_path / "training-root"
    tracker.failures.clear()
    tracker.lockouts.clear()
    api_config._config = prev


def _reload() -> None:
    api_config.reload_config()


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    )


def _app():
    from stateset_agents.api.main import create_app

    return create_app()


@pytest.fixture
def captured_train(monkeypatch):
    """Replace the trainer so a job records what it would run with."""
    calls: list[dict[str, Any]] = []

    async def fake_train(**kwargs: Any) -> Any:
        calls.append(kwargs)
        # Stand in for the trainer's artifact publication, so the API can
        # check real directory existence as well as resume ownership.
        output = Path(kwargs["config_overrides"]["output_dir"])
        (output / "checkpoint-0").mkdir(parents=True)
        return kwargs.get("agent")

    import importlib

    # ``stateset_agents.training.train`` the attribute is the function; the
    # service imports it from the module.
    train_module = importlib.import_module("stateset_agents.training.train")
    monkeypatch.setattr(train_module, "train", fake_train)
    return calls


def _training_body(**extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "agent_config": {"model_name": "stub://x"},
        "environment_scenarios": [{"user_responses": ["hi"]}],
        "reward_config": {},
        "num_episodes": 1,
    }
    body.update(extra)
    return body


async def _wait_for_job(client, training_id: str, key: str) -> dict[str, Any]:
    import asyncio

    for _ in range(200):
        status = (await client.get(f"/training/{training_id}", headers=_h(key))).json()
        if status.get("status") not in ("running", "cancelling"):
            return status
        await asyncio.sleep(0.01)
    raise AssertionError("training job did not finish")


# --- 1. training overrides --------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "output_dir",
        "overwrite_output_dir",
        "report_to",
        "wandb_project",
        "resume_from_checkpoint",
    ],
)
async def test_training_overrides_reject_server_controlled_keys(secure_env, key):
    async with _client(_app()) as client:
        response = await client.post(
            "/training",
            json=_training_body(training_config_overrides={key: "/tmp/escape"}),
            headers=_h(TRAINER),
        )
    assert response.status_code == 422
    assert key in response.text


async def test_training_output_dir_is_server_controlled(secure_env, captured_train):
    async with _client(_app()) as client:
        response = await client.post(
            "/training",
            json=_training_body(training_config_overrides={"learning_rate": 1e-5}),
            headers=_h(TRAINER),
        )
        assert response.status_code == 202, response.text
        training_id = response.json()["training_id"]
        await _wait_for_job(client, training_id, TRAINER)

    overrides = captured_train[-1]["config_overrides"]
    assert overrides["learning_rate"] == 1e-5
    output_dir = Path(overrides["output_dir"]).resolve()
    assert output_dir == (secure_env / training_id).resolve()


async def test_resume_outside_training_root_is_rejected(secure_env, captured_train):
    async with _client(_app()) as client:
        response = await client.post(
            "/training",
            json=_training_body(resume_from_checkpoint="/etc"),
            headers=_h(TRAINER),
        )
        traversal = await client.post(
            "/training",
            json=_training_body(resume_from_checkpoint="../../etc"),
            headers=_h(TRAINER),
        )
    assert response.status_code == 400
    assert traversal.status_code == 400
    assert captured_train == []


async def test_resume_by_job_id_and_under_root(secure_env, captured_train):
    async with _client(_app()) as client:
        first = await client.post(
            "/training", json=_training_body(), headers=_h(TRAINER)
        )
        first_id = first.json()["training_id"]
        await _wait_for_job(client, first_id, TRAINER)

        by_id = await client.post(
            "/training",
            json=_training_body(resume_from_checkpoint=first_id),
            headers=_h(TRAINER),
        )
        assert by_id.status_code == 202, by_id.text
        await _wait_for_job(client, by_id.json()["training_id"], TRAINER)

        other_user = await client.post(
            "/training",
            json=_training_body(resume_from_checkpoint=first_id),
            headers=_h(TRAINER2),
        )

        nested = f"{first_id}/checkpoint-0"
        by_path = await client.post(
            "/training",
            json=_training_body(resume_from_checkpoint=nested),
            headers=_h(TRAINER),
        )
        assert by_path.status_code == 202, by_path.text
        await _wait_for_job(client, by_path.json()["training_id"], TRAINER)

    # Another trainer cannot resume from a job it cannot see.
    assert other_user.status_code == 400
    resumed = [Path(c["resume_from_checkpoint"]) for c in captured_train[1:]]
    assert resumed[0] == (secure_env / first_id).resolve()
    assert resumed[1] == (secure_env / nested).resolve()


# --- 2. agent ownership -----------------------------------------------------


async def test_agents_are_private_to_their_owner(secure_env):
    async with _client(_app()) as client:
        created = await client.post(
            "/agents",
            json={"model_name": "stub://alice", "system_prompt": "alice secret"},
            headers=_h(ALICE),
        )
        assert created.status_code == 201, created.text
        agent_id = created.json()["agent_id"]

        listed = (await client.get("/agents", headers=_h(BOB))).json()
        assert all(item["agent_id"] != agent_id for item in listed["items"])
        assert "alice secret" not in str(listed)

        assert (
            await client.get(f"/agents/{agent_id}", headers=_h(BOB))
        ).status_code == 404
        assert (
            await client.delete(f"/agents/{agent_id}", headers=_h(BOB))
        ).status_code == 404
        assert (
            await client.get(f"/agents/{agent_id}", headers=_h(ALICE))
        ).status_code == 200
        own = (await client.get("/agents", headers=_h(ALICE))).json()
        assert any(item["agent_id"] == agent_id for item in own["items"])

        admin = (await client.get("/agents", headers=_h(ADMIN))).json()
        assert any(item["agent_id"] == agent_id for item in admin["items"])
        assert (
            await client.delete(f"/agents/{agent_id}", headers=_h(ADMIN))
        ).status_code == 204


# --- 3. external model loading ----------------------------------------------


async def test_production_rejects_unapproved_models(secure_env, monkeypatch):
    monkeypatch.setenv("API_ENVIRONMENT", "production")
    monkeypatch.setenv("API_CORS_ORIGINS", "https://app.example.com")
    _reload()
    async with _client(_app()) as client:
        agent = await client.post(
            "/agents", json={"model_name": "attacker/any-repo"}, headers=_h(ALICE)
        )
        training = await client.post(
            "/training",
            json=_training_body(agent_config={"model_name": "attacker/any-repo"}),
            headers=_h(TRAINER),
        )
    assert agent.status_code == 403
    assert training.status_code == 403


def _policy(name: str, roles: list[str]) -> bool:
    from stateset_agents.api.model_policy import client_model_uses_stub

    return client_model_uses_stub(name, roles=roles)


def test_model_policy(secure_env, monkeypatch):
    from fastapi import HTTPException

    # Development without opt-in keeps the stub backend (unchanged behaviour).
    assert _policy("attacker/any-repo", ["user"]) is True
    assert _policy("stub://x", ["user"]) is True

    monkeypatch.setenv("API_ALLOW_EXTERNAL_MODELS", "true")
    monkeypatch.setenv("API_MODEL_ALLOWLIST", "Qwen/*,gpt2")
    _reload()
    assert _policy("Qwen/Qwen2.5-0.5B-Instruct", ["trainer"]) is False
    assert _policy("gpt2", ["admin"]) is False
    for name, roles in (
        ("Qwen/Qwen2.5-0.5B-Instruct", ["user"]),  # role too low
        ("attacker/any-repo", ["trainer"]),  # not allow-listed
    ):
        with pytest.raises(HTTPException) as exc:
            _policy(name, roles)
        assert exc.value.status_code == 403

    # Without an allow-list, local filesystem paths are still refused.
    monkeypatch.delenv("API_MODEL_ALLOWLIST")
    _reload()
    assert _policy("attacker/any-repo", ["trainer"]) is False
    for path in ("/etc", "../models/x", "~/secret", "./local"):
        with pytest.raises(HTTPException):
            _policy(path, ["trainer"])


# --- 4. training lab --------------------------------------------------------


async def test_training_lab_isolates_experiments(secure_env):
    async with _client(_app()) as client:
        created = await client.post(
            "/api/lab/experiments", json={"name": "alice-exp"}, headers=_h(ALICE)
        )
        exp_id = created.json()["id"]

        listed = (await client.get("/api/lab/experiments", headers=_h(BOB))).json()
        assert [e["id"] for e in listed] == []
        board = (await client.get("/api/lab/leaderboard", headers=_h(BOB))).json()
        assert board == []
        compare = await client.post("/api/lab/compare", json=[exp_id], headers=_h(BOB))
        assert compare.json()["experiments"] == []

        for method, path, kwargs in (
            ("get", f"/api/lab/experiments/{exp_id}", {}),
            ("post", f"/api/lab/experiments/{exp_id}/start", {}),
            ("post", f"/api/lab/experiments/{exp_id}/stop", {}),
            ("post", f"/api/lab/experiments/{exp_id}/clone", {"json": {}}),
            ("patch", f"/api/lab/experiments/{exp_id}/config", {"json": {}}),
            ("get", f"/api/lab/experiments/{exp_id}/logs", {}),
            ("get", f"/api/lab/experiments/{exp_id}/metrics", {}),
            ("get", f"/api/lab/experiments/{exp_id}/export", {}),
            ("delete", f"/api/lab/experiments/{exp_id}", {}),
        ):
            response = await getattr(client, method)(path, headers=_h(BOB), **kwargs)
            assert response.status_code == 404, (method, path, response.text)

        assert (
            await client.get(f"/api/lab/experiments/{exp_id}", headers=_h(ALICE))
        ).status_code == 200
        assert (
            await client.get(f"/api/lab/experiments/{exp_id}", headers=_h(ADMIN))
        ).status_code == 200


async def test_training_lab_playground_sessions_are_private(secure_env):
    async with _client(_app()) as client:
        started = await client.post(
            "/api/lab/playground/chat",
            json={"message": "hello from alice"},
            headers=_h(ALICE),
        )
        assert started.status_code == 200
        session_id = started.json()["session_id"]

        # Another user cannot read, append to, or clear the session.
        assert (
            await client.get(f"/api/lab/playground/{session_id}", headers=_h(BOB))
        ).status_code == 404
        hijack = await client.post(
            "/api/lab/playground/chat",
            json={"session_id": session_id, "message": "bob writes here"},
            headers=_h(BOB),
        )
        assert hijack.status_code == 404
        assert (
            await client.delete(f"/api/lab/playground/{session_id}", headers=_h(BOB))
        ).status_code == 404

        own = await client.get(f"/api/lab/playground/{session_id}", headers=_h(ALICE))
        assert own.status_code == 200
        contents = [m["content"] for m in own.json()["history"]]
        assert "bob writes here" not in contents
        assert (
            await client.get(f"/api/lab/playground/{session_id}", headers=_h(ADMIN))
        ).status_code == 200
        assert (
            await client.delete(f"/api/lab/playground/{session_id}", headers=_h(ALICE))
        ).status_code == 200


async def test_training_lab_validates_clone_and_patch(secure_env):
    async with _client(_app()) as client:
        exp_id = (
            await client.post(
                "/api/lab/experiments", json={"name": "x"}, headers=_h(ALICE)
            )
        ).json()["id"]
        clone = await client.post(
            f"/api/lab/experiments/{exp_id}/clone",
            json={"training": {"num_episodes": 10**12}},
            headers=_h(ALICE),
        )
        patch = await client.patch(
            f"/api/lab/experiments/{exp_id}/config",
            json={"num_episodes": 10**12},
            headers=_h(ALICE),
        )
        negative_lr = await client.patch(
            f"/api/lab/experiments/{exp_id}/config",
            json={"learning_rate": -1},
            headers=_h(ALICE),
        )
        ok = await client.patch(
            f"/api/lab/experiments/{exp_id}/config",
            json={"num_episodes": 20},
            headers=_h(ALICE),
        )
        current = (
            await client.get(f"/api/lab/experiments/{exp_id}", headers=_h(ALICE))
        ).json()
    assert clone.status_code == 422
    assert patch.status_code == 422
    assert negative_lr.status_code == 422
    assert ok.status_code == 200
    assert current["training"]["num_episodes"] == 20


# --- 5. lockout isolation ---------------------------------------------------


async def test_valid_key_does_not_reset_other_credential_lockouts(secure_env):
    async with _client(_app()) as client:
        messages = set()
        for _ in range(30):
            bad = await client.get("/agents", headers=_h("guess-guess-guess"))
            messages.add(bad.text)
            ok = await client.get("/agents", headers=_h(ALICE))
            assert ok.status_code == 200
    assert any("temporarily locked" in message for message in messages)


# --- 7. WebSocket query tokens and /metrics -------------------------------


class _FakeWebSocket:
    def __init__(self, *, query: dict | None = None, headers: dict | None = None):
        self.query_params = query or {}
        self.headers = headers or {}
        self.close_code: int | None = None
        self.sent: list[str] = []
        self._pinged = False

    async def accept(self) -> None:
        pass

    async def close(self, code: int) -> None:
        self.close_code = code

    async def send_text(self, message: str) -> None:
        self.sent.append(message)

    async def receive_text(self) -> str:
        from starlette.websockets import WebSocketDisconnect

        if not self._pinged:
            self._pinged = True
            return "ping"
        raise WebSocketDisconnect()


async def test_lab_websocket_ignores_query_credentials_by_default(
    secure_env, monkeypatch
):
    from stateset_agents.api.routers.training_lab import experiment_ws

    async with _client(_app()) as client:
        exp_id = (
            await client.post(
                "/api/lab/experiments", json={"name": "x"}, headers=_h(ALICE)
            )
        ).json()["id"]

    query = _FakeWebSocket(query={"api_key": ALICE})
    await experiment_ws(query, exp_id)  # type: ignore[arg-type]
    assert query.close_code == 4401

    header = _FakeWebSocket(headers={"x-api-key": ALICE})
    await experiment_ws(header, exp_id)  # type: ignore[arg-type]
    assert header.close_code is None

    monkeypatch.setenv("API_LAB_WS_QUERY_TOKEN", "true")
    _reload()
    opted_in = _FakeWebSocket(query={"api_key": ALICE})
    await experiment_ws(opted_in, exp_id)  # type: ignore[arg-type]
    assert opted_in.close_code is None


async def test_lab_websocket_enforces_experiment_ownership(secure_env):
    from stateset_agents.api.routers.training_lab import experiment_ws

    async with _client(_app()) as client:
        exp_id = (
            await client.post(
                "/api/lab/experiments", json={"name": "x"}, headers=_h(ALICE)
            )
        ).json()["id"]
    bob = _FakeWebSocket(headers={"x-api-key": BOB})
    await experiment_ws(bob, exp_id)  # type: ignore[arg-type]
    assert bob.close_code == 4404
    alice = _FakeWebSocket(headers={"x-api-key": ALICE})
    await experiment_ws(alice, exp_id)  # type: ignore[arg-type]
    assert alice.close_code is None


async def test_metrics_requires_auth_unless_public(secure_env, monkeypatch):
    async with _client(_app()) as client:
        anonymous = await client.get("/metrics")
        authed = await client.get("/metrics", headers=_h(ALICE))
    assert anonymous.status_code == 401
    _assert_prometheus_handler(authed)

    monkeypatch.setenv("API_PUBLIC_METRICS", "true")
    _reload()
    async with _client(_app()) as client:
        public = await client.get("/metrics")
    _assert_prometheus_handler(public)


def _assert_prometheus_handler(response: httpx.Response) -> None:
    """The Prometheus handler answered (404 only when the client is absent)."""
    try:
        import prometheus_client  # noqa: F401
    except ImportError:
        assert response.status_code == 404
        assert "Prometheus client not installed" in response.text
    else:
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")
