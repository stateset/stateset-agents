"""
Tests for API dependency injection functions.

Covers the shared dependencies extracted during the API hardening:
- get_inference_service
- get_agent_service
- require_auth_if_enabled
- Unified AuthenticatedUser model
"""

from fastapi import Depends, FastAPI

from stateset_agents.api.auth import AuthenticatedUser
from stateset_agents.api.dependencies import (
    get_agent_service,
    get_inference_service,
    require_auth_if_enabled,
)
from tests.api.asgi_client import SyncASGIClient


# ============================================================================
# AuthenticatedUser tests
# ============================================================================


class TestAuthenticatedUser:
    """Tests for the unified AuthenticatedUser Pydantic model."""

    def test_direct_construction(self):
        user = AuthenticatedUser(
            user_id="u1",
            roles=["admin"],
            auth_method="api_key",
        )
        assert user.user_id == "u1"
        assert user.roles == ["admin"]
        assert user.email is None
        assert user.name is None
        assert user.metadata is None

    def test_construction_with_all_fields(self):
        user = AuthenticatedUser(
            user_id="u2",
            roles=["user"],
            api_key="sk-abc***xyz",
            auth_method="jwt",
            email="u2@example.com",
            name="User Two",
            metadata={"org": "acme"},
        )
        assert user.email == "u2@example.com"
        assert user.name == "User Two"
        assert user.metadata == {"org": "acme"}

    def test_from_dict_basic(self):
        user = AuthenticatedUser.from_dict(
            {"user_id": "u3", "roles": ["trainer"], "email": "u3@x.com"}
        )
        assert user.user_id == "u3"
        assert user.roles == ["trainer"]
        assert user.email == "u3@x.com"

    def test_from_dict_uses_sub_fallback(self):
        user = AuthenticatedUser.from_dict({"sub": "sub-user", "roles": []})
        assert user.user_id == "sub-user"

    def test_from_dict_extra_fields_go_to_metadata(self):
        user = AuthenticatedUser.from_dict(
            {"user_id": "u4", "roles": [], "custom_key": 42, "org": "acme"}
        )
        assert user.metadata == {"custom_key": 42, "org": "acme"}

    def test_from_dict_no_extra_fields_metadata_is_none(self):
        user = AuthenticatedUser.from_dict({"user_id": "u5", "roles": ["admin"]})
        assert user.metadata is None

    def test_default_auth_method(self):
        """auth_method defaults to 'none'."""
        user = AuthenticatedUser(user_id="u6", roles=[])
        assert user.auth_method == "none"

    def test_model_dump_roundtrip(self):
        """Ensure model_dump produces a dict that from_dict can reconstruct."""
        original = AuthenticatedUser(
            user_id="rt",
            roles=["admin"],
            auth_method="jwt",
            email="rt@x.com",
        )
        reconstructed = AuthenticatedUser.from_dict(original.model_dump())
        assert reconstructed.user_id == original.user_id
        assert reconstructed.roles == original.roles
        assert reconstructed.email == original.email

    def test_import_from_dependencies_works(self):
        """Verify that importing from dependencies still works (re-export)."""
        from stateset_agents.api.dependencies import AuthenticatedUser as Dep

        assert Dep is AuthenticatedUser


# ============================================================================
# get_inference_service tests
# ============================================================================


class TestGetInferenceService:
    """Tests for the shared inference service dependency."""

    def test_returns_service_from_app_state(self):
        from stateset_agents.api.services.inference_service import (
            InferenceConfig,
            InferenceService,
        )

        app = FastAPI()
        expected = InferenceService(InferenceConfig(backend="stub"))
        app.state.inference_service = expected

        @app.get("/test")
        def endpoint(svc=Depends(get_inference_service)):
            return {"is_stub": svc.is_stub, "same": svc is expected}

        with SyncASGIClient(app) as client:
            resp = client.get("/test")
            assert resp.status_code == 200
            assert resp.json()["is_stub"] is True

    def test_creates_service_lazily_if_missing(self):
        app = FastAPI()

        @app.get("/test")
        def endpoint(svc=Depends(get_inference_service)):
            return {"ok": True}

        with SyncASGIClient(app) as client:
            resp = client.get("/test")
            assert resp.status_code == 200


# ============================================================================
# get_agent_service tests
# ============================================================================


class TestGetAgentService:
    """Tests for the shared agent service dependency."""

    def test_returns_service_from_app_state(self):
        from stateset_agents.api.services.agent_service import AgentService
        from stateset_agents.utils.security import SecurityMonitor

        app = FastAPI()
        expected = AgentService(SecurityMonitor())
        app.state.agent_service = expected

        @app.get("/test")
        def endpoint(svc=Depends(get_agent_service)):
            return {"has_agents": isinstance(svc.agents, dict)}

        with SyncASGIClient(app) as client:
            resp = client.get("/test")
            assert resp.status_code == 200
            assert resp.json()["has_agents"] is True

    def test_creates_service_lazily_if_missing(self):
        app = FastAPI()

        @app.get("/test")
        def endpoint(svc=Depends(get_agent_service)):
            return {"has_agents": isinstance(svc.agents, dict)}

        with SyncASGIClient(app) as client:
            resp = client.get("/test")
            assert resp.status_code == 200
            assert resp.json()["has_agents"] is True


# ============================================================================
# require_auth_if_enabled tests
# ============================================================================


class TestRequireAuthIfEnabled:
    """Tests for the shared auth-gate dependency."""

    def test_allows_anonymous_when_auth_disabled(self, monkeypatch):
        monkeypatch.setenv("API_REQUIRE_AUTH", "false")

        from stateset_agents.api import config as api_config

        prev = api_config._config
        api_config.reload_config()

        app = FastAPI()

        @app.get("/test")
        async def endpoint(user=Depends(require_auth_if_enabled)):
            return {"user": user.user_id if user else None}

        try:
            with SyncASGIClient(app) as client:
                resp = client.get("/test")
                assert resp.status_code == 200
                assert resp.json()["user"] is None
        finally:
            api_config._config = prev

    def test_rejects_anonymous_when_auth_enabled(self, monkeypatch):
        monkeypatch.setenv("API_REQUIRE_AUTH", "true")

        from stateset_agents.api import config as api_config

        prev = api_config._config
        api_config.reload_config()

        app = FastAPI()

        @app.get("/test")
        async def endpoint(user=Depends(require_auth_if_enabled)):
            return {"ok": True}

        try:
            with SyncASGIClient(app) as client:
                resp = client.get("/test")
                assert resp.status_code == 401
        finally:
            api_config._config = prev
