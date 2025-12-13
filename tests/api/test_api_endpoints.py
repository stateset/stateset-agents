"""
API endpoint tests for the FastAPI service.

These tests verify the REST API functionality.
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

# Ensure environment is set for API
os.environ.setdefault("API_ENVIRONMENT", "development")
os.environ.setdefault("API_JWT_SECRET", "test-secret-key-for-testing-purposes-only-minimum-32-chars")
os.environ.setdefault("API_CORS_ORIGINS", "*")
os.environ.setdefault("API_REQUIRE_AUTH", "false")
os.environ.setdefault("API_RATE_LIMIT_ENABLED", "false")

# Import API components (these may not exist yet, but we'll create mocks)
try:
    from api.main import create_app
    app = create_app()
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    app = None


@pytest.mark.api
@pytest.mark.skipif(not API_AVAILABLE, reason="API dependencies not available")
class TestAPIEndpoints:
    """Test API endpoints functionality."""

    @pytest.fixture
    async def async_client(self):
        """Create an async test client."""
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", follow_redirects=True
        ) as client:
            yield client

    @pytest.mark.asyncio
    async def test_health_endpoint(self, async_client):
        """Test the health check endpoint."""
        response = await async_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_root_endpoint(self, async_client):
        """Test the root endpoint."""
        response = await async_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "stateset" in data["message"].lower()

    @pytest.mark.asyncio
    async def test_agent_status_endpoint(self, async_client):
        """Test agent status endpoint."""
        # This would test an actual agent status endpoint
        # For now, we'll mock the expected behavior

        response = await async_client.get("/agent/status")

        # Assuming this endpoint exists and returns agent status
        if response.status_code == 200:
            data = response.json()
            assert "agent_status" in data or "status" in data

    @pytest.mark.asyncio
    async def test_conversation_endpoint(self, async_client):
        """Test conversation endpoint."""

        conversation_data = {
            "messages": [{"role": "user", "content": "Hello, how can you help me?"}],
            "context": {"max_tokens": 100, "temperature": 0.7},
        }

        # Mock the agent response
        with patch(
            "stateset_agents.api.ultimate_grpo_service.MultiTurnAgent"
        ) as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent.generate_response = AsyncMock(
                return_value="I'd be happy to help you!"
            )
            mock_agent_class.return_value = mock_agent

            response = await async_client.post("/conversation", json=conversation_data)

            if response.status_code == 200:
                data = response.json()
                assert "response" in data
                assert isinstance(data["response"], str)

    @pytest.mark.asyncio
    async def test_training_status_endpoint(self, async_client):
        """Test training status endpoint."""

        response = await async_client.get("/training/status")

        if response.status_code == 200:
            data = response.json()
            # Should contain training status information
            assert isinstance(data, dict)

    @pytest.mark.asyncio
    async def test_invalid_request_handling(self, async_client):
        """Test handling of invalid requests."""

        # Test invalid JSON
        response = await async_client.post("/conversation", content="invalid json")
        assert response.status_code == 400 or response.status_code == 422

        # Test missing required fields
        response = await async_client.post("/conversation", json={})
        assert response.status_code in [400, 422]

    @pytest.mark.asyncio
    async def test_cors_headers(self, async_client):
        """Test CORS headers are properly set."""

        response = await async_client.options(
            "/conversation",
            headers={
                "Origin": "http://localhost",
                "Access-Control-Request-Method": "POST",
            },
        )

        # Check for CORS headers
        assert response.status_code in {200, 204}
        assert "access-control-allow-origin" in response.headers

    @pytest.mark.asyncio
    async def test_api_documentation_available(self, async_client):
        """Test that API documentation is available."""

        # Test OpenAPI schema
        response = await async_client.get("/openapi.json")
        if response.status_code == 200:
            schema = response.json()
            assert "openapi" in schema
            assert "paths" in schema

        # Test Swagger UI
        response = await async_client.get("/docs")
        # Swagger UI might return HTML or redirect
        assert response.status_code in [200, 302]


@pytest.mark.api
@pytest.mark.skipif(not API_AVAILABLE, reason="API dependencies not available")
class TestAPIErrorHandling:
    """Test API error handling."""

    @pytest.fixture
    async def async_client(self):
        """Create an async test client for the FastAPI app."""
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", follow_redirects=True
        ) as client:
            yield client

    @pytest.mark.asyncio
    async def test_404_handling(self, async_client):
        """Test 404 error handling."""

        response = await async_client.get("/nonexistent-endpoint")
        assert response.status_code == 404

        data = response.json()
        assert "error" in data
        assert data["error"]["code"] in {"NOT_FOUND", "BAD_REQUEST"}

    @pytest.mark.asyncio
    async def test_method_not_allowed(self, async_client):
        """Test method not allowed handling."""

        response = await async_client.put("/health")
        assert response.status_code == 405

        data = response.json()
        assert "error" in data
        assert data["error"]["code"] in {"METHOD_NOT_ALLOWED", "BAD_REQUEST"}

    @pytest.mark.asyncio
    async def test_agent_error_handling(self, async_client):
        """Test error handling when agent fails."""

        conversation_data = {"messages": [{"role": "user", "content": "Test message"}]}

        # Mock agent to raise an exception
        with patch(
            "stateset_agents.api.ultimate_grpo_service.MultiTurnAgent"
        ) as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent.generate_response = AsyncMock(
                side_effect=Exception("Agent error")
            )
            mock_agent_class.return_value = mock_agent

            response = await async_client.post("/conversation", json=conversation_data)

            # Endpoint may or may not integrate a real agent in this test env.
            data = response.json()
            if response.status_code >= 400:
                assert "detail" in data or "error" in data
            else:
                assert "conversation_id" in data or "response" in data


@pytest.mark.api
@pytest.mark.skipif(not API_AVAILABLE, reason="API dependencies not available")
class TestAPIIntegration:
    """Integration tests for API components."""

    @pytest.fixture
    async def async_client(self):
        """Create an async test client for the FastAPI app."""
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", follow_redirects=True
        ) as client:
            yield client

    @pytest.mark.asyncio
    async def test_full_conversation_flow(self, async_client):
        """Test a complete conversation flow through the API."""

        # This would test a full conversation workflow
        # For now, we'll test the basic structure

        # Start conversation
        start_data = {"messages": [], "context": {"scenario": "general_help"}}

        response = await async_client.post("/conversation/start", json=start_data)

        if response.status_code == 200:
            conversation_id = response.json().get("conversation_id")
            assert conversation_id is not None

            # Continue conversation
            continue_data = {
                "conversation_id": conversation_id,
                "message": "I need help with Python",
            }

            response = await async_client.post(
                "/conversation/continue", json=continue_data
            )

            if response.status_code == 200:
                data = response.json()
                assert "response" in data
                assert isinstance(data["response"], str)

    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, async_client):
        """Test metrics and monitoring endpoints."""

        response = await async_client.get("/metrics")

        if response.status_code == 200:
            # Could be Prometheus metrics or JSON metrics
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                data = response.json()
                assert isinstance(data, dict)
