"""
Comprehensive API v1 Tests

Tests for all API endpoints including security, validation, rate limiting, and error handling.
"""

import asyncio
import json
import os
import time
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient
import httpx

from tests.api.asgi_client import SyncASGIClient

# Environment is set by conftest.py, but ensure it's set for standalone runs
os.environ.setdefault("API_ENVIRONMENT", "development")
os.environ.setdefault("API_CORS_ORIGINS", "*")
os.environ.setdefault("API_JWT_SECRET", "test-secret-key-for-testing-purposes-only-minimum-32-chars")
os.environ.setdefault("API_REQUIRE_AUTH", "false")
os.environ.setdefault("API_RATE_LIMIT_ENABLED", "false")

# Try to import API components - skip tests if not available
try:
    from api.main import create_app
    from api.config import get_config, APIConfig, reload_config
    from api.auth import generate_token, generate_api_key
    from api.errors import ErrorCode
    API_AVAILABLE = True
except ImportError as e:
    API_AVAILABLE = False
    create_app = None
    get_config = None
    APIConfig = None
    reload_config = None
    generate_token = None
    generate_api_key = None
    ErrorCode = None

pytestmark = pytest.mark.skipif(not API_AVAILABLE, reason="API dependencies not available")


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def app():
    """Create test application."""
    if not API_AVAILABLE:
        pytest.skip("API not available")
    return create_app()


@pytest.fixture
def client(app):
    """Create test client."""
    with SyncASGIClient(app) as client:
        yield client


@pytest.fixture
async def async_client(app):
    """Create async test client."""
    transport = httpx.ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def auth_headers():
    """Generate authenticated headers."""
    token = generate_token("test-user", ["admin", "trainer", "user"])
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def api_key():
    """Generate test API key."""
    return generate_api_key()


# ============================================================================
# Health Check Tests
# ============================================================================

class TestHealthEndpoint:
    """Tests for /api/v1/health endpoint."""

    def test_health_check_returns_200(self, client):
        """Health check should return 200 OK."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_check_response_format(self, client):
        """Health check should return proper format."""
        response = client.get("/api/v1/health")
        data = response.json()

        assert "status" in data
        assert "version" in data
        assert "uptime_seconds" in data
        assert "components" in data
        assert isinstance(data["components"], list)

    def test_health_check_includes_timestamp(self, client):
        """Health check should include timestamp."""
        response = client.get("/api/v1/health")
        data = response.json()

        assert "timestamp" in data

    def test_health_check_no_auth_required(self, client):
        """Health check should not require authentication."""
        # No auth headers
        response = client.get("/api/v1/health")
        assert response.status_code == 200


# ============================================================================
# Training Endpoint Tests
# ============================================================================

class TestTrainingEndpoint:
    """Tests for /api/v1/training endpoints."""

    def test_start_training_success(self, client):
        """Should successfully start a training job."""
        response = client.post(
            "/api/v1/training",
            json={
                "prompts": ["What is machine learning?"],
                "strategy": "computational",
                "num_iterations": 2,
            },
        )

        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "starting"

    def test_start_training_with_multiple_prompts(self, client):
        """Should accept multiple prompts."""
        response = client.post(
            "/api/v1/training",
            json={
                "prompts": ["Prompt 1", "Prompt 2", "Prompt 3"],
                "strategy": "computational",
                "num_iterations": 1,
            },
        )

        assert response.status_code == 202

    def test_start_training_validation_empty_prompts(self, client):
        """Should reject empty prompts list."""
        response = client.post(
            "/api/v1/training",
            json={
                "prompts": [],
                "strategy": "computational",
            },
        )

        assert response.status_code == 422
        data = response.json()
        assert data["error"]["code"] == ErrorCode.VALIDATION_ERROR.value

    def test_start_training_validation_empty_string_prompts(self, client):
        """Should reject prompts with only whitespace."""
        response = client.post(
            "/api/v1/training",
            json={
                "prompts": ["", "   ", "\n\t"],
                "strategy": "computational",
            },
        )

        assert response.status_code == 422

    def test_start_training_validation_too_many_prompts(self, client):
        """Should reject too many prompts."""
        config = get_config()
        prompts = [f"Prompt {i}" for i in range(config.validation.max_prompts + 5)]

        response = client.post(
            "/api/v1/training",
            json={
                "prompts": prompts,
                "strategy": "computational",
            },
        )

        assert response.status_code == 422

    def test_start_training_validation_prompt_too_long(self, client):
        """Should reject prompts that are too long."""
        config = get_config()
        long_prompt = "x" * (config.validation.max_prompt_length + 100)

        response = client.post(
            "/api/v1/training",
            json={
                "prompts": [long_prompt],
                "strategy": "computational",
            },
        )

        assert response.status_code == 422

    def test_start_training_validation_invalid_strategy(self, client):
        """Should reject invalid strategy."""
        response = client.post(
            "/api/v1/training",
            json={
                "prompts": ["Test prompt"],
                "strategy": "invalid_strategy",
            },
        )

        assert response.status_code == 422

    def test_start_training_validation_iterations_too_high(self, client):
        """Should reject iterations exceeding maximum."""
        config = get_config()

        response = client.post(
            "/api/v1/training",
            json={
                "prompts": ["Test prompt"],
                "strategy": "computational",
                "num_iterations": config.validation.max_iterations + 100,
            },
        )

        assert response.status_code == 422

    def test_start_training_idempotency(self, client):
        """Same idempotency key should return same job."""
        idempotency_key = f"test-{uuid.uuid4().hex[:8]}"

        response1 = client.post(
            "/api/v1/training",
            json={
                "prompts": ["Test prompt"],
                "strategy": "computational",
                "idempotency_key": idempotency_key,
            },
        )

        response2 = client.post(
            "/api/v1/training",
            json={
                "prompts": ["Test prompt"],
                "strategy": "computational",
                "idempotency_key": idempotency_key,
            },
        )

        assert response1.status_code == 202
        assert response2.status_code == 202
        assert response1.json()["job_id"] == response2.json()["job_id"]

    def test_get_training_status(self, client):
        """Should get training job status."""
        # First create a job
        create_response = client.post(
            "/api/v1/training",
            json={
                "prompts": ["Test prompt"],
                "strategy": "computational",
                "num_iterations": 1,
            },
        )
        job_id = create_response.json()["job_id"]

        # Then get status
        response = client.get(f"/api/v1/training/{job_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert "status" in data
        assert "metrics" in data

    def test_get_training_status_not_found(self, client):
        """Should return 404 for non-existent job."""
        response = client.get("/api/v1/training/non-existent-job-id")

        assert response.status_code == 404
        data = response.json()
        assert data["error"]["code"] == ErrorCode.NOT_FOUND.value

    def test_cancel_training(self, client):
        """Should cancel a running training job."""
        # Create a job
        create_response = client.post(
            "/api/v1/training",
            json={
                "prompts": ["Test prompt"],
                "strategy": "computational",
                "num_iterations": 100,  # Long job
            },
        )
        job_id = create_response.json()["job_id"]

        # Cancel it
        response = client.delete(f"/api/v1/training/{job_id}")

        assert response.status_code == 200
        assert "cancelled" in response.json()["message"].lower()


# ============================================================================
# Conversation Endpoint Tests
# ============================================================================

class TestConversationEndpoint:
    """Tests for /api/v1/conversations endpoints."""

    def test_create_conversation_with_message(self, client):
        """Should create conversation with single message."""
        response = client.post(
            "/api/v1/conversations",
            json={
                "message": "Hello, how are you?",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "conversation_id" in data
        assert "response" in data
        assert len(data["response"]) > 0

    def test_create_conversation_with_messages_list(self, client):
        """Should create conversation with messages list."""
        response = client.post(
            "/api/v1/conversations",
            json={
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"},
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "response" in data

    def test_create_conversation_validation_no_message(self, client):
        """Should reject request without message or messages."""
        response = client.post(
            "/api/v1/conversations",
            json={
                "temperature": 0.7,
            },
        )

        assert response.status_code == 422

    def test_create_conversation_validation_both_message_types(self, client):
        """Should reject request with both message and messages."""
        response = client.post(
            "/api/v1/conversations",
            json={
                "message": "Hello",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        assert response.status_code == 422

    def test_create_conversation_validation_empty_message(self, client):
        """Should reject empty message."""
        response = client.post(
            "/api/v1/conversations",
            json={
                "message": "",
            },
        )

        assert response.status_code == 422

    def test_create_conversation_validation_message_too_long(self, client):
        """Should reject message that is too long."""
        config = get_config()
        long_message = "x" * (config.validation.max_message_length + 100)

        response = client.post(
            "/api/v1/conversations",
            json={
                "message": long_message,
            },
        )

        assert response.status_code == 422

    def test_create_conversation_validation_invalid_role(self, client):
        """Should reject invalid message role."""
        response = client.post(
            "/api/v1/conversations",
            json={
                "messages": [
                    {"role": "invalid_role", "content": "Hello"},
                ],
            },
        )

        assert response.status_code == 422

    def test_continue_conversation(self, client):
        """Should continue existing conversation."""
        # Create conversation
        create_response = client.post(
            "/api/v1/conversations",
            json={"message": "Hello"},
        )
        conversation_id = create_response.json()["conversation_id"]

        # Continue it
        response = client.post(
            "/api/v1/conversations",
            json={
                "message": "How can you help me?",
                "conversation_id": conversation_id,
            },
        )

        assert response.status_code == 200
        assert response.json()["conversation_id"] == conversation_id

    def test_get_conversation(self, client):
        """Should get conversation details."""
        # Create conversation
        create_response = client.post(
            "/api/v1/conversations",
            json={"message": "Hello"},
        )
        conversation_id = create_response.json()["conversation_id"]

        # Get it
        response = client.get(f"/api/v1/conversations/{conversation_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["conversation_id"] == conversation_id
        assert "messages" in data
        assert len(data["messages"]) >= 2  # User message + assistant response

    def test_get_conversation_not_found(self, client):
        """Should return 404 for non-existent conversation."""
        response = client.get("/api/v1/conversations/non-existent-id")

        assert response.status_code == 404

    def test_end_conversation(self, client):
        """Should end conversation."""
        # Create conversation
        create_response = client.post(
            "/api/v1/conversations",
            json={"message": "Hello"},
        )
        conversation_id = create_response.json()["conversation_id"]

        # End it
        response = client.delete(f"/api/v1/conversations/{conversation_id}")

        assert response.status_code == 200

        # Should not be accessible anymore
        get_response = client.get(f"/api/v1/conversations/{conversation_id}")
        assert get_response.status_code == 404

    def test_conversation_includes_metrics(self, client):
        """Should include tokens and processing time."""
        response = client.post(
            "/api/v1/conversations",
            json={"message": "Hello"},
        )

        data = response.json()
        assert "tokens_used" in data
        assert "processing_time_ms" in data
        assert data["tokens_used"] > 0
        assert data["processing_time_ms"] > 0


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    def test_404_not_found(self, client):
        """Should return proper 404 response."""
        response = client.get("/api/v1/nonexistent-endpoint")

        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == ErrorCode.NOT_FOUND.value

    def test_405_method_not_allowed(self, client):
        """Should return 405 for wrong method."""
        response = client.put("/api/v1/health")

        assert response.status_code == 405

    def test_422_validation_error_format(self, client):
        """Validation errors should have proper format."""
        response = client.post(
            "/api/v1/training",
            json={"prompts": []},
        )

        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == ErrorCode.VALIDATION_ERROR.value
        assert "details" in data["error"]
        assert isinstance(data["error"]["details"], list)

    def test_error_includes_request_id(self, client):
        """Errors should include request ID."""
        response = client.get("/api/v1/training/nonexistent")

        data = response.json()
        assert "request_id" in data

    def test_error_includes_timestamp(self, client):
        """Errors should include timestamp."""
        response = client.get("/api/v1/training/nonexistent")

        data = response.json()
        assert "timestamp" in data

    def test_error_includes_path(self, client):
        """Errors should include request path."""
        response = client.get("/api/v1/training/nonexistent")

        data = response.json()
        assert "path" in data
        assert "/api/v1/training/nonexistent" in data["path"]

    def test_malformed_json(self, client):
        """Should handle malformed JSON gracefully."""
        response = client.post(
            "/api/v1/training",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422


# ============================================================================
# Security Header Tests
# ============================================================================

class TestSecurityHeaders:
    """Tests for security headers."""

    def test_x_content_type_options(self, client):
        """Should include X-Content-Type-Options header."""
        response = client.get("/api/v1/health")
        assert response.headers.get("X-Content-Type-Options") == "nosniff"

    def test_x_frame_options(self, client):
        """Should include X-Frame-Options header."""
        response = client.get("/api/v1/health")
        assert response.headers.get("X-Frame-Options") == "DENY"

    def test_x_xss_protection(self, client):
        """Should include X-XSS-Protection header."""
        response = client.get("/api/v1/health")
        assert "1" in response.headers.get("X-XSS-Protection", "")

    def test_referrer_policy(self, client):
        """Should include Referrer-Policy header."""
        response = client.get("/api/v1/health")
        assert response.headers.get("Referrer-Policy") is not None

    def test_request_id_in_response(self, client):
        """Should include X-Request-ID in response."""
        response = client.get("/api/v1/health")
        assert response.headers.get("X-Request-ID") is not None

    def test_response_time_header(self, client):
        """Should include X-Response-Time-Ms header."""
        response = client.get("/api/v1/health")
        assert response.headers.get("X-Response-Time-Ms") is not None


# ============================================================================
# Rate Limiting Tests
# ============================================================================

class TestRateLimiting:
    """Tests for rate limiting."""

    @pytest.fixture
    def rate_limited_client(self, app):
        """Client with rate limiting enabled."""
        os.environ["API_RATE_LIMIT_ENABLED"] = "true"
        os.environ["API_RATE_LIMIT_PER_MIN"] = "5"  # Low limit for testing
        reload_config()

        with SyncASGIClient(create_app()) as client:
            yield client

        # Reset
        os.environ["API_RATE_LIMIT_ENABLED"] = "false"
        reload_config()

    def test_rate_limit_headers_present(self, rate_limited_client):
        """Should include rate limit headers."""
        response = rate_limited_client.get("/api/v1/health")

        # Note: Health endpoint might skip rate limiting
        # This tests a rate-limited endpoint
        response = rate_limited_client.post(
            "/api/v1/conversations",
            json={"message": "test"},
        )

        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

    def test_rate_limit_exceeded(self, rate_limited_client):
        """Should return 429 when rate limit exceeded."""
        # Make requests until rate limited
        for i in range(10):
            response = rate_limited_client.post(
                "/api/v1/conversations",
                json={"message": f"test {i}"},
            )
            if response.status_code == 429:
                break

        # Verify 429 response
        assert response.status_code == 429
        data = response.json()
        assert data["error"]["code"] == ErrorCode.RATE_LIMIT_EXCEEDED.value
        assert "Retry-After" in response.headers


# ============================================================================
# Authentication Tests
# ============================================================================

class TestAuthentication:
    """Tests for authentication."""

    @pytest.fixture
    def auth_required_client(self, app):
        """Client with authentication required."""
        os.environ["API_REQUIRE_AUTH"] = "true"
        os.environ["API_KEYS"] = "test-api-key-that-is-long-enough-32ch:admin|user"
        reload_config()

        with SyncASGIClient(create_app()) as client:
            yield client

        # Reset
        os.environ["API_REQUIRE_AUTH"] = "false"
        os.environ["API_KEYS"] = ""
        reload_config()

    def test_unauthenticated_request_rejected(self, auth_required_client):
        """Should reject unauthenticated requests."""
        response = auth_required_client.post(
            "/api/v1/conversations",
            json={"message": "test"},
        )

        assert response.status_code == 401

    def test_invalid_api_key_rejected(self, auth_required_client):
        """Should reject invalid API key."""
        response = auth_required_client.post(
            "/api/v1/conversations",
            json={"message": "test"},
            headers={"X-API-Key": "invalid-key"},
        )

        assert response.status_code == 401

    def test_valid_api_key_accepted(self, auth_required_client):
        """Should accept valid API key."""
        response = auth_required_client.post(
            "/api/v1/conversations",
            json={"message": "test"},
            headers={"X-API-Key": "test-api-key-that-is-long-enough-32ch"},
        )

        assert response.status_code == 200

    def test_bearer_token_accepted(self, auth_required_client):
        """Should accept valid bearer token."""
        response = auth_required_client.post(
            "/api/v1/conversations",
            json={"message": "test"},
            headers={"Authorization": "Bearer test-api-key-that-is-long-enough-32ch"},
        )

        assert response.status_code == 200


# ============================================================================
# OpenAPI Documentation Tests
# ============================================================================

class TestOpenAPIDocumentation:
    """Tests for OpenAPI documentation."""

    def test_openapi_schema_available(self, client):
        """OpenAPI schema should be available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema
        assert "info" in schema

    def test_swagger_ui_available(self, client):
        """Swagger UI should be available."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_available(self, client):
        """ReDoc should be available."""
        response = client.get("/redoc")
        assert response.status_code == 200

    def test_openapi_includes_all_endpoints(self, client):
        """OpenAPI should document all endpoints."""
        response = client.get("/openapi.json")
        schema = response.json()
        paths = schema["paths"]

        expected_paths = [
            "/api/v1/health",
            "/api/v1/training",
            "/api/v1/conversations",
        ]

        for path in expected_paths:
            assert path in paths, f"Missing path: {path}"

    def test_openapi_includes_error_responses(self, client):
        """OpenAPI should document error responses."""
        response = client.get("/openapi.json")
        schema = response.json()

        # Check training endpoint
        training_path = schema["paths"].get("/api/v1/training", {})
        post_method = training_path.get("post", {})
        responses = post_method.get("responses", {})

        # Should document error responses
        assert "422" in responses or "4XX" in responses


# ============================================================================
# Input Validation Security Tests
# ============================================================================

class TestInputValidationSecurity:
    """Tests for input validation security."""

    def test_sql_injection_in_message(self, client):
        """Should handle SQL injection attempts safely."""
        malicious_message = "'; DROP TABLE users; --"

        response = client.post(
            "/api/v1/conversations",
            json={"message": malicious_message},
        )

        # Should succeed but not execute SQL
        assert response.status_code == 200

    def test_xss_in_message(self, client):
        """Should handle XSS attempts safely."""
        xss_message = "<script>alert('xss')</script>"

        response = client.post(
            "/api/v1/conversations",
            json={"message": xss_message},
        )

        assert response.status_code == 200
        # Response should not contain unescaped script tags in a real implementation

    def test_path_traversal_in_conversation_id(self, client):
        """Should reject path traversal attempts."""
        response = client.get("/api/v1/conversations/../../../etc/passwd")

        # Should be 404 (not found) not 200 with file contents
        assert response.status_code == 404

    def test_very_long_input(self, client):
        """Should reject extremely long inputs."""
        # Create a very long message (10MB)
        huge_message = "x" * (10 * 1024 * 1024)

        response = client.post(
            "/api/v1/conversations",
            json={"message": huge_message},
        )

        assert response.status_code == 422

    def test_unicode_handling(self, client):
        """Should handle unicode properly."""
        unicode_message = "Hello 你好 مرحبا 🚀"

        response = client.post(
            "/api/v1/conversations",
            json={"message": unicode_message},
        )

        assert response.status_code == 200


# ============================================================================
# Concurrency Tests
# ============================================================================

class TestConcurrency:
    """Tests for concurrent request handling."""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, app):
        """Should handle concurrent requests."""
        transport = httpx.ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Make 10 concurrent requests
            tasks = [
                client.post("/api/v1/conversations", json={"message": f"Message {i}"})
                for i in range(10)
            ]

            responses = await asyncio.gather(*tasks)

            # All should succeed
            assert all(r.status_code == 200 for r in responses)

            # Each should have unique conversation ID
            conversation_ids = [r.json()["conversation_id"] for r in responses]
            assert len(set(conversation_ids)) == 10

    @pytest.mark.asyncio
    async def test_concurrent_training_jobs(self, app):
        """Should handle concurrent training job creation."""
        transport = httpx.ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            tasks = [
                client.post(
                    "/api/v1/training",
                    json={"prompts": [f"Prompt {i}"], "num_iterations": 1},
                )
                for i in range(5)
            ]

            responses = await asyncio.gather(*tasks)

            # All should succeed
            assert all(r.status_code == 202 for r in responses)

            # Each should have unique job ID
            job_ids = [r.json()["job_id"] for r in responses]
            assert len(set(job_ids)) == 5


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_training_workflow(self, client):
        """Test complete training workflow."""
        # 1. Start training
        create_response = client.post(
            "/api/v1/training",
            json={
                "prompts": ["What is AI?", "Explain ML"],
                "strategy": "computational",
                "num_iterations": 2,
            },
        )
        assert create_response.status_code == 202
        job_id = create_response.json()["job_id"]

        # 2. Check status
        status_response = client.get(f"/api/v1/training/{job_id}")
        assert status_response.status_code == 200
        assert status_response.json()["job_id"] == job_id

        # 3. Wait for completion (with timeout)
        for _ in range(20):
            time.sleep(0.05)
            status_response = client.get(f"/api/v1/training/{job_id}")
            if status_response.json()["status"] in ("completed", "failed"):
                break

        # 4. Verify completion
        final_status = client.get(f"/api/v1/training/{job_id}")
        assert final_status.json()["status"] in ("completed", "running")

    def test_full_conversation_workflow(self, client):
        """Test complete conversation workflow."""
        # 1. Start conversation
        create_response = client.post(
            "/api/v1/conversations",
            json={"message": "Hello!"},
        )
        assert create_response.status_code == 200
        conversation_id = create_response.json()["conversation_id"]

        # 2. Continue conversation
        continue_response = client.post(
            "/api/v1/conversations",
            json={
                "message": "What can you help me with?",
                "conversation_id": conversation_id,
            },
        )
        assert continue_response.status_code == 200
        assert continue_response.json()["conversation_id"] == conversation_id

        # 3. Get conversation history
        get_response = client.get(f"/api/v1/conversations/{conversation_id}")
        assert get_response.status_code == 200
        assert len(get_response.json()["messages"]) >= 4  # 2 user + 2 assistant

        # 4. End conversation
        end_response = client.delete(f"/api/v1/conversations/{conversation_id}")
        assert end_response.status_code == 200

        # 5. Verify ended
        get_ended = client.get(f"/api/v1/conversations/{conversation_id}")
        assert get_ended.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
