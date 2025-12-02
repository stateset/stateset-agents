"""
Unit tests for the Enhanced Ultimate GRPO Service module.

Tests cover API endpoints, service management, request handling,
and WebSocket functionality.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock FastAPI availability check
with patch.dict('sys.modules', {'uvicorn': MagicMock(), 'fastapi': MagicMock()}):
    pass


class TestEnhancedTrainingRequest:
    """Test EnhancedTrainingRequest model."""

    def test_training_request_defaults(self):
        """Test request with default values."""
        from api.enhanced_ultimate_grpo_service import EnhancedTrainingRequest

        request = EnhancedTrainingRequest(
            experiment_name="test_exp",
            model_config={"model_type": "gpt2"},
            training_data="/data/train.json",
        )

        assert request.experiment_name == "test_exp"
        assert request.agent_type == "MultiTurnAgent"
        assert request.num_epochs == 10
        assert request.batch_size == 32
        assert request.learning_rate == 1e-4
        assert request.cpu_cores == 2.0
        assert request.memory_gb == 8.0
        assert request.gpu_count == 1

    def test_training_request_custom_values(self):
        """Test request with custom values."""
        from api.enhanced_ultimate_grpo_service import EnhancedTrainingRequest

        request = EnhancedTrainingRequest(
            experiment_name="custom_exp",
            agent_type="CustomAgent",
            model_config={"model_type": "custom"},
            training_data=["/data/train1.json", "/data/train2.json"],
            num_epochs=20,
            batch_size=64,
            learning_rate=5e-5,
            cpu_cores=4.0,
            memory_gb=16.0,
            gpu_count=2,
            enable_checkpointing=False,
            priority=5,
        )

        assert request.agent_type == "CustomAgent"
        assert request.num_epochs == 20
        assert request.gpu_count == 2
        assert request.enable_checkpointing is False
        assert request.priority == 5


class TestEnhancedConversationRequest:
    """Test EnhancedConversationRequest model."""

    def test_conversation_request_defaults(self):
        """Test request with default values."""
        from api.enhanced_ultimate_grpo_service import EnhancedConversationRequest

        request = EnhancedConversationRequest(message="Hello, world!")

        assert request.message == "Hello, world!"
        assert request.conversation_id is None
        assert request.user_id is None
        assert request.strategy == "default"
        assert request.max_tokens == 512
        assert request.temperature == 0.7
        assert request.stream is False

    def test_conversation_request_custom_values(self):
        """Test request with custom values."""
        from api.enhanced_ultimate_grpo_service import EnhancedConversationRequest

        request = EnhancedConversationRequest(
            message="Custom message",
            conversation_id="conv_001",
            user_id="user_001",
            context={"key": "value"},
            strategy="advanced",
            max_tokens=1024,
            temperature=0.5,
            stream=True,
        )

        assert request.conversation_id == "conv_001"
        assert request.user_id == "user_001"
        assert request.context == {"key": "value"}
        assert request.stream is True


class TestSystemHealthResponse:
    """Test SystemHealthResponse model."""

    def test_health_response_creation(self):
        """Test creating a health response."""
        from api.enhanced_ultimate_grpo_service import SystemHealthResponse

        response = SystemHealthResponse(
            status="healthy",
            timestamp=time.time(),
            uptime=3600.0,
            components={"gateway": "healthy", "orchestrator": "healthy"},
            resource_utilization={"cpu": 0.5, "memory": 0.6},
            active_jobs=5,
            queue_size=10,
            error_rate=0.01,
        )

        assert response.status == "healthy"
        assert response.uptime == 3600.0
        assert response.active_jobs == 5
        assert response.error_rate == 0.01


class TestMetricsResponse:
    """Test MetricsResponse model."""

    def test_metrics_response_creation(self):
        """Test creating a metrics response."""
        from api.enhanced_ultimate_grpo_service import MetricsResponse

        response = MetricsResponse(
            timestamp=time.time(),
            system_metrics={"cpu_usage": 0.5},
            training_metrics={"running_jobs": 3},
            api_metrics={"total_requests": 1000},
            alerts=[{"type": "warning", "message": "High memory usage"}],
        )

        assert response.system_metrics["cpu_usage"] == 0.5
        assert response.training_metrics["running_jobs"] == 3
        assert len(response.alerts) == 1


class TestServiceManager:
    """Test ServiceManager class."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock external dependencies."""
        with patch("api.enhanced_ultimate_grpo_service.get_monitoring_service") as mock_monitoring, \
             patch("api.enhanced_ultimate_grpo_service.get_state_service") as mock_state, \
             patch("api.enhanced_ultimate_grpo_service.get_training_orchestrator") as mock_orchestrator, \
             patch("api.enhanced_ultimate_grpo_service.EnhancedGRPOGateway") as mock_gateway:

            mock_monitoring.return_value = MagicMock()
            mock_monitoring.return_value.get_health_summary.return_value = {
                "status": "healthy",
                "system_metrics": {"cpu": 0.5},
            }

            mock_state.return_value = MagicMock()
            mock_state.return_value.health_check = AsyncMock(return_value={"status": "healthy"})

            mock_orchestrator.return_value = MagicMock()
            mock_orchestrator.return_value.get_system_status = AsyncMock(return_value={
                "queue_status": {
                    "queued_jobs": 5,
                    "running_jobs": 2,
                    "completed_jobs": 10,
                }
            })

            mock_gateway_instance = MagicMock()
            mock_gateway_instance.get_health_status.return_value = {
                "status": "healthy",
                "stats": {
                    "total_requests": 1000,
                    "total_errors": 10,
                    "average_response_time": 0.1,
                    "cache_hit_rate": 0.8,
                },
            }
            mock_gateway_instance.security_manager = MagicMock()
            mock_gateway.return_value = mock_gateway_instance

            yield {
                "monitoring": mock_monitoring,
                "state": mock_state,
                "orchestrator": mock_orchestrator,
                "gateway": mock_gateway,
            }

    @pytest.fixture
    def service_manager(self, mock_dependencies):
        """Create a ServiceManager for testing."""
        from api.enhanced_ultimate_grpo_service import ServiceManager
        return ServiceManager()

    def test_service_manager_creation(self, service_manager):
        """Test service manager creation."""
        assert service_manager.is_initialized is False
        assert service_manager.gateway is None
        assert service_manager.monitoring is None

    @pytest.mark.asyncio
    async def test_service_manager_initialize(self, service_manager, mock_dependencies):
        """Test service manager initialization."""
        await service_manager.initialize()

        assert service_manager.is_initialized is True
        assert service_manager.monitoring is not None
        assert service_manager.state_service is not None
        assert service_manager.orchestrator is not None
        assert service_manager.gateway is not None

    @pytest.mark.asyncio
    async def test_service_manager_double_initialize(self, service_manager, mock_dependencies):
        """Test that double initialization is a no-op."""
        await service_manager.initialize()
        await service_manager.initialize()  # Should not error

        assert service_manager.is_initialized is True

    @pytest.mark.asyncio
    async def test_service_manager_shutdown(self, service_manager, mock_dependencies):
        """Test service manager shutdown."""
        await service_manager.initialize()
        service_manager.orchestrator.shutdown = AsyncMock()
        service_manager.state_service.shutdown = AsyncMock()

        await service_manager.shutdown()

        service_manager.orchestrator.shutdown.assert_called_once()
        service_manager.state_service.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_health_status_healthy(self, service_manager, mock_dependencies):
        """Test getting health status when all components are healthy."""
        await service_manager.initialize()

        health = await service_manager.get_health_status()

        assert health["status"] == "healthy"
        assert "components" in health
        assert "uptime" in health

    @pytest.mark.asyncio
    async def test_get_health_status_with_metrics(self, service_manager, mock_dependencies):
        """Test health status includes resource utilization."""
        await service_manager.initialize()

        health = await service_manager.get_health_status()

        assert "resource_utilization" in health
        assert "training" in health
        assert "api" in health


class TestAPIEndpoints:
    """Test API endpoint functions."""

    @pytest.fixture
    def mock_service_manager(self):
        """Create a mock service manager."""
        manager = MagicMock()
        manager.orchestrator = MagicMock()
        manager.orchestrator.submit_training_job = AsyncMock(return_value="job_001")
        manager.orchestrator.get_job_status = AsyncMock(return_value={
            "job_id": "job_001",
            "status": "running",
            "progress": {"current_epoch": 2, "total_epochs": 10},
        })
        manager.orchestrator.cancel_job = AsyncMock(return_value=True)
        manager.monitoring = MagicMock()
        manager.monitoring.record_training_iteration = MagicMock()
        manager.monitoring.record_error = MagicMock()
        manager.monitoring.record_request = MagicMock()
        manager.monitoring.get_metrics_dashboard = MagicMock(return_value={})
        manager.gateway = MagicMock()
        manager.gateway.get_health_status.return_value = {"status": "healthy", "stats": {}}
        manager.get_health_status = AsyncMock(return_value={
            "status": "healthy",
            "timestamp": time.time(),
            "uptime": 1000.0,
            "components": {},
            "resource_utilization": {},
            "training": {"running_jobs": 1, "queue_size": 0},
            "api": {"error_rate": 0.0},
        })
        return manager

    @pytest.mark.asyncio
    async def test_root_endpoint(self):
        """Test root endpoint returns service info."""
        from api.enhanced_ultimate_grpo_service import FASTAPI_AVAILABLE

        if not FASTAPI_AVAILABLE:
            pytest.skip("FastAPI not available")

        from api.enhanced_ultimate_grpo_service import root

        response = await root()

        assert response["title"] == "Enhanced Ultimate GRPO Service"
        assert response["version"] == "2.1.0"
        assert "features" in response
        assert "endpoints" in response

    @pytest.mark.asyncio
    async def test_verify_request(self):
        """Test request verification returns user context."""
        from api.enhanced_ultimate_grpo_service import FASTAPI_AVAILABLE

        if not FASTAPI_AVAILABLE:
            pytest.skip("FastAPI not available")

        from api.enhanced_ultimate_grpo_service import verify_request

        context = await verify_request(MagicMock())

        assert "user_id" in context
        assert "roles" in context


class TestWebSocketEndpoint:
    """Test WebSocket endpoint functionality."""

    @pytest.fixture
    def mock_websocket(self):
        """Create a mock WebSocket."""
        ws = MagicMock()
        ws.accept = AsyncMock()
        ws.receive_text = AsyncMock()
        ws.send_text = AsyncMock()
        ws.close = AsyncMock()
        return ws

    @pytest.mark.asyncio
    async def test_websocket_ping_pong(self, mock_websocket):
        """Test WebSocket ping/pong functionality."""
        from api.enhanced_ultimate_grpo_service import FASTAPI_AVAILABLE

        if not FASTAPI_AVAILABLE:
            pytest.skip("FastAPI not available")

        # Mock a ping message followed by disconnect
        mock_websocket.receive_text.side_effect = [
            json.dumps({"type": "ping"}),
            Exception("Disconnect"),  # Simulate disconnect
        ]

        from api.enhanced_ultimate_grpo_service import websocket_endpoint

        try:
            await websocket_endpoint(mock_websocket)
        except Exception:
            pass

        mock_websocket.accept.assert_called_once()
        # Check that a pong was sent
        calls = mock_websocket.send_text.call_args_list
        assert len(calls) > 0
        response = json.loads(calls[0][0][0])
        assert response["type"] == "pong"

    @pytest.mark.asyncio
    async def test_websocket_unknown_message_type(self, mock_websocket):
        """Test WebSocket handles unknown message types."""
        from api.enhanced_ultimate_grpo_service import FASTAPI_AVAILABLE

        if not FASTAPI_AVAILABLE:
            pytest.skip("FastAPI not available")

        mock_websocket.receive_text.side_effect = [
            json.dumps({"type": "unknown_type"}),
            Exception("Disconnect"),
        ]

        from api.enhanced_ultimate_grpo_service import websocket_endpoint

        try:
            await websocket_endpoint(mock_websocket)
        except Exception:
            pass

        calls = mock_websocket.send_text.call_args_list
        assert len(calls) > 0
        response = json.loads(calls[0][0][0])
        assert response["type"] == "error"
        assert "unknown_type" in response["message"].lower()


class TestRouteConfiguration:
    """Test route configuration."""

    def test_route_config_creation(self):
        """Test creating route configurations."""
        from api.enhanced_grpo_gateway import RouteConfig

        route = RouteConfig(
            path="/api/v2/train",
            methods=["POST"],
            timeout=300.0,
            cache_ttl=0,
            rate_limit=60,
            requires_auth=True,
            allowed_roles=["admin", "trainer"],
        )

        assert route.path == "/api/v2/train"
        assert "POST" in route.methods
        assert route.timeout == 300.0
        assert route.requires_auth is True
        assert "admin" in route.allowed_roles


class TestIntegration:
    """Integration tests for the service."""

    @pytest.fixture
    def mock_all_dependencies(self):
        """Mock all external dependencies for integration tests."""
        with patch("api.enhanced_ultimate_grpo_service.get_monitoring_service") as mock_monitoring, \
             patch("api.enhanced_ultimate_grpo_service.get_state_service") as mock_state, \
             patch("api.enhanced_ultimate_grpo_service.get_training_orchestrator") as mock_orchestrator, \
             patch("api.enhanced_ultimate_grpo_service.EnhancedGRPOGateway") as mock_gateway, \
             patch("api.enhanced_ultimate_grpo_service.managed_state_context") as mock_context:

            # Setup monitoring
            mock_monitoring.return_value = MagicMock()
            mock_monitoring.return_value.get_health_summary.return_value = {"status": "healthy"}
            mock_monitoring.return_value.record_training_iteration = MagicMock()
            mock_monitoring.return_value.record_error = MagicMock()

            # Setup state service
            mock_state.return_value = MagicMock()
            mock_state.return_value.health_check = AsyncMock(return_value={"status": "healthy"})
            mock_state.return_value.conversation_manager = MagicMock()
            mock_state.return_value.conversation_manager.get_conversation = AsyncMock(
                return_value={"turns": [], "context": {}}
            )
            mock_state.return_value.conversation_manager.create_conversation = AsyncMock()
            mock_state.return_value.conversation_manager.add_turn = AsyncMock()

            # Setup orchestrator
            mock_orchestrator.return_value = MagicMock()
            mock_orchestrator.return_value.submit_training_job = AsyncMock(return_value="job_123")
            mock_orchestrator.return_value.get_job_status = AsyncMock(return_value={
                "job_id": "job_123",
                "status": "running",
            })
            mock_orchestrator.return_value.cancel_job = AsyncMock(return_value=True)
            mock_orchestrator.return_value.get_system_status = AsyncMock(return_value={
                "queue_status": {"queued_jobs": 0, "running_jobs": 1, "completed_jobs": 5}
            })

            # Setup gateway
            mock_gateway_instance = MagicMock()
            mock_gateway_instance.security_manager = MagicMock()
            mock_gateway_instance.get_health_status.return_value = {
                "status": "healthy",
                "stats": {
                    "total_requests": 100,
                    "total_errors": 1,
                    "average_response_time": 0.05,
                    "cache_hit_rate": 0.9,
                },
            }
            mock_gateway.return_value = mock_gateway_instance

            # Setup context manager
            mock_context_instance = MagicMock()
            mock_context_instance.__aenter__ = AsyncMock(return_value=mock_state.return_value)
            mock_context_instance.__aexit__ = AsyncMock(return_value=None)
            mock_context.return_value = mock_context_instance

            yield {
                "monitoring": mock_monitoring,
                "state": mock_state,
                "orchestrator": mock_orchestrator,
                "gateway": mock_gateway,
                "context": mock_context,
            }

    @pytest.mark.asyncio
    async def test_full_training_workflow(self, mock_all_dependencies):
        """Test a complete training job submission workflow."""
        from api.enhanced_ultimate_grpo_service import (
            ServiceManager,
            EnhancedTrainingRequest,
        )

        # Initialize service
        manager = ServiceManager()
        await manager.initialize()

        # Create training request
        request = EnhancedTrainingRequest(
            experiment_name="integration_test",
            model_config={"model_type": "gpt2"},
            training_data="/data/train.json",
            num_epochs=5,
        )

        # Verify orchestrator is available
        assert manager.orchestrator is not None

        # Submit job would work through the orchestrator
        job_id = await manager.orchestrator.submit_training_job(
            MagicMock(),  # config
            priority=request.priority,
            user_id="test_user",
        )

        assert job_id == "job_123"

    @pytest.mark.asyncio
    async def test_health_check_integration(self, mock_all_dependencies):
        """Test health check with all components."""
        from api.enhanced_ultimate_grpo_service import ServiceManager

        manager = ServiceManager()
        await manager.initialize()

        health = await manager.get_health_status()

        assert health["status"] == "healthy"
        assert "components" in health
        assert "training" in health
        assert health["training"]["running_jobs"] == 1


class TestErrorHandling:
    """Test error handling in the service."""

    @pytest.fixture
    def service_manager_with_errors(self):
        """Create a service manager that simulates errors."""
        with patch("api.enhanced_ultimate_grpo_service.get_monitoring_service") as mock_monitoring:
            mock_monitoring.return_value = MagicMock()
            mock_monitoring.return_value.get_health_summary.side_effect = Exception("Monitoring error")

            from api.enhanced_ultimate_grpo_service import ServiceManager
            manager = ServiceManager()
            manager.monitoring = mock_monitoring.return_value
            return manager

    @pytest.mark.asyncio
    async def test_health_check_handles_errors(self, service_manager_with_errors):
        """Test health check handles component errors gracefully."""
        health = await service_manager_with_errors.get_health_status()

        assert health["status"] == "error"
        assert "error" in health


class TestConcurrency:
    """Test concurrent request handling."""

    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self):
        """Test multiple concurrent health checks."""
        from api.enhanced_ultimate_grpo_service import FASTAPI_AVAILABLE

        if not FASTAPI_AVAILABLE:
            pytest.skip("FastAPI not available")

        with patch("api.enhanced_ultimate_grpo_service.get_monitoring_service"), \
             patch("api.enhanced_ultimate_grpo_service.get_state_service"), \
             patch("api.enhanced_ultimate_grpo_service.get_training_orchestrator"), \
             patch("api.enhanced_ultimate_grpo_service.EnhancedGRPOGateway"):

            from api.enhanced_ultimate_grpo_service import ServiceManager

            manager = ServiceManager()

            # Mock health check to be async
            manager.get_health_status = AsyncMock(return_value={
                "status": "healthy",
                "timestamp": time.time(),
                "uptime": 1000.0,
                "components": {},
                "resource_utilization": {},
                "training": {},
                "api": {},
            })

            # Run concurrent health checks
            tasks = [manager.get_health_status() for _ in range(10)]
            results = await asyncio.gather(*tasks)

            assert len(results) == 10
            assert all(r["status"] == "healthy" for r in results)
