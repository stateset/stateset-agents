"""
Enhanced Ultimate GRPO Service - Next-Generation AI Agent Training Platform

This service integrates all advanced components: API gateway, monitoring, state management,
training orchestration, and comprehensive observability for production-ready GRPO services.
"""

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

try:
    import uvicorn
    from fastapi import (
        BackgroundTasks,
        Depends,
        FastAPI,
        HTTPException,
        WebSocket,
        WebSocketDisconnect,
    )
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, ConfigDict, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from stateset_agents.core.advanced_monitoring import (
    AdvancedMonitoringService,
    get_monitoring_service,
    monitor_async_function,
)
from stateset_agents.core.enhanced_state_management import (
    DistributedStateService,
    get_state_service,
    managed_state_context,
)
from stateset_agents.core.error_handling import ErrorHandler, GRPOException
from stateset_agents.core.performance_optimizer import OptimizationLevel, PerformanceOptimizer
from stateset_agents.training.advanced_training_orchestrator import (
    AdvancedTrainingOrchestrator,
    ResourceRequirement,
    ResourceType,
    TrainingConfig,
    get_training_orchestrator,
)

# Import our enhanced components
from stateset_agents.api.enhanced_grpo_gateway import (
    EnhancedGRPOGateway,
    LoadBalancingStrategy,
    RouteConfig,
    SecurityManager,
    ServiceInstance,
)

logger = logging.getLogger(__name__)


# API Models
class EnhancedTrainingRequest(BaseModel):
    """Enhanced training request with comprehensive configuration"""

    model_config = ConfigDict(populate_by_name=True)

    experiment_name: str
    agent_type: str = "MultiTurnAgent"
    model_configuration: Dict[str, Any] = Field(..., alias="model_config")
    training_data: Union[str, List[str]]

    # Training parameters
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"

    # GRPO specific
    grpo_epsilon: float = 0.2
    grpo_value_loss_coef: float = 0.5
    grpo_entropy_coef: float = 0.01

    # Resource requirements
    cpu_cores: float = 2.0
    memory_gb: float = 8.0
    gpu_count: int = 1
    max_runtime_hours: Optional[int] = None

    # Advanced settings
    enable_checkpointing: bool = True
    enable_early_stopping: bool = True
    enable_wandb: bool = True
    priority: int = 1


class EnhancedConversationRequest(BaseModel):
    """Enhanced conversation request"""

    message: str
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    strategy: str = "default"
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False


class SystemHealthResponse(BaseModel):
    """System health response"""

    status: str
    timestamp: float
    uptime: float
    components: Dict[str, Any]
    resource_utilization: Dict[str, float]
    active_jobs: int
    queue_size: int
    error_rate: float


class MetricsResponse(BaseModel):
    """Metrics response"""

    timestamp: float
    system_metrics: Dict[str, float]
    training_metrics: Dict[str, float]
    api_metrics: Dict[str, float]
    alerts: List[Dict[str, Any]]


class ServiceManager:
    """Centralized service management"""

    def __init__(self):
        # Core services
        self.gateway: Optional[EnhancedGRPOGateway] = None
        self.monitoring: Optional[AdvancedMonitoringService] = None
        self.state_service: Optional[DistributedStateService] = None
        self.orchestrator: Optional[AdvancedTrainingOrchestrator] = None

        # Service state
        self.is_initialized = False
        self.start_time = time.time()
        self.error_handler = ErrorHandler()
        self.performance_optimizer = PerformanceOptimizer(OptimizationLevel.BALANCED)

    async def initialize(self):
        """Initialize all services"""
        if self.is_initialized:
            return

        logger.info("Initializing Enhanced GRPO Service Manager...")

        try:
            # Initialize core services
            self.monitoring = get_monitoring_service()
            self.state_service = get_state_service()
            self.orchestrator = get_training_orchestrator()

            # Initialize gateway
            self.gateway = EnhancedGRPOGateway(
                enable_metrics=True, enable_security=True
            )

            # Setup security
            if self.gateway.security_manager:
                # Add demo API keys
                self.gateway.security_manager.add_api_key(
                    "demo_admin_key",
                    ["admin", "trainer"],
                    {"description": "Demo admin key"},
                )
                self.gateway.security_manager.add_api_key(
                    "demo_user_key", ["user"], {"description": "Demo user key"}
                )

            # Configure routes
            self._configure_gateway_routes()

            # Setup service instances (for load balancing)
            self._setup_service_instances()

            self.is_initialized = True
            logger.info("Enhanced GRPO Service Manager initialized successfully")

        except Exception as e:
            error_context = self.error_handler.handle_error(
                e, "service_manager", "initialize"
            )
            logger.error(f"Service initialization failed: {error_context.error_id}")
            raise

    def _configure_gateway_routes(self):
        """Configure gateway routes"""
        if not self.gateway:
            return

        routes = [
            RouteConfig(
                path="/api/v2/train",
                methods=["POST"],
                timeout=300.0,
                cache_ttl=0,
                rate_limit=60,
                requires_auth=True,
                allowed_roles=["admin", "trainer"],
            ),
            RouteConfig(
                path="/api/v2/chat",
                methods=["POST"],
                timeout=30.0,
                cache_ttl=300,
                rate_limit=100,
                requires_auth=False,
            ),
            RouteConfig(
                path="/api/v2/jobs/*",
                methods=["GET", "DELETE"],
                timeout=10.0,
                cache_ttl=60,
                rate_limit=200,
                requires_auth=True,
                allowed_roles=["admin", "trainer", "user"],
            ),
            RouteConfig(
                path="/api/v2/health",
                methods=["GET"],
                timeout=5.0,
                cache_ttl=30,
                rate_limit=1000,
            ),
            RouteConfig(
                path="/api/v2/metrics",
                methods=["GET"],
                timeout=10.0,
                cache_ttl=60,
                rate_limit=500,
                requires_auth=True,
                allowed_roles=["admin"],
            ),
        ]

        for route in routes:
            self.gateway.add_route(route)

    def _setup_service_instances(self):
        """Setup service instances for load balancing"""
        if not self.gateway:
            return

        # Add primary instance
        primary_instance = ServiceInstance(
            id="grpo-primary", host="localhost", port=8001, weight=1.0
        )
        self.gateway.add_service_instance(primary_instance)

        # In production, you would add more instances here

    async def shutdown(self):
        """Graceful shutdown of all services"""
        logger.info("Shutting down Enhanced GRPO Service Manager...")

        try:
            # Shutdown orchestrator first (stop training jobs)
            if self.orchestrator:
                await self.orchestrator.shutdown()

            # Shutdown state service
            if self.state_service:
                await self.state_service.shutdown()

            # Gateway cleanup is handled by FastAPI

            logger.info("Service Manager shutdown complete")

        except Exception as e:
            logger.error(f"Shutdown error: {e}")

    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        status = {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime": time.time() - self.start_time,
            "components": {},
            "resource_utilization": {},
            "training": {},
            "api": {},
        }

        try:
            # Check monitoring service
            if self.monitoring:
                monitoring_health = self.monitoring.get_health_summary()
                status["components"]["monitoring"] = monitoring_health["status"]

                if "system_metrics" in monitoring_health:
                    status["resource_utilization"] = monitoring_health["system_metrics"]

            # Check state service
            if self.state_service:
                state_health = await self.state_service.health_check()
                status["components"]["state_service"] = state_health["status"]

            # Check orchestrator
            if self.orchestrator:
                orchestrator_status = await self.orchestrator.get_system_status()
                status["components"]["orchestrator"] = "healthy"
                status["training"] = {
                    "queue_size": orchestrator_status["queue_status"]["queued_jobs"],
                    "running_jobs": orchestrator_status["queue_status"]["running_jobs"],
                    "completed_jobs": orchestrator_status["queue_status"][
                        "completed_jobs"
                    ],
                }

            # Check gateway
            if self.gateway:
                gateway_health = self.gateway.get_health_status()
                status["components"]["gateway"] = gateway_health["status"]
                status["api"] = {
                    "total_requests": gateway_health["stats"]["total_requests"],
                    "error_rate": gateway_health["stats"]["total_errors"]
                    / max(1, gateway_health["stats"]["total_requests"]),
                    "average_response_time": gateway_health["stats"][
                        "average_response_time"
                    ],
                    "cache_hit_rate": gateway_health["stats"]["cache_hit_rate"],
                }

            # Determine overall status
            component_statuses = [
                comp for comp in status["components"].values() if isinstance(comp, str)
            ]
            if any(s == "unhealthy" for s in component_statuses):
                status["status"] = "unhealthy"
            elif any(s == "degraded" for s in component_statuses):
                status["status"] = "degraded"

        except Exception as e:
            status["status"] = "error"
            status["error"] = str(e)

        return status


_service_manager: Optional[ServiceManager] = None


def get_service_manager() -> ServiceManager:
    """Return a lazily-initialized global ServiceManager."""
    global _service_manager
    if _service_manager is None:
        _service_manager = ServiceManager()
    return _service_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager"""
    # Startup
    logger.info("🚀 Starting Enhanced Ultimate GRPO Service")
    await get_service_manager().initialize()

    yield

    # Shutdown
    logger.info("🛑 Shutting down Enhanced Ultimate GRPO Service")
    await get_service_manager().shutdown()


# Create FastAPI app
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Enhanced Ultimate GRPO Service",
        description="Next-Generation AI Agent Training Platform with Advanced Gateway, Monitoring, and Orchestration",
        version="2.1.0",
        lifespan=lifespan,
    )

    # Add middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request dependency for rate limiting and authentication
    async def verify_request(request):
        """Request verification and rate limiting"""
        # This would integrate with the gateway for authentication
        # For now, return a simple user context
        return {"user_id": "demo_user", "roles": ["user"]}

    @app.get("/")
    async def root():
        """Root endpoint with comprehensive service information"""
        return {
            "title": "Enhanced Ultimate GRPO Service",
            "version": "2.1.0",
            "description": "Next-Generation AI Agent Training Platform",
            "status": "operational",
            "features": {
                "🚀 Advanced API Gateway": "Intelligent load balancing, caching, and security",
                "📊 Real-time Monitoring": "Comprehensive metrics, alerts, and observability",
                "🔄 State Management": "Distributed caching with consistency guarantees",
                "🎯 Training Orchestration": "Dynamic resource allocation and job scheduling",
                "🛡️ Enhanced Security": "API key management and threat detection",
                "⚡ Performance Optimization": "Memory management and computational efficiency",
                "🔍 Distributed Tracing": "End-to-end request tracking",
                "📈 Experiment Tracking": "W&B and MLflow integration",
            },
            "endpoints": {
                "training": "/api/v2/train",
                "conversations": "/api/v2/chat",
                "job_management": "/api/v2/jobs",
                "health": "/api/v2/health",
                "metrics": "/api/v2/metrics",
                "system": "/api/v2/system",
                "websocket": "/ws/v2",
            },
            "documentation": {"openapi": "/docs", "redoc": "/redoc"},
        }

    @app.post("/api/v2/train")
    @monitor_async_function("enhanced_grpo_service.train")
    async def train_agent(
        request: EnhancedTrainingRequest,
        background_tasks: BackgroundTasks,
        user_context=Depends(verify_request),
    ):
        """Submit enhanced training job with comprehensive configuration"""
        service_manager = get_service_manager()

        async with managed_state_context() as state_service:
            try:
                # Create training configuration
                training_config = TrainingConfig(
                    experiment_name=request.experiment_name,
                    agent_type=request.agent_type,
                    model_config=request.model_configuration,
                    training_data=request.training_data,
                    num_epochs=request.num_epochs,
                    batch_size=request.batch_size,
                    learning_rate=request.learning_rate,
                    optimizer=request.optimizer,
                    scheduler=request.scheduler,
                    grpo_epsilon=request.grpo_epsilon,
                    grpo_value_loss_coef=request.grpo_value_loss_coef,
                    grpo_entropy_coef=request.grpo_entropy_coef,
                    enable_checkpointing=request.enable_checkpointing,
                    enable_early_stopping=request.enable_early_stopping,
                    enable_wandb=request.enable_wandb,
                    resource_requirements=[
                        ResourceRequirement(ResourceType.CPU, request.cpu_cores),
                        ResourceRequirement(ResourceType.MEMORY, request.memory_gb),
                        ResourceRequirement(ResourceType.GPU, float(request.gpu_count)),
                    ],
                    max_runtime=request.max_runtime_hours * 3600
                    if request.max_runtime_hours
                    else None,
                )

                # Submit job to orchestrator
                job_id = await service_manager.orchestrator.submit_training_job(
                    training_config,
                    priority=request.priority,
                    user_id=user_context["user_id"],
                )

                # Record metrics
                if service_manager.monitoring:
                    service_manager.monitoring.record_training_iteration(
                        request.agent_type, "grpo", {"job_submitted": 1}
                    )

                return {
                    "job_id": job_id,
                    "status": "submitted",
                    "experiment_name": request.experiment_name,
                    "estimated_start_time": time.time() + 60,  # Estimate
                    "configuration": {
                        "agent_type": request.agent_type,
                        "num_epochs": request.num_epochs,
                        "batch_size": request.batch_size,
                        "resources": {
                            "cpu_cores": request.cpu_cores,
                            "memory_gb": request.memory_gb,
                            "gpu_count": request.gpu_count,
                        },
                    },
                    "tracking": {
                        "experiment_url": f"https://wandb.ai/project/{request.experiment_name}"
                        if request.enable_wandb
                        else None
                    },
                }

            except Exception as e:
                if service_manager.monitoring:
                    service_manager.monitoring.record_error(
                        "training", type(e).__name__, e
                    )
                raise HTTPException(
                    status_code=500, detail=f"Training submission failed: {str(e)}"
                )

    @app.post("/api/v2/chat")
    @monitor_async_function("enhanced_grpo_service.chat")
    async def enhanced_chat(
        request: EnhancedConversationRequest, user_context=Depends(verify_request)
    ):
        """Enhanced conversational interface with streaming support"""
        service_manager = get_service_manager()

        async with managed_state_context() as state_service:
            try:
                conversation_manager = state_service.conversation_manager

                # Handle conversation context
                if request.conversation_id:
                    conversation = await conversation_manager.get_conversation(
                        request.conversation_id
                    )
                    if not conversation:
                        raise HTTPException(
                            status_code=404, detail="Conversation not found"
                        )
                else:
                    # Create new conversation
                    conversation_id = str(uuid.uuid4())
                    await conversation_manager.create_conversation(
                        conversation_id,
                        request.user_id or user_context["user_id"],
                        request.context,
                    )
                    request.conversation_id = conversation_id

                # Add user turn
                await conversation_manager.add_turn(
                    request.conversation_id, "user", request.message
                )

                # Generate response (placeholder - would use actual model)
                response_text = f"Enhanced response to: {request.message[:50]}... (Strategy: {request.strategy})"

                # Add assistant turn
                await conversation_manager.add_turn(
                    request.conversation_id,
                    "assistant",
                    response_text,
                    {
                        "strategy": request.strategy,
                        "temperature": request.temperature,
                        "max_tokens": request.max_tokens,
                    },
                )

                # Get updated conversation
                conversation = await conversation_manager.get_conversation(
                    request.conversation_id
                )

                # Record metrics
                if service_manager.monitoring:
                    service_manager.monitoring.record_request(
                        "POST", "/api/v2/chat", 200, 0.5
                    )

                return {
                    "conversation_id": request.conversation_id,
                    "response": response_text,
                    "context": conversation["context"],
                    "turn_count": len(conversation["turns"]),
                    "metadata": {
                        "strategy": request.strategy,
                        "response_time": 0.5,
                        "tokens_generated": len(response_text.split()),
                        "model_version": "enhanced-v2.1",
                    },
                }

            except Exception as e:
                if service_manager.monitoring:
                    service_manager.monitoring.record_error("chat", type(e).__name__, e)
                raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

    @app.get("/api/v2/jobs/{job_id}")
    @monitor_async_function("enhanced_grpo_service.get_job_status")
    async def get_job_status(job_id: str, user_context=Depends(verify_request)):
        """Get detailed training job status"""
        service_manager = get_service_manager()

        try:
            status = await service_manager.orchestrator.get_job_status(job_id)
            if not status:
                raise HTTPException(status_code=404, detail="Job not found")

            return {
                **status,
                "links": {
                    "cancel": f"/api/v2/jobs/{job_id}/cancel",
                    "logs": f"/api/v2/jobs/{job_id}/logs",
                    "artifacts": f"/api/v2/jobs/{job_id}/artifacts",
                },
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/v2/jobs/{job_id}")
    @monitor_async_function("enhanced_grpo_service.cancel_job")
    async def cancel_job(job_id: str, user_context=Depends(verify_request)):
        """Cancel a training job"""
        service_manager = get_service_manager()

        try:
            success = await service_manager.orchestrator.cancel_job(job_id)
            if not success:
                raise HTTPException(
                    status_code=404, detail="Job not found or cannot be cancelled"
                )

            return {
                "job_id": job_id,
                "status": "cancelled",
                "cancelled_at": time.time(),
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v2/health", response_model=SystemHealthResponse)
    @monitor_async_function("enhanced_grpo_service.health_check")
    async def health_check():
        """Comprehensive system health check"""
        service_manager = get_service_manager()

        try:
            health_status = await service_manager.get_health_status()

            return SystemHealthResponse(
                status=health_status["status"],
                timestamp=health_status["timestamp"],
                uptime=health_status["uptime"],
                components=health_status["components"],
                resource_utilization=health_status.get("resource_utilization", {}),
                active_jobs=health_status.get("training", {}).get("running_jobs", 0),
                queue_size=health_status.get("training", {}).get("queue_size", 0),
                error_rate=health_status.get("api", {}).get("error_rate", 0.0),
            )

        except Exception as e:
            return SystemHealthResponse(
                status="error",
                timestamp=time.time(),
                uptime=time.time() - service_manager.start_time,
                components={"error": str(e)},
                resource_utilization={},
                active_jobs=0,
                queue_size=0,
                error_rate=1.0,
            )

    @app.get("/api/v2/metrics", response_model=MetricsResponse)
    @monitor_async_function("enhanced_grpo_service.get_metrics")
    async def get_metrics(user_context=Depends(verify_request)):
        """Get comprehensive system metrics"""
        service_manager = get_service_manager()

        try:
            current_time = time.time()

            # Get monitoring dashboard
            dashboard = {}
            if service_manager.monitoring:
                dashboard = service_manager.monitoring.get_metrics_dashboard()

            # Get orchestrator metrics
            orchestrator_status = {}
            if service_manager.orchestrator:
                orchestrator_status = (
                    await service_manager.orchestrator.get_system_status()
                )

            # Get gateway metrics
            gateway_stats = {}
            if service_manager.gateway:
                gateway_health = service_manager.gateway.get_health_status()
                gateway_stats = gateway_health.get("stats", {})

            return MetricsResponse(
                timestamp=current_time,
                system_metrics=dashboard.get("system", {}),
                training_metrics={
                    "queue_size": orchestrator_status.get("queue_status", {}).get(
                        "queued_jobs", 0
                    ),
                    "running_jobs": orchestrator_status.get("queue_status", {}).get(
                        "running_jobs", 0
                    ),
                    "completed_jobs": orchestrator_status.get("queue_status", {}).get(
                        "completed_jobs", 0
                    ),
                    **orchestrator_status.get("resource_utilization", {}),
                },
                api_metrics={
                    "total_requests": gateway_stats.get("total_requests", 0),
                    "error_rate": gateway_stats.get("total_errors", 0)
                    / max(1, gateway_stats.get("total_requests", 1)),
                    "average_response_time": gateway_stats.get(
                        "average_response_time", 0
                    ),
                    "cache_hit_rate": gateway_stats.get("cache_hit_rate", 0),
                    "active_connections": gateway_stats.get("active_connections", 0),
                },
                alerts=dashboard.get("alerts", []),
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v2/system/status")
    async def get_system_status(user_context=Depends(verify_request)):
        """Get detailed system status"""
        service_manager = get_service_manager()

        try:
            return {
                "service_manager": await service_manager.get_health_status(),
                "orchestrator": await service_manager.orchestrator.get_system_status()
                if service_manager.orchestrator
                else None,
                "gateway": service_manager.gateway.get_health_status()
                if service_manager.gateway
                else None,
                "state_service": await service_manager.state_service.health_check()
                if service_manager.state_service
                else None,
                "monitoring": service_manager.monitoring.get_health_summary()
                if service_manager.monitoring
                else None,
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.websocket("/ws/v2")
    async def websocket_endpoint(websocket: WebSocket):
        """Enhanced WebSocket endpoint with real-time updates"""
        service_manager = get_service_manager()
        await websocket.accept()

        try:
            while True:
                # Receive message
                data = await websocket.receive_text()
                message_data = json.loads(data)

                message_type = message_data.get("type")

                if message_type == "health":
                    # Send health status
                    health = await service_manager.get_health_status()
                    await websocket.send_text(
                        json.dumps({"type": "health_response", "data": health})
                    )

                elif message_type == "metrics":
                    # Send real-time metrics
                    if service_manager.monitoring:
                        dashboard = service_manager.monitoring.get_metrics_dashboard()
                        await websocket.send_text(
                            json.dumps({"type": "metrics_response", "data": dashboard})
                        )

                elif message_type == "job_status":
                    # Send job status updates
                    job_id = message_data.get("job_id")
                    if job_id and service_manager.orchestrator:
                        status = await service_manager.orchestrator.get_job_status(
                            job_id
                        )
                        await websocket.send_text(
                            json.dumps({"type": "job_status_response", "data": status})
                        )

                elif message_type == "ping":
                    # Ping/pong
                    await websocket.send_text(
                        json.dumps({"type": "pong", "timestamp": time.time()})
                    )

                else:
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "error",
                                "message": f"Unknown message type: {message_type}",
                            }
                        )
                    )

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await websocket.close()


def main():
    """Main entry point for the enhanced service"""
    if not FASTAPI_AVAILABLE:
        logger.error("FastAPI is required but not available")
        return

    print("\n" + "=" * 100)
    print("🚀 ENHANCED ULTIMATE GRPO SERVICE - NEXT-GENERATION AI TRAINING PLATFORM")
    print("=" * 100)
    print("\n🎯 REVOLUTIONARY FEATURES:")
    print("  ⚡ Intelligent API Gateway with Multi-Strategy Load Balancing")
    print("  📊 Real-time Monitoring with Prometheus & Distributed Tracing")
    print("  🔄 Advanced State Management with Redis & Consistency Guarantees")
    print("  🎯 Dynamic Training Orchestration with Resource-Aware Scheduling")
    print("  🛡️ Enterprise Security with API Key Management & Threat Detection")
    print("  📈 Comprehensive Experiment Tracking with W&B & MLflow Integration")
    print("  ⚡ Performance Optimization with Memory Management & Auto-scaling")
    print("  🔍 Distributed Tracing with OpenTelemetry & Jaeger")
    print("  📱 Real-time WebSocket API for Live Updates")
    print("  🔧 Fault-Tolerant Design with Circuit Breakers & Auto-Recovery")
    print("\n🔧 API ENDPOINTS:")
    print("  📋 Training:      POST /api/v2/train")
    print("  💬 Chat:          POST /api/v2/chat")
    print("  📊 Job Status:    GET  /api/v2/jobs/{id}")
    print("  🏥 Health:        GET  /api/v2/health")
    print("  📈 Metrics:       GET  /api/v2/metrics")
    print("  🔧 System:        GET  /api/v2/system/status")
    print("  🌐 WebSocket:     WS   /ws/v2")
    print("\n📚 DOCUMENTATION:")
    print("  🔍 Interactive:   http://localhost:8002/docs")
    print("  📖 ReDoc:         http://localhost:8002/redoc")
    print("=" * 100 + "\n")

    # Start the enhanced service
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info", access_log=True)


if __name__ == "__main__":
    main()
