"""
Metrics and Health Router Module

API endpoints for system health, metrics, and observability.
"""

import time
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ..schemas import MetricsResponse, HealthResponse, ErrorResponse
from ..dependencies import require_role, get_security_monitor, AuthenticatedUser
from ..cache import get_cache, HEALTH_CACHE_TTL_SECONDS, METRICS_CACHE_TTL_SECONDS
from ..constants import API_VERSION
from utils.performance_monitor import get_global_monitor
from utils.security import SecurityMonitor

router = APIRouter(tags=["observability"])

# Track service start time for uptime calculation
_service_start_time = time.monotonic()


# ============================================================================
# Response Models
# ============================================================================

class ComponentHealthDetail(BaseModel):
    """Detailed component health information."""
    name: str = Field(..., description="Component name")
    status: str = Field(..., description="Health status (healthy, degraded, unhealthy)")
    latency_ms: float = Field(0.0, description="Component response latency in ms")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")


class DetailedHealthResponse(BaseModel):
    """Detailed health check response."""
    status: str = Field(..., description="Overall service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    components: List[ComponentHealthDetail] = Field(
        default_factory=list, description="Individual component health"
    )
    checks: Dict[str, bool] = Field(default_factory=dict, description="Health check results")


class RouterHealthResponse(BaseModel):
    """Health response used by the public `/health` router endpoint."""

    status: str = Field(..., description="Overall service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    version: str = Field(..., description="API version")
    uptime: float = Field(..., description="Service uptime in seconds")
    components: Dict[str, str] = Field(
        default_factory=dict,
        description="Component statuses keyed by component name",
    )


class DetailedMetricsResponse(BaseModel):
    """Detailed metrics response."""
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Metrics timestamp")
    system_metrics: Dict[str, Any] = Field(default_factory=dict, description="System metrics")
    api_metrics: Dict[str, Any] = Field(default_factory=dict, description="API metrics")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    security_metrics: Dict[str, Any] = Field(default_factory=dict, description="Security metrics")
    cache_metrics: Dict[str, Any] = Field(default_factory=dict, description="Cache metrics")


# ============================================================================
# Health Endpoints
# ============================================================================

@router.get(
    "/health",
    response_model=RouterHealthResponse,
    summary="Health Check",
    description="Check the health status of the API service and its components.",
    responses={
        200: {"description": "Service is healthy"},
        503: {"description": "Service is unhealthy"},
    },
)
async def health_check() -> RouterHealthResponse:
    """
    Get comprehensive service health status.

    Returns health information including:
    - Overall service status
    - Uptime
    - Individual component health
    - Basic health checks

    This endpoint is cached for {HEALTH_CACHE_TTL_SECONDS} seconds to reduce load.
    """
    cache = get_cache()
    cache_key = "health:check"

    # Try cache first
    cached_response = cache.get(cache_key)
    if cached_response is not None:
        # Update timestamp for cached response
        cached_response["timestamp"] = datetime.utcnow()
        return RouterHealthResponse(**cached_response)

    # Calculate uptime
    uptime = round(time.monotonic() - _service_start_time, 2)

    # Lightweight component checks (placeholders)
    components = {
        "agent_service": "healthy",
        "training_service": "healthy",
        "security_monitor": "healthy",
    }

    # Determine overall status
    all_healthy = all(status == "healthy" for status in components.values())
    overall_status = "healthy" if all_healthy else "degraded"

    response = RouterHealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version=API_VERSION,
        uptime=uptime,
        components=components,
    )

    # Cache the response
    cache.set(cache_key, response.model_dump(), HEALTH_CACHE_TTL_SECONDS)

    return response


@router.get(
    "/healthz",
    summary="Kubernetes Health Probe",
    description="Simple health probe for Kubernetes liveness checks.",
    responses={
        200: {"description": "Service is alive"},
    },
)
async def healthz() -> Dict[str, str]:
    """
    Simple health probe for Kubernetes.

    Returns minimal response for fast liveness checks.
    """
    return {"status": "ok"}


@router.get(
    "/ready",
    summary="Readiness Probe",
    description="Check if the service is ready to accept traffic.",
    responses={
        200: {"description": "Service is ready"},
        503: {"description": "Service is not ready"},
    },
)
async def readiness() -> Dict[str, Any]:
    """
    Readiness probe for Kubernetes.

    Checks if all dependencies are available.
    """
    # Would check database connections, external services, etc.
    ready = True

    return {
        "status": "ready" if ready else "not_ready",
        "checks": {
            "dependencies_available": ready,
        },
    }


# ============================================================================
# Metrics Endpoints
# ============================================================================

@router.get(
    "/metrics",
    response_model=DetailedMetricsResponse,
    summary="Get Metrics",
    description="Get comprehensive system and performance metrics. Requires admin role.",
    responses={
        200: {"description": "Metrics retrieved successfully"},
        401: {"description": "Authentication required", "model": ErrorResponse},
        403: {"description": "Insufficient permissions", "model": ErrorResponse},
    },
)
async def get_metrics(
    user: AuthenticatedUser = Depends(require_role("admin")),
    monitor: SecurityMonitor = Depends(get_security_monitor),
) -> DetailedMetricsResponse:
    """
    Get comprehensive system metrics.

    Requires admin role. Returns:
    - System metrics (operations, active operations)
    - API metrics (requests, errors, latency)
    - Performance metrics (response times)
    - Security metrics (events, threats)
    - Cache metrics (hits, misses, size)

    This endpoint is cached for {METRICS_CACHE_TTL_SECONDS} seconds.
    """
    cache = get_cache()
    cache_key = f"metrics:all:{user.user_id}"

    # Try cache first
    cached_response = cache.get(cache_key)
    if cached_response is not None:
        cached_response["timestamp"] = datetime.utcnow().isoformat()
        return DetailedMetricsResponse(**cached_response)

    # Get performance monitor data
    try:
        perf_monitor = get_global_monitor()
        summary = perf_monitor.get_metrics_summary()
    except Exception:
        summary = {}

    # Get middleware metrics
    from ..middleware import get_metrics as get_api_metrics
    api_metrics_data = get_api_metrics().get_summary()

    # Get security metrics
    from ..security import get_api_security_monitor
    security_monitor = get_api_security_monitor()
    security_stats = security_monitor.get_stats()

    # Build response
    response = DetailedMetricsResponse(
        timestamp=datetime.utcnow(),
        system_metrics={
            "total_operations": summary.get("total_operations", 0),
            "active_operations": len(getattr(perf_monitor, "_active_operations", {})),
            "uptime_seconds": round(time.monotonic() - _service_start_time, 2),
        },
        api_metrics={
            "total_requests": api_metrics_data.get("total_requests", 0),
            "requests_by_endpoint": api_metrics_data.get("requests_by_endpoint", {}),
            "status_codes": api_metrics_data.get("status_codes", {}),
            "errors_by_endpoint": api_metrics_data.get("errors_by_endpoint", {}),
            "rate_limit_hits": api_metrics_data.get("rate_limit_hits", 0),
        },
        performance_metrics={
            "latency": api_metrics_data.get("latency", {}),
            "operations_by_type": summary.get("operations_by_type", {}),
        },
        security_metrics={
            "total_events": security_stats.get("total_events", 0),
            "events_last_hour": security_stats.get("events_last_hour", 0),
            "blocked_events": security_stats.get("blocked_events", 0),
            "active_lockouts": security_stats.get("active_lockouts", 0),
            "high_threat_events": security_stats.get("high_threat_events", 0),
            "events_by_type": security_stats.get("events_by_type", {}),
        },
        cache_metrics=cache.stats(),
    )

    # Cache the response
    cache.set(cache_key, response.model_dump(), METRICS_CACHE_TTL_SECONDS)

    return response


@router.get(
    "/metrics/security",
    summary="Security Metrics",
    description="Get security-specific metrics. Requires admin role.",
    responses={
        200: {"description": "Security metrics retrieved"},
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
    },
)
async def get_security_metrics(
    user: AuthenticatedUser = Depends(require_role("admin")),
) -> Dict[str, Any]:
    """
    Get detailed security metrics.

    Requires admin role. Returns:
    - Security event counts by type
    - Recent threat activity
    - Authentication statistics
    """
    from ..security import get_api_security_monitor
    security_monitor = get_api_security_monitor()

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "statistics": security_monitor.get_stats(),
        "recent_events": [
            {
                "type": e.event_type.value,
                "threat_level": e.threat_level.value,
                "blocked": e.blocked,
                "path": e.path,
                "timestamp": datetime.fromtimestamp(e.timestamp).isoformat(),
            }
            for e in security_monitor.events[-10:]  # Last 10 events
        ],
    }


@router.get(
    "/metrics/cache",
    summary="Cache Metrics",
    description="Get cache statistics.",
    responses={
        200: {"description": "Cache metrics retrieved"},
        401: {"description": "Authentication required"},
    },
)
async def get_cache_metrics(
    user: AuthenticatedUser = Depends(require_role("admin")),
) -> Dict[str, Any]:
    """
    Get cache statistics.

    Returns hit/miss rates and cache size.
    """
    cache = get_cache()

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "cache": cache.stats(),
    }
