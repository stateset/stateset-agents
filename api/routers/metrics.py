import asyncio
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict
from ..schemas import MetricsResponse, HealthResponse, ErrorResponse
from ..dependencies import require_role, get_security_monitor
from utils.performance_monitor import get_global_monitor
from utils.security import SecurityMonitor

router = APIRouter(tags=["observability"])

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the health status of the API service",
)
async def health_check():
    """Get service health status."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().timestamp(),
        version="2.0.0",
        uptime=asyncio.get_event_loop().time(),
        components={
            "agent_service": "healthy",
            "training_service": "healthy",
            "security_monitor": "healthy",
        },
    )

@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Get Metrics",
    description="Get system and performance metrics",
    responses={
        200: {"description": "Metrics retrieved successfully"},
        401: {"description": "Authentication required", "model": ErrorResponse},
    },
)
async def get_metrics(
    user_data: Dict = Depends(require_role("admin")),
    monitor: SecurityMonitor = Depends(get_security_monitor)
):
    """Get system metrics."""
    try:
        perf_monitor = get_global_monitor()
        summary = perf_monitor.get_metrics_summary()

        return MetricsResponse(
            timestamp=datetime.utcnow().timestamp(),
            system_metrics={
                "total_operations": summary.get("total_operations", 0),
                "active_operations": len(perf_monitor._active_operations),
            },
            performance_metrics={
                "avg_response_time": summary.get("operations_by_type", {})
                .get("agent_response", {})
                .get("avg_duration", 0)
            },
            security_events=len(monitor.events),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get metrics")
