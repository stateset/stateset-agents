"""
API Middleware Module

Centralized middleware for security, observability, and request handling.
"""

import time
import uuid
import logging
from typing import Callable, Optional, Dict, Any, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from fastapi import FastAPI

from .config import get_config, APIConfig
from .constants import (
    SECURITY_HEADERS,
    HSTS_HEADER,
    HEADER_REQUEST_ID,
    HEADER_CORRELATION_ID,
    HEADER_RESPONSE_TIME,
    HEADER_RATE_LIMIT,
    HEADER_RATE_LIMIT_REMAINING,
    HEADER_RATE_LIMIT_RESET,
    HEADER_RETRY_AFTER,
    RATE_LIMIT_DEQUE_MAXLEN,
    PERCENTILE_P50,
    PERCENTILE_P95,
    PERCENTILE_P99,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Security Headers Middleware
# ============================================================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add comprehensive security headers to all responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Add all security headers from constants
        for header, value in SECURITY_HEADERS.items():
            response.headers[header] = value

        # Add HSTS in production
        config = get_config()
        if config.is_production():
            response.headers["Strict-Transport-Security"] = HSTS_HEADER

        return response


# ============================================================================
# Request Context Middleware
# ============================================================================

@dataclass
class RequestContext:
    """Request context with tracing information."""
    request_id: str
    start_time: float
    client_ip: str
    user_agent: str
    path: str
    method: str
    user_id: Optional[str] = None
    roles: list = field(default_factory=list)
    api_key: Optional[str] = None

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (time.monotonic() - self.start_time) * 1000


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach request context and trace ID to all requests."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get or generate request ID
        request_id = (
            request.headers.get(HEADER_REQUEST_ID) or
            request.headers.get(HEADER_CORRELATION_ID) or
            str(uuid.uuid4())
        )

        # Create request context
        context = RequestContext(
            request_id=request_id,
            start_time=time.monotonic(),
            client_ip=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("User-Agent", "unknown"),
            path=str(request.url.path),
            method=request.method,
        )

        # Attach to request state
        request.state.context = context
        request.state.request_id = request_id

        # Process request
        response = await call_next(request)

        # Add tracing headers to response
        response.headers[HEADER_REQUEST_ID] = request_id
        response.headers[HEADER_RESPONSE_TIME] = f"{context.elapsed_ms():.2f}"

        return response


# ============================================================================
# Rate Limiting Middleware
# ============================================================================

class SlidingWindowRateLimiter:
    """Thread-safe sliding window rate limiter."""

    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self.windows: Dict[str, deque] = defaultdict(deque)

    def is_allowed(self, key: str, limit: int) -> Tuple[bool, int]:
        """
        Check if request is allowed under rate limit.

        Returns:
            Tuple of (allowed, remaining_requests)
        """
        now = time.monotonic()
        window = self.windows[key]
        cutoff = now - self.window_seconds

        # Remove expired entries
        while window and window[0] <= cutoff:
            window.popleft()

        remaining = max(0, limit - len(window))

        if len(window) >= limit:
            return False, 0

        window.append(now)
        return True, remaining - 1

    def reset(self, key: str) -> None:
        """Reset rate limit for a key."""
        self.windows.pop(key, None)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware with sliding window algorithm."""

    def __init__(self, app: FastAPI, limiter: Optional[SlidingWindowRateLimiter] = None):
        super().__init__(app)
        if limiter is not None:
            self.limiter = limiter
        else:
            config = get_config()
            self.limiter = SlidingWindowRateLimiter(
                window_seconds=config.rate_limit.window_seconds
            )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        config = get_config()

        if not config.rate_limit.enabled:
            return await call_next(request)

        # Skip rate limiting for health checks
        if request.url.path in ("/health", "/healthz", "/ready"):
            return await call_next(request)

        # Get rate limit key (API key or IP)
        api_key = self._extract_api_key(request)
        client_ip = request.client.host if request.client else "unknown"
        limit_key = api_key or client_ip

        # Check rate limit
        allowed, remaining = self.limiter.is_allowed(
            limit_key,
            config.rate_limit.requests_per_minute
        )

        if not allowed:
            logger.warning(
                "Rate limit exceeded",
                extra={
                    "client_ip": client_ip,
                    "path": request.url.path,
                    "limit_key": limit_key[:16] + "..." if api_key else limit_key,
                }
            )

            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": "Too many requests. Please retry after a short delay.",
                    },
                    "request_id": getattr(request.state, "request_id", str(uuid.uuid4())),
                    "retry_after_seconds": config.rate_limit.window_seconds,
                },
                headers={
                    HEADER_RETRY_AFTER: str(config.rate_limit.window_seconds),
                    HEADER_RATE_LIMIT: str(config.rate_limit.requests_per_minute),
                    HEADER_RATE_LIMIT_REMAINING: "0",
                    HEADER_RATE_LIMIT_RESET: str(int(time.time()) + config.rate_limit.window_seconds),
                }
            )

        # Add rate limit headers to response
        response = await call_next(request)
        response.headers[HEADER_RATE_LIMIT] = str(config.rate_limit.requests_per_minute)
        response.headers[HEADER_RATE_LIMIT_REMAINING] = str(remaining)
        response.headers[HEADER_RATE_LIMIT_RESET] = str(int(time.time()) + config.rate_limit.window_seconds)

        return response

    def _extract_api_key(self, request: Request) -> Optional[str]:
        """Extract API key from request headers."""
        # Check Authorization header
        auth_header = request.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer "):
            return auth_header.split(" ", 1)[1].strip()

        # Check X-API-Key header
        api_key = request.headers.get("X-API-Key")
        return api_key.strip() if api_key else None


# ============================================================================
# Metrics Middleware
# ============================================================================

@dataclass
class APIMetrics:
    """API metrics collector."""
    request_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    status_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    latencies: deque = field(default_factory=lambda: deque(maxlen=RATE_LIMIT_DEQUE_MAXLEN))
    error_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    rate_limit_hits: int = 0

    def record_request(self, path: str, method: str, status_code: int, latency_ms: float) -> None:
        """Record a request."""
        key = f"{method}:{path}"
        self.request_counts[key] += 1
        self.status_counts[status_code] += 1
        self.latencies.append(latency_ms)

        if status_code >= 400:
            self.error_counts[key] += 1

    def record_rate_limit_hit(self) -> None:
        """Record a rate limit hit."""
        self.rate_limit_hits += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        latencies_list = list(self.latencies)

        if latencies_list:
            avg_latency = sum(latencies_list) / len(latencies_list)
            sorted_latencies = sorted(latencies_list)
            p50_index = int(PERCENTILE_P50 * (len(sorted_latencies) - 1))
            p95_index = int(PERCENTILE_P95 * (len(sorted_latencies) - 1))
            p99_index = int(PERCENTILE_P99 * (len(sorted_latencies) - 1))
        else:
            avg_latency = 0
            sorted_latencies = []
            p50_index = p95_index = p99_index = 0

        return {
            "total_requests": sum(self.request_counts.values()),
            "requests_by_endpoint": dict(self.request_counts),
            "status_codes": dict(self.status_counts),
            "errors_by_endpoint": dict(self.error_counts),
            "rate_limit_hits": self.rate_limit_hits,
            "latency": {
                "avg_ms": round(avg_latency, 2),
                "p50_ms": round(sorted_latencies[p50_index], 2) if sorted_latencies else 0,
                "p95_ms": round(sorted_latencies[p95_index], 2) if sorted_latencies else 0,
                "p99_ms": round(sorted_latencies[p99_index], 2) if sorted_latencies else 0,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }


# Global metrics instance
_metrics = APIMetrics()


def get_metrics() -> APIMetrics:
    """Get the global metrics instance."""
    return _metrics


class MetricsMiddleware(BaseHTTPMiddleware):
    """Collect API metrics."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.monotonic()

        response = await call_next(request)

        # Record metrics
        latency_ms = (time.monotonic() - start_time) * 1000
        _metrics.record_request(
            path=request.url.path,
            method=request.method,
            status_code=response.status_code,
            latency_ms=latency_ms,
        )

        return response


# ============================================================================
# Request Logging Middleware
# ============================================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Structured request/response logging."""

    # Paths to exclude from logging
    EXCLUDED_PATHS = {"/health", "/healthz", "/ready", "/metrics"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip logging for health checks and metrics
        if request.url.path in self.EXCLUDED_PATHS:
            return await call_next(request)

        context = getattr(request.state, "context", None)
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        # Log request
        logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": context.client_ip if context else "unknown",
                "user_agent": context.user_agent if context else "unknown",
            }
        )

        response = await call_next(request)

        # Log response
        elapsed_ms = context.elapsed_ms() if context else 0
        log_level = logging.WARNING if response.status_code >= 400 else logging.INFO

        logger.log(
            log_level,
            "Request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "elapsed_ms": round(elapsed_ms, 2),
            }
        )

        return response


# ============================================================================
# Setup Function
# ============================================================================

def setup_middleware(app: FastAPI) -> None:
    """Set up all middleware in the correct order."""
    config = get_config()

    # Order matters! Middleware is executed in reverse order of addition
    # So the last added middleware runs first

    # 1. Request logging (runs last, sees final response)
    app.add_middleware(RequestLoggingMiddleware)

    # 2. Metrics collection
    app.add_middleware(MetricsMiddleware)

    # 3. Rate limiting
    if config.rate_limit.enabled:
        app.add_middleware(RateLimitMiddleware)

    # 4. Request context (runs early, sets up tracing)
    app.add_middleware(RequestContextMiddleware)

    # 5. Security headers (runs first, adds headers to all responses)
    app.add_middleware(SecurityHeadersMiddleware)

    logger.info("Middleware configured successfully")
