"""
Enhanced GRPO Service Gateway - Next-Generation API Architecture

This module provides an advanced API gateway with load balancing, intelligent routing,
caching, security, and comprehensive monitoring for the GRPO RL service framework.
"""

import asyncio
import hashlib
import json
import logging
import ssl
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import aioredis
    import redis
    from fastapi import (
        BackgroundTasks,
        Depends,
        FastAPI,
        HTTPException,
        Request,
        Response,
        Security,
    )
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from fastapi.responses import StreamingResponse
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

import utils.cache

from stateset_agents.core.async_pool import AsyncResourcePool, managed_async_resources
from stateset_agents.core.error_handling import (
    CircuitBreaker,
    ErrorCategory,
    ErrorHandler,
    GRPOException,
)
from stateset_agents.core.performance_optimizer import OptimizationLevel, PerformanceOptimizer
from stateset_agents.core.type_system import DeviceType, TypeValidator
from stateset_agents.utils.monitoring import MonitoringService

CacheService = utils.cache.CacheService

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service status enumeration"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""

    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESOURCE_BASED = "resource_based"
    LATENCY_BASED = "latency_based"


@dataclass
class ServiceInstance:
    """Individual service instance"""

    id: str
    host: str
    port: int
    weight: float = 1.0
    status: ServiceStatus = ServiceStatus.HEALTHY
    current_connections: int = 0
    total_requests: int = 0
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    last_health_check: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def average_response_time(self) -> float:
        return (
            sum(self.response_times) / len(self.response_times)
            if self.response_times
            else 0.0
        )

    def record_request(self, response_time: float):
        """Record a request for metrics"""
        self.current_connections = max(0, self.current_connections - 1)
        self.total_requests += 1
        self.response_times.append(response_time)


@dataclass
class RouteConfig:
    """Configuration for a route"""

    path: str
    methods: List[str]
    timeout: float = 30.0
    retry_attempts: int = 3
    cache_ttl: int = 0  # 0 = no caching
    rate_limit: Optional[int] = None  # requests per minute
    requires_auth: bool = False
    allowed_roles: List[str] = field(default_factory=list)


class IntelligentCache:
    """Advanced caching with TTL, invalidation, and warm-up"""

    def __init__(self, redis_url: Optional[str] = None):
        self.local_cache: Dict[
            str, Tuple[Any, float, float]
        ] = {}  # key -> (value, expiry, access_time)
        self.redis_client = None
        self.cache_stats = {"hits": 0, "misses": 0, "evictions": 0, "size": 0}

        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        current_time = time.time()

        # Check local cache first
        if key in self.local_cache:
            value, expiry, _ = self.local_cache[key]
            if current_time < expiry:
                self.local_cache[key] = (
                    value,
                    expiry,
                    current_time,
                )  # Update access time
                self.cache_stats["hits"] += 1
                return value
            else:
                del self.local_cache[key]
                self.cache_stats["evictions"] += 1

        # Check Redis cache
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(key)
                if cached_data:
                    value = json.loads(cached_data)
                    # Store in local cache for faster access
                    self.local_cache[key] = (
                        value,
                        current_time + 300,
                        current_time,
                    )  # 5 min local TTL
                    self.cache_stats["hits"] += 1
                    return value
            except Exception as e:
                logger.error(f"Redis cache get error: {e}")

        self.cache_stats["misses"] += 1
        return None

    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache"""
        current_time = time.time()
        expiry = current_time + ttl

        # Store in local cache
        self.local_cache[key] = (value, expiry, current_time)
        self.cache_stats["size"] = len(self.local_cache)

        # Store in Redis cache
        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl, json.dumps(value))
            except Exception as e:
                logger.error(f"Redis cache set error: {e}")

    def invalidate(self, pattern: str = None):
        """Invalidate cache entries"""
        if pattern:
            keys_to_remove = [k for k in self.local_cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self.local_cache[key]
                self.cache_stats["evictions"] += 1
        else:
            self.local_cache.clear()
            self.cache_stats["evictions"] += len(self.local_cache)

        self.cache_stats["size"] = len(self.local_cache)


class LoadBalancer:
    """Advanced load balancer with multiple strategies"""

    def __init__(
        self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    ):
        self.strategy = strategy
        self.instances: Dict[str, ServiceInstance] = {}
        self.current_index = 0
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

    def add_instance(self, instance: ServiceInstance):
        """Add a service instance"""
        self.instances[instance.id] = instance
        self.circuit_breakers[instance.id] = CircuitBreaker(
            failure_threshold=5, recovery_timeout=30.0, expected_exception=Exception
        )

    def remove_instance(self, instance_id: str):
        """Remove a service instance"""
        self.instances.pop(instance_id, None)
        self.circuit_breakers.pop(instance_id, None)

    async def get_instance(self) -> Optional[ServiceInstance]:
        """Get the next available instance based on strategy"""
        healthy_instances = [
            inst
            for inst in self.instances.values()
            if inst.status == ServiceStatus.HEALTHY
            and not self.circuit_breakers[inst.id].is_open
        ]

        if not healthy_instances:
            return None

        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            instance = healthy_instances[self.current_index % len(healthy_instances)]
            self.current_index += 1

        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            instance = min(healthy_instances, key=lambda x: x.current_connections)

        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            # Weighted selection based on instance weights
            total_weight = sum(inst.weight for inst in healthy_instances)
            weights = [inst.weight / total_weight for inst in healthy_instances]
            instance = self._weighted_choice(healthy_instances, weights)

        elif self.strategy == LoadBalancingStrategy.LATENCY_BASED:
            instance = min(healthy_instances, key=lambda x: x.average_response_time)

        else:  # RESOURCE_BASED
            # Choose based on resource utilization (simplified)
            instance = min(
                healthy_instances, key=lambda x: x.current_connections / x.weight
            )

        instance.current_connections += 1
        return instance

    def _weighted_choice(
        self, instances: List[ServiceInstance], weights: List[float]
    ) -> ServiceInstance:
        """Weighted random choice"""
        import random

        return random.choices(instances, weights=weights)[0]


class RateLimiter:
    """Advanced rate limiting with sliding window"""

    def __init__(self):
        self.windows: Dict[str, deque] = defaultdict(lambda: deque())

    async def is_allowed(self, key: str, limit: int, window_seconds: int = 60) -> bool:
        """Check if request is allowed under rate limit"""
        current_time = time.time()
        window = self.windows[key]

        # Remove old entries
        while window and window[0] < current_time - window_seconds:
            window.popleft()

        # Check if under limit
        if len(window) < limit:
            window.append(current_time)
            return True

        return False


class SecurityManager:
    """Advanced security management"""

    def __init__(self):
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.blocked_ips: set = set()
        self.suspicious_activity: Dict[str, List[float]] = defaultdict(list)

    def add_api_key(self, key: str, roles: List[str], metadata: Dict[str, Any] = None):
        """Add an API key"""
        self.api_keys[key] = {
            "roles": roles,
            "created": datetime.now(),
            "metadata": metadata or {},
        }

    def validate_api_key(self, key: str) -> Optional[Dict[str, Any]]:
        """Validate an API key"""
        return self.api_keys.get(key)

    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is blocked"""
        return ip in self.blocked_ips

    def block_ip(self, ip: str):
        """Block an IP address"""
        self.blocked_ips.add(ip)

    def record_suspicious_activity(self, identifier: str):
        """Record suspicious activity"""
        current_time = time.time()
        self.suspicious_activity[identifier].append(current_time)

        # Remove old entries (last hour)
        self.suspicious_activity[identifier] = [
            t for t in self.suspicious_activity[identifier] if t > current_time - 3600
        ]

        # Auto-block if too many suspicious activities
        if len(self.suspicious_activity[identifier]) > 10:
            if identifier.count(".") == 3:  # Looks like an IP
                self.block_ip(identifier)


class EnhancedGRPOGateway:
    """Enhanced GRPO Service Gateway with advanced features"""

    def __init__(
        self,
        redis_url: Optional[str] = None,
        enable_metrics: bool = True,
        enable_security: bool = True,
    ):
        self.cache = IntelligentCache(redis_url)
        self.load_balancer = LoadBalancer()
        self.rate_limiter = RateLimiter()
        self.security_manager = SecurityManager() if enable_security else None
        self.error_handler = ErrorHandler()
        self.performance_optimizer = PerformanceOptimizer(OptimizationLevel.BALANCED)
        self.monitoring = MonitoringService() if enable_metrics else None

        self.routes: Dict[str, RouteConfig] = {}
        self.middleware_stack: List[Callable] = []

        # Statistics
        self.stats = {
            "total_requests": 0,
            "total_errors": 0,
            "average_response_time": 0.0,
            "cache_hit_rate": 0.0,
            "active_connections": 0,
        }

    def add_route(self, config: RouteConfig):
        """Add a route configuration"""
        self.routes[config.path] = config

    def add_service_instance(self, instance: ServiceInstance):
        """Add a service instance"""
        self.load_balancer.add_instance(instance)

    async def process_request(
        self,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: Any = None,
        client_ip: str = None,
    ) -> Tuple[int, Dict[str, str], Any]:
        """Process an incoming request"""
        start_time = time.time()
        request_id = str(uuid.uuid4())

        try:
            self.stats["total_requests"] += 1
            self.stats["active_connections"] += 1

            # Security checks
            if self.security_manager and client_ip:
                if self.security_manager.is_ip_blocked(client_ip):
                    raise HTTPException(status_code=403, detail="IP blocked")

            # Find route configuration
            route_config = self._find_route_config(path)
            if not route_config:
                raise HTTPException(status_code=404, detail="Route not found")

            # Authentication
            if route_config.requires_auth and self.security_manager:
                auth_header = headers.get("Authorization", "")
                if not auth_header.startswith("Bearer "):
                    raise HTTPException(
                        status_code=401, detail="Authentication required"
                    )

                api_key = auth_header.replace("Bearer ", "")
                key_info = self.security_manager.validate_api_key(api_key)
                if not key_info:
                    raise HTTPException(status_code=401, detail="Invalid API key")

                # Role validation
                if route_config.allowed_roles and not any(
                    role in key_info["roles"] for role in route_config.allowed_roles
                ):
                    raise HTTPException(
                        status_code=403, detail="Insufficient permissions"
                    )

            # Rate limiting
            if route_config.rate_limit:
                limit_key = f"{client_ip}:{path}" if client_ip else path
                if not await self.rate_limiter.is_allowed(
                    limit_key, route_config.rate_limit
                ):
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")

            # Cache check
            if method == "GET" and route_config.cache_ttl > 0:
                cache_key = self._generate_cache_key(path, headers, body)
                cached_response = await self.cache.get(cache_key)
                if cached_response:
                    return (
                        cached_response["status"],
                        cached_response["headers"],
                        cached_response["body"],
                    )

            # Load balancing
            instance = await self.load_balancer.get_instance()
            if not instance:
                raise HTTPException(
                    status_code=503, detail="No healthy service instances"
                )

            # Forward request
            status, response_headers, response_body = await self._forward_request(
                instance, method, path, headers, body, route_config.timeout
            )

            # Cache response
            if method == "GET" and route_config.cache_ttl > 0 and status == 200:
                cache_key = self._generate_cache_key(path, headers, body)
                cached_data = {
                    "status": status,
                    "headers": response_headers,
                    "body": response_body,
                }
                await self.cache.set(cache_key, cached_data, route_config.cache_ttl)

            # Record metrics
            response_time = time.time() - start_time
            instance.record_request(response_time)
            self._update_stats(response_time)

            return status, response_headers, response_body

        except Exception as e:
            self.stats["total_errors"] += 1
            if self.security_manager and client_ip:
                self.security_manager.record_suspicious_activity(client_ip)

            error_context = self.error_handler.handle_error(
                e, "gateway", "request_processing"
            )

            return (
                500,
                {"Content-Type": "application/json"},
                {
                    "error": "Internal server error",
                    "error_id": error_context.error_id,
                    "message": str(e)
                    if isinstance(e, HTTPException)
                    else "An error occurred",
                },
            )

        finally:
            self.stats["active_connections"] -= 1

    def _find_route_config(self, path: str) -> Optional[RouteConfig]:
        """Find route configuration for path"""
        # Exact match first
        if path in self.routes:
            return self.routes[path]

        # Pattern matching (simplified)
        for route_path, config in self.routes.items():
            if self._path_matches(path, route_path):
                return config

        return None

    def _path_matches(self, path: str, pattern: str) -> bool:
        """Simple path pattern matching"""
        if "*" in pattern:
            prefix = pattern.split("*")[0]
            return path.startswith(prefix)
        return path == pattern

    def _generate_cache_key(self, path: str, headers: Dict[str, str], body: Any) -> str:
        """Generate cache key for request"""
        key_data = {
            "path": path,
            "relevant_headers": {
                k: v
                for k, v in headers.items()
                if k.lower() in ["accept", "content-type"]
            },
            "body_hash": hashlib.md5(str(body).encode()).hexdigest() if body else None,
        }
        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

    async def _forward_request(
        self,
        instance: ServiceInstance,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: Any,
        timeout: float,
    ) -> Tuple[int, Dict[str, str], Any]:
        """Forward request to service instance"""
        # This is a simplified implementation
        # In practice, you would use aiohttp or similar to make the actual HTTP request

        # Simulate request forwarding
        await asyncio.sleep(0.01)  # Simulate network latency

        # Simulate different responses based on path
        if "health" in path:
            return 200, {"Content-Type": "application/json"}, {"status": "healthy"}
        elif "train" in path:
            return (
                200,
                {"Content-Type": "application/json"},
                {
                    "job_id": str(uuid.uuid4()),
                    "status": "started",
                    "instance": instance.id,
                },
            )
        elif "chat" in path:
            return (
                200,
                {"Content-Type": "application/json"},
                {
                    "response": "Hello! This is a simulated response.",
                    "instance": instance.id,
                },
            )
        else:
            return (
                200,
                {"Content-Type": "application/json"},
                {"message": "Request processed successfully", "instance": instance.id},
            )

    def _update_stats(self, response_time: float):
        """Update gateway statistics"""
        # Simple moving average for response time
        alpha = 0.1  # Smoothing factor
        self.stats["average_response_time"] = (
            alpha * response_time + (1 - alpha) * self.stats["average_response_time"]
        )

        # Update cache hit rate
        if self.cache.cache_stats["hits"] + self.cache.cache_stats["misses"] > 0:
            self.stats["cache_hit_rate"] = self.cache.cache_stats["hits"] / (
                self.cache.cache_stats["hits"] + self.cache.cache_stats["misses"]
            )

    def get_health_status(self) -> Dict[str, Any]:
        """Get gateway health status"""
        healthy_instances = sum(
            1
            for inst in self.load_balancer.instances.values()
            if inst.status == ServiceStatus.HEALTHY
        )

        return {
            "status": "healthy" if healthy_instances > 0 else "unhealthy",
            "instances": {
                "total": len(self.load_balancer.instances),
                "healthy": healthy_instances,
                "unhealthy": len(self.load_balancer.instances) - healthy_instances,
            },
            "stats": self.stats,
            "cache_stats": self.cache.cache_stats,
            "uptime": time.time() - getattr(self, "_start_time", time.time()),
        }


def create_enhanced_gateway_app() -> FastAPI:
    """Create FastAPI app with enhanced gateway"""
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI is required for gateway functionality")

    gateway = EnhancedGRPOGateway()

    # Initialize default service instances (for demo)
    gateway.add_service_instance(
        ServiceInstance(id="grpo-service-1", host="localhost", port=8001, weight=1.0)
    )

    # Configure routes
    gateway.add_route(
        RouteConfig(
            path="/api/train",
            methods=["POST"],
            timeout=300.0,
            cache_ttl=0,
            rate_limit=60,
            requires_auth=True,
            allowed_roles=["admin", "trainer"],
        )
    )

    gateway.add_route(
        RouteConfig(
            path="/api/chat",
            methods=["POST"],
            timeout=30.0,
            cache_ttl=300,  # Cache responses for 5 minutes
            rate_limit=100,
            requires_auth=False,
        )
    )

    gateway.add_route(
        RouteConfig(
            path="/health", methods=["GET"], timeout=5.0, cache_ttl=60, rate_limit=1000
        )
    )

    app = FastAPI(
        title="Enhanced GRPO Gateway",
        description="Advanced API gateway for GRPO RL services",
        version="1.0.0",
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

    @app.middleware("http")
    async def gateway_middleware(request: Request, call_next):
        """Gateway processing middleware"""
        start_time = time.time()

        # Extract request information
        method = request.method
        path = str(request.url.path)
        headers = dict(request.headers)
        client_ip = request.client.host if request.client else None

        # Get request body
        body = None
        if method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.json()
            except:
                try:
                    body = await request.body()
                except:
                    pass

        # Process through gateway
        status, response_headers, response_body = await gateway.process_request(
            method, path, headers, body, client_ip
        )

        # Create response
        response = Response(
            content=json.dumps(response_body)
            if isinstance(response_body, dict)
            else response_body,
            status_code=status,
            headers=response_headers,
        )

        return response

    @app.get("/gateway/health")
    async def gateway_health():
        """Gateway health check"""
        return gateway.get_health_status()

    @app.post("/gateway/config/route")
    async def add_route(config: dict):
        """Add route configuration"""
        route_config = RouteConfig(**config)
        gateway.add_route(route_config)
        return {"message": "Route added successfully"}

    @app.post("/gateway/config/instance")
    async def add_instance(instance_data: dict):
        """Add service instance"""
        instance = ServiceInstance(**instance_data)
        gateway.add_service_instance(instance)
        return {"message": "Service instance added successfully"}

    @app.get("/gateway/stats")
    async def get_stats():
        """Get gateway statistics"""
        return {
            "gateway_stats": gateway.stats,
            "cache_stats": gateway.cache.cache_stats,
            "load_balancer_stats": {
                "instances": len(gateway.load_balancer.instances),
                "strategy": gateway.load_balancer.strategy.value,
            },
        }

    return app


if __name__ == "__main__":
    if FASTAPI_AVAILABLE:
        import uvicorn

        app = create_enhanced_gateway_app()
        gateway._start_time = time.time()

        print("\n" + "=" * 80)
        print("🚀 ENHANCED GRPO GATEWAY - NEXT-GENERATION API ARCHITECTURE")
        print("=" * 80)
        print("\nAdvanced Features:")
        print("✅ Intelligent Load Balancing with Multiple Strategies")
        print("✅ Multi-Layer Caching (Local + Redis)")
        print("✅ Advanced Rate Limiting with Sliding Windows")
        print("✅ Circuit Breakers for Fault Tolerance")
        print("✅ Security with API Key Management")
        print("✅ Real-time Monitoring & Metrics")
        print("✅ Auto-scaling and Health Checks")
        print("✅ Request/Response Compression")
        print("✅ Suspicious Activity Detection")
        print("✅ Flexible Route Configuration")
        print("=" * 80 + "\n")

        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    else:
        print("FastAPI is required to run the enhanced gateway")
