"""
Ultimate GRPO Service API - Integrating All Innovations

This service provides a comprehensive API that integrates all the latest
innovations from the /grpo directory into the GRPO Agent Framework.
"""

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
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
        Request,
        WebSocket,
        WebSocketDisconnect,
    )
    from fastapi.encoders import jsonable_encoder
    from fastapi.exceptions import RequestValidationError
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, field_validator

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Lightweight shims so the module can be imported without FastAPI installed
    class HTTPException(Exception):
        def __init__(self, status_code: int = 500, detail: Any = None):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

    class Request:
        def __init__(self):
            self.headers = {}
            self.client = None
            self.url = type("URL", (), {"path": "/"})()
            self.state = type("state", (), {})()

    class WebSocket:
        pass

    class WebSocketDisconnect(Exception):
        pass

    class BackgroundTasks:
        def add_task(self, *_args, **_kwargs):
            return None

    def Depends(dependency):
        return dependency

    class BaseModel:
        pass

    def Field(default=None, default_factory=None):
        return default if default_factory is None else default_factory()

    def field_validator(*_args, **_kwargs):
        def decorator(func):
            return func
        return decorator

    def jsonable_encoder(obj):
        return obj

    class JSONResponse:
        def __init__(self, status_code: int = 200, content: Any = None):
            self.status_code = status_code
            self.content = content

import utils.monitoring

from stateset_agents.core.agent import Agent
from stateset_agents.core.computational_engine import (
    ComputationalGRPOEngine,
    create_computational_engine,
)
from stateset_agents.core.environment import Environment
from stateset_agents.core.multiturn_agent import DialogueDatabase, MultiTurnAgent

MonitoringService = utils.monitoring.MonitoringService
import utils.cache

CacheService = utils.cache.CacheService

logger = logging.getLogger(__name__)


def _get_int_from_env(env_var: str, default: int) -> int:
    """Parse an integer environment variable safely."""
    try:
        return int(os.getenv(env_var, str(default)))
    except ValueError:
        logger.warning("Invalid value for %s, using default %s", env_var, default)
        return default


@dataclass
class APIConfig:
    """Runtime configuration for the API surface."""

    api_keys: Dict[str, List[str]]
    rate_limit_per_minute: int
    max_prompts: int
    max_prompt_length: int
    max_message_length: int
    max_iterations: int
    allow_anonymous: bool

    @classmethod
    def from_env(cls) -> "APIConfig":
        raw_keys = os.getenv("GRPO_API_KEYS", "")
        parsed_keys: Dict[str, List[str]] = {}

        for item in raw_keys.split(","):
            cleaned = item.strip()
            if not cleaned:
                continue

            key, _, roles_raw = cleaned.partition(":")
            roles = [role.strip() for role in roles_raw.split("|") if role.strip()]
            parsed_keys[key] = roles or ["user"]

        return cls(
            api_keys=parsed_keys,
            rate_limit_per_minute=_get_int_from_env("GRPO_RATE_LIMIT_PER_MIN", 60),
            max_prompts=_get_int_from_env("GRPO_MAX_PROMPTS", 8),
            max_prompt_length=_get_int_from_env("GRPO_MAX_PROMPT_CHARS", 4000),
            max_message_length=_get_int_from_env("GRPO_MAX_MESSAGE_CHARS", 4000),
            max_iterations=_get_int_from_env("GRPO_MAX_ITERATIONS", 50),
            allow_anonymous=os.getenv("GRPO_ALLOW_ANONYMOUS", "false").lower()
            in {"1", "true", "yes", "on"},
        )


class SlidingWindowRateLimiter:
    """In-memory sliding window rate limiter."""

    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self.windows: Dict[str, deque] = defaultdict(deque)

    def allow(self, key: str, limit: int) -> bool:
        """Return True if the request is within the rate limit."""
        now = time.monotonic()
        window = self.windows[key]
        cutoff = now - self.window_seconds

        while window and window[0] <= cutoff:
            window.popleft()

        if len(window) >= limit:
            return False

        window.append(now)
        return True


class APIMetrics:
    """Lightweight request metrics tracker."""

    def __init__(self) -> None:
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.status_counts: Dict[int, int] = defaultdict(int)
        self.latencies: deque = deque(maxlen=1000)
        self.rate_limit_hits: int = 0

    def record(self, path: str, status_code: int, latency: float) -> None:
        """Record a request/response pair."""
        self.request_counts[path] += 1
        self.status_counts[status_code] += 1
        self.latencies.append(latency)

    def snapshot(self) -> Dict[str, Any]:
        """Return a serializable view of metrics."""
        average_latency = (
            sum(self.latencies) / len(self.latencies) if self.latencies else 0.0
        )
        percentile_95 = 0.0
        if self.latencies:
            sorted_latencies = sorted(self.latencies)
            index = int(0.95 * (len(sorted_latencies) - 1))
            percentile_95 = sorted_latencies[index]

        return {
            "total_requests": sum(self.request_counts.values()),
            "requests_by_path": dict(self.request_counts),
            "status_codes": dict(self.status_counts),
            "average_latency_seconds": round(average_latency, 4),
            "p95_latency_seconds": round(percentile_95, 4),
            "rate_limit_hits": self.rate_limit_hits,
        }


@dataclass
class RequestContext:
    """Captured request metadata after auth/rate limiting."""

    request_id: str
    user_id: str
    roles: List[str]
    api_key: Optional[str]
    client: str


API_CONFIG = APIConfig.from_env()
RATE_LIMITER = SlidingWindowRateLimiter()
API_METRICS = APIMetrics()


class LightweightDemoEngine:
    """Lightweight demo engine to keep the API responsive in constrained envs."""

    def __init__(self) -> None:
        self.scale_factor = 1.0
        self.total_trajectories = 0
        self.total_reward = 0.0

    async def train_iteration(self, prompts: List[str]) -> Dict[str, Any]:
        """Simulate a training iteration without heavy compute."""
        trajectories = len(prompts)
        average_reward = 0.5
        computation_used = trajectories * self.scale_factor

        self.total_trajectories += trajectories
        self.total_reward += average_reward * trajectories

        return {
            "trajectories_generated": trajectories,
            "average_reward": average_reward,
            "total_computation_used": computation_used,
            "scale_factor": self.scale_factor,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Return simple engine metrics."""
        average_reward = (
            self.total_reward / self.total_trajectories
            if self.total_trajectories
            else 0.0
        )
        return {
            "total_trajectories": self.total_trajectories,
            "average_reward": average_reward,
            "scale_factor": self.scale_factor,
        }

    def scale_computation(self, scale_factor: float) -> Dict[str, Any]:
        """Adjust the simulated computation scale."""
        self.scale_factor = scale_factor
        return {"scale_factor": scale_factor}

    def cleanup(self) -> None:
        """Cleanup hook for parity with real engines."""
        return None


def _build_error_response(
    request: Request, status_code: int, message: str, details: Optional[Any] = None
) -> JSONResponse:
    """Consistent error envelope across the API."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    payload = {
        "error": {"message": message, "status_code": status_code},
        "request_id": request_id,
        "path": str(request.url.path),
        "timestamp": datetime.utcnow().isoformat(),
    }
    if details is not None:
        payload["error"]["details"] = jsonable_encoder(details)

    return JSONResponse(status_code=status_code, content=payload)

# TTL-enabled dictionary for bounded state storage
class TTLDict(dict):
    """Dictionary with automatic cleanup of expired entries."""

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 10000):
        super().__init__()
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.Lock()

    def __setitem__(self, key, value):
        with self._lock:
            # Enforce max size by removing oldest entries
            if len(self) >= self.max_size:
                self._evict_oldest(len(self) - self.max_size + 1)
            super().__setitem__(key, value)
            self._timestamps[key] = time.time()

    def __getitem__(self, key):
        with self._lock:
            # Check if expired
            if key in self._timestamps:
                if time.time() - self._timestamps[key] > self.ttl_seconds:
                    super().pop(key, None)
                    self._timestamps.pop(key, None)
                    raise KeyError(key)
            return super().__getitem__(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def setdefault(self, key, default=None):
        with self._lock:
            now = time.time()
            timestamp = self._timestamps.get(key)
            if timestamp is not None and now - timestamp <= self.ttl_seconds:
                return super().__getitem__(key)

            # Insert/replace without calling __setitem__ (avoid deadlock).
            if key not in self and len(self) >= self.max_size:
                self._evict_oldest(len(self) - self.max_size + 1)
            super().__setitem__(key, default)
            self._timestamps[key] = now
            return default

    def _evict_oldest(self, count: int = 1):
        """Remove the oldest entries."""
        if not self._timestamps:
            return
        sorted_keys = sorted(self._timestamps.keys(), key=lambda k: self._timestamps[k])
        for key in sorted_keys[:count]:
            super().pop(key, None)
            self._timestamps.pop(key, None)

    def cleanup_expired(self):
        """Remove all expired entries."""
        with self._lock:
            now = time.time()
            expired_keys = [
                k for k, ts in self._timestamps.items()
                if now - ts > self.ttl_seconds
            ]
            for key in expired_keys:
                super().pop(key, None)
                self._timestamps.pop(key, None)
            return len(expired_keys)


# Global state with TTL for automatic cleanup
services = {}
active_engines = {}
active_conversations = TTLDict(ttl_seconds=3600, max_size=10000)  # 1 hour TTL, max 10K
training_jobs = TTLDict(ttl_seconds=86400, max_size=1000)  # 24 hour TTL, max 1K


def _extract_api_key(request: Request) -> Optional[str]:
    """Get API key from headers (supports Bearer and X-API-Key)."""
    auth_header = request.headers.get("authorization") or ""
    if auth_header.lower().startswith("bearer "):
        return auth_header.split(" ", 1)[1].strip()
    api_key = request.headers.get("x-api-key")
    return api_key.strip() if api_key else None


async def verify_request(request: Request) -> RequestContext:
    """Authenticate request and enforce rate limits."""
    client_ip = request.client.host if request.client else "unknown"
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    api_key = _extract_api_key(request)

    if API_CONFIG.api_keys:
        if not api_key or api_key not in API_CONFIG.api_keys:
            raise HTTPException(
                status_code=401, detail="A valid API key is required for this API"
            )
        roles = API_CONFIG.api_keys[api_key]
        user_id = request.headers.get("x-user-id", client_ip)
    elif API_CONFIG.allow_anonymous:
        roles = ["anonymous"]
        user_id = request.headers.get("x-user-id", client_ip)
    else:
        raise HTTPException(
            status_code=401,
            detail="API key required. Set GRPO_ALLOW_ANONYMOUS=true to enable unauthenticated access.",
        )

    limit_key = api_key or client_ip
    if not RATE_LIMITER.allow(limit_key, API_CONFIG.rate_limit_per_minute):
        API_METRICS.rate_limit_hits += 1
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please retry after a short delay.",
        )

    return RequestContext(
        request_id=request_id,
        user_id=user_id,
        roles=roles,
        api_key=api_key,
        client=client_ip,
    )


# API Models
class TrainingRequest(BaseModel):
    """Request model for training"""

    prompts: List[str]
    strategy: str = "computational"
    num_iterations: int = 1
    parallel_batch_size: Optional[int] = None
    use_neural_rewards: bool = True
    use_ruler_rewards: bool = False
    distributed_config: Optional[Dict[str, Any]] = None

    @field_validator("prompts")
    @classmethod
    def validate_prompts(cls, prompts: List[str]) -> List[str]:
        cleaned = [prompt.strip() for prompt in prompts if prompt and prompt.strip()]
        if not cleaned:
            raise ValueError("At least one prompt is required")
        if len(cleaned) > API_CONFIG.max_prompts:
            raise ValueError(
                f"Maximum {API_CONFIG.max_prompts} prompts are supported per request"
            )
        for prompt in cleaned:
            if len(prompt) > API_CONFIG.max_prompt_length:
                raise ValueError(
                    f"Prompt exceeds {API_CONFIG.max_prompt_length} characters"
                )
        return cleaned

    @field_validator("num_iterations")
    @classmethod
    def validate_iterations(cls, iterations: int) -> int:
        if iterations < 1:
            raise ValueError("num_iterations must be positive")
        if iterations > API_CONFIG.max_iterations:
            raise ValueError(
                f"num_iterations cannot exceed {API_CONFIG.max_iterations}"
            )
        return iterations

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, strategy: str) -> str:
        allowed = {"computational", "distributed"}
        if strategy not in allowed:
            raise ValueError(
                f"strategy must be one of: {', '.join(sorted(allowed))}"
            )
        return strategy

    @field_validator("parallel_batch_size")
    @classmethod
    def validate_parallel_batch_size(
        cls, parallel_batch_size: Optional[int]
    ) -> Optional[int]:
        if parallel_batch_size is not None and parallel_batch_size < 1:
            raise ValueError("parallel_batch_size must be positive when provided")
        return parallel_batch_size


class ConversationRequest(BaseModel):
    """Request model for conversations"""

    message: str
    conversation_id: Optional[str] = None
    strategy: str = "default"
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

    @field_validator("message")
    @classmethod
    def validate_message(cls, message: str) -> str:
        cleaned = message.strip()
        if not cleaned:
            raise ValueError("message cannot be empty")
        if len(cleaned) > API_CONFIG.max_message_length:
            raise ValueError(
                f"message exceeds {API_CONFIG.max_message_length} characters"
            )
        return cleaned


class TrainingResponse(BaseModel):
    """Response model for training"""

    job_id: str
    status: str
    iterations_completed: int = 0
    total_trajectories: int = 0
    average_reward: float = 0.0
    computation_used: float = 0.0
    metrics: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    request_id: Optional[str] = None


class ConversationResponse(BaseModel):
    """Response model for conversations"""

    conversation_id: str
    response: str
    context: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ScaleRequest(BaseModel):
    """Request model for scaling computation"""

    scale_factor: float
    apply_to_all: bool = False

    @field_validator("scale_factor")
    @classmethod
    def validate_scale_factor(cls, scale_factor: float) -> float:
        if scale_factor <= 0:
            raise ValueError("scale_factor must be positive")
        return scale_factor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for FastAPI app"""
    # Startup
    logger.info("🚀 Starting Ultimate GRPO Service")

    # Initialize services
    enable_prometheus = (
        os.getenv("GRPO_ENABLE_PROMETHEUS", "false").lower() == "true"
    )
    services["monitoring"] = MonitoringService(enable_prometheus=enable_prometheus)
    services["cache"] = CacheService()

    # Initialize example components
    model_config = {"model_type": "gpt-oss", "model_name": "openai/gpt-oss-120b"}

    # Create computational engine (lightweight by default to avoid heavy spawning)
    from stateset_agents.core.agent import Agent
    from stateset_agents.core.environment import Environment
    from stateset_agents.core.reward import RewardFunction

    use_lightweight_engine = (
        os.getenv("GRPO_API_LIGHT_ENGINE", "true").lower() != "false"
    )

    if use_lightweight_engine:
        services["demo_engine"] = LightweightDemoEngine()
    else:
        class DemoAgent(Agent):
            async def generate_response(self, prompt: str) -> str:
                return f"Response to: {prompt[:50]}..."

        class DemoEnvironment(Environment):
            async def reset(self) -> Dict[str, Any]:
                return {"state": "initial"}

            async def step(self, action: str) -> Dict[str, Any]:
                return {"reward": 0.5, "done": False}

            async def get_reward(self, trajectory) -> float:
                return 0.5

        class DemoReward(RewardFunction):
            async def compute_reward(self, turns, context=None):
                from stateset_agents.core.reward import RewardResult

                return RewardResult(score=0.5, breakdown={})

        demo_agent = DemoAgent(model_config)
        demo_environment = DemoEnvironment()
        demo_reward = DemoReward()

        services["demo_engine"] = create_computational_engine(
            demo_agent, demo_environment, demo_reward, num_workers=1
        )

    logger.info(
        "Initialized demo engine (%s)",
        "lightweight" if use_lightweight_engine else "computational",
    )

    # Create multi-turn agent
    services["multiturn_agent"] = MultiTurnAgent(
        model_config, dialogue_database=DialogueDatabase([])
    )

    logger.info("✅ Ultimate GRPO Service initialized")

    yield

    # Shutdown
    logger.info("🛑 Shutting down Ultimate GRPO Service")

    # Cleanup engines
    for engine in active_engines.values():
        if hasattr(engine, "cleanup"):
            engine.cleanup()

    logger.info("✅ Ultimate GRPO Service shutdown complete")


# Create FastAPI app
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Ultimate GRPO Service",
        description="Comprehensive GRPO training and inference API with all latest innovations",
        version="2.0.0",
        lifespan=lifespan,
    )

    # Add CORS middleware with secure configuration
    # In production, set GRPO_CORS_ORIGINS environment variable to comma-separated allowed origins
    cors_origins_env = os.getenv("GRPO_CORS_ORIGINS", "")
    if cors_origins_env:
        cors_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]
    else:
        # Default to permissive for development; restrict in production
        cors_origins = ["*"] if os.getenv("GRPO_ENV", "development") == "development" else []

    # Only allow credentials when origins are explicitly specified (not wildcard)
    allow_credentials = cors_origins and cors_origins != ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=allow_credentials,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Request-ID", "X-User-ID"],
    )

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        """Attach request id, capture metrics, and normalize errors."""
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        start_time = time.monotonic()

        try:
            response = await call_next(request)
        except RequestValidationError as exc:
            response = _build_error_response(
                request,
                422,
                "Invalid request payload",
                details=exc.errors(),
            )
        except HTTPException as exc:
            response = _build_error_response(
                request, exc.status_code, str(exc.detail)
            )
        except Exception as exc:  # pragma: no cover - safety net
            logger.exception("Unhandled exception for request %s", request_id)
            response = _build_error_response(
                request, 500, "Internal server error, please retry later."
            )

        API_METRICS.record(
            request.url.path, response.status_code, time.monotonic() - start_time
        )
        response.headers["x-request-id"] = request_id
        return response

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Return consistent error envelopes for HTTP errors."""
        return _build_error_response(request, exc.status_code, str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        """Normalize validation errors into the API envelope."""
        return _build_error_response(
            request,
            422,
            "Invalid request payload",
            details=exc.errors(),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        """Catch-all handler to avoid leaking stack traces."""
        logger.exception("Unhandled exception on path %s", request.url.path)
        return _build_error_response(
            request, 500, "Internal server error, please retry later."
        )
else:
    app = None


if FASTAPI_AVAILABLE:
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "title": "Ultimate GRPO Service",
            "version": "2.0.0",
            "description": "Comprehensive GRPO training and inference API",
            "innovations": [
                "🧠 Neural Reward Models",
                "⚖️ RULER LLM Judges",
                "💬 Multi-Turn Conversations",
                "🔄 Distributed Training",
                "⚡ Computational Engine",
                "🎯 Multi-Objective Rewards",
                "🚀 Auto-Scaling",
            ],
            "security": {
                "auth_enabled": bool(API_CONFIG.api_keys),
                "allow_anonymous": API_CONFIG.allow_anonymous,
                "rate_limit_per_minute": API_CONFIG.rate_limit_per_minute,
            },
            "endpoints": {
                "training": "/api/train",
                "conversations": "/api/chat",
                "scaling": "/api/scale",
                "metrics": "/api/metrics",
                "websocket": "/ws",
            },
        }


    @app.post("/api/train", response_model=TrainingResponse)
    async def train_agent(
        request: TrainingRequest,
        background_tasks: BackgroundTasks,
        user_context: RequestContext = Depends(verify_request),
    ):
        """
        Launch advanced GRPO training with all innovations
        """
        job_id = str(uuid.uuid4())

        # Initialize job tracking
        training_jobs[job_id] = {
            "status": "starting",
            "strategy": request.strategy,
            "iterations_completed": 0,
            "total_trajectories": 0,
            "start_time": datetime.utcnow(),
            "request_id": user_context.request_id,
            "user_id": user_context.user_id,
            "error": None,
            "results": [],
        }

        # Launch training based on strategy
        if request.strategy == "computational":
            background_tasks.add_task(
                run_computational_training,
                job_id,
                request.prompts,
                request.num_iterations,
                request.use_neural_rewards,
                request.use_ruler_rewards,
            )
        elif request.strategy == "distributed":
            background_tasks.add_task(
                run_distributed_training,
                job_id,
                request.prompts,
                request.num_iterations,
                request.distributed_config or {},
            )
        else:
            raise HTTPException(
                status_code=400, detail=f"Unknown training strategy: {request.strategy}"
            )

        return TrainingResponse(
            job_id=job_id,
            status="started",
            iterations_completed=0,
            total_trajectories=0,
            average_reward=0.0,
            computation_used=0.0,
            metrics={"strategy": request.strategy},
            started_at=training_jobs[job_id]["start_time"],
            completed_at=None,
            request_id=user_context.request_id,
        )


    @app.post("/api/chat", response_model=ConversationResponse)
    async def chat(
        request: ConversationRequest,
        user_context: RequestContext = Depends(verify_request),
    ):
        """
        Advanced multi-turn conversational interface
        """
        multiturn_agent = services.get("multiturn_agent")
        if not multiturn_agent:
            raise HTTPException(status_code=500, detail="Multi-turn agent not initialized")

        # Start or continue conversation
        if request.conversation_id:
            # Continue existing conversation
            try:
                turns = await multiturn_agent.continue_conversation(
                    request.conversation_id, request.message, strategy=request.strategy
                )
                response = turns[-1]["content"] if turns else "No response generated"

                context = multiturn_agent.get_conversation_summary(request.conversation_id)
                active_conversations.setdefault(
                    request.conversation_id,
                    {
                        "user_id": request.user_id or user_context.user_id,
                        "started_at": datetime.utcnow(),
                        "strategy": request.strategy,
                    },
                )

            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
        else:
            # Start new conversation
            conversation_context = await multiturn_agent.start_conversation(
                user_id=request.user_id or user_context.user_id,
                initial_context=request.context,
            )

            # Generate first response
            response = await multiturn_agent.generate_multiturn_response(
                conversation_context.conversation_id,
                request.message,
                strategy=request.strategy,
            )

            # Update conversation
            conversation_context.add_turn({"role": "assistant", "content": response})

            context = conversation_context.get_context_summary()
            request.conversation_id = conversation_context.conversation_id
            active_conversations[request.conversation_id] = {
                "user_id": request.user_id or user_context.user_id,
                "started_at": datetime.utcnow(),
                "strategy": request.strategy,
            }

        return ConversationResponse(
            conversation_id=request.conversation_id,
            response=response,
            context=context,
            metadata={
                "strategy": request.strategy,
                "timestamp": datetime.now().isoformat(),
            },
        )


    @app.post("/api/scale")
    async def scale_computation(
        request: ScaleRequest, user_context: RequestContext = Depends(verify_request)
    ):
        """
        Scale computational resources across engines
        """
        results = {}

        if request.apply_to_all:
            # Scale all engines
            for engine_id, engine in active_engines.items():
                if hasattr(engine, "scale_computation"):
                    try:
                        result = engine.scale_computation(request.scale_factor)
                        results[engine_id] = result
                    except Exception as e:
                        results[engine_id] = {"error": str(e)}

            # Scale demo engine
            if "demo_engine" in services:
                try:
                    result = services["demo_engine"].scale_computation(request.scale_factor)
                    results["demo_engine"] = result
                except Exception as e:
                    results["demo_engine"] = {"error": str(e)}
        else:
            # Scale demo engine only
            if "demo_engine" in services:
                try:
                    result = services["demo_engine"].scale_computation(request.scale_factor)
                    results["demo_engine"] = result
                except Exception as e:
                    results["demo_engine"] = {"error": str(e)}

        return {
            "message": "Computational resources scaled",
            "scale_factor": request.scale_factor,
            "results": results,
            "philosophy": "Computation is the key to long-term improvement",
        }


    @app.get("/api/metrics")
    async def get_metrics(user_context: RequestContext = Depends(verify_request)):
        """Get comprehensive system metrics"""
        metrics = {
            "system": {
                "active_engines": len(active_engines),
                "active_conversations": len(active_conversations),
                "training_jobs": len(training_jobs),
                "services_initialized": len(services),
            },
            "training_jobs": {},
            "engines": {},
            "conversations": {},
        }

        # Training job metrics
        for job_id, job in training_jobs.items():
            metrics["training_jobs"][job_id] = {
                "status": job["status"],
                "strategy": job["strategy"],
                "iterations_completed": job["iterations_completed"],
                "total_trajectories": job["total_trajectories"],
            }

        # Engine metrics
        for engine_id, engine in active_engines.items():
            if hasattr(engine, "get_metrics"):
                metrics["engines"][engine_id] = engine.get_metrics()

        # Demo engine metrics
        if "demo_engine" in services:
            metrics["demo_engine"] = services["demo_engine"].get_metrics()

        # Conversation metrics
        if "multiturn_agent" in services:
            agent = services["multiturn_agent"]
            metrics["conversations"] = {
                "active_count": len(agent.get_active_conversations()),
                "strategies_available": list(agent.strategies.keys()),
                "tools_registered": list(agent.tools.keys()),
            }

        metrics["api"] = API_METRICS.snapshot()
        metrics["rate_limit"] = {
            "requests_per_minute": API_CONFIG.rate_limit_per_minute,
            "auth_enabled": bool(API_CONFIG.api_keys),
            "allow_anonymous": API_CONFIG.allow_anonymous,
        }

        return metrics


    @app.get("/api/status/{job_id}", response_model=TrainingResponse)
    async def get_training_status(
        job_id: str, user_context: RequestContext = Depends(verify_request)
    ):
        """Get training job status"""
        if job_id not in training_jobs:
            raise HTTPException(status_code=404, detail="Training job not found")

        job = training_jobs[job_id]

        # Calculate metrics
        if job["results"]:
            avg_reward = sum(r.get("average_reward", 0) for r in job["results"]) / len(
                job["results"]
            )
            total_computation = sum(
                r.get("total_computation_used", 0) for r in job["results"]
            )
            latest_metrics = job["results"][-1] if job["results"] else {}
        else:
            avg_reward = 0.0
            total_computation = 0.0
            latest_metrics = {}

        return TrainingResponse(
            job_id=job_id,
            status=job["status"],
            iterations_completed=job["iterations_completed"],
            total_trajectories=job["total_trajectories"],
            average_reward=avg_reward,
            computation_used=total_computation,
            metrics=latest_metrics,
            error=job.get("error"),
            started_at=job.get("start_time"),
            completed_at=job.get("completed_at"),
            request_id=job.get("request_id"),
        )


    @app.delete("/api/conversations/{conversation_id}")
    async def end_conversation(
        conversation_id: str, user_context: RequestContext = Depends(verify_request)
    ):
        """End a conversation"""
        multiturn_agent = services.get("multiturn_agent")
        if not multiturn_agent:
            raise HTTPException(status_code=500, detail="Multi-turn agent not initialized")

        context = multiturn_agent.end_conversation(conversation_id)
        if not context:
            raise HTTPException(status_code=404, detail="Conversation not found")

        active_conversations.pop(conversation_id, None)

        return {
            "message": "Conversation ended",
            "conversation_id": conversation_id,
            "final_summary": context.get_context_summary(),
        }


    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time interactions"""
        # Authenticate before accepting connection
        if API_CONFIG.api_keys:
            api_key = websocket.headers.get("x-api-key") or websocket.headers.get(
                "authorization"
            )
            if api_key and api_key.lower().startswith("bearer "):
                api_key = api_key.split(" ", 1)[1].strip()

            if not api_key or api_key not in API_CONFIG.api_keys:
                await websocket.close(code=1008, reason="Unauthorized")
                return

            # Rate limit WebSocket connections
            limit_key = f"ws:{api_key}"
            if not RATE_LIMITER.allow(limit_key, API_CONFIG.rate_limit_per_minute):
                await websocket.close(code=1008, reason="Rate limit exceeded")
                return

        await websocket.accept()

        # Constants for WebSocket security
        MAX_MESSAGE_SIZE = 65536  # 64KB max message size
        MAX_MESSAGES_PER_SECOND = 10
        message_timestamps = deque(maxlen=MAX_MESSAGES_PER_SECOND)

        try:
            while True:
                # Receive message
                data = await websocket.receive_text()

                # Validate message size
                if len(data) > MAX_MESSAGE_SIZE:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Message exceeds maximum size of {MAX_MESSAGE_SIZE} bytes"
                    })
                    continue

                # Rate limit messages
                now = time.monotonic()
                message_timestamps.append(now)
                if len(message_timestamps) >= MAX_MESSAGES_PER_SECOND:
                    oldest = message_timestamps[0]
                    if now - oldest < 1.0:  # More than MAX messages in 1 second
                        await websocket.send_json({
                            "type": "error",
                            "message": "Message rate limit exceeded"
                        })
                        continue

                # Parse JSON with error handling
                try:
                    message_data = json.loads(data)
                except json.JSONDecodeError as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Invalid JSON: {str(e)[:100]}"
                    })
                    continue

                # Validate message structure
                if not isinstance(message_data, dict):
                    await websocket.send_json({
                        "type": "error",
                        "message": "Message must be a JSON object"
                    })
                    continue

                # Handle different message types
                if message_data.get("type") == "chat":
                    # Handle chat message
                    try:
                        chat_data = message_data.get("data", {})
                        if not isinstance(chat_data, dict):
                            await websocket.send_json({
                                "type": "error",
                                "message": "chat data must be an object"
                            })
                            continue

                        request = ConversationRequest(**chat_data)
                        multiturn_agent = services.get("multiturn_agent")
                        if not multiturn_agent:
                            await websocket.send_json({
                                "type": "error",
                                "message": "Multi-turn agent not initialized"
                            })
                            continue

                        # Direct handling instead of calling endpoint that expects HTTP context
                        conversation_id = request.conversation_id
                        if conversation_id:
                            turns = await multiturn_agent.continue_conversation(
                                conversation_id, request.message, strategy=request.strategy
                            )
                            response_text = turns[-1]["content"] if turns else "No response"
                            context = multiturn_agent.get_conversation_summary(conversation_id)
                        else:
                            ctx = await multiturn_agent.start_conversation(
                                user_id=request.user_id or "ws_user",
                                initial_context=request.context,
                            )
                            response_text = await multiturn_agent.generate_multiturn_response(
                                ctx.conversation_id, request.message, strategy=request.strategy
                            )
                            ctx.add_turn({"role": "assistant", "content": response_text})
                            context = ctx.get_context_summary()
                            conversation_id = ctx.conversation_id

                        await websocket.send_json({
                            "type": "chat_response",
                            "data": {
                                "conversation_id": conversation_id,
                                "response": response_text,
                                "context": context,
                                "metadata": {"strategy": request.strategy}
                            }
                        })

                    except ValueError as e:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Invalid chat request: {str(e)[:200]}"
                        })
                    except Exception as e:
                        logger.error(f"WebSocket chat error: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": "Failed to process chat request"
                        })

                elif message_data.get("type") == "metrics":
                    # Send metrics (direct implementation instead of endpoint call)
                    try:
                        metrics = {
                            "system": {
                                "active_engines": len(active_engines),
                                "active_conversations": len(active_conversations),
                                "training_jobs": len(training_jobs),
                            },
                            "api": API_METRICS.snapshot(),
                        }
                        if "demo_engine" in services:
                            metrics["demo_engine"] = services["demo_engine"].get_metrics()
                        await websocket.send_json({"type": "metrics_response", "data": metrics})
                    except Exception as e:
                        logger.error(f"WebSocket metrics error: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": "Failed to retrieve metrics"
                        })

                elif message_data.get("type") == "ping":
                    # Ping/pong
                    await websocket.send_text(
                        json.dumps(
                            {"type": "pong", "timestamp": datetime.now().isoformat()}
                        )
                    )

                else:
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "error",
                                "message": f"Unknown message type: {message_data.get('type')}",
                            }
                        )
                    )

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await websocket.close()

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "monitoring": "monitoring" in services,
                "cache": "cache" in services,
                "demo_engine": "demo_engine" in services,
                "multiturn_agent": "multiturn_agent" in services,
            },
        }


# Main entry point
def main():
    """Main entry point for the service"""
    if not FASTAPI_AVAILABLE:
        logger.error("FastAPI is required but not available")
        return

    print("\n" + "=" * 80)
    print("🚀 ULTIMATE GRPO SERVICE - ALL INNOVATIONS INTEGRATED")
    print("=" * 80)
    print("\nInnovations included:")
    print("1. 🧠 Neural Reward Models - Learning from trajectory data")
    print("2. ⚖️ RULER LLM Judges - Sophisticated evaluation with custom rubrics")
    print("3. 💬 Multi-Turn Conversations - Advanced dialogue management")
    print("4. 🔄 Distributed Training - Multi-GPU scaling with fault tolerance")
    print("5. ⚡ Computational Engine - Parallel trajectory generation")
    print("6. 🎯 Multi-Objective Rewards - Sophisticated reward composition")
    print("7. 🚀 Auto-Scaling - Dynamic resource allocation")
    print("8. 📊 Real-time Metrics - Comprehensive monitoring")
    print("9. 🔌 WebSocket Support - Real-time interactions")
    print("10. 🛠️ Tool Integration - Extensible agent capabilities")
    print("\nPhilosophy: Computation > Hand-crafted Knowledge")
    print("=" * 80 + "\n")

    # Start the service
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")


if __name__ == "__main__":
    main()
