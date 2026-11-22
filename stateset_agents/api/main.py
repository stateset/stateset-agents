import asyncio
import copy
import inspect
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

from stateset_agents.utils.async_calls import run_async_owned

from .application_lifecycle import ApplicationLifecycle, ApplicationLifecycleMiddleware
from .config import ConfigurationError, get_config
from .errors import setup_exception_handlers
from .middleware import setup_middleware
from .openapi import add_documentation_routes, setup_openapi
from .resilience import HealthChecker, HealthStatus, get_all_circuit_stats
from .resource_lifecycle import ResourceOwnershipError
from .routers import (
    agents,
    distributed_rollouts,
    finetune,
    messages,
    metrics,
    openai,
    training,
    training_lab,
    v1,
)
from .services.inference_service import InferenceConfig, InferenceService

logger = logging.getLogger(__name__)

API_LIFESPAN_EXCEPTIONS = (ImportError, OSError, RuntimeError, ValueError)


class LazyApp:
    """ASGI-compatible proxy that creates the FastAPI app on first use."""

    def __init__(self) -> None:
        self._app: FastAPI | None = None

    def _get_app(self) -> FastAPI:
        if self._app is None:
            self._app = create_app()
        return self._app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        await self._get_app()(scope, receive, send)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_app(), name)


def _lifecycle(app: FastAPI) -> ApplicationLifecycle:
    owner = getattr(app.state, "lifecycle", None)
    if owner is None:
        owner = ApplicationLifecycle()
        app.state.lifecycle = owner
    if not isinstance(owner, ApplicationLifecycle):
        raise ResourceOwnershipError(
            "app.state.lifecycle must be an ApplicationLifecycle"
        )
    return owner


def _restart_dependencies(app: FastAPI, owner: ApplicationLifecycle) -> None:
    from .conversation_store import ConversationStore
    from .services.training_service import TrainingService

    supported = {
        "health_checker": HealthChecker,
        "inference_service": InferenceService,
        "training_service": TrainingService,
        "v1_conversation_store": ConversationStore,
    }
    factories = getattr(app.state, "lifecycle_factories", {})
    if not isinstance(factories, dict) or set(factories) - supported.keys():
        raise ValueError(
            "lifecycle_factories must map supported dependency names to factories"
        )
    if any(not callable(factory) for factory in factories.values()):
        raise TypeError("Lifecycle factories must be callable")
    # Validate all custom replacements before constructing anything. Each new
    # resource is then bound by the caller immediately after construction.
    for name, native_type in supported.items():
        current = getattr(app.state, name, None)
        if owner.was_retired(name, current) and type(current) is not native_type:
            if name not in factories or not callable(factories[name]):
                raise ResourceOwnershipError(
                    f"Replace closed custom {name} or provide app.state.lifecycle_factories[{name!r}]"
                )


def _dependency(
    app: FastAPI, owner: ApplicationLifecycle, name: str, default: Any
) -> Any:
    from .conversation_store import ConversationStore
    from .services.training_service import TrainingService

    current = getattr(app.state, name, None)
    if current is None or owner.was_retired(name, current):
        factory = getattr(app.state, "lifecycle_factories", {}).get(name)
        if factory is not None:
            replacement = factory()
            if inspect.isawaitable(replacement):
                if inspect.iscoroutine(replacement):
                    replacement.close()
                raise TypeError(
                    "Lifecycle factories must construct resources synchronously"
                )
            if replacement is None or replacement is current:
                raise ResourceOwnershipError(
                    "Lifecycle factories must return a fresh resource"
                )
            current = replacement
        elif current is None:
            current = default()
        elif type(current) is HealthChecker or type(current) is TrainingService:
            current = current.clone_for_restart()
        elif type(current) is InferenceService:
            current = InferenceService(copy.deepcopy(current.config))
        elif type(current) is ConversationStore:
            path = (
                getattr(app.state.config, "v1_conversation_db_path", None)
                or current.path
            )
            current = ConversationStore(
                path, ttl_seconds=current.ttl_seconds, max_items=current.max_items
            )
        else:
            raise ResourceOwnershipError(
                f"Closed custom {name} requires a fresh resource"
            )
    setattr(app.state, name, current)
    return current


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Own one complete application generation, including rollback and request drain."""
    owner = _lifecycle(app)
    generation = owner.begin()  # Rejected overlap must not clean up the active app.
    primary: BaseException | None = None
    try:
        with owner.phase(generation, "configuration"):
            config = getattr(app.state, "config", None)
            if config is None:
                config = get_config()
                app.state.config = config
            strict_startup = (
                os.getenv("STATESET_AGENTS_STRICT_STARTUP", "false").lower()
                in {"1", "true", "on", "yes"}
                or config.is_production()
            )
            _restart_dependencies(app, owner)

        with owner.phase(generation, "dependencies"):
            health_checker = _dependency(app, owner, "health_checker", HealthChecker)
            close_health = getattr(health_checker, "aclose", None)
            if close_health is not None:
                owner.bind(generation, "health_checker", health_checker, close_health)
            inference_service = _dependency(
                app,
                owner,
                "inference_service",
                lambda: InferenceService(InferenceConfig.from_env()),
            )
            owner.bind(
                generation,
                "inference_service",
                inference_service,
                inference_service.aclose,
                critical=False,
            )
            # Retain injected training/journal owners even if an earlier startup
            # step fails. Missing services are constructed at their normal phase.
            for name in ("training_service", "v1_conversation_store"):
                if getattr(app.state, name, None) is not None:
                    resource = _dependency(app, owner, name, lambda: None)
                    owner.bind(generation, name, resource, resource.aclose)

        logger.info("Starting StateSet Agents API v%s", config.api_version)
        with owner.phase(generation, "cache"):
            cache_status = None
            try:
                from .distributed_cache import (
                    CacheConfig,
                    acquire_cache,
                    get_cache_status,
                )

                cache_status = get_cache_status
                cache_lease = await acquire_cache(CacheConfig.from_env())
                owner.bind(
                    generation, "cache", cache_lease, cache_lease.aclose, critical=False
                )
            except ResourceOwnershipError:
                raise
            except API_LIFESPAN_EXCEPTIONS as error:
                if strict_startup or (
                    cache_status is not None and cache_status()["state"] == "failed"
                ):
                    raise
                logger.warning(
                    "Cache initialization skipped (%s)", type(error).__name__
                )

        with owner.phase(generation, "database"):
            database_status = None
            try:
                from .persistence import (
                    DatabaseConfig,
                    acquire_database,
                    get_database_status,
                )

                database_status = get_database_status
                database_lease = await acquire_database(DatabaseConfig.from_env())
                owner.bind(
                    generation,
                    "database",
                    database_lease,
                    database_lease.aclose,
                    critical=False,
                )
            except ResourceOwnershipError:
                raise
            except API_LIFESPAN_EXCEPTIONS as error:
                if strict_startup or (
                    database_status is not None
                    and database_status()["state"] == "failed"
                ):
                    raise
                logger.warning(
                    "Database initialization skipped (%s)", type(error).__name__
                )

        with owner.phase(generation, "conversations"):
            conversation_path = getattr(config, "v1_conversation_db_path", None)
            if (
                conversation_path
                or "v1_conversation_store"
                in getattr(app.state, "lifecycle_factories", {})
            ) and getattr(app.state, "v1_conversation_store", None) is None:
                from .conversation_store import ConversationStore
                from .routers.v1 import (
                    V1_CONVERSATION_TTL_SECONDS,
                    V1_MAX_CONVERSATIONS,
                )

                def create_store() -> ConversationStore:
                    if conversation_path is None:
                        raise ValueError("A conversation journal path is required")
                    return ConversationStore(
                        conversation_path,
                        ttl_seconds=V1_CONVERSATION_TTL_SECONDS,
                        max_items=V1_MAX_CONVERSATIONS,
                    )

                store = _dependency(
                    app,
                    owner,
                    "v1_conversation_store",
                    create_store,
                )
                owner.bind(generation, "v1_conversation_store", store, store.aclose)

        with owner.phase(generation, "agent"):
            from stateset_agents.utils.security import (
                SecurityMonitor as _SecurityMonitor,
            )

            from .services.agent_service import AgentService

            agent_service = getattr(app.state, "agent_service", None)
            if agent_service is None:
                agent_service = AgentService(_SecurityMonitor())
                app.state.agent_service = agent_service

        with owner.phase(generation, "checkpoint"):
            default_ckpt = os.environ.get("STATESET_DEFAULT_CHECKPOINT")
            if default_ckpt:
                default_base = os.environ.get("STATESET_DEFAULT_BASE_MODEL")
                previous = getattr(app.state, "_default_checkpoint_binding", None)
                current_agent = getattr(agent_service, "agents", {}).get("default")
                request = (agent_service, default_ckpt, default_base)
                if (
                    previous is None
                    or previous[:3] != request
                    or previous[3] is not current_agent
                ):
                    try:
                        await run_async_owned(
                            agent_service.register_default_checkpoint_agent(
                                checkpoint_path=default_ckpt,
                                base_model=default_base,
                            )
                        )
                        registered = getattr(agent_service, "agents", {}).get("default")
                        if registered is not None:
                            app.state._default_checkpoint_binding = (
                                *request,
                                registered,
                            )
                    except Exception as error:
                        logger.error(
                            "Default checkpoint initialization failed (%s)",
                            type(error).__name__,
                        )

        with owner.phase(generation, "training"):
            from .services.training_service import TrainingService

            if getattr(app.state, "training_service", None) is None:
                training_service = _dependency(
                    app,
                    owner,
                    "training_service",
                    lambda: TrainingService(
                        max_concurrent_jobs=getattr(
                            config, "training_max_concurrent_jobs", 1
                        ),
                        queue_path=getattr(config, "training_queue_path", None),
                    ),
                )
                owner.bind(
                    generation,
                    "training_service",
                    training_service,
                    training_service.aclose,
                )

        with owner.phase(generation, "health_checks"):
            health_checker.add_check("api", lambda: True)
            health_checker.add_check(
                "inference_backend", inference_service.check_health
            )
        app.state.started_at = time.time()
        owner.serving(generation)
        yield
    except BaseException as error:
        primary = error
        owner.record_error(generation, error)
        raise
    finally:
        logger.info("Shutting down StateSet Agents API")
        try:
            await owner.aclose(generation)
        except (Exception, asyncio.CancelledError) as cleanup_error:
            if primary is None:
                raise
            logger.warning(
                "Application cleanup failed while preserving the primary error (%s)",
                type(cleanup_error).__name__,
            )


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    config = get_config()

    for warning in config.validate():
        logger.warning("Configuration warning: %s", warning)

    if config.is_production() and config.has_no_auth_credentials():
        raise ConfigurationError(
            "Authentication is required but no credential source is configured "
            "(no API keys and no JWT secret). Set API_KEYS or API_JWT_SECRET, "
            "or disable auth with API_REQUIRE_AUTH=false."
        )

    health_checker = HealthChecker()
    app = FastAPI(
        title=config.title,
        description="""
        ## StateSet Agents API

        A comprehensive REST API for training and deploying AI agents using
        reinforcement learning techniques.

        ### Features

        - 🤖 **Agent Management**: Create and manage AI agents
        - 💬 **Conversations**: Interactive conversations with agents
        - 🎯 **Training**: Train agents using reinforcement learning
        - 📊 **Monitoring**: Real-time metrics and monitoring
        - 🔒 **Security**: Authentication and authorization
        - 📈 **Analytics**: Performance analytics and insights
        """,
        version=config.api_version,
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.health_checker = health_checker
    owner = _lifecycle(app)
    app.state.started_at = time.time()
    if not getattr(app.state, "inference_service", None):
        app.state.inference_service = InferenceService(InferenceConfig.from_env())

    # Middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(ApplicationLifecycleMiddleware, owner=owner)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors.allowed_origins,
        allow_credentials=config.cors.allow_credentials,
        allow_methods=config.cors.allowed_methods,
        allow_headers=config.cors.allowed_headers,
        max_age=config.cors.max_age,
    )

    # Advanced Middleware (Rate Limiting, Security Headers, etc.)
    setup_middleware(app)

    # Include routers
    app.include_router(agents.router)
    app.include_router(agents.conversation_router)
    app.include_router(training.router)
    app.include_router(finetune.router)
    app.include_router(metrics.router)
    app.include_router(v1.router)
    app.include_router(messages.router)
    app.include_router(openai.router)
    app.include_router(distributed_rollouts.router)
    if config.enable_training_lab:
        app.include_router(training_lab.router)
        app.include_router(training_lab.ws_router)

    # Mount the interactive 3D Weight Studio (non-v1 browser workbench)
    try:
        from stateset_agents.weight_studio.router import mount_studio_ui

        mount_studio_ui(app)
    except Exception as error:  # noqa: BLE001 - optional static/studio mount
        logger.debug("Weight Studio mount skipped (%s)", type(error).__name__)

    # Compatibility aliases for legacy tests/clients
    app.add_api_route(
        "/conversation",
        v1.chat_v1,
        methods=["POST"],
        response_model=v1.ConversationResponseV1,
        include_in_schema=False,
    )

    # Setup OpenAPI documentation
    setup_openapi(app)
    add_documentation_routes(app)

    # Setup exception handlers
    setup_exception_handlers(app)

    # Root endpoint
    @app.get(
        "/",
        summary="API Root",
        description="Welcome endpoint with API information and available endpoints.",
        tags=["health"],
    )
    async def root() -> dict[str, Any]:
        """Get API information."""
        return {
            "message": "Welcome to StateSet Agents API",
            "version": config.api_version,
            "documentation": "/docs",
            "openapi": "/openapi.json",
            "health": "/health",
            "endpoints": {
                "agents": "/agents",
                "conversations": "/conversations",
                "training": "/training",
                "metrics": "/metrics",
                "messages": "/v1/messages",
                "chat_completions": "/v1/chat/completions",
            },
        }

    # Ready endpoint for Kubernetes
    @app.get(
        "/ready",
        summary="Readiness Check",
        description="Check if the API is ready to receive traffic.",
        tags=["health"],
    )
    async def readiness_check(request: Request) -> Any:
        """Kubernetes readiness probe endpoint."""
        try:
            checker = getattr(request.app.state, "health_checker", None)
            results = (
                await asyncio.wait_for(
                    checker.check_all(), timeout=metrics.HEALTH_TIMEOUT_SECONDS
                )
                if checker is not None
                else {}
            )
            overall_status = HealthChecker.aggregate_status(results)
        except Exception:
            overall_status = HealthStatus.UNHEALTHY

        if overall_status == HealthStatus.UNHEALTHY:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": "Dependencies unhealthy"},
            )

        return {"status": "ready"}

    # Liveness endpoint for Kubernetes
    @app.get(
        "/live",
        summary="Liveness Check",
        description="Check if the API is alive.",
        tags=["health"],
    )
    async def liveness_check() -> dict[str, Any]:
        """Kubernetes liveness probe endpoint."""
        return {
            "status": "alive",
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

    @app.get("/lifecycle", summary="Application Lifecycle", tags=["health"])
    async def application_lifecycle(request: Request) -> dict[str, Any]:
        """Inspect application ownership without starting dependency probes."""
        return _lifecycle(request.app).get_status()

    # Circuit breaker status endpoint
    @app.get(
        "/circuits",
        summary="Circuit Breaker Status",
        description="Get the status of all circuit breakers.",
        tags=["metrics"],
    )
    async def circuit_status() -> dict[str, Any]:
        """Get circuit breaker status for monitoring."""
        return {
            "circuits": get_all_circuit_stats(),
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

    return app


app = LazyApp()

if __name__ == "__main__":
    import uvicorn

    config = get_config()
    uvicorn.run(
        "stateset_agents.api.main:app",
        host=config.host,
        port=config.port,
        reload=not config.is_production(),
        log_level=config.observability.log_level.lower(),
    )
