"""Own API application generations, request admission, and dependency cleanup."""

from __future__ import annotations

import asyncio
import copy
import threading
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from stateset_agents.utils.async_calls import drain_owned_operation, run_async_owned

from .resource_lifecycle import ResourceOwnershipError

_SCOPES: ContextVar[tuple[object, ...]] = ContextVar(
    "application_ownership", default=()
)


@dataclass
class _Resource:
    value: Any
    close: Callable[[], Awaitable[None]]
    critical: bool


@dataclass
class ApplicationGeneration:
    """Capture one lifespan so its late exit cannot close a successor."""

    number: int
    loop: asyncio.AbstractEventLoop
    startup_task: asyncio.Task[Any] | None
    owner: ApplicationLifecycle = field(repr=False)
    state: str = "starting"
    phase: str | None = "configuration"
    resources: dict[str, _Resource] = field(default_factory=dict)
    startup_error_type: str | None = None
    serving_error_type: str | None = None
    cleanup_errors: list[dict[str, str]] = field(default_factory=list)
    completed_cleanup: list[str] = field(default_factory=list)
    cleanup_task: asyncio.Task[None] | None = None
    cleanup_start_error: BaseException | None = field(default=None, repr=False)


class ApplicationLifecycle:
    """Serialize lifespans and retain cleanup independently of close observers."""

    _ORDER = (
        "health_checker",
        "v1_conversation_store",
        "training_service",
        "cache",
        "database",
        "inference_service",
    )

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._generation: ApplicationGeneration | None = None
        self._active: set[object] = set()
        self._idle = asyncio.Event()
        self._idle.set()
        self._retired: dict[str, Any] = {}

    def _check_callback(self) -> None:
        if self in _SCOPES.get():
            raise ResourceOwnershipError(
                "Application callbacks cannot close or restart their own owner"
            )

    def begin(self) -> ApplicationGeneration:
        """Admit one new generation only after the previous cleanup succeeded."""
        self._check_callback()
        with self._lock:
            previous = self._generation
            if previous is not None and (
                previous.state != "closed"
                or previous.cleanup_task is None
                or not previous.cleanup_task.done()
                or previous.cleanup_task.cancelled()
                or previous.cleanup_task.exception() is not None
            ):
                raise ResourceOwnershipError(
                    f"Application is {previous.state}; another lifespan cannot start"
                )
            if self._active:
                raise ResourceOwnershipError(
                    "Application has active unmanaged requests"
                )
            self._idle = asyncio.Event()
            self._idle.set()
            generation = ApplicationGeneration(
                number=1 if previous is None else previous.number + 1,
                loop=asyncio.get_running_loop(),
                startup_task=asyncio.current_task(),
                owner=self,
            )
            self._generation = generation
            return generation

    @contextmanager
    def phase(self, generation: ApplicationGeneration, name: str) -> Iterator[None]:
        """Describe owned startup work and fence reentrant callbacks."""
        with self._lock:
            if self._generation is not generation or generation.state != "starting":
                raise ResourceOwnershipError(
                    "Application startup no longer owns this generation"
                )
            generation.phase = name
        token = _SCOPES.set((*_SCOPES.get(), self))
        try:
            yield
        finally:
            _SCOPES.reset(token)

    def was_retired(self, name: str, resource: Any) -> bool:
        """Check an exact successfully closed dependency, not just its name."""
        with self._lock:
            return resource is not None and self._retired.get(name) is resource

    def bind(
        self,
        generation: ApplicationGeneration,
        name: str,
        resource: Any,
        close: Callable[[], Awaitable[None]],
        *,
        critical: bool = True,
    ) -> None:
        """Capture cleanup before another startup step may fail."""
        with self._lock:
            if self._generation is not generation or generation.state != "starting":
                raise ResourceOwnershipError(
                    "Application dependency publication is stale"
                )
            if name not in self._ORDER:
                raise ValueError(f"Unsupported application dependency: {name}")
            if name in generation.resources:
                raise ResourceOwnershipError(
                    f"Application dependency {name} is already owned"
                )
            generation.resources[name] = _Resource(resource, close, critical)

    def serving(self, generation: ApplicationGeneration) -> None:
        """Open request admission only after every startup step has returned."""
        with self._lock:
            if self._generation is not generation or generation.state != "starting":
                raise ResourceOwnershipError("Application startup is stale")
            generation.state, generation.phase = "serving", None

    def record_error(
        self, generation: ApplicationGeneration, error: BaseException
    ) -> None:
        """Keep the primary failure separate from later cleanup failures."""
        with self._lock:
            if generation.state == "starting":
                generation.startup_error_type = type(error).__name__
            else:
                generation.serving_error_type = type(error).__name__

    @contextmanager
    def request(self) -> Iterator[None]:
        """Retain an admitted request through response and cancellation cleanup."""
        with self._lock:
            generation = self._generation
            if generation is not None:
                if generation.state != "serving":
                    raise ResourceOwnershipError(f"Application is {generation.state}")
                if generation.loop is not asyncio.get_running_loop():
                    raise ResourceOwnershipError(
                        "Application belongs to another event loop"
                    )
            identity = object()
            self._active.add(identity)
            self._idle.clear()
        token = _SCOPES.set((*_SCOPES.get(), self))
        try:
            yield
        finally:
            _SCOPES.reset(token)
            with self._lock:
                self._active.remove(identity)
                if not self._active:
                    self._idle.set()

    async def aclose(self, generation: ApplicationGeneration | None = None) -> None:
        """Join one shutdown; failed cleanup is retained without replay."""
        self._check_callback()
        with self._lock:
            selected = generation or self._generation
            if selected is None:
                return
            if selected.owner is not self:
                raise ResourceOwnershipError(
                    "Application generation belongs to another owner"
                )
            if selected.loop is not asyncio.get_running_loop():
                raise ResourceOwnershipError(
                    "Application cleanup belongs to another event loop"
                )
            if (
                selected.state == "starting"
                and selected.startup_task is not asyncio.current_task()
            ):
                raise ResourceOwnershipError(
                    "Cancel and join the owning startup task before closing"
                )
            if selected.cleanup_start_error is not None:
                raise selected.cleanup_start_error
            if selected.cleanup_task is None:
                selected.state, selected.phase = "stopping", "requests"
                work = self._shutdown(selected)
                try:
                    selected.cleanup_task = asyncio.create_task(work)
                except BaseException as error:
                    work.close()
                    selected.state, selected.phase = "failed", None
                    selected.cleanup_start_error = error
                    selected.cleanup_errors.append(
                        {"component": "shutdown", "error_type": type(error).__name__}
                    )
                    raise
            task = selected.cleanup_task

        async def join() -> None:
            await asyncio.shield(task)

        await drain_owned_operation(join())

    async def _shutdown(self, generation: ApplicationGeneration) -> None:
        failure: BaseException | None = None
        await self._idle.wait()
        for name in self._ORDER:
            resource = generation.resources.get(name)
            if resource is None:
                continue
            with self._lock:
                generation.phase = name
            token = _SCOPES.set((*_SCOPES.get(), self))
            try:
                await resource.close()
            except (Exception, asyncio.CancelledError) as error:
                with self._lock:
                    generation.cleanup_errors.append(
                        {"component": name, "error_type": type(error).__name__}
                    )
                if failure is None:
                    failure = error
                if resource.critical:
                    break
            else:
                with self._lock:
                    generation.completed_cleanup.append(name)
                    self._retired[name] = resource.value
            finally:
                _SCOPES.reset(token)
        with self._lock:
            generation.state = "failed" if failure is not None else "closed"
            generation.phase = None
        if failure is not None:
            raise failure

    def get_status(self) -> dict[str, Any]:
        """Return detached lifecycle evidence with no credentials or request data."""
        with self._lock:
            generation = self._generation
            if generation is None:
                return {
                    "generation": 0,
                    "state": "new",
                    "phase": None,
                    "active_requests": len(self._active),
                    "cleanup_complete": False,
                    "startup_error_type": None,
                    "serving_error_type": None,
                    "cleanup_errors": [],
                    "completed_cleanup": [],
                    "pending_cleanup": [],
                }
            return copy.deepcopy(
                {
                    "generation": generation.number,
                    "state": generation.state,
                    "phase": generation.phase,
                    "active_requests": len(self._active),
                    "cleanup_complete": generation.state == "closed",
                    "startup_error_type": generation.startup_error_type,
                    "serving_error_type": generation.serving_error_type,
                    "cleanup_errors": generation.cleanup_errors,
                    "completed_cleanup": generation.completed_cleanup,
                    "pending_cleanup": [
                        name
                        for name in self._ORDER
                        if name in generation.resources
                        and name not in generation.completed_cleanup
                    ],
                }
            )


class ApplicationLifecycleMiddleware:
    """Fence routes and streams while keeping passive lifecycle/liveness available."""

    def __init__(self, app: ASGIApp, owner: ApplicationLifecycle) -> None:
        self.app, self.owner = app, owner

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Retain the complete ASGI callback through cancellation cleanup."""
        if scope["type"] not in {"http", "websocket"} or (
            scope["type"] == "http"
            and scope.get("path") in {"/live", "/healthz", "/lifecycle"}
        ):
            await self.app(scope, receive, send)
            return
        # Only admission errors become 503. A route's ownership error must still
        # pass through its normal exception handling after admission.
        admission = self.owner.request()
        try:
            admission.__enter__()
        except ResourceOwnershipError:
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1013})
            else:
                state = self.owner.get_status()["state"]
                content = (
                    {"status": "not_ready", "reason": f"Application is {state}"}
                    if scope.get("path") == "/ready"
                    else {"detail": f"Application is {state}"}
                )
                await JSONResponse(content, status_code=503)(scope, receive, send)
            return
        try:

            async def dispatch() -> None:
                await self.app(scope, receive, send)

            await run_async_owned(dispatch())
        finally:
            admission.__exit__(None, None, None)
