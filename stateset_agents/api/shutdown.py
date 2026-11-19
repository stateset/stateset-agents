"""
Graceful Shutdown Handler

Manages graceful shutdown of API services with proper cleanup.
"""

import asyncio
import logging
import math
import signal
import sys
from collections.abc import Callable, Coroutine
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from time import monotonic
from typing import Any

from stateset_agents.exceptions import INFERENCE_EXCEPTIONS
from stateset_agents.utils.async_calls import drain_owned_operation

logger = logging.getLogger(__name__)

SHUTDOWN_EXCEPTIONS = INFERENCE_EXCEPTIONS


class ShutdownPhase(Enum):
    """Shutdown phases in order of execution."""

    STOP_ACCEPTING = "stop_accepting"  # Stop accepting new requests
    DRAIN_REQUESTS = "drain_requests"  # Wait for in-flight requests
    CLEANUP_JOBS = "cleanup_jobs"  # Cancel/complete running jobs
    CLEANUP_CONNECTIONS = "cleanup_connections"  # Close WebSocket connections
    CLEANUP_STATE = "cleanup_state"  # Cleanup state and caches
    CLEANUP_ENGINES = "cleanup_engines"  # Cleanup computational engines
    FINAL = "final"  # Final cleanup


@dataclass
class ShutdownTask:
    """A task to run during shutdown."""

    name: str
    phase: ShutdownPhase
    callback: Callable[[], Coroutine[Any, Any, None]]
    timeout_seconds: float = 30.0
    critical: bool = False  # If True, failure aborts shutdown


@dataclass
class ShutdownState:
    """Current shutdown state."""

    is_shutting_down: bool = False
    started_at: datetime | None = None
    current_phase: ShutdownPhase | None = None
    completed_tasks: list[str] = field(default_factory=list)
    failed_tasks: list[str] = field(default_factory=list)
    in_flight_requests: int = 0
    deadline_exceeded: bool = False


# Context variable for tracking in-flight requests
_in_flight_count: ContextVar[int] = ContextVar("in_flight_count", default=0)


class GracefulShutdownManager:
    """
    Manages graceful shutdown of the application.

    Supports:
    - Signal handling (SIGTERM, SIGINT)
    - Phased shutdown with timeouts
    - In-flight request tracking
    - Async cleanup callbacks
    """

    # Default timeout for each phase
    DEFAULT_PHASE_TIMEOUT = 30.0

    def __init__(
        self,
        drain_timeout: float = 30.0,
        total_timeout: float = 120.0,
    ) -> None:
        """
        Initialize shutdown manager.

        Args:
            drain_timeout: Max time to wait for in-flight requests.
            total_timeout: Shared time budget for all phases. Expiry cancels the
                active callback and prevents later callbacks from starting, but
                still waits for cancellation cleanup. Zero admits no callbacks;
                positive infinity disables the total deadline.
        """
        if math.isnan(total_timeout) or total_timeout < 0:
            raise ValueError("total_timeout must be nonnegative and not NaN")
        self.drain_timeout = drain_timeout
        self.total_timeout = total_timeout
        self.state = ShutdownState()
        self._tasks: dict[ShutdownPhase, list[ShutdownTask]] = {
            phase: [] for phase in ShutdownPhase
        }
        self._shutdown_event = asyncio.Event()
        self._signal_handlers_installed = False
        self._shutdown_task: asyncio.Task[None] | None = None
        self._deadline: float | None = None

    def register_task(
        self,
        name: str,
        phase: ShutdownPhase,
        callback: Callable[[], Coroutine[Any, Any, None]],
        timeout_seconds: float = 30.0,
        critical: bool = False,
    ) -> None:
        """
        Register a shutdown task.

        Args:
            name: Task name for logging.
            phase: Shutdown phase to run in.
            callback: Async function to call.
            timeout_seconds: Task timeout.
            critical: If True, failure aborts shutdown.
        """
        if self._shutdown_task is not None:
            raise RuntimeError("Cannot register tasks after shutdown starts")
        task = ShutdownTask(
            name=name,
            phase=phase,
            callback=callback,
            timeout_seconds=timeout_seconds,
            critical=critical,
        )
        self._tasks[phase].append(task)
        logger.debug("Registered shutdown task: %s (phase: %s)", name, phase.value)

    def install_signal_handlers(self) -> None:
        """Install signal handlers for graceful shutdown."""
        if self._signal_handlers_installed:
            return

        loop = asyncio.get_event_loop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self._handle_signal, sig)
                logger.debug("Installed signal handler for %s", sig.name)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                signal.signal(sig, self._sync_handle_signal)
                logger.debug("Installed sync signal handler for %s", sig.name)

        self._signal_handlers_installed = True

    def _handle_signal(self, sig: signal.Signals) -> None:
        """Handle shutdown signal (async context)."""
        logger.info("Received signal %s, initiating graceful shutdown", sig.name)
        self._shutdown_event.set()

    def _sync_handle_signal(self, sig: int, frame: Any) -> None:
        """Handle shutdown signal (sync context)."""
        logger.info("Received signal %d, initiating graceful shutdown", sig)
        # Schedule shutdown in the event loop
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(self._shutdown_event.set)
        except RuntimeError:
            # No running loop, exit directly
            sys.exit(0)

    def request_shutdown(self) -> None:
        """Request application shutdown."""
        logger.info("Shutdown requested programmatically")
        self._shutdown_event.set()

    @property
    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress."""
        return self.state.is_shutting_down

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()

    def increment_in_flight(self) -> None:
        """Increment in-flight request count."""
        self.state.in_flight_requests += 1

    def decrement_in_flight(self) -> None:
        """Decrement in-flight request count."""
        self.state.in_flight_requests = max(0, self.state.in_flight_requests - 1)

    async def execute_shutdown(self) -> None:
        """Join one shutdown sequence, retaining cleanup through caller cancellation."""
        if self._shutdown_task is None:
            self.state.is_shutting_down = True
            self.state.started_at = datetime.utcnow()
            self._deadline = monotonic() + self.total_timeout
            self._shutdown_event.set()
            self._shutdown_task = asyncio.create_task(self._execute_shutdown())
        task = self._shutdown_task

        async def join() -> None:
            await task

        await drain_owned_operation(join())

    def _remaining_time(self) -> float:
        """Return the shared budget, failing before another effect can start."""
        if self._deadline is None:
            return self.total_timeout
        remaining = self._deadline - monotonic()
        if remaining <= 0:
            self.state.deadline_exceeded = True
            raise asyncio.TimeoutError("Total shutdown timeout exceeded")
        return remaining

    async def _execute_shutdown(self) -> None:
        """Run phases once; repeated callers observe this same outcome."""
        logger.info("Starting graceful shutdown sequence")

        try:
            for phase in ShutdownPhase:
                self._remaining_time()
                self.state.current_phase = phase
                logger.info("Shutdown phase: %s", phase.value)

                if phase == ShutdownPhase.DRAIN_REQUESTS:
                    await self._drain_requests()

                # Execute tasks for this phase
                tasks = self._tasks[phase]
                if tasks:
                    await self._execute_phase_tasks(tasks)

            self._remaining_time()
            if self.state.failed_tasks:
                logger.warning(
                    "Shutdown completed with failed tasks: %s", self.state.failed_tasks
                )
            else:
                logger.info("Graceful shutdown completed successfully")

        except SHUTDOWN_EXCEPTIONS as e:
            logger.exception("Error during shutdown: %s", e)
            raise

        finally:
            if self._deadline is not None and monotonic() >= self._deadline:
                self.state.deadline_exceeded = True
            self.state.current_phase = None

    async def _drain_requests(self) -> None:
        """Wait for in-flight requests to complete."""
        if self.state.in_flight_requests == 0:
            logger.info("No in-flight requests to drain")
            return

        logger.info(
            "Draining %d in-flight requests (timeout: %.1fs)",
            self.state.in_flight_requests,
            self.drain_timeout,
        )

        drain_deadline = monotonic() + self.drain_timeout
        while self.state.in_flight_requests > 0:
            try:
                remaining = self._remaining_time()
            except asyncio.TimeoutError:
                self.state.failed_tasks.append(ShutdownPhase.DRAIN_REQUESTS.value)
                raise
            drain_remaining = drain_deadline - monotonic()
            if drain_remaining <= 0:
                logger.warning(
                    "Drain timeout reached with %d requests remaining",
                    self.state.in_flight_requests,
                )
                self.state.failed_tasks.append(ShutdownPhase.DRAIN_REQUESTS.value)
                break

            await asyncio.sleep(min(0.1, drain_remaining, remaining))

        logger.info("Request draining complete")

    async def _invoke_task(self, task: ShutdownTask) -> None:
        """Check admission again after scheduling, before callback effects."""
        self._remaining_time()
        await task.callback()

    async def _execute_phase_tasks(self, tasks: list[ShutdownTask]) -> None:
        """Execute all tasks for a shutdown phase."""
        for task in tasks:
            timeout = min(task.timeout_seconds, self._remaining_time())
            logger.debug("Running shutdown task: %s", task.name)

            try:
                await asyncio.wait_for(
                    self._invoke_task(task),
                    timeout=timeout,
                )
                self.state.completed_tasks.append(task.name)
                logger.info("Completed shutdown task: %s", task.name)

            except asyncio.TimeoutError as exc:
                logger.warning(
                    "Shutdown task %s timed out after %.1fs",
                    task.name,
                    timeout,
                )
                self.state.failed_tasks.append(task.name)
                self._remaining_time()

                if task.critical:
                    raise RuntimeError(
                        f"Critical shutdown task {task.name} timed out"
                    ) from exc

            except asyncio.CancelledError:
                self.state.failed_tasks.append(task.name)
                raise
            except Exception as e:
                logger.exception("Shutdown task %s failed: %s", task.name, e)
                self.state.failed_tasks.append(task.name)

                if task.critical or not isinstance(e, SHUTDOWN_EXCEPTIONS):
                    raise

            # A callback may suppress cancellation or block the loop. Neither
            # grants a fresh budget or authorizes a later cleanup phase.
            self._remaining_time()

    def get_status(self) -> dict[str, Any]:
        """Get shutdown status."""
        task = self._shutdown_task
        if task is None:
            outcome = "not_started"
        elif not task.done():
            outcome = "running"
        elif task.cancelled() or task.exception() is not None:
            outcome = "failed"
        else:
            outcome = (
                "completed" if not self.state.failed_tasks else "completed_with_errors"
            )
        return {
            "outcome": outcome,
            "deadline_exceeded": self.state.deadline_exceeded
            or (
                outcome == "running"
                and self._deadline is not None
                and monotonic() >= self._deadline
            ),
            "is_shutting_down": self.state.is_shutting_down,
            "started_at": (
                self.state.started_at.isoformat() if self.state.started_at else None
            ),
            "current_phase": (
                self.state.current_phase.value if self.state.current_phase else None
            ),
            "in_flight_requests": self.state.in_flight_requests,
            "completed_tasks": list(self.state.completed_tasks),
            "failed_tasks": list(self.state.failed_tasks),
        }


# Global singleton
_shutdown_manager: GracefulShutdownManager | None = None


def get_shutdown_manager(
    drain_timeout: float = 30.0,
    total_timeout: float = 120.0,
) -> GracefulShutdownManager:
    """Get the global shutdown manager instance."""
    global _shutdown_manager
    if _shutdown_manager is None:
        _shutdown_manager = GracefulShutdownManager(
            drain_timeout=drain_timeout,
            total_timeout=total_timeout,
        )
    return _shutdown_manager


def reset_shutdown_manager() -> None:
    """Reset shutdown manager (for testing)."""
    global _shutdown_manager
    _shutdown_manager = None
