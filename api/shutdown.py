"""
Graceful Shutdown Handler

Manages graceful shutdown of API services with proper cleanup.
"""

import asyncio
import logging
import signal
import sys
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


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
    started_at: Optional[datetime] = None
    current_phase: Optional[ShutdownPhase] = None
    completed_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)
    in_flight_requests: int = 0


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
    ):
        """
        Initialize shutdown manager.

        Args:
            drain_timeout: Max time to wait for in-flight requests.
            total_timeout: Max total shutdown time.
        """
        self.drain_timeout = drain_timeout
        self.total_timeout = total_timeout
        self.state = ShutdownState()
        self._tasks: Dict[ShutdownPhase, List[ShutdownTask]] = {
            phase: [] for phase in ShutdownPhase
        }
        self._shutdown_event = asyncio.Event()
        self._signal_handlers_installed = False

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
        """Execute the shutdown sequence."""
        if self.state.is_shutting_down:
            logger.warning("Shutdown already in progress")
            return

        self.state.is_shutting_down = True
        self.state.started_at = datetime.utcnow()

        logger.info("Starting graceful shutdown sequence")

        try:
            for phase in ShutdownPhase:
                self.state.current_phase = phase
                logger.info("Shutdown phase: %s", phase.value)

                if phase == ShutdownPhase.DRAIN_REQUESTS:
                    await self._drain_requests()

                # Execute tasks for this phase
                tasks = self._tasks[phase]
                if tasks:
                    await self._execute_phase_tasks(tasks)

            logger.info("Graceful shutdown completed successfully")

        except Exception as e:
            logger.exception("Error during shutdown: %s", e)
            raise

        finally:
            self.state.current_phase = None

    async def _drain_requests(self) -> None:
        """Wait for in-flight requests to complete."""
        if self.state.in_flight_requests == 0:
            logger.info("No in-flight requests to drain")
            return

        logger.info(
            "Draining %d in-flight requests (timeout: %ds)",
            self.state.in_flight_requests,
            self.drain_timeout,
        )

        start = asyncio.get_event_loop().time()
        while self.state.in_flight_requests > 0:
            elapsed = asyncio.get_event_loop().time() - start
            if elapsed >= self.drain_timeout:
                logger.warning(
                    "Drain timeout reached with %d requests remaining",
                    self.state.in_flight_requests,
                )
                break

            await asyncio.sleep(0.1)

        logger.info("Request draining complete")

    async def _execute_phase_tasks(self, tasks: List[ShutdownTask]) -> None:
        """Execute all tasks for a shutdown phase."""
        for task in tasks:
            logger.debug("Running shutdown task: %s", task.name)

            try:
                await asyncio.wait_for(
                    task.callback(),
                    timeout=task.timeout_seconds,
                )
                self.state.completed_tasks.append(task.name)
                logger.info("Completed shutdown task: %s", task.name)

            except asyncio.TimeoutError:
                logger.warning(
                    "Shutdown task %s timed out after %.1fs",
                    task.name,
                    task.timeout_seconds,
                )
                self.state.failed_tasks.append(task.name)

                if task.critical:
                    raise RuntimeError(f"Critical shutdown task {task.name} timed out")

            except Exception as e:
                logger.exception("Shutdown task %s failed: %s", task.name, e)
                self.state.failed_tasks.append(task.name)

                if task.critical:
                    raise

    def get_status(self) -> Dict[str, Any]:
        """Get shutdown status."""
        return {
            "is_shutting_down": self.state.is_shutting_down,
            "started_at": (
                self.state.started_at.isoformat() if self.state.started_at else None
            ),
            "current_phase": (
                self.state.current_phase.value if self.state.current_phase else None
            ),
            "in_flight_requests": self.state.in_flight_requests,
            "completed_tasks": self.state.completed_tasks,
            "failed_tasks": self.state.failed_tasks,
        }


# Global singleton
_shutdown_manager: Optional[GracefulShutdownManager] = None


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
