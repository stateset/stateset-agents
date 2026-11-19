"""
Comprehensive Monitoring System for GRPO Agent Framework

This module provides advanced monitoring capabilities including metrics collection,
alerting, health checks, and integration with external monitoring systems.
"""

from __future__ import annotations

import asyncio
import contextvars
import copy
import inspect
import logging
import math
import statistics
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, cast

from stateset_agents.utils.async_calls import drain_owned_operation, run_sync_owned

# Optional imports for enhanced functionality
try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

try:
    from prometheus_client import REGISTRY, Counter, Gauge, Histogram, start_http_server

    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


def _prometheus_get_or_create(
    metric_cls: Any, name: str, documentation: str, **kwargs: Any
) -> Any:
    """
    Create a Prometheus metric or reuse an existing one if already registered.

    `prometheus_client` uses a process-global default registry. In tests and
    multi-service processes we may instantiate monitoring helpers multiple times.
    Registering the same metric twice raises ValueError, so we reuse the existing
    collector when present.
    """
    if not HAS_PROMETHEUS:  # pragma: no cover
        raise RuntimeError("Prometheus client not installed")

    try:
        return metric_cls(name, documentation, **kwargs)
    except ValueError:
        # Registry keys include a base name without `_total` for Counters.
        base_name = name[:-6] if name.endswith("_total") else name
        collector = REGISTRY._names_to_collectors.get(
            base_name
        ) or REGISTRY._names_to_collectors.get(name)
        if collector is None or not isinstance(collector, metric_cls):
            raise
        return collector


PSUTIL_EXCEPTIONS: tuple[type[BaseException], ...] = ()
if HAS_PSUTIL and psutil is not None:
    PSUTIL_EXCEPTIONS = (psutil.Error,)

MONITORING_EXCEPTIONS: tuple[type[BaseException], ...] = (
    RuntimeError,
    ValueError,
    TypeError,
    OSError,
    asyncio.TimeoutError,
)


class MetricType(Enum):
    """Types of metrics"""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Metric data structure"""

    name: str
    type: MetricType
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    help_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "type": self.type.value,
            "value": self.value,
            "labels": self.labels,
            "timestamp": self.timestamp.isoformat(),
            "help_text": self.help_text,
        }


@dataclass
class Alert:
    """Alert data structure"""

    id: str
    name: str
    severity: AlertSeverity
    message: str
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
            "labels": self.labels,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


class UtilityHealthState(str, Enum):
    """Admission and cleanup ownership of utility health checks."""

    READY = "ready"
    CLOSING = "closing"
    CLOSED = "closed"
    FAILED = "failed"


_health_probe_context: contextvars.ContextVar[tuple[object, ...]] = (
    contextvars.ContextVar("utility_health_probe_owners", default=())
)


@dataclass(eq=False)
class _UtilityHealthProbe:
    name: str
    critical: bool
    loop: asyncio.AbstractEventLoop
    task: asyncio.Task[dict[str, Any]] | None = None
    observers: int = 0
    cancel_requested: bool = False
    waiting_for_sync: bool = False
    worker_ident: int | None = None
    executor: ThreadPoolExecutor | None = None
    done: threading.Event = field(default_factory=threading.Event)
    cleanup_error: BaseException | None = None


class _HealthOwner:
    """Share one blocking cleanup outcome, with an asynchronous observer API."""

    def __init__(self) -> None:
        self._health_lock = threading.Condition(threading.RLock())
        self._health_state = UtilityHealthState.READY
        self._health_close_started = False
        self._health_close_done = False
        self._health_close_error: BaseException | None = None
        self._health_closer_ident: int | None = None

    @property
    def state(self) -> UtilityHealthState:
        """Inspect admission without invoking a dependency."""
        with self._health_lock:
            return self._health_state

    def _owned_probes(self) -> list[_UtilityHealthProbe]:
        raise NotImplementedError

    def _owns_current_execution(self) -> bool:
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        with self._health_lock:
            if (
                self._health_close_started
                and not self._health_close_done
                and self._health_closer_ident == threading.get_ident()
            ):
                return True
            return any(
                (current is not None and probe.task is current)
                or probe.worker_ident == threading.get_ident()
                or probe in _health_probe_context.get()
                for probe in self._owned_probes()
            )

    def _assert_not_probe(self) -> None:
        if self._owns_current_execution():
            raise RuntimeError("A health probe cannot close or await its own owner")

    def _assert_can_close(self) -> None:
        self._assert_not_probe()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if any(probe.loop is loop for probe in self._owned_probes()):
            raise RuntimeError("Use aclose() to drain health probes on this event loop")

    def _stop_admission(self) -> None:
        with self._health_lock:
            if not self._health_close_done:
                self._health_state = UtilityHealthState.CLOSING

    def get_status(self) -> dict[str, Any]:
        """Return a detached view of probe admission and retained work."""
        with self._health_lock:
            probes = self._owned_probes()
            return {
                "state": self._health_state.value,
                "active_probes": len(probes),
                "cancelling_probes": sum(probe.cancel_requested for probe in probes),
                "observers": sum(probe.observers for probe in probes),
                "pending_checks": [probe.name for probe in probes],
                "cleanup_error_type": (
                    type(self._health_close_error).__name__
                    if self._health_close_error
                    else None
                ),
            }

    def _close_owned(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        """Stop admission and join probes once, retaining cleanup failures."""
        self._assert_can_close()
        with self._health_lock:
            if self._health_close_started:
                while not self._health_close_done:
                    self._health_lock.wait()
                if self._health_close_error is not None:
                    raise self._health_close_error
                return
            self._health_close_started = True
            self._health_closer_ident = threading.get_ident()
            self._stop_admission()
        error: BaseException | None = None
        try:
            self._close_owned()
        except BaseException as exc:
            error = exc
        finally:
            with self._health_lock:
                self._health_close_error = error
                self._health_state = (
                    UtilityHealthState.FAILED if error else UtilityHealthState.CLOSED
                )
                self._health_close_done = True
                self._health_lock.notify_all()
        if error is not None:
            raise error

    async def aclose(self) -> None:
        """Drain probes off-loop before propagating caller cancellation."""
        self._assert_not_probe()
        self._stop_admission()
        await run_sync_owned(self.close)


@dataclass
class HealthCheck(_HealthOwner):
    """A health definition with one owned probe shared by concurrent observers.

    Close the check before closing its event loop or dependencies. A request
    deadline retires that observer; synchronous calls and async cleanup remain
    owned until they finish. ``None`` marks an unconfigured dependency.
    """

    name: str
    check_func: Callable[[], bool | Awaitable[bool]] | None
    timeout: float = 5.0
    critical: bool = False
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        super().__init__()
        self._probe: _UtilityHealthProbe | None = None

    def _owned_probes(self) -> list[_UtilityHealthProbe]:
        with self._health_lock:
            return (
                [self._probe] if self._probe and not self._probe.done.is_set() else []
            )

    @staticmethod
    def _observation(
        name: str,
        critical: bool,
        status: str,
        duration: float = 0.0,
        error: str | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": name,
            "status": status,
            "duration_seconds": duration,
            "critical": critical,
            "timestamp": datetime.now().isoformat(),
        }
        if error is not None:
            result["error"] = error
        return result

    async def run(self) -> dict[str, Any]:
        """Observe a shared probe within this call's captured deadline."""
        self._assert_not_probe()
        with self._health_lock:
            if self._health_state is not UtilityHealthState.READY:
                return self._observation(
                    self.name,
                    self.critical,
                    "unavailable",
                    error=f"Health check is {self._health_state.value}",
                )
            timeout = self.timeout
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                raise TypeError(
                    "Health check timeout must be a finite nonnegative number"
                )
            if not math.isfinite(timeout) or timeout < 0:
                raise ValueError("Health check timeout must be finite and nonnegative")
            probe = self._probe
            if probe is not None and probe.task is not None and probe.task.done():
                self._probe_done(probe, probe.task)
                if probe.cleanup_error is not None:
                    return self._observation(
                        probe.name,
                        probe.critical,
                        "unavailable",
                        error="Previous health check cleanup failed",
                    )
            if probe is not None and not probe.done.is_set():
                if probe.cancel_requested:
                    return self._observation(
                        probe.name,
                        probe.critical,
                        "unavailable",
                        error="Previous health check cleanup is still in progress",
                    )
                if probe.loop is not asyncio.get_running_loop():
                    raise RuntimeError(
                        "Active health probes belong to another event loop"
                    )
            else:
                check_func = self.check_func
                if check_func is None:
                    return self._observation(
                        self.name,
                        self.critical,
                        "unavailable",
                        error=self.unavailable_reason
                        or "Health check is not configured",
                    )
                if not callable(check_func):
                    raise TypeError("Health check must be callable or None")
                probe = _UtilityHealthProbe(
                    self.name, self.critical, asyncio.get_running_loop()
                )
                self._probe = probe
                operation = self._run_probe(probe, check_func)
                try:
                    probe.task = asyncio.create_task(operation)
                except BaseException:
                    operation.close()
                    self._probe = None
                    raise
                probe.task.add_done_callback(lambda task: self._probe_done(probe, task))
            probe.observers += 1
        assert probe.task is not None
        try:
            done, _ = await asyncio.wait({probe.task}, timeout=timeout)
            if not done:
                return self._observation(
                    probe.name,
                    probe.critical,
                    "timeout",
                    timeout,
                    "Health check timed out",
                )
            if probe.task.cancelled():
                return self._observation(
                    probe.name,
                    probe.critical,
                    "cancelled",
                    error="Health probe was cancelled",
                )
            return copy.deepcopy(probe.task.result())
        finally:
            with self._health_lock:
                probe.observers -= 1
                if not probe.done.is_set() and probe.observers == 0:
                    self._cancel_probe(probe)

    def _cancel_probe(self, probe: _UtilityHealthProbe) -> None:
        if probe.cancel_requested or probe.done.is_set():
            return
        probe.cancel_requested = True

        def cancel() -> None:
            # A synchronous callable can return an awaitable. Retain its waiter
            # until that result is retrieved and safely consumed or closed.
            if (
                probe.task is not None
                and not probe.task.done()
                and not probe.waiting_for_sync
            ):
                probe.task.cancel()

        probe.loop.call_soon_threadsafe(cancel)

    async def _run_sync(
        self, probe: _UtilityHealthProbe, check_func: Callable[[], Any]
    ) -> Any:
        executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="utility-health"
        )
        probe.executor = executor

        def invoke() -> Any:
            with self._health_lock:
                probe.worker_ident = threading.get_ident()
            try:
                return check_func()
            finally:
                with self._health_lock:
                    probe.worker_ident = None

        async def wait_owned(future: asyncio.Future[Any]) -> Any:
            while not future.done():
                try:
                    await asyncio.wait({future})
                except asyncio.CancelledError:
                    probe.cancel_requested = True
            return future.result()

        probe.waiting_for_sync = True
        result: Any = None
        try:
            context = contextvars.copy_context()
            future = asyncio.get_running_loop().run_in_executor(
                executor, context.run, invoke
            )
            result = await wait_owned(future)
            return result
        finally:
            try:
                operation = asyncio.to_thread(
                    executor.shutdown, wait=True, cancel_futures=True
                )
                try:
                    cleanup = asyncio.create_task(operation)
                except BaseException:
                    operation.close()
                    raise
                await wait_owned(cleanup)
                probe.executor = None
            except BaseException as error:
                probe.cleanup_error = error
                if inspect.isawaitable(result):
                    await self._discard_awaitable(result)
                raise
            finally:
                probe.waiting_for_sync = False

    @staticmethod
    async def _discard_awaitable(result: Awaitable[Any]) -> None:
        if (
            inspect.iscoroutine(result)
            and inspect.getcoroutinestate(result) == inspect.CORO_CREATED
        ):
            result.close()
        else:
            task = asyncio.ensure_future(result)
            task.cancel()

            async def drain() -> None:
                await asyncio.gather(task, return_exceptions=True)

            await drain_owned_operation(drain())

    async def _run_probe(
        self, probe: _UtilityHealthProbe, check_func: Callable[[], Any]
    ) -> dict[str, Any]:
        token = _health_probe_context.set((*_health_probe_context.get(), probe))
        try:
            return await self._measure_probe(probe, check_func)
        finally:
            _health_probe_context.reset(token)

    async def _measure_probe(
        self, probe: _UtilityHealthProbe, check_func: Callable[[], Any]
    ) -> dict[str, Any]:
        probe.task = asyncio.current_task()
        # Never invoke user code under an admission lock with an eager factory.
        await asyncio.sleep(0)
        start = time.monotonic()
        try:
            if self._is_retired(probe):
                raise asyncio.CancelledError
            if inspect.iscoroutinefunction(check_func) or inspect.iscoroutinefunction(
                getattr(check_func, "__call__", None)  # noqa: B004
            ):
                result = check_func()
            else:
                result = await self._run_sync(probe, check_func)
            if inspect.isawaitable(result):
                if self._is_retired(probe):
                    await self._discard_awaitable(result)
                    raise asyncio.CancelledError
                result = await result
            if self._is_retired(probe):
                raise asyncio.CancelledError
            return self._observation(
                probe.name,
                probe.critical,
                "healthy" if result else "unhealthy",
                time.monotonic() - start,
            )
        except MONITORING_EXCEPTIONS as error:
            return self._observation(
                probe.name,
                probe.critical,
                "error",
                time.monotonic() - start,
                str(error),
            )

    def _is_retired(self, probe: _UtilityHealthProbe) -> bool:
        # Admission can change from another thread or while awaiting a callback.
        with self._health_lock:
            return probe.cancel_requested

    def _probe_done(
        self, probe: _UtilityHealthProbe, task: asyncio.Task[dict[str, Any]]
    ) -> None:
        if not task.cancelled():
            task.exception()  # Retrieve failures even when all observers timed out.
        with self._health_lock:
            probe.done.set()
            if probe.cleanup_error is not None:
                self._health_state = UtilityHealthState.FAILED
                self._health_close_error = probe.cleanup_error

    def get_status(self) -> dict[str, Any]:
        """Inspect probe work and any executor whose cleanup is unconfirmed."""
        with self._health_lock:
            status = super().get_status()
            status["name"] = self.name
            status["configured"] = callable(self.check_func)
            status["executor_owned"] = (
                self._probe is not None and self._probe.executor is not None
            )
            return status

    def _close_owned(self) -> None:
        with self._health_lock:
            probe = self._probe
            if probe is None:
                return
            self._cancel_probe(probe)
        while not probe.done.wait(timeout=0.1):
            if not probe.loop.is_running():
                raise RuntimeError(
                    "Health probe cleanup is unconfirmed; its event loop stopped"
                )
        if probe.cleanup_error is not None:
            raise probe.cleanup_error


@dataclass
class SystemMetrics:
    """Recent system resource samples."""

    cpu_usage: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    memory_usage: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    disk_usage: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    network_io: deque[dict[str, float]] = field(
        default_factory=lambda: deque(maxlen=100)
    )


@dataclass
class ApplicationMetrics:
    """Recent application-level metrics."""

    requests_total: int = 0
    requests_per_second: deque[float] = field(default_factory=lambda: deque(maxlen=60))
    response_times: deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    active_conversations: int = 0
    training_iterations: int = 0
    reward_scores: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0


class ThreadMonitorState(str, Enum):
    """Admission and cleanup ownership of an automatic monitoring thread."""

    RUNNING = "running"
    CLOSING = "closing"
    CLOSED = "closed"
    FAILED = "failed"


class MonitoringStartupError(RuntimeError):
    """Retain a partially initialized monitor when rollback reports a failure."""

    def __init__(self, owner: Any) -> None:
        super().__init__("Monitoring startup failed; inspect the retained owner")
        self.owner = owner


class _ThreadOwner:
    """Own one automatic worker and one synchronous cleanup outcome."""

    def __init__(self) -> None:
        self._lifecycle = threading.Condition(threading.RLock())
        self._stop_event = threading.Event()
        self._worker_done = threading.Event()
        self._worker_done.set()
        self._thread: threading.Thread | None = None
        self._worker_error: BaseException | None = None
        self._state = ThreadMonitorState.RUNNING
        self._close_started = False
        self._close_done = False
        self._close_error: BaseException | None = None
        self._closer_ident: int | None = None

    @property
    def state(self) -> ThreadMonitorState:
        """Observe worker admission and cleanup without starting work."""
        with self._lifecycle:
            return self._state

    def get_status(self) -> dict[str, Any]:
        """Return a detached view of worker and cleanup ownership."""
        with self._lifecycle:
            return {
                "state": self._state.value,
                "thread_alive": self._thread is not None and self._thread.is_alive(),
                "worker_pending": not self._worker_done.is_set(),
                "stop_requested": self._stop_event.is_set(),
                "worker_error_type": (
                    type(self._worker_error).__name__ if self._worker_error else None
                ),
                "cleanup_error_type": (
                    type(self._close_error).__name__ if self._close_error else None
                ),
            }

    def _owns_current_thread(self) -> bool:
        with self._lifecycle:
            return self._thread is threading.current_thread() or (
                self._close_started
                and not self._close_done
                and self._closer_ident == threading.get_ident()
            )

    def _assert_can_close(self) -> None:
        if self._owns_current_thread():
            raise RuntimeError(
                "A monitoring worker cannot close or replace its own owner"
            )

    def _start_worker(self, name: str, operation: Callable[[], None]) -> None:
        def run() -> None:
            error: BaseException | None = None
            try:
                operation()
                if not self._stop_event.is_set():
                    error = RuntimeError("Monitoring worker stopped unexpectedly")
            except BaseException as exc:
                error = exc
            finally:
                with self._lifecycle:
                    if error is not None:
                        self._worker_error = self._worker_error or error
                        if self._state is ThreadMonitorState.RUNNING:
                            self._state = ThreadMonitorState.FAILED
                    self._worker_done.set()
                    self._lifecycle.notify_all()
                if error is not None:
                    logging.getLogger(__name__).error(
                        "%s worker failed (%s)", name, type(error).__name__
                    )

        with self._lifecycle:
            if self._state is not ThreadMonitorState.RUNNING:
                raise RuntimeError(f"Monitoring owner is {self._state.value}")
            if self._thread is not None:
                return
            thread = threading.Thread(target=run, daemon=True, name=name)
            self._thread = thread
            self._worker_done.clear()
            try:
                thread.start()
            except BaseException as error:
                self._worker_error = error
                self._state = ThreadMonitorState.FAILED
                self._stop_event.set()
                if thread.ident is None:
                    self._worker_done.set()
                raise

    def _request_stop(self) -> None:
        with self._lifecycle:
            if not self._close_done:
                self._state = ThreadMonitorState.CLOSING
                self._stop_event.set()

    def _close_resources(self) -> None:
        """Release dependencies only after this owner's worker has joined."""

    def close(self) -> None:
        """Stop and join owned work once; concurrent callers share its outcome.

        This blocks until the worker's current sample or callback returns. Use
        ``aclose`` when calling from an event loop.
        """
        self._assert_can_close()
        with self._lifecycle:
            if self._close_started:
                while not self._close_done:
                    self._lifecycle.wait()
                if self._close_error is not None:
                    raise self._close_error
                return
            self._close_started = True
            self._closer_ident = threading.get_ident()
            self._request_stop()

        error: BaseException | None = None
        try:
            if self._thread is not None and self._thread.ident is not None:
                self._thread.join()
            if not self._worker_done.is_set():
                raise RuntimeError("Monitoring worker cleanup is unconfirmed")
            try:
                self._close_resources()
            except BaseException as exc:
                error = exc
            error = self._worker_error or error
        except BaseException as exc:
            # An interrupted join must not close dependencies of a live worker.
            error = exc
        finally:
            with self._lifecycle:
                self._close_error = error
                self._state = (
                    ThreadMonitorState.FAILED if error else ThreadMonitorState.CLOSED
                )
                self._close_done = True
                self._lifecycle.notify_all()
        if error is not None:
            raise error

    async def aclose(self) -> None:
        """Drain synchronous cleanup off-loop despite repeated caller cancellation."""
        self._assert_can_close()
        self._request_stop()
        await run_sync_owned(self.close)

    def _rollback_startup(self, startup_error: BaseException) -> None:
        try:
            self.close()
        except BaseException as error:
            control = (
                error
                if not isinstance(error, Exception)
                else startup_error if not isinstance(startup_error, Exception) else None
            )
            if control is not None:
                cast(Any, control).monitoring_owner = self
                if control is error:
                    raise
                raise control from error
            raise MonitoringStartupError(self) from error


class MetricsCollector(_ThreadOwner):
    """Collects and manages metrics with thread-safe operations"""

    def __init__(self, enable_prometheus: bool = True):
        super().__init__()
        self.enable_prometheus = enable_prometheus and HAS_PROMETHEUS
        self.metrics: dict[str, list[Metric]] = defaultdict(list)
        self.prometheus_metrics: dict[str, Any] = {}

        # Thread lock for protecting shared state
        self._lock = threading.RLock()

        # System metrics
        self.system_metrics = SystemMetrics()

        # Application metrics
        self.app_metrics = ApplicationMetrics()

        # Initialize Prometheus metrics
        try:
            if self.enable_prometheus:
                self._setup_prometheus_metrics()
            # Preserve automatic startup, retaining the worker until close.
            self._start_system_monitoring()
        except BaseException as error:
            self._rollback_startup(error)
            raise

    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        if not HAS_PROMETHEUS:
            return

        # Request metrics
        self.prometheus_metrics["requests_total"] = _prometheus_get_or_create(
            Counter,
            "grpo_requests_total",
            "Total number of requests",
            labelnames=["method", "endpoint", "status"],
        )

        self.prometheus_metrics["request_duration"] = _prometheus_get_or_create(
            Histogram,
            "grpo_request_duration_seconds",
            "Request duration in seconds",
            labelnames=["method", "endpoint"],
        )

        # System metrics
        self.prometheus_metrics["cpu_usage"] = _prometheus_get_or_create(
            Gauge, "grpo_cpu_usage_percent", "CPU usage percentage"
        )

        self.prometheus_metrics["memory_usage"] = _prometheus_get_or_create(
            Gauge, "grpo_memory_usage_bytes", "Memory usage in bytes"
        )

        # Application metrics
        self.prometheus_metrics["active_conversations"] = _prometheus_get_or_create(
            Gauge, "grpo_active_conversations", "Number of active conversations"
        )

        self.prometheus_metrics["training_iterations"] = _prometheus_get_or_create(
            Counter, "grpo_training_iterations_total", "Total training iterations"
        )

        self.prometheus_metrics["reward_score"] = _prometheus_get_or_create(
            Histogram, "grpo_reward_score", "Reward scores distribution"
        )

        self.prometheus_metrics["errors_total"] = _prometheus_get_or_create(
            Counter,
            "grpo_errors_total",
            "Total number of errors",
            labelnames=["error_type"],
        )

    def _start_system_monitoring(self):
        """Start system metrics collection in background"""
        if self.state is not ThreadMonitorState.RUNNING:
            raise RuntimeError(f"Monitoring owner is {self.state.value}")
        if not HAS_PSUTIL:
            return

        # Create a logger for the monitoring thread
        monitoring_logger = logging.getLogger(__name__ + ".system_monitoring")

        def collect_system_metrics():
            while not self._stop_event.is_set():
                try:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)

                    # Memory usage
                    memory = psutil.virtual_memory()

                    # Disk usage
                    disk = psutil.disk_usage("/")

                    # Network I/O
                    net_io = psutil.net_io_counters()

                    # Thread-safe update of all metrics
                    with self._lock:
                        self.system_metrics.cpu_usage.append(cpu_percent)
                        self.system_metrics.memory_usage.append(memory.percent)
                        self.system_metrics.disk_usage.append(disk.percent)
                        self.system_metrics.network_io.append(
                            {
                                "bytes_sent": float(net_io.bytes_sent),
                                "bytes_recv": float(net_io.bytes_recv),
                                "timestamp": time.time(),
                            }
                        )

                    # Update Prometheus metrics (these are thread-safe by design)
                    if self.enable_prometheus:
                        self.prometheus_metrics["cpu_usage"].set(cpu_percent)
                        self.prometheus_metrics["memory_usage"].set(memory.used)

                except MONITORING_EXCEPTIONS + PSUTIL_EXCEPTIONS as e:
                    if self._stop_event.is_set():
                        raise
                    monitoring_logger.warning(f"Error collecting system metrics: {e}")

                self._stop_event.wait(30)  # Wake promptly when close requests stop.

        # Start in background thread
        self._start_worker("utility-system-metrics", collect_system_metrics)

    def record_metric(self, metric: Metric):
        """Record a metric"""
        metric = copy.deepcopy(metric)

        # Update Prometheus metrics
        if self.enable_prometheus and metric.name in self.prometheus_metrics:
            prom_metric = self.prometheus_metrics[metric.name]
            label_names = getattr(prom_metric, "_labelnames", ())
            if label_names:
                prom_metric = prom_metric.labels(
                    **{name: metric.labels.get(name, "") for name in label_names}
                )

            if metric.type == MetricType.COUNTER:
                prom_metric.inc(metric.value)
            elif metric.type == MetricType.GAUGE:
                prom_metric.set(metric.value)
            elif metric.type == MetricType.HISTOGRAM:
                prom_metric.observe(metric.value)
        with self._lock:
            self.metrics[metric.name].append(metric)

    def increment_counter(
        self, name: str, value: float = 1.0, labels: dict[str, str] | None = None
    ) -> None:
        """Increment a counter metric"""
        metric = Metric(
            name=name, type=MetricType.COUNTER, value=value, labels=labels or {}
        )
        self.record_metric(metric)

    def set_gauge(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Set a gauge metric"""
        metric = Metric(
            name=name, type=MetricType.GAUGE, value=value, labels=labels or {}
        )
        self.record_metric(metric)

    def observe_histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Observe a histogram metric"""
        metric = Metric(
            name=name, type=MetricType.HISTOGRAM, value=value, labels=labels or {}
        )
        self.record_metric(metric)

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of all metrics"""
        with self._lock:
            return self._metrics_summary()

    def _metrics_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": {},
            "application_metrics": {},
            "custom_metrics": {},
        }

        # System metrics
        if HAS_PSUTIL:
            summary["system_metrics"] = {
                "cpu_usage": {
                    "current": (
                        list(self.system_metrics.cpu_usage)[-1]
                        if self.system_metrics.cpu_usage
                        else 0
                    ),
                    "average": (
                        statistics.mean(self.system_metrics.cpu_usage)
                        if self.system_metrics.cpu_usage
                        else 0
                    ),
                    "max": (
                        max(self.system_metrics.cpu_usage)
                        if self.system_metrics.cpu_usage
                        else 0
                    ),
                },
                "memory_usage": {
                    "current": (
                        list(self.system_metrics.memory_usage)[-1]
                        if self.system_metrics.memory_usage
                        else 0
                    ),
                    "average": (
                        statistics.mean(self.system_metrics.memory_usage)
                        if self.system_metrics.memory_usage
                        else 0
                    ),
                    "max": (
                        max(self.system_metrics.memory_usage)
                        if self.system_metrics.memory_usage
                        else 0
                    ),
                },
                "disk_usage": {
                    "current": (
                        list(self.system_metrics.disk_usage)[-1]
                        if self.system_metrics.disk_usage
                        else 0
                    ),
                    "average": (
                        statistics.mean(self.system_metrics.disk_usage)
                        if self.system_metrics.disk_usage
                        else 0
                    ),
                    "max": (
                        max(self.system_metrics.disk_usage)
                        if self.system_metrics.disk_usage
                        else 0
                    ),
                },
            }

        # Application metrics
        summary["application_metrics"] = {
            "requests_total": self.app_metrics.requests_total,
            "requests_per_second": (
                list(self.app_metrics.requests_per_second)[-1]
                if self.app_metrics.requests_per_second
                else 0
            ),
            "average_response_time": (
                statistics.mean(self.app_metrics.response_times)
                if self.app_metrics.response_times
                else 0
            ),
            "active_conversations": self.app_metrics.active_conversations,
            "training_iterations": self.app_metrics.training_iterations,
            "average_reward_score": (
                statistics.mean(self.app_metrics.reward_scores)
                if self.app_metrics.reward_scores
                else 0
            ),
            "error_count": self.app_metrics.error_count,
        }

        # Custom metrics
        custom_metrics = cast(dict[str, Any], summary["custom_metrics"])
        for metric_name, metric_list in self.metrics.items():
            if metric_list:
                latest_metric = metric_list[-1]
                custom_metrics[metric_name] = {
                    "type": latest_metric.type.value,
                    "value": latest_metric.value,
                    "labels": copy.deepcopy(latest_metric.labels),
                    "timestamp": latest_metric.timestamp.isoformat(),
                }

        return summary


class AlertManager:
    """Manages alerts and notifications with thread-safe operations"""

    def __init__(self):
        self.alerts: deque = deque(maxlen=10000)  # Bounded to prevent memory leak
        self.alert_rules: list[dict[str, Any]] = []
        self.notification_handlers: list[Callable[[Alert], None]] = []
        self._lock = threading.RLock()
        self._logger = logging.getLogger(__name__ + ".AlertManager")

    def add_alert_rule(
        self,
        name: str,
        condition: Callable[[dict[str, Any]], bool],
        severity: AlertSeverity,
        message: str,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Add an alert rule"""
        rule = {
            "name": name,
            "condition": condition,
            "severity": severity,
            "message": message,
            "labels": labels or {},
        }
        self.alert_rules.append(rule)

    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add a notification handler"""
        self.notification_handlers.append(handler)

    def check_alerts(self, metrics: dict[str, Any]):
        """Check all alert rules against current metrics (thread-safe)"""
        # Take a snapshot of rules to avoid holding lock during condition evaluation
        with self._lock:
            rules_snapshot = list(self.alert_rules)

        for rule in rules_snapshot:
            try:
                condition_met = rule["condition"](metrics)

                with self._lock:
                    if condition_met:
                        # Check if alert already exists
                        existing_alert = next(
                            (
                                a
                                for a in self.alerts
                                if a.name == rule["name"] and not a.resolved
                            ),
                            None,
                        )

                        if not existing_alert:
                            # Create new alert
                            alert = Alert(
                                id=str(uuid.uuid4()),
                                name=rule["name"],
                                severity=rule["severity"],
                                message=rule["message"],
                                labels=rule["labels"],
                            )

                            self.alerts.append(alert)

                            # Send notifications (outside lock to avoid deadlock)
                            handlers_copy = list(self.notification_handlers)
                    else:
                        existing_alert = None
                        handlers_copy = []

                        # Check if we should resolve existing alert
                        for a in self.alerts:
                            if a.name == rule["name"] and not a.resolved:
                                a.resolved = True
                                a.resolved_at = datetime.now()
                                break

                # Send notifications outside the lock
                if condition_met and not existing_alert:
                    for handler in handlers_copy:
                        try:
                            handler(alert)
                        except MONITORING_EXCEPTIONS as e:
                            self._logger.warning(f"Notification handler error: {e}")

            except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
                self._logger.warning(
                    f"Error checking alert rule {rule.get('name')}: {e}"
                )

    def get_active_alerts(self) -> list[Alert]:
        """Get all active (unresolved) alerts"""
        return [a for a in self.alerts if not a.resolved]

    def get_alert_summary(self) -> dict[str, Any]:
        """Get alert summary"""
        active_alerts = self.get_active_alerts()

        return {
            "total_alerts": len(self.alerts),
            "active_alerts": len(active_alerts),
            "alerts_by_severity": {
                severity.value: len(
                    [a for a in active_alerts if a.severity == severity]
                )
                for severity in AlertSeverity
            },
            "recent_alerts": [a.to_dict() for a in active_alerts[-10:]],
        }


class HealthChecker(_HealthOwner):
    """Own registered checks and their probes until explicit shutdown."""

    def __init__(self) -> None:
        super().__init__()
        self.health_checks: list[HealthCheck] = []
        self._owned_checks: dict[int, HealthCheck] = {}

    def add_health_check(self, health_check: HealthCheck) -> None:
        """Register a check and retain cleanup ownership if it is later removed."""
        with self._health_lock:
            if self._health_state is not UtilityHealthState.READY:
                raise RuntimeError("Health checker is closing or closed")
            self.health_checks.append(health_check)
            self._owned_checks[id(health_check)] = health_check

    def _checks_snapshot(self) -> list[HealthCheck]:
        with self._health_lock:
            for check in self.health_checks:
                self._owned_checks[id(check)] = check
            return list(self._owned_checks.values())

    def _owned_probes(self) -> list[_UtilityHealthProbe]:
        return [
            probe
            for check in self._checks_snapshot()
            for probe in check._owned_probes()
        ]

    def _stop_admission(self) -> None:
        with self._health_lock:
            super()._stop_admission()
            for check in self._checks_snapshot():
                check._stop_admission()

    def _close_owned(self) -> None:
        error: BaseException | None = None
        for check in self._checks_snapshot():
            try:
                check.close()
            except BaseException as exc:
                error = error or exc
        if error is not None:
            raise error

    def get_status(self) -> dict[str, Any]:
        """Inspect current registrations and retained check cleanup owners."""
        with self._health_lock:
            status = super().get_status()
            status["registered_checks"] = [check.name for check in self.health_checks]
            status["owned_checks"] = [
                check.get_status() for check in self._checks_snapshot()
            ]
            return status

    async def run_all_checks(self) -> dict[str, Any]:
        """Observe a captured registration list without owning caller deadlines."""
        self._assert_not_probe()
        with self._health_lock:
            self._checks_snapshot()
            checks = tuple(self.health_checks)
        results = []

        for check in checks:
            with self._health_lock:
                accepting = self._health_state is UtilityHealthState.READY
                state = self._health_state.value
            result = (
                await check.run()
                if accepting
                else check._observation(
                    check.name,
                    check.critical,
                    "unavailable",
                    error=f"Health checker is {state}",
                )
            )
            results.append(result)

        # Overall health status
        all_healthy = bool(results) and all(r["status"] == "healthy" for r in results)

        critical_failing = any(
            r["status"] != "healthy" and r["critical"] for r in results
        )

        overall_status = (
            "healthy"
            if all_healthy
            else (
                "critical"
                if critical_failing
                else "degraded" if results else "unavailable"
            )
        )

        return {
            "overall_status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "checks": results,
            "summary": {
                "total_checks": len(results),
                "healthy": len([r for r in results if r["status"] == "healthy"]),
                "unhealthy": len([r for r in results if r["status"] != "healthy"]),
                "critical_failing": len(
                    [r for r in results if r["status"] != "healthy" and r["critical"]]
                ),
            },
        }


class MonitoringService(_ThreadOwner):
    """Main monitoring service that orchestrates all monitoring functionality"""

    def __init__(
        self,
        enable_prometheus: bool = True,
        # Default to a non-API port to avoid clashing with the gateway's 8000.
        prometheus_port: int = 8001,
        enable_health_checks: bool = True,
        enable_alerts: bool = True,
    ):
        super().__init__()
        self.enable_prometheus = enable_prometheus
        self.prometheus_port = prometheus_port
        self.enable_health_checks = enable_health_checks
        self.enable_alerts = enable_alerts

        self._logger = logging.getLogger(__name__ + ".MonitoringService")
        self._prometheus_server: Any | None = None
        self._prometheus_thread: threading.Thread | None = None
        self._listener_unmanaged = False
        self._listener_error: BaseException | None = None
        self._listener_start_attempted = False
        self._listener_start_done = threading.Event()
        self._listener_start_done.set()
        self._listener_starter_ident: int | None = None

        try:
            self.metrics_collector = MetricsCollector(enable_prometheus)
            self.alert_manager = AlertManager() if enable_alerts else None
            self.health_checker = HealthChecker() if enable_health_checks else None
            self._start_prometheus()
            self._setup_default_health_checks()
            self._setup_default_alert_rules()
            self._start_monitoring_loop()
        except BaseException as error:
            self._rollback_startup(error)
            raise

    def _start_prometheus(self) -> None:
        with self._lifecycle:
            if self._state is not ThreadMonitorState.RUNNING:
                raise RuntimeError(f"Monitoring owner is {self._state.value}")
            if self._listener_start_attempted:
                return
            self._listener_start_attempted = True
            if not (self.enable_prometheus and HAS_PROMETHEUS):
                return
            self._listener_start_done.clear()
            self._listener_starter_ident = threading.get_ident()
            port = self.prometheus_port
        try:
            result = start_http_server(port)
            if result is None:
                # Older clients start a server without returning its handles.
                # Preserve startup while making unconfirmed cleanup explicit.
                self._listener_unmanaged = True
                raise RuntimeError("Prometheus client did not return listener handles")
            self._prometheus_server, self._prometheus_thread = result
            self._logger.info("Prometheus metrics server started on port %s", port)
        except (OSError, ValueError, RuntimeError) as e:
            self._listener_error = e
            self._logger.error("Unable to manage Prometheus metrics listener: %s", e)
        finally:
            with self._lifecycle:
                self._listener_starter_ident = None
                self._listener_start_done.set()

    def _owns_current_thread(self) -> bool:
        collector = getattr(self, "metrics_collector", None)
        health_checker = getattr(self, "health_checker", None)
        return (
            super()._owns_current_thread()
            or self._prometheus_thread is threading.current_thread()
            or self._listener_starter_ident == threading.get_ident()
            or (collector is not None and collector._owns_current_thread())
            or (health_checker is not None and health_checker._owns_current_execution())
        )

    def _assert_can_close(self) -> None:
        super()._assert_can_close()
        health_checker = getattr(self, "health_checker", None)
        if health_checker is not None:
            health_checker._assert_can_close()

    def _request_stop(self) -> None:
        super()._request_stop()
        health_checker = getattr(self, "health_checker", None)
        if health_checker is not None:
            health_checker._stop_admission()

    async def aclose(self) -> None:
        """Drain health probes and background threads before releasing resources."""
        _ThreadOwner._assert_can_close(self)
        self._request_stop()
        await run_sync_owned(self.close)

    def get_status(self) -> dict[str, Any]:
        """Inspect the worker, metric collector, and owned Prometheus listener."""
        status = super().get_status()
        collector = getattr(self, "metrics_collector", None)
        status["metrics_collector"] = (
            collector.get_status() if collector is not None else None
        )
        health_checker = getattr(self, "health_checker", None)
        status["health_checker"] = (
            health_checker.get_status() if health_checker is not None else None
        )
        with self._lifecycle:
            listener_thread = self._prometheus_thread
            status["prometheus"] = {
                "starting": not self._listener_start_done.is_set(),
                "server_owned": self._prometheus_server is not None,
                "thread_alive": listener_thread is not None
                and listener_thread.is_alive(),
                "cleanup_unconfirmed": self._listener_unmanaged,
                "error_type": (
                    type(self._listener_error).__name__
                    if self._listener_error
                    else None
                ),
            }
        return status

    def _close_listener(self) -> None:
        if self._listener_unmanaged:
            raise RuntimeError(
                "Prometheus listener cleanup is unconfirmed; no handles were returned"
            )
        if self._prometheus_server is None:
            return
        try:
            self._prometheus_server.shutdown()
            try:
                self._prometheus_server.server_close()
            finally:
                if self._prometheus_thread is not None:
                    self._prometheus_thread.join()
        except BaseException as error:
            self._listener_error = error
            raise
        self._prometheus_server = None
        self._prometheus_thread = None

    def _close_resources(self) -> None:
        self._listener_start_done.wait()
        health_checker = getattr(self, "health_checker", None)
        if health_checker is not None:
            # Unconfirmed probes may still use these dependencies. Retain them
            # if draining health work fails, rather than claiming shutdown.
            health_checker.close()
        error: BaseException | None = None
        collector = getattr(self, "metrics_collector", None)
        if collector is not None:
            try:
                collector.close()
            except BaseException as exc:
                error = exc
        try:
            self._close_listener()
        except BaseException as exc:
            error = error or exc
        if error is not None:
            raise error

    def _setup_default_health_checks(self):
        """Setup default health checks"""
        if not self.health_checker:
            return

        self.health_checker.add_health_check(
            HealthCheck("database", None, critical=True)
        )

        self.health_checker.add_health_check(HealthCheck("cache", None, critical=False))

        # System resources check
        def system_resources_check():
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent

            return cpu_percent < 90 and memory_percent < 90

        self.health_checker.add_health_check(
            HealthCheck(
                "system_resources",
                system_resources_check if HAS_PSUTIL else None,
                critical=False,
                unavailable_reason="psutil is not installed",
            )
        )

    def _setup_default_alert_rules(self):
        """Setup default alert rules"""
        if not self.alert_manager:
            return

        # High CPU usage
        def high_cpu_condition(metrics):
            cpu_usage = (
                metrics.get("system_metrics", {}).get("cpu_usage", {}).get("current", 0)
            )
            return cpu_usage > 80

        self.alert_manager.add_alert_rule(
            "high_cpu_usage",
            high_cpu_condition,
            AlertSeverity.MEDIUM,
            "CPU usage is above 80%",
        )

        # High memory usage
        def high_memory_condition(metrics):
            memory_usage = (
                metrics.get("system_metrics", {})
                .get("memory_usage", {})
                .get("current", 0)
            )
            return memory_usage > 85

        self.alert_manager.add_alert_rule(
            "high_memory_usage",
            high_memory_condition,
            AlertSeverity.MEDIUM,
            "Memory usage is above 85%",
        )

        # High error rate
        def high_error_rate_condition(metrics):
            error_count = metrics.get("application_metrics", {}).get("error_count", 0)
            return error_count > 100

        self.alert_manager.add_alert_rule(
            "high_error_rate",
            high_error_rate_condition,
            AlertSeverity.HIGH,
            "Error count is above 100",
        )

    def _start_monitoring_loop(self):
        """Start the monitoring loop"""

        def monitoring_loop():
            while not self._stop_event.is_set():
                try:
                    # Get current metrics
                    metrics = self.metrics_collector.get_metrics_summary()

                    # Check alerts
                    if self.alert_manager:
                        self.alert_manager.check_alerts(metrics)

                except MONITORING_EXCEPTIONS as e:
                    if self._stop_event.is_set():
                        raise
                    self._logger.warning(f"Error in monitoring loop: {e}")

                self._stop_event.wait(60)

        # Start in background thread
        self._start_worker("utility-monitoring", monitoring_loop)

    @contextmanager
    def track_request(self, method: str, endpoint: str):
        """Context manager for tracking API requests"""
        start_time = time.time()

        try:
            yield

            # Record successful request
            duration = time.time() - start_time
            self.metrics_collector.increment_counter(
                "requests_total",
                labels={"method": method, "endpoint": endpoint, "status": "success"},
            )
            self.metrics_collector.observe_histogram(
                "request_duration",
                duration,
                labels={"method": method, "endpoint": endpoint},
            )

            # Update app metrics
            with self.metrics_collector._lock:
                self.metrics_collector.app_metrics.requests_total += 1
                self.metrics_collector.app_metrics.response_times.append(duration)

        except MONITORING_EXCEPTIONS as e:
            # Record failed request
            duration = time.time() - start_time
            self.metrics_collector.increment_counter(
                "requests_total",
                labels={"method": method, "endpoint": endpoint, "status": "error"},
            )
            self.metrics_collector.increment_counter(
                "errors_total", labels={"error_type": type(e).__name__}
            )

            # Update app metrics
            with self.metrics_collector._lock:
                self.metrics_collector.app_metrics.error_count += 1

            raise

    async def log_metric(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Log a custom metric"""
        metric = Metric(
            name=name, type=MetricType.GAUGE, value=value, labels=labels or {}
        )
        self.metrics_collector.record_metric(metric)

    async def log_event(self, event_type: str, metadata: dict[str, Any]):
        """Log an event"""
        self.metrics_collector.increment_counter(
            "events_total", labels={"event_type": event_type}
        )

        # Could also send to external systems like Elasticsearch
        print(f"Event logged: {event_type} - {metadata}")

    async def get_metrics(self) -> dict[str, Any]:
        """Get all metrics"""
        return self.metrics_collector.get_metrics_summary()

    async def get_health_status(self) -> dict[str, Any]:
        """Get health status"""
        if not self.health_checker:
            return {"status": "health_checks_disabled"}

        return await self.health_checker.run_all_checks()

    async def get_alerts(self) -> dict[str, Any]:
        """Get alert summary"""
        if not self.alert_manager:
            return {"alerts": "disabled"}

        return self.alert_manager.get_alert_summary()

    async def get_dashboard_data(self) -> dict[str, Any]:
        """Get comprehensive dashboard data"""
        return {
            "timestamp": datetime.now().isoformat(),
            "lifecycle": self.get_status(),
            "metrics": await self.get_metrics(),
            "health": await self.get_health_status(),
            "alerts": await self.get_alerts(),
        }


# Global monitoring instance
_global_monitoring: MonitoringService | None = None
_global_monitoring_lock = threading.RLock()
_global_replacing = False
_global_initializing = False
_global_creation_error: BaseException | None = None


def get_monitoring_service() -> MonitoringService:
    """Get global monitoring service"""
    global _global_monitoring, _global_initializing, _global_creation_error

    with _global_monitoring_lock:
        if _global_replacing or _global_initializing:
            raise RuntimeError(
                "Global monitoring initialization or replacement is in progress"
            )
        if _global_creation_error is not None:
            raise _global_creation_error
        if _global_monitoring is not None:
            return _global_monitoring
        _global_initializing = True
    try:
        owner = MonitoringService()
        with _global_monitoring_lock:
            _global_monitoring = owner
        return owner
    except BaseException as error:
        with _global_monitoring_lock:
            _global_creation_error = error
        raise
    finally:
        with _global_monitoring_lock:
            _global_initializing = False


def setup_monitoring(
    enable_prometheus: bool = True,
    prometheus_port: int = 8000,
    enable_health_checks: bool = True,
    enable_alerts: bool = True,
) -> MonitoringService:
    """Setup global monitoring"""
    global _global_monitoring, _global_replacing, _global_creation_error

    with _global_monitoring_lock:
        if _global_replacing or _global_initializing:
            raise RuntimeError(
                "Global monitoring initialization or replacement is in progress"
            )
        previous = _global_monitoring
        failed_owner = getattr(
            _global_creation_error,
            "owner",
            getattr(_global_creation_error, "monitoring_owner", None),
        )
        for owner in (previous, failed_owner):
            check_owner = getattr(owner, "_assert_can_close", None)
            if check_owner is not None:
                check_owner()
        _global_replacing = True
    creating = False
    try:
        # Do not hold the registry lock while joining callbacks. A callback that
        # asks for the global service gets an explicit transition error.
        if previous is not None:
            previous.close()
        if failed_owner is not None and failed_owner is not previous:
            failed_owner.close()
        creating = True
        replacement = MonitoringService(
            enable_prometheus=enable_prometheus,
            prometheus_port=prometheus_port,
            enable_health_checks=enable_health_checks,
            enable_alerts=enable_alerts,
        )
        with _global_monitoring_lock:
            _global_monitoring = replacement
            _global_creation_error = None
        return replacement
    except BaseException as error:
        if creating:
            with _global_monitoring_lock:
                _global_creation_error = error
        raise
    finally:
        with _global_monitoring_lock:
            _global_replacing = False
