"""Utility monitoring owns its automatic workers and listener until joined."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from stateset_agents.utils import monitoring


@pytest.fixture(autouse=True)
def isolate_global_transitions(monkeypatch):
    monkeypatch.setattr(monitoring, "_global_initializing", False)
    monkeypatch.setattr(monitoring, "_global_replacing", False)
    monkeypatch.setattr(monitoring, "_global_creation_error", None)


class FakeThread:
    def __init__(self, target=None, *, daemon=None, name=None):
        self.target = target
        self.daemon = daemon
        self.name = name or "thread"
        self.ident = None
        self.alive = False
        self.joins = 0

    def start(self):
        self.ident = id(self)
        self.alive = True

    def is_alive(self):
        return self.alive

    def join(self):
        self.joins += 1
        if self.target:
            # The close path has already signalled stop, so loop targets exit.
            self.target()
            self.target = None
        self.alive = False


@pytest.fixture
def fake_threads(monkeypatch):
    threads = []

    def create(*args, **kwargs):
        thread = FakeThread(*args, **kwargs)
        threads.append(thread)
        return thread

    monkeypatch.setattr(
        monitoring,
        "threading",
        SimpleNamespace(
            Thread=create,
            Event=threading.Event,
            RLock=threading.RLock,
            Condition=threading.Condition,
            current_thread=threading.current_thread,
            get_ident=threading.get_ident,
        ),
    )
    return threads


@pytest.mark.parametrize("kind", ["collector", "service"])
def test_repeated_start_does_not_duplicate_automatic_workers(
    monkeypatch, fake_threads, kind
):
    monkeypatch.setattr(monitoring, "HAS_PSUTIL", kind == "collector")
    if kind == "collector":
        owner = monitoring.MetricsCollector(enable_prometheus=False)
        start = owner._start_system_monitoring
    else:
        owner = monitoring.MonitoringService(enable_prometheus=False)
        start = owner._start_monitoring_loop
    try:
        start()
        assert len(fake_threads) == 1
        owner.close()
        assert fake_threads[0].joins == 1
        assert owner.get_status()["state"] == "closed"
        with pytest.raises(RuntimeError, match="closing|closed"):
            start()
    finally:
        close = getattr(owner, "close", None)
        if close:
            close()


def test_service_closes_its_listener_and_joins_every_owned_thread(
    monkeypatch, fake_threads
):
    monkeypatch.setattr(monitoring, "HAS_PSUTIL", False)
    monkeypatch.setattr(monitoring, "HAS_PROMETHEUS", True)
    monkeypatch.setattr(
        monitoring.MetricsCollector, "_setup_prometheus_metrics", lambda _: None
    )
    events = []
    listener = FakeThread()
    listener.start()
    server = SimpleNamespace(
        shutdown=lambda: events.append("shutdown"),
        server_close=lambda: events.append("socket closed"),
    )
    monkeypatch.setattr(
        monitoring, "start_http_server", lambda port: (server, listener), raising=False
    )
    owner = monitoring.MonitoringService()
    owner.close()
    owner.close()
    assert events == ["shutdown", "socket closed"]
    assert listener.joins == 1 and not listener.is_alive()
    assert all(thread.joins == 1 and not thread.is_alive() for thread in fake_threads)
    assert owner.metrics_collector.get_status()["state"] == "closed"
    assert owner.get_status()["state"] == "closed"


def test_global_replacement_closes_previous_owner_before_constructing_new(monkeypatch):
    events = []
    previous = SimpleNamespace(close=lambda: events.append("closed old"))
    replacement = object()

    def construct(**kwargs):
        assert events == ["closed old"]
        events.append("created new")
        return replacement

    monkeypatch.setattr(monitoring, "_global_monitoring", previous)
    monkeypatch.setattr(monitoring, "MonitoringService", construct)
    assert monitoring.setup_monitoring(enable_prometheus=False) is replacement
    assert monitoring.get_monitoring_service() is replacement
    assert events == ["closed old", "created new"]


def configure_sampling(monkeypatch, sample):
    monkeypatch.setattr(monitoring, "HAS_PSUTIL", True)
    monkeypatch.setattr(
        monitoring,
        "psutil",
        SimpleNamespace(
            cpu_percent=sample,
            virtual_memory=lambda: SimpleNamespace(percent=20, used=100),
            disk_usage=lambda _: SimpleNamespace(percent=30),
            net_io_counters=lambda: SimpleNamespace(bytes_sent=1, bytes_recv=2),
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["collector", "service"])
async def test_cancelled_and_concurrent_closers_retain_live_sample(monkeypatch, kind):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def sample(interval):
        entered.set()
        assert release.wait(timeout=3)
        calls.append("sample")
        return 25

    configure_sampling(monkeypatch, sample)
    owner = (
        monitoring.MetricsCollector(enable_prometheus=False)
        if kind == "collector"
        else monitoring.MonitoringService(enable_prometheus=False)
    )
    collector = owner if kind == "collector" else owner.metrics_collector
    waiters = []
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        waiters = [asyncio.create_task(owner.aclose()) for _ in range(2)]
        waiters.append(asyncio.create_task(asyncio.to_thread(owner.close)))
        assert await asyncio.to_thread(collector._stop_event.wait, 2)
        waiters[0].cancel()
        await asyncio.sleep(0)
        waiters[0].cancel()
        await asyncio.sleep(0)
        assert not any(task.done() for task in waiters)
        assert collector.get_status()["worker_pending"]
        assert collector.get_status()["thread_alive"]
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await waiters[0]
        await asyncio.gather(*waiters[1:])
        assert calls == ["sample"]
        assert owner.get_status()["state"] == "closed"
        assert not collector.get_status()["thread_alive"]
        assert not collector.get_status()["worker_pending"]
    finally:
        release.set()
        await owner.aclose()
        await asyncio.gather(*waiters, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_sampling_worker_cannot_close_its_parent_service(monkeypatch, use_async):
    proceed, checked = threading.Event(), threading.Event()
    owners, errors = [], []

    def sample(interval):
        assert proceed.wait(timeout=3)
        try:
            if use_async:
                asyncio.run(owners[0].aclose())
            else:
                owners[0].close()
        except RuntimeError as error:
            errors.append(str(error))
        checked.set()
        return 25

    configure_sampling(monkeypatch, sample)
    owner = monitoring.MonitoringService(enable_prometheus=False)
    owners.append(owner)
    try:
        proceed.set()
        assert await asyncio.to_thread(checked.wait, 2)
        assert len(errors) == 1 and "own owner" in errors[0]
        assert owner.get_status()["state"] == "running"
    finally:
        proceed.set()
        await owner.aclose()


@pytest.mark.asyncio
async def test_worker_failure_remains_visible_and_cannot_restart(monkeypatch):
    failure = AssertionError("sampler bug")

    def sample(interval):
        raise failure

    configure_sampling(monkeypatch, sample)
    owner = monitoring.MetricsCollector(enable_prometheus=False)
    assert await asyncio.to_thread(owner._worker_done.wait, 2)
    assert owner.get_status()["state"] == "failed"
    assert owner.get_status()["worker_error_type"] == "AssertionError"
    with pytest.raises(RuntimeError, match="failed"):
        owner._start_system_monitoring()
    for _ in range(2):
        with pytest.raises(AssertionError) as caught:
            await owner.aclose()
        assert caught.value is failure
    assert not owner.get_status()["thread_alive"]


def test_failed_thread_start_retains_owner_and_drains_started_effects(
    monkeypatch, fake_threads
):
    monkeypatch.setattr(monitoring, "HAS_PSUTIL", True)
    original_start = FakeThread.start

    def start(thread):
        original_start(thread)
        raise OSError("start outcome uncertain")

    monkeypatch.setattr(FakeThread, "start", start)
    with pytest.raises(monitoring.MonitoringStartupError) as caught:
        monitoring.MetricsCollector(enable_prometheus=False)
    owner = caught.value.owner
    assert owner.get_status()["state"] == "failed"
    assert not owner.get_status()["worker_pending"]
    assert not fake_threads[0].is_alive()
    assert fake_threads[0].joins == 1


@pytest.mark.parametrize("error", [KeyboardInterrupt(), asyncio.CancelledError()])
def test_control_flow_during_startup_preserves_exception_and_owner(
    monkeypatch, fake_threads, error
):
    monkeypatch.setattr(monitoring, "HAS_PSUTIL", True)
    original_start = FakeThread.start

    def start(thread):
        original_start(thread)
        raise error

    monkeypatch.setattr(FakeThread, "start", start)
    with pytest.raises(type(error)) as caught:
        monitoring.MetricsCollector(enable_prometheus=False)
    assert caught.value is error
    assert error.monitoring_owner.get_status()["state"] == "failed"
    assert not error.monitoring_owner.get_status()["worker_pending"]
    assert fake_threads[0].joins == 1


@pytest.mark.parametrize("phase", ["shutdown", "server_close"])
def test_listener_cleanup_failure_is_retained_without_replay(
    monkeypatch, fake_threads, phase
):
    monkeypatch.setattr(monitoring, "HAS_PSUTIL", False)
    monkeypatch.setattr(monitoring, "HAS_PROMETHEUS", True)
    monkeypatch.setattr(
        monitoring.MetricsCollector, "_setup_prometheus_metrics", lambda _: None
    )
    calls = []
    failure = OSError("listener cleanup uncertain")
    listener = FakeThread()
    listener.start()

    def shutdown():
        calls.append("shutdown")
        if phase == "shutdown":
            raise failure

    def server_close():
        calls.append("server_close")
        if phase == "server_close":
            raise failure

    server = SimpleNamespace(shutdown=shutdown, server_close=server_close)
    monkeypatch.setattr(
        monitoring, "start_http_server", lambda port: (server, listener), raising=False
    )
    owner = monitoring.MonitoringService()
    for _ in range(2):
        with pytest.raises(OSError) as caught:
            owner.close()
        assert caught.value is failure
    status = owner.get_status()
    assert status["state"] == "failed"
    assert status["prometheus"]["server_owned"]
    assert status["prometheus"]["error_type"] == "OSError"
    assert calls == (
        ["shutdown"] if phase == "shutdown" else ["shutdown", "server_close"]
    )
    assert listener.joins == (0 if phase == "shutdown" else 1)
    assert owner.metrics_collector.state.value == "closed"


def test_legacy_listener_without_handles_cannot_claim_successful_close(
    monkeypatch, fake_threads
):
    monkeypatch.setattr(monitoring, "HAS_PSUTIL", False)
    monkeypatch.setattr(monitoring, "HAS_PROMETHEUS", True)
    monkeypatch.setattr(
        monitoring.MetricsCollector, "_setup_prometheus_metrics", lambda _: None
    )
    calls = []
    monkeypatch.setattr(
        monitoring,
        "start_http_server",
        lambda port: calls.append("start"),
        raising=False,
    )
    owner = monitoring.MonitoringService()
    owner._start_prometheus()
    assert calls == ["start"]
    with pytest.raises(RuntimeError, match="unconfirmed"):
        owner.close()
    assert owner.get_status()["state"] == "failed"
    assert owner.get_status()["prometheus"]["cleanup_unconfirmed"]


def test_constructor_failure_closes_already_started_listener(monkeypatch, fake_threads):
    monkeypatch.setattr(monitoring, "HAS_PSUTIL", False)
    monkeypatch.setattr(monitoring, "HAS_PROMETHEUS", True)
    monkeypatch.setattr(
        monitoring.MetricsCollector, "_setup_prometheus_metrics", lambda _: None
    )
    events = []
    server = SimpleNamespace(
        shutdown=lambda: events.append("shutdown"),
        server_close=lambda: events.append("closed"),
    )
    listener = FakeThread()
    listener.start()
    monkeypatch.setattr(
        monitoring, "start_http_server", lambda port: (server, listener), raising=False
    )
    failure = ValueError("alert setup failed")

    def fail(self):
        raise failure

    monkeypatch.setattr(
        monitoring.MonitoringService, "_setup_default_alert_rules", fail
    )
    with pytest.raises(ValueError) as caught:
        monitoring.MonitoringService()
    assert caught.value is failure
    assert events == ["shutdown", "closed"]
    assert listener.joins == 1


def test_failed_global_cleanup_keeps_the_previous_owner(monkeypatch):
    failure = RuntimeError("old cleanup failed")
    calls = []

    def close():
        raise failure

    previous = SimpleNamespace(close=close)
    monkeypatch.setattr(monitoring, "_global_monitoring", previous)
    monkeypatch.setattr(
        monitoring, "MonitoringService", lambda **kwargs: calls.append("new")
    )
    with pytest.raises(RuntimeError) as caught:
        monitoring.setup_monitoring()
    assert caught.value is failure
    assert monitoring.get_monitoring_service() is previous
    assert not calls


def test_metric_snapshots_detach_caller_and_summary_labels(monkeypatch):
    monkeypatch.setattr(monitoring, "HAS_PSUTIL", False)
    owner = monitoring.MetricsCollector(enable_prometheus=False)
    metric = monitoring.Metric(
        "sample", monitoring.MetricType.GAUGE, 2, {"source": "one"}
    )
    try:
        owner.record_metric(metric)
        metric.labels["source"] = "caller mutation"
        first = owner.get_metrics_summary()
        assert first["custom_metrics"]["sample"]["labels"] == {"source": "one"}
        first["custom_metrics"]["sample"]["labels"]["source"] = "observer mutation"
        assert owner.get_metrics_summary()["custom_metrics"]["sample"]["labels"] == {
            "source": "one"
        }
    finally:
        owner.close()


@pytest.mark.asyncio
async def test_close_retains_listener_initialization_until_handles_arrive(
    monkeypatch, fake_threads
):
    monkeypatch.setattr(monitoring, "HAS_PSUTIL", False)
    monkeypatch.setattr(monitoring, "HAS_PROMETHEUS", True)
    monkeypatch.setattr(
        monitoring.MetricsCollector, "_setup_prometheus_metrics", lambda _: None
    )
    entered, release = threading.Event(), threading.Event()
    owners, calls = [], []
    server = SimpleNamespace(
        shutdown=lambda: calls.append("shutdown"),
        server_close=lambda: calls.append("closed"),
    )
    listener = FakeThread()
    listener.start()

    class Service(monitoring.MonitoringService):
        def _start_prometheus(self):
            owners.append(self)
            super()._start_prometheus()

    def start(port):
        entered.set()
        assert release.wait(timeout=3)
        return server, listener

    monkeypatch.setattr(monitoring, "start_http_server", start, raising=False)
    constructing = asyncio.create_task(asyncio.to_thread(Service))
    closing = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        owner = owners[0]
        closing = asyncio.create_task(owner.aclose())
        assert await asyncio.to_thread(owner._stop_event.wait, 2)
        assert owner.get_status()["prometheus"]["starting"]
        assert not closing.done()
        release.set()
        with pytest.raises(RuntimeError, match="closing|closed"):
            await constructing
        await closing
        assert calls == ["shutdown", "closed"]
        assert listener.joins == 1
        assert owner.get_status()["state"] == "closed"
    finally:
        release.set()
        await asyncio.gather(
            constructing, *([closing] if closing else []), return_exceptions=True
        )
        if owners:
            await owners[0].aclose()


@pytest.mark.asyncio
async def test_global_replacement_does_not_hold_registry_lock_while_joining_callback(
    monkeypatch,
):
    monkeypatch.setattr(monitoring, "HAS_PSUTIL", False)
    entered, release = threading.Event(), threading.Event()
    original = monitoring.MetricsCollector.get_metrics_summary
    errors = []

    def summarize(collector):
        entered.set()
        assert release.wait(timeout=3)
        try:
            monitoring.get_monitoring_service()
        except RuntimeError as error:
            errors.append(str(error))
        return original(collector)

    monkeypatch.setattr(monitoring.MetricsCollector, "get_metrics_summary", summarize)
    previous = monitoring.MonitoringService(enable_prometheus=False)
    monkeypatch.setattr(monitoring, "_global_monitoring", previous)
    replacement = object()

    def construct(**kwargs):
        assert previous.get_status()["state"] == "closed"
        return replacement

    monkeypatch.setattr(monitoring, "MonitoringService", construct)
    replacing = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        replacing = asyncio.create_task(asyncio.to_thread(monitoring.setup_monitoring))
        assert await asyncio.to_thread(previous._stop_event.wait, 2)
        release.set()
        assert await replacing is replacement
        assert len(errors) == 1 and "in progress" in errors[0]
    finally:
        release.set()
        await previous.aclose()
        if replacing:
            await asyncio.gather(replacing, return_exceptions=True)


@pytest.mark.asyncio
async def test_first_global_initialization_does_not_lock_out_callbacks(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    replacement = object()
    calls = []

    def construct():
        calls.append("create")
        entered.set()
        assert release.wait(timeout=3)
        return replacement

    monkeypatch.setattr(monitoring, "_global_monitoring", None)
    monkeypatch.setattr(monitoring, "MonitoringService", construct)
    initializing = asyncio.create_task(
        asyncio.to_thread(monitoring.get_monitoring_service)
    )
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        for action in (monitoring.get_monitoring_service, monitoring.setup_monitoring):
            with pytest.raises(RuntimeError, match="in progress"):
                await asyncio.wait_for(asyncio.to_thread(action), timeout=1)
        release.set()
        assert await initializing is replacement
        assert monitoring.get_monitoring_service() is replacement
        assert calls == ["create"]
    finally:
        release.set()
        await asyncio.gather(initializing, return_exceptions=True)


def test_failed_initialization_requires_explicit_setup_to_retry(monkeypatch):
    failure = ValueError("configuration rejected")
    calls = []
    replacement = object()

    def construct(**kwargs):
        calls.append("create")
        if len(calls) == 1:
            raise failure
        return replacement

    monkeypatch.setattr(monitoring, "_global_monitoring", None)
    monkeypatch.setattr(monitoring, "MonitoringService", construct)
    for _ in range(2):
        with pytest.raises(ValueError) as caught:
            monitoring.get_monitoring_service()
        assert caught.value is failure
    assert calls == ["create"]
    assert monitoring.setup_monitoring() is replacement
    assert monitoring.get_monitoring_service() is replacement
    assert calls == ["create", "create"]


def test_failed_initialization_keeps_owner_when_cleanup_is_unconfirmed(monkeypatch):
    cleanup_error = RuntimeError("listener cleanup unconfirmed")
    calls = []

    def close():
        raise cleanup_error

    owner = SimpleNamespace(close=close)
    failure = monitoring.MonitoringStartupError(owner)

    def construct(**kwargs):
        calls.append("create")
        raise failure

    monkeypatch.setattr(monitoring, "_global_monitoring", None)
    monkeypatch.setattr(monitoring, "MonitoringService", construct)
    with pytest.raises(monitoring.MonitoringStartupError) as caught:
        monitoring.get_monitoring_service()
    assert caught.value.owner is owner
    with pytest.raises(RuntimeError) as caught:
        monitoring.setup_monitoring()
    assert caught.value is cleanup_error
    with pytest.raises(monitoring.MonitoringStartupError) as caught:
        monitoring.get_monitoring_service()
    assert caught.value.owner is owner
    assert calls == ["create"]
