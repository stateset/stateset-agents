"""Background monitoring owns startup, worker failures, and shutdown."""

import asyncio
import threading

import pytest

from stateset_agents.core import advanced_monitoring as monitoring


@pytest.fixture(autouse=True)
def disable_exporters(monkeypatch):
    monkeypatch.setattr(monitoring, "PROMETHEUS_AVAILABLE", False)
    monkeypatch.setattr(monitoring, "OPENTELEMETRY_AVAILABLE", False)


def make_owner(kind, monkeypatch, operation):
    if kind == "metrics":
        owner = monitoring.MetricsCollector()
        monkeypatch.setattr(owner, "_cleanup_old_metrics", operation)
    elif kind == "alerts":
        owner = monitoring.AlertManager(monitoring.MetricsCollector())
        monkeypatch.setattr(owner, "_check_alerts", operation)
    else:
        owner = monitoring.AdvancedMonitoringService(enable_tracing=False)
        monkeypatch.setattr(owner, "_monitoring_loop", operation)
    return owner


async def cleanup(owner):
    """Also clean up pre-change owners when demonstrating regressions."""
    if hasattr(owner, "aclose"):
        try:
            await owner.aclose()
        except Exception:
            pass
        return
    tasks = []
    for candidate in (
        owner,
        getattr(owner, "metrics_collector", None),
        getattr(owner, "alert_manager", None),
    ):
        for name in ("_cleanup_task", "_alert_task", "_monitoring_task"):
            task = getattr(candidate, name, None)
            if task is not None:
                task.cancel()
                tasks.append(task)
    await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["metrics", "alerts", "service"])
async def test_close_joins_workers_and_prevents_restart(monkeypatch, kind):
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = []

    async def operation():
        calls.append("started")
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cleaning.set()
            await release.wait()
            calls.append("cleaned")

    owner = make_owner(kind, monkeypatch, operation)
    closing = None
    try:
        await owner.start()
        await asyncio.wait_for(entered.wait(), timeout=2)
        await owner.start()
        assert calls == ["started"]
        closing = asyncio.create_task(owner.aclose())
        await asyncio.wait_for(cleaning.wait(), timeout=2)
        assert not closing.done()
        with pytest.raises(RuntimeError, match="closing|closed"):
            await owner.start()
        release.set()
        await asyncio.wait_for(closing, timeout=2)
        await owner.aclose()
        assert calls == ["started", "cleaned"]
        assert owner.get_status()["state"] == "closed"
        with pytest.raises(RuntimeError, match="closing|closed"):
            await owner.start()
    finally:
        release.set()
        await cleanup(owner)
        if closing:
            await asyncio.gather(closing, return_exceptions=True)


@pytest.mark.asyncio
async def test_dead_worker_cannot_be_silently_replaced(monkeypatch):
    entered, fail = asyncio.Event(), asyncio.Event()
    calls = []

    async def operation():
        calls.append("started")
        entered.set()
        await fail.wait()
        raise AssertionError("worker failed")

    owner = make_owner("metrics", monkeypatch, operation)
    try:
        await owner.start()
        await asyncio.wait_for(entered.wait(), timeout=2)
        fail.set()
        await asyncio.gather(owner._cleanup_task, return_exceptions=True)
        with pytest.raises(RuntimeError, match="failed"):
            await owner.start()
        assert calls == ["started"]
        assert owner.get_status()["state"] == "failed"
    finally:
        fail.set()
        await cleanup(owner)


def test_unstarted_monitoring_does_not_claim_healthy_background_workers():
    service = monitoring.AdvancedMonitoringService(enable_tracing=False)
    assert service.get_health_summary()["status"] != "healthy"


@pytest.mark.asyncio
async def test_concurrent_cancelled_close_waiters_share_cleanup(monkeypatch):
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = []

    async def operation():
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cleaning.set()
            await release.wait()
            calls.append("cleaned")

    owner = make_owner("metrics", monkeypatch, operation)
    waiters = []
    try:
        await owner.start()
        await entered.wait()
        waiters = [asyncio.create_task(owner.aclose()) for _ in range(2)]
        await asyncio.wait_for(cleaning.wait(), timeout=2)
        waiters[0].cancel()
        await asyncio.sleep(0)
        waiters[0].cancel()
        await asyncio.sleep(0)
        assert not any(waiter.done() for waiter in waiters)
        assert owner.get_status()["active_workers"] == 1
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await waiters[0]
        await waiters[1]
        assert calls == ["cleaned"]
    finally:
        release.set()
        await cleanup(owner)
        await asyncio.gather(*waiters, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("closing", [False, True])
async def test_admitted_startup_is_shared_and_retained(monkeypatch, closing):
    owner = monitoring.AdvancedMonitoringService(enable_tracing=False)
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    original_start = owner.metrics_collector._start_background_tasks

    async def delayed():
        calls.append("startup")
        entered.set()
        await release.wait()
        await original_start()

    async def park():
        await asyncio.Future()

    monkeypatch.setattr(owner.metrics_collector, "_start_background_tasks", delayed)
    monkeypatch.setattr(owner, "_monitoring_loop", park)
    first = asyncio.create_task(owner.start())
    second = close = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        second = asyncio.create_task(owner.start())
        await asyncio.sleep(0)
        first.cancel()
        await asyncio.sleep(0)
        first.cancel()
        await asyncio.sleep(0)
        assert not first.done() and not second.done()
        if closing:
            close = asyncio.create_task(owner.aclose())
            await asyncio.sleep(0)
            assert owner.state.value == "closing"
            assert not close.done()
            with pytest.raises(RuntimeError, match="closing"):
                await owner.start()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        await second
        assert calls == ["startup"]
        if close:
            await close
            assert owner.state.value == "closed"
        else:
            assert owner.state.value == "running"
    finally:
        release.set()
        await cleanup(owner)
        await asyncio.gather(
            first, *[task for task in (second, close) if task], return_exceptions=True
        )


@pytest.mark.asyncio
async def test_startup_failure_rolls_back_already_started_workers(monkeypatch):
    owner = monitoring.AdvancedMonitoringService(enable_tracing=False)
    started, cleaned = asyncio.Event(), asyncio.Event()
    failure = ValueError("alert startup failed")

    async def operation():
        started.set()
        try:
            await asyncio.Future()
        finally:
            cleaned.set()

    async def fail():
        await started.wait()
        raise failure

    monkeypatch.setattr(owner, "_monitoring_loop", operation)
    monkeypatch.setattr(owner.alert_manager, "_start_background_tasks", fail)
    try:
        with pytest.raises(ValueError) as caught:
            await owner.start()
        assert caught.value is failure
        assert cleaned.is_set()
        assert owner.get_status()["active_workers"] == 0
        assert owner.metrics_collector.state.value == "closed"
        assert owner.alert_manager.state.value == "failed"
        assert owner.state.value == "failed"
        with pytest.raises(ValueError) as repeated:
            await owner.aclose()
        assert repeated.value is failure
    finally:
        await cleanup(owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_alert_cleanup", [False, True])
async def test_service_drains_producer_then_alerts_then_metrics(
    monkeypatch, fail_alert_cleanup
):
    owner = monitoring.AdvancedMonitoringService(enable_tracing=False)
    entered = {name: asyncio.Event() for name in ("monitoring", "alerts", "metrics")}
    cleaning, release = asyncio.Event(), asyncio.Event()
    events = []
    failure = ValueError("notification cleanup failed")

    async def operation(name):
        entered[name].set()
        try:
            await asyncio.Future()
        finally:
            if name == "monitoring":
                cleaning.set()
                await release.wait()
            events.append(name)
            if name == "alerts" and fail_alert_cleanup:
                raise failure

    monkeypatch.setattr(owner, "_monitoring_loop", lambda: operation("monitoring"))
    monkeypatch.setattr(
        owner.alert_manager, "_check_alerts", lambda: operation("alerts")
    )
    monkeypatch.setattr(
        owner.metrics_collector, "_cleanup_old_metrics", lambda: operation("metrics")
    )
    closing = None
    try:
        await owner.start()
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())), timeout=2
        )
        closing = asyncio.create_task(owner.aclose())
        await asyncio.wait_for(cleaning.wait(), timeout=2)
        assert not events
        assert owner.alert_manager.state.value == "running"
        assert owner.metrics_collector.state.value == "running"
        release.set()
        if fail_alert_cleanup:
            with pytest.raises(ValueError) as caught:
                await closing
            assert caught.value is failure
            with pytest.raises(ValueError) as repeated:
                await owner.aclose()
            assert repeated.value is failure
            assert owner.state.value == "failed"
        else:
            await closing
            assert owner.state.value == "closed"
        assert events == ["monitoring", "alerts", "metrics"]
        assert owner.metrics_collector.state.value == "closed"
    finally:
        release.set()
        await cleanup(owner)
        if closing:
            await asyncio.gather(closing, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["return", "cancel", "error"])
async def test_unexpected_child_exit_is_visible_in_service_health(monkeypatch, outcome):
    owner = monitoring.AdvancedMonitoringService(enable_tracing=False)
    entered, finish = asyncio.Event(), asyncio.Event()

    async def operation():
        entered.set()
        await finish.wait()
        if outcome == "error":
            raise AssertionError("worker bug")
        if outcome == "cancel":
            raise asyncio.CancelledError

    async def park():
        await asyncio.Future()

    monkeypatch.setattr(owner, "_monitoring_loop", park)
    monkeypatch.setattr(owner.alert_manager, "_check_alerts", operation)
    try:
        await owner.start()
        await entered.wait()
        assert owner.get_health_summary()["status"] == "healthy"
        finish.set()
        await asyncio.gather(owner.alert_manager._alert_task, return_exceptions=True)
        summary = owner.get_health_summary()
        assert summary["status"] == "unhealthy"
        assert summary["lifecycle"]["state"] == "failed"
        assert summary["lifecycle"]["children"]["alerts"]["state"] == "failed"
        with pytest.raises(RuntimeError, match="failed"):
            await owner.start()
    finally:
        finish.set()
        await cleanup(owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("during_startup", [False, True])
async def test_self_close_is_rejected_even_with_eager_scheduling(
    monkeypatch, eager, during_startup
):
    if eager and not hasattr(asyncio, "eager_task_factory"):
        pytest.skip("Eager task factory requires Python 3.12")
    owner = monitoring.MetricsCollector()
    entered = asyncio.Event()
    original_start = owner._start_background_tasks

    async def operation():
        with pytest.raises(RuntimeError, match="own owner"):
            await owner.aclose()
        entered.set()
        if during_startup:
            await original_start()
        else:
            await asyncio.Future()

    monkeypatch.setattr(
        owner,
        "_start_background_tasks" if during_startup else "_cleanup_old_metrics",
        operation,
    )
    loop = asyncio.get_running_loop()
    original_factory = loop.get_task_factory()
    try:
        if eager:
            loop.set_task_factory(asyncio.eager_task_factory)
        await asyncio.wait_for(owner.start(), timeout=2)
        await asyncio.wait_for(entered.wait(), timeout=2)
        assert owner.state.value == "running"
    finally:
        loop.set_task_factory(original_factory)
        await cleanup(owner)


@pytest.mark.asyncio
async def test_cpu_sampling_keeps_event_loop_responsive_and_is_drained(monkeypatch):
    owner = monitoring.AdvancedMonitoringService(
        enable_alerts=False, enable_tracing=False
    )
    loop = asyncio.get_running_loop()
    entered, closing_entered = asyncio.Event(), asyncio.Event()
    release = threading.Event()
    event_loop_thread = threading.get_ident()
    calls = []

    def sample_cpu(interval):
        assert threading.get_ident() != event_loop_thread
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(timeout=3)
        calls.append("sample finished")
        return 25.0

    original_close = owner._close

    async def closing():
        closing_entered.set()
        await original_close()

    monkeypatch.setattr(monitoring.psutil, "cpu_percent", sample_cpu)
    monkeypatch.setattr(owner, "_close", closing)
    close = None
    try:
        await owner.start()
        await asyncio.wait_for(entered.wait(), timeout=2)
        close = asyncio.create_task(owner.aclose())
        await asyncio.wait_for(closing_entered.wait(), timeout=2)
        await asyncio.sleep(0)
        assert not close.done()
        assert owner.get_status()["active_workers"] == 1
        assert not calls
        release.set()
        await asyncio.wait_for(close, timeout=2)
        assert calls == ["sample finished"]
        assert owner.state.value == "closed"
    finally:
        release.set()
        await cleanup(owner)
        if close:
            await asyncio.gather(close, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_cleanup", [False, True])
async def test_collection_cancellation_cannot_restart_the_polling_loop(
    monkeypatch, fail_cleanup
):
    owner = monitoring.AdvancedMonitoringService(enable_tracing=False)
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    failure = RuntimeError("collection cleanup failed")
    calls = []

    async def collect():
        calls.append("collect")
        entered.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cleaning.set()
            await release.wait()
            if fail_cleanup:
                raise failure from None
            # Cancellation-resistant providers can return after unwinding.

    monkeypatch.setattr(owner.metrics_collector, "collect_system_metrics", collect)
    close = None
    try:
        await owner.start()
        await asyncio.wait_for(entered.wait(), timeout=2)
        close = asyncio.create_task(owner.aclose())
        await asyncio.wait_for(cleaning.wait(), timeout=2)
        release.set()
        done, _ = await asyncio.wait({close}, timeout=2)
        assert close in done
        if fail_cleanup:
            with pytest.raises(RuntimeError) as caught:
                await close
            assert caught.value is failure
        else:
            await close
            assert owner.state.value == "closed"
        assert calls == ["collect"]
    finally:
        release.set()
        if close and not close.done():
            owner._monitoring_task.cancel()
        await cleanup(owner)
        if close:
            await asyncio.gather(close, return_exceptions=True)


@pytest.mark.asyncio
async def test_close_during_startup_accepts_cooperative_worker_exit(monkeypatch):
    owner = monitoring.AdvancedMonitoringService(enable_tracing=False)
    entered, release = asyncio.Event(), asyncio.Event()
    original_start = owner._start_background_tasks

    async def delayed():
        entered.set()
        await release.wait()
        await original_start()

    monkeypatch.setattr(owner, "_start_background_tasks", delayed)
    start = asyncio.create_task(owner.start())
    close = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        close = asyncio.create_task(owner.aclose())
        await asyncio.sleep(0)
        assert owner.state.value == "closing"
        release.set()
        await asyncio.wait_for(asyncio.gather(start, close), timeout=2)
        assert owner.state.value == "closed"
        assert owner.get_status()["error_type"] is None
    finally:
        release.set()
        await cleanup(owner)
        await asyncio.gather(start, *([close] if close else []), return_exceptions=True)


@pytest.mark.asyncio
async def test_standalone_alert_manager_does_not_close_its_borrowed_collector():
    collector = monitoring.MetricsCollector()
    manager = monitoring.AlertManager(collector)
    try:
        await collector.start()
        await manager.start()
        await manager.aclose()
        assert collector.state.value == "running"
    finally:
        await cleanup(manager)
        await cleanup(collector)
