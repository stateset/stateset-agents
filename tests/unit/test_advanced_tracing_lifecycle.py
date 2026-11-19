"""Advanced tracing retains span and exporter ownership through cleanup."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from stateset_agents.core import advanced_monitoring_tracing as tracing
from stateset_agents.core.advanced_monitoring_tracing import DistributedTracer


@pytest.fixture(autouse=True)
def capture_exit_hooks(monkeypatch):
    callbacks = []
    monkeypatch.setattr(
        tracing,
        "atexit",
        SimpleNamespace(
            register=lambda callback: callbacks.append(callback),
            unregister=lambda callback: callbacks.remove(callback),
        ),
    )
    return callbacks


class SDKSpan:
    def __init__(self, name, context=None):
        self.name = name
        self.parent = context
        self.attributes = {}
        self.ends = 0

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def get_span_context(self):
        return SimpleNamespace(span_id=id(self), trace_id=123)

    def end(self):
        self.ends += 1


class SDKTracer:
    def __init__(self):
        self.spans = []

    def start_span(self, name, context=None):
        span = SDKSpan(name, context)
        self.spans.append(span)
        return span


class Provider:
    def __init__(self, *, shutdown_on_exit=True):
        self.shutdown_on_exit = shutdown_on_exit
        self.tracer = SDKTracer()
        self.processors = []
        self.shutdowns = 0

    def add_span_processor(self, processor):
        self.processors.append(processor)

    def get_tracer(self, name):
        return self.tracer

    def shutdown(self):
        self.shutdowns += 1
        for processor in self.processors:
            processor.shutdown()


class Exporter:
    def __init__(self, **kwargs):
        self.shutdowns = 0

    def shutdown(self):
        self.shutdowns += 1


class Processor:
    def __init__(self, exporter):
        self.exporter = exporter
        self.shutdowns = 0

    def shutdown(self):
        self.shutdowns += 1
        self.exporter.shutdown()


class TraceModule:
    def __init__(self):
        self.global_provider = Provider()
        self.global_sets = []

    def set_tracer_provider(self, provider):
        self.global_sets.append(provider)
        self.global_provider = provider

    def get_tracer_provider(self):
        return self.global_provider

    def get_tracer(self, name):
        return self.global_provider.get_tracer(name)

    def set_span_in_context(self, span):
        return span


def make_tracer(**overrides):
    module = overrides["trace_module"] if "trace_module" in overrides else TraceModule()
    providers, exporters, processors = [], [], []

    def provider(**kwargs):
        value = Provider(**kwargs)
        providers.append(value)
        return value

    def exporter(**kwargs):
        value = Exporter(**kwargs)
        exporters.append(value)
        return value

    def processor(value):
        result = Processor(value)
        processors.append(result)
        return result

    options = {
        "opentelemetry_available": True,
        "trace_module": module,
        "tracer_provider_cls": provider,
        "jaeger_exporter_cls": exporter,
        "batch_span_processor_cls": processor,
    }
    options.update(overrides)
    tracer = DistributedTracer(**options)
    return tracer, SimpleNamespace(
        module=module, providers=providers, exporters=exporters, processors=processors
    )


def test_local_finish_ends_sdk_span_exactly_once():
    tracer, sdk = make_tracer()
    span = tracer.start_span("work")
    upstream = sdk.providers[0].tracer.spans[0]
    tracer.finish_span(span.span_id)
    finished = span.end_time
    tracer.finish_span(span.span_id)
    assert upstream.ends == 1
    assert span.end_time == finished
    assert tracer.completed_spans == [span]


def test_sdk_metadata_failure_does_not_lose_the_created_span(monkeypatch):
    tracer, sdk = make_tracer()

    def broken(self, key, value):
        raise ValueError("invalid metadata")

    monkeypatch.setattr(SDKSpan, "set_attribute", broken)
    span = tracer.start_span("work", tags={"x": "value"})
    upstream = sdk.providers[0].tracer.spans[0]
    tracer.finish_span(span.span_id)
    assert upstream.ends == 1


def test_explicit_parent_is_preserved_in_the_sdk_trace():
    tracer, sdk = make_tracer()
    parent = tracer.start_span("parent")
    child = tracer.start_span("child", parent_span_id=parent.span_id)
    upstream = sdk.providers[0].tracer.spans
    assert upstream[1].parent is upstream[0]
    assert child.trace_id == parent.trace_id


def test_tracing_does_not_modify_callers_tags_or_return_summary_aliases():
    tracer, _ = make_tracer()
    tags = {"items": [1]}
    span = tracer.start_span("work", tags=tags)
    assert tags == {"items": [1]}
    tracer.finish_span(span.span_id)
    summary = tracer.get_trace_summary(span.trace_id)
    summary["spans"][0]["tags"]["items"].append(2)
    assert span.tags["items"] == [1]


@pytest.mark.asyncio
async def test_close_owns_private_provider_and_ends_unfinished_spans():
    tracer, sdk = make_tracer()
    span = tracer.start_span("unfinished")
    assert not sdk.module.global_sets
    assert not sdk.providers[0].shutdown_on_exit
    await tracer.aclose()
    await tracer.aclose()
    assert sdk.providers[0].tracer.spans[0].ends == 1
    assert sdk.providers[0].shutdowns == 1
    assert sdk.processors[0].shutdowns == 1
    assert sdk.exporters[0].shutdowns == 1
    assert sdk.module.global_provider.shutdowns == 0
    assert span.tags["interrupted"] is True
    assert tracer.get_status()["state"] == "closed"


@pytest.mark.asyncio
async def test_cancelled_trace_finishes_sdk_span_and_records_cancellation():
    tracer, sdk = make_tracer()
    entered = asyncio.Event()

    async def work():
        async with tracer.trace("work"):
            entered.set()
            await asyncio.Future()

    task = asyncio.create_task(work())
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sdk.providers[0].tracer.spans[0].ends == 1
        assert tracer.completed_spans[0].tags["cancelled"] is True
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await tracer.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phase",
    ["provider", "exporter", "processor", "attach_before", "attach_after", "tracer"],
)
async def test_partial_setup_retains_returned_resources_for_cleanup(monkeypatch, phase):
    failure = ValueError("setup failed")

    def fail(*args, **kwargs):
        raise failure

    if phase in ("provider", "exporter", "processor"):
        factory = {"provider": Provider, "exporter": Exporter, "processor": Processor}[
            phase
        ]
        monkeypatch.setattr(factory, "__init__", fail)
    elif phase.startswith("attach"):
        original = Provider.add_span_processor

        def attach(provider, processor):
            if phase == "attach_after":
                original(provider, processor)
            raise failure

        monkeypatch.setattr(Provider, "add_span_processor", attach)
    else:
        monkeypatch.setattr(Provider, "get_tracer", fail)

    # The process-global provider is outside this setup, including failure to
    # construct the local provider.
    module = SimpleNamespace()
    tracer, sdk = make_tracer(trace_module=module)
    assert tracer.get_status()["setup_error_type"] == "ValueError"
    assert not tracer.get_status()["sdk_available"]
    local = tracer.start_span("local fallback")
    tracer.finish_span(local.span_id)
    await tracer.aclose()
    await tracer.aclose()
    assert tracer.get_status()["state"] == "closed"
    assert not tracer.get_status()["provider_owned"]
    if sdk.providers:
        # Before confirmed attachment, cleanup belongs to the returned leaf
        # resource. The empty private provider owns no threads or exit hooks.
        assert sdk.providers[0].shutdowns == (
            0 if phase in {"processor", "attach_before", "attach_after"} else 1
        )
    if sdk.exporters:
        assert sdk.exporters[0].shutdowns == 1
    if sdk.processors:
        assert sdk.processors[0].shutdowns == 1


@pytest.mark.asyncio
async def test_failed_span_end_is_retained_without_replay_or_masking_work(monkeypatch):
    tracer, sdk = make_tracer()
    end_failure = ValueError("end outcome unknown")
    work_failure = LookupError("application failed")

    def end(span):
        span.ends += 1
        raise end_failure

    monkeypatch.setattr(SDKSpan, "end", end)
    with pytest.raises(LookupError) as caught:
        async with tracer.trace("work") as span:
            raise work_failure
    assert caught.value is work_failure
    tracer.finish_span(span.span_id)
    assert sdk.providers[0].tracer.spans[0].ends == 1
    assert tracer.get_status()["failed_span_ends"] == [span.span_id]
    for _ in range(2):
        with pytest.raises(ValueError) as cleanup_error:
            await tracer.aclose()
        assert cleanup_error.value is end_failure
    assert sdk.providers[0].tracer.spans[0].ends == 1
    assert sdk.providers[0].shutdowns == 1
    assert tracer.get_status()["pending_sdk_spans"] == [span.span_id]
    assert tracer.get_status()["state"] == "failed"


@pytest.mark.asyncio
async def test_failed_provider_shutdown_retains_handles_and_the_same_outcome(
    monkeypatch,
):
    tracer, sdk = make_tracer()
    failure = OSError("shutdown uncertain")
    original = Provider.shutdown

    def shutdown(provider):
        original(provider)
        raise failure

    monkeypatch.setattr(Provider, "shutdown", shutdown)
    for _ in range(2):
        with pytest.raises(OSError) as caught:
            await tracer.aclose()
        assert caught.value is failure
    status = tracer.get_status()
    assert status["state"] == "failed"
    assert status["provider_owned"] and status["processor_owned"]
    assert status["exporter_owned"] and not status["sdk_available"]
    assert status["shutdown_error_type"] == "OSError"
    assert sdk.providers[0].shutdowns == 1
    assert sdk.processors[0].shutdowns == 1
    assert sdk.exporters[0].shutdowns == 1


@pytest.mark.asyncio
async def test_cancelled_close_waiters_cannot_detach_blocking_sdk_shutdown(monkeypatch):
    tracer, sdk = make_tracer()
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    event_loop_thread = threading.get_ident()
    original = Provider.shutdown

    def shutdown(provider):
        assert threading.get_ident() != event_loop_thread
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(timeout=3)
        original(provider)

    monkeypatch.setattr(Provider, "shutdown", shutdown)
    first = asyncio.create_task(tracer.aclose())
    second = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        second = asyncio.create_task(tracer.aclose())
        first.cancel()
        await asyncio.sleep(0)
        first.cancel()
        await asyncio.sleep(0)
        assert not first.done() and not second.done()
        assert tracer.get_status()["state"] == "closing"
        with pytest.raises(RuntimeError, match="closing"):
            tracer.start_span("late")
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        await second
        assert tracer.get_status()["state"] == "closed"
        assert sdk.providers[0].shutdowns == 1
        with pytest.raises(RuntimeError, match="closed"):
            tracer.start_span("late")
    finally:
        release.set()
        await asyncio.gather(
            first, *([second] if second else []), return_exceptions=True
        )
        await tracer.aclose()


@pytest.mark.asyncio
async def test_close_inside_trace_marks_interruption_without_waiting_for_work():
    tracer, sdk = make_tracer()
    with pytest.raises(ValueError, match="later application failure"):
        async with tracer.trace("work") as span:
            await tracer.aclose()
            finished = span.end_time
            assert span.tags["interrupted"] is True
            raise ValueError("later application failure")
    assert span.end_time == finished
    assert "error" not in span.tags
    assert sdk.providers[0].tracer.spans[0].ends == 1


def test_sdk_end_is_not_repeated_by_reentrant_finish(monkeypatch):
    tracer, sdk = make_tracer()
    span = tracer.start_span("work")

    def end(upstream):
        upstream.ends += 1
        tracer.finish_span(span.span_id)

    monkeypatch.setattr(SDKSpan, "end", end)
    tracer.finish_span(span.span_id)
    assert sdk.providers[0].tracer.spans[0].ends == 1
    assert tracer.completed_spans == [span]


def test_local_span_completion_preserves_the_first_timestamp():
    tracer = DistributedTracer()
    span = tracer.start_span("work")
    span.finish()
    finished, duration = span.end_time, span.duration
    span.finish()
    tracer.finish_span(span.span_id)
    assert (span.end_time, span.duration) == (finished, duration)


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_failure", [False, True])
async def test_monitoring_closes_exporters_after_workers_even_on_failure(
    monkeypatch, worker_failure
):
    from stateset_agents.core import advanced_monitoring as monitoring

    monkeypatch.setattr(monitoring, "PROMETHEUS_AVAILABLE", False)
    service = monitoring.AdvancedMonitoringService(enable_tracing=False)
    tracer, sdk = make_tracer()
    service.tracer = tracer
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    events = []
    failure = ValueError("worker cleanup failed")
    original_end, original_shutdown = SDKSpan.end, Provider.shutdown

    def end(span):
        events.append("span")
        original_end(span)

    def shutdown(provider):
        events.append("provider")
        original_shutdown(provider)

    async def worker():
        async with tracer.trace("worker"):
            entered.set()
            try:
                await asyncio.Future()
            finally:
                cleaning.set()
                await release.wait()
                events.append("worker")
                if worker_failure:
                    raise failure

    monkeypatch.setattr(SDKSpan, "end", end)
    monkeypatch.setattr(Provider, "shutdown", shutdown)
    monkeypatch.setattr(service, "_monitoring_loop", worker)
    closing = None
    try:
        await service.start()
        await asyncio.wait_for(entered.wait(), timeout=2)
        closing = asyncio.create_task(service.aclose())
        await asyncio.wait_for(cleaning.wait(), timeout=2)
        assert tracer.get_status()["state"] == "ready"
        assert not events
        release.set()
        if worker_failure:
            with pytest.raises(ValueError) as caught:
                await closing
            assert caught.value is failure
            assert service.state.value == "failed"
        else:
            await closing
            assert service.state.value == "closed"
        assert events == ["worker", "span", "provider"]
        assert service.get_status()["tracing"]["state"] == "closed"
        assert sdk.exporters[0].shutdowns == 1
    finally:
        release.set()
        try:
            await service.aclose()
        except ValueError:
            pass
        if closing:
            await asyncio.gather(closing, return_exceptions=True)


@pytest.mark.asyncio
async def test_tracer_failure_is_visible_in_monitoring_shutdown(monkeypatch):
    from stateset_agents.core import advanced_monitoring as monitoring

    service = monitoring.AdvancedMonitoringService(
        enable_metrics=False, enable_alerts=False, enable_tracing=False
    )
    tracer, sdk = make_tracer()
    service.tracer = tracer
    failure = RuntimeError("exporter cleanup failed")

    def shutdown(provider):
        provider.shutdowns += 1
        raise failure

    monkeypatch.setattr(Provider, "shutdown", shutdown)
    for _ in range(2):
        with pytest.raises(RuntimeError) as caught:
            await service.aclose()
        assert caught.value is failure
    status = service.get_status()
    assert status["state"] == "failed"
    assert status["tracing"]["state"] == "failed"
    assert status["tracing"]["provider_owned"]
    assert sdk.providers[0].shutdowns == 1


@pytest.mark.asyncio
async def test_closing_one_tracer_preserves_sibling_and_global_provider():
    module = TraceModule()
    first, first_sdk = make_tracer(trace_module=module)
    second, second_sdk = make_tracer(trace_module=module)
    try:
        first.start_span("first")
        await first.aclose()
        async with second.trace("still available"):
            pass
        assert second.get_status()["state"] == "ready"
        assert second_sdk.providers[0].shutdowns == 0
        assert second_sdk.providers[0].tracer.spans[0].ends == 1
        assert module.global_provider.shutdowns == 0
        assert not module.global_sets
    finally:
        await first.aclose()
        await second.aclose()
    assert first_sdk.exporters[0].shutdowns == second_sdk.exporters[0].shutdowns == 1


@pytest.mark.asyncio
async def test_sdk_end_cancellation_is_propagated_after_resource_cleanup(monkeypatch):
    tracer, sdk = make_tracer()

    def end(span):
        span.ends += 1
        raise asyncio.CancelledError

    monkeypatch.setattr(SDKSpan, "end", end)
    span = tracer.start_span("work")
    for _ in range(2):
        with pytest.raises(asyncio.CancelledError):
            await tracer.aclose()
    assert sdk.providers[0].shutdowns == 1
    assert sdk.providers[0].tracer.spans[0].ends == 1
    assert tracer.get_status()["state"] == "failed"
    assert tracer.get_status()["pending_sdk_spans"] == [span.span_id]


@pytest.mark.asyncio
async def test_exit_fallback_closes_once_and_async_close_observes_it(
    capture_exit_hooks,
):
    tracer, sdk = make_tracer()
    span = tracer.start_span("unfinished")
    callback = capture_exit_hooks[0]
    callback()
    assert capture_exit_hooks == [callback]
    callback()
    await tracer.aclose()
    assert not capture_exit_hooks
    assert span.tags["interrupted"] is True
    assert sdk.providers[0].tracer.spans[0].ends == 1
    assert sdk.providers[0].shutdowns == 1
    assert tracer.state.value == "closed"


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_first", [False, True])
async def test_failed_cleanup_is_not_replayed_by_exit_or_async_close(
    monkeypatch, capture_exit_hooks, exit_first
):
    tracer, sdk = make_tracer()
    failure = OSError("cleanup uncertain")
    callback = capture_exit_hooks[0]

    def shutdown(provider):
        provider.shutdowns += 1
        raise failure

    monkeypatch.setattr(Provider, "shutdown", shutdown)
    if exit_first:
        callback()
    with pytest.raises(OSError) as caught:
        await tracer.aclose()
    assert caught.value is failure
    callback()
    assert sdk.providers[0].shutdowns == 1
    assert tracer.state.value == "failed"
    assert not capture_exit_hooks
