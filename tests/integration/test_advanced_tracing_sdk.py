"""Exercise real SDK export and context propagation without network services."""

import asyncio
import json
import subprocess
import sys
import textwrap

import pytest

from stateset_agents.core.advanced_monitoring_tracing import DistributedTracer

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_real_sdk_exports_finished_spans_without_replacing_global_provider():
    trace = pytest.importorskip("opentelemetry.trace")
    sdk = pytest.importorskip("opentelemetry.sdk.trace")
    exports = pytest.importorskip("opentelemetry.sdk.trace.export")
    memory = pytest.importorskip(
        "opentelemetry.sdk.trace.export.in_memory_span_exporter"
    )

    class Exporter(memory.InMemorySpanExporter):
        def __init__(self):
            super().__init__()
            self.shutdowns = 0

        def shutdown(self):
            self.shutdowns += 1
            super().shutdown()

    exporter = Exporter()
    previous_global = trace.get_tracer_provider()
    tracer = DistributedTracer(
        opentelemetry_available=True,
        trace_module=trace,
        tracer_provider_cls=sdk.TracerProvider,
        jaeger_exporter_cls=lambda **kwargs: exporter,
        batch_span_processor_cls=exports.BatchSpanProcessor,
    )
    try:
        assert trace.get_tracer_provider() is previous_global
        async with tracer.trace("parent") as parent:
            async with tracer.trace("child", parent_span_id=parent.span_id):
                pass
        with pytest.raises(ValueError, match="work failed"):
            async with tracer.trace("error"):
                raise ValueError("work failed")
        with pytest.raises(asyncio.CancelledError):
            async with tracer.trace("cancelled"):
                raise asyncio.CancelledError
        await tracer.aclose()
        await tracer.aclose()
        spans = {span.name: span for span in exporter.get_finished_spans()}
        assert set(spans) == {"parent", "child", "error", "cancelled"}
        assert all(span.end_time is not None for span in spans.values())
        assert spans["child"].context.trace_id == spans["parent"].context.trace_id
        assert spans["child"].parent.span_id == spans["parent"].context.span_id
        assert spans["error"].status.status_code is trace.StatusCode.ERROR
        assert spans["error"].events[0].name == "exception"
        assert spans["cancelled"].attributes["cancelled"] is True
        assert str(spans["parent"].context.trace_id) == parent.tags["otel_trace_id"]
        assert exporter.shutdowns == 1
        assert tracer.get_status()["pending_sdk_spans"] == []
        assert trace.get_tracer_provider() is previous_global
    finally:
        await tracer.aclose()


def test_real_sdk_process_exit_fallback_exports_unfinished_span(tmp_path):
    pytest.importorskip("opentelemetry.sdk.trace")
    result = tmp_path / "shutdown.json"
    program = textwrap.dedent("""
        import json
        import sys
        from pathlib import Path
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor, SpanExporter, SpanExportResult,
        )
        from stateset_agents.core.advanced_monitoring_tracing import DistributedTracer

        global_provider = trace.get_tracer_provider()

        class Exporter(SpanExporter):
            def __init__(self):
                self.spans = []

            def export(self, spans):
                self.spans.extend(spans)
                return SpanExportResult.SUCCESS

            def shutdown(self):
                Path(sys.argv[1]).write_text(json.dumps({
                    "names": [span.name for span in self.spans],
                    "interrupted": [span.attributes["interrupted"] for span in self.spans],
                    "global_unchanged": trace.get_tracer_provider() is global_provider,
                }))

        exporter = Exporter()
        tracer = DistributedTracer(
            opentelemetry_available=True,
            trace_module=trace,
            tracer_provider_cls=TracerProvider,
            batch_span_processor_cls=BatchSpanProcessor,
            jaeger_exporter_cls=lambda **kwargs: exporter,
        )
        tracer.start_span("unfinished")
        """)
    subprocess.run(
        [sys.executable, "-c", program, str(result)],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert json.loads(result.read_text()) == {
        "names": ["unfinished"],
        "interrupted": [True],
        "global_unchanged": True,
    }
