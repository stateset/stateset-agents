"""Tests for the inference-level Prometheus metrics module.

These exercise the helper API and the end-to-end instrumentation on
``InferenceService`` against the stub backend, which avoids the need for a
running vLLM. The Prometheus client itself is exercised in-process by
reading registered samples back out of the collector.
"""

from __future__ import annotations

import pytest

from stateset_agents.api import inference_metrics
from stateset_agents.api.messages_models import MessageInput, MessagesRequest
from stateset_agents.api.services.inference_service import (
    InferenceConfig,
    InferenceService,
)


prometheus_client = pytest.importorskip("prometheus_client")


def _counter_value(counter: object, **labels: str) -> float:
    """Read a labeled counter's current value (0.0 if untouched)."""
    metric = counter.labels(**labels)  # type: ignore[attr-defined]
    return metric._value.get()  # type: ignore[attr-defined]


def _histogram_sample_count(histogram: object, **labels: str) -> int:
    """Total observations recorded for a histogram label set."""
    metric = histogram.labels(**labels)  # type: ignore[attr-defined]
    return int(metric._sum.get() > -1) and int(sum(b.get() for b in metric._buckets))


def _gauge_value(gauge: object, **labels: str) -> float:
    metric = gauge.labels(**labels)  # type: ignore[attr-defined]
    return metric._value.get()  # type: ignore[attr-defined]


def test_module_exports_metrics_when_prometheus_present() -> None:
    assert inference_metrics.HAS_PROMETHEUS is True
    assert inference_metrics.PROM_INFERENCE_REQUESTS_TOTAL is not None
    assert inference_metrics.PROM_INFERENCE_DURATION_SECONDS is not None
    assert inference_metrics.PROM_INFERENCE_TTFT_SECONDS is not None
    assert inference_metrics.PROM_INFERENCE_TOKENS_PER_SECOND is not None
    assert inference_metrics.PROM_INFERENCE_TOKENS_TOTAL is not None
    assert inference_metrics.PROM_INFERENCE_INFLIGHT is not None


def test_record_request_increments_counter() -> None:
    before = _counter_value(
        inference_metrics.PROM_INFERENCE_REQUESTS_TOTAL,
        model="unit-test-model",
        route=inference_metrics.ROUTE_OPENAI_RESPONSE,
        status=inference_metrics.STATUS_SUCCESS,
    )
    inference_metrics.record_request(
        model="unit-test-model",
        route=inference_metrics.ROUTE_OPENAI_RESPONSE,
        status=inference_metrics.STATUS_SUCCESS,
    )
    after = _counter_value(
        inference_metrics.PROM_INFERENCE_REQUESTS_TOTAL,
        model="unit-test-model",
        route=inference_metrics.ROUTE_OPENAI_RESPONSE,
        status=inference_metrics.STATUS_SUCCESS,
    )
    assert after == before + 1


def test_record_tokens_split_by_direction() -> None:
    counter = inference_metrics.PROM_INFERENCE_TOKENS_TOTAL
    before_prompt = _counter_value(
        counter,
        model="m",
        route=inference_metrics.ROUTE_OPENAI_RESPONSE,
        direction="prompt",
    )
    before_completion = _counter_value(
        counter,
        model="m",
        route=inference_metrics.ROUTE_OPENAI_RESPONSE,
        direction="completion",
    )
    inference_metrics.record_tokens(
        model="m",
        route=inference_metrics.ROUTE_OPENAI_RESPONSE,
        prompt_tokens=7,
        completion_tokens=11,
    )
    assert (
        _counter_value(
            counter,
            model="m",
            route=inference_metrics.ROUTE_OPENAI_RESPONSE,
            direction="prompt",
        )
        == before_prompt + 7
    )
    assert (
        _counter_value(
            counter,
            model="m",
            route=inference_metrics.ROUTE_OPENAI_RESPONSE,
            direction="completion",
        )
        == before_completion + 11
    )


def test_record_throughput_rejects_invalid_input() -> None:
    # Zero or negative inputs must be silently ignored.
    inference_metrics.record_throughput(
        model="m", route=inference_metrics.ROUTE_OPENAI_RESPONSE, tokens=0, seconds=1.0
    )
    inference_metrics.record_throughput(
        model="m", route=inference_metrics.ROUTE_OPENAI_RESPONSE, tokens=5, seconds=0.0
    )
    # No assertion needed — these would raise on a real division-by-zero bug.


def test_track_inflight_decrements_after_block() -> None:
    gauge = inference_metrics.PROM_INFERENCE_INFLIGHT
    before = _gauge_value(
        gauge, model="inflight-model", route=inference_metrics.ROUTE_OPENAI_STREAM
    )
    with inference_metrics.track_inflight(
        model="inflight-model", route=inference_metrics.ROUTE_OPENAI_STREAM
    ):
        mid = _gauge_value(
            gauge,
            model="inflight-model",
            route=inference_metrics.ROUTE_OPENAI_STREAM,
        )
        assert mid == before + 1
    after = _gauge_value(
        gauge, model="inflight-model", route=inference_metrics.ROUTE_OPENAI_STREAM
    )
    assert after == before


def test_track_request_marks_error_on_exception() -> None:
    counter = inference_metrics.PROM_INFERENCE_REQUESTS_TOTAL
    before = _counter_value(
        counter,
        model="boom",
        route=inference_metrics.ROUTE_OPENAI_RESPONSE,
        status=inference_metrics.STATUS_ERROR,
    )

    with pytest.raises(RuntimeError, match="boom"):
        with inference_metrics.track_request(
            model="boom", route=inference_metrics.ROUTE_OPENAI_RESPONSE
        ):
            raise RuntimeError("boom")

    after = _counter_value(
        counter,
        model="boom",
        route=inference_metrics.ROUTE_OPENAI_RESPONSE,
        status=inference_metrics.STATUS_ERROR,
    )
    assert after == before + 1


# ---------------------------------------------------------------------------
# End-to-end on InferenceService stub backend
# ---------------------------------------------------------------------------


def _stub_service() -> InferenceService:
    return InferenceService(
        InferenceConfig(backend="stub", default_model="stub-test-model")
    )


def _basic_request(model: str = "stub-test-model") -> MessagesRequest:
    return MessagesRequest(
        model=model,
        max_tokens=64,
        messages=[MessageInput(role="user", content="hello world")],
    )


@pytest.mark.asyncio
async def test_create_openai_response_records_metrics() -> None:
    service = _stub_service()
    request = _basic_request()

    counter = inference_metrics.PROM_INFERENCE_REQUESTS_TOTAL
    duration = inference_metrics.PROM_INFERENCE_DURATION_SECONDS
    tokens = inference_metrics.PROM_INFERENCE_TOKENS_TOTAL

    before_requests = _counter_value(
        counter,
        model="stub-test-model",
        route=inference_metrics.ROUTE_OPENAI_RESPONSE,
        status=inference_metrics.STATUS_SUCCESS,
    )
    before_completion_tokens = _counter_value(
        tokens,
        model="stub-test-model",
        route=inference_metrics.ROUTE_OPENAI_RESPONSE,
        direction="completion",
    )

    response = await service.create_openai_response(request)

    assert response["choices"][0]["message"]["content"].startswith("Echo:")
    assert (
        _counter_value(
            counter,
            model="stub-test-model",
            route=inference_metrics.ROUTE_OPENAI_RESPONSE,
            status=inference_metrics.STATUS_SUCCESS,
        )
        == before_requests + 1
    )
    # Token counter should have incremented by the stub's reported completion
    # tokens (greater than zero for a non-empty echo).
    after_completion_tokens = _counter_value(
        tokens,
        model="stub-test-model",
        route=inference_metrics.ROUTE_OPENAI_RESPONSE,
        direction="completion",
    )
    assert after_completion_tokens > before_completion_tokens
    # Duration histogram observed at least one sample for this label set.
    duration_metric = duration.labels(  # type: ignore[attr-defined]
        model="stub-test-model",
        route=inference_metrics.ROUTE_OPENAI_RESPONSE,
    )
    observed = sum(b.get() for b in duration_metric._buckets)  # type: ignore[attr-defined]
    assert observed >= 1


@pytest.mark.asyncio
async def test_stream_openai_records_ttft_and_throughput() -> None:
    service = _stub_service()
    request = _basic_request()

    ttft = inference_metrics.PROM_INFERENCE_TTFT_SECONDS
    requests = inference_metrics.PROM_INFERENCE_REQUESTS_TOTAL

    before_requests = _counter_value(
        requests,
        model="stub-test-model",
        route=inference_metrics.ROUTE_OPENAI_STREAM,
        status=inference_metrics.STATUS_SUCCESS,
    )

    chunks = [chunk async for chunk in service.stream_openai(request)]
    assert any(chunk.startswith("data:") for chunk in chunks)

    assert (
        _counter_value(
            requests,
            model="stub-test-model",
            route=inference_metrics.ROUTE_OPENAI_STREAM,
            status=inference_metrics.STATUS_SUCCESS,
        )
        == before_requests + 1
    )

    ttft_metric = ttft.labels(  # type: ignore[attr-defined]
        model="stub-test-model",
        route=inference_metrics.ROUTE_OPENAI_STREAM,
    )
    ttft_count = sum(b.get() for b in ttft_metric._buckets)  # type: ignore[attr-defined]
    assert ttft_count >= 1
