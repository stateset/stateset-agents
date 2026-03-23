"""
Model Evaluation and Metrics Framework for StateSet Agents

This module provides comprehensive evaluation capabilities including:
- Response quality metrics
- Latency and throughput benchmarking
- A/B testing framework
- Automated evaluation pipelines

Example:
    >>> from stateset_agents.core.evaluation import AgentEvaluator, EvaluationConfig
    >>>
    >>> evaluator = AgentEvaluator(EvaluationConfig(
    ...     metrics=["relevance", "coherence", "helpfulness"],
    ...     num_samples=100,
    ... ))
    >>>
    >>> results = await evaluator.evaluate(agent, test_dataset)
    >>> print(results.summary())
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from typing import Any

from .evaluation_metrics import (
    METRIC_REGISTRY,
    CoherenceMetric,
    ExactMatchMetric,
    HelpfulnessMetric,
    LatencyMetric,
    Metric,
    RelevanceMetric,
    SafetyMetric,
    get_metric,
)
from .evaluation_models import (
    EvaluationConfig,
    EvaluationResults,
    EvaluationSample,
    MetricResult,
    MetricType,
    SampleResult,
)

logger = logging.getLogger(__name__)

EVALUATION_EXCEPTIONS = (
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
    KeyError,
    OSError,
)


def _percentile(values: list[float], percentile: float) -> float:
    """Compute a percentile using linear interpolation."""
    ordered_values = sorted(values)
    if len(ordered_values) == 1:
        return ordered_values[0]

    rank = (len(ordered_values) - 1) * percentile
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(ordered_values) - 1)
    weight = rank - lower_index
    lower_value = ordered_values[lower_index]
    upper_value = ordered_values[upper_index]
    return lower_value + (upper_value - lower_value) * weight


class AgentEvaluator:
    """Main evaluation class for agents."""

    def __init__(self, config: EvaluationConfig | None = None):
        """Initialize evaluator."""
        self.config = config or EvaluationConfig()
        self.metrics = [get_metric(name) for name in self.config.metrics]
        self._response_cache: dict[str, str] = {}

    async def evaluate(
        self,
        agent: Any,
        samples: list[EvaluationSample],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> EvaluationResults:
        """Run evaluation on an agent."""
        start_time = datetime.now()

        if self.config.num_samples is not None:
            samples = samples[: self.config.num_samples]

        indexed_results: list[tuple[int, SampleResult]] = []
        semaphore = asyncio.Semaphore(self.config.parallel_workers)

        async def evaluate_sample(
            sample: EvaluationSample,
            index: int,
        ) -> tuple[int, SampleResult]:
            async with semaphore:
                result = await self._evaluate_single(agent, sample)
            return index, result

        tasks = [
            asyncio.create_task(evaluate_sample(sample, index))
            for index, sample in enumerate(samples)
        ]

        for completed_count, task in enumerate(asyncio.as_completed(tasks), start=1):
            index, result = await task
            indexed_results.append((index, result))
            if progress_callback:
                progress_callback(completed_count, len(samples))

        results = [
            result for _, result in sorted(indexed_results, key=lambda item: item[0])
        ]
        aggregate = self._aggregate_metrics(results)
        confidence_intervals: dict[str, tuple[float, float]] = {}

        if self.config.compute_confidence_intervals:
            confidence_intervals = self._compute_confidence_intervals(results)

        return EvaluationResults(
            config=self.config,
            samples=results,
            aggregate_metrics=aggregate,
            confidence_intervals=confidence_intervals,
            start_time=start_time,
            end_time=datetime.now(),
        )

    async def _evaluate_single(
        self,
        agent: Any,
        sample: EvaluationSample,
    ) -> SampleResult:
        """Evaluate a single sample."""
        input_text = (
            sample.input if isinstance(sample.input, str) else json.dumps(sample.input)
        )
        cache_key = f"{id(agent)}:{input_text}"

        try:
            if self.config.cache_responses and cache_key in self._response_cache:
                output = self._response_cache[cache_key]
                latency_ms = 0.0
            else:
                start = time.time()
                output = await asyncio.wait_for(
                    agent.generate_response(sample.input),
                    timeout=self.config.timeout_seconds,
                )
                latency_ms = (time.time() - start) * 1000

                if self.config.cache_responses:
                    self._response_cache[cache_key] = output

            context = {"latency_ms": latency_ms, **sample.context}
            metrics = []

            for metric in self.metrics:
                try:
                    result = await metric.compute(
                        input_text,
                        output,
                        sample.expected_output,
                        context,
                    )
                    metrics.append(result)
                except EVALUATION_EXCEPTIONS as exc:
                    logger.warning("Metric %s failed: %s", metric.name, exc)
                    metrics.append(
                        MetricResult(metric.name, 0.0, metadata={"error": str(exc)})
                    )

            return SampleResult(
                sample_id=sample.id,
                input=sample.input,
                output=output,
                expected_output=sample.expected_output,
                metrics=metrics,
                latency_ms=latency_ms,
            )
        except asyncio.TimeoutError:
            return SampleResult(
                sample_id=sample.id,
                input=sample.input,
                output="",
                expected_output=sample.expected_output,
                metrics=[],
                latency_ms=self.config.timeout_seconds * 1000,
                error="Timeout",
            )
        except EVALUATION_EXCEPTIONS as exc:
            return SampleResult(
                sample_id=sample.id,
                input=sample.input,
                output="",
                expected_output=sample.expected_output,
                metrics=[],
                latency_ms=0,
                error=str(exc),
            )

    def _aggregate_metrics(self, results: list[SampleResult]) -> dict[str, float]:
        """Aggregate metrics across all samples."""
        metric_values: dict[str, list[float]] = defaultdict(list)

        for result in results:
            if result.error:
                continue
            for metric in result.metrics:
                metric_values[metric.metric_name].append(metric.value)

        aggregated = {
            name: statistics.mean(values)
            for name, values in metric_values.items()
            if values
        }

        latencies = [result.latency_ms for result in results if result.error is None]
        if latencies:
            aggregated["avg_latency_ms"] = statistics.mean(latencies)
            aggregated["p50_latency_ms"] = statistics.median(latencies)
            aggregated["p95_latency_ms"] = _percentile(latencies, 0.95)

        return aggregated

    def _compute_confidence_intervals(
        self,
        results: list[SampleResult],
    ) -> dict[str, tuple[float, float]]:
        """Compute confidence intervals for metrics."""
        metric_values: dict[str, list[float]] = defaultdict(list)

        for result in results:
            if result.error:
                continue
            for metric in result.metrics:
                metric_values[metric.metric_name].append(metric.value)

        intervals = {}
        z_score = statistics.NormalDist().inv_cdf(
            0.5 + (self.config.confidence_level / 2.0)
        )

        for name, values in metric_values.items():
            if len(values) < 2:
                continue

            mean = statistics.mean(values)
            standard_deviation = statistics.stdev(values)
            margin = z_score * (standard_deviation / (len(values) ** 0.5))
            intervals[name] = (mean - margin, mean + margin)

        return intervals

    def clear_cache(self) -> None:
        """Clear response cache."""
        self._response_cache.clear()


class ABTestRunner:
    """A/B testing framework for comparing agents."""

    def __init__(self, config: EvaluationConfig | None = None):
        """Initialize A/B test runner."""
        self.config = config or EvaluationConfig()
        self.evaluator = AgentEvaluator(config)

    async def compare(
        self,
        agent_a: Any,
        agent_b: Any,
        samples: list[EvaluationSample],
        primary_metric: str = "helpfulness",
    ) -> dict[str, Any]:
        """Compare two agents."""
        results_a = await self.evaluator.evaluate(agent_a, samples)
        results_b = await self.evaluator.evaluate(agent_b, samples)

        comparison = {}
        for metric_name in self.config.metrics:
            value_a = results_a.aggregate_metrics.get(metric_name, 0)
            value_b = results_b.aggregate_metrics.get(metric_name, 0)
            comparison[metric_name] = {
                "agent_a": value_a,
                "agent_b": value_b,
                "difference": value_b - value_a,
                "improvement_pct": (
                    ((value_b - value_a) / value_a * 100) if value_a > 0 else 0
                ),
            }

        primary_a = results_a.aggregate_metrics.get(primary_metric, 0)
        primary_b = results_b.aggregate_metrics.get(primary_metric, 0)
        if primary_b > primary_a:
            winner = "agent_b"
        elif primary_a > primary_b:
            winner = "agent_a"
        else:
            winner = "tie"

        return {
            "winner": winner,
            "primary_metric": primary_metric,
            "comparison": comparison,
            "results_a": results_a.to_dict(),
            "results_b": results_b.to_dict(),
        }


__all__ = [
    "ABTestRunner",
    "AgentEvaluator",
    "CoherenceMetric",
    "EvaluationConfig",
    "EvaluationResults",
    "EvaluationSample",
    "ExactMatchMetric",
    "HelpfulnessMetric",
    "LatencyMetric",
    "METRIC_REGISTRY",
    "Metric",
    "MetricResult",
    "MetricType",
    "RelevanceMetric",
    "SafetyMetric",
    "SampleResult",
    "get_metric",
]
