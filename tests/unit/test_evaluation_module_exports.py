"""Focused tests for the evaluation module boundary and regressions."""

from __future__ import annotations

import asyncio

import pytest

from stateset_agents.core.evaluation import (
    AgentEvaluator,
    EvaluationConfig,
    EvaluationResults,
    EvaluationSample,
    MetricResult,
    RelevanceMetric,
    SampleResult,
)
from stateset_agents.core.evaluation_metrics import RelevanceMetric as SplitRelevanceMetric
from stateset_agents.core.evaluation_models import (
    EvaluationConfig as SplitEvaluationConfig,
)
from stateset_agents.core.evaluation_models import (
    EvaluationResults as SplitEvaluationResults,
)


class DelayedAgent:
    """Agent stub that completes prompts out of order."""

    async def generate_response(self, prompt):
        if prompt == "slow":
            await asyncio.sleep(0.02)
        return f"response:{prompt}"


def test_evaluation_module_reexports_split_models_and_metrics():
    """Public evaluation module should keep exporting split symbols."""
    assert EvaluationConfig is SplitEvaluationConfig
    assert EvaluationResults is SplitEvaluationResults
    assert RelevanceMetric is SplitRelevanceMetric


@pytest.mark.asyncio
async def test_evaluator_preserves_input_order_for_unsorted_sample_ids():
    """Results should follow input order, not lexical sample-id order."""
    evaluator = AgentEvaluator(EvaluationConfig(metrics=["relevance"], parallel_workers=2))
    samples = [
        EvaluationSample(id="b", input="slow"),
        EvaluationSample(id="a", input="fast"),
    ]

    results = await evaluator.evaluate(DelayedAgent(), samples)

    assert [sample.sample_id for sample in results.samples] == ["b", "a"]


def test_confidence_level_controls_interval_width_and_summary_label():
    """Configured confidence levels should affect interval math and labels."""
    sample_results = [
        SampleResult(
            sample_id=str(index),
            input="input",
            output="output",
            expected_output="expected",
            metrics=[MetricResult("relevance", value)],
            latency_ms=10.0,
        )
        for index, value in enumerate([0.2, 0.4, 0.6, 0.8], start=1)
    ]

    evaluator_90 = AgentEvaluator(EvaluationConfig(confidence_level=0.90))
    evaluator_99 = AgentEvaluator(EvaluationConfig(confidence_level=0.99))

    interval_90 = evaluator_90._compute_confidence_intervals(sample_results)["relevance"]
    interval_99 = evaluator_99._compute_confidence_intervals(sample_results)["relevance"]

    width_90 = interval_90[1] - interval_90[0]
    width_99 = interval_99[1] - interval_99[0]

    assert width_99 > width_90

    summary = EvaluationResults(
        config=EvaluationConfig(confidence_level=0.99),
        samples=sample_results,
        aggregate_metrics={"relevance": 0.5},
        confidence_intervals={"relevance": interval_99},
    ).summary()

    assert "99% CI" in summary
