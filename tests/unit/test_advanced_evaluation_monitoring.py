"""Regression tests for AdvancedEvaluator's continuous-monitoring hand-off."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from stateset_agents.core.enhanced.advanced_evaluation import (
    AdvancedEvaluator,
    EvaluationMetrics,
)


def _evaluator_with_qualities(qualities: dict[str, float]) -> AdvancedEvaluator:
    evaluator = AdvancedEvaluator()
    evaluator.test_suite.run_test_suite = AsyncMock(return_value={})

    async def fake_metrics(agent, test_results, environment):
        return EvaluationMetrics(response_quality=qualities[agent.name])

    evaluator._calculate_comprehensive_metrics = fake_metrics
    evaluator.monitor.start_monitoring = AsyncMock()
    return evaluator


@pytest.mark.asyncio
async def test_monitoring_receives_best_agent_not_its_result() -> None:
    """Regression: the monitor used to be handed an EvaluationResult."""
    agents = [SimpleNamespace(name="weak"), SimpleNamespace(name="strong")]
    environment = MagicMock()
    evaluator = _evaluator_with_qualities({"weak": 0.2, "strong": 0.9})

    results = await evaluator.comprehensive_evaluation(
        agents,
        environment,
        {"num_test_runs": 1, "include_monitoring": True, "comparative_analysis": False},
    )

    assert evaluator.monitoring_task is not None
    await evaluator.monitoring_task
    assert [r.agent_name for r in results] == ["weak", "strong"]
    evaluator.monitor.start_monitoring.assert_awaited_once_with(agents[1], environment)


@pytest.mark.asyncio
async def test_monitoring_skipped_when_no_agents() -> None:
    evaluator = _evaluator_with_qualities({})

    results = await evaluator.comprehensive_evaluation(
        [],
        MagicMock(),
        {"num_test_runs": 1, "include_monitoring": True, "comparative_analysis": False},
    )

    assert results == []
    assert evaluator.monitoring_task is None
    evaluator.monitor.start_monitoring.assert_not_called()
