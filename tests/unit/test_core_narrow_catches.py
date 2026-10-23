"""Core handlers no longer swallow programmer errors (TypeError and friends).

Each test pins a behaviour that changed when a broad catch was narrowed: the
bug now surfaces where it happens instead of being retried, logged or turned
into a degraded result.
"""

from __future__ import annotations

from typing import Any

import pytest

from stateset_agents.core.async_pool import AsyncResourceFactory, AsyncResourcePool
from stateset_agents.core.error_handling import ResourceException
from stateset_agents.core.structured_output import (
    StructuredOutputError,
    StructuredOutputMixin,
)

_SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}}


class _Agent(StructuredOutputMixin):
    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.calls = 0

    async def generate_response(self, messages: Any, context: Any = None) -> str:
        self.calls += 1
        raise self.error


@pytest.mark.asyncio
async def test_structured_output_does_not_retry_a_programmer_error() -> None:
    agent = _Agent(TypeError("generate_response() got an unexpected keyword"))
    with pytest.raises(TypeError, match="unexpected keyword"):
        await agent.generate_structured("hi", json_schema=_SCHEMA)
    assert agent.calls == 1


@pytest.mark.asyncio
async def test_structured_output_still_retries_generation_failures() -> None:
    agent = _Agent(RuntimeError("model server hiccup"))
    with pytest.raises(StructuredOutputError):
        await agent.generate_structured("hi", json_schema=_SCHEMA)
    assert agent.calls == 3


class _BuggyFactory(AsyncResourceFactory[object]):
    def __init__(self, fail_after: int) -> None:
        self.created = 0
        self.cleaned: list[object] = []
        self.fail_after = fail_after

    async def create_resource(self, **kwargs: Any) -> object:
        if self.created >= self.fail_after:
            raise TypeError("create_resource() missing 1 required argument")
        self.created += 1
        return object()

    async def cleanup_resource(self, resource: object) -> None:
        self.cleaned.append(resource)


@pytest.mark.asyncio
async def test_pool_factory_bug_is_not_a_failed_connection() -> None:
    pool = AsyncResourcePool(_BuggyFactory(fail_after=0), min_size=0)
    with pytest.raises(TypeError, match="missing 1 required"):
        await pool._create_resource()
    assert pool._stats.failed_connections == 0


@pytest.mark.asyncio
async def test_pool_initialisation_cleans_up_whatever_stops_it() -> None:
    factory = _BuggyFactory(fail_after=1)
    pool = AsyncResourcePool(factory, min_size=2, health_check_interval=0)
    with pytest.raises(TypeError):
        await pool.initialize()
    assert len(factory.cleaned) == 1


@pytest.mark.asyncio
async def test_pool_still_wraps_connection_failures() -> None:
    class Down(AsyncResourceFactory[object]):
        async def create_resource(self, **kwargs: Any) -> object:
            raise OSError("connection refused")

    pool = AsyncResourcePool(Down(), min_size=1, health_check_interval=0)
    with pytest.raises(ResourceException):
        await pool.initialize()


@pytest.mark.asyncio
async def test_generation_config_tolerates_a_non_finite_token_id() -> None:
    pytest.importorskip("transformers")
    from stateset_agents.core.agent import AgentConfig, MultiTurnAgent

    agent = MultiTurnAgent(AgentConfig(model_name="stub://ids", use_stub_model=True))
    await agent.initialize()
    agent.tokenizer.pad_token_id = float("inf")  # int(inf) -> OverflowError
    config = agent._build_generation_config()
    assert config.pad_token_id is None


def test_stub_model_shape_probe_only_tolerates_shape_errors() -> None:
    from stateset_agents.core.agent_backends import StubModel

    class BrokenIds:
        @property
        def shape(self) -> Any:
            raise RuntimeError("device lost")

    model = StubModel()
    with pytest.raises(RuntimeError, match="device lost"):
        model(input_ids=BrokenIds())
    # A flat list of ids still falls back to the default shape.
    assert model(input_ids=[1, 2, 3]) is not None


class _Echo:
    async def generate_response(self, prompt: Any) -> str:
        return "hello"


class _BuggyMetric:
    name = "buggy"

    async def compute(self, *args: Any) -> Any:
        raise AttributeError("'NoneType' object has no attribute 'lower'")


class _JudgeDownMetric:
    name = "judge"

    async def compute(self, *args: Any) -> Any:
        raise RuntimeError("judge unavailable")


@pytest.mark.asyncio
async def test_evaluation_does_not_score_a_metric_bug_as_zero() -> None:
    from stateset_agents.core.evaluation import AgentEvaluator
    from stateset_agents.core.evaluation_models import EvaluationSample

    evaluator = AgentEvaluator()
    evaluator.metrics = [_BuggyMetric()]
    with pytest.raises(AttributeError, match="no attribute 'lower'"):
        await evaluator._evaluate_single(_Echo(), EvaluationSample(id="1", input="hi"))


@pytest.mark.asyncio
async def test_evaluation_records_an_expected_metric_failure() -> None:
    from stateset_agents.core.evaluation import AgentEvaluator
    from stateset_agents.core.evaluation_models import EvaluationSample

    evaluator = AgentEvaluator()
    evaluator.metrics = [_JudgeDownMetric()]
    result = await evaluator._evaluate_single(
        _Echo(), EvaluationSample(id="1", input="hi")
    )
    assert result.metrics[0].metadata == {"error": "judge unavailable"}
