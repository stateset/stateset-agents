"""Local transport checks for the reward scale consumed by RLAIF."""

import math
import sys
from unittest.mock import AsyncMock

import pytest

from stateset_agents.rewards.llm_judge import JudgeConfig, LLMJudge, ResultCache


def _judge(*replies, **settings):
    judge = LLMJudge(JudgeConfig(provider="local", **settings))
    judge.backend._call_api = AsyncMock(side_effect=replies)
    return judge


@pytest.mark.asyncio
async def test_rlaif_reads_bounded_judge_score_across_extreme_intervals():
    from stateset_agents.training.rlaif_trainer import RLAIFTrainer

    top = sys.float_info.max
    trainer = RLAIFTrainer.__new__(RLAIFTrainer)
    judge = trainer.judge = _judge(
        '{"score":0}',
        '{"score":0}',
        "Criterion 1, score: -2.5e-1",
        score_min=-top,
        score_max=top,
    )
    assert await trainer.compute_reward("q", "a") == 0.5
    assert await trainer.compute_reward("q", "a") == 0.5
    judge.config.score_min = 0.0
    assert await trainer.compute_reward("q", "a") == 0.0
    judge.config.score_min, judge.config.score_max = -1.0, 1.0
    assert await trainer.compute_reward("q", "a") == 0.375
    assert judge.backend._call_api.await_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity", "1e999"])
@pytest.mark.parametrize("normalize", [True, False])
async def test_nonfinite_judge_results_are_rejected_before_cache(bad, normalize):
    judge = _judge(
        '{"score":"' + bad + '"}', '{"score": 0.25}', normalize_scores=normalize
    )
    with pytest.raises(ValueError, match="finite"):
        await judge.evaluate("q", "a")
    assert await judge.evaluate("q", "a") == 0.25
    assert await judge.evaluate("q", "a") == 0.25
    assert judge.backend._call_api.await_count == 2


@pytest.mark.asyncio
async def test_judge_batches_keep_fallback_at_its_input_position():
    judge = _judge('{"score": 0}', '{"score": "NaN"}', '{"score": 1}')
    with pytest.raises(ValueError, match="one response"):
        await judge.evaluate_batch(["first", "second"], ["only"])
    judge.backend._call_api.assert_not_awaited()
    assert await judge.evaluate_batch(["a", "b", "c"], ["x", "y", "z"]) == [0, 0.5, 1]


@pytest.mark.asyncio
async def test_judge_free_text_nonfinite_and_legacy_ten_point_scores():
    judge = _judge("Score: NaN", "Criterion 1, score: 7/10")
    with pytest.raises(ValueError, match="finite"):
        await judge.evaluate("q", "r")
    assert await judge.evaluate("q", "r") == 0.7


@pytest.mark.asyncio
async def test_missing_score_stays_neutral_on_nonunit_rubrics_and_can_be_retried():
    judge = _judge("unavailable", "{}", '{"score": -2}', score_min=-2, score_max=0)
    assert await judge.evaluate("q", "r") == 0.5
    assert await judge.evaluate("q", "r") == 0.5
    assert await judge.evaluate("q", "r") == 0
    assert judge.backend._call_api.await_count == 3


@pytest.mark.asyncio
async def test_inflight_score_uses_the_rubric_sent_to_the_judge():
    judge = _judge()

    async def change_rubric(messages):
        judge.config.score_max = 20.0
        return "Score: 7"

    judge.backend._call_api = AsyncMock(side_effect=change_rubric)
    assert await judge.evaluate("q", "r") == 0.7
    assert await judge.evaluate("q", "r") == 0.35


@pytest.mark.asyncio
@pytest.mark.parametrize("bounds", [(0, 0), (2, 1), (-math.inf, 1), (0, math.nan)])
async def test_invalid_normalized_interval_precedes_model_call(bounds):
    judge = _judge(score_min=bounds[0], score_max=bounds[1])
    with pytest.raises(ValueError, match="bounds"):
        await judge.evaluate("q", "r")
    judge.backend._call_api.assert_not_awaited()


def test_judge_cache_preserves_boundaries_and_repeated_inserts():
    cache = ResultCache(2)
    cache.set("a|b", "c", "rubric", 0.25)
    cache.set("a|b", "c", "rubric", 0.5)
    cache.set("a", "b|c", "rubric", 0.75)
    assert cache.get("a|b", "c", "rubric") == 0.5
    assert cache.get("a", "b|c", "rubric") == 0.75
    cache.set("new", "text", "rubric", 1)
    assert cache.get("a|b", "c", "rubric") is None
    assert cache.get("a", "b|c", "rubric") == 0.75
