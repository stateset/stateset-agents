"""A judge that returns no verdict must not become a cached low reward."""

import pytest

from stateset_agents.rewards.llm_judge import JudgeConfig, JudgeProvider, LLMJudge


def _judge(**kwargs):
    return LLMJudge(JudgeConfig(provider=JudgeProvider.LOCAL, **kwargs))


@pytest.mark.asyncio
async def test_failed_judgement_is_neutral_on_a_one_to_ten_scale():
    judge = _judge(score_min=1.0, score_max=10.0)
    assert await judge.evaluate("q", "r") == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_failed_judgement_is_not_cached():
    judge = _judge()
    calls = []
    original = judge.backend._call_api

    async def counting(messages):
        calls.append(1)
        return await original(messages)

    judge.backend._call_api = counting
    await judge.evaluate("q", "r")
    await judge.evaluate("q", "r")
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_raw_scale_failure_uses_scale_midpoint():
    judge = _judge(score_min=1.0, score_max=10.0, normalize_scores=False)
    assert await judge.evaluate("q", "r") == pytest.approx(5.5)
