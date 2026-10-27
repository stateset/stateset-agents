"""A judge that returns no verdict fails closed and is never cached."""

import pytest

from stateset_agents.rewards.llm_judge import (
    JudgeConfig,
    JudgeProvider,
    JudgeUnavailableError,
    LLMJudge,
)


def _judge(**kwargs):
    return LLMJudge(JudgeConfig(provider=JudgeProvider.LOCAL, **kwargs))


@pytest.mark.asyncio
@pytest.mark.parametrize("normalize_scores", [True, False])
async def test_failed_judgement_raises_instead_of_scoring_the_midpoint(
    normalize_scores,
):
    judge = _judge(score_min=1.0, score_max=10.0, normalize_scores=normalize_scores)
    with pytest.raises(JudgeUnavailableError, match="Local model not configured"):
        await judge.evaluate("q", "r")


@pytest.mark.asyncio
async def test_failed_judgement_is_not_cached():
    judge = _judge()
    calls = []
    original = judge.backend._call_api

    async def counting(messages):
        calls.append(1)
        return await original(messages)

    judge.backend._call_api = counting
    for _ in range(2):
        with pytest.raises(JudgeUnavailableError):
            await judge.evaluate("q", "r")
    assert len(calls) == 2
