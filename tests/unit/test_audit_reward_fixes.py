"""Regressions for reward defects found in the 2026-11 project audit."""

from __future__ import annotations

import pytest

from stateset_agents.rewards.kimi_k25_reward import CodeFormattingHeuristic
from stateset_agents.rewards.llm_judge import ResultCache
from stateset_agents.rewards.llm_judge_adapter import LLMJudgeReward
from stateset_agents.rewards.llm_reward import RulerRewardFunction
from stateset_agents.rewards.nsr_verifier import extract_verdict


@pytest.mark.parametrize(
    "reply",
    ["I can’t approve this refund.", "We won’t approve it."],
)
def test_typographic_apostrophe_still_negates_an_approval(reply: str) -> None:
    assert extract_verdict(reply) == "denied"
    assert extract_verdict(reply.replace("’", "'")) == "denied"


def test_result_cache_survives_resetting_a_key_and_evicts_least_recent() -> None:
    cache = ResultCache(2)
    cache.set("q", "r", "c", 1)
    cache.set("q", "r", "c", 2)  # used to leave a stale second eviction entry
    cache.set("q2", "r", "c", 3)
    assert cache.get("q", "r", "c") == 2  # now the most recently used
    cache.set("q3", "r", "c", 4)
    assert cache.get("q2", "r", "c") is None
    assert cache.get("q", "r", "c") == 2 and cache.get("q3", "r", "c") == 4
    assert len(cache.cache) == 2


def test_result_cache_keys_do_not_collide_across_part_boundaries() -> None:
    cache = ResultCache(4)
    cache.set("a|b", "c", "k", "first")
    assert cache.get("a", "b|c", "k") is None
    with pytest.raises(ValueError, match="at least 1"):
        ResultCache(0)


@pytest.mark.asyncio
async def test_code_formatting_does_not_zero_a_non_coding_reply() -> None:
    component = CodeFormattingHeuristic(require_code_blocks=True)
    turns = [{"role": "assistant", "content": "Your order ships tomorrow."}]
    assert await component.compute_score(turns, {}) == 1.0
    assert await component.compute_score(turns, {"is_coding_task": True}) == 0.0


@pytest.fixture
def litellm_present(monkeypatch: pytest.MonkeyPatch) -> None:
    import stateset_agents.rewards.llm_reward as llm_reward

    monkeypatch.setattr(llm_reward, "LITELLM_AVAILABLE", True)


@pytest.mark.usefixtures("litellm_present")
def test_ruler_cache_key_separates_custom_rubrics_and_temperatures() -> None:
    strict = RulerRewardFunction(custom_rubric="strict rubric")
    lenient = RulerRewardFunction(custom_rubric="lenient rubric")
    assert strict._hash_trajectory([], None) != lenient._hash_trajectory([], None)
    warm = RulerRewardFunction(custom_rubric="strict rubric", temperature=0.9)
    assert strict._hash_trajectory([], None) != warm._hash_trajectory([], None)


@pytest.mark.usefixtures("litellm_present")
def test_ruler_rejects_zero_attempts_and_reports_rates_within_bounds() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        RulerRewardFunction(max_retries=0)
    judge = RulerRewardFunction()
    judge.total_calls, judge.cache_hits, judge.fallback_uses = 1, 3, 1
    metrics = judge.get_metrics()
    assert metrics["cache_hit_rate"] == pytest.approx(0.6)
    assert metrics["fallback_rate"] == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_judge_adapter_accepts_a_judge_without_config() -> None:
    class Judge:
        async def evaluate(self, query: str, response: str) -> float:
            return 0.5

    reward = LLMJudgeReward(Judge())
    turns = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    result = await reward.compute_reward(turns)
    assert result.metadata["judge_model"] == "unknown"
