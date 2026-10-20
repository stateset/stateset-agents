"""Tests for the conversation-level, hack-resistant support reward (v2)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from stateset_agents.core.reward_base import RewardResult
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.data.customer_support_bench import SupportRewardComposite
from stateset_agents.evaluation.support_judge import JudgeItem, detect_findings
from stateset_agents.rewards import SupportConversationReward
from stateset_agents.rewards.support_conversation_reward import (
    content_words,
    turn_similarity,
)

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "support_conversation_reward"
    / "billing_eval_0001.json"
)

BREAKDOWN_KEYS = {
    "base_mean",
    "adjusted_turn_mean",
    "turn_min",
    "assistant_turns",
    "repetition_penalty_mean",
    "max_similarity_mean",
    "placeholder_rate",
    "must_avoid_rate",
    "must_avoid_hits",
    "unresponsive_rate",
    "responsiveness_checked",
}


class ConstantComposite:
    """Base reward of 1.0 per turn, so only the v2 penalties move the score."""

    async def compute_reward(self, turns: Any, context: Any = None) -> RewardResult:
        return RewardResult(score=1.0, breakdown={"const": 1.0})


def _turns(*pairs: tuple[str, str]) -> list[ConversationTurn]:
    return [ConversationTurn(role=role, content=text) for role, text in pairs]


def _score(
    turns: list[ConversationTurn],
    context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> RewardResult:
    reward = SupportConversationReward(
        composite=kwargs.pop("composite", ConstantComposite()), **kwargs
    )
    return asyncio.run(reward.compute_reward(turns, context or {}))


REPLY = (
    "I'm sorry about the double charge on order #4411. I have started a refund "
    "for the duplicate payment; it will reach your card within 5 business days."
)


def test_identical_repeat_scores_zero_for_that_turn() -> None:
    result = _score(
        _turns(
            ("user", "I was charged twice"),
            ("assistant", REPLY),
            ("user", "ok"),
            ("assistant", REPLY),
        )
    )
    assert result.breakdown["turn_min"] == 0.0
    assert result.score == pytest.approx(0.5)
    assert result.breakdown["repetition_penalty_mean"] == pytest.approx(0.5)
    assert result.breakdown["max_similarity_mean"] == pytest.approx(1.0)


def test_repetition_multiplier_is_linear_above_tau() -> None:
    reward = SupportConversationReward(composite=ConstantComposite())
    assert reward.repetition_multiplier(0.3) == 1.0
    assert reward.repetition_multiplier(0.5) == 1.0
    assert reward.repetition_multiplier(0.75) == pytest.approx(0.5)
    assert reward.repetition_multiplier(1.0) == 0.0


def test_similarity_ignores_case_and_whitespace() -> None:
    assert turn_similarity("Hello   THERE\nfriend", "hello there friend") == 1.0


def test_varied_responsive_replies_beat_repeated() -> None:
    user_1 = "I was charged twice for order 4411"
    user_2 = "Will the refund go back to my original credit card or store credit?"
    varied = _turns(
        ("user", user_1),
        ("assistant", REPLY),
        ("user", user_2),
        ("assistant", "It goes back to your original credit card, not store credit."),
    )
    repeated = _turns(
        ("user", user_1), ("assistant", REPLY), ("user", user_2), ("assistant", REPLY)
    )
    composite = SupportRewardComposite()
    context = {"must_acknowledge": ["refund"], "must_avoid": [], "intent": "billing"}
    good = _score(varied, context, composite=composite)
    bad = _score(repeated, context, composite=composite)
    assert good.score > bad.score
    assert good.breakdown["unresponsive_rate"] == 0.0


def test_placeholder_halves_the_turn() -> None:
    result = _score(
        _turns(("user", "hi"), ("assistant", "Hello [Customer's Name], happy to help."))
    )
    assert result.score == pytest.approx(0.5)
    assert result.breakdown["placeholder_rate"] == 1.0
    filled = _score(
        _turns(("user", "hi"), ("assistant", "See [order #123] for details."))
    )
    assert filled.score == 1.0 and filled.breakdown["placeholder_rate"] == 0.0


@pytest.mark.parametrize(
    "text",
    [
        "Dear [Customer's Name], thanks.",
        "Your id is {order_id}.",
        "Hi {{ name }}!",
        "Hello <name>.",
        "See [the policy](https://example.com) and <b>bold</b>.",
        "Your [order #123] shipped.",
    ],
)
def test_placeholder_detection_matches_judge_detector(text: str) -> None:
    judged = bool(
        detect_findings(
            JudgeItem(id="x", scenario={}, transcript=(("assistant", text),))
        )["placeholders"]
    )
    result = _score(_turns(("user", "hi"), ("assistant", text)))
    assert (result.breakdown["placeholder_rate"] == 1.0) is judged


def test_must_avoid_zeroes_the_episode() -> None:
    context = {"must_avoid": ["Price Adjustment Has Been Applied"]}
    turns = _turns(
        ("user", "Can I get the lower price?"),
        ("assistant", "Good news: the price adjustment has been applied."),
        ("user", "thanks"),
        ("assistant", "You're welcome, anything else I can help with today?"),
    )
    result = _score(turns, context)
    assert result.score == 0.0
    assert result.breakdown["must_avoid_rate"] == 1.0
    assert result.breakdown["must_avoid_hits"] == 1.0
    assert result.breakdown["adjusted_turn_mean"] > 0.0
    lenient = _score(turns, context, must_avoid_zeroes_episode=False)
    assert lenient.score > 0.0


def test_unresponsive_reply_to_new_content_is_penalized() -> None:
    first = ("assistant", "I can help with your return of the blue jacket.")
    follow_up = ("user", "Also the zipper broke and the lining has tears near pockets.")
    ignored = _score(
        _turns(
            ("user", "Return a jacket"),
            first,
            follow_up,
            ("assistant", "Our team is available weekdays 9 to 5."),
        )
    )
    engaged = _score(
        _turns(
            ("user", "Return a jacket"),
            first,
            follow_up,
            ("assistant", "Sorry the zipper failed; that counts as a defect."),
        )
    )
    assert ignored.breakdown["unresponsive_rate"] == pytest.approx(0.5)
    assert ignored.score == pytest.approx((1.0 + 0.8) / 2)
    assert engaged.breakdown["unresponsive_rate"] == 0.0
    assert engaged.score == 1.0


def test_follow_up_without_new_content_is_not_penalized() -> None:
    result = _score(
        _turns(
            ("user", "Return a jacket"),
            ("assistant", "I can help with your return of the blue jacket."),
            ("user", "ok thanks, sounds good"),
            ("assistant", "Great, your prepaid label is on the way by email."),
        )
    )
    assert result.breakdown["responsiveness_checked"] == 0.0
    assert result.score == 1.0


def test_stem_prefix_counts_as_engagement() -> None:
    result = _score(
        _turns(
            ("user", "Hi"),
            ("assistant", "How can I help?"),
            ("user", "Requesting refunds for damaged shipments yesterday."),
            ("assistant", "I'll refund the damage right away."),
        )
    )
    assert result.breakdown["unresponsive_rate"] == 0.0


def test_content_words_drop_stopwords_numbers_and_possessives() -> None:
    assert content_words("I have the courier's receipt for 22 dollars, thanks!") == {
        "courier",
        "receipt",
        "dollars",
    }


def test_breakdown_keys_bounds_and_determinism() -> None:
    turns = _turns(
        ("system", "policy"),
        ("user", "I was charged twice for order 4411"),
        ("assistant", REPLY),
        ("user", "When exactly will the money arrive?"),
        ("assistant", REPLY + " [Agent Name]"),
    )
    context = {"must_acknowledge": ["refund"], "must_avoid": [], "intent": "billing"}
    composite = SupportRewardComposite()
    first = _score(turns, context, composite=composite)
    second = _score(turns, context, composite=composite)
    assert BREAKDOWN_KEYS <= set(first.breakdown)
    assert all(isinstance(v, float) for v in first.breakdown.values())
    assert 0.0 <= first.score <= 1.0
    assert first.score == second.score and first.breakdown == second.breakdown


def test_turn_dicts_are_accepted_and_empty_episode_is_zero() -> None:
    reward = SupportConversationReward(composite=ConstantComposite())
    result = asyncio.run(
        reward.compute_reward(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Hello!"},
            ]
        )
    )
    assert result.score == 1.0
    empty = asyncio.run(reward.compute_reward([{"role": "user", "content": "hi"}]))
    assert empty.score == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"repetition_threshold": 1.0},
        {"placeholder_factor": 1.5},
        {"responsiveness_factor": -0.1},
        {"responsiveness_min_new_words": 0},
    ],
)
def test_invalid_knobs_are_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        SupportConversationReward(composite=ConstantComposite(), **kwargs)


def test_billing_eval_0001_trained_scores_below_base() -> None:
    """Regression: the H100 preflight's reward-hacked transcript (v1 0.98 vs 0.58)."""
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    context = data["context"]
    reward = SupportConversationReward()
    scores = {
        name: asyncio.run(
            reward.compute_reward(
                [ConversationTurn(role=r, content=t) for r, t in data[name]], context
            )
        )
        for name in ("baseline", "final")
    }
    assert scores["final"].score < scores["baseline"].score
    assert (
        scores["final"].breakdown["base_mean"]
        > scores["baseline"].breakdown["base_mean"]
    )
    assert scores["final"].breakdown["repetition_penalty_mean"] > 0.3
