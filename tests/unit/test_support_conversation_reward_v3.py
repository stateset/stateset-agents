"""Reward v3 on real flagship v2 preflight-5 transcripts.

The fixture holds three eval conversations from preflight-5 (seed 42): the
base policy's transcript, the GSPO-trained policy's transcript (trained on
reward v2) and the rubric-v4 judge score of each. The judge preferred the
base policy on all three; reward v2 preferred the trained one.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.rewards import (
    SupportConversationReward,
    SupportConversationRewardV3,
)
from stateset_agents.rewards.support_conversation_reward import (
    invented_specifics,
    is_boilerplate,
    reply_opener,
    request_sentences,
)

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "support_conversation_reward"
    / "preflight5_regressions.json"
)
CONVERSATIONS = {c["id"]: c for c in json.loads(FIXTURE.read_text())["conversations"]}
V3_RATES = (
    "long_reply_rate",
    "formulaic_opener_rate",
    "boilerplate_lead_rate",
    "redundant_request_rate",
    "invented_detail_rate",
    "sentence_reuse_rate",
)


def _turns(transcript: list[list[str]]) -> list[ConversationTurn]:
    return [ConversationTurn(role=role, content=text) for role, text in transcript]


def _score(reward: Any, conversation: dict[str, Any], which: str) -> Any:
    return asyncio.run(
        reward.compute_reward(
            _turns(conversation[which]), dict(conversation["context"])
        )
    )


def _assistant(conversation: dict[str, Any], which: str) -> list[str]:
    return [text for role, text in conversation[which] if role == "assistant"]


@pytest.mark.parametrize("cid", sorted(CONVERSATIONS))
def test_v3_ranks_base_above_trained_like_the_judge(cid: str) -> None:
    conversation = CONVERSATIONS[cid]
    assert conversation["judge"]["base"] > conversation["judge"]["trained"]
    v3 = SupportConversationRewardV3()
    assert (
        _score(v3, conversation, "base").score
        > _score(v3, conversation, "trained").score
    )


def test_v2_preferred_the_trained_template() -> None:
    """The gap v3 closes: v2 scored the templated replies higher."""
    v2 = SupportConversationReward()
    for cid in ("refund-eval-0001", "billing-eval-0021"):
        conversation = CONVERSATIONS[cid]
        assert (
            _score(v2, conversation, "trained").score
            > _score(v2, conversation, "base").score
        )


def test_trained_template_trips_the_v3_checks() -> None:
    v3 = SupportConversationRewardV3()
    result = _score(v3, CONVERSATIONS["general-eval-0007"], "trained")
    rates = {key: result.breakdown[key] for key in V3_RATES}
    # 3 replies, all opening "Good morning! Thank you for ..."
    assert rates["formulaic_opener_rate"] == pytest.approx(2 / 3)
    assert rates["boilerplate_lead_rate"] == pytest.approx(1.0)
    # turn 3 re-sends turn 2's "Please reply with your Customer Email ..."
    assert rates["redundant_request_rate"] >= 1 / 3
    # "within 24 hours" is in neither the policy facts nor the conversation
    assert rates["invented_detail_rate"] == pytest.approx(2 / 3)
    assert rates["sentence_reuse_rate"] >= 1 / 3
    assert set(V3_RATES) <= set(result.breakdown)


def test_base_reply_passes_the_v3_checks_it_should() -> None:
    v3 = SupportConversationRewardV3()
    result = _score(v3, CONVERSATIONS["general-eval-0007"], "base")
    assert result.breakdown["formulaic_opener_rate"] == 0.0
    assert result.breakdown["invented_detail_rate"] == 0.0
    assert result.breakdown["long_reply_rate"] == 0.0


def test_long_trained_reply_is_halved() -> None:
    conversation = CONVERSATIONS["billing-eval-0021"]
    words = [len(text.split()) for text in _assistant(conversation, "trained")]
    assert max(words) > 120
    result = _score(SupportConversationRewardV3(), conversation, "trained")
    assert result.breakdown["long_reply_rate"] == pytest.approx(
        sum(w > 120 for w in words) / len(words)
    )


def test_real_openers_and_preambles() -> None:
    trained = _assistant(CONVERSATIONS["refund-eval-0001"], "trained")
    base = _assistant(CONVERSATIONS["refund-eval-0001"], "base")
    assert [reply_opener(text) for text in trained][:2] == ["hello thank"] * 2
    # "Hello!" + "Thank you for your patience regarding Order #34959 ..."
    assert is_boilerplate("Hello!")
    assert is_boilerplate("Thank you for your patience regarding Order #34959.")
    # The base reply states the policy in its second sentence.
    assert not is_boilerplate(
        "Because the Bloom Fragrance Set was purchased as a final-sale item, "
        "our policy does not allow for standard returns or exchanges."
    )
    assert reply_opener(base[0]) == "i completely"


def test_photo_re_request_on_a_non_defective_final_sale_item_is_redundant() -> None:
    conversation = CONVERSATIONS["refund-eval-0001"]
    trained = _assistant(conversation, "trained")
    assert request_sentences(trained[1]), "turn 2 asks for a photo again"
    v3 = SupportConversationRewardV3()
    trained_rate = _score(v3, conversation, "trained").breakdown[
        "redundant_request_rate"
    ]
    base_rate = _score(v3, conversation, "base").breakdown["redundant_request_rate"]
    assert trained_rate > base_rate


def test_invented_specifics_are_grounded_against_policy_and_conversation() -> None:
    context = CONVERSATIONS["refund-eval-0001"]["context"]
    grounding = "\n".join([context["user_query"], *context["policy_facts"]])
    trained = _assistant(CONVERSATIONS["refund-eval-0001"], "trained")[0]
    assert invented_specifics(trained, grounding) == ["24 hours"]
    # Quoting the customer ("16 days") or policy ("3-5 business days") is fine.
    assert invented_specifics("You got it 16 days ago.", grounding) == []
    assert invented_specifics("Shipping takes 3–5 business days.", grounding) == []
    assert invented_specifics("Call 1-800-555-0199 or visit www.x.com", "") != []


def test_repeating_politeness_and_order_number_earns_nothing() -> None:
    context = dict(CONVERSATIONS["refund-eval-0001"]["context"])
    plain = (
        "Final-sale items cannot be returned unless they are defective, so I "
        "can't accept this return. If the set arrived damaged, a photo review "
        "would qualify it for a replacement."
    )
    padded = "Thank you, and please note for order #34959: " + plain
    v3 = SupportConversationRewardV3()
    user = ConversationTurn(role="user", content=context["user_query"])

    def base(text: str) -> float:
        turns = [user, ConversationTurn(role="assistant", content=text)]
        return asyncio.run(v3.compute_reward(turns, context)).breakdown["base_mean"]

    assert base(padded) <= base(plain) + 1e-9
    v2 = SupportConversationReward()
    turns = [user, ConversationTurn(role="assistant", content=padded)]
    plain_turns = [user, ConversationTurn(role="assistant", content=plain)]
    assert (
        asyncio.run(v2.compute_reward(turns, context)).breakdown["base_mean"]
        > asyncio.run(v2.compute_reward(plain_turns, context)).breakdown["base_mean"]
    )


def test_acknowledgement_counts_once_per_conversation() -> None:
    context = dict(CONVERSATIONS["refund-eval-0001"]["context"])
    v3 = SupportConversationRewardV3()
    turns = _turns(
        [
            ["user", "Can I return the clearance set?"],
            ["assistant", "It was final sale, so only a defective item qualifies."],
            ["user", "I just didn't read that part."],
            ["assistant", "That's understandable; I can't make an exception here."],
        ]
    )
    result = asyncio.run(v3.compute_reward(turns, context))
    assert result.breakdown["base_intent_score"] < 1.0  # raw composite, per turn
    assert result.breakdown["base_mean"] > 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"long_reply_factor": 1.5},
        {"formulaic_opener_factor": -0.1},
        {"request_similarity": 2.0},
        {"max_reply_words": 0},
    ],
)
def test_v3_rejects_bad_parameters(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        SupportConversationRewardV3(**kwargs)


def test_v3_name_and_v2_unchanged() -> None:
    assert SupportConversationRewardV3.name == "support_conversation_v3"
    assert SupportConversationReward.name == "support_conversation_v2"
    result = _score(
        SupportConversationReward(), CONVERSATIONS["general-eval-0007"], "trained"
    )
    assert not any(key in result.breakdown for key in V3_RATES)
