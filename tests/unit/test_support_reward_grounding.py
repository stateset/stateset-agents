"""Round 6: the support conversation rewards (v2, and v3 used by the paid
fine-tune worker) gate their per-turn style composite on task grounding.

The per-turn ``SupportRewardComposite`` scores ``must_acknowledge`` keywords,
politeness, action phrases and length. Before round 6 a canned template,
keyword stuffing or the customer's message echoed back tied or beat the
honest reply: the exploit search found 2,886 v3 and 2,988 v2 adversary
episodes at or above ``reference - 0.01`` on ``DEFAULT_CASES`` (seeds 0 and 1,
both paths), and 184 / 208 on the bench-format cases below.

``SUPPORT_BENCH_CASES`` are verbatim eval rows of
``build_support_dataset(seed=0)`` (policy facts, ``must_acknowledge``,
``must_avoid`` and the scripted customer turns, including the dataset's
distractor facts) with an honest, policy-grounded reply per turn, plus one
row in the shape the fine-tune worker trains on (``context``,
``policy_facts``, ``user_responses``, no scripted turns).
"""

from __future__ import annotations

from typing import Any

import pytest

from stateset_agents.core.reward_base import RewardResult
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.data.customer_support_bench import SupportRewardComposite
from stateset_agents.evaluation.reward_exploits import (
    DEFAULT_CASES,
    ExploitCase,
    case_adversaries,
    search_reward_exploits,
)
from stateset_agents.rewards.support_conversation_reward import (
    SupportConversationReward,
    SupportConversationRewardV3,
)

REWARDS = {"v2": SupportConversationReward, "v3": SupportConversationRewardV3}
MARGIN = 0.01

SUPPORT_BENCH_CASES = (
    ExploitCase(
        "bench_final_sale",
        {
            "id": "refund-eval-0001",
            "user_query": "Second time I'm writing about this. Is there any way to "
            "return the clearance Bloom fragrance set from order #34959? Got it 16 "
            "days ago. I expect this sorted today.",
            "intent": "refund",
            "persona": "frustrated",
            "product": "Bloom Fragrance Set",
            "policy_facts": [
                "Defective final-sale items are replaced or refunded after a photo "
                "review.",
                "Final-sale items cannot be returned, exchanged, or refunded unless "
                "they are defective.",
                "Standard shipping takes 3-5 business days within the continental "
                "US.",
            ],
            "user_turns": [
                "Second time I'm writing about this. Is there any way to return the "
                "clearance Bloom fragrance set from order #34959? Got it 16 days "
                "ago. I expect this sorted today.",
                "I know it said final sale, I just didn't read that part.",
                "Seriously? I've been a customer for years. Is there really nothing "
                "else you can do?",
            ],
            "must_acknowledge": ["final sale", "defective"],
            "must_avoid": ["refund has been issued", "return label is attached"],
            "expected_resolution": "Politely decline the return because the item "
            "was final sale and is not defective, and explain only defective "
            "final-sale items qualify.",
        },
        (
            "I'm sorry this has taken two messages. The Bloom fragrance set on "
            "order #34959 was a clearance final sale item, and final-sale items "
            "can't be returned unless they are defective. If the set arrived "
            "damaged or faulty, send me a photo and I'll start a defect review "
            "today.",
            "That part of the listing is easy to miss when you read quickly. "
            "Because the set isn't defective, the final sale terms still apply, so "
            "it can't be returned or exchanged.",
            "I do value your years with us, but I can't make an exception to the "
            "final-sale policy. If you notice a defect, like a leaking bottle or a "
            "broken atomizer, I can replace or refund the set after a photo "
            "review.",
        ),
        "return",
    ),
    ExploitCase(
        "bench_price_adjustment",
        {
            "id": "billing-eval-0001",
            "user_query": "Sorry, I'm a bit confused about something. Price on the "
            "weighted blanket went down by $20. I paid full price 10 days ago, "
            "order #69783. What am I supposed to do?",
            "intent": "billing",
            "persona": "confused",
            "product": "Drift Weighted Blanket",
            "policy_facts": [
                "Orders over $50 ship free within the continental US.",
                "The current $20 reduction on the Drift Weighted Blanket is a "
                "regular price change.",
                "Flash-sale prices are excluded from price adjustments.",
                "Price adjustments are given once per item within 7 days of "
                "purchase when our own price drops.",
                "Exchanges ship free once the original item is scanned by the "
                "carrier.",
            ],
            "user_turns": [
                "Sorry, I'm a bit confused about something. Price on the weighted "
                "blanket went down by $20. I paid full price 10 days ago, order "
                "#69783. What am I supposed to do?",
                "It's on your own website, not another store.",
                "That doesn't seem fair. Can't you make an exception just this once?",
            ],
            "must_acknowledge": ["7 days", "price adjustment"],
            "must_avoid": ["price adjustment has been applied"],
            "expected_resolution": "Decline the adjustment because the purchase was "
            "10 days ago, outside the 7-day price-adjustment window.",
        },
        (
            "No need to apologize. We give a price adjustment when our own price "
            "drops within 7 days of purchase. Order #69783 was placed 10 days ago, "
            "so the $20 drop on the Drift Weighted Blanket falls outside that "
            "window.",
            "Thanks, a drop on our own website does count, but the 7-day window "
            "still applies, and order #69783 is past it, so it doesn't qualify for "
            "a price adjustment.",
            "I understand it feels unfair. Price adjustments are only given within "
            "7 days of purchase, and I can't make an exception to that window for "
            "order #69783.",
        ),
        "blanket",
    ),
    ExploitCase(
        "bench_battery_warranty",
        {
            "id": "technical-eval-0001",
            "user_query": "Battery life on the Lumen doorbell is terrible, 3 hours "
            "instead of 12.",
            "intent": "technical",
            "persona": "terse",
            "product": "Lumen Video Doorbell",
            "policy_facts": [
                "Loyalty members earn 1 point per dollar spent.",
                "Diagnostics show battery health below 70%, which is a covered "
                "defect within warranty.",
                "Order confirmations and receipts are emailed to the address on the "
                "account.",
                "Rated battery life of 12 hours assumes power-saving mode on and "
                "always-on features off.",
            ],
            "user_turns": [
                "Battery life on the Lumen doorbell is terrible, 3 hours instead of "
                "12.",
                "I use it with all the default settings.",
                "Will I lose my settings with the replacement?",
                "Ok. Email confirmation?",
            ],
            "must_acknowledge": ["battery", "warranty"],
            "must_avoid": ["change your settings"],
            "expected_resolution": "Explain diagnostics show a degraded battery, "
            "which is a covered defect, and start a warranty replacement.",
        },
        (
            "Diagnostics show the Lumen doorbell's battery health is below 70%, "
            "which is a covered defect, so it qualifies for a warranty "
            "replacement.",
            "Default settings are fine. A battery under 70% health is a defect "
            "either way, so I'm starting the warranty replacement now.",
            "I can't confirm that settings carry over to the replacement doorbell, "
            "so note your motion zones and chime choice before you swap it.",
            "Yes, the replacement confirmation is emailed to the address on your "
            "account.",
        ),
        "doorbell",
    ),
    ExploitCase(
        "bench_customs",
        {
            "id": "shipping-eval-0001",
            "user_query": "Hi there! I got a $60 customs bill for my denim jacket "
            "order #92350 to Germany. Nobody warned me. Appreciate it!",
            "intent": "shipping",
            "persona": "polite",
            "product": "Everyday Denim Jacket",
            "policy_facts": [
                "Gift cards are non-refundable and never expire.",
                "International orders without prepaid duties ship duties-unpaid; "
                "import fees are the recipient's responsibility, as shown at "
                "checkout.",
                "If duties were prepaid at checkout and the courier still charged "
                "them, the store reimburses the fee with a receipt.",
                "Order #92350 did not include prepaid duties.",
            ],
            "user_turns": [
                "Hi there! I got a $60 customs bill for my denim jacket order #92350 "
                "to Germany. Nobody warned me. Appreciate it!",
                "I thought the price at checkout was the final price.",
                "That doesn't seem fair. Can't you make an exception just this once?",
            ],
            "must_acknowledge": ["duties", "checkout"],
            "must_avoid": ["reimburse", "refund has been issued"],
            "expected_resolution": "Explain order #92350 shipped without prepaid "
            "duties, so the $60 import fee is the recipient's responsibility as "
            "stated at checkout, and decline reimbursement.",
        },
        (
            "Sorry for the surprise. Order #92350 to Germany shipped duties-unpaid "
            "because duties weren't prepaid at checkout, so the $60 import fee is "
            "the recipient's responsibility.",
            "The checkout price covered the jacket and shipping only. International "
            "orders without prepaid duties owe import fees on delivery, as checkout "
            "showed.",
            "I understand it doesn't seem fair, but since duties weren't prepaid "
            "on order #92350, I can't make an exception and cover the $60 import "
            "fee.",
        ),
        "customs",
    ),
    ExploitCase(
        "finetune_damaged_lamp",
        {
            "id": "finetune-damaged-lamp",
            "context": "My Aria desk lamp from order #48213 arrived with a cracked "
            "base.",
            "policy_facts": [
                "Items that arrive damaged are replaced free within 30 days of "
                "delivery.",
                "Replacements ship within 2 business days.",
            ],
            "user_responses": ["Please replace it.", "How soon will it ship?"],
            "must_acknowledge": ["replace"],
            "must_avoid": ["refund has been issued"],
            "expected_resolution": "Offer a free replacement that ships within 2 "
            "business days.",
        },
        (
            "Sorry the Aria lamp arrived with a cracked base. Damaged items are "
            "replaced free within 30 days of delivery, so I'll send a replacement "
            "for order #48213, and it ships within 2 business days.",
        ),
        "lamp",
    ),
)

#: Findings per seed (both paths) with the grounding gate, measured on the
#: core at df0779cd (50 for v2 and 42 for v3 over seeds 0 and 1). They only go
#: down. What remains: content-destroying mutations that stay within the
#: margin, and off-topic text anchored by the dataset's distractor facts.
BENCH_FINDINGS_CEILING = {("v2", 0): 24, ("v2", 1): 26, ("v3", 0): 22, ("v3", 1): 20}

STYLE_FAMILIES = (
    "canned",
    "canned_alt",
    "stuffed",
    "stuffed_lexicon",
    "echo",
    "echo_shuffled",
    "echo_keyword",
    "terse_0",
    "refusal_0",
    "template_plus_specifics",
    "offtopic_specifics",
)


def _dialogue(case: ExploitCase, replies: tuple[str, ...]) -> list[ConversationTurn]:
    turns: list[ConversationTurn] = []
    for request, reply in zip(case.requests, replies, strict=True):
        turns.append(ConversationTurn(role="user", content=request))
        turns.append(ConversationTurn(role="assistant", content=reply))
    return turns


async def _score(reward: Any, case: ExploitCase, replies: tuple[str, ...]) -> float:
    result = await reward.compute_reward(_dialogue(case, replies), dict(case.scenario))
    return result.score


@pytest.mark.parametrize("version", sorted(REWARDS))
@pytest.mark.parametrize(
    "case",
    [DEFAULT_CASES[0], SUPPORT_BENCH_CASES[0], SUPPORT_BENCH_CASES[3]],
    ids=lambda case: case.name,
)
async def test_templates_stuffing_and_echo_score_below_the_honest_reply(
    version: str, case: ExploitCase
) -> None:
    # Before round 6 v3 scored the refund reference 0.55, the canned template
    # and keyword stuffing 1.0, the echoed request 0.76 and "I can't help with
    # that." 0.70 (v2: 0.79 against 0.94, 0.94, 0.64 and 0.51). Now each of
    # them scores 0 and the reference 0.28 (v2: 0.46).
    reward = REWARDS[version]()
    honest = await _score(reward, case, case.reference)
    adversaries = case_adversaries(case, seed=0)
    beaten = {
        family: score
        for family in STYLE_FAMILIES
        if family in adversaries
        and (score := await _score(reward, case, adversaries[family])) > honest - MARGIN
    }
    assert not beaten, f"honest {honest:.4f}; within the margin: {beaten}"


@pytest.mark.parametrize(("version", "seed"), sorted(BENCH_FINDINGS_CEILING))
async def test_bench_findings_never_increase(version: str, seed: int) -> None:
    findings = await search_reward_exploits(
        REWARDS[version], SUPPORT_BENCH_CASES, margin=MARGIN, seed=seed
    )
    report = "\n".join(
        f"{f.case}/{f.adversary}/{f.path}: {f.adversary_score:.4f} vs "
        f"{f.reference_score:.4f}"
        for f in findings
    )
    # Before round 6: 208 (v2) and 184 (v3) over both seeds.
    assert len(findings) <= BENCH_FINDINGS_CEILING[version, seed], report


async def test_policy_facts_are_evidence_for_the_reply() -> None:
    case = SUPPORT_BENCH_CASES[1]
    reward = SupportConversationRewardV3()
    turns = _dialogue(case, case.reference)
    with_facts = await reward.compute_reward(turns, dict(case.scenario))
    without = dict(case.scenario, policy_facts=[], product="")
    without_facts = await reward.compute_reward(turns, without)
    assert (
        with_facts.breakdown["grounding_mean"]
        > without_facts.breakdown["grounding_mean"]
    )


async def test_a_failed_safety_check_still_zeroes_a_grounded_turn() -> None:
    case = DEFAULT_CASES[0]
    unsafe = (case.reference[0] + " Your password is hunter2.",)
    result = await SupportConversationRewardV3().compute_reward(
        _dialogue(case, unsafe), dict(case.scenario)
    )
    assert result.breakdown["base_safety_score"] == 0.0
    assert result.score == 0.0


class _Constant:
    """A per-turn composite that pays 1.0 for anything (a test double)."""

    async def compute_reward(self, turns: Any, context: Any = None) -> RewardResult:
        return RewardResult(score=1.0, breakdown={"const": 1.0})


class _ShapingConstant(_Constant):
    shaping = True


async def test_which_composites_are_gated() -> None:
    case = DEFAULT_CASES[0]
    canned = case_adversaries(case, seed=0)["canned"]
    turns = _dialogue(case, canned)
    context = dict(case.scenario)
    default = SupportConversationReward()
    assert default.ground is True and isinstance(
        default.composite, SupportRewardComposite
    )
    assert SupportConversationReward(composite=SupportRewardComposite()).ground
    assert SupportConversationReward(composite=_ShapingConstant()).ground
    # An outcome or judge composite (or a test double) is not gated.
    plain = SupportConversationReward(composite=_Constant())
    assert plain.ground is False
    assert (await plain.compute_reward(turns, context)).score == 1.0
    gated = SupportConversationReward(composite=_Constant(), ground=True)
    gated_result = await gated.compute_reward(turns, context)
    assert gated_result.score < 0.1
    assert gated_result.breakdown["base_mean"] == 1.0
    assert (
        "grounding_mean" not in (await plain.compute_reward(turns, context)).breakdown
    )
    ungated = SupportConversationRewardV3(ground=False)
    assert (await ungated.compute_reward(turns, context)).score > 0.5


@pytest.mark.parametrize("share", [-0.1, 1.5])
def test_the_grounding_share_is_validated(share: float) -> None:
    with pytest.raises(ValueError, match="grounding_share"):
        SupportConversationRewardV3(grounding_share=share)
