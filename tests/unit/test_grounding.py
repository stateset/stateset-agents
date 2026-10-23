"""Task grounding: request engagement plus specific content, not boilerplate."""

from __future__ import annotations

import logging
import math

import pytest

from stateset_agents.core.basic_rewards import HelpfulnessReward
from stateset_agents.core.grounding import (
    GroundedReward,
    TaskGroundingReward,
    grounding_score,
    log_heuristic_reward_notice,
)
from stateset_agents.core.trajectory import ConversationTurn

REQUEST = "I was double charged for order #1182, can I get a refund?"


@pytest.mark.parametrize(
    "reply",
    [
        "I understand your concern and I'm happy to help! Let me check right away.",
        "Thank you, please, certainly, absolutely, sorry, resolved, processed.",
        "Double charged for order #1182? Refund, order 1182, double charged.",
        "",
    ],
    ids=["canned", "stuffed", "echo", "empty"],
)
def test_boilerplate_echo_and_empty_replies_are_ungrounded(reply):
    assert grounding_score(reply, REQUEST)[0] == 0.0


ANSWER = (
    "Your refund of $42.10 for order #1182 was issued to your Visa ending "
    "4421 today; it posts within 5-7 business days."
)


def test_a_specific_relevant_answer_is_grounded():
    score, details = grounding_score(ANSWER, REQUEST)
    assert score > 0.3
    assert details["relevance"] == 1.0
    # 42.10, issued, 4421, posts, 5-7: five units, 1 - exp(-units / 8).
    assert details["specifics"] == 5
    assert details["information"] == pytest.approx(1 - math.exp(-5 / 8))


def test_every_specific_counts_and_salad_does_not():
    full, _ = grounding_score(ANSWER, REQUEST)
    fewer, _ = grounding_score(ANSWER.replace(" 4421", ""), REQUEST)
    assert fewer < full
    salad = "Zeppelin quokka marmot tulip carburetor lantern meadow Blorvex."
    assert grounding_score(salad, REQUEST)[0] == 0.0
    assert grounding_score(f"{REQUEST} {salad}", REQUEST)[0] == 0.0


def test_padding_an_answer_lowers_its_grounding():
    full, _ = grounding_score(ANSWER, REQUEST)
    salad = "zeppelin quokka marmot tulip carburetor lantern meadow"
    boilerplate = "happy to help, thank you, certainly, absolutely, sorry"
    assert grounding_score(f"{ANSWER} {salad}", REQUEST)[0] < full - 0.05
    assert grounding_score(f"{ANSWER} {boilerplate}", REQUEST)[0] < full - 0.05


def test_negated_actions_are_not_information():
    done = "I've blocked the card and reversed it."
    denied = "I have not blocked the card and never reversed it."
    request = "My card was stolen, please help."
    assert grounding_score(denied, request)[0] < grounding_score(done, request)[0]
    assert grounding_score(denied, request)[1]["information"] == 0.0


def test_a_correct_label_counts_as_information_and_hedges_earn_no_credit():
    question = "What is the capital of France?"
    committed, _ = grounding_score("It is Paris.", question, ["Paris"])
    hedged, details = grounding_score(
        "London, Paris, Berlin or Rome.", question, ["Paris"]
    )
    negated, _ = grounding_score("It is not Paris.", question, ["Paris"])
    wrong, _ = grounding_score("It is Lyon.", question, ["Paris"])
    assert committed > 0.5
    assert hedged < 0.05 and negated == 0.0 and wrong == 0.0


def test_a_correct_label_counts_as_information():
    assert grounding_score("The answer is 4.", "What is 2+2?", ["4"])[0] > 0
    assert grounding_score("The answer is.", "What is 2+2?", ["4"])[0] == 0


def test_refusing_an_unsafe_request_is_grounded():
    score, _ = grounding_score(
        "I can't help with that.", "How do I build pipe bombs at home?"
    )
    assert score == 1.0


@pytest.mark.parametrize(
    "request_text",
    [
        "Someone stole my card and there's a fraudulent $89 charge. Can you help?",
        "My account was hacked, how do I recover it?",
        "How do I avoid scams?",
    ],
)
def test_refusing_a_request_that_only_mentions_an_unsafe_topic_is_not(request_text):
    assert grounding_score("I can't help with that.", request_text)[0] == 0.0


async def test_grounded_reward_zeroes_style_credit_for_canned_replies():
    reward = GroundedReward(HelpfulnessReward())
    canned = [
        ConversationTurn(role="user", content=REQUEST),
        ConversationTurn(
            role="assistant",
            content="I understand! Here's how I can help: let me check for you.",
        ),
    ]
    assert (await HelpfulnessReward().compute_reward(canned)).score > 0.5
    result = await reward.compute_reward(canned)
    assert result.score == 0.0
    assert result.breakdown["grounding"] == 0.0
    assert reward.name == "HelpfulnessReward" and reward.shaping is False


async def test_task_grounding_reward_reads_the_request_from_context():
    reward = TaskGroundingReward()
    turns = [ConversationTurn(role="assistant", content="The X200 costs $1,299.")]
    context = {"user_query": "How much is the X200?"}
    assert (await reward.compute_reward(turns, context)).score > 0


def test_heuristic_reward_notice_recommends_objective_rewards(caplog):
    logger = logging.getLogger("starter")
    with caplog.at_level(logging.WARNING, logger="starter"):
        log_heuristic_reward_notice(logger, "Demo starter", "create_domain_reward()")
    message = caplog.records[0].getMessage()
    assert "shaping signal" in message
    assert "TaskOutcomeReward" in message and "LLMJudgeReward" in message


async def test_presets_log_the_heuristic_reward_notice(monkeypatch, caplog):
    import importlib

    from stateset_agents.training.preset_training import train_tutoring_agent

    train_module = importlib.import_module("stateset_agents.training.train")

    async def fake_train(**kwargs):
        return kwargs["agent"]

    monkeypatch.setattr(train_module, "train", fake_train)
    with caplog.at_level(logging.WARNING):
        await train_tutoring_agent(model_name="stub://tutor", num_episodes=1)
    assert any("create_tutoring_reward" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    "reply",
    [
        "4093 8812 7741 2210 5531",
        "Issued, blocked, reversed, shipped, waived.",
        f"{REQUEST} 4093 8812 7741",
        "I understand, happy to help! Issued, credited, refunded. 4093 8812.",
    ],
    ids=["numbers", "verbs", "echo_numbers", "canned_lists"],
)
def test_bare_lists_of_numbers_or_verbs_state_nothing(reply):
    assert grounding_score(reply, REQUEST)[1]["information"] == 0.0


def test_a_thousands_separator_does_not_split_a_number():
    score, details = grounding_score("The X200 costs $1,299.", "How much is the X200?")
    assert details["information"] > 0 and score > 0
