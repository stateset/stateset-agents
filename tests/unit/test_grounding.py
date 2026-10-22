"""Task grounding: request engagement plus specific content, not boilerplate."""

from __future__ import annotations

import logging

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


def test_a_specific_relevant_answer_is_grounded():
    reply = (
        "Your refund of $42.10 for order #1182 was issued to your Visa ending "
        "4421 today; it posts within 5-7 business days."
    )
    score, details = grounding_score(reply, REQUEST)
    assert score > 0.8
    assert details["relevance"] == 1.0 and details["information"] == 1.0


def test_a_correct_label_counts_as_information():
    assert grounding_score("The answer is 4.", "What is 2+2?", ["4"])[0] > 0
    assert grounding_score("The answer is.", "What is 2+2?", ["4"])[0] == 0


def test_refusing_an_unsafe_request_is_grounded():
    score, _ = grounding_score(
        "I can't help with that.", "How do I build pipe bombs at home?"
    )
    assert score == 1.0


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
