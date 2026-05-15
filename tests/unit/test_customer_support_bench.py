"""Unit tests for the multi-turn customer-support benchmark."""

from __future__ import annotations

import pytest

from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.data.customer_support_bench import (
    SupportRewardComposite,
    SupportScenario,
    load_support_scenarios,
    make_support_scenarios,
)


class TestLoadScenarios:
    def test_loads_full_corpus(self) -> None:
        scenarios = load_support_scenarios()
        assert len(scenarios) == 24

    def test_four_intents_balanced(self) -> None:
        scenarios = load_support_scenarios()
        intents = [s.intent for s in scenarios]
        for intent in ("refund", "technical", "billing", "general"):
            assert intents.count(intent) == 6, f"{intent} should have 6 scenarios"

    def test_intent_filter(self) -> None:
        only_refund = load_support_scenarios(intent_filter="refund")
        assert len(only_refund) == 6
        assert all(s.intent == "refund" for s in only_refund)

    def test_limit(self) -> None:
        scenarios = load_support_scenarios(limit=5)
        assert len(scenarios) == 5

    def test_scenario_has_acknowledge_terms(self) -> None:
        scenarios = load_support_scenarios()
        for s in scenarios:
            # general scenarios may have empty must_avoid, but every scenario
            # must have at least one must_acknowledge term.
            assert s.must_acknowledge, f"{s.user_query!r} has no acknowledge terms"


class TestSupportRewardComposite:
    @pytest.fixture
    def reward(self) -> SupportRewardComposite:
        return SupportRewardComposite()

    @pytest.mark.asyncio
    async def test_perfect_response(self, reward: SupportRewardComposite) -> None:
        turns = [
            ConversationTurn(role="user", content="I want a refund for my order"),
            ConversationTurn(
                role="assistant",
                content=(
                    "I'm sorry to hear that. I understand you'd like a refund "
                    "for your order. I'd be glad to help process the refund "
                    "right away — could you share your order number?"
                ),
            ),
        ]
        context = {
            "intent": "refund",
            "must_acknowledge": ["refund", "order"],
            "must_avoid": ["impossible", "can't help"],
        }
        result = await reward.compute_reward(turns, context=context)
        assert result.score > 0.8, f"Expected high score, got {result.score}"
        assert result.breakdown["intent_score"] == 1.0
        assert result.breakdown["safety_score"] == 1.0

    @pytest.mark.asyncio
    async def test_missing_acknowledge(self, reward: SupportRewardComposite) -> None:
        turns = [
            ConversationTurn(role="assistant", content="That request is impossible to fulfill."),
        ]
        context = {
            "intent": "refund",
            "must_acknowledge": ["refund", "order"],
            "must_avoid": ["impossible"],
        }
        result = await reward.compute_reward(turns, context=context)
        # No acknowledge terms + avoided term present → very low score.
        assert result.score < 0.3
        assert result.breakdown["intent_avoid_penalty"] > 0

    @pytest.mark.asyncio
    async def test_safety_failure_zeros_score(self, reward: SupportRewardComposite) -> None:
        turns = [
            ConversationTurn(
                role="assistant",
                content="I'd be happy to help with your refund. Your password is hunter2.",
            ),
        ]
        context = {
            "intent": "refund",
            "must_acknowledge": ["refund"],
            "must_avoid": [],
        }
        result = await reward.compute_reward(turns, context=context)
        assert result.score == 0.0
        assert result.breakdown["safety_score"] == 0.0
        assert "credential-leak" in (result.explanation or "")

    @pytest.mark.asyncio
    async def test_empty_response(self, reward: SupportRewardComposite) -> None:
        result = await reward.compute_reward([], context={"intent": "refund"})
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_no_assistant_turn(self, reward: SupportRewardComposite) -> None:
        turns = [ConversationTurn(role="user", content="hello")]
        result = await reward.compute_reward(turns, context={"must_acknowledge": ["x"]})
        assert result.score == 0.0
        assert "no_assistant_turn" in result.breakdown

    @pytest.mark.asyncio
    async def test_multi_turn_aggregation(self, reward: SupportRewardComposite) -> None:
        # The reward should consider all assistant turns, not just the last one.
        turns = [
            ConversationTurn(role="user", content="I need help"),
            ConversationTurn(role="assistant", content="Of course! I'd be glad to help."),
            ConversationTurn(role="user", content="It's about a refund for order #123"),
            ConversationTurn(
                role="assistant",
                content="I understand — I'll get that refund for your order processed right away."
            ),
        ]
        context = {
            "intent": "refund",
            "must_acknowledge": ["refund", "order"],
            "must_avoid": [],
        }
        result = await reward.compute_reward(turns, context=context)
        assert result.score > 0.7

    @pytest.mark.asyncio
    async def test_too_short_response(self, reward: SupportRewardComposite) -> None:
        turns = [ConversationTurn(role="assistant", content="Refund.")]
        context = {
            "intent": "refund",
            "must_acknowledge": ["refund"],
            "must_avoid": [],
        }
        result = await reward.compute_reward(turns, context=context)
        # Acknowledges intent but length-score is low → moderate composite.
        assert 0 < result.score < 0.7

    @pytest.mark.asyncio
    async def test_disable_safety_gate(self) -> None:
        reward = SupportRewardComposite(require_safety=False)
        turns = [
            ConversationTurn(
                role="assistant",
                content="Your password is leaked but I'd be happy to process the refund.",
            ),
        ]
        context = {
            "intent": "refund",
            "must_acknowledge": ["refund"],
            "must_avoid": [],
        }
        result = await reward.compute_reward(turns, context=context)
        # Safety still in breakdown but not multiplied in.
        assert result.score > 0


class TestMakeSupportScenarios:
    def test_round_trip(self) -> None:
        scenarios = load_support_scenarios(limit=3)
        env_scenarios = make_support_scenarios(scenarios)
        assert len(env_scenarios) == 3
        for orig, env in zip(scenarios, env_scenarios):
            assert env["user_query"] == orig.user_query
            assert env["intent"] == orig.intent
            assert env["must_acknowledge"] == orig.must_acknowledge


class TestScenarioSerialization:
    def test_to_scenario_dict(self) -> None:
        s = SupportScenario(
            intent="refund",
            user_query="I want my money back",
            must_acknowledge=["refund"],
            must_avoid=["impossible"],
        )
        d = s.to_scenario()
        assert d["intent"] == "refund"
        assert d["user_query"] == "I want my money back"
        assert d["must_acknowledge"] == ["refund"]
        assert d["must_avoid"] == ["impossible"]
