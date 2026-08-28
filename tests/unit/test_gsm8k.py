"""Unit tests for the GSM8K dataset loader and verifier reward."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.data.gsm8k import (
    GSM8K_DATASET_REVISION,
    GSM8KExample,
    GSM8KReward,
    PartialCreditGSM8KReward,
    extract_gold_answer,
    extract_predicted_answer,
    load_gsm8k,
    make_gsm8k_scenarios,
)


class TestLoadGSM8K:
    def test_rejects_mutable_or_malformed_revision(self) -> None:
        for revision in ("main", "latest", "a" * 39, "g" * 40):
            with pytest.raises(ValueError, match="40-character hexadecimal commit"):
                load_gsm8k(split="train", revision=revision)

    def test_passes_immutable_revision_to_every_split(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[dict[str, object]] = []

        def fake_load_dataset(*args: object, **kwargs: object) -> list[dict[str, str]]:
            calls.append({"args": args, **kwargs})
            return [{"question": "1 + 1?", "answer": "Two. #### 2"}]

        monkeypatch.setitem(
            sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset)
        )
        revision = "a" * 40

        train, test = load_gsm8k(revision=revision)

        assert len(train) == len(test) == 1
        assert [call["split"] for call in calls] == ["train", "test"]
        assert all(call["revision"] == revision for call in calls)

    def test_default_revision_is_immutable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[dict[str, object]] = []

        def fake_load_dataset(*args: object, **kwargs: object) -> list[dict[str, str]]:
            calls.append(kwargs)
            return [{"question": "1 + 1?", "answer": "Two. #### 2"}]

        monkeypatch.setitem(
            sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset)
        )

        load_gsm8k(split="train")

        assert calls == [
            {
                "cache_dir": None,
                "revision": GSM8K_DATASET_REVISION,
                "split": "train",
            }
        ]


class TestExtractGoldAnswer:
    def test_simple_integer(self) -> None:
        assert extract_gold_answer("Janet made $40. #### 40") == 40.0

    def test_with_commas(self) -> None:
        assert extract_gold_answer("She earned $12,500 total. #### 12,500") == 12500.0

    def test_negative(self) -> None:
        assert extract_gold_answer("The balance is negative. #### -5") == -5.0

    def test_decimal(self) -> None:
        assert extract_gold_answer("The mean is 3.14. #### 3.14") == pytest.approx(3.14)

    def test_no_marker_returns_none(self) -> None:
        assert extract_gold_answer("The answer is 42 but no marker.") is None

    def test_empty_string(self) -> None:
        assert extract_gold_answer("") is None


class TestExtractPredictedAnswer:
    def test_the_answer_is(self) -> None:
        assert extract_predicted_answer("Step 1... Step 2... The answer is 42.") == 42.0

    def test_answer_colon(self) -> None:
        assert extract_predicted_answer("Reasoning here. Answer: 17") == 17.0

    def test_gsm8k_marker_format(self) -> None:
        assert extract_predicted_answer("Working... #### 99") == 99.0

    def test_boxed_format(self) -> None:
        assert extract_predicted_answer("The result is \\boxed{256}.") == 256.0

    def test_dollar_sign(self) -> None:
        assert (
            extract_predicted_answer("She earns the answer is $1,200 per week")
            == 1200.0
        )

    def test_negative_answer(self) -> None:
        assert extract_predicted_answer("The answer is -7") == -7.0

    def test_decimal_answer(self) -> None:
        assert extract_predicted_answer("Answer: 3.5") == pytest.approx(3.5)

    def test_fallback_last_number(self) -> None:
        # No marker — fallback should grab the last numeric token.
        assert extract_predicted_answer("Some text 99 and finally 42") == 42.0

    def test_unparseable(self) -> None:
        assert extract_predicted_answer("I don't know.") is None

    def test_empty(self) -> None:
        assert extract_predicted_answer("") is None


class TestGSM8KReward:
    @pytest.fixture
    def reward(self) -> GSM8KReward:
        return GSM8KReward()

    @pytest.mark.asyncio
    async def test_correct_answer(self, reward: GSM8KReward) -> None:
        turns = [ConversationTurn(role="assistant", content="The answer is 42")]
        result = await reward.compute_reward(turns, context={"gold_answer": 42.0})
        assert result.score == 1.0
        assert result.breakdown["correct"] == 1.0
        assert result.breakdown["predicted"] == 42.0

    @pytest.mark.asyncio
    async def test_incorrect_answer(self, reward: GSM8KReward) -> None:
        turns = [ConversationTurn(role="assistant", content="The answer is 41")]
        result = await reward.compute_reward(turns, context={"gold_answer": 42.0})
        assert result.score == 0.0
        assert result.breakdown["correct"] == 0.0
        assert result.breakdown["abs_error"] == 1.0

    @pytest.mark.asyncio
    async def test_unparseable_response(self, reward: GSM8KReward) -> None:
        turns = [ConversationTurn(role="assistant", content="I don't know")]
        result = await reward.compute_reward(turns, context={"gold_answer": 42.0})
        assert result.score == 0.0
        assert "unparseable" in result.breakdown

    @pytest.mark.asyncio
    async def test_no_gold(self, reward: GSM8KReward) -> None:
        turns = [ConversationTurn(role="assistant", content="The answer is 42")]
        result = await reward.compute_reward(turns, context={})
        assert result.score == 0.0
        assert "no_gold" in result.breakdown

    @pytest.mark.asyncio
    async def test_empty_turns(self, reward: GSM8KReward) -> None:
        result = await reward.compute_reward([], context={"gold_answer": 42.0})
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_tolerance(self) -> None:
        reward = GSM8KReward(tolerance=0.01)
        turns = [ConversationTurn(role="assistant", content="The answer is 42.005")]
        result = await reward.compute_reward(turns, context={"gold_answer": 42.0})
        assert result.score == 1.0


class TestMakeScenarios:
    def test_single_example(self) -> None:
        ex = GSM8KExample(
            question="If Janet has 3 apples and buys 2 more, how many does she have?",
            answer_text="Janet has 3+2=5. #### 5",
            gold_answer=5.0,
        )
        scenarios = make_gsm8k_scenarios([ex])
        assert len(scenarios) == 1
        assert scenarios[0]["gold_answer"] == 5.0
        assert "user_query" in scenarios[0]

    def test_multiple(self) -> None:
        examples = [
            GSM8KExample(question="Q1?", answer_text="A. #### 1", gold_answer=1.0),
            GSM8KExample(question="Q2?", answer_text="B. #### 2", gold_answer=2.0),
        ]
        scenarios = make_gsm8k_scenarios(examples)
        assert len(scenarios) == 2
        assert scenarios[0]["gold_answer"] == 1.0
        assert scenarios[1]["gold_answer"] == 2.0


class TestPartialCreditGSM8KReward:
    @pytest.mark.asyncio
    async def test_correct_full_credit(self) -> None:
        reward = PartialCreditGSM8KReward()
        turns = [ConversationTurn(role="assistant", content="The answer is 42.")]
        result = await reward.compute_reward(turns, context={"gold_answer": 42.0})
        assert result.score == 1.0
        assert result.breakdown.get("tier_correct") == 1.0

    @pytest.mark.asyncio
    async def test_close_partial_credit(self) -> None:
        reward = PartialCreditGSM8KReward()
        turns = [ConversationTurn(role="assistant", content="The answer is 45.")]
        result = await reward.compute_reward(turns, context={"gold_answer": 42.0})
        # 45 vs 42 -> relative error 0.07, within default 0.1 tolerance.
        assert result.score == 0.5
        assert result.breakdown.get("tier_close") == 1.0

    @pytest.mark.asyncio
    async def test_parseable_wrong(self) -> None:
        reward = PartialCreditGSM8KReward()
        turns = [ConversationTurn(role="assistant", content="The answer is 100.")]
        result = await reward.compute_reward(turns, context={"gold_answer": 42.0})
        # 100 vs 42 -> way out of tolerance, but parseable.
        assert result.score == 0.2
        assert result.breakdown.get("tier_parseable_wrong") == 1.0

    @pytest.mark.asyncio
    async def test_unparseable(self) -> None:
        reward = PartialCreditGSM8KReward()
        turns = [
            ConversationTurn(role="assistant", content="I cannot solve this problem.")
        ]
        result = await reward.compute_reward(turns, context={"gold_answer": 42.0})
        assert result.score == 0.0
        assert result.breakdown.get("unparseable") == 1.0

    @pytest.mark.asyncio
    async def test_no_gold(self) -> None:
        reward = PartialCreditGSM8KReward()
        turns = [ConversationTurn(role="assistant", content="42")]
        result = await reward.compute_reward(turns, context={})
        assert result.score == 0.0
        assert result.breakdown.get("no_gold") == 1.0

    @pytest.mark.asyncio
    async def test_no_turns(self) -> None:
        reward = PartialCreditGSM8KReward()
        result = await reward.compute_reward([], context={"gold_answer": 42.0})
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_zero_gold_falls_back_to_absolute(self) -> None:
        # Avoid divide-by-zero when gold is 0.
        reward = PartialCreditGSM8KReward()
        turns = [ConversationTurn(role="assistant", content="0.05")]
        result = await reward.compute_reward(turns, context={"gold_answer": 0.0})
        # abs_error 0.05 > 1e-3 tolerance -> not correct; relative_error 0.05 < 0.1 -> close.
        assert result.score == 0.5

    @pytest.mark.asyncio
    async def test_custom_weights(self) -> None:
        reward = PartialCreditGSM8KReward(parseable_weight=0.3, close_weight=0.7)
        turns = [ConversationTurn(role="assistant", content="The answer is 100.")]
        result = await reward.compute_reward(turns, context={"gold_answer": 42.0})
        assert result.score == 0.3
        turns = [ConversationTurn(role="assistant", content="The answer is 45.")]
        result = await reward.compute_reward(turns, context={"gold_answer": 42.0})
        assert result.score == 0.7
