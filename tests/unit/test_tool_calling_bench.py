"""Unit tests for the tool-calling benchmark module."""

from __future__ import annotations

import pytest

from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.data.tool_calling_bench import (
    SAMPLE_TOOLS,
    ToolCallReward,
    extract_tool_call,
    load_tool_call_scenarios,
    make_tool_call_scenarios,
)


class TestSampleTools:
    def test_three_tools(self) -> None:
        names = {t["name"] for t in SAMPLE_TOOLS}
        assert names == {"get_weather", "calculator", "search"}

    def test_each_tool_has_required_keys(self) -> None:
        for tool in SAMPLE_TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool
            assert isinstance(tool["parameters"], dict)


class TestLoadScenarios:
    def test_loads_full_corpus(self) -> None:
        scenarios = load_tool_call_scenarios()
        assert len(scenarios) == 8

    def test_each_scenario_has_expected_fields(self) -> None:
        for s in load_tool_call_scenarios():
            assert s.user_query
            assert s.expected_tool in {"get_weather", "calculator", "search"}
            assert isinstance(s.expected_params, dict)

    def test_tool_filter(self) -> None:
        only_calc = load_tool_call_scenarios(tool_filter="calculator")
        assert all(s.expected_tool == "calculator" for s in only_calc)
        assert len(only_calc) > 0

    def test_limit(self) -> None:
        few = load_tool_call_scenarios(limit=3)
        assert len(few) == 3


class TestExtractToolCall:
    def test_valid_block(self) -> None:
        response = (
            "Sure! Let me check.\n\n"
            "```json\n"
            '{"tool": "get_weather", "parameters": {"city": "San Francisco"}}\n'
            "```\n"
        )
        call = extract_tool_call(response)
        assert call is not None
        assert call["tool"] == "get_weather"
        assert call["parameters"]["city"] == "San Francisco"

    def test_no_block_returns_none(self) -> None:
        assert extract_tool_call("I don't know what tool to use.") is None

    def test_malformed_json_returns_none(self) -> None:
        response = "```json\n{not valid json\n```"
        assert extract_tool_call(response) is None

    def test_missing_tool_key_returns_none(self) -> None:
        response = '```json\n{"parameters": {"x": 1}}\n```'
        assert extract_tool_call(response) is None

    def test_first_block_wins(self) -> None:
        response = (
            '```json\n{"tool": "a", "parameters": {}}\n```\n'
            '```json\n{"tool": "b", "parameters": {}}\n```'
        )
        call = extract_tool_call(response)
        assert call is not None
        assert call["tool"] == "a"

    def test_empty_returns_none(self) -> None:
        assert extract_tool_call("") is None


class TestToolCallReward:
    @pytest.fixture
    def reward(self) -> ToolCallReward:
        return ToolCallReward()

    @pytest.mark.asyncio
    async def test_perfect_response(self, reward: ToolCallReward) -> None:
        response = (
            '```json\n{"tool": "calculator", "parameters": {"expression": "17 * 24"}}\n```\n'
            "The answer is 408."
        )
        turns = [ConversationTurn(role="assistant", content=response)]
        context = {
            "expected_tool": "calculator",
            "expected_params": {"expression": "17 * 24"},
            "expected_outcome": "408",
        }
        result = await reward.compute_reward(turns, context=context)
        assert result.score == pytest.approx(1.0)
        assert result.breakdown["tool_selection"] == 1.0
        assert result.breakdown["param_correctness"] == 1.0
        assert result.breakdown["outcome"] == 1.0

    @pytest.mark.asyncio
    async def test_wrong_tool(self, reward: ToolCallReward) -> None:
        response = (
            '```json\n{"tool": "search", "parameters": {"query": "17 * 24"}}\n```'
        )
        turns = [ConversationTurn(role="assistant", content=response)]
        context = {
            "expected_tool": "calculator",
            "expected_params": {"expression": "17 * 24"},
            "expected_outcome": "408",
        }
        result = await reward.compute_reward(turns, context=context)
        assert result.breakdown["tool_selection"] == 0.0
        assert result.score < 0.5

    @pytest.mark.asyncio
    async def test_wrong_params(self, reward: ToolCallReward) -> None:
        response = '```json\n{"tool": "calculator", "parameters": {"expression": "wrong"}}\n```'
        turns = [ConversationTurn(role="assistant", content=response)]
        context = {
            "expected_tool": "calculator",
            "expected_params": {"expression": "17 * 24"},
            "expected_outcome": "408",
        }
        result = await reward.compute_reward(turns, context=context)
        assert result.breakdown["tool_selection"] == 1.0
        assert result.breakdown["param_correctness"] == 0.0

    @pytest.mark.asyncio
    async def test_no_tool_call(self, reward: ToolCallReward) -> None:
        response = "I'm not sure how to help."
        turns = [ConversationTurn(role="assistant", content=response)]
        context = {
            "expected_tool": "calculator",
            "expected_params": {"expression": "17 * 24"},
            "expected_outcome": "408",
        }
        result = await reward.compute_reward(turns, context=context)
        assert result.score < 0.01

    @pytest.mark.asyncio
    async def test_outcome_in_text(self, reward: ToolCallReward) -> None:
        # Right tool, right params, outcome present.
        response = (
            '```json\n{"tool": "calculator", "parameters": {"expression": "17 * 24"}}\n```\n'
            "Result: 408"
        )
        turns = [ConversationTurn(role="assistant", content=response)]
        context = {
            "expected_tool": "calculator",
            "expected_params": {"expression": "17 * 24"},
            "expected_outcome": "408",
        }
        result = await reward.compute_reward(turns, context=context)
        assert result.breakdown["outcome"] == 1.0

    @pytest.mark.asyncio
    async def test_empty_turns(self, reward: ToolCallReward) -> None:
        result = await reward.compute_reward([], context={"expected_tool": "x"})
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_weights_sum_to_one_by_default(self, reward: ToolCallReward) -> None:
        total = (
            reward.tool_selection_weight
            + reward.param_correctness_weight
            + reward.outcome_weight
        )
        assert total == pytest.approx(1.0)


class TestMakeScenarios:
    def test_round_trip(self) -> None:
        scenarios = load_tool_call_scenarios(limit=3)
        env = make_tool_call_scenarios(scenarios)
        assert len(env) == 3
        for orig, e in zip(scenarios, env, strict=True):
            assert e["user_query"] == orig.user_query
            assert e["expected_tool"] == orig.expected_tool
