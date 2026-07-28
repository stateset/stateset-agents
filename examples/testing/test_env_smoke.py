"""Smoke tests for every bundled scenario — can each one be reset + stepped?

The cheapest way to catch a malformed scenario JSONL before it tanks a
multi-hour training run: try every scenario through the env once.
"""

from __future__ import annotations

import pytest

from stateset_agents.core import ConversationEnvironment, MultiTurnAgent
from stateset_agents.core.agent_config import AgentConfig
from stateset_agents.data import (
    SupportRewardComposite,
    ToolCallReward,
    load_support_scenarios,
    load_tool_call_scenarios,
    make_support_scenarios,
    make_tool_call_scenarios,
)


@pytest.fixture
async def stub_agent():
    agent = MultiTurnAgent(
        AgentConfig(
            model_name="stub://smoke",
            use_stub_model=True,
        )
    )
    await agent.initialize()
    return agent


@pytest.mark.parametrize("idx", list(range(24)))
async def test_every_support_scenario_round_trips(stub_agent, idx):
    """Each of the 24 bundled support scenarios must reset, step once,
    and produce a reward in [0, 1]."""
    scenarios = make_support_scenarios(load_support_scenarios())
    env = ConversationEnvironment(
        scenarios=[scenarios[idx]],
        reward_fn=SupportRewardComposite(),
        max_turns=1,
    )
    state = await env.reset(scenario=scenarios[idx])
    response = await stub_agent.generate_response(
        state.context["scenario"]["user_query"]
    )
    payload = await env.step(response)
    assert payload["done"] is True
    assert 0.0 <= float(payload["reward"]) <= 1.0


@pytest.mark.parametrize("idx", list(range(8)))
async def test_every_tool_scenario_round_trips(stub_agent, idx):
    """Each of the 8 bundled tool-call scenarios must reset + step + score."""
    scenarios = make_tool_call_scenarios(load_tool_call_scenarios())
    env = ConversationEnvironment(
        scenarios=[scenarios[idx]],
        reward_fn=ToolCallReward(),
        max_turns=1,
    )
    state = await env.reset(scenario=scenarios[idx])
    response = await stub_agent.generate_response(
        state.context["scenario"]["user_query"]
    )
    payload = await env.step(response)
    assert payload["done"] is True
    assert 0.0 <= float(payload["reward"]) <= 1.0


def test_support_scenarios_load_count():
    """Guard against accidental deletions from the bundled corpus."""
    scenarios = load_support_scenarios()
    assert len(scenarios) == 24, f"Expected 24 support scenarios, got {len(scenarios)}"


def test_tool_scenarios_load_count():
    scenarios = load_tool_call_scenarios()
    assert len(scenarios) == 8, f"Expected 8 tool scenarios, got {len(scenarios)}"


def test_support_scenarios_have_intents():
    """Every support scenario must declare an intent — the rubric depends on it."""
    for s in load_support_scenarios():
        assert s.intent in {"refund", "technical", "billing", "general"}
