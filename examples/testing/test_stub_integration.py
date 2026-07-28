"""End-to-end stub integration tests — no mocks, no GPU.

These run the *real* `MultiTurnAgent`, `ConversationEnvironment`, and a real
`RewardFunction` against the stub backend. They're not unit tests — they're
the cheapest possible check that the wiring is intact across a refactor.

Pattern: if you change anything in `core/agent.py`, `core/environment.py`,
or `core/reward.py`, run these before you push.
"""

from __future__ import annotations

import pytest

from stateset_agents.core import ConversationEnvironment, MultiTurnAgent
from stateset_agents.core.agent_config import AgentConfig
from stateset_agents.data import (
    SupportRewardComposite,
    load_support_scenarios,
    make_support_scenarios,
)


@pytest.fixture
async def stub_agent():
    agent = MultiTurnAgent(
        AgentConfig(
            model_name="stub://integration-tests",
            use_stub_model=True,
        )
    )
    await agent.initialize()
    yield agent
    # Stub backends don't need teardown — but for real agents you'd close pools here.


@pytest.fixture
def support_env():
    return ConversationEnvironment(
        scenarios=make_support_scenarios(load_support_scenarios()[:3]),
        reward_fn=SupportRewardComposite(),
        max_turns=2,
    )


async def test_stub_agent_initializes_with_property(stub_agent):
    """The stub backend exposes its identity via `_is_stub_backend`."""
    assert stub_agent._is_stub_backend is True


async def test_stub_agent_response_is_string(stub_agent):
    response = await stub_agent.generate_response("Say hi")
    assert isinstance(response, str)
    assert len(response) > 0


async def test_env_reset_returns_state(support_env):
    state = await support_env.reset(scenario=support_env.scenarios[0])
    assert state.episode_id
    assert state.context["scenario"]["user_query"]


async def test_env_step_returns_payload(support_env, stub_agent):
    state = await support_env.reset(scenario=support_env.scenarios[0])
    response = await stub_agent.generate_response(
        state.context["scenario"]["user_query"]
    )
    payload = await support_env.step(response)
    assert {"state", "reward", "done"}.issubset(payload.keys())
    assert isinstance(payload["reward"], (int, float))
    assert isinstance(payload["done"], bool)


async def test_episode_terminates_by_max_turns(support_env, stub_agent):
    """An episode must reach `done=True` within `max_turns` steps. Guards against
    runaway loops in trainer rollouts."""
    await support_env.reset(scenario=support_env.scenarios[0])
    steps = 0
    done = False
    while not done and steps < support_env.max_turns + 1:
        response = await stub_agent.generate_response("anything")
        payload = await support_env.step(response)
        done = bool(payload["done"])
        steps += 1
    assert (
        done
    ), f"Episode did not terminate after {steps} steps (max_turns={support_env.max_turns})"
    assert steps <= support_env.max_turns


async def test_rewards_in_expected_range(support_env, stub_agent):
    """Whatever the stub responds with, the rubric must produce a score in [0, 1]."""
    for scenario in support_env.scenarios[:2]:
        await support_env.reset(scenario=scenario)
        response = await stub_agent.generate_response(scenario["user_query"])
        payload = await support_env.step(response)
        assert 0.0 <= float(payload["reward"]) <= 1.0
