import asyncio
from typing import Any, Dict, List, Optional

import pytest

from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from stateset_agents.core.computational_engine import ComputationalGRPOEngine
from stateset_agents.core.environment import ConversationTurn, Environment, EnvironmentState, EpisodeStatus
from stateset_agents.core.reward import RewardFunction, RewardResult


class StubEnvironment(Environment):
    async def reset(self, scenario: Optional[Dict[str, Any]] = None) -> EnvironmentState:
        return EnvironmentState(
            episode_id="stub-episode",
            turn_count=0,
            status=EpisodeStatus.ONGOING,
        )

    async def step(
        self,
        state: EnvironmentState,
        action: ConversationTurn,
    ) -> tuple[EnvironmentState, ConversationTurn, float, bool]:
        state.turn_count += 1
        reply = ConversationTurn(role="system", content="ack")
        return state, reply, 0.0, True

    async def get_reward(self, trajectory) -> float:  # type: ignore[override]
        return 0.42


class StubReward(RewardFunction):
    async def compute_reward(
        self, turns: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None
    ) -> RewardResult:
        return RewardResult(score=0.5, breakdown={}, metadata={})


@pytest.mark.asyncio
async def test_computational_engine_uses_stub_agent_string_prompt():
    agent = MultiTurnAgent(
        AgentConfig(
            model_name="stub://engine-test",
            use_stub_model=True,
            stub_responses=["Stub response from engine"],
        )
    )
    await agent.initialize()

    engine = ComputationalGRPOEngine(
        agent=agent,
        environment=StubEnvironment(),
        reward_function=StubReward(),
        num_workers=1,
        use_learned_rewards=True,
    )

    trajectory = await engine._generate_single_trajectory("Hello world")
    assert trajectory.prompt == "Hello world"
    assert "Stub response" in trajectory.response
    assert trajectory.learned_reward == pytest.approx(0.5)
