from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from stateset_agents.core.environment import (
    ConversationEnvironment,
    Environment,
    EnvironmentState,
    EpisodeStatus,
)
from stateset_agents.core.trajectory import MultiTurnTrajectory
from stateset_agents.training.evaluation import EvaluationConfig, evaluate_agent


class DummyAgent:
    async def generate_response(self, history: Any, context: Any) -> str:
        return "hello world"


async def test_evaluate_agent_returns_metrics_with_concurrency() -> None:
    scenario: Dict[str, Any] = {
        "id": "s1",
        "topic": "demo",
        "context": "Demo",
        "user_responses": ["ok"],
    }
    env = ConversationEnvironment(scenarios=[scenario], max_turns=1)
    agent = DummyAgent()

    metrics = await evaluate_agent(
        agent=agent,
        environment=env,
        scenarios=[scenario],
        config=EvaluationConfig(num_episodes=3, max_turns=1, seed=123, concurrency=2),
    )

    assert metrics["eval_num_episodes"] == 3.0
    assert "eval_reward" in metrics
    assert "eval_episode_length" in metrics


async def test_evaluate_agent_falls_back_without_clone() -> None:
    class NoCloneEnvironment(Environment):
        def __init__(self):
            super().__init__(max_turns=1)
            self._in_episode = False

        async def reset(self, scenario: Optional[Dict[str, Any]] = None) -> EnvironmentState:
            return EnvironmentState(
                episode_id="ep1",
                turn_count=0,
                status=EpisodeStatus.ONGOING,
                context={"prompt": "Hello"},
            )

        async def step(self, state: EnvironmentState, action: Any):
            new_state = state.copy()
            new_state.status = EpisodeStatus.COMPLETED
            return new_state, 0.0, True, {}

        async def run_episode(self, agent_fn, scenario=None, max_turns=None):
            if self._in_episode:
                raise RuntimeError("run_episode called concurrently without clone()")
            self._in_episode = True
            try:
                await asyncio.sleep(0.01)
                return MultiTurnTrajectory(turns=[], total_reward=0.0)
            finally:
                self._in_episode = False

    env = NoCloneEnvironment()
    agent = DummyAgent()

    metrics = await evaluate_agent(
        agent=agent,
        environment=env,
        config=EvaluationConfig(num_episodes=2, concurrency=2),
    )

    assert metrics["eval_num_episodes"] == 2.0
