from __future__ import annotations

from typing import Any, Dict, List

from stateset_agents.core.environment import ConversationEnvironment
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

