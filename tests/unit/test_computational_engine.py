from typing import Any

import pytest

from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from stateset_agents.core.computational_engine import ComputationalGRPOEngine
from stateset_agents.core.environment import (
    ConversationTurn,
    Environment,
    EnvironmentState,
    EpisodeStatus,
)
from stateset_agents.core.reward import RewardFunction, RewardResult


class StubEnvironment(Environment):
    async def reset(self, scenario: dict[str, Any] | None = None) -> EnvironmentState:
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
        self, turns: list[dict[str, Any]], context: dict[str, Any] | None = None
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


def _engine(agent: Any) -> ComputationalGRPOEngine:
    with pytest.warns(DeprecationWarning):
        return ComputationalGRPOEngine(
            agent=agent,
            environment=StubEnvironment(),
            reward_function=StubReward(),
            num_workers=1,
        )


def test_engine_is_deprecated_with_a_date_and_points_at_the_trainers() -> None:
    with pytest.warns(DeprecationWarning, match=r"removed in v0\.71\.0") as caught:
        ComputationalGRPOEngine(
            agent=object(),  # type: ignore[arg-type]
            environment=StubEnvironment(),
            reward_function=StubReward(),
            num_workers=1,
        )
    assert "stateset_agents.train()" in str(caught[0].message)
    assert caught[0].filename == __file__


@pytest.mark.asyncio
async def test_update_policy_does_not_change_weights() -> None:
    """The documented behaviour: a diagnostic loss, never a gradient step."""
    torch = pytest.importorskip("torch")
    from stateset_agents.core.computational_engine import ComputationalTrajectory

    class TinyLM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(16, 4)

        def forward(self, input_ids, labels=None, **_: Any):
            logits = self.embed(input_ids) @ self.embed.weight.T
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, 16), labels.view(-1)
            )
            return type("Out", (), {"loss": loss})()

    class Tokenizer:
        def __call__(self, text: str, **_: Any):
            return {"input_ids": torch.tensor([[1, 2, 3]])}

    class Agent:
        model = TinyLM()
        tokenizer = Tokenizer()

    agent = Agent()
    before = agent.model.embed.weight.detach().clone()
    engine = _engine(agent)
    trajectory = ComputationalTrajectory(
        id="t",
        prompt="p",
        response="r",
        raw_reward_signal=None,
        learned_reward=1.0,
        computational_cost=0.0,
        timestamp=__import__("datetime").datetime.now(),
    )
    import numpy as np

    loss = await engine._update_policy([trajectory], np.array([1.0]))
    assert loss > 0
    assert torch.equal(agent.model.embed.weight, before)
    assert agent.model.embed.weight.grad is None


@pytest.mark.asyncio
async def test_update_policy_propagates_programmer_errors() -> None:
    """A malformed tokenizer output is a bug, not a skippable trajectory."""
    torch = pytest.importorskip("torch")
    from stateset_agents.core.computational_engine import ComputationalTrajectory

    class Agent:
        model = torch.nn.Linear(2, 2)

        class tokenizer:  # noqa: N801 - a callable stand-in
            def __new__(cls, text: str, **_: Any):
                return {"attention_mask": torch.ones(1, 3)}  # no input_ids

    engine = _engine(Agent())
    trajectory = ComputationalTrajectory(
        id="t",
        prompt="p",
        response="r",
        raw_reward_signal=None,
        learned_reward=1.0,
        computational_cost=0.0,
        timestamp=__import__("datetime").datetime.now(),
    )
    import numpy as np

    with pytest.raises(KeyError, match="input_ids"):
        await engine._update_policy([trajectory], np.array([1.0]))


def test_subclassing_base_trainer_is_deprecated() -> None:
    pytest.importorskip("torch")
    from stateset_agents.training.base_trainer import BaseTrainer

    with pytest.warns(DeprecationWarning, match=r"BaseTrainer.*removed in v0\.71\.0"):

        class _Legacy(BaseTrainer):  # type: ignore[type-arg]
            async def train_step(self, batch: Any) -> dict[str, float]:
                return {}

            async def train(self) -> dict[str, Any]:
                return {}
