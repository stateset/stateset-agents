import pytest

from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.environments.symbolic_physics import SymbolicPhysicsEnvironment
from stateset_agents.rewards.symbolic_physics_reward import (
    SymbolicPhysicsRewardFunction,
    SymbolicRewardConfig,
)


@pytest.mark.asyncio
async def test_symbolic_reward_equivalence() -> None:
    reward_fn = SymbolicPhysicsRewardFunction(
        config=SymbolicRewardConfig(num_samples=6)
    )
    context = {
        "variables": ["s", "t"],
        "target_expression": "s + t",
        "constraints": [],
    }
    turn = ConversationTurn(role="assistant", content="s + t")
    result = await reward_fn.compute_reward(turns=[turn], context=context)
    assert result.score > 0.9


@pytest.mark.asyncio
async def test_symbolic_environment_done_on_success() -> None:
    tasks = [
        {
            "id": "toy",
            "prompt": "Return s + t",
            "variables": ["s", "t"],
            "target_expression": "s + t",
            "constraints": [],
        }
    ]
    reward_fn = SymbolicPhysicsRewardFunction(
        config=SymbolicRewardConfig(num_samples=4)
    )
    env = SymbolicPhysicsEnvironment(
        tasks=tasks, reward_fn=reward_fn, max_turns=1, success_threshold=0.9
    )
    state = await env.reset()
    turn = ConversationTurn(role="assistant", content="s + t")
    _, reward, done, _ = await env.step(state, turn)
    assert done is True
    assert reward > 0.9


@pytest.mark.asyncio
async def test_symbolic_reward_sum_constraint() -> None:
    reward_fn = SymbolicPhysicsRewardFunction(
        config=SymbolicRewardConfig(num_samples=6)
    )
    context = {
        "variables": ["s", "t"],
        "constraints": [{"type": "sum", "terms": ["s", "t"]}],
    }
    turn = ConversationTurn(role="assistant", content="s + t")
    result = await reward_fn.compute_reward(turns=[turn], context=context)
    assert result.score > 0.8


@pytest.mark.asyncio
async def test_symbolic_reward_sign_flip_product_constraint() -> None:
    reward_fn = SymbolicPhysicsRewardFunction(
        config=SymbolicRewardConfig(num_samples=6)
    )
    context = {
        "variables": ["s", "t"],
        "constraints": [
            {"type": "product", "factors": ["s", "t"]},
            {"type": "sign_flip", "variables": ["s", "t"], "parity": "odd"},
        ],
    }
    turn = ConversationTurn(role="assistant", content="s * t")
    result = await reward_fn.compute_reward(turns=[turn], context=context)
    assert result.score > 0.8
