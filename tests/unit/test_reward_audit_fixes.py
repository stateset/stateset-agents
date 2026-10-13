"""Regression tests for reward / environment correctness fixes."""

from __future__ import annotations

import datetime
import json
import types

import pytest

from stateset_agents.core.basic_rewards import TaskCompletionReward
from stateset_agents.core.conversation_environment import ConversationEnvironment
from stateset_agents.core.reward_base import RewardFunction, RewardResult, RewardType
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.environments.symbolic_physics import SymbolicPhysicsEnvironment
from stateset_agents.rewards.symbolic_physics_reward import (
    SymbolicPhysicsRewardFunction,
)
from stateset_agents.training.reward_calibration import (
    MultiRewardCalibrator,
    RewardNormalizer,
)

torch = pytest.importorskip("torch")


def _assistant(content: str) -> list[ConversationTurn]:
    return [ConversationTurn(role="assistant", content=content)]


# --- RULER judge receives whole trajectories -------------------------------


def _ruler(monkeypatch):
    import stateset_agents.rewards.llm_reward as lr

    monkeypatch.setattr(lr, "LITELLM_AVAILABLE", True)
    reward = lr.RulerRewardFunction(model="judge", max_retries=1)
    prompts: list[str] = []

    async def fake_completion(messages):
        prompts.append(messages[1]["content"])
        n = messages[1]["content"].count("<trajectory id=")
        scores = [
            {"trajectory_id": str(i + 1), "explanation": "ok", "score": 0.1 * (i + 1)}
            for i in range(n)
        ]
        content = json.dumps({"scores": scores})
        message = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    reward._make_completion_call = fake_completion
    return reward, prompts


@pytest.mark.asyncio
async def test_ruler_scores_single_trajectory_with_judge(monkeypatch):
    reward, prompts = _ruler(monkeypatch)
    turns = [
        ConversationTurn(role="user", content="Where is my order?"),
        ConversationTurn(role="assistant", content="It ships tomorrow."),
    ]
    result = await reward.compute_reward(turns)

    assert "fallback_used" not in result.breakdown
    assert result.score == pytest.approx(0.1)
    assert len(prompts) == 1
    assert prompts[0].count("<trajectory id=") == 1
    assert "It ships tomorrow." in prompts[0]


@pytest.mark.asyncio
async def test_ruler_batch_returns_one_result_per_trajectory(monkeypatch):
    reward, prompts = _ruler(monkeypatch)
    a = [
        ConversationTurn(role="user", content="Hi"),
        ConversationTurn(role="assistant", content="Hello A"),
    ]
    b = [
        ConversationTurn(role="user", content="Hi"),
        ConversationTurn(role="assistant", content="Hello B"),
    ]
    results = await reward.compute_batch_rewards([a, b])

    assert [r.score for r in results] == pytest.approx([0.1, 0.2])
    assert prompts[0].count("<trajectory id=") == 2


# --- Symbolic physics reward is deterministic --------------------------------


@pytest.mark.asyncio
async def test_symbolic_reward_is_deterministic_for_identical_answers():
    reward = SymbolicPhysicsRewardFunction()
    ctx = {"variables": ["x", "y"], "target_expression": "x*y"}
    scores = {
        (await reward.compute_reward(_assistant("x*abs(y)"), ctx)).score
        for _ in range(6)
    }
    assert len(scores) == 1


@pytest.mark.asyncio
async def test_symmetry_constraint_recomputes_derived_variables():
    reward = SymbolicPhysicsRewardFunction()
    ctx = {
        "variables": ["x", "y", "d"],
        "derived_variables": {"d": "x - y"},
        "constraints": [{"type": "antisymmetry", "pairs": [["x", "y"]]}],
    }
    direct = await reward.compute_reward(_assistant("x - y"), ctx)
    derived = await reward.compute_reward(_assistant("d"), ctx)
    assert derived.score == pytest.approx(direct.score)


@pytest.mark.asyncio
async def test_sign_flip_constraint_recomputes_derived_variables():
    reward = SymbolicPhysicsRewardFunction()
    ctx = {
        "variables": ["x", "y", "s"],
        "derived_variables": {"s": "x * x * x"},
        "constraints": [{"type": "sign_flip", "variable": "x", "parity": "odd"}],
    }
    direct = await reward.compute_reward(_assistant("x * x * x * y"), ctx)
    derived = await reward.compute_reward(_assistant("s * y"), ctx)
    assert derived.score == pytest.approx(direct.score)


# --- ConversationEnvironment scores the reply against what it answers -------


class _Spy(RewardFunction):
    def __init__(self) -> None:
        super().__init__(1.0, RewardType.IMMEDIATE, "spy")
        self.calls: list[list[tuple[str, str]]] = []

    async def compute_reward(self, turns, context=None):
        self.calls.append([(t.role, t.content) for t in turns])
        return RewardResult(score=1.0)


@pytest.mark.asyncio
async def test_conversation_env_step_reward_sees_answered_user_turn():
    spy = _Spy()
    env = ConversationEnvironment(
        [{"user_responses": ["U-reply-A", "U-reply-B"]}], max_turns=2, reward_fn=spy
    )
    count = [0]

    async def agent(history, ctx):
        count[0] += 1
        return f"assistant-{count[0]}"

    trajectory = await env.run_episode(agent)
    replies = [t.content for t in trajectory.turns if t.role == "user"]

    assert spy.calls[0] == [("assistant", "assistant-1")]
    # The second reply is scored against the user message it answers, not
    # against the simulated reply that follows it.
    assert spy.calls[1] == [
        ("assistant", "assistant-1"),
        ("user", replies[0]),
        ("assistant", "assistant-2"),
    ]


# --- Reward calibration -----------------------------------------------------


class _LengthReward(RewardFunction):
    def __init__(self) -> None:
        super().__init__(1.0, RewardType.IMMEDIATE, "len")

    async def compute_reward(self, turns, context=None):
        return RewardResult(score=float(len(turns[-1].content)))


@pytest.mark.asyncio
async def test_calibrated_functions_use_precalibration_stats():
    calibrator = MultiRewardCalibrator([_LengthReward()])
    episodes = [_assistant("x" * k) for k in range(10, 60)]
    stats = (await calibrator.calibrate(episodes))["len"]
    mean, std = stats.mean, stats.std

    calibrated = calibrator.get_calibrated_functions()[0]
    result = await calibrated.compute_reward(_assistant("x" * 50))

    expected = (50 - mean) / std
    assert result.score == pytest.approx(expected)


@pytest.mark.parametrize("buffer_size", [10, 50, 150])
def test_reward_normalizer_tracks_full_small_buffer(buffer_size):
    normalizer = RewardNormalizer(buffer_size=buffer_size)
    for i in range(500):
        normalizer.add_reward(float(i % 2))
    assert normalizer.stats.std == pytest.approx(0.5, abs=0.02)
    assert normalizer.stats.mean == pytest.approx(0.5, abs=0.02)


# --- Neural reward trainer --------------------------------------------------


def _trajectory(i: int, metadata: dict):
    from stateset_agents.training.neural_reward_trainer import TrajectoryData

    return TrajectoryData(
        id=str(i),
        prompt=f"p{i}",
        response=f"r{i}" * (i % 7 + 1),
        raw_reward=float(i % 3) / 2.0,
        learned_reward=0.0,
        computational_cost=0.0,
        timestamp=datetime.datetime.now(),
        metadata=metadata,
    )


def test_neural_reward_trains_on_heterogeneous_metadata():
    from stateset_agents.training.neural_reward_trainer import (
        NeuralRewardModel,
        NeuralRewardTrainer,
    )

    trainer = NeuralRewardTrainer(NeuralRewardModel(), max_epochs=2)
    data = [_trajectory(i, {"a": 1} if i % 2 else {"b": None}) for i in range(20)]
    results = trainer.train(data[:16], data[16:], verbose=False)
    assert results["epochs_trained"] >= 1


def test_neural_reward_retrain_resets_early_stopping():
    from stateset_agents.training.neural_reward_trainer import (
        NeuralRewardModel,
        NeuralRewardTrainer,
    )

    torch.manual_seed(0)
    trainer = NeuralRewardTrainer(NeuralRewardModel(), max_epochs=3, patience=2)
    data = [_trajectory(i, {}) for i in range(24)]
    trainer.train(data[:16], data[16:], verbose=False)
    # A previous run's stale best loss / patience must not cut the next run short.
    trainer.best_val_loss = -1.0
    trainer.patience_counter = trainer.patience
    before = len(trainer.train_losses)
    trainer.train(data[:16], data[16:], verbose=False)
    assert len(trainer.train_losses) - before > 1


def test_transformer_reward_dataset_items_collate_with_mixed_metadata():
    from torch.utils.data import DataLoader

    from stateset_agents.training.transformer_reward_model import (
        RewardDataset,
        RewardExample,
    )

    class _Tok:
        def __call__(self, text, max_length, **_):
            ids = torch.zeros(1, max_length, dtype=torch.long)
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    examples = [
        RewardExample(prompt="p", response="r", reward=1.0, metadata={"a": 1}),
        RewardExample(prompt="p", response="r", reward=0.0, metadata={"b": None}),
    ]
    batch = next(iter(DataLoader(RewardDataset(examples, _Tok(), 4), batch_size=2)))
    assert batch["reward"].shape == (2,)


# --- TaskCompletionReward ---------------------------------------------------


@pytest.mark.asyncio
async def test_task_completion_counts_only_required_actions():
    reward = TaskCompletionReward(task_criteria={"required_actions": ["issue_refund"]})
    unrelated = await reward.compute_reward(
        _assistant("hi"), {"completed_actions": ["greet", "lookup_order", "greet"]}
    )
    assert unrelated.score == 0.0

    done = await reward.compute_reward(
        _assistant("hi"), {"completed_actions": ["issue_refund", "issue_refund"]}
    )
    assert done.score == 1.0


# --- Symbolic physics environment shows the task ----------------------------


@pytest.mark.asyncio
async def test_symbolic_env_without_scenario_shows_sampled_task():
    env = SymbolicPhysicsEnvironment(
        [
            {
                "task_id": "t1",
                "prompt": "Find kinetic energy in terms of m and v",
                "variables": ["m", "v"],
                "target_expression": "0.5*m*v**2",
            }
        ],
        seed=0,
    )
    seen = []

    async def agent(history, ctx):
        seen.append(history)
        return "0.5*m*v**2"

    await env.run_episode(agent)
    assert "kinetic energy" in seen[0][0]["content"]
