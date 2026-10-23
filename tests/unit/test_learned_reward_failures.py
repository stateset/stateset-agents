"""Learned reward models never stand in a score, and never retrain silently.

``NeuralRewardFunction`` and ``LearnedRewardFunction`` caught TypeError,
AttributeError, KeyError, ValueError and RuntimeError from the model and
returned 0.5 -- a bug in the reward model trained the policy towards a
constant -- and ``NeuralRewardFunction`` retrained itself from its replay
buffer every 100 calls, so the reward moved under the policy without the
caller asking. Now an outage (I/O, out of memory, a timeout) raises
``RewardUnavailableError`` (which trainers drop or propagate per the reward
contract), anything else propagates, and online retraining is opt-in
(``retrain_online=True``).
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.core.reward_context import RewardUnavailableError  # noqa: E402
from stateset_agents.core.trajectory import ConversationTurn  # noqa: E402
from stateset_agents.training.neural_reward_trainer import (  # noqa: E402
    NeuralRewardFunction,
)

TURNS = [
    ConversationTurn(role="user", content="q"),
    ConversationTurn(role="assistant", content="a"),
]


def _raising(error: BaseException) -> Any:
    def predict(*args: Any, **kwargs: Any) -> float:
        raise error

    return predict


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [TypeError("t"), AttributeError("a"), KeyError("k"), ValueError("v")]
)
async def test_neural_reward_bugs_propagate(error: BaseException) -> None:
    reward = NeuralRewardFunction()
    reward.model.predict_reward = _raising(error)  # type: ignore[method-assign]
    with pytest.raises(type(error)):
        await reward.compute_reward(TURNS)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [OSError("disk"), torch.cuda.OutOfMemoryError("oom"), TimeoutError()]
)
async def test_neural_reward_outage_is_unavailable(error: BaseException) -> None:
    reward = NeuralRewardFunction()
    reward.model.predict_reward = _raising(error)  # type: ignore[method-assign]
    with pytest.raises(RewardUnavailableError) as caught:
        await reward.compute_reward(TURNS)
    assert caught.value.__cause__ is error


@pytest.mark.asyncio
async def test_neural_reward_does_not_retrain_itself_by_default() -> None:
    reward = NeuralRewardFunction(min_trajectories=0)
    calls: list[int] = []

    async def update() -> None:
        calls.append(1)

    reward._update_model = update  # type: ignore[method-assign]
    for _ in range(250):
        await reward.compute_reward(TURNS)
    assert calls == []
    assert reward.update_count == 0


@pytest.mark.asyncio
async def test_online_retraining_is_opt_in() -> None:
    reward = NeuralRewardFunction(
        min_trajectories=0, retrain_online=True, update_frequency=100
    )
    calls: list[int] = []

    async def update() -> None:
        calls.append(1)

    reward._update_model = update  # type: ignore[method-assign]
    for _ in range(250):
        await reward.compute_reward(TURNS)
    assert len(calls) == 2


def test_fallback_reward_is_deprecated() -> None:
    with pytest.warns(DeprecationWarning, match="fallback_reward"):
        NeuralRewardFunction(fallback_reward=0.3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        NeuralRewardFunction()


class _Trainer:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.config = type("Config", (), {"base_model": "tiny"})()

    def predict(self, prompt: str, response: str) -> float:
        if self.error is not None:
            raise self.error
        return 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [TypeError("t"), AttributeError("a"), KeyError("k"), RuntimeError("bug")]
)
async def test_learned_reward_bugs_propagate(error: BaseException) -> None:
    from stateset_agents.training.transformer_reward_model import (
        LearnedRewardFunction,
    )

    reward = LearnedRewardFunction(trainer=_Trainer(error))  # type: ignore[arg-type]
    with pytest.raises(type(error)):
        await reward.compute_reward(TURNS)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [OSError("disk"), torch.cuda.OutOfMemoryError("oom")])
async def test_learned_reward_outage_is_unavailable(error: BaseException) -> None:
    from stateset_agents.training.transformer_reward_model import (
        LearnedRewardFunction,
    )

    reward = LearnedRewardFunction(trainer=_Trainer(error))  # type: ignore[arg-type]
    with pytest.raises(RewardUnavailableError):
        await reward.compute_reward(TURNS)


@pytest.mark.asyncio
async def test_learned_reward_scores_normally() -> None:
    from stateset_agents.training.transformer_reward_model import (
        LearnedRewardFunction,
    )

    reward = LearnedRewardFunction(trainer=_Trainer())  # type: ignore[arg-type]
    result = await reward.compute_reward(TURNS)
    assert result.score == pytest.approx(0.5)  # sigmoid(0)
    assert "error" not in result.metadata
