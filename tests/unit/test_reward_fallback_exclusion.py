"""A reward-fallback episode is never trained on (docs/REWARD_CONTRACT.md).

A reward that scores with an opted-in fallback (``NSRVerifierConfig(
error_score=...)``, ``RulerRewardFunction(fallback_enabled=True)``, a
renormalising composite) flags its result ``metadata["reward_fallback"]``
(``reward_context.REWARD_FALLBACK_KEY``), and composers list left-out
components (``failed_components``); environments and ``score_episode`` copy
both to the trajectory. No trainer read them, so flagged episodes trained
like any other. Now:

* MultiTurn GRPO (and through it the DDP trainer) drops a flagged episode at
  rollout time, like an outage, and the GRPO update excludes any flagged
  trajectory that still reaches it;
* SingleTurn GRPO drops a flagged generation from its group;
* GSPO, DAPO, GEPO and VAPO drop a flagged reply from its group (a reward
  callable may return a ``RewardResult`` to flag one);
* the count is reported in the metrics and a step in which every episode is
  flagged raises ``RewardUnavailableError``.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from stateset_agents.core.conversation_environment import (  # noqa: E402
    ConversationEnvironment,
)
from stateset_agents.core.reward_base import (  # noqa: E402
    RewardFunction,
    RewardResult,
    RewardType,
)
from stateset_agents.core.reward_context import (  # noqa: E402
    FAILED_COMPONENTS_KEY,
    REWARD_FALLBACK_KEY,
    RewardUnavailableError,
)
from stateset_agents.core.trajectory import (  # noqa: E402
    ConversationTurn,
    MultiTurnTrajectory,
    TrajectoryGroup,
)
from stateset_agents.training.config import TrainingConfig  # noqa: E402
from stateset_agents.training.multi_turn_trainer import (  # noqa: E402
    MultiTurnGRPOTrainer,
)
from stateset_agents.training.spine.rollouts import (  # noqa: E402
    exclude_flagged_groups,
)
from stateset_agents.training.spine.scoring import (  # noqa: E402
    CallableScorer,
    ResponseScorer,
    trajectory_reward_fallback,
)

# --- shared pieces ---------------------------------------------------------


class FlagsSome(RewardFunction):
    """Scores 0.5; replies containing ``flag`` get an opted-in fallback."""

    def __init__(self, flag: str = "FALLBACK") -> None:
        super().__init__(reward_type=RewardType.SPARSE, name="flags_some")
        self.flag = flag

    async def compute_reward(self, turns: Any, context: Any = None) -> RewardResult:
        last = [t for t in turns if getattr(t, "role", None) == "assistant"][-1]
        if self.flag in str(last.content):
            return RewardResult(score=0.0, metadata={REWARD_FALLBACK_KEY: "judge down"})
        return RewardResult(score=0.5)


def flagging_callable(flag: str = "FALLBACK") -> Any:
    def reward(prompt: str, response: str) -> Any:
        if flag in response:
            return RewardResult(score=0.0, metadata={REWARD_FALLBACK_KEY: "outage"})
        return sum(ch.isdigit() for ch in response) / max(len(response), 1)

    return reward


def _trajectory(text: str, reward: float, **metadata: Any) -> MultiTurnTrajectory:
    return MultiTurnTrajectory(
        turns=[
            ConversationTurn(role="user", content="q"),
            ConversationTurn(role="assistant", content=text),
        ],
        total_reward=reward,
        metadata=dict(metadata),
    )


# --- scorers and the group filter ------------------------------------------


@pytest.mark.asyncio
async def test_scorers_report_the_fallback_flag() -> None:
    scorer = ResponseScorer(FlagsSome())
    assert await scorer.score_checked("q", "fine") == (0.5, None)
    assert await scorer.score_checked("q", "FALLBACK") == (0.0, "judge down")
    assert await scorer.score("q", "FALLBACK") == 0.0  # the float API is unchanged

    callable_scorer = CallableScorer(flagging_callable())
    assert await callable_scorer.score_checked("q", "12a") == (
        pytest.approx(2 / 3),
        None,
    )
    assert (await callable_scorer.score_checked("q", "FALLBACK")).fallback == "outage"
    assert await callable_scorer.score("q", "FALLBACK") == 0.0


def test_trajectory_flags_include_left_out_components() -> None:
    assert trajectory_reward_fallback(_trajectory("a", 1.0)) is None
    flagged = _trajectory("a", 1.0, **{REWARD_FALLBACK_KEY: "renormalized"})
    assert trajectory_reward_fallback(flagged) == "renormalized"
    partial = _trajectory("a", 1.0, **{FAILED_COMPONENTS_KEY: ["judge"]})
    assert "judge" in trajectory_reward_fallback(partial)


def test_group_filter_drops_flagged_episodes_and_empty_groups() -> None:
    clean = [_trajectory("a", 1.0), _trajectory("b", 0.0)]
    flagged = _trajectory("c", 0.3, **{REWARD_FALLBACK_KEY: "x"})
    groups = [
        TrajectoryGroup(scenario_id="s1", trajectories=[*clean, flagged]),
        TrajectoryGroup(scenario_id="s2", trajectories=[flagged]),
    ]
    kept, excluded = exclude_flagged_groups(groups)
    assert excluded == 2
    assert [g.scenario_id for g in kept] == ["s1"]
    assert kept[0].trajectories == clean
    assert kept[0].group_id == groups[0].group_id
    assert len(groups[0].trajectories) == 3  # the input is not mutated


# --- MultiTurn GRPO ----------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_turn_rollouts_drop_flagged_episodes() -> None:
    scenario = {"id": "s1", "user_turns": ["hi"]}
    replies = iter(["ok", "FALLBACK", "ok", "ok"])

    class Agent:
        async def generate_response(self, history: Any, context: Any) -> str:
            return next(replies)

    reward = FlagsSome()
    trainer = MultiTurnGRPOTrainer(
        agent=Agent(),
        environment=ConversationEnvironment([scenario], max_turns=1, reward_fn=reward),
        reward_fn=reward,
        config=TrainingConfig(num_generations=4, batched_rollouts=False),
    )
    [group] = await trainer.generate_trajectories([scenario], num_generations=4)
    assert len(group.trajectories) == 3
    assert all(trajectory_reward_fallback(t) is None for t in group.trajectories)
    assert trainer.reward_fallback_episodes == 1


@pytest.mark.asyncio
async def test_multi_turn_rollouts_raise_when_every_episode_is_flagged() -> None:
    scenario = {"id": "s1", "user_turns": ["hi"]}

    class Agent:
        async def generate_response(self, history: Any, context: Any) -> str:
            return "FALLBACK"

    reward = FlagsSome()
    trainer = MultiTurnGRPOTrainer(
        agent=Agent(),
        environment=ConversationEnvironment([scenario], max_turns=1, reward_fn=reward),
        reward_fn=reward,
        config=TrainingConfig(num_generations=2, batched_rollouts=False),
    )
    with pytest.raises(RewardUnavailableError, match="reward fallback"):
        await trainer.generate_trajectories([scenario], num_generations=2)


def _update_trainer(monkeypatch: pytest.MonkeyPatch) -> Any:
    from types import SimpleNamespace

    from stateset_agents.training import multi_turn_trainer

    model = torch.nn.Linear(2, 1)
    monkeypatch.setattr(
        multi_turn_trainer,
        "compute_token_old_logprobs",
        lambda *args: [model.weight.detach().clone()],
    )
    trainer = MultiTurnGRPOTrainer(
        agent=SimpleNamespace(model=model, tokenizer=None),
        environment=None,
        reward_fn=None,
        config=TrainingConfig(num_generations=2, bf16=False),
    )
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    seen: list[list[TrajectoryGroup]] = []

    def loss(groups: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        seen.append(list(groups))
        return {"total_loss": model.weight.square().sum(), "path": "token"}

    trainer.compute_grpo_loss = loss  # type: ignore[method-assign]
    trainer.seen = seen
    return trainer


@pytest.mark.asyncio
async def test_grpo_update_excludes_flagged_trajectories_from_the_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _update_trainer(monkeypatch)
    flagged = _trajectory("c", 9.0, **{REWARD_FALLBACK_KEY: "judge down"})
    group = TrajectoryGroup(
        scenario_id="s",
        trajectories=[_trajectory("a", 1.0), _trajectory("b", 0.0), flagged],
    )
    metrics = await trainer.training_step([group])
    assert flagged not in trainer.seen[0][0].trajectories
    assert len(trainer.seen[0][0].trajectories) == 2
    assert metrics["reward_fallback_episodes"] == 1


@pytest.mark.asyncio
async def test_grpo_update_raises_when_every_trajectory_is_flagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _update_trainer(monkeypatch)
    group = TrajectoryGroup(
        scenario_id="s",
        trajectories=[
            _trajectory("a", 1.0, **{REWARD_FALLBACK_KEY: "x"}),
            _trajectory("b", 0.0, **{FAILED_COMPONENTS_KEY: ["judge"]}),
        ],
    )
    with pytest.raises(RewardUnavailableError, match="reward fallback"):
        await trainer.training_step([group])
    assert trainer.seen == []


# --- SingleTurn GRPO ---------------------------------------------------------


@pytest.mark.asyncio
async def test_single_turn_drops_flagged_generations() -> None:
    from stateset_agents.training.single_turn_trainer import SingleTurnGRPOTrainer

    replies = iter(["ok", "FALLBACK", "fine"])

    class Agent:
        model = None
        tokenizer = None

        async def generate_response(self, prompt: Any) -> str:
            return next(replies)

    trainer = object.__new__(SingleTurnGRPOTrainer)
    trainer.agent = Agent()
    trainer.reward_fn = FlagsSome()
    trainer.environment = None
    group, _best = await trainer._generate_trajectory_group("q", 3)
    assert [t.turns[-1].content for t in group.trajectories] == ["ok", "fine"]
    assert trainer.reward_fallback_episodes == 1

    trainer.agent = type("All", (), {"generate_response": _always_fallback})()
    with pytest.raises(RewardUnavailableError, match="reward fallback"):
        await trainer._generate_trajectory_group("q", 2)


async def _always_fallback(self: Any, prompt: Any) -> str:
    return "FALLBACK"


# --- prompt/response trainers ------------------------------------------------


def _scripted(texts: list[str], rollouts: Any) -> Any:
    """Rollouts whose response texts are ``texts`` (token ids unchanged)."""

    def make(prompt: str, *args: Any) -> list[dict[str, Any]]:
        out = rollouts(prompt, *args)
        for item, text in zip(out, texts, strict=True):
            item["response"] = text
        return out

    return make


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["dapo", "gepo", "vapo"])
async def test_callable_trainers_drop_flagged_replies(name: str) -> None:
    from tests.unit.test_dedicated_trainer_nonfinite_grads import (
        _dapo,
        _gepo,
        _gepo_rollouts,
        _token_rollouts,
        _vapo,
    )

    build = {"dapo": _dapo, "gepo": _gepo, "vapo": _vapo}[name]
    trainer = build(num_gradient_updates=1)
    trainer.reward_fn = flagging_callable()
    if name == "dapo":
        trainer.config.use_dynamic_sampling = False
    rollouts = _gepo_rollouts if name == "gepo" else _token_rollouts
    make = _scripted(["12a", "FALLBACK", "9z"], rollouts)
    seen: list[int] = []

    async def generate(prompt: str, *args: Any) -> list[dict[str, Any]]:
        group = make(prompt, *args)
        seen.append(len(group))
        return group

    trainer.generate_group_responses = generate
    metrics = await trainer.train_step(["q"])
    assert metrics["reward_fallback_responses"] == 1
    assert metrics["optimizer_updates"] == 1.0
    assert metrics["average_reward"] == pytest.approx((2 / 3 + 0.5) / 2)

    all_flagged = _scripted(["FALLBACK"] * 3, rollouts)

    async def generate_flagged(prompt: str, *args: Any) -> list[dict[str, Any]]:
        return all_flagged(prompt, *args)

    trainer.generate_group_responses = generate_flagged
    with pytest.raises(RewardUnavailableError, match="reward fallback"):
        await trainer.train_step(["q"])


@pytest.mark.asyncio
async def test_gspo_drops_flagged_replies() -> None:
    from tests.unit.test_dedicated_trainer_nonfinite_grads import (
        _gspo,
        _gspo_classes,
    )

    trainer = _gspo(_gspo_classes()[0], num_iterations=1)
    trainer.reward_model = FlagsSome(flag="FALLBACK")

    async def generate(prompt: str, count: int) -> list[tuple[str, float]]:
        return [("12a", -5.0), ("FALLBACK", -5.0), ("9z", -5.0)][:count]

    trainer.generator.generate_group_responses = generate
    trainer.config.num_generations = 3
    metrics = await trainer.train_step(["q"], num_groups=1)
    assert metrics["reward_fallback_responses"] == 1.0
    assert metrics["advantage_group_count"] == 1.0

    async def flagged(prompt: str, count: int) -> list[tuple[str, float]]:
        return [("FALLBACK", -5.0)] * count

    trainer.generator.generate_group_responses = flagged
    with pytest.raises(RewardUnavailableError, match="reward fallback"):
        await trainer.train_step(["q"], num_groups=1)
