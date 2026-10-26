"""A failing component never counts as a zero score in CompositeReward: by
default the composite is unavailable too; with the opt-in
``outage_policy = "renormalize"`` its weight is spread over the survivors."""

import pytest

from stateset_agents.core.reward_base import (
    CompositeReward,
    RewardFunction,
    RewardResult,
    RewardUnavailableError,
)


class _Fixed(RewardFunction):
    def __init__(self, score, weight=0.5, fail=False, name="fixed"):
        super().__init__(weight=weight, name=name)
        self.score, self.fail = score, fail

    async def compute_reward(self, turns, context=None):
        if self.fail:
            raise RewardUnavailableError("boom")
        return RewardResult(score=self.score, breakdown={})


@pytest.mark.asyncio
async def test_failed_component_weight_is_redistributed():
    healthy = CompositeReward([_Fixed(1.0, name="a"), _Fixed(1.0, name="b")])
    degraded = CompositeReward(
        [_Fixed(1.0, name="a"), _Fixed(1.0, fail=True, name="b")]
    )
    degraded.outage_policy = "renormalize"
    assert (await healthy.compute_reward([])).score == pytest.approx(1.0)
    result = await degraded.compute_reward([])
    assert result.score == pytest.approx(1.0)
    assert result.metadata["failed_components"] == ["b"]


@pytest.mark.asyncio
async def test_unequal_weights_keep_configured_scale():
    reward = CompositeReward(
        [
            _Fixed(0.4, weight=2.0, name="a"),
            _Fixed(1.0, weight=1.0, fail=True, name="b"),
        ]
    )
    reward.outage_policy = "renormalize"
    # survivor carries the full configured weight of 3.0
    assert (await reward.compute_reward([])).score == pytest.approx(1.2)


# --- an outage drops the episode unless renormalizing is opted into (round 5) --

import asyncio  # noqa: E402

from stateset_agents.core.conversation_environment import (  # noqa: E402
    ConversationEnvironment,
)
from stateset_agents.core.errors import ConfigurationError  # noqa: E402
from stateset_agents.core.reward_base import RewardType  # noqa: E402
from stateset_agents.core.reward_context import (  # noqa: E402
    REWARD_FALLBACK_KEY,
    score_episode,
)
from stateset_agents.core.trajectory import ConversationTurn  # noqa: E402


class _Judge(RewardFunction):
    def __init__(self, up: bool) -> None:
        super().__init__(0.8, RewardType.CUMULATIVE, "judge")
        self.up = up

    async def compute_reward(self, turns, context=None):
        if not self.up:
            raise RewardUnavailableError("judge 503")
        return RewardResult(score=0.2)


def _judged(up: bool) -> CompositeReward:
    return CompositeReward([_Judge(up), _Fixed(1.0, weight=0.2, name="concise")])


TURNS = [
    ConversationTurn(role="user", content="Explain refunds."),
    ConversationTurn(role="assistant", content="Bad answer."),
]


@pytest.mark.asyncio
async def test_a_component_outage_makes_the_composite_unavailable():
    """With the judge up "Bad answer." scores 0.36; with it down the
    conciseness term alone used to stand in for the whole reward (1.0)."""
    up = await _judged(True).compute_reward(TURNS)
    assert up.score == pytest.approx(0.36)
    with pytest.raises(RewardUnavailableError, match="judge"):
        await _judged(False).compute_reward(TURNS)


@pytest.mark.parametrize("attach", [True, False])
def test_an_outage_drops_the_episode_on_both_paths(attach):
    scenario = {"id": "s", "user_query": "Explain refunds."}
    reward = _judged(False)
    env = ConversationEnvironment(
        [scenario], max_turns=1, reward_fn=reward if attach else None
    )

    async def agent(history, context):
        return "Bad answer."

    trajectory = asyncio.run(env.run_episode(agent, scenario))
    if attach:
        assert "judge 503" in trajectory.metadata["reward_error"]
    with pytest.raises(RewardUnavailableError):
        asyncio.run(score_episode(trajectory, reward, env, scenario))


@pytest.mark.parametrize("attach", [True, False])
def test_renormalizing_is_an_opt_in_recorded_on_the_result_and_episode(attach):
    reward = _judged(False)
    reward.outage_policy = "renormalize"
    result = asyncio.run(reward.compute_reward(TURNS))
    assert result.score == pytest.approx(1.0)
    assert result.metadata["failed_components"] == ["judge"]
    assert result.breakdown["unavailable_components"] == 1.0
    assert "judge" in result.metadata[REWARD_FALLBACK_KEY]

    scenario = {"id": "s", "user_query": "Explain refunds."}
    env = ConversationEnvironment(
        [scenario], max_turns=1, reward_fn=reward if attach else None
    )

    async def agent(history, context):
        return "Bad answer."

    trajectory = asyncio.run(env.run_episode(agent, scenario))
    asyncio.run(score_episode(trajectory, reward, env, scenario))
    assert trajectory.metadata["reward_failed_components"] == ["judge"]
    assert "judge" in trajectory.metadata[REWARD_FALLBACK_KEY]


@pytest.mark.asyncio
async def test_an_unknown_outage_policy_is_a_configuration_error():
    reward = _judged(False)
    reward.outage_policy = "ignore"
    with pytest.raises(ConfigurationError, match="outage_policy"):
        await reward.compute_reward(TURNS)


def test_the_sync_wrapper_uses_typed_errors_not_message_text():
    from stateset_agents.core.reward_base import SyncRewardCallError

    reward = _Fixed(0.5)
    assert reward([]).score == 0.5

    async def inside_a_loop():
        return reward([])

    with pytest.raises(SyncRewardCallError):
        asyncio.run(inside_a_loop())
    assert issubclass(SyncRewardCallError, RuntimeError)

    class Raises(RewardFunction):
        async def compute_reward(self, turns, context=None):
            raise RuntimeError("no running event loop")

    # A reward's own error propagates as it is, whatever its message says.
    with pytest.raises(RuntimeError, match="no running event loop"):
        Raises()([])


def test_the_sync_wrapper_does_not_depend_on_the_loop_probe_message(monkeypatch):
    """Whatever text ``get_running_loop`` uses, no running loop means run."""

    def no_loop():
        raise RuntimeError("some other wording entirely")

    monkeypatch.setattr(asyncio, "get_running_loop", no_loop)
    assert _Fixed(0.25)([]).score == 0.25


def test_a_reward_error_from_the_sync_wrapper_has_no_chained_context():
    """``asyncio.run`` runs outside any handler, so a reward's own error is not
    chained to the loop probe's RuntimeError."""

    class Raises(RewardFunction):
        async def compute_reward(self, turns, context=None):
            raise ValueError("bad data")

    with pytest.raises(ValueError, match="bad data") as caught:
        Raises()([])
    assert caught.value.__context__ is None


# --- a child's score is range-checked even when it declares no range (round 6) --

from stateset_agents.core.reward_context import (  # noqa: E402
    UNIT_SCORE_RANGE,
    check_child_score_range,
    score_response,
)


class _Wide(_Fixed):
    score_range = (0.0, 10.0)


@pytest.mark.parametrize("score", [7.0, -0.5, 1.0 + 1e-6])
async def test_an_undeclared_child_outside_the_unit_range_raises(score):
    # Before round 6 the 7.0 was weighted into the total (3.5 + 0.5 = 4.0).
    composite = CompositeReward([_Fixed(score, name="wild"), _Fixed(1.0)])
    with pytest.raises(ValueError, match="wild scored .* declares no score_range"):
        await composite.compute_reward([])


@pytest.mark.parametrize("score", [0.0, 0.5, 1.0, 1.0 + 1e-12])
async def test_an_undeclared_child_inside_the_unit_range_is_summed(score):
    composite = CompositeReward([_Fixed(score), _Fixed(1.0)])
    assert (await composite.compute_reward([])).score == pytest.approx(
        0.5 * score + 0.5
    )


async def test_a_declared_child_is_held_to_its_own_range():
    composite = CompositeReward([_Wide(7.0, name="wide"), _Fixed(1.0)])
    assert (await composite.compute_reward([])).score == pytest.approx(4.0)
    over = CompositeReward([_Wide(11.0, name="wide"), _Fixed(1.0)])
    with pytest.raises(ValueError, match="outside its declared score_range"):
        await over.compute_reward([])


async def test_a_nested_composite_spans_its_weights():
    # Weights summing to 2 make the inner composite span [0, 2]; its own
    # children are checked, so the outer composite accepts its 2.0.
    inner = CompositeReward([_Fixed(1.0, weight=1.0), _Fixed(1.0, weight=1.0)])
    outer = CompositeReward([inner])
    assert (await outer.compute_reward([])).score == pytest.approx(2.0)
    bad_inner = CompositeReward([_Fixed(7.0, name="wild")])
    with pytest.raises(ValueError, match="wild scored"):
        await CompositeReward([bad_inner]).compute_reward([])


async def test_the_top_level_still_leaves_undeclared_rewards_unchecked():
    # The [0, 1] default holds composite children only (docs/REWARD_CONTRACT.md).
    scenario = {"id": "s", "user_query": "hi"}
    assert await score_response(_Fixed(7.0), "hi", "x", scenario) == 7.0


def test_the_child_check_helper():
    assert UNIT_SCORE_RANGE == (0.0, 1.0)
    assert check_child_score_range(_Fixed(0.3), 0.3) == 0.3
    assert check_child_score_range(_Fixed(7.0), 7.0, default=None) == 7.0
    assert check_child_score_range(_Wide(7.0), 7.0) == 7.0
    assert check_child_score_range(_Fixed(1.5), 1.5, default=(0.0, 2.0)) == 1.5
    with pytest.raises(ValueError):
        check_child_score_range(_Fixed(7.0), 7.0)
    with pytest.raises(ValueError, match="declared"):
        check_child_score_range(_Wide(11.0), 11.0, default=None)
