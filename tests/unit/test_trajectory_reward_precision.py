"""Turn rewards must accumulate without rounding."""

import pytest

from stateset_agents.core.trajectory import ConversationTurn, MultiTurnTrajectory


def test_small_turn_rewards_are_not_rounded_away():
    traj = MultiTurnTrajectory()
    for _ in range(3):
        traj.add_turn(ConversationTurn(role="assistant", content="x"), reward=0.04)
    assert traj.total_reward == pytest.approx(0.12)
    traj.add_turn(ConversationTurn(role="assistant", content="y"), reward=0.26)
    assert traj.total_reward == pytest.approx(0.38)


def test_decimal_sums_stay_exact():
    traj = MultiTurnTrajectory()
    traj.add_turn(ConversationTurn(role="assistant", content="a"), reward=0.1)
    traj.add_turn(ConversationTurn(role="assistant", content="b"), reward=0.2)
    assert traj.total_reward == 0.3
