"""
Regression tests for the legacy ``grpo_agent_framework`` compatibility shims.
"""

import importlib


def test_grpo_core_shim_reexports_core_symbols():
    legacy_core = importlib.import_module("grpo_agent_framework.core")

    from grpo_agent_framework.core import (
        CompositeReward,
        ConversationTurn,
        Environment,
        MultiTurnTrajectory,
        RewardFunction,
        Trajectory,
    )
    from stateset_agents.core import (
        CompositeReward as canonical_composite,
        ConversationTurn as canonical_turn,
        Environment as canonical_env,
        MultiTurnTrajectory as canonical_multi_turn,
        RewardFunction as canonical_reward,
        Trajectory as canonical_traj,
    )

    assert Environment is canonical_env
    assert RewardFunction is canonical_reward
    assert CompositeReward is canonical_composite
    assert Trajectory is canonical_traj
    assert MultiTurnTrajectory is canonical_multi_turn
    assert ConversationTurn is canonical_turn
    assert legacy_core.Environment is canonical_env
