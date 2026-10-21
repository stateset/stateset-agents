"""A reward function without reward_type is applied exactly once."""

from stateset_agents.core.conversation_environment import ConversationEnvironment
from stateset_agents.core.task_environment import TaskEnvironment


class _Untyped:
    async def compute_reward(self, turns, context=None):
        return 1.0


def test_step_scoring_environment_skips_final_reward():
    env = ConversationEnvironment(scenarios=[{"id": "s"}], reward_fn=_Untyped())
    assert env._should_compute_step_reward()
    assert not env._should_compute_final_reward()


def test_other_environments_score_untyped_rewards_at_episode_end():
    env = TaskEnvironment(
        tasks=[{"id": "t"}],
        success_criteria=lambda turns, ctx: True,
        reward_fn=_Untyped(),
    )
    assert env._should_compute_final_reward()
