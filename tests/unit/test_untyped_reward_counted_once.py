"""A reward function without reward_type is applied exactly once."""

from stateset_agents.core.conversation_environment import ConversationEnvironment
from stateset_agents.core.task_environment import TaskEnvironment


class _Untyped:
    async def compute_reward(self, turns, context=None):
        return 1.0


def test_conversation_environment_scores_untyped_rewards_at_episode_end():
    # Untyped rewards are episode-level, as when a trainer holds them, so the
    # total does not depend on where the reward is attached.
    env = ConversationEnvironment(scenarios=[{"id": "s"}], reward_fn=_Untyped())
    assert not env._should_compute_step_reward()
    assert env._should_compute_final_reward()


def test_other_environments_score_untyped_rewards_at_episode_end():
    env = TaskEnvironment(
        tasks=[{"id": "t"}],
        success_criteria=lambda turns, ctx: True,
        reward_fn=_Untyped(),
    )
    assert env._should_compute_final_reward()


def test_a_step_scoring_subclass_scores_untyped_rewards_per_step():
    class StepScoring(ConversationEnvironment):
        scores_steps = True

    env = StepScoring(scenarios=[{"id": "s"}], reward_fn=_Untyped())
    assert env._should_compute_step_reward()
    assert not env._should_compute_final_reward()
