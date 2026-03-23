"""Tests for the multimodal conversation environment."""

import pytest

from stateset_agents.core.environments.multimodal_environment import (
    MultimodalConversationEnvironment,
    MultimodalScenario,
    _create_text_message,
)


class TestMultimodalConversationEnvironment:
    def test_requires_at_least_one_scenario(self):
        with pytest.raises(ValueError, match="at least one scenario"):
            MultimodalConversationEnvironment([])

    @pytest.mark.asyncio
    async def test_step_returns_none_when_episode_is_done(self):
        env = MultimodalConversationEnvironment(
            [
                MultimodalScenario(
                    scenario_id="scenario-1",
                    task_description="Describe the image.",
                    user_input=_create_text_message("What is in this image?"),
                    expected_response="A cat on a sofa.",
                )
            ],
            max_turns=1,
        )

        initial_message = await env.reset()
        next_message, reward, done, info = await env.step("A cat on a sofa.")

        assert initial_message.role == "user"
        assert next_message is None
        assert done is True
        assert isinstance(reward, float)
        assert info["current_turn"] == 1
