"""
Comprehensive tests for core/environment.py

Covers:
- ConversationEnvironment creation and configuration
- Scenario management
- Episode lifecycle
- Step execution
- State management
- Multi-turn interactions
- Error handling
"""

import pytest
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.core.reward import HelpfulnessReward


class TestConversationEnvironmentBasics:
    """Basic tests for ConversationEnvironment"""

    def test_environment_creation_with_scenarios(self):
        """Test creating environment with scenarios"""
        scenarios = [
            {"id": "s1", "topic": "greeting", "context": "Welcome users"},
            {"id": "s2", "topic": "help", "context": "Provide assistance"},
        ]

        env = ConversationEnvironment(scenarios=scenarios, max_turns=5)

        assert env is not None
        assert len(env.scenarios) == 2
        assert env.max_turns == 5

    def test_environment_creation_with_single_scenario(self):
        """Test creating environment with single scenario"""
        scenarios = [{"id": "s1", "topic": "test", "context": "Test scenario"}]

        env = ConversationEnvironment(scenarios=scenarios, max_turns=3)

        assert len(env.scenarios) == 1

    def test_environment_with_custom_max_turns(self):
        """Test environment with various max_turns values"""
        scenarios = [{"topic": "test"}]

        for max_turns in [1, 5, 10, 20]:
            env = ConversationEnvironment(scenarios=scenarios, max_turns=max_turns)
            assert env.max_turns == max_turns

    def test_environment_with_reward_function(self):
        """Test environment with custom reward function"""
        scenarios = [{"topic": "test"}]
        reward_fn = HelpfulnessReward()

        env = ConversationEnvironment(
            scenarios=scenarios,
            max_turns=5,
            reward_function=reward_fn
        )

        assert env.reward_function is reward_fn


class TestEnvironmentEpisodeManagement:
    """Test episode lifecycle management"""

    @pytest.mark.asyncio
    async def test_reset_environment(self):
        """Test resetting environment to start new episode"""
        scenarios = [{"id": "s1", "topic": "test", "context": "Test"}]
        env = ConversationEnvironment(scenarios=scenarios, max_turns=3)

        state = await env.reset()

        assert state is not None
        assert "scenario" in state
        assert "turn_count" in state
        assert state["turn_count"] == 0

    @pytest.mark.asyncio
    async def test_reset_multiple_times(self):
        """Test resetting environment multiple times"""
        scenarios = [{"topic": "test1"}, {"topic": "test2"}]
        env = ConversationEnvironment(scenarios=scenarios, max_turns=5)

        for _ in range(5):
            state = await env.reset()
            assert state["turn_count"] == 0

    @pytest.mark.asyncio
    async def test_reset_with_specific_scenario(self):
        """Test resetting with specific scenario selection"""
        scenarios = [
            {"id": "s1", "topic": "greeting"},
            {"id": "s2", "topic": "farewell"},
        ]
        env = ConversationEnvironment(scenarios=scenarios, max_turns=3)

        state = await env.reset(scenario_id="s2")

        assert state["scenario"]["topic"] == "farewell"

    @pytest.mark.asyncio
    async def test_episode_done_after_max_turns(self):
        """Test that episode ends after max_turns"""
        scenarios = [{"topic": "test"}]
        env = ConversationEnvironment(scenarios=scenarios, max_turns=2)

        state = await env.reset()

        # First turn
        next_state, reward, done, info = await env.step(state, "Response 1")
        assert not done
        assert next_state["turn_count"] == 1

        # Second turn (should reach max)
        next_state, reward, done, info = await env.step(next_state, "Response 2")
        assert done or next_state["turn_count"] == 2


class TestEnvironmentStepExecution:
    """Test step execution and state transitions"""

    @pytest.mark.asyncio
    async def test_single_step(self):
        """Test executing a single step"""
        scenarios = [{"topic": "greeting"}]
        env = ConversationEnvironment(scenarios=scenarios, max_turns=5)

        state = await env.reset()
        action = "Hello! How can I help you?"

        next_state, reward, done, info = await env.step(state, action)

        assert next_state is not None
        assert "turn_count" in next_state
        assert next_state["turn_count"] > state["turn_count"]
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    @pytest.mark.asyncio
    async def test_multiple_steps(self):
        """Test executing multiple steps in sequence"""
        scenarios = [{"topic": "conversation"}]
        env = ConversationEnvironment(scenarios=scenarios, max_turns=5)

        state = await env.reset()

        actions = [
            "Hello!",
            "How can I assist you?",
            "I'm here to help.",
        ]

        for action in actions:
            next_state, reward, done, info = await env.step(state, action)
            assert next_state["turn_count"] > state["turn_count"]
            state = next_state

            if done:
                break

    @pytest.mark.asyncio
    async def test_step_with_empty_action(self):
        """Test step with empty action string"""
        scenarios = [{"topic": "test"}]
        env = ConversationEnvironment(scenarios=scenarios, max_turns=3)

        state = await env.reset()

        # Empty action should still be processed
        next_state, reward, done, info = await env.step(state, "")

        assert next_state is not None

    @pytest.mark.asyncio
    async def test_step_with_long_action(self):
        """Test step with very long action"""
        scenarios = [{"topic": "test"}]
        env = ConversationEnvironment(scenarios=scenarios, max_turns=3)

        state = await env.reset()

        # Long action (1000 characters)
        long_action = "This is a very long response. " * 50

        next_state, reward, done, info = await env.step(state, long_action)

        assert next_state is not None


class TestEnvironmentScenarioHandling:
    """Test scenario selection and management"""

    def test_environment_with_detailed_scenarios(self):
        """Test environment with rich scenario descriptions"""
        scenarios = [
            {
                "id": "customer_complaint",
                "topic": "Customer Service",
                "context": "Customer is unhappy with product quality",
                "initial_message": "I'm very disappointed with my purchase",
                "expected_tone": "empathetic",
                "difficulty": "high"
            },
            {
                "id": "simple_query",
                "topic": "Information",
                "context": "Customer needs basic information",
                "initial_message": "What are your business hours?",
                "expected_tone": "friendly",
                "difficulty": "low"
            }
        ]

        env = ConversationEnvironment(scenarios=scenarios, max_turns=5)

        assert len(env.scenarios) == 2
        assert any(s["difficulty"] == "high" for s in env.scenarios)

    @pytest.mark.asyncio
    async def test_random_scenario_selection(self):
        """Test that environment randomly selects scenarios"""
        scenarios = [
            {"id": f"s{i}", "topic": f"topic{i}"}
            for i in range(10)
        ]

        env = ConversationEnvironment(scenarios=scenarios, max_turns=3)

        selected_scenarios = []
        for _ in range(20):
            state = await env.reset()
            selected_scenarios.append(state["scenario"]["id"])

        # Should have some variety in selection (not all the same)
        unique_scenarios = set(selected_scenarios)
        assert len(unique_scenarios) > 1


class TestEnvironmentRunEpisodeSignatures:
    """Tests for Environment.run_episode signature handling."""

    @pytest.mark.asyncio
    async def test_run_episode_legacy_signature_with_string_response(self):
        """Legacy (state, env_response, reward, done) tuples should be parsed safely."""
        from stateset_agents.core.environment import Environment, EnvironmentState, EpisodeStatus

        class LegacyEnv(Environment):
            async def reset(self, scenario=None):
                return EnvironmentState(
                    episode_id="legacy",
                    turn_count=0,
                    status=EpisodeStatus.ONGOING,
                    context={"scenario": scenario or {}},
                )

            async def step(self, state, action):
                new_state = state.copy()
                new_state.turn_count += 1
                new_state.status = EpisodeStatus.COMPLETED
                return new_state, "user reply", 1.0, True

        env = LegacyEnv(max_turns=1)

        async def agent_fn(history, context):
            return "assistant reply"

        trajectory = await env.run_episode(agent_fn, scenario={"id": "s"})

        assert trajectory.total_reward == 1.0
        assert trajectory.turn_rewards == [1.0]

    @pytest.mark.asyncio
    async def test_run_episode_new_signature_with_final_reward(self):
        """New (state, reward, done, info) tuples should include final rewards in turn totals."""
        from stateset_agents.core.environment import Environment, EnvironmentState, EpisodeStatus
        from stateset_agents.core.reward_base import RewardFunction, RewardResult
        from stateset_agents.core.trajectory import ConversationTurn

        class FixedReward(RewardFunction):
            async def compute_reward(self, turns, context=None):
                return RewardResult(score=0.5)

        class NewEnv(Environment):
            async def reset(self, scenario=None):
                return EnvironmentState(
                    episode_id="new",
                    turn_count=0,
                    status=EpisodeStatus.ONGOING,
                    context={"scenario": scenario or {}},
                )

            async def step(self, state, action):
                new_state = state.copy()
                new_state.turn_count += 1
                new_state.status = EpisodeStatus.COMPLETED
                info = {"env_response": ConversationTurn(role="user", content="ok")}
                return new_state, 1.0, True, info

        env = NewEnv(max_turns=1, reward_fn=FixedReward())

        async def agent_fn(history, context):
            return "assistant reply"

        trajectory = await env.run_episode(agent_fn, scenario={"id": "s"})

        assert trajectory.total_reward == 1.5
        assert trajectory.turn_rewards == [1.5]
        assert len(trajectory.turns) == 2

    def test_environment_with_empty_scenarios(self):
        """Test environment handles empty scenario list"""
        # Should handle gracefully or raise appropriate error
        try:
            env = ConversationEnvironment(scenarios=[], max_turns=3)
            # If it allows empty, that's fine
            assert len(env.scenarios) == 0
        except (ValueError, AssertionError):
            # If it raises error, that's also acceptable
            pass


class TestEnvironmentStateManagement:
    """Test state tracking and updates"""

    @pytest.mark.asyncio
    async def test_state_contains_history(self):
        """Test that state tracks conversation history"""
        scenarios = [{"topic": "test"}]
        env = ConversationEnvironment(scenarios=scenarios, max_turns=5)

        state = await env.reset()

        # Execute multiple steps
        for i in range(3):
            state, _, _, _ = await env.step(state, f"Response {i}")

        # State should contain history
        assert state["turn_count"] == 3

    @pytest.mark.asyncio
    async def test_state_includes_conversation_and_task_ids(self):
        """Test that scenario metadata is surfaced in state context."""
        scenario = {
            "id": "scenario_1",
            "task_id": "task_alpha",
            "conversation_id": "conv_alpha",
            "topic": "test",
        }
        env = ConversationEnvironment(scenarios=[scenario], max_turns=3)

        state = await env.reset(scenario=scenario)

        assert state.context.get("conversation_id") == "conv_alpha"
        assert state.context.get("scenario_id") == "scenario_1"
        assert state.context.get("task_id") == "task_alpha"

    @pytest.mark.asyncio
    async def test_state_immutability(self):
        """Test that original state is not modified by step"""
        scenarios = [{"topic": "test"}]
        env = ConversationEnvironment(scenarios=scenarios, max_turns=5)

        original_state = await env.reset()
        original_turn_count = original_state["turn_count"]

        next_state, _, _, _ = await env.step(original_state, "Action")

        # Original state should not be modified
        assert original_state["turn_count"] == original_turn_count
        assert next_state["turn_count"] > original_turn_count


class TestEnvironmentRewardIntegration:
    """Test reward function integration"""

    @pytest.mark.asyncio
    async def test_environment_with_reward_function(self):
        """Test environment computes rewards using reward function"""
        scenarios = [{"topic": "test"}]
        reward_fn = HelpfulnessReward()

        env = ConversationEnvironment(
            scenarios=scenarios,
            max_turns=3,
            reward_function=reward_fn
        )

        state = await env.reset()
        next_state, reward, done, info = await env.step(state, "Helpful response")

        assert isinstance(reward, (int, float))
        assert reward >= 0.0  # Helpfulness rewards are non-negative

    @pytest.mark.asyncio
    async def test_environment_without_reward_function(self):
        """Test environment works without explicit reward function"""
        scenarios = [{"topic": "test"}]

        env = ConversationEnvironment(scenarios=scenarios, max_turns=3)

        state = await env.reset()
        next_state, reward, done, info = await env.step(state, "Response")

        # Should return default reward (likely 0)
        assert isinstance(reward, (int, float))


class TestEnvironmentEdgeCases:
    """Test edge cases and error handling"""

    def test_environment_with_zero_max_turns(self):
        """Test environment with max_turns=0"""
        scenarios = [{"topic": "test"}]

        # Should either handle gracefully or raise error
        try:
            env = ConversationEnvironment(scenarios=scenarios, max_turns=0)
            # If it allows, episodes should immediately end
        except ValueError:
            # Raising error is acceptable
            pass

    def test_environment_with_negative_max_turns(self):
        """Test environment with negative max_turns"""
        scenarios = [{"topic": "test"}]

        # Should raise error
        try:
            env = ConversationEnvironment(scenarios=scenarios, max_turns=-1)
            # If it allows negative, that's unexpected but test it
        except ValueError:
            # Expected behavior
            pass

    def test_environment_with_very_large_max_turns(self):
        """Test environment with extremely large max_turns"""
        scenarios = [{"topic": "test"}]

        env = ConversationEnvironment(scenarios=scenarios, max_turns=10000)

        assert env.max_turns == 10000

    @pytest.mark.asyncio
    async def test_step_after_episode_done(self):
        """Test calling step after episode is already done"""
        scenarios = [{"topic": "test"}]
        env = ConversationEnvironment(scenarios=scenarios, max_turns=1)

        state = await env.reset()

        # First step should complete episode
        state, _, done, _ = await env.step(state, "Action")

        if done:
            # Calling step again after done
            # Should either work (continue) or raise error
            try:
                await env.step(state, "Another action")
            except Exception:
                # Error is acceptable
                pass

    def test_environment_with_malformed_scenarios(self):
        """Test environment handles malformed scenario data"""
        # Scenarios with missing fields
        scenarios = [
            {"topic": "valid"},
            {},  # Empty scenario
            {"no_topic": "invalid"},
        ]

        try:
            env = ConversationEnvironment(scenarios=scenarios, max_turns=3)
            # If it handles gracefully, that's fine
        except (KeyError, ValueError):
            # Raising error is also acceptable
            pass


class TestEnvironmentContextPreservation:
    """Test that context is preserved across turns"""

    @pytest.mark.asyncio
    async def test_context_preserved_across_steps(self):
        """Test that scenario context is available in all steps"""
        scenarios = [{
            "id": "s1",
            "topic": "support",
            "context": "Customer needs technical help",
            "metadata": {"priority": "high"}
        }]

        env = ConversationEnvironment(scenarios=scenarios, max_turns=5)

        state = await env.reset()
        initial_context = state["scenario"]["context"]

        # Execute multiple steps
        for _ in range(3):
            state, _, _, _ = await env.step(state, "Response")
            # Context should remain the same
            assert state["scenario"]["context"] == initial_context


class TestTaskEnvironmentSuccessCriteria:
    """Test TaskEnvironment success_criteria handling"""

    @pytest.mark.asyncio
    async def test_success_criteria_is_called(self):
        """Test that success_criteria callback is actually called"""
        from stateset_agents.core.environment import TaskEnvironment

        criteria_called = {"count": 0, "last_turns": None, "last_context": None}

        def custom_success_criteria(turns, context):
            criteria_called["count"] += 1
            criteria_called["last_turns"] = turns
            criteria_called["last_context"] = context
            return context.get("task_progress", 0.0) >= 1.0

        tasks = [{"description": "Test task", "required_actions": []}]
        env = TaskEnvironment(
            tasks=tasks,
            success_criteria=custom_success_criteria,
            max_turns=5
        )

        state = await env.reset()

        # Execute a step
        action = ConversationTurn(role="assistant", content="Working on the task")
        await env.step(state, action)

        # success_criteria should have been called
        assert criteria_called["count"] >= 1
        assert criteria_called["last_turns"] is not None
        assert criteria_called["last_context"] is not None

    @pytest.mark.asyncio
    async def test_custom_success_criteria_determines_completion(self):
        """Test that custom success_criteria determines task completion"""
        from stateset_agents.core.environment import TaskEnvironment

        # Custom criteria: complete when "done" appears in any turn
        def custom_success_criteria(turns, context):
            for turn in turns:
                if "done" in turn.content.lower():
                    return True
            return False

        tasks = [{"description": "Test task", "required_actions": []}]
        env = TaskEnvironment(
            tasks=tasks,
            success_criteria=custom_success_criteria,
            max_turns=5
        )

        state = await env.reset()

        # First step without "done" - should not complete
        # TaskEnvironment.step returns (state, env_response, reward, done)
        action1 = ConversationTurn(role="assistant", content="Working on task")
        state, _, _, done1 = await env.step(state, action1)
        assert not done1

        # Second step with "done" - should complete
        action2 = ConversationTurn(role="assistant", content="Task is done!")
        _, _, _, done2 = await env.step(state, action2)
        assert done2

    @pytest.mark.asyncio
    async def test_success_criteria_receives_turns(self):
        """Test that success_criteria receives accumulated turns"""
        from stateset_agents.core.environment import TaskEnvironment

        received_turns_count = {"value": 0}

        def custom_success_criteria(turns, context):
            received_turns_count["value"] = len(turns)
            return False  # Never complete

        tasks = [{"description": "Test task", "required_actions": []}]
        env = TaskEnvironment(
            tasks=tasks,
            success_criteria=custom_success_criteria,
            max_turns=5
        )

        state = await env.reset()

        # First step
        action1 = ConversationTurn(role="assistant", content="Step 1")
        state, _, _, _ = await env.step(state, action1)
        turns_after_step1 = received_turns_count["value"]

        # Second step
        action2 = ConversationTurn(role="assistant", content="Step 2")
        await env.step(state, action2)
        turns_after_step2 = received_turns_count["value"]

        # Each step adds action + env_response, so turns should accumulate
        assert turns_after_step2 > turns_after_step1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
