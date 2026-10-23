"""Batch rollout integration keeps sandbox observations and evaluator state apart."""

import copy
import unittest
from types import SimpleNamespace

import pytest

from stateset_agents.core.environment_base import (
    Environment,
    EnvironmentState,
    EpisodeStatus,
)
from stateset_agents.core.trajectory import ConversationTurn

pytestmark = pytest.mark.integration


class Sandbox(Environment):
    supports_concurrent_episodes = True

    def __init__(self, *, legacy=False, reward=None):
        super().__init__(max_turns=2, reward_fn=reward)
        self.legacy = legacy
        self.states = {}

    async def reset(self, scenario=None):
        state = EnvironmentState(
            episode_id=scenario["record_id"],
            turn_count=0,
            status=EpisodeStatus.ONGOING,
            context={
                "scenario": copy.deepcopy(scenario),
                "expected_response": "private-answer",
                "expected_tool_calls": ["private-tool"],
                "tool_trace": [],
                "messages": [{"role": "user", "content": scenario["record_id"]}],
            },
        )
        self.states[state.episode_id] = state
        return state

    async def step(self, state, action):
        expected = [] if state.turn_count == 0 else ["lookup"]
        if state.context["tool_trace"] != expected:
            raise AssertionError("policy modified the trusted tool trace")
        state.context["tool_trace"].append(action.content)
        state.turn_count += 1
        done = state.turn_count == 2
        if done:
            state.status = EpisodeStatus.COMPLETED
        observation = ConversationTurn(role="tool", content="facts-" + state.episode_id)
        if self.legacy:
            return state, None if done else observation, float(done), done
        return (
            state,
            float(done),
            done,
            {
                "messages": (
                    []
                    if done
                    else [{"role": observation.role, "content": observation.content}]
                )
            },
        )


class EnvironmentBatchIsolation(unittest.IsolatedAsyncioTestCase):
    async def test_single_and_batched_rollouts_preserve_privacy_and_observations(self):
        scenarios = [
            {
                "record_id": identity,
                "expected_response": "private-answer",
                "expected_tool_calls": ["private-tool"],
                "public": {"nested": []},
            }
            for identity in ("case-a", "case-b")
        ]
        for legacy in (False, True):
            for batched in (False, True):
                with self.subTest(legacy=legacy, batched=batched):
                    original = copy.deepcopy(scenarios)
                    env = Sandbox(legacy=legacy)
                    batch_sizes = []

                    async def replies(histories, contexts, observed=batch_sizes):
                        observed.append(len(histories))
                        result = []
                        for history, context in zip(histories, contexts, strict=True):
                            identity = context["scenario"]["record_id"]
                            for value in (context, context["scenario"]):
                                self.assertNotIn("expected_response", value)
                                self.assertNotIn("expected_tool_calls", value)
                            self.assertEqual(history[0]["content"], identity)
                            finished_lookup = len(history) > 1
                            if finished_lookup:
                                self.assertEqual(
                                    history[-1],
                                    {"role": "tool", "content": "facts-" + identity},
                                )
                            context["tool_trace"].append("forged")
                            context["scenario"]["public"]["nested"].append("forged")
                            history[0]["content"] = "forged"
                            result.append("finish" if finished_lookup else "lookup")
                        return result

                    if batched:
                        trajectories = await env.run_episodes_batched(
                            replies, scenarios
                        )
                        self.assertEqual(batch_sizes, [2, 2])
                    else:

                        async def single(history, context):
                            return (await replies([history], [context]))[0]

                        trajectories = [
                            await env.run_episode(single, scenario)
                            for scenario in scenarios
                        ]
                    self.assertEqual(scenarios, original)
                    for trajectory in trajectories:
                        self.assertEqual(trajectory.total_reward, 1.0)
                        self.assertEqual(trajectory.turn_rewards, [0.0, 1.0])
                        self.assertEqual(
                            [turn.role for turn in trajectory.turns],
                            ["user", "assistant", "tool", "assistant"],
                        )
                        self.assertEqual(
                            trajectory.final_state["tool_trace"], ["lookup", "finish"]
                        )
                        self.assertEqual(
                            trajectory.final_state["scenario"]["public"]["nested"], []
                        )
                        self.assertEqual(
                            trajectory.final_state["expected_response"],
                            "private-answer",
                        )

    async def test_strict_reward_context_and_errors_are_preserved(self):
        class Reward:
            requires_environment_context = True
            fail_on_error = True
            fail = False
            observed = None

            async def compute_reward(self, turns, context=None):
                self.observed = context
                if self.fail:
                    raise ValueError("reward unavailable")
                return SimpleNamespace(score=0.75)

        reward = Reward()
        env = Sandbox(reward=reward)
        context = {"private": "evaluator-state"}
        trajectory = SimpleNamespace(turns=[], final_state=context)
        self.assertEqual(await env.get_reward(trajectory), 0.75)
        # The recorded state, built like score_episode's context.
        self.assertEqual(reward.observed, context)
        reward.fail = True
        with self.assertRaisesRegex(ValueError, "reward unavailable"):
            await env.get_reward(trajectory)

        async def finish(history, context):
            return "lookup" if len(history) == 1 else "finish"

        with self.assertRaisesRegex(ValueError, "reward unavailable"):
            await env.run_episode(finish, {"record_id": "strict"})

    async def test_strict_prompt_failure_stops_before_policy_execution(self):
        class BrokenPrompt(Sandbox):
            async def get_initial_prompt(self, scenario=None):
                raise ValueError("prompt unavailable")

        env = BrokenPrompt(reward=SimpleNamespace(fail_on_error=True))

        async def forbidden(*args):
            self.fail("policy ran after strict prompt failure")

        with self.assertRaisesRegex(ValueError, "prompt unavailable"):
            await env.run_episodes_batched(forbidden, [{"record_id": "strict"}])


if __name__ == "__main__":
    unittest.main()
