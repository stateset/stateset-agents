"""Tool-use environment: the policy calls typed tools, then answers (provisional).

An episode opens with the scenario's question as a user turn (after a system
prompt that lists the tools and the call format). Each assistant turn is
parsed by :func:`stateset_agents.core.tool_calls.parse_tool_calls`:

* a turn with tool calls (``<tool_call>{...}</tool_call>`` blocks, a
  ``{"tool": ..., "args": ...}`` JSON action, or structured ``tool_calls``)
  runs each call through the :class:`~stateset_agents.core.tool_registry.ToolRegistry`
  and answers with one ``tool``-role turn per call. A malformed call gets a
  structured ``malformed_call`` error observation, never a crash;
* a turn without a tool call is the final answer and ends the episode.

Every attempted call, malformed or not, spends one unit of the
``max_tool_calls`` budget. A call attempted once the budget is spent gets a
``budget_exhausted`` observation and ends the episode without an answer
(status ``FAILED``); ``max_turns`` (default ``max_tool_calls + 1``) also
bounds the episode.

The environment adds no reward of its own: the episode is scored by the
reward attached to it or held by the trainer, through the shared reward path
(``reward_context.score_episode``). Use a verifier such as
:class:`~stateset_agents.rewards.math_verifier.MathAnswerVerifier` or
:class:`~stateset_agents.rewards.code_verifier.SandboxedCodeVerifier`
(SPARSE rewards, applied once to the finished conversation); both score 0
unless ``termination`` is ``"final_answer"``. The final state records the
executed evidence: ``tool_log`` (every call with its arguments, result or
error), ``executed_tool_calls`` (the calls that ran and returned a result, as
``{"tool": name, "args": arguments}``, the trace
:class:`~stateset_agents.rewards.task_outcome.TaskOutcomeReward` compares with
a scenario's ``expected_tool_calls``), ``tool_calls_used``, ``final_answer``
and ``termination`` (``"final_answer"``, ``"budget_exhausted"`` or
``"max_turns"``).

The policy's context is an allowlist (:attr:`ToolUseEnvironment.policy_state_keys`,
:attr:`~ToolUseEnvironment.policy_scenario_keys`): the question, the tool log
and the scenario's descriptive keys. Reference data (``ground_truth``,
``tests``, ``expected_answer``, ``label``...) never reaches it, whatever its
name; ``visible_context_keys`` opts extra keys in explicitly.

Tool observations are environment output, never policy output: the trainer's
loss supervises assistant tokens only (``docs/TOOL_USE_RL.md`` states the
masking guarantee and the tests that pin it).
"""

from __future__ import annotations

import copy
import dataclasses
import random
import uuid
from collections.abc import Iterable, Sequence
from typing import Any

from ..environment_base import (
    POLICY_SCENARIO_KEYS,
    POLICY_STATE_KEYS,
    STEP_ENVIRONMENT_REWARD,
    Environment,
    EnvironmentState,
    EpisodeStatus,
    _context_keys,
)
from ..errors import ConfigurationError
from ..reward_base import RewardFunction
from ..tool_calls import (
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    ToolCallError,
    parse_tool_calls,
)
from ..tool_registry import (
    DEFAULT_MAX_OBSERVATION_CHARS,
    ToolObservation,
    ToolRegistry,
    ToolSpec,
)
from ..trajectory import ConversationTurn

__all__ = ["PRIVATE_SCENARIO_KEYS", "ToolUseEnvironment"]

#: Well-known scenario keys that hold reference data. Kept for reference: the
#: policy's context is an allowlist, so these and any other unlisted key
#: (``expected_answer``, ``correct_facts``, ``label``...) are never shown.
PRIVATE_SCENARIO_KEYS = frozenset(
    {
        "answer",
        "expected_response",
        "expected_tool_calls",
        "ground_truth",
        "hidden_tests",
        "reference",
        "solution",
        "tests",
    }
)
_QUESTION_KEYS = ("question", "prompt", "user_query", "query", "task")


def _question(scenario: dict[str, Any]) -> str:
    for key in _QUESTION_KEYS:
        value = scenario.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ConfigurationError(
        f"tool-use scenario {scenario.get('id')!r} needs a question "
        f"(one of {list(_QUESTION_KEYS)})"
    )


def _executed_call(entry: dict[str, Any]) -> dict[str, Any] | None:
    """A tool-log entry as executed evidence, if the call ran and returned.

    Calls that never ran (malformed, unknown tool, invalid arguments, over
    budget) and calls that failed or timed out are not executed outcomes.
    """
    if entry.get("ok") is not True:
        return None
    return {"tool": entry.get("name"), "args": copy.deepcopy(entry.get("arguments"))}


class ToolUseEnvironment(Environment):
    """Multi-turn tool use scored by a verifier (provisional API).

    Args:
        scenarios: Episodes to draw from; each needs a question (``question``,
            ``prompt``, ``user_query``, ``query`` or ``task``) and carries the
            verifier's reference data (``ground_truth``, ``tests``...).
        tools: A :class:`ToolRegistry` or the :class:`ToolSpec` objects to
            register.
        max_tool_calls: Calls (well-formed or not) an episode may attempt.
        max_turns: Assistant turns per episode; default ``max_tool_calls + 1``.
        reward_fn: Reward applied by the environment (optional; a trainer
            may hold it instead, with the same result).
        system_prompt: Overrides the generated tool instructions; a scenario's
            own ``system_prompt`` overrides both.
        json_actions: Also accept a whole-reply ``{"tool", "args"}`` action.
        max_observation_chars: Truncate longer observations.
        seed: Seeds scenario selection when ``reset()`` gets no scenario.
        visible_context_keys: Extra state or scenario keys to show the policy
            (an explicit opt-in; reference data is hidden by default).
    """

    supports_concurrent_episodes = True
    policy_state_keys = POLICY_STATE_KEYS | {
        "tool_log",
        "tool_calls_used",
        "final_answer",
        "termination",
    }
    policy_scenario_keys = POLICY_SCENARIO_KEYS | {"entry_point", "examples"}

    def __init__(
        self,
        scenarios: Sequence[dict[str, Any]],
        tools: ToolRegistry | Iterable[ToolSpec],
        *,
        max_tool_calls: int = 4,
        max_turns: int | None = None,
        reward_fn: RewardFunction | None = None,
        system_prompt: str | None = None,
        json_actions: bool = True,
        max_observation_chars: int = DEFAULT_MAX_OBSERVATION_CHARS,
        seed: int | None = None,
        timeout_seconds: float | None = None,
        visible_context_keys: Iterable[str] | None = None,
    ) -> None:
        if not scenarios:
            raise ConfigurationError("ToolUseEnvironment needs at least one scenario")
        if isinstance(max_tool_calls, bool) or int(max_tool_calls) < 0:
            raise ConfigurationError("max_tool_calls must be a non-negative integer")
        turns = int(max_turns) if max_turns is not None else int(max_tool_calls) + 1
        if turns < 1:
            raise ConfigurationError("max_turns must be at least 1")
        super().__init__(turns, reward_fn, timeout_seconds=timeout_seconds)
        for scenario in scenarios:
            _question(scenario)
        self.scenarios = [dict(scenario) for scenario in scenarios]
        self.registry = (
            tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
        )
        if not len(self.registry):
            raise ConfigurationError("ToolUseEnvironment needs at least one tool")
        self.max_tool_calls = int(max_tool_calls)
        self.system_prompt = system_prompt
        self.json_actions = bool(json_actions)
        self.max_observation_chars = int(max_observation_chars)
        self.seed = seed
        self._rng = random.Random(seed)
        self.visible_context_keys = _context_keys(visible_context_keys)

    # -- episode lifecycle -------------------------------------------------

    async def reset(self, scenario: dict[str, Any] | None = None) -> EnvironmentState:
        chosen = (
            dict(scenario)
            if scenario is not None
            else dict(self._rng.choice(self.scenarios))
        )
        question = _question(chosen)
        episode_id = str(uuid.uuid4())
        state = EnvironmentState(
            episode_id=episode_id,
            turn_count=0,
            status=EpisodeStatus.ONGOING,
            context={
                "scenario": chosen,
                "messages": [{"role": "user", "content": question}],
                "transcript": [{"role": "user", "content": question}],
                "tool_log": [],
                "executed_tool_calls": [],
                "tool_calls_used": 0,
                "final_answer": None,
                "termination": None,
            },
        )
        self.active_episodes[episode_id] = state
        return state

    async def get_initial_prompt(self, scenario: dict[str, Any] | None = None) -> str:
        scenario = scenario or {}
        override = scenario.get("system_prompt") or self.system_prompt
        if isinstance(override, str):
            return override
        return self.tool_instructions()

    def tool_instructions(self) -> str:
        """The default system prompt: the tools and how to call them."""
        forms = f'{TOOL_CALL_OPEN}{{"name": "<tool>", "arguments": {{...}}}}{TOOL_CALL_CLOSE}'
        if self.json_actions:
            forms += ' or the whole reply {"tool": "<tool>", "args": {...}}'
        return (
            f"You can call tools. To call one, write {forms}. Each result "
            "arrives as a tool message. When you know the answer, reply with "
            f"the final answer and no tool call. At most {self.max_tool_calls} "
            f"tool calls.\nTools:\n{self.registry.describe()}"
        )

    async def step(
        self, state: EnvironmentState, action: ConversationTurn
    ) -> tuple[EnvironmentState, float, bool, dict[str, Any]]:
        """Run the turn's tool calls, or take it as the final answer."""
        new_state = state.copy()
        context = new_state.context
        context["tool_log"] = copy.deepcopy(context.get("tool_log", []))
        context["executed_tool_calls"] = copy.deepcopy(
            context.get("executed_tool_calls", [])
        )
        prior = [
            ConversationTurn(role=m["role"], content=m["content"])
            for m in context.get("transcript", [])
        ]
        new_state.turn_count += 1
        parsed = parse_tool_calls(action, json_actions=self.json_actions)
        observations: list[ToolObservation] = []
        if parsed.attempted:
            observations = await self._run_calls(new_state, parsed.items)
        else:
            context["final_answer"] = parsed.text
            self._finish(new_state, "final_answer", EpisodeStatus.COMPLETED)
        if not self._finished(new_state) and new_state.turn_count >= self.max_turns:
            self._finish(new_state, "max_turns", EpisodeStatus.TIMEOUT)
        # The conversation so far as plain messages (JSON-friendly state).
        context["transcript"] = [
            *context.get("transcript", []),
            {"role": "assistant", "content": str(action.content or "")},
            *(o.to_message(self.max_observation_chars) for o in observations),
        ]
        step_reward = await self._step_reward(new_state, [*prior, action])
        info = {
            "messages": [
                o.to_message(self.max_observation_chars) for o in observations
            ],
            "tool_calls": [o.to_log() for o in observations],
        }
        return new_state, step_reward, self._finished(new_state), info

    # -- helpers -------------------------------------------------------------

    async def _run_calls(
        self, state: EnvironmentState, items: Sequence[Any]
    ) -> list[ToolObservation]:
        observations: list[ToolObservation] = []
        context = state.context
        for item in items:
            if int(context["tool_calls_used"]) >= self.max_tool_calls:
                observations.append(
                    ToolObservation.failure(
                        getattr(item, "name", ""),
                        item.call_id,
                        "budget_exhausted",
                        f"the budget of {self.max_tool_calls} tool calls is spent",
                    )
                )
                self._finish(state, "budget_exhausted", EpisodeStatus.FAILED)
                break
            # Ids are unique within the episode: call_<n> for the n-th attempt.
            item = dataclasses.replace(
                item, call_id=f"call_{context['tool_calls_used']}"
            )
            context["tool_calls_used"] = int(context["tool_calls_used"]) + 1
            observation = (
                ToolObservation.malformed(item)
                if isinstance(item, ToolCallError)
                else await self.registry.execute(item)
            )
            observations.append(observation)
            entry = observation.to_log()
            context["tool_log"].append(entry)
            executed = _executed_call(entry)
            if executed is not None:
                context["executed_tool_calls"].append(executed)
        return observations

    async def _step_reward(
        self, state: EnvironmentState, turns: list[ConversationTurn]
    ) -> float:
        """Score the step for an IMMEDIATE/DENSE reward; 0.0 otherwise."""
        state.metadata[STEP_ENVIRONMENT_REWARD] = 0.0
        self._snapshot_step_state(state)
        if self.reward_fn is not None and self._should_compute_step_reward():
            return await self._apply_step_reward(
                state, self._step_reward_turns(state, turns)
            )
        return 0.0

    @staticmethod
    def _finish(state: EnvironmentState, how: str, status: EpisodeStatus) -> None:
        state.context["termination"] = how
        state.status = status

    @staticmethod
    def _finished(state: EnvironmentState) -> bool:
        return state.status != EpisodeStatus.ONGOING

    def clone(self) -> ToolUseEnvironment:
        return ToolUseEnvironment(
            self.scenarios,
            self.registry,
            max_tool_calls=self.max_tool_calls,
            max_turns=self.max_turns,
            reward_fn=self.reward_fn,
            system_prompt=self.system_prompt,
            json_actions=self.json_actions,
            max_observation_chars=self.max_observation_chars,
            seed=self.seed,
            timeout_seconds=self.timeout_seconds,
            visible_context_keys=self.visible_context_keys,
        )
