"""
Task-oriented environment implementation.
"""

from __future__ import annotations

import json
import random
import re
import uuid
from collections.abc import Callable
from typing import Any

from .environment_base import (
    STEP_ENVIRONMENT_REWARD,
    Environment,
    EnvironmentState,
    EpisodeStatus,
    _context_keys,
    policy_visible,
)
from .reward import RewardFunction
from .trajectory import ConversationTurn

_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}")
#: Shares of the default environment reward: progress, then completion.
_PROGRESS_SHARE = 2.0 / 3.0
_COMPLETION_SHARE = 1.0 / 3.0


def _names(value: Any) -> list[str]:
    """Action names from a string, a list of strings, or ``{"name": ...}``."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        name = value.get("name") or value.get("tool")
        if isinstance(name, str):
            return [name]
        function = value.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            return [function["name"]]
        return []
    if isinstance(value, (list, tuple)):
        return [name for item in value for name in _names(item)]
    return []


def declared_actions(turn: ConversationTurn) -> set[str]:
    """Actions an agent turn explicitly declares, lower-cased.

    Explicit signals are structured tool calls (``turn.tool_calls``), an
    ``"action"``/``"actions"`` entry in ``turn.metadata``, or a JSON object in
    the reply with an ``"action"``, ``"actions"`` or ``"tool"`` key, e.g.
    ``{"action": "load_data"}``. Prose that merely mentions an action does not
    declare it.
    """
    return set(_declared_in_order(turn))


def _declared_in_order(turn: ConversationTurn) -> list[str]:
    """:func:`declared_actions` in the order the turn declares them."""
    names: list[str] = _names(turn.tool_calls or [])
    metadata = turn.metadata if isinstance(turn.metadata, dict) else {}
    for key in ("action", "actions"):
        names += _names(metadata.get(key))
    content = turn.content if isinstance(turn.content, str) else ""
    candidates = [content.strip()] + _JSON_OBJECT_RE.findall(content)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (ValueError, RecursionError):
            continue
        if isinstance(parsed, dict):
            for key in ("action", "actions", "tool"):
                names += _names(parsed.get(key))
    ordered: list[str] = []
    for name in names:
        key = name.strip().lower()
        if key and key not in ordered:
            ordered.append(key)
    return ordered


def _required_name(action: Any) -> str | None:
    if isinstance(action, str):
        return action
    if isinstance(action, dict) and isinstance(action.get("name"), str):
        return str(action["name"])
    return None


class TaskEnvironment(Environment):
    """Environment for task-oriented interactions.

    A task lists ``required_actions`` (names, or mappings with a ``"name"``).
    An action is completed only when the agent declares it explicitly: a
    structured tool call, an ``"action"`` entry in the turn's metadata, or a
    JSON object such as ``{"action": "load_data"}`` in the reply (see
    :func:`declared_actions`).

    ``match_action_keywords=True`` (a keyword argument) additionally completes
    an action whenever one of its ``"keywords"`` appears in a reply. That is
    gameable, since a reply can list every keyword without doing anything, and
    is meant only for demos with free-text agents.

    At most one action executes per step (``max_actions_per_step``, default
    1): declaring every action in one reply is not doing them, so only the
    first action a reply declares executes (re-declaring a finished one makes
    no progress). Pass ``max_actions_per_step=None`` for environments that
    really execute batches.

    The environment's own reward totals ``task_reward_scale`` (default 1.0)
    over an episode that completes the task: two thirds paid as progress
    (spread over the actions) and one third on completion, so it sits on the
    same scale as a reward function instead of dwarfing it.
    ``task_reward_scale=15.0`` restores the pre-0.70 scale (10 per full
    progress plus a bonus of 5).

    The policy's context shows the task's description, instructions, goal
    and type, never ``required_actions`` or reference data
    (``visible_context_keys=`` opts extra keys in). ``seed`` seeds task
    selection when ``reset()`` gets no scenario.
    """

    #: The policy sees its own progress (the actions that counted so far),
    #: never the list of required ones.
    policy_state_keys = Environment.policy_state_keys | {
        "completed_actions",
        "task_progress",
    }

    def __init__(
        self,
        tasks: list[dict[str, Any]],
        success_criteria: Callable[[list[ConversationTurn], dict[str, Any]], bool],
        max_turns: int = 20,
        reward_fn: RewardFunction | None = None,
        **kwargs,
    ):
        self.match_action_keywords = bool(kwargs.pop("match_action_keywords", False))
        per_step = kwargs.pop("max_actions_per_step", 1)
        if per_step is not None and (isinstance(per_step, bool) or int(per_step) < 1):
            raise ValueError("max_actions_per_step must be a positive integer or None")
        self.max_actions_per_step: int | None = (
            None if per_step is None else int(per_step)
        )
        self.task_reward_scale = float(kwargs.pop("task_reward_scale", 1.0))
        self.seed: int | None = kwargs.pop("seed", None)
        self.visible_context_keys = _context_keys(
            kwargs.pop("visible_context_keys", None)
        )
        super().__init__(max_turns, reward_fn, **kwargs)
        self.tasks = tasks
        self.success_criteria = success_criteria
        self.current_task: dict[str, Any] | None = None
        self._last_state: EnvironmentState | None = None
        self._rng = random.Random(self.seed)

    async def reset(self, scenario: dict[str, Any] | None = None) -> EnvironmentState:
        """Reset for a new task"""
        selected_scenario = (
            scenario if scenario is not None else self._rng.choice(self.tasks)
        )

        self.current_task = selected_scenario
        episode_id = str(uuid.uuid4())
        scenario_conv_id = None
        scenario_conv_id = selected_scenario.get("conversation_id")
        conversation_id = (
            str(scenario_conv_id)
            if scenario_conv_id is not None and str(scenario_conv_id)
            else episode_id
        )

        state = EnvironmentState(
            episode_id=episode_id,
            turn_count=0,
            status=EpisodeStatus.ONGOING,
            context={
                "conversation_id": conversation_id,
                "task": selected_scenario,
                "task_goal": selected_scenario.get("goal"),
                "task_type": selected_scenario.get("type"),
                "required_actions": selected_scenario.get("required_actions", []),
                "completed_actions": [],
                "task_progress": 0.0,
                "turns": [],
            },
        )
        task_id = selected_scenario.get("task_id", selected_scenario.get("id"))
        if task_id is not None:
            state.context["task_id"] = task_id

        self.active_episodes[episode_id] = state
        self._last_state = state
        return state

    async def step(self, state: EnvironmentState, action: ConversationTurn) -> Any:
        """Process agent action and update task state.

        The step reward is the task-progress reward plus, for an IMMEDIATE or
        DENSE ``reward_fn``, that function's score for this action. Other
        reward functions are applied once at episode end.
        """
        new_state = state.copy()
        new_state.turn_count += 1

        prior_turns = list(new_state.context.get("turns", []))
        turns = prior_turns + [action]

        await self._update_task_progress(action, new_state)

        env_response = await self._generate_task_response(action, new_state)
        turns.append(env_response)
        new_state.context["turns"] = turns

        step_reward = await self._calculate_task_reward(action, new_state)
        new_state.metadata[STEP_ENVIRONMENT_REWARD] = step_reward
        self._snapshot_step_state(new_state)
        if self.reward_fn is not None and self._should_compute_step_reward():
            step_reward += await self._apply_step_reward(
                new_state, self._step_reward_turns(new_state, prior_turns + [action])
            )

        task_complete = await self._check_task_completion(new_state)
        done = task_complete or new_state.turn_count >= self.max_turns

        if done:
            new_state.status = (
                EpisodeStatus.COMPLETED if task_complete else EpisodeStatus.TIMEOUT
            )

        return new_state, env_response, step_reward, done

    async def get_initial_prompt(self, scenario: dict[str, Any] | None = None) -> str:
        """The task's description and instructions, as the policy may see them
        (reference-like keys inside either are dropped, see
        :func:`~stateset_agents.core.environment_base.policy_visible`)."""
        active_scenario = scenario or self.current_task or {}
        shown = self.visible_context_keys
        description = policy_visible(
            active_scenario.get("description", "Complete the given task."), shown
        )
        task_prompt = f"Task: {description}"
        instructions = policy_visible(active_scenario.get("instructions"), shown)
        if instructions:
            task_prompt += f"\n\nInstructions: {instructions}"

        return task_prompt

    async def _update_task_progress(
        self, action: ConversationTurn, state: EnvironmentState
    ) -> None:
        """Complete the required actions this turn declares (see class doc).

        The first ``max_actions_per_step`` actions the turn declares execute.
        """
        required_actions = state.context.get("required_actions", [])
        # Copy: the previous state's list must not change under it.
        completed_actions = list(state.context.get("completed_actions", []))
        signalled = self._actions_this_turn(action, required_actions)
        if self.max_actions_per_step is not None:
            # The first actions the turn declares execute (repeating a done
            # one included); the rest of a batch does not.
            signalled = signalled[: self.max_actions_per_step]
        for req_action in signalled:
            if req_action not in completed_actions:
                completed_actions.append(req_action)

        state.context["completed_actions"] = completed_actions
        state.context["task_progress"] = len(completed_actions) / max(
            1, len(required_actions)
        )

    def _actions_this_turn(
        self, action: ConversationTurn, required_actions: list[Any]
    ) -> list[Any]:
        """Required actions the turn signals, in the order it signals them."""
        by_name: dict[str, Any] = {}
        for req_action in required_actions:
            name = _required_name(req_action)
            if name is not None:
                by_name.setdefault(name.strip().lower(), req_action)
        signalled = [
            by_name[name] for name in _declared_in_order(action) if name in by_name
        ]
        if self.match_action_keywords:
            content = (action.content or "").lower()
            signalled += [
                req_action
                for req_action in required_actions
                if isinstance(req_action, dict)
                and req_action not in signalled
                and any(
                    str(keyword).lower() in content
                    for keyword in req_action.get("keywords", [])
                )
            ]
        return signalled

    async def _generate_task_response(
        self, action: ConversationTurn, state: EnvironmentState
    ) -> ConversationTurn:
        """Generate environment response for task"""
        progress = state.context.get("task_progress", 0.0)

        if progress == 1.0:
            response = "Task completed successfully!"
        elif progress > 0.5:
            response = "Good progress! Continue with the remaining steps."
        else:
            response = "Please proceed with the task requirements."

        return ConversationTurn(
            role="system", content=response, metadata={"task_progress": progress}
        )

    async def _calculate_task_reward(
        self, action: ConversationTurn, state: EnvironmentState
    ) -> float:
        """Reward progress made by this step, plus a one-time completion bonus.

        The bonus is paid on the step that completes the task, not on every
        later step, so stalling after completion earns nothing. A completed
        episode totals ``task_reward_scale``: two thirds for progress, one
        third for completion.
        """
        progress = state.context.get("task_progress", 0.0)
        previous_progress = state.context.get("previous_progress", 0.0)
        progress_delta = progress - previous_progress

        state.context["previous_progress"] = progress

        reward = progress_delta * _PROGRESS_SHARE
        if progress >= 1.0 > previous_progress:
            reward += _COMPLETION_SHARE

        return float(reward * self.task_reward_scale)

    async def _check_task_completion(self, state: EnvironmentState) -> bool:
        """Check if task is completed using success_criteria"""
        turns = state.context.get("turns", [])
        return self.success_criteria(turns, state.context)

    def clone(self) -> TaskEnvironment:
        """Create a new TaskEnvironment with the same configuration."""
        return TaskEnvironment(
            tasks=list(self.tasks),
            success_criteria=self.success_criteria,
            max_turns=self.max_turns,
            reward_fn=self.reward_fn,
            timeout_seconds=self.timeout_seconds,
            match_action_keywords=self.match_action_keywords,
            max_actions_per_step=self.max_actions_per_step,
            task_reward_scale=self.task_reward_scale,
            seed=self.seed,
            visible_context_keys=self.visible_context_keys,
        )
