"""The context contract every caller of ``RewardFunction.compute_reward`` uses.

Environments, trainers and evaluators all score episodes with the same reward
functions, so they must hand those functions the same context. The contract:

* The scenario's own keys sit at the top level, so rewards that read
  ``context["ground_truth"]`` or ``context["task"]`` find them whichever caller
  scored the episode.
* The environment's episode state is laid over them (it wins a key clash,
  except that an environment value of ``None`` never hides a scenario value).
* The scenario itself is also available as ``context["scenario"]``.

Rewards that declare ``requires_environment_context`` judge executed evidence
(tool traces, policy violations). They receive the environment state plus
``context["scenario"]`` but never scenario keys lifted to the top level, so
reference data in a scenario cannot pose as evidence the environment recorded.

When an environment scores a single step it adds
``context[REWARD_SCOPE_KEY] == LATEST_TURN_SCOPE``: the conversation so far is
still passed for context, but rewards that aggregate over assistant turns score
only the newest one (see :func:`assistant_turns_in_scope`). Summing those step
rewards then weights every turn once.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

#: Context key an environment sets when it asks for one step's reward.
REWARD_SCOPE_KEY = "reward_scope"
#: ``REWARD_SCOPE_KEY`` value: score only the newest assistant turn.
LATEST_TURN_SCOPE = "latest_turn"


def requires_environment_context(reward_fn: Any) -> bool:
    """Whether ``reward_fn`` judges executed environment evidence."""
    return bool(getattr(reward_fn, "requires_environment_context", False))


def _resolve_scenario(
    environment_context: Mapping[str, Any], scenario: Any
) -> dict[str, Any] | None:
    if isinstance(scenario, dict):
        return scenario
    for key in ("scenario", "task"):
        candidate = environment_context.get(key)
        if isinstance(candidate, dict):
            return candidate
    return None


def build_reward_context(
    environment_context: Mapping[str, Any] | None = None,
    scenario: Mapping[str, Any] | None = None,
    *,
    reward_fn: Any = None,
    latest_turn_only: bool = False,
) -> dict[str, Any]:
    """Return the context to pass to ``reward_fn.compute_reward``.

    Args:
        environment_context: The episode's environment state
            (``EnvironmentState.context`` or ``MultiTurnTrajectory.final_state``).
        scenario: The scenario the episode ran. Defaults to the environment
            state's ``"scenario"`` (or ``"task"``) entry.
        reward_fn: The reward that will receive the context; rewards with
            ``requires_environment_context`` get no lifted scenario keys.
        latest_turn_only: Mark the context as scoring one step, so rewards
            score only the newest assistant turn.
    """
    environment = dict(environment_context or {})
    resolved = _resolve_scenario(environment, scenario)
    context: dict[str, Any] = {}
    if resolved is not None and not requires_environment_context(reward_fn):
        context.update(resolved)
    for key, value in environment.items():
        if value is None and context.get(key) is not None:
            continue
        context[key] = value
    if resolved is not None:
        context["scenario"] = resolved
    if latest_turn_only:
        context[REWARD_SCOPE_KEY] = LATEST_TURN_SCOPE
    return context


def trajectory_reward_context(
    trajectory: Any,
    scenario: Mapping[str, Any] | None = None,
    *,
    reward_fn: Any = None,
) -> dict[str, Any]:
    """Build the reward context for a finished trajectory.

    Uses the trajectory's ``final_state`` as the environment state and, unless
    ``scenario`` is given, the scenario recorded in its metadata or initial
    state.
    """
    final_state = getattr(trajectory, "final_state", None)
    environment = final_state if isinstance(final_state, Mapping) else None
    if not isinstance(scenario, dict):
        metadata = getattr(trajectory, "metadata", None)
        initial = getattr(trajectory, "initial_state", None)
        for source in (metadata, initial):
            if isinstance(source, Mapping) and isinstance(source.get("scenario"), dict):
                scenario = source["scenario"]
                break
    return build_reward_context(environment, scenario, reward_fn=reward_fn)


def scores_latest_turn_only(context: Mapping[str, Any] | None) -> bool:
    """Whether ``context`` asks for the newest assistant turn's score only."""
    return isinstance(context, Mapping) and (
        context.get(REWARD_SCOPE_KEY) == LATEST_TURN_SCOPE
    )


def assistant_turns_in_scope(
    turns: Sequence[Any], context: Mapping[str, Any] | None = None
) -> list[Any]:
    """Assistant turns a reward should score under ``context``.

    All assistant turns normally; only the newest when the caller is scoring
    a single step.
    """
    assistant = [turn for turn in turns if getattr(turn, "role", None) == "assistant"]
    if scores_latest_turn_only(context):
        return assistant[-1:]
    return assistant


def environment_applied_reward(
    trajectory: Any, reward_fn: Any, environment: Any
) -> bool:
    """Whether ``environment`` already scored ``trajectory`` with ``reward_fn``.

    True only when ``reward_fn`` is the environment's own reward function and
    the environment recorded applying it, in which case scoring the episode
    again would double the cost of the reward (an LLM judge, a budget) and
    replace the environment's reward with one computed differently.
    """
    if reward_fn is None or reward_fn is not getattr(environment, "reward_fn", None):
        return False
    metadata = getattr(trajectory, "metadata", None)
    return isinstance(metadata, Mapping) and bool(metadata.get("reward_fn_applied"))


def set_episode_reward(trajectory: Any, score: float) -> None:
    """Replace a trajectory's reward with one episode-level ``score``.

    The score is credited to the final step, so ``sum(turn_rewards)`` keeps
    matching ``total_reward``.
    """
    previous = getattr(trajectory, "turn_rewards", None) or []
    steps = max(1, len(previous))
    turn_rewards = [0.0] * (steps - 1) + [float(score)]
    trajectory.total_reward = float(score)
    trajectory.turn_rewards = turn_rewards
    if hasattr(trajectory, "rewards"):
        trajectory.rewards = turn_rewards


__all__ = [
    "LATEST_TURN_SCOPE",
    "REWARD_SCOPE_KEY",
    "assistant_turns_in_scope",
    "build_reward_context",
    "environment_applied_reward",
    "requires_environment_context",
    "scores_latest_turn_only",
    "set_episode_reward",
    "trajectory_reward_context",
]
