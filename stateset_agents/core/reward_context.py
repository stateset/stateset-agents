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
only the newest one (see :func:`assistant_turns_in_scope`), and the episode
takes the mean of those step scores, so every turn weighs the same.

:func:`score_episode` is the one place an episode's reward is settled for
training, evaluation and the distributed trainer alike.
"""

from __future__ import annotations

import inspect
import math
import numbers
from collections.abc import Mapping, Sequence
from typing import Any

from .trajectory import ConversationTurn

#: ``RewardResult.metadata`` key a reward sets (to a short reason) when its
#: score is an opted-in fallback rather than a real verdict, e.g.
#: ``NSRVerifierConfig(error_score=0.0)`` or ``RulerRewardFunction(
#: fallback_enabled=True)`` during an outage. Environments and
#: ``score_episode`` copy it to the trajectory's metadata.
REWARD_FALLBACK_KEY = "reward_fallback"

#: Context key an environment sets when it asks for one step's reward.
REWARD_SCOPE_KEY = "reward_scope"
#: ``REWARD_SCOPE_KEY`` value: score only the newest assistant turn.
LATEST_TURN_SCOPE = "latest_turn"


def requires_environment_context(reward_fn: Any) -> bool:
    """Whether ``reward_fn`` judges executed environment evidence."""
    # Only an explicit True counts: a mock's auto-attribute is not a claim.
    return getattr(reward_fn, "requires_environment_context", False) is True


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


def _real(value: Any) -> float | None:
    """A Python or NumPy/torch scalar as a float, or None if it is not one."""
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Real):
        return float(value)
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "ndim", 0) == 0:
        scalar: Any = item()
        if isinstance(scalar, bool):
            return None
        if isinstance(scalar, numbers.Real):
            return float(scalar)
    return None


def reward_score(result: Any) -> float:
    """The finite score of a reward result: a ``RewardResult`` (anything with
    ``.score``, or its ``.total_reward`` alias), a mapping with ``"score"`` (or
    ``"total_reward"``), or a real scalar (Python, NumPy or a 0-d tensor).

    Raises ``TypeError`` for anything else (``None``, a string, a list) and
    ``ValueError`` for a non-finite score: neither is ever replaced by a
    stand-in value. Every caller (environments, ``Environment.get_reward``,
    ``score_episode``) uses this one parser.
    """
    if isinstance(result, Mapping):
        candidates = [result.get("score"), result.get("total_reward")]
    else:
        # ``total_reward`` is RewardResult's compatibility alias.
        candidates = [
            getattr(result, "score", None),
            getattr(result, "total_reward", None),
            result,
        ]
    score = next((value for value in map(_real, candidates) if value is not None), None)
    if score is None:
        raise TypeError(
            f"Unexpected reward result {type(result).__name__!r}: expected a "
            "RewardResult, a mapping with 'score' or a real number"
        )
    if not math.isfinite(score):
        raise ValueError(f"reward score must be finite, got {score}")
    return score


def reward_fallback(result: Any) -> str | None:
    """The fallback reason a reward result is flagged with, if any."""
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, Mapping) and metadata.get(REWARD_FALLBACK_KEY):
        return str(metadata[REWARD_FALLBACK_KEY])
    return None


def note_fallback(metadata: dict[str, Any], result: Any) -> None:
    """Record ``result``'s fallback flag (if any) in episode ``metadata``."""
    reason = reward_fallback(result)
    if reason is not None:
        metadata[REWARD_FALLBACK_KEY] = reason


def normalize_turns(turns: Any) -> list[ConversationTurn]:
    """Conversation turns as ``ConversationTurn`` objects.

    Accepts ``ConversationTurn`` objects and ``{"role": ..., "content": ...}``
    mappings (the shape ``Trajectory.add_turn`` and many callers use), so
    rewards that read ``turn.role`` never see a dict and silently score 0.
    """
    normalized: list[ConversationTurn] = []
    for turn in list(turns or []):
        if isinstance(turn, ConversationTurn):
            normalized.append(turn)
        elif isinstance(turn, Mapping):
            normalized.append(
                ConversationTurn(
                    role=str(turn.get("role", "user")),
                    content="" if turn.get("content") is None else str(turn["content"]),
                    metadata=dict(turn.get("metadata") or {}),
                )
            )
        elif hasattr(turn, "role") and hasattr(turn, "content"):
            normalized.append(turn)
        else:
            raise TypeError(f"Unsupported conversation turn: {type(turn).__name__!r}")
    return normalized


def _scores_per_step(reward_fn: Any, environment: Any) -> bool:
    """Whether an environment would apply ``reward_fn`` after every step."""
    reward_type = getattr(reward_fn, "reward_type", None)
    if reward_type is None:
        return bool(getattr(environment, "scores_steps", False))
    return getattr(reward_type, "value", reward_type) in ("immediate", "dense")


async def _call(reward_fn: Any, turns: list[Any], context: dict[str, Any]) -> Any:
    result = reward_fn.compute_reward(turns, context)
    if inspect.isawaitable(result):
        result = await result
    return result


def response_reward_context(
    prompt: str,
    scenario: Mapping[str, Any] | None = None,
    *,
    reward_fn: Any = None,
) -> dict[str, Any]:
    """The reward context for one reply to one prompt (single-turn trainers).

    The standard shape: the scenario's keys lifted to the top level, with
    ``"user_query"`` set to the prompt unless the scenario has one, and the
    scenario under ``"scenario"``. ``GSPOTrainer``, ``GSPOTokenTrainer`` and
    the TRL reward bridge build their context here, so scenario labels such
    as ``ground_truth`` reach the reward as they do in a multi-turn episode.
    """
    query = {"user_query": str(prompt), **dict(scenario or {})}
    return build_reward_context(None, query, reward_fn=reward_fn)


async def score_episode(
    trajectory: Any,
    reward_fn: Any,
    environment: Any = None,
    scenario: Mapping[str, Any] | None = None,
) -> float:
    """Give a finished episode its one reward and return the episode total.

    Training, periodic evaluation, post-training evaluation and the
    distributed trainer all score episodes here, so the same trajectory gets
    the same reward on every path:

    * Without a ``reward_fn``, or when ``reward_fn`` is the environment's own
      and the environment applied it, the environment's reward stands.
    * Otherwise ``reward_fn`` scores the episode exactly as an environment
      would have: IMMEDIATE/DENSE rewards score each assistant turn (the
      conversation so far, newest turn in scope) and contribute their mean;
      CUMULATIVE/SPARSE rewards score the whole episode once. That score is
      added to the environment's own step rewards
      (``metadata["environment_rewards"]``), so the total does not depend on
      whether the reward was attached to the environment or the trainer.

    The trajectory's ``total_reward``, ``turn_rewards`` and
    ``metadata["reward_breakdown"]`` are updated. A failed reward (here or
    recorded by the environment as ``reward_error``) or a non-finite total
    raises: callers drop or report the episode, never score it 0.
    """
    metadata = getattr(trajectory, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
    if reward_fn is None or environment_applied_reward(
        trajectory, reward_fn, environment
    ):
        error = metadata.get("reward_error")
        if error:
            raise RuntimeError(f"Episode reward failed: {error}")
        total = float(getattr(trajectory, "total_reward", 0.0))
        if not math.isfinite(total):
            raise ValueError(f"Episode reward must be finite, got {total}")
        return total

    turns = normalize_turns(getattr(trajectory, "turns", None))
    context = trajectory_reward_context(trajectory, scenario, reward_fn=reward_fn)
    intrinsic = metadata.get("environment_rewards")
    if not isinstance(intrinsic, list) or not intrinsic:
        steps = len(getattr(trajectory, "turn_rewards", None) or []) or 1
        intrinsic = [0.0] * steps
    intrinsic = [float(value) for value in intrinsic]

    assistant = [
        i for i, turn in enumerate(turns) if getattr(turn, "role", None) == "assistant"
    ]
    breakdown: Any = None
    fallback: str | None = None
    if _scores_per_step(reward_fn, environment) and assistant:
        step_context = {**context, REWARD_SCOPE_KEY: LATEST_TURN_SCOPE}
        scores = []
        for index in assistant:
            result = await _call(reward_fn, turns[: index + 1], step_context)
            scores.append(reward_score(result))
            breakdown = getattr(result, "breakdown", None)
            fallback = reward_fallback(result) or fallback
        parts = [score / len(scores) for score in scores]
        if len(parts) != len(intrinsic):
            # Steps and assistant turns do not line up; credit the mean once.
            parts = [0.0] * (len(intrinsic) - 1) + [math.fsum(parts)]
    else:
        result = await _call(reward_fn, turns, context)
        parts = [0.0] * (len(intrinsic) - 1) + [reward_score(result)]
        breakdown = getattr(result, "breakdown", None)
        fallback = reward_fallback(result)

    turn_rewards = [base + part for base, part in zip(intrinsic, parts, strict=True)]
    total = math.fsum(turn_rewards)
    if not math.isfinite(total):
        raise ValueError(f"Episode reward must be finite, got {total}")
    trajectory.total_reward = total
    if hasattr(trajectory, "turn_rewards"):
        trajectory.turn_rewards = turn_rewards
    if hasattr(trajectory, "rewards"):
        trajectory.rewards = turn_rewards
    if isinstance(getattr(trajectory, "metadata", None), dict):
        trajectory.metadata["reward_breakdown"] = (
            dict(breakdown) if isinstance(breakdown, Mapping) else {}
        )
        error = trajectory.metadata.pop("reward_error", None)
        if error:
            # The environment's own reward failed; this one scored.
            trajectory.metadata["environment_reward_error"] = error
        if fallback is not None:
            trajectory.metadata[REWARD_FALLBACK_KEY] = fallback
    return total


__all__ = [
    "LATEST_TURN_SCOPE",
    "REWARD_FALLBACK_KEY",
    "REWARD_SCOPE_KEY",
    "assistant_turns_in_scope",
    "build_reward_context",
    "environment_applied_reward",
    "normalize_turns",
    "note_fallback",
    "reward_fallback",
    "requires_environment_context",
    "response_reward_context",
    "reward_score",
    "score_episode",
    "scores_latest_turn_only",
    "trajectory_reward_context",
]
