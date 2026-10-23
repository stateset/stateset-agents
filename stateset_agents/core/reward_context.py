"""The context contract every caller of ``RewardFunction.compute_reward`` uses.

Environments, trainers and evaluators all score episodes with the same reward
functions, so they must hand those functions the same context. The contract:

* The scenario's own keys sit at the top level, so rewards that read
  ``context["ground_truth"]`` or ``context["task"]`` find them whichever caller
  scored the episode.
* The environment's episode state is laid over them (it wins a key clash,
  except that an environment value of ``None`` never hides a scenario value).
  ``termination`` (:data:`TERMINATION_KEY`) is never lifted from the
  scenario: how an episode ended comes from the environment's state alone.
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

from .errors import ErrorCode, RewardError
from .trajectory import ConversationTurn


class RewardUnavailableError(RewardError, RuntimeError):
    """A reward could not be computed for an external reason (judge or network
    outage, a verifier that cannot be reached).

    This is the only failure reward composers may absorb: ``CompositeReward``
    and ``MultiObjectiveRewardFunction`` raise it in turn by default, and with
    ``outage_policy = "renormalize"`` leave the component out, spread its
    weight over the rest and flag the result. Every other exception is a bug
    or bad data and propagates, so it is never mistaken for an outage.

    A :class:`~stateset_agents.core.errors.RewardError` (so a
    ``StateSetError``) that is still a ``RuntimeError`` and prints its bare
    message, so existing ``except RuntimeError`` handlers are unchanged.
    """

    message_only_str = True
    default_code = ErrorCode.RWD_UNAVAILABLE


#: ``RewardResult.metadata`` key a reward sets (to a short reason) when its
#: score is a fallback rather than a real verdict, e.g.
#: ``NSRVerifierConfig(error_score=0.0)`` or ``RulerRewardFunction(
#: fallback_enabled=True)`` during an outage, ``LLMJudgeRewardWithFallback``
#: when its heuristic replaces an unavailable judge, or a composer that
#: renormalized around an outage. Environments, ``score_episode`` and
#: ``score_trajectory`` copy it to the trajectory's metadata.
REWARD_FALLBACK_KEY = "reward_fallback"

#: ``RewardResult.metadata`` key a composer sets to the names of components
#: it left out during an outage (with an opted-in renormalizing policy).
#: Environments and ``score_episode`` copy it to the trajectory's metadata as
#: :data:`FAILED_COMPONENTS_KEY`.
COMPONENT_FAILURES_KEY = "failed_components"
#: Trajectory metadata key listing the reward components an outage left out.
FAILED_COMPONENTS_KEY = "reward_failed_components"

#: Context key an environment sets when it asks for one step's reward.
REWARD_SCOPE_KEY = "reward_scope"
#: ``REWARD_SCOPE_KEY`` value: score only the newest assistant turn.
LATEST_TURN_SCOPE = "latest_turn"

#: Episode-state key in which a multi-turn environment records how an
#: episode ended (``ToolUseEnvironment``: ``"final_answer"``,
#: ``"budget_exhausted"``, ``"max_turns"``; ``None`` while it runs).
TERMINATION_KEY = "termination"
#: :data:`TERMINATION_KEY` value of an episode that ended with a final answer.
FINAL_ANSWER_TERMINATION = "final_answer"


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
        # How the episode ended is the environment's record alone: a
        # scenario's own ``termination`` is never lifted, so it cannot make
        # an unanswered episode (``termination`` None) look answered.
        context.pop(TERMINATION_KEY, None)
    for key, value in environment.items():
        if value is None and context.get(key) is not None:
            continue
        context[key] = value
    if resolved is not None:
        context["scenario"] = resolved
    if latest_turn_only:
        context[REWARD_SCOPE_KEY] = LATEST_TURN_SCOPE
    return context


def _trajectory_scenario(
    trajectory: Any, scenario: Mapping[str, Any] | None = None
) -> Mapping[str, Any] | None:
    """The scenario a trajectory ran: ``scenario`` or the one it recorded."""
    if isinstance(scenario, dict):
        return scenario
    for source in (
        getattr(trajectory, "metadata", None),
        getattr(trajectory, "initial_state", None),
    ):
        if isinstance(source, Mapping):
            recorded = source.get("scenario")
            if isinstance(recorded, dict):
                return recorded
    return None


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
    return build_reward_context(
        environment, _trajectory_scenario(trajectory, scenario), reward_fn=reward_fn
    )


def unanswered_termination(context: Mapping[str, Any] | None) -> str | None:
    """How the episode ended, when the context says it ended without a final
    answer; ``None`` otherwise.

    Verifiers score such an episode 0: its last turn is not an answer, however
    right the text beside its last tool call may be. A context without
    :data:`TERMINATION_KEY` (a single reply, an environment that does not
    record one) is not affected.
    """
    if not isinstance(context, Mapping) or TERMINATION_KEY not in context:
        return None
    termination = context.get(TERMINATION_KEY)
    if termination == FINAL_ANSWER_TERMINATION:
        return None
    return str(termination)


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


def reward_score(result: Any, reward_fn: Any = None) -> float:
    """The finite score of a reward result: a ``RewardResult`` (anything with
    ``.score``, or its ``.total_reward`` alias), a mapping with ``"score"`` (or
    ``"total_reward"``), or a real scalar (Python, NumPy or a 0-d tensor).

    Raises ``TypeError`` for anything else (``None``, a string, a list) and
    ``ValueError`` for a non-finite score: neither is ever replaced by a
    stand-in value. With ``reward_fn``, a score outside the range that reward
    declares (``score_range = (low, high)``) raises ``ValueError`` too; it is
    never clamped. Every caller (environments, ``Environment.get_reward``,
    ``score_episode``, ``score_response``) uses this one parser.
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
    if reward_fn is not None:
        check_score_range(reward_fn, score)
    return score


#: Rounding slack allowed at the edges of a declared score range.
_RANGE_SLACK = 1e-9


def declared_score_range(reward_fn: Any) -> tuple[float, float] | None:
    """The ``(low, high)`` range ``reward_fn`` declares, if it declares one."""
    declared = getattr(reward_fn, "score_range", None)
    if not isinstance(declared, tuple) or len(declared) != 2:
        return None
    low, high = (_real(bound) for bound in declared)
    if low is None or high is None:
        return None
    return low, high


def check_score_range(reward_fn: Any, score: float) -> float:
    """``score``, or ``ValueError`` when it lies outside ``reward_fn``'s
    declared ``score_range`` (docs/REWARD_CONTRACT.md: never clamped)."""
    declared = declared_score_range(reward_fn)
    if declared is None:
        return score
    low, high = declared
    slack = _RANGE_SLACK * max(1.0, abs(low), abs(high))
    if not low - slack <= score <= high + slack:
        name = getattr(reward_fn, "name", None) or type(reward_fn).__name__
        raise ValueError(
            f"reward {name} scored {score}, outside its declared score_range "
            f"[{low}, {high}]"
        )
    return score


#: The range a composer holds a child reward to when the child declares none
#: (docs/REWARD_CONTRACT.md: a score is in ``[0, 1]`` unless declared).
UNIT_SCORE_RANGE: tuple[float, float] = (0.0, 1.0)


def check_child_score_range(
    reward_fn: Any,
    score: float,
    default: tuple[float, float] | None = UNIT_SCORE_RANGE,
) -> float:
    """``score``, or ``ValueError`` when a composer's child scored outside its
    declared ``score_range`` or, when it declares none, outside ``default``
    (``[0, 1]``; ``None`` leaves an undeclared child unchecked).

    :func:`check_score_range` leaves an undeclared top-level reward
    unchecked; a composer cannot, because it would weight and sum a child's
    out-of-range score (a 7.0) silently. Never clamps."""
    if declared_score_range(reward_fn) is not None or default is None:
        return check_score_range(reward_fn, score)
    low, high = default
    slack = _RANGE_SLACK * max(1.0, abs(low), abs(high))
    if not low - slack <= score <= high + slack:
        name = getattr(reward_fn, "name", None) or type(reward_fn).__name__
        raise ValueError(
            f"reward {name} scored {score}, outside [{low}, {high}], the range "
            "a composite holds a child to when it declares no score_range "
            "(declare score_range on the reward if it scores another range)"
        )
    return score


def reward_fallback(result: Any) -> str | None:
    """The fallback reason a reward result is flagged with, if any."""
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, Mapping) and metadata.get(REWARD_FALLBACK_KEY):
        return str(metadata[REWARD_FALLBACK_KEY])
    return None


def failed_components(result: Any) -> list[str]:
    """The components a composer left out of ``result`` during an outage."""
    metadata = getattr(result, "metadata", None)
    failed = (
        metadata.get(COMPONENT_FAILURES_KEY) if isinstance(metadata, Mapping) else None
    )
    if not isinstance(failed, (list, tuple)):
        return []
    return [str(name) for name in failed]


def note_fallback(metadata: dict[str, Any], result: Any) -> None:
    """Record ``result``'s fallback flag and left-out components (if any) in
    episode ``metadata``."""
    reason = reward_fallback(result)
    if reason is not None:
        metadata[REWARD_FALLBACK_KEY] = reason
    failed = failed_components(result)
    if failed:
        metadata[FAILED_COMPONENTS_KEY] = failed


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
    scenario under ``"scenario"``. ``scenario`` may also be an environment
    state that carries its scenario under ``"scenario"`` or ``"task"``; the
    state is then laid over the scenario as in :func:`build_reward_context`. ``GSPOTrainer``, ``GSPOTokenTrainer`` and
    the TRL reward bridge build their context here, so scenario labels such
    as ``ground_truth`` reach the reward as they do in a multi-turn episode.
    """
    mapping = dict(scenario or {})
    nested = _resolve_scenario(mapping, None)
    if nested is not None:
        # An environment state that carries its scenario (single-turn
        # trainers pass the reset state's context): lay it over the scenario
        # exactly as an episode's reward context does.
        context = build_reward_context(mapping, nested, reward_fn=reward_fn)
        context.setdefault("user_query", str(prompt))
        return context
    query = {"user_query": str(prompt), **mapping}
    return build_reward_context(None, query, reward_fn=reward_fn)


def _prompt_text(prompt: Any) -> str:
    """The user query of a prompt: the text, or a conversation's last user turn."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, Sequence):
        users = [t for t in normalize_turns(prompt) if t.role == "user"]
        return str(users[-1].content) if users else ""
    return str(prompt)


def response_turns(prompt: Any, response: Any) -> list[ConversationTurn]:
    """The conversation one reply to one prompt forms: ``[user, assistant]``.

    ``prompt`` is the user's text, or the chat messages so far (system and
    user turns, as TRL's conversational format passes them); ``response`` is
    the reply's text or its assistant ``ConversationTurn`` (kept as is, with
    any generated token ids). The user turn is always included, as it is in
    an episode, so rewards that read the request (correctness, grounding,
    judges) see it on every path.
    """
    if isinstance(prompt, str):
        turns = [ConversationTurn(role="user", content=prompt)]
    elif isinstance(prompt, Sequence):
        turns = normalize_turns(prompt)
    else:
        turns = [ConversationTurn(role="user", content=str(prompt))]
    if isinstance(response, ConversationTurn):
        reply = response
    else:
        reply = ConversationTurn(role="assistant", content=str(response))
    return [*turns, reply]


async def score_response(
    reward_fn: Any,
    prompt: Any,
    response: Any,
    scenario: Mapping[str, Any] | None = None,
) -> float:
    """Score one reply to one prompt exactly as a one-turn episode is scored.

    The single-response paths (``SingleTurnGRPOTrainer``, ``GSPOTrainer``,
    ``GSPOTokenTrainer``, the TRL reward bridge, ``ComputationalGRPOEngine``
    and the auto-research adapters) all score here: the turns are
    :func:`response_turns`, the context is :func:`response_reward_context`
    (scenario labels lifted, ``user_query`` set), and IMMEDIATE/DENSE rewards
    get the latest-turn scope an environment step would give them. The score
    is parsed by :func:`reward_score`, so a failure, a ``None`` or a
    non-finite score raises; nothing stands in for it.
    """
    turns = response_turns(prompt, response)
    context = response_reward_context(
        _prompt_text(prompt), scenario, reward_fn=reward_fn
    )
    if _scores_per_step(reward_fn, None):
        context[REWARD_SCOPE_KEY] = LATEST_TURN_SCOPE
    return reward_score(await _call(reward_fn, turns, context), reward_fn)


async def score_trajectory(
    reward_fn: Any,
    trajectory: Any,
    scenario: Mapping[str, Any] | None = None,
) -> float:
    """Score a finished trajectory's whole conversation once.

    The context is :func:`trajectory_reward_context` (the recorded final state
    and scenario), the same context :func:`score_episode` gives a
    CUMULATIVE/SPARSE reward, also for rewards with
    ``requires_environment_context``. ``Environment.get_reward`` scores here.
    Nothing is replayed per step and the trajectory's rewards are not
    modified; a failure, a ``None`` or a non-finite score raises. A fallback
    score is flagged on the trajectory's metadata as in :func:`score_episode`
    (:data:`REWARD_FALLBACK_KEY`, :data:`FAILED_COMPONENTS_KEY`).
    """
    turns = getattr(trajectory, "turns", None)
    if not turns and hasattr(trajectory, "__iter__"):
        turns = list(trajectory)
    context = trajectory_reward_context(trajectory, scenario, reward_fn=reward_fn)
    result = await _call(reward_fn, normalize_turns(turns), context)
    score = reward_score(result, reward_fn)
    metadata = getattr(trajectory, "metadata", None)
    if isinstance(metadata, dict):
        note_fallback(metadata, result)
    return score


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
            # The environment recorded an outage; the episode has no reward.
            raise RewardUnavailableError(f"Episode reward failed: {error}")
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
    failed: list[str] = []
    if _scores_per_step(reward_fn, environment) and assistant:
        # Replay each step against the state it was scored at, recorded by
        # the environment in state_history; fall back to the final state.
        states = getattr(trajectory, "state_history", None)
        per_step_states = (
            states
            if isinstance(states, list) and len(states) == len(assistant)
            else None
        )
        final_step_context = {**context, REWARD_SCOPE_KEY: LATEST_TURN_SCOPE}
        scores = []
        for step, index in enumerate(assistant):
            step_context = (
                build_reward_context(
                    per_step_states[step],
                    _trajectory_scenario(trajectory, scenario),
                    reward_fn=reward_fn,
                    latest_turn_only=True,
                )
                if per_step_states is not None
                and isinstance(per_step_states[step], Mapping)
                else final_step_context
            )
            result = await _call(reward_fn, turns[: index + 1], step_context)
            scores.append(reward_score(result, reward_fn))
            breakdown = getattr(result, "breakdown", None)
            fallback = reward_fallback(result) or fallback
            failed = failed_components(result) or failed
        parts = [score / len(scores) for score in scores]
        if len(parts) != len(intrinsic):
            # Steps and assistant turns do not line up; credit the mean once.
            parts = [0.0] * (len(intrinsic) - 1) + [math.fsum(parts)]
    else:
        result = await _call(reward_fn, turns, context)
        parts = [0.0] * (len(intrinsic) - 1) + [reward_score(result, reward_fn)]
        breakdown = getattr(result, "breakdown", None)
        fallback = reward_fallback(result)
        failed = failed_components(result)

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
        # The flags describe this score: drop any the environment's own
        # (different) reward left before recording this reward's.
        trajectory.metadata.pop(REWARD_FALLBACK_KEY, None)
        trajectory.metadata.pop(FAILED_COMPONENTS_KEY, None)
        if fallback is not None:
            trajectory.metadata[REWARD_FALLBACK_KEY] = fallback
        if failed:
            trajectory.metadata[FAILED_COMPONENTS_KEY] = failed
    return total


__all__ = [
    "COMPONENT_FAILURES_KEY",
    "FAILED_COMPONENTS_KEY",
    "FINAL_ANSWER_TERMINATION",
    "LATEST_TURN_SCOPE",
    "RewardUnavailableError",
    "REWARD_FALLBACK_KEY",
    "REWARD_SCOPE_KEY",
    "TERMINATION_KEY",
    "UNIT_SCORE_RANGE",
    "assistant_turns_in_scope",
    "build_reward_context",
    "check_child_score_range",
    "check_score_range",
    "declared_score_range",
    "environment_applied_reward",
    "failed_components",
    "normalize_turns",
    "note_fallback",
    "reward_fallback",
    "requires_environment_context",
    "response_reward_context",
    "response_turns",
    "reward_score",
    "score_episode",
    "score_response",
    "score_trajectory",
    "scores_latest_turn_only",
    "trajectory_reward_context",
    "unanswered_termination",
]
