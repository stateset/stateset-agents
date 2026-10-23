"""Reward scorers: the one adapter layer native trainers' rollouts score through.

Built only on the public helpers in :mod:`stateset_agents.core.reward_context`
(no parallel scoring path):

* :class:`EpisodeScorer` -- a finished episode's one reward
  (``score_episode``): ``MultiTurnGRPOTrainer``, the distributed trainers.
* :class:`ResponseScorer` -- a reply to a prompt, scored exactly as a one-turn
  episode (``score_response``: ``[user, assistant]`` turns, the standard
  reward context, latest-turn scope for IMMEDIATE/DENSE rewards):
  ``SingleTurnGRPOTrainer``, ``GSPOTrainer``, ``GSPOTokenTrainer`` and the TRL
  reward bridge.
* :class:`CallableScorer` -- a plain ``reward_fn(prompt, response)`` callable,
  sync or async: ``DAPOTrainer``, ``GEPOTrainer``, ``VAPOTrainer``.

Scorers never absorb a failure. What a trainer drops (only
``RewardUnavailableError``) and when an all-dropped step raises stays the
trainer's rollout policy.

A reward may also score with an opted-in fallback (a judge outage scored
by ``error_score``, a renormalising composite): its result is flagged
(``reward_context.REWARD_FALLBACK_KEY``, or left-out ``failed_components``)
and the episode must never be trained on. ``score_checked`` returns a
:class:`ScoredReply` carrying that flag; :func:`trajectory_reward_fallback`
reads it from an episode scored by ``score_episode``.

Importing this module does not import ``torch``.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import sys
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from stateset_agents.core.reward_context import (
    FAILED_COMPONENTS_KEY,
    REWARD_FALLBACK_KEY,
    failed_components,
    reward_fallback,
    reward_score,
    score_episode,
    score_response,
)


def reward_outage_exceptions() -> tuple[type[BaseException], ...]:
    """Failures of a reward-model prediction that are outages, not bugs: I/O
    (``OSError``, which includes connection errors), memory exhaustion (CPU,
    or CUDA when ``torch`` is loaded) and timeouts. A learned reward turns
    them into ``RewardUnavailableError``; any other exception propagates."""
    outages: list[type[BaseException]] = [OSError, MemoryError, asyncio.TimeoutError]
    cuda = getattr(sys.modules.get("torch"), "cuda", None)
    oom = getattr(cuda, "OutOfMemoryError", None)
    if isinstance(oom, type) and issubclass(oom, BaseException):
        outages.append(oom)
    return tuple(outages)


class ScoredReply(NamedTuple):
    """One reply's reward and, for an opted-in fallback score, its reason."""

    score: float
    #: Why the score is a fallback (never train on it); ``None`` if genuine.
    fallback: str | None = None


def result_reward_fallback(result: Any) -> str | None:
    """A reward result's fallback reason: its ``reward_fallback`` flag, or the
    components a composer left out of it (``failed_components``)."""
    reason = reward_fallback(result)
    if reason is not None:
        return reason
    left_out = failed_components(result)
    return f"scored without {', '.join(left_out)}" if left_out else None


def trajectory_reward_fallback(trajectory: Any) -> str | None:
    """A scored episode's fallback reason (``REWARD_FALLBACK_KEY`` or
    ``FAILED_COMPONENTS_KEY`` in its metadata), or ``None``."""
    metadata = getattr(trajectory, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    reason = metadata.get(REWARD_FALLBACK_KEY)
    if reason:
        return str(reason)
    left_out = metadata.get(FAILED_COMPONENTS_KEY)
    if isinstance(left_out, (list, tuple)) and left_out:
        return f"scored without {', '.join(str(name) for name in left_out)}"
    return None


class _ResultRecorder:
    """Stands in for a reward function so ``score_response`` (the one
    single-response scoring path) is reused unchanged while the raw result,
    and so its fallback flag, is kept. Every other attribute is the reward
    function's own."""

    def __init__(self, reward_fn: Any) -> None:
        self._reward_fn = reward_fn
        self.result: Any = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._reward_fn, name)

    def compute_reward(self, turns: Any, context: Any = None) -> Any:
        result = self._reward_fn.compute_reward(turns, context)
        if inspect.isawaitable(result):
            return self._keep(result)
        self.result = result
        return result

    async def _keep(self, pending: Any) -> Any:
        self.result = await pending
        return self.result


class EpisodeScorer:
    """Settle a finished episode's reward (``reward_context.score_episode``).

    The same trajectory earns the same reward on every path that scores
    episodes: training rollouts, evaluation and the distributed trainers.
    ``trajectory.total_reward`` (and its turn rewards and breakdown) are
    updated; the total is returned.
    """

    def __init__(self, reward_fn: Any, environment: Any = None) -> None:
        self.reward_fn = reward_fn
        self.environment = environment

    async def score(
        self, trajectory: Any, scenario: Mapping[str, Any] | None = None
    ) -> float:
        return await score_episode(
            trajectory, self.reward_fn, self.environment, scenario
        )


class ResponseScorer:
    """Score replies to a prompt as one-turn episodes (``score_response``).

    ``prompt`` is the user's text or the conversation so far; a reply is its
    text or its assistant ``ConversationTurn``. ``scenario`` carries the
    labels (``ground_truth``...) the reward context lifts to the top level.
    """

    def __init__(self, reward_fn: Any) -> None:
        self.reward_fn = reward_fn

    async def score(
        self,
        prompt: Any,
        response: Any,
        scenario: Mapping[str, Any] | None = None,
    ) -> float:
        return await score_response(self.reward_fn, prompt, response, scenario)

    async def score_checked(
        self,
        prompt: Any,
        response: Any,
        scenario: Mapping[str, Any] | None = None,
    ) -> ScoredReply:
        """:meth:`score` plus the result's fallback flag (:class:`ScoredReply`)."""
        recorder = _ResultRecorder(self.reward_fn)
        score = await score_response(recorder, prompt, response, scenario)
        return ScoredReply(score, result_reward_fallback(recorder.result))

    async def score_group(
        self,
        prompt: Any,
        responses: Sequence[Any],
        scenario: Mapping[str, Any] | None = None,
    ) -> list[float]:
        """Every reply's score, in order; the replies are scored concurrently."""
        return list(
            await asyncio.gather(
                *(self.score(prompt, response, scenario) for response in responses)
            )
        )


class CallableScorer:
    """Score replies with a ``reward_fn(prompt, response)`` callable.

    The callable may return a number, a ``RewardResult`` (whose fallback flag
    :meth:`score_checked` reports) or an awaitable of either (async reward
    functions). A NaN or infinite score raises ``ValueError``, as
    ``reward_context.reward_score`` does for ``RewardFunction`` results:
    nothing stands in for it.
    """

    def __init__(self, reward_fn: Any) -> None:
        self.reward_fn = reward_fn

    async def score(self, prompt: str, response: str) -> float:
        return (await self.score_checked(prompt, response)).score

    async def score_checked(self, prompt: str, response: str) -> ScoredReply:
        """The reply's score and, for a flagged ``RewardResult``, its reason."""
        reward = self.reward_fn(prompt, response)
        if inspect.isawaitable(reward):
            reward = await reward
        if hasattr(reward, "score") or isinstance(reward, Mapping):
            return ScoredReply(
                reward_score(reward, self.reward_fn), result_reward_fallback(reward)
            )
        score = float(reward)
        if not math.isfinite(score):
            raise ValueError(f"reward score must be finite, got {score}")
        return ScoredReply(score)


__all__ = [
    "CallableScorer",
    "EpisodeScorer",
    "ResponseScorer",
    "ScoredReply",
    "result_reward_fallback",
    "reward_outage_exceptions",
    "trajectory_reward_fallback",
]
