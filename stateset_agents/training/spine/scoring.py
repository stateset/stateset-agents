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

Importing this module does not import ``torch``.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Mapping, Sequence
from typing import Any

from stateset_agents.core.reward_context import score_episode, score_response


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

    The callable may return a number or an awaitable of one (async reward
    functions); the result is converted with ``float``.
    """

    def __init__(self, reward_fn: Any) -> None:
        self.reward_fn = reward_fn

    async def score(self, prompt: str, response: str) -> float:
        reward = self.reward_fn(prompt, response)
        if inspect.isawaitable(reward):
            reward = await reward
        return float(reward)


__all__ = ["CallableScorer", "EpisodeScorer", "ResponseScorer"]
