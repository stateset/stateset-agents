"""
Parallel trajectory generation and scoring (deprecated; it does not train).

:class:`ComputationalGRPOEngine` generates responses for a batch of prompts
in parallel, scores them through the shared reward path
(:func:`~stateset_agents.core.reward_context.score_response`) and computes
group-relative advantages. It does **not** update the policy: its
``_update_policy`` runs a forward pass per trajectory, reports the mean
advantage-weighted NLL as ``policy_loss`` and never calls ``backward()`` or
an optimizer step (without torch, a model or a tokenizer the reported value
is ``-mean(advantages)``). The only parameter changes come from an agent's own
``update_policy`` / ``update_from_feedback`` hooks, if it defines them.

Deprecated in v0.70.0 and removed in v0.71.0. To train, use the trainers in
:mod:`stateset_agents.training` -- ``stateset_agents.train(...)`` or
``MultiTurnGRPOTrainer`` / ``SingleTurnGRPOTrainer`` -- which run real GRPO
updates.
"""

import asyncio
import logging
import math
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._deprecation import warn_deprecated
from ..exceptions import LOSS_EXCEPTIONS
from .agent import Agent
from .environment import Environment
from .reward import RewardFunction
from .reward_context import RewardUnavailableError, score_response
from .trajectory import Trajectory

logger = logging.getLogger(__name__)

_REMOVAL = "0.71.0"
_REPLACEMENT = (
    "stateset_agents.train() or stateset_agents.training.MultiTurnGRPOTrainer "
    "(this engine never applies a gradient update)"
)


@dataclass
class ComputationalTrajectory:
    """A trajectory focused on computational efficiency and learning"""

    id: str
    prompt: str
    response: str
    raw_reward_signal: float | None  # Environment signal (None: no reward_fn)
    learned_reward: float  # Reward from learned reward model
    computational_cost: float  # Tracks computation used
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "response": self.response,
            "raw_reward_signal": self.raw_reward_signal,
            "learned_reward": self.learned_reward,
            "computational_cost": self.computational_cost,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class ComputationalGRPOEngine:
    """Parallel rollout-and-score loop that does not train the policy.

    Deprecated (removed in v0.71.0): see the module docstring. Use
    ``stateset_agents.train()`` or the trainers in
    :mod:`stateset_agents.training`.
    """

    def __init__(
        self,
        agent: Agent,
        environment: Environment,
        reward_function: RewardFunction,
        num_workers: int | None = None,
        trajectory_batch_size: int = 32,
        use_learned_rewards: bool = True,
    ):
        warn_deprecated(
            "stateset_agents.core.computational_engine.ComputationalGRPOEngine",
            removal=_REMOVAL,
            replacement=_REPLACEMENT,
        )
        self.agent = agent
        self.environment = environment
        self.reward_function = reward_function
        self.num_workers = num_workers or (os.cpu_count() or 1)
        self._semaphore = asyncio.Semaphore(self.num_workers)
        self.trajectory_batch_size = trajectory_batch_size
        self.use_learned_rewards = use_learned_rewards

        # Initialize components
        self.trajectory_buffer: deque[ComputationalTrajectory] = deque(maxlen=100000)
        self.generation_count = 0
        self.total_computation = 0.0

        # Metrics
        self.metrics = {
            "trajectories_per_second": 0.0,
            "average_reward": 0.0,
            "computation_efficiency": 0.0,
            "learning_progress": 0.0,
        }

    async def generate_trajectory_batch(
        self, prompts: list[str]
    ) -> list[ComputationalTrajectory]:
        """
        Generate trajectories in parallel, maximizing computational efficiency
        """
        start_time = time.time()

        # Generate trajectories in parallel
        tasks = []
        for prompt in prompts:
            task = asyncio.create_task(self._generate_single_trajectory(prompt))
            tasks.append(task)

        # Collect results; a trajectory whose reward is unavailable (an
        # outage) is dropped and counted, never kept with a stand-in reward.
        # Any other error (a bug, bad data, a non-finite reward) propagates.
        results = await asyncio.gather(*tasks, return_exceptions=True)
        trajectories: list[ComputationalTrajectory] = []
        failed = 0
        for result in results:
            if isinstance(result, RewardUnavailableError):
                failed += 1
                logger.warning(
                    "Dropping trajectory: its reward is unavailable: %s", result
                )
            elif isinstance(result, BaseException):
                raise result
            else:
                trajectories.append(result)
        self.metrics["failed_trajectories"] = float(failed)

        # Update metrics
        # Windows clocks can report zero elapsed time for fast batches.
        elapsed = max(time.time() - start_time, 1e-9)
        self.metrics["trajectories_per_second"] = len(trajectories) / elapsed
        self.total_computation += elapsed * self.num_workers

        return trajectories

    async def _generate_single_trajectory(self, prompt: str) -> ComputationalTrajectory:
        """Generate a single trajectory asynchronously"""
        async with self._semaphore:
            # Generate response using agent
            response = await self.agent.generate_response(prompt)

            # Raw reward signal from the environment, when it has one.
            raw_reward = await self._get_environmental_reward(prompt, response)

            # The learned reward is the training signal. A failure propagates
            # (the batch drops this trajectory); no other value stands in.
            if self.use_learned_rewards:
                # The shared single-response path (score_response).
                learned_reward = await score_response(
                    self.reward_function, prompt, response
                )
            elif raw_reward is None:
                raise ValueError(
                    "use_learned_rewards=False needs an environment reward_fn"
                )
            else:
                learned_reward = raw_reward
            if not math.isfinite(learned_reward):
                raise ValueError(f"reward must be finite, got {learned_reward}")

            # Track computational cost
            computational_cost = len(prompt) + len(response)  # Simplified

            trajectory = ComputationalTrajectory(
                id=str(uuid.uuid4()),
                prompt=prompt,
                response=response,
                raw_reward_signal=raw_reward,
                learned_reward=learned_reward,
                computational_cost=computational_cost,
                timestamp=datetime.now(),
                metadata={
                    "generation_method": "parallel",
                    "worker_id": self._get_worker_id(),
                },
            )

            return trajectory

    async def _get_environmental_reward(
        self, prompt: str, response: str
    ) -> float | None:
        """Get reward signal from environment (not hand-crafted features)"""
        # In practice, this would come from:
        # - User feedback
        # - Task completion metrics
        # - Environmental outcomes
        # NOT from hand-crafted heuristics

        # An environment without a reward function has no raw signal; that is
        # recorded as None rather than a fabricated neutral score.
        if (
            getattr(self.environment, "reward_fn", None) is None
            and type(self.environment).get_reward is Environment.get_reward
        ):
            return None

        trajectory = Trajectory()
        trajectory.add_turn({"role": "user", "content": prompt})
        trajectory.add_turn({"role": "assistant", "content": response})
        # Failures propagate: the batch drops this trajectory.
        return float(await self.environment.get_reward(trajectory))

    async def train_iteration(self, prompts: list[str]) -> dict[str, Any]:
        """Generate, score and compute advantages for one batch of prompts.

        Despite the name no policy update happens here (see
        :meth:`_update_policy`); ``policy_loss`` in the result is a diagnostic.
        """
        iteration_start = time.time()

        # Generate trajectory batch in parallel
        trajectories = await self.generate_trajectory_batch(prompts)
        if prompts and not trajectories:
            raise RuntimeError(
                "No trajectory in the batch has a reward: all "
                f"{len(prompts)} reward computations failed"
            )

        # Add to buffer
        self.trajectory_buffer.extend(trajectories)

        # Update agent from all trajectories
        if len(self.trajectory_buffer) >= self.trajectory_batch_size:
            await self._update_agent(trajectories)

        # Compute advantages using learned rewards
        advantages = self._compute_advantages(trajectories)

        # Update policy
        policy_loss = await self._update_policy(trajectories, advantages)

        # Calculate metrics
        iteration_time = time.time() - iteration_start
        avg_reward = (
            float(np.mean([t.learned_reward for t in trajectories]))
            if trajectories
            else 0.0
        )

        results = {
            "iteration_time": iteration_time,
            "trajectories_generated": len(trajectories),
            "trajectories_failed": int(self.metrics.get("failed_trajectories", 0)),
            "average_reward": avg_reward,
            "policy_loss": policy_loss,
            "total_computation_used": self.total_computation,
            "trajectories_per_second": self.metrics["trajectories_per_second"],
        }

        return results

    async def _update_agent(self, trajectories: list[ComputationalTrajectory]):
        """Update the agent from trajectory feedback"""
        # Convert trajectories to agent-compatible format
        training_data = []
        for traj in trajectories:
            training_data.append(
                {
                    "prompt": traj.prompt,
                    "response": traj.response,
                    "reward": traj.learned_reward,
                    "metadata": traj.metadata,
                }
            )

        # Update agent if it supports online learning
        if hasattr(self.agent, "update_from_feedback"):
            await self.agent.update_from_feedback(training_data)

    def _compute_advantages(
        self, trajectories: list[ComputationalTrajectory]
    ) -> NDArray[np.float64]:
        """Compute advantages for GRPO"""
        rewards = np.array([t.learned_reward for t in trajectories], dtype=float)
        if rewards.size == 0:
            return np.zeros(0, dtype=float)
        baseline = np.mean(rewards)
        advantages = rewards - baseline

        # Normalize advantages
        if np.std(advantages) > 0:
            advantages = advantages / np.std(advantages)

        normalized_advantages: NDArray[np.float64] = np.asarray(advantages, dtype=float)
        return normalized_advantages

    def _get_worker_id(self) -> str:
        """Return a stable worker identifier for async trajectory generation."""
        current_task = asyncio.current_task()
        if current_task is None:
            return "worker_main"
        return f"worker_{current_task.get_name()}"

    async def _update_policy(
        self, trajectories: list[ComputationalTrajectory], advantages: np.ndarray
    ) -> float:
        """Report an advantage-weighted NLL; this does **not** update the policy.

        Each trajectory gets one forward pass and ``advantage * loss`` is
        averaged into the returned value with ``.item()``: no ``backward()``,
        no optimizer step, so the model's weights never change here. Without
        torch, a model or a tokenizer the return value is ``-mean(advantages)``
        (not a loss). An agent's own ``update_policy(trajectories,
        advantages)`` hook, if it has one, is called either way.

        A forward pass that fails at runtime (``LOSS_EXCEPTIONS``: OOM, shape
        or numeric errors) skips that trajectory; a ``TypeError`` /
        ``KeyError`` / ``AttributeError`` is a bug and propagates.
        """
        try:
            # Lazy import torch for optional dependency
            import torch
        except ImportError:
            logger.warning("PyTorch not available, using simulated policy update")
            # Fallback to agent's custom update if available
            if hasattr(self.agent, "update_policy"):
                await self.agent.update_policy(trajectories, advantages)
            return float(-np.mean(advantages))

        # Check if agent has a model we can update
        if not hasattr(self.agent, "model") or self.agent.model is None:
            logger.warning("Agent has no model to update")
            if hasattr(self.agent, "update_policy"):
                await self.agent.update_policy(trajectories, advantages)
            return float(-np.mean(advantages))

        # Actual policy gradient update
        model = self.agent.model
        tokenizer = self.agent.tokenizer if hasattr(self.agent, "tokenizer") else None

        if tokenizer is None:
            logger.warning("No tokenizer available for policy update")
            return float(-np.mean(advantages))

        model.train()

        # Get device - try from model attribute first, then from parameters
        if hasattr(model, "device"):
            device = (
                model.device
                if isinstance(model.device, torch.device)
                else torch.device(model.device)
            )
        else:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        total_loss = 0.0
        num_updates = 0

        # Process each trajectory
        for traj, advantage in zip(trajectories, advantages, strict=False):
            try:
                # Tokenize the conversation
                full_text = f"User: {traj.prompt}\nAssistant: {traj.response}"
                inputs = tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                )
                inputs = {
                    k: v.to(device) if hasattr(v, "to") else v
                    for k, v in inputs.items()
                }

                # Forward pass
                outputs = model(**inputs, labels=inputs["input_ids"])

                # Policy loss: -advantage * log_prob
                # The negative log likelihood is already computed in outputs.loss
                policy_loss = advantage * outputs.loss

                total_loss += policy_loss.item()
                num_updates += 1

                # No backward pass and no optimizer step: this value is a
                # diagnostic only (see the docstring).

            except LOSS_EXCEPTIONS as e:
                logger.warning(f"Failed to compute policy loss for trajectory: {e}")
                continue

        model.eval()

        avg_loss = total_loss / num_updates if num_updates > 0 else 0.0

        # Also call agent's update method if available
        if hasattr(self.agent, "update_policy"):
            await self.agent.update_policy(trajectories, advantages)

        return float(avg_loss)

    def scale_computation(self, scale_factor: float):
        """Scale computational resources"""
        previous_workers = self.num_workers
        new_workers = max(1, int(previous_workers * scale_factor))
        self.num_workers = new_workers
        self._semaphore = asyncio.Semaphore(new_workers)

        return {
            "previous_workers": previous_workers,
            "current_workers": self.num_workers,
            "scale_factor": scale_factor,
        }

    def get_metrics(self) -> dict[str, Any]:
        """Get engine metrics"""
        return {
            "engine_metrics": self.metrics,
            # Convenience top-level duplicates used by some tests
            "trajectories_per_second": self.metrics.get("trajectories_per_second", 0.0),
            "average_reward": self.metrics.get("average_reward", 0.0),
            "computation_efficiency": self.metrics.get("computation_efficiency", 0.0),
            "learning_progress": self.metrics.get("learning_progress", 0.0),
            "total_trajectories": len(self.trajectory_buffer),
            "computation_used": self.total_computation,
            "active_workers": self.num_workers,
            "philosophy_alignment": {
                "uses_learned_rewards": self.use_learned_rewards,
                "parallel_computation": True,
                "hand_crafted_features": False,
                "scales_with_compute": True,
            },
        }

    def cleanup(self):
        """Cleanup resources safely"""
        return None


# Convenience functions
def create_computational_engine(
    agent: Agent,
    environment: Environment,
    reward_function: RewardFunction,
    num_workers: int | None = None,
    **kwargs,
) -> ComputationalGRPOEngine:
    """Create a :class:`ComputationalGRPOEngine` (deprecated, removed in v0.71.0)."""
    return ComputationalGRPOEngine(
        agent=agent,
        environment=environment,
        reward_function=reward_function,
        num_workers=num_workers,
        **kwargs,
    )
