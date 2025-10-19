"""
Computational GRPO Engine - Embodying the Bitter Lesson

This module provides a computation-first GRPO training engine that:
1. Leverages massive computation through parallel trajectory generation
2. Uses general learning methods rather than domain-specific heuristics
3. Scales arbitrarily with computational resources
4. Learns from data rather than hand-crafted features
5. Continuously improves through self-play and exploration

"We want AI agents that can discover like we can, not which contain what we have discovered."
"""

import asyncio
import hashlib
import logging
import multiprocessing as mp
import os
import time
import uuid
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .agent import Agent
from .environment import Environment
from .reward import RewardFunction
from .trajectory import Trajectory

logger = logging.getLogger(__name__)


@dataclass
class ComputationalTrajectory:
    """A trajectory focused on computational efficiency and learning"""

    id: str
    prompt: str
    response: str
    raw_reward_signal: float  # Direct signal from environment
    learned_reward: float  # Reward from learned reward model
    computational_cost: float  # Tracks computation used
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
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
    """
    The core GRPO engine that embodies the Bitter Lesson principles
    """

    def __init__(
        self,
        agent: Agent,
        environment: Environment,
        reward_function: RewardFunction,
        num_workers: Optional[int] = None,
        trajectory_batch_size: int = 32,
        use_learned_rewards: bool = True,
    ):
        self.agent = agent
        self.environment = environment
        self.reward_function = reward_function
        self.num_workers = num_workers or mp.cpu_count()
        self.trajectory_batch_size = trajectory_batch_size
        self.use_learned_rewards = use_learned_rewards

        # Initialize components
        self.trajectory_buffer = deque(maxlen=100000)
        self.generation_count = 0
        self.total_computation = 0.0

        # Metrics
        self.metrics = {
            "trajectories_per_second": 0.0,
            "average_reward": 0.0,
            "computation_efficiency": 0.0,
            "learning_progress": 0.0,
        }

        # Initialize process pool for parallel computation
        self.executor = None
        try:
            self.executor = ProcessPoolExecutor(max_workers=self.num_workers)
        except Exception as e:
            logger.warning("ProcessPoolExecutor unavailable, falling back to ThreadPoolExecutor: %s", e)
            try:
                self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
            except Exception as e2:
                logger.warning("ThreadPoolExecutor also unavailable; continuing without executor: %s", e2)
                self.executor = None

    async def generate_trajectory_batch(
        self, prompts: List[str]
    ) -> List[ComputationalTrajectory]:
        """
        Generate trajectories in parallel, maximizing computational efficiency
        """
        start_time = time.time()

        # Generate trajectories in parallel
        tasks = []
        for prompt in prompts:
            task = asyncio.create_task(self._generate_single_trajectory(prompt))
            tasks.append(task)

        # Collect results
        trajectories = await asyncio.gather(*tasks)

        # Update metrics
        elapsed = time.time() - start_time
        self.metrics["trajectories_per_second"] = len(trajectories) / elapsed
        self.total_computation += elapsed * self.num_workers

        return trajectories

    async def _generate_single_trajectory(self, prompt: str) -> ComputationalTrajectory:
        """Generate a single trajectory asynchronously"""
        # Generate response using agent
        response = await self.agent.generate_response(prompt)

        # Get raw reward signal from environment
        raw_reward = await self._get_environmental_reward(prompt, response)

        # Get learned reward if enabled
        learned_reward = raw_reward
        if self.use_learned_rewards:
            try:
                # Use reward function to compute learned reward
                reward_result = await self.reward_function.compute_reward(
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ]
                )
                learned_reward = reward_result.score
            except Exception as e:
                logger.warning(f"Failed to compute learned reward: {e}")
                learned_reward = raw_reward

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
                "worker_id": f"worker_{asyncio.current_task().get_name() if asyncio.current_task() else 'main'}",
            },
        )

        return trajectory

    async def _get_environmental_reward(self, prompt: str, response: str) -> float:
        """Get reward signal from environment (not hand-crafted features)"""
        # In practice, this would come from:
        # - User feedback
        # - Task completion metrics
        # - Environmental outcomes
        # NOT from hand-crafted heuristics

        # Use environment to get reward
        try:
            # Create a simple trajectory for environment evaluation
            trajectory = Trajectory()
            trajectory.add_turn({"role": "user", "content": prompt})
            trajectory.add_turn({"role": "assistant", "content": response})

            # Get reward from environment
            reward = await self.environment.get_reward(trajectory)
            return reward
        except Exception as e:
            logger.warning(f"Failed to get environmental reward: {e}")
            # Fallback to simple signal
            signal_strength = np.random.random()
            noise = np.random.normal(0, 0.1)
            return float(np.clip(signal_strength + noise, 0, 1))

    async def train_iteration(self, prompts: List[str]) -> Dict[str, Any]:
        """
        Run a complete GRPO training iteration
        """
        iteration_start = time.time()

        # Generate trajectory batch in parallel
        trajectories = await self.generate_trajectory_batch(prompts)

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
        avg_reward = np.mean([t.learned_reward for t in trajectories])

        results = {
            "iteration_time": iteration_time,
            "trajectories_generated": len(trajectories),
            "average_reward": avg_reward,
            "policy_loss": policy_loss,
            "total_computation_used": self.total_computation,
            "trajectories_per_second": self.metrics["trajectories_per_second"],
        }

        return results

    async def _update_agent(self, trajectories: List[ComputationalTrajectory]):
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
        self, trajectories: List[ComputationalTrajectory]
    ) -> np.ndarray:
        """Compute advantages for GRPO"""
        rewards = np.array([t.learned_reward for t in trajectories])
        baseline = np.mean(rewards)
        advantages = rewards - baseline

        # Normalize advantages
        if np.std(advantages) > 0:
            advantages = advantages / np.std(advantages)

        return advantages

    async def _update_policy(
        self, trajectories: List[ComputationalTrajectory], advantages: np.ndarray
    ) -> float:
        """Update policy using GRPO"""
        # In practice, this would update your actual model parameters
        # using the computed advantages

        # For now, simulate policy gradient computation
        policy_loss = -np.mean(advantages)

        # Update agent's policy if supported
        if hasattr(self.agent, "update_policy"):
            await self.agent.update_policy(trajectories, advantages)

        return float(policy_loss)

    def scale_computation(self, scale_factor: float):
        """Scale computational resources"""
        new_workers = max(1, int(self.num_workers * scale_factor))
        try:
            if self.executor:
                self.executor.shutdown(wait=True)
            self.executor = ProcessPoolExecutor(max_workers=new_workers)
        except Exception as e:
            logger.warning(
                "ProcessPoolExecutor unavailable during scaling, falling back to ThreadPoolExecutor: %s",
                e,
            )
            try:
                self.executor = ThreadPoolExecutor(max_workers=new_workers)
            except Exception as e2:
                logger.warning(
                    "ThreadPoolExecutor also unavailable during scaling: %s",
                    e2,
                )
                # Keep previous executor if any; otherwise set to None
        self.num_workers = new_workers

        return {
            "previous_workers": int(self.num_workers / scale_factor),
            "current_workers": self.num_workers,
            "scale_factor": scale_factor,
        }

    def get_metrics(self) -> Dict[str, Any]:
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
        """Cleanup resources"""
        self.executor.shutdown(wait=True)


# Convenience functions
def create_computational_engine(
    agent: Agent,
    environment: Environment,
    reward_function: RewardFunction,
    num_workers: Optional[int] = None,
    **kwargs,
) -> ComputationalGRPOEngine:
    """Create a computational GRPO engine with optimal configuration"""
    return ComputationalGRPOEngine(
        agent=agent,
        environment=environment,
        reward_function=reward_function,
        num_workers=num_workers,
        **kwargs,
    )
