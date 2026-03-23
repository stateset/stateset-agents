"""Reusable Weights & Biases logger helpers."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

import numpy as np

from stateset_agents.core.agent_config import AgentConfig
from stateset_agents.core.trajectory import MultiTurnTrajectory, TrajectoryGroup

logger = logging.getLogger(__name__)


def _wandb_support():
    from . import wandb_integration as support

    return support


class WandBLogger:
    """Comprehensive W&B logger for GRPO agent training."""

    def __init__(
        self,
        project: str = "grpo-agent-framework",
        entity: str | None = None,
        api_key: str | None = None,
        enabled: bool = True,
        **kwargs,
    ):
        del kwargs
        self.project = project
        self.entity = entity
        self.enabled = bool(enabled)
        self.run = None
        self.start_time = None
        self.offline_mode = os.getenv("WANDB_MODE", "").lower() == "offline"

        support = _wandb_support()
        if self.enabled:
            if not support.WANDB_INSTALLED:
                logger.warning(
                    "wandb package not available. Install with: pip install wandb"
                )
                self.enabled = False
            elif not support._load_wandb():
                logger.warning("wandb import failed. W&B logging will be disabled.")
                self.enabled = False

        if api_key:
            os.environ["WANDB_API_KEY"] = api_key

        if self.enabled and not os.getenv("WANDB_API_KEY") and not self.offline_mode:
            logger.warning("WANDB_API_KEY not found. W&B logging will be disabled.")
            self.enabled = False

        logger.info("W&B Logger initialized (enabled: %s)", self.enabled)

    def init_run(
        self,
        config: dict[str, Any],
        name: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        group: str | None = None,
        job_type: str = "train",
    ):
        """Initialize a new W&B run."""
        if not self.enabled:
            return

        support = _wandb_support()
        try:
            clean_config = self._clean_config(config)

            init_kwargs: dict[str, Any] = {
                "project": self.project,
                "entity": self.entity,
                "name": name,
                "tags": tags or [],
                "notes": notes,
                "group": group,
                "job_type": job_type,
                "config": clean_config,
                "reinit": True,
            }
            if self.offline_mode:
                init_kwargs["mode"] = "offline"

            self.run = support.wandb.init(**init_kwargs)
            self.start_time = datetime.now()
            logger.info("W&B run initialized: %s", self.run.name)
        except support.WANDB_EXCEPTIONS as e:
            logger.error("Failed to initialize W&B run: %s", e)
            self.enabled = False

    def _clean_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Clean configuration dictionary for W&B serialization."""
        support = _wandb_support()
        clean_config = {}

        for key, value in config.items():
            try:
                if callable(value) or hasattr(value, "__dict__"):
                    if hasattr(value, "__dict__"):
                        clean_config[key] = {
                            k: v
                            for k, v in value.__dict__.items()
                            if not callable(v) and not k.startswith("_")
                        }
                    else:
                        clean_config[key] = str(value)
                else:
                    clean_config[key] = value
            except support.WANDB_EXCEPTIONS:
                continue

        return clean_config

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        prefix: str | None = None,
    ):
        """Log metrics to W&B."""
        if not self.enabled or not self.run:
            return

        support = _wandb_support()
        try:
            if prefix:
                metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

            numeric_metrics = {}
            for k, v in metrics.items():
                try:
                    if isinstance(v, (int, float, np.number)):
                        numeric_metrics[k] = float(v)
                    elif hasattr(v, "item"):
                        numeric_metrics[k] = float(v.item())
                    else:
                        numeric_metrics[k] = v
                except support.WANDB_EXCEPTIONS:
                    continue

            self.run.log(numeric_metrics, step=step)
        except support.WANDB_EXCEPTIONS as e:
            logger.error("Failed to log metrics: %s", e)

    def log_agent_config(self, agent_config: AgentConfig):
        """Log agent configuration details."""
        if not self.enabled:
            return

        support = _wandb_support()
        try:
            config_dict = {
                "agent/model_name": agent_config.model_name,
                "agent/system_prompt_length": len(agent_config.system_prompt)
                if agent_config.system_prompt
                else 0,
                "agent/temperature": agent_config.temperature,
                "agent/max_new_tokens": agent_config.max_new_tokens,
                "agent/torch_dtype": agent_config.torch_dtype,
                "agent/device_map": str(agent_config.device_map),
                "agent/use_peft": agent_config.use_peft,
            }

            if agent_config.peft_config:
                config_dict.update(
                    {
                        f"agent/peft_{k}": v
                        for k, v in agent_config.peft_config.items()
                        if isinstance(v, (str, int, float, bool))
                    }
                )

            self.log_metrics(config_dict)
        except support.WANDB_EXCEPTIONS as e:
            logger.error("Failed to log agent config: %s", e)

    def log_training_step(
        self,
        losses: dict[str, float],
        learning_rate: float,
        step: int,
        trajectory_groups: list[TrajectoryGroup] | None = None,
    ):
        """Log training step metrics."""
        if not self.enabled:
            return

        support = _wandb_support()
        try:
            metrics = {
                "train/learning_rate": learning_rate,
                "train/step": step,
                **{f"train/{k}": v for k, v in losses.items()},
            }

            if trajectory_groups:
                traj_metrics = self._compute_trajectory_metrics(trajectory_groups)
                metrics.update({f"train/{k}": v for k, v in traj_metrics.items()})

            self.log_metrics(metrics, step=step)
        except support.WANDB_EXCEPTIONS as e:
            logger.error("Failed to log training step: %s", e)

    def log_evaluation(
        self,
        eval_metrics: dict[str, float],
        step: int,
        trajectories: list[MultiTurnTrajectory] | None = None,
    ):
        """Log evaluation metrics."""
        if not self.enabled:
            return

        support = _wandb_support()
        try:
            metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}

            if trajectories:
                traj_analysis = self._analyze_trajectories(trajectories)
                metrics.update(
                    {f"eval_analysis/{k}": v for k, v in traj_analysis.items()}
                )

            self.log_metrics(metrics, step=step)
        except support.WANDB_EXCEPTIONS as e:
            logger.error("Failed to log evaluation: %s", e)

    def _compute_trajectory_metrics(
        self, trajectory_groups: list[TrajectoryGroup]
    ) -> dict[str, float]:
        """Compute metrics from trajectory groups."""
        if not trajectory_groups:
            return {}

        all_rewards = []
        group_sizes = []
        reward_diversities = []

        for group in trajectory_groups:
            if hasattr(group, "trajectories"):
                rewards = []
                for traj in group.trajectories:
                    if hasattr(traj, "total_reward"):
                        rewards.append(traj.total_reward)
                    elif hasattr(traj, "reward"):
                        rewards.append(traj.reward)

                if rewards:
                    all_rewards.extend(rewards)
                    group_sizes.append(len(rewards))
                    if len(rewards) > 1:
                        reward_diversities.append(np.std(rewards))

        if not all_rewards:
            return {}

        metrics = {
            "num_groups": len(trajectory_groups),
            "total_trajectories": len(all_rewards),
            "avg_group_size": np.mean(group_sizes) if group_sizes else 0,
            "reward_mean": np.mean(all_rewards),
            "reward_std": np.std(all_rewards),
            "reward_min": np.min(all_rewards),
            "reward_max": np.max(all_rewards),
            "reward_median": np.median(all_rewards),
        }

        if reward_diversities:
            metrics["reward_diversity_mean"] = np.mean(reward_diversities)
            metrics["reward_diversity_std"] = np.std(reward_diversities)

        return metrics

    def _analyze_trajectories(
        self, trajectories: list[MultiTurnTrajectory]
    ) -> dict[str, float]:
        """Analyze trajectory characteristics."""
        if not trajectories:
            return {}

        episode_lengths = []
        total_rewards = []
        turn_counts = []

        for traj in trajectories:
            episode_lengths.append(traj.episode_length)
            total_rewards.append(traj.total_reward)

            user_turns = len([t for t in traj.turns if t.role == "user"])
            assistant_turns = len([t for t in traj.turns if t.role == "assistant"])
            turn_counts.append({"user": user_turns, "assistant": assistant_turns})

        return {
            "num_episodes": len(trajectories),
            "avg_episode_length": np.mean(episode_lengths),
            "episode_length_std": np.std(episode_lengths),
            "avg_total_reward": np.mean(total_rewards),
            "total_reward_std": np.std(total_rewards),
            "avg_user_turns": np.mean([tc["user"] for tc in turn_counts]),
            "avg_assistant_turns": np.mean([tc["assistant"] for tc in turn_counts]),
        }

    def log_model_checkpoint(
        self,
        checkpoint_path: str,
        step: int,
        metrics: dict[str, float] | None = None,
        is_best: bool = False,
    ):
        """Log model checkpoint information."""
        if not self.enabled:
            return

        support = _wandb_support()
        try:
            checkpoint_metrics = {
                "checkpoint/step": step,
                "checkpoint/is_best": is_best,
                "checkpoint/path": checkpoint_path,
                "checkpoint/timestamp": datetime.now().isoformat(),
            }

            if metrics:
                checkpoint_metrics.update(
                    {f"checkpoint/{k}": v for k, v in metrics.items()}
                )

            self.log_metrics(checkpoint_metrics, step=step)
        except support.WANDB_EXCEPTIONS as e:
            logger.error("Failed to log checkpoint: %s", e)

    def log_conversation_examples(
        self, trajectories: list[MultiTurnTrajectory], step: int, num_examples: int = 3
    ):
        """Log conversation examples as W&B tables."""
        if not self.enabled or not trajectories:
            return

        support = _wandb_support()
        try:
            examples = trajectories[:num_examples]
            table_data = []
            for i, traj in enumerate(examples):
                conversation = ""
                for turn in traj.turns:
                    conversation += f"{turn.role.capitalize()}: {turn.content}\n\n"

                table_data.append(
                    [i, traj.total_reward, traj.episode_length, conversation.strip()]
                )

            table = support.wandb.Table(
                columns=["Example", "Total Reward", "Episode Length", "Conversation"],
                data=table_data,
            )
            self.log_metrics({f"examples/conversations_step_{step}": table}, step=step)
        except support.WANDB_EXCEPTIONS as e:
            logger.error("Failed to log conversation examples: %s", e)

    def log_reward_distribution(
        self, rewards: list[float], step: int, name: str = "rewards"
    ):
        """Log reward distribution histogram."""
        if not self.enabled or not rewards:
            return

        support = _wandb_support()
        try:
            self.log_metrics(
                {
                    f"{name}/distribution": support.wandb.Histogram(rewards),
                    f"{name}/mean": np.mean(rewards),
                    f"{name}/std": np.std(rewards),
                },
                step=step,
            )
        except support.WANDB_EXCEPTIONS as e:
            logger.error("Failed to log reward distribution: %s", e)

    def finish_run(self, summary: dict[str, Any] | None = None):
        """Finish the W&B run and log summary."""
        if not self.enabled or not self.run:
            return

        support = _wandb_support()
        try:
            if self.start_time:
                duration = (datetime.now() - self.start_time).total_seconds()
                if summary is None:
                    summary = {}
                summary["training_duration_seconds"] = duration
                summary["training_duration_hours"] = duration / 3600

            if summary:
                for key, value in summary.items():
                    self.run.summary[key] = value

            self.run.finish()
            logger.info("W&B run finished")
        except support.WANDB_EXCEPTIONS as e:
            logger.error("Failed to finish W&B run: %s", e)

    def log_system_metrics(self):
        """Log system metrics (GPU, memory, etc.)."""
        if not self.enabled:
            return

        support = _wandb_support()
        try:
            import psutil
            import torch

            metrics = {
                "system/cpu_percent": psutil.cpu_percent(),
                "system/memory_percent": psutil.virtual_memory().percent,
            }

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                    memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                    metrics[f"gpu_{i}/memory_allocated_gb"] = memory_allocated
                    metrics[f"gpu_{i}/memory_reserved_gb"] = memory_reserved

            self.log_metrics(metrics)
        except support.WANDB_EXCEPTIONS as e:
            logger.error("Failed to log system metrics: %s", e)


def init_wandb(
    project: str = "grpo-agent-framework",
    entity: str | None = None,
    api_key: str | None = None,
    **kwargs,
) -> WandBLogger:
    """Initialize a W&B logger with convenience defaults."""
    return WandBLogger(project=project, entity=entity, api_key=api_key, **kwargs)
