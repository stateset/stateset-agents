"""Configuration helpers for Weights & Biases integration."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def _parse_env_tags(raw_tags: str | None) -> list[str]:
    """Parse WANDB_TAGS into a normalized list without blank entries."""
    if not raw_tags:
        return []
    return [tag.strip() for tag in raw_tags.split(",") if tag.strip()]


@dataclass
class WandBConfig:
    """Configuration for Weights & Biases integration."""

    project: str = "grpo-agent-framework"
    entity: str | None = None
    name: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: str | None = None

    log_frequency: int = 10
    log_trajectories: bool = True
    log_rewards: bool = True
    log_model_gradients: bool = True
    log_model_parameters: bool = False
    log_system_metrics: bool = True

    create_reward_plots: bool = True
    create_trajectory_plots: bool = True
    create_loss_plots: bool = True
    plot_frequency: int = 100

    log_code: bool = True
    log_model_architecture: bool = True
    save_model_artifacts: bool = True
    watch_model: bool = True

    offline_mode: bool = False
    save_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "entity": self.entity,
            "name": self.name,
            "tags": self.tags,
            "notes": self.notes,
            "log_frequency": self.log_frequency,
            "log_trajectories": self.log_trajectories,
            "log_rewards": self.log_rewards,
            "log_model_gradients": self.log_model_gradients,
            "log_model_parameters": self.log_model_parameters,
            "log_system_metrics": self.log_system_metrics,
            "create_reward_plots": self.create_reward_plots,
            "create_trajectory_plots": self.create_trajectory_plots,
            "create_loss_plots": self.create_loss_plots,
            "plot_frequency": self.plot_frequency,
            "log_code": self.log_code,
            "log_model_architecture": self.log_model_architecture,
            "save_model_artifacts": self.save_model_artifacts,
            "watch_model": self.watch_model,
            "offline_mode": self.offline_mode,
            "save_dir": self.save_dir,
        }


def create_wandb_config(
    project: str, name: str | None = None, tags: list[str] | None = None, **kwargs
) -> WandBConfig:
    """Create a W&B configuration with sensible defaults."""
    return WandBConfig(
        project=project,
        name=name or f"grpo-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        tags=tags or [],
        **kwargs,
    )


def setup_wandb_from_env() -> WandBConfig | None:
    """Setup W&B configuration from environment variables."""
    offline_mode = os.getenv("WANDB_MODE", "").lower() == "offline"
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key and not offline_mode:
        logger.warning("WANDB_API_KEY not found in environment")
        return None

    return WandBConfig(
        project=os.getenv("WANDB_PROJECT", "grpo-agent-framework"),
        entity=os.getenv("WANDB_ENTITY"),
        name=os.getenv("WANDB_RUN_NAME"),
        tags=_parse_env_tags(os.getenv("WANDB_TAGS")),
        notes=os.getenv("WANDB_NOTES"),
        offline_mode=offline_mode,
    )
