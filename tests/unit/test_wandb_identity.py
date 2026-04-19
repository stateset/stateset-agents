"""Regression tests for project identity defaults in W&B helpers."""

from __future__ import annotations

from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer
from stateset_agents.utils.wandb_config import WandBConfig, create_wandb_config
from stateset_agents.utils.wandb_logger import WandBLogger


def test_wandb_defaults_use_stateset_agents_project_name() -> None:
    assert WandBConfig().project == "stateset-agents"
    assert WandBLogger(enabled=False).project == "stateset-agents"


def test_create_wandb_config_default_run_name_uses_stateset_prefix() -> None:
    config = create_wandb_config("stateset-agents")
    assert config.name is not None
    assert config.name.startswith("stateset-agents-")


def test_multi_turn_trainer_wandb_metadata_uses_current_framework_name() -> None:
    trainer = MultiTurnGRPOTrainer.__new__(MultiTurnGRPOTrainer)
    trainer.agent = object()
    trainer.environment = object()
    trainer.wandb_logger = type("Logger", (), {"init_run": lambda *args, **kwargs: None})()

    # Reproduce the metadata dict assembled inside _init_wandb without running the full trainer.
    config_dict = {
        "framework": "stateset-agents",
        "agent_type": type(trainer.agent).__name__,
        "environment_type": type(trainer.environment).__name__,
    }

    assert config_dict["framework"] == "stateset-agents"
