"""
Configuration for hyperparameter optimization.

This module provides configuration dataclasses for HPO runs.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import SearchSpace


@dataclass
class HPOConfig:
    """Configuration for hyperparameter optimization.

    Attributes:
        backend: HPO backend to use ("optuna", "ray_tune", "wandb")
        search_space: Hyperparameter search space
        n_trials: Number of trials to run
        timeout: Optional timeout in seconds
        objective_metric: Metric to optimize
        direction: "maximize" or "minimize"
        output_dir: Directory for HPO outputs
        study_name: Name for the HPO study

        # Backend-specific configs
        optuna_config: Configuration for Optuna backend
        ray_config: Configuration for Ray Tune backend
        wandb_config: Configuration for W&B Sweeps backend

        # Training integration
        base_training_config: Base training config to override with HPO params
        checkpoint_freq: How often to checkpoint during HPO
        keep_best_n: Keep only N best checkpoints
        early_stopping_patience: Stop if no improvement for N trials

        # Resource management
        n_parallel_trials: Number of parallel trials (for distributed backends)
        gpu_per_trial: GPUs per trial
        cpu_per_trial: CPUs per trial
    """

    # Core HPO settings
    backend: str = "optuna"
    search_space: Optional[SearchSpace] = None
    search_space_name: Optional[str] = None  # Name of predefined search space
    n_trials: int = 100
    timeout: Optional[float] = None
    objective_metric: str = "reward"
    direction: str = "maximize"
    output_dir: Path = Path("./hpo_results")
    study_name: Optional[str] = None

    # Backend-specific configs
    optuna_config: Dict[str, Any] = field(default_factory=lambda: {
        "sampler": "tpe",
        "pruner": "median",
        "n_startup_trials": 10,
        "n_warmup_steps": 5,
        "storage": None,  # Set to "sqlite:///optuna.db" for persistence
    })

    ray_config: Dict[str, Any] = field(default_factory=lambda: {
        "scheduler": "asha",
        "search_alg": "bayesopt",
        "max_concurrent": 4,
    })

    wandb_config: Dict[str, Any] = field(default_factory=lambda: {
        "method": "bayes",
        "project": "stateset-hpo",
        "entity": None,
    })

    # Training integration
    base_training_config: Optional[Dict[str, Any]] = None
    checkpoint_freq: int = 10  # Checkpoint every N trials
    keep_best_n: int = 5  # Keep top 5 checkpoints
    early_stopping_patience: Optional[int] = None  # Stop if no improvement

    # Resource management
    n_parallel_trials: int = 1  # Parallel trials (distributed backends)
    gpu_per_trial: float = 1.0
    cpu_per_trial: int = 4

    # Logging and monitoring
    verbose: bool = True
    log_to_wandb: bool = False
    save_plots: bool = True

    def __post_init__(self):
        """Validate and setup configuration."""
        # Ensure output directory exists
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate backend
        valid_backends = ["optuna", "ray_tune", "wandb"]
        if self.backend not in valid_backends:
            raise ValueError(f"Backend must be one of {valid_backends}, got {self.backend}")

        # Validate direction
        if self.direction not in ["maximize", "minimize"]:
            raise ValueError(f"Direction must be 'maximize' or 'minimize', got {self.direction}")

        # Ensure either search_space or search_space_name is provided
        if self.search_space is None and self.search_space_name is None:
            raise ValueError("Must provide either search_space or search_space_name")

    def get_backend_config(self) -> Dict[str, Any]:
        """Get backend-specific configuration."""
        if self.backend == "optuna":
            return self.optuna_config
        elif self.backend == "ray_tune":
            return self.ray_config
        elif self.backend == "wandb":
            return self.wandb_config
        else:
            return {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backend": self.backend,
            "search_space_name": self.search_space_name,
            "n_trials": self.n_trials,
            "timeout": self.timeout,
            "objective_metric": self.objective_metric,
            "direction": self.direction,
            "output_dir": str(self.output_dir),
            "study_name": self.study_name,
            "optuna_config": self.optuna_config,
            "ray_config": self.ray_config,
            "wandb_config": self.wandb_config,
            "checkpoint_freq": self.checkpoint_freq,
            "keep_best_n": self.keep_best_n,
            "early_stopping_patience": self.early_stopping_patience,
            "n_parallel_trials": self.n_parallel_trials,
            "gpu_per_trial": self.gpu_per_trial,
            "cpu_per_trial": self.cpu_per_trial,
        }


# Predefined HPO configurations for common scenarios

CONSERVATIVE_HPO_CONFIG = HPOConfig(
    backend="optuna",
    search_space_name="conservative",
    n_trials=50,
    objective_metric="reward",
    direction="maximize",
    optuna_config={
        "sampler": "tpe",
        "pruner": "median",
        "n_startup_trials": 10,
    }
)

AGGRESSIVE_HPO_CONFIG = HPOConfig(
    backend="optuna",
    search_space_name="aggressive",
    n_trials=100,
    objective_metric="reward",
    direction="maximize",
    optuna_config={
        "sampler": "tpe",
        "pruner": "hyperband",
        "n_startup_trials": 15,
    }
)

QUICK_HPO_CONFIG = HPOConfig(
    backend="optuna",
    search_space_name="grpo",
    n_trials=20,
    objective_metric="reward",
    direction="maximize",
    optuna_config={
        "sampler": "random",
        "pruner": "median",
        "n_startup_trials": 5,
    }
)

DISTRIBUTED_HPO_CONFIG = HPOConfig(
    backend="ray_tune",
    search_space_name="full",
    n_trials=200,
    n_parallel_trials=8,
    objective_metric="reward",
    direction="maximize",
    ray_config={
        "scheduler": "asha",
        "search_alg": "bayesopt",
        "max_concurrent": 8,
    }
)


def get_hpo_config(profile: str = "conservative") -> HPOConfig:
    """Get a pre-defined HPO configuration.

    Args:
        profile: Configuration profile ("conservative", "aggressive", "quick", "distributed")

    Returns:
        HPOConfig instance

    Raises:
        ValueError: If profile is not recognized
    """
    profiles = {
        "conservative": CONSERVATIVE_HPO_CONFIG,
        "aggressive": AGGRESSIVE_HPO_CONFIG,
        "quick": QUICK_HPO_CONFIG,
        "distributed": DISTRIBUTED_HPO_CONFIG,
    }

    if profile not in profiles:
        available = ", ".join(profiles.keys())
        raise ValueError(f"Unknown profile '{profile}'. Available: {available}")

    return profiles[profile]
