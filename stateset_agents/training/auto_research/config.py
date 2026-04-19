"""
Configuration for the autonomous research loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AutoResearchConfig:
    """Configuration for the autonomous research loop.

    Attributes:
        time_budget: Wall-clock seconds per experiment (training + eval).
        max_experiments: Maximum number of experiments to run (0 = unlimited).
        max_wall_clock: Total wall-clock budget in seconds (0 = unlimited).
        objective_metric: Metric name to optimize (from evaluation results).
        direction: "maximize" or "minimize".
        improvement_threshold: Minimum relative improvement to count as "better".
        improvement_patience: Stop after this many consecutive non-improvements
            (0 = disabled). Useful for detecting plateaus.
        output_dir: Directory to write checkpoints, logs, and results.
        proposer: Proposer strategy name.
        search_space_name: Predefined search space name, or None.
        search_space_overrides: Dict of SearchDimension-like dicts to override/add.
        eval_episodes: Number of evaluation episodes per experiment.
        eval_seed: Fixed seed for reproducible evaluation.
        eval_concurrency: Parallel evaluation episodes.
        selection_promotion_zscore: Stability penalty applied to the locked
            selection metric before promotion. Higher values are stricter.
        experiment_isolation: Execution isolation mode for subprocess runs
            ("worktree" or "shared").
        runtime_environment: Runtime launcher for subprocess runs ("venv" or
            "uv").
        save_checkpoints: Whether to save model checkpoints for kept experiments.
        log_to_wandb: Whether to log experiment results to W&B.
        wandb_project: W&B project name (if logging enabled).
        trainer_algorithm: Training algorithm ("gspo", "grpo", "dapo", "vapo"),
            or "auto" to let the proposer choose per-experiment.
        calibrate_rewards: Normalize reward scores to prevent scale drift.
        calibration_method: "z_score", "min_max", or "percentile".
        base_config_overrides: Fixed overrides applied to every experiment.
    """

    # Experiment loop
    time_budget: int = 300
    max_experiments: int = 0
    max_wall_clock: int = 0
    objective_metric: str = "eval_reward"
    direction: str = "maximize"
    improvement_threshold: float = 0.0
    improvement_patience: int = 0

    # Output
    output_dir: str = "./auto_research_results"

    # Proposer
    proposer: str = "perturbation"
    search_space_name: str | None = "grpo"
    search_space_overrides: dict[str, Any] | None = None

    # Evaluation
    eval_episodes: int = 10
    eval_seed: int = 42
    eval_concurrency: int = 1
    selection_promotion_zscore: float = 1.0
    experiment_isolation: str = "worktree"
    runtime_environment: str = "venv"

    # Checkpointing
    save_checkpoints: bool = True

    # Logging
    log_to_wandb: bool = False
    wandb_project: str = "auto-research"

    # Training algorithm — "gspo", "grpo", "dapo", "vapo", or "auto"
    # When "auto", the proposer can include "algorithm" in proposed params.
    trainer_algorithm: str = "gspo"

    # Reward calibration
    calibrate_rewards: bool = False
    calibration_method: str = "percentile"

    # Fixed overrides for every experiment
    base_config_overrides: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        """Return a list of validation warnings."""
        warnings: list[str] = []
        if self.direction not in ("maximize", "minimize"):
            warnings.append(
                f"direction must be 'maximize' or 'minimize', got {self.direction!r}"
            )
        if self.time_budget <= 0:
            warnings.append("time_budget must be positive")
        valid_proposers = (
            "random", "grid", "bayesian", "perturbation",
            "adaptive", "smart", "llm",
        )
        if self.proposer not in valid_proposers:
            warnings.append(f"Unknown proposer strategy: {self.proposer!r}")
        valid_algos = ("gspo", "grpo", "dapo", "vapo", "auto")
        if self.trainer_algorithm not in valid_algos:
            warnings.append(
                f"Unknown trainer_algorithm: {self.trainer_algorithm!r}"
            )
        if self.eval_episodes <= 0:
            warnings.append("eval_episodes must be positive")
        if self.eval_concurrency <= 0:
            warnings.append("eval_concurrency must be positive")
        if self.selection_promotion_zscore < 0.0:
            warnings.append("selection_promotion_zscore must be >= 0")
        if self.experiment_isolation not in {"shared", "worktree"}:
            warnings.append("experiment_isolation must be one of: shared, worktree")
        if self.runtime_environment not in {"uv", "venv"}:
            warnings.append("runtime_environment must be one of: uv, venv")
        if self.improvement_patience < 0:
            warnings.append("improvement_patience must be >= 0")
        return warnings

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (for saving to JSON/YAML)."""
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_file(cls, path: str | Path) -> AutoResearchConfig:
        """Load configuration from a YAML or JSON file.

        Supports two formats:
        1. Flat: all fields at top level
        2. Nested: fields under an "auto_research" key

        Example YAML:
            time_budget: 300
            proposer: smart
            search_space_name: auto_research

        Or nested:
            auto_research:
              time_budget: 300
              proposer: smart

        Args:
            path: Path to YAML (.yaml/.yml) or JSON (.json) config file.

        Returns:
            AutoResearchConfig populated from the file.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()

        if suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "PyYAML required for YAML config files. "
                    "Install with: pip install pyyaml"
                ) from exc
            data = yaml.safe_load(text) or {}
        elif suffix in (".json",):
            import json

            data = json.loads(text) or {}
        else:
            raise ValueError(f"Unsupported config format: {suffix}")

        # Support nested format
        if "auto_research" in data and isinstance(data["auto_research"], dict):
            data = data["auto_research"]

        # Map common aliases
        if "algorithm" in data and "trainer_algorithm" not in data:
            data["trainer_algorithm"] = data.pop("algorithm")
        if "search_space" in data and "search_space_name" not in data:
            data["search_space_name"] = data.pop("search_space")
        if "patience" in data and "improvement_patience" not in data:
            data["improvement_patience"] = data.pop("patience")
        if "wandb" in data and "log_to_wandb" not in data:
            data["log_to_wandb"] = data.pop("wandb")

        # Filter to only known fields
        from dataclasses import fields as dc_fields

        known = {f.name for f in dc_fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}

        return cls(**filtered)
