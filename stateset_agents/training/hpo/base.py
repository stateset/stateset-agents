"""
Base classes and abstractions for hyperparameter optimization.

This module provides the foundational abstractions for HPO in StateSet Agents:
- HPOBackend: Abstract interface for different HPO engines
- SearchSpace: Hyperparameter search space definitions
- HPOResult: Results from HPO runs
- HPOCallback: Hooks for HPO events
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Union
import json
from pathlib import Path


class SearchSpaceType(str, Enum):
    """Types of hyperparameter search distributions."""
    FLOAT = "float"
    INT = "int"
    CATEGORICAL = "categorical"
    LOGUNIFORM = "loguniform"
    UNIFORM = "uniform"
    CHOICE = "choice"


@dataclass
class SearchDimension:
    """Defines a single hyperparameter dimension in the search space.

    Attributes:
        name: Parameter name (e.g., "learning_rate")
        type: Distribution type (float, int, categorical, etc.)
        low: Lower bound for numeric parameters
        high: Upper bound for numeric parameters
        choices: List of choices for categorical parameters
        log_scale: Whether to use log scale for numeric parameters
        default: Default value (optional)
    """
    name: str
    type: SearchSpaceType
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    log_scale: bool = False
    default: Optional[Any] = None

    def __post_init__(self):
        """Validate the search dimension configuration."""
        if self.type in [SearchSpaceType.FLOAT, SearchSpaceType.INT,
                        SearchSpaceType.UNIFORM, SearchSpaceType.LOGUNIFORM]:
            if self.low is None or self.high is None:
                raise ValueError(f"Numeric parameter {self.name} requires low and high bounds")
        elif self.type in [SearchSpaceType.CATEGORICAL, SearchSpaceType.CHOICE]:
            if not self.choices:
                raise ValueError(f"Categorical parameter {self.name} requires choices")


@dataclass
class SearchSpace:
    """Defines the complete hyperparameter search space.

    Example:
        >>> search_space = SearchSpace([
        ...     SearchDimension("learning_rate", SearchSpaceType.LOGUNIFORM, 1e-6, 1e-3),
        ...     SearchDimension("batch_size", SearchSpaceType.CHOICE, choices=[16, 32, 64]),
        ...     SearchDimension("optimizer", SearchSpaceType.CATEGORICAL,
        ...                     choices=["adam", "adamw", "sgd"])
        ... ])
    """
    dimensions: List[SearchDimension]

    def get_dimension(self, name: str) -> Optional[SearchDimension]:
        """Get a dimension by name."""
        for dim in self.dimensions:
            if dim.name == name:
                return dim
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "dimensions": [
                {
                    "name": dim.name,
                    "type": dim.type.value,
                    "low": dim.low,
                    "high": dim.high,
                    "choices": dim.choices,
                    "log_scale": dim.log_scale,
                    "default": dim.default
                }
                for dim in self.dimensions
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchSpace":
        """Create from dictionary representation."""
        dimensions = []
        for dim_data in data["dimensions"]:
            dimensions.append(SearchDimension(
                name=dim_data["name"],
                type=SearchSpaceType(dim_data["type"]),
                low=dim_data.get("low"),
                high=dim_data.get("high"),
                choices=dim_data.get("choices"),
                log_scale=dim_data.get("log_scale", False),
                default=dim_data.get("default")
            ))
        return cls(dimensions)


@dataclass
class HPOResult:
    """Results from a single HPO trial.

    Attributes:
        trial_id: Unique identifier for this trial
        params: Hyperparameters tested
        metrics: Performance metrics achieved
        best_metric: Best metric value (for optimization)
        training_time: Time taken for training (seconds)
        status: Trial status (success, failed, pruned)
        checkpoint_path: Path to saved checkpoint (if any)
        metadata: Additional metadata
    """
    trial_id: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    best_metric: float
    training_time: float
    status: str = "success"
    checkpoint_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "trial_id": self.trial_id,
            "params": self.params,
            "metrics": self.metrics,
            "best_metric": self.best_metric,
            "training_time": self.training_time,
            "status": self.status,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "metadata": self.metadata
        }

    def save(self, path: Path):
        """Save result to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class HPOSummary:
    """Summary of an entire HPO run.

    Attributes:
        best_params: Best hyperparameters found
        best_metric: Best metric value achieved
        best_trial_id: ID of best trial
        total_trials: Total number of trials run
        successful_trials: Number of successful trials
        failed_trials: Number of failed trials
        pruned_trials: Number of pruned trials
        total_time: Total time spent (seconds)
        all_results: All trial results
    """
    best_params: Dict[str, Any]
    best_metric: float
    best_trial_id: str
    total_trials: int
    successful_trials: int
    failed_trials: int = 0
    pruned_trials: int = 0
    total_time: float = 0.0
    all_results: List[HPOResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "best_params": self.best_params,
            "best_metric": self.best_metric,
            "best_trial_id": self.best_trial_id,
            "total_trials": self.total_trials,
            "successful_trials": self.successful_trials,
            "failed_trials": self.failed_trials,
            "pruned_trials": self.pruned_trials,
            "total_time": self.total_time,
            "all_results": [r.to_dict() for r in self.all_results]
        }

    def save(self, path: Path):
        """Save summary to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_summary(self):
        """Print a formatted summary to console."""
        print("\n" + "="*60)
        print("HPO RUN SUMMARY")
        print("="*60)
        print(f"Best Trial ID: {self.best_trial_id}")
        print(f"Best Metric: {self.best_metric:.4f}")
        print(f"\nBest Hyperparameters:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        print(f"\nTrials: {self.total_trials} total, {self.successful_trials} successful, "
              f"{self.failed_trials} failed, {self.pruned_trials} pruned")
        print(f"Total Time: {self.total_time:.2f}s")
        print("="*60 + "\n")


class HPOCallback(Protocol):
    """Protocol for HPO callbacks."""

    def on_trial_start(self, trial_id: str, params: Dict[str, Any]) -> None:
        """Called when a trial starts."""
        ...

    def on_trial_end(self, result: HPOResult) -> None:
        """Called when a trial ends."""
        ...

    def on_hpo_start(self, search_space: SearchSpace, n_trials: int) -> None:
        """Called when HPO starts."""
        ...

    def on_hpo_end(self, summary: HPOSummary) -> None:
        """Called when HPO ends."""
        ...


class HPOBackend(ABC):
    """Abstract base class for HPO backends.

    This defines the interface that all HPO backends must implement.
    Concrete implementations include Optuna, Ray Tune, and W&B Sweeps.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        objective_metric: str = "reward",
        direction: str = "maximize",
        callbacks: Optional[List[HPOCallback]] = None
    ):
        """Initialize the HPO backend.

        Args:
            search_space: Hyperparameter search space
            objective_metric: Metric to optimize (e.g., "reward", "loss")
            direction: "maximize" or "minimize"
            callbacks: Optional callbacks for HPO events
        """
        self.search_space = search_space
        self.objective_metric = objective_metric
        self.direction = direction
        self.callbacks = callbacks or []
        self.results: List[HPOResult] = []

    @abstractmethod
    async def suggest_params(self, trial_id: str) -> Dict[str, Any]:
        """Suggest hyperparameters for a new trial.

        Args:
            trial_id: Unique identifier for this trial

        Returns:
            Dictionary of suggested hyperparameters
        """
        pass

    @abstractmethod
    async def report_result(self, result: HPOResult) -> None:
        """Report the result of a trial.

        Args:
            result: Trial result including metrics and params
        """
        pass

    @abstractmethod
    async def should_prune(self, trial_id: str, intermediate_metrics: Dict[str, float]) -> bool:
        """Determine if a trial should be pruned early.

        Args:
            trial_id: Trial identifier
            intermediate_metrics: Metrics from intermediate checkpoints

        Returns:
            True if trial should be pruned
        """
        pass

    @abstractmethod
    async def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        n_trials: int = 100,
        timeout: Optional[float] = None
    ) -> HPOSummary:
        """Run the HPO optimization.

        Args:
            objective_fn: Function that takes params and returns metric value
            n_trials: Number of trials to run
            timeout: Optional timeout in seconds

        Returns:
            HPOSummary with best params and full results
        """
        pass

    def _notify_trial_start(self, trial_id: str, params: Dict[str, Any]):
        """Notify callbacks of trial start."""
        for callback in self.callbacks:
            callback.on_trial_start(trial_id, params)

    def _notify_trial_end(self, result: HPOResult):
        """Notify callbacks of trial end."""
        for callback in self.callbacks:
            callback.on_trial_end(result)

    def _notify_hpo_start(self, n_trials: int):
        """Notify callbacks of HPO start."""
        for callback in self.callbacks:
            callback.on_hpo_start(self.search_space, n_trials)

    def _notify_hpo_end(self, summary: HPOSummary):
        """Notify callbacks of HPO end."""
        for callback in self.callbacks:
            callback.on_hpo_end(summary)


# Type aliases for convenience
ObjectiveFunction = Callable[[Dict[str, Any]], float]
AsyncObjectiveFunction = Callable[[Dict[str, Any]], Any]  # Returns awaitable
