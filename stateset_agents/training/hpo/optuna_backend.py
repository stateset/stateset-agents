"""
Optuna backend for hyperparameter optimization.

Optuna is a state-of-the-art HPO framework with:
- Tree-structured Parzen Estimator (TPE) for efficient search
- MedianPruner for early stopping of unpromising trials
- Distributed optimization support
- Rich visualization capabilities
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
import logging

from .base import (
    HPOBackend,
    HPOCallback,
    HPOResult,
    HPOSummary,
    SearchSpace,
    SearchDimension,
    SearchSpaceType,
)

# Optional import - gracefully handle if not installed
try:
    import optuna
    from optuna.pruners import MedianPruner, HyperbandPruner, PercentilePruner
    from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate,
    )
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None  # type: ignore
    if TYPE_CHECKING:
        import optuna


logger = logging.getLogger(__name__)


def _resolve_async_value(value: Any) -> Any:
    """Resolve possibly-async objective values to a concrete result."""
    if not asyncio.iscoroutine(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)
    # If we're already inside an event loop, run the coroutine in a worker thread.
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, value).result()


def _require_optuna():
    """Check if Optuna is available."""
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna is required for this HPO backend. "
            "Install it with: pip install optuna"
        )


class OptunaBackend(HPOBackend):
    """Optuna backend for hyperparameter optimization.

    Features:
    - Tree-structured Parzen Estimator (TPE) for Bayesian optimization
    - Multiple pruning strategies for early stopping
    - Distributed optimization via study storage
    - Rich visualization and analysis tools

    Example:
        >>> from training.hpo.optuna_backend import OptunaBackend
        >>> from training.hpo.search_spaces import create_grpo_search_space
        >>>
        >>> search_space = create_grpo_search_space()
        >>> backend = OptunaBackend(
        ...     search_space=search_space,
        ...     study_name="grpo_optimization",
        ...     sampler="tpe",
        ...     pruner="median"
        ... )
        >>>
        >>> def objective(params):
        ...     # Train with params and return metric
        ...     return train_and_evaluate(params)
        >>>
        >>> summary = await backend.optimize(objective, n_trials=100)
        >>> print(f"Best params: {summary.best_params}")
    """

    def __init__(
        self,
        search_space: SearchSpace,
        objective_metric: str = "reward",
        direction: str = "maximize",
        callbacks: Optional[List[HPOCallback]] = None,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        sampler: str = "tpe",
        pruner: str = "median",
        n_startup_trials: int = 10,
        n_warmup_steps: int = 5,
        pruning_interval: int = 1,
        load_if_exists: bool = False,
    ):
        """Initialize Optuna backend.

        Args:
            search_space: Hyperparameter search space
            objective_metric: Metric to optimize
            direction: "maximize" or "minimize"
            callbacks: Optional callbacks for HPO events
            study_name: Name for the Optuna study (for persistence)
            storage: Storage URL (e.g., "sqlite:///optuna.db" for persistence)
            sampler: Sampling algorithm ("tpe", "random", "cmaes")
            pruner: Pruning algorithm ("median", "hyperband", "percentile", "none")
            n_startup_trials: Number of random trials before using sampler
            n_warmup_steps: Number of warmup steps before pruning
            pruning_interval: How often to check for pruning
            load_if_exists: Load existing study if it exists
        """
        super().__init__(search_space, objective_metric, direction, callbacks)

        self.study_name = study_name or f"hpo_{int(time.time())}"
        self.storage = storage
        self.sampler_name = sampler
        self.pruner_name = pruner
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps
        self.pruning_interval = pruning_interval
        self.load_if_exists = load_if_exists

        # Create Optuna components only if Optuna is installed.
        # This keeps the backend importable/constructible in lightweight environments,
        # and raises a clear error only when optimization is invoked.
        self.sampler = None
        self.pruner = None
        if OPTUNA_AVAILABLE:
            self.sampler = self._create_sampler()
            self.pruner = self._create_pruner()
        self.study: Optional[optuna.Study] = None

    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create the Optuna sampler."""
        if self.sampler_name == "tpe":
            return TPESampler(
                n_startup_trials=self.n_startup_trials,
                multivariate=True,
                seed=42
            )
        elif self.sampler_name == "random":
            return RandomSampler(seed=42)
        elif self.sampler_name == "cmaes":
            return CmaEsSampler(seed=42)
        else:
            raise ValueError(f"Unknown sampler: {self.sampler_name}")

    def _create_pruner(self) -> optuna.pruners.BasePruner:
        """Create the Optuna pruner."""
        if self.pruner_name == "median":
            return MedianPruner(
                n_startup_trials=self.n_startup_trials,
                n_warmup_steps=self.n_warmup_steps,
                interval_steps=self.pruning_interval
            )
        elif self.pruner_name == "hyperband":
            return HyperbandPruner(
                min_resource=self.n_warmup_steps,
                reduction_factor=3
            )
        elif self.pruner_name == "percentile":
            return PercentilePruner(
                percentile=25.0,
                n_startup_trials=self.n_startup_trials,
                n_warmup_steps=self.n_warmup_steps
            )
        elif self.pruner_name == "none":
            return optuna.pruners.NopPruner()
        else:
            raise ValueError(f"Unknown pruner: {self.pruner_name}")

    def _create_study(self) -> optuna.Study:
        """Create or load an Optuna study."""
        _require_optuna()
        direction = "maximize" if self.direction == "maximize" else "minimize"

        return optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=self.sampler,
            pruner=self.pruner,
            direction=direction,
            load_if_exists=self.load_if_exists
        )

    def _suggest_param(self, trial: optuna.Trial, dim: SearchDimension) -> Any:
        """Suggest a parameter value for a search dimension."""
        if dim.type == SearchSpaceType.FLOAT or dim.type == SearchSpaceType.UNIFORM:
            return trial.suggest_float(
                dim.name,
                dim.low,
                dim.high,
                log=False
            )
        elif dim.type == SearchSpaceType.LOGUNIFORM:
            return trial.suggest_float(
                dim.name,
                dim.low,
                dim.high,
                log=True
            )
        elif dim.type == SearchSpaceType.INT:
            return trial.suggest_int(
                dim.name,
                int(dim.low),
                int(dim.high),
                log=dim.log_scale
            )
        elif dim.type in [SearchSpaceType.CATEGORICAL, SearchSpaceType.CHOICE]:
            return trial.suggest_categorical(dim.name, dim.choices)
        else:
            raise ValueError(f"Unknown search space type: {dim.type}")

    async def suggest_params(self, trial_id: str) -> Dict[str, Any]:
        """Suggest hyperparameters for a new trial.

        Args:
            trial_id: Unique identifier (used as Optuna trial number)

        Returns:
            Dictionary of suggested hyperparameters
        """
        # For Optuna, we don't suggest params independently
        # This method is not used in the main optimization loop
        raise NotImplementedError("Use optimize() method for Optuna backend")

    async def report_result(self, result: HPOResult) -> None:
        """Report the result of a trial.

        Args:
            result: Trial result including metrics and params
        """
        self.results.append(result)

    async def should_prune(
        self,
        trial_id: str,
        intermediate_metrics: Dict[str, float]
    ) -> bool:
        """Determine if a trial should be pruned.

        Optuna handles pruning internally, so this always returns False.
        """
        return False

    def _objective_wrapper(
        self,
        trial: optuna.Trial,
        objective_fn: Callable[[Dict[str, Any]], float]
    ) -> float:
        """Wrapper that converts Optuna trial to params dict."""
        # Suggest parameters based on search space
        params = {}
        for dim in self.search_space.dimensions:
            params[dim.name] = self._suggest_param(trial, dim)

        trial_id = str(trial.number)

        # Notify callbacks
        self._notify_trial_start(trial_id, params)

        start_time = time.time()

        try:
            # Run objective function
            metric_value = _resolve_async_value(objective_fn(params))
            training_time = time.time() - start_time

            # Create result
            result = HPOResult(
                trial_id=trial_id,
                params=params,
                metrics={self.objective_metric: metric_value},
                best_metric=metric_value,
                training_time=training_time,
                status="success"
            )

            # Store result
            self.results.append(result)

            # Notify callbacks
            self._notify_trial_end(result)

            # Check for pruning (optional intermediate reporting)
            if hasattr(objective_fn, "report_intermediate"):
                # Allow objective function to report intermediate values
                pass

            return metric_value

        except optuna.TrialPruned:
            training_time = time.time() - start_time
            result = HPOResult(
                trial_id=trial_id,
                params=params,
                metrics={},
                best_metric=float('-inf') if self.direction == "maximize" else float('inf'),
                training_time=training_time,
                status="pruned"
            )
            self.results.append(result)
            self._notify_trial_end(result)
            raise

        except Exception as e:
            logger.error(f"Trial {trial_id} failed: {e}")
            training_time = time.time() - start_time
            result = HPOResult(
                trial_id=trial_id,
                params=params,
                metrics={},
                best_metric=float('-inf') if self.direction == "maximize" else float('inf'),
                training_time=training_time,
                status="failed",
                metadata={"error": str(e)}
            )
            self.results.append(result)
            self._notify_trial_end(result)
            raise

    async def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        n_trials: int = 100,
        timeout: Optional[float] = None,
    ) -> HPOSummary:
        """Run the HPO optimization using Optuna.

        Args:
            objective_fn: Function that takes params dict and returns metric value
            n_trials: Number of trials to run
            timeout: Optional timeout in seconds

        Returns:
            HPOSummary with best params and all results

        Example:
            >>> def objective(params):
            ...     lr = params['learning_rate']
            ...     # Train model and return reward
            ...     return train(lr)
            >>>
            >>> summary = await backend.optimize(objective, n_trials=50)
        """
        _require_optuna()
        self.study = self._create_study()
        self.results = []

        # Notify callbacks
        self._notify_hpo_start(n_trials)

        start_time = time.time()

        # Run optimization (Optuna handles synchronization)
        logger.info(f"Starting Optuna optimization: {n_trials} trials")

        # Wrap objective for async compatibility
        def sync_objective_wrapper(trial):
            return self._objective_wrapper(trial, objective_fn)

        try:
            self.study.optimize(
                sync_objective_wrapper,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=True,
            )
        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")

        total_time = time.time() - start_time

        # Extract best trial
        best_trial = self.study.best_trial
        best_params = best_trial.params
        best_metric = best_trial.value

        # Count trial statuses
        successful_trials = len([r for r in self.results if r.status == "success"])
        failed_trials = len([r for r in self.results if r.status == "failed"])
        pruned_trials = len([r for r in self.results if r.status == "pruned"])

        # Create summary
        summary = HPOSummary(
            best_params=best_params,
            best_metric=best_metric,
            best_trial_id=str(best_trial.number),
            total_trials=len(self.results),
            successful_trials=successful_trials,
            failed_trials=failed_trials,
            pruned_trials=pruned_trials,
            total_time=total_time,
            all_results=self.results
        )

        # Notify callbacks
        self._notify_hpo_end(summary)

        logger.info(f"Optimization complete: best metric = {best_metric:.4f}")

        return summary

    def plot_optimization_history(self, save_path: Optional[Path] = None):
        """Plot optimization history.

        Args:
            save_path: Optional path to save plot
        """
        if self.study is None:
            raise ValueError("No study available. Run optimize() first.")

        fig = plot_optimization_history(self.study)
        if save_path:
            fig.write_image(str(save_path))
        return fig

    def plot_param_importances(self, save_path: Optional[Path] = None):
        """Plot hyperparameter importances.

        Args:
            save_path: Optional path to save plot
        """
        if self.study is None:
            raise ValueError("No study available. Run optimize() first.")

        fig = plot_param_importances(self.study)
        if save_path:
            fig.write_image(str(save_path))
        return fig

    def plot_parallel_coordinate(self, save_path: Optional[Path] = None):
        """Plot parallel coordinate plot of trials.

        Args:
            save_path: Optional path to save plot
        """
        if self.study is None:
            raise ValueError("No study available. Run optimize() first.")

        fig = plot_parallel_coordinate(self.study)
        if save_path:
            fig.write_image(str(save_path))
        return fig

    def get_study(self) -> optuna.Study:
        """Get the underlying Optuna study object.

        Returns:
            Optuna Study object for advanced analysis
        """
        if self.study is None:
            raise ValueError("No study available. Run optimize() first.")
        return self.study
