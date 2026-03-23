"""
Experiment proposers for the autonomous research loop.

A proposer suggests the next set of hyperparameters to try, given the
history of previous experiments and a search space definition.
"""

from __future__ import annotations

import logging
import math
import random
from abc import ABC, abstractmethod
from itertools import product
from typing import Any

logger = logging.getLogger(__name__)


def _dim_names(search_space: Any) -> set[str]:
    """Return the set of dimension names in a search space."""
    return {d.name for d in search_space.dimensions}


class ExperimentProposer(ABC):
    """Abstract base for experiment proposers."""

    @abstractmethod
    def propose(
        self,
        current_best: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], str]:
        """Propose the next experiment configuration.

        Args:
            current_best: The current best hyperparameter configuration.
            history: List of previous experiment records (params, objective, status).

        Returns:
            A tuple of (proposed_params, description) where description is a
            short human-readable summary of what changed.
        """
        ...


def _sample_dimension(dim: Any) -> Any:
    """Sample a random value from a search dimension."""
    from stateset_agents.training.hpo.base import SearchSpaceType

    if dim.type in (SearchSpaceType.CATEGORICAL, SearchSpaceType.CHOICE):
        if not dim.choices:
            return dim.default
        return random.choice(dim.choices)
    elif dim.type == SearchSpaceType.LOGUNIFORM:
        return math.exp(random.uniform(math.log(dim.low), math.log(dim.high)))
    elif dim.type == SearchSpaceType.INT:
        return random.randint(int(dim.low), int(dim.high))
    elif dim.type in (SearchSpaceType.FLOAT, SearchSpaceType.UNIFORM):
        return random.uniform(dim.low, dim.high)
    if dim.default is not None:
        return dim.default
    # Last resort: midpoint of bounds
    if dim.low is not None and dim.high is not None:
        return (dim.low + dim.high) / 2.0
    return 0


class RandomProposer(ExperimentProposer):
    """Randomly samples from the search space, ignoring history."""

    def __init__(self, search_space: Any):
        self.search_space = search_space

    def propose(
        self,
        current_best: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], str]:
        # Start from current_best but only keep keys in search space
        known = _dim_names(self.search_space)
        params = {k: v for k, v in current_best.items() if k in known}
        changes: list[str] = []

        for dim in self.search_space.dimensions:
            value = _sample_dimension(dim)
            if value != params.get(dim.name):
                changes.append(f"{dim.name}={value}")
            params[dim.name] = value

        desc = ", ".join(changes) if changes else "random sample (no changes)"
        return params, f"random: {desc}"


class PerturbationProposer(ExperimentProposer):
    """Perturbs the current best config by a small random amount.

    This is the default proposer — it makes small, focused changes to
    explore the neighborhood of the current best configuration. Inspired
    by Population Based Training (PBT) perturbation.
    """

    def __init__(
        self,
        search_space: Any,
        perturbation_factor: float = 0.2,
        num_params_to_change: int = 2,
    ):
        self.search_space = search_space
        self.perturbation_factor = perturbation_factor
        self.num_params_to_change = num_params_to_change

    def propose(
        self,
        current_best: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], str]:
        # Only keep keys that exist in the search space
        known = _dim_names(self.search_space)
        params = {k: v for k, v in current_best.items() if k in known}
        dims = list(self.search_space.dimensions)

        # Ensure all search space dims have a value
        for dim in dims:
            if dim.name not in params:
                params[dim.name] = dim.default if dim.default is not None else _sample_dimension(dim)

        # Pick a random subset of dimensions to perturb
        n = min(self.num_params_to_change, len(dims))
        to_change = random.sample(dims, n)

        changes: list[str] = []
        for dim in to_change:
            old_val = params.get(dim.name)
            new_val = self._perturb(dim, old_val)
            if new_val != old_val:
                changes.append(f"{dim.name}: {old_val} → {new_val}")
                params[dim.name] = new_val

        if not changes:
            # Force at least one random change
            dim = random.choice(dims)
            new_val = _sample_dimension(dim)
            changes.append(f"{dim.name}: {params.get(dim.name)} → {new_val}")
            params[dim.name] = new_val

        # Validate all params have correct types for their dimensions
        for dim in dims:
            val = params.get(dim.name)
            if val is not None and not self._is_valid_type(dim, val):
                params[dim.name] = _sample_dimension(dim)

        desc = "; ".join(changes)
        return params, f"perturb: {desc}"

    @staticmethod
    def _is_valid_type(dim: Any, value: Any) -> bool:
        """Check if a value is a valid type for the given dimension."""
        from stateset_agents.training.hpo.base import SearchSpaceType

        if dim.type in (SearchSpaceType.CATEGORICAL, SearchSpaceType.CHOICE):
            return dim.choices and value in dim.choices
        # INT, FLOAT, UNIFORM, LOGUNIFORM all require numeric values
        return isinstance(value, (int, float))

    def _perturb(self, dim: Any, current_value: Any) -> Any:
        from stateset_agents.training.hpo.base import SearchSpaceType

        if dim.type in (SearchSpaceType.CATEGORICAL, SearchSpaceType.CHOICE):
            candidates = [c for c in dim.choices if c != current_value]
            return random.choice(candidates) if candidates else current_value

        if current_value is None or not isinstance(current_value, (int, float)):
            return _sample_dimension(dim)

        # Numeric perturbation
        factor = 1.0 + random.uniform(
            -self.perturbation_factor, self.perturbation_factor
        )
        new_val = float(current_value) * factor

        # Clamp to bounds
        if dim.low is not None:
            new_val = max(dim.low, new_val)
        if dim.high is not None:
            new_val = min(dim.high, new_val)

        if dim.type == SearchSpaceType.INT:
            new_val = round(new_val)

        return new_val


class GridProposer(ExperimentProposer):
    """Systematic grid search over the search space."""

    def __init__(self, search_space: Any, points_per_dim: int = 3):
        self.search_space = search_space
        self.points_per_dim = max(2, points_per_dim)
        self._grid: list[dict[str, Any]] | None = None
        self._index = 0

    def propose(
        self,
        current_best: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], str]:
        if self._grid is None:
            self._grid = self._build_grid()

        if self._index >= len(self._grid):
            random.shuffle(self._grid)
            self._index = 0

        # Only keep search-space keys from current_best
        known = _dim_names(self.search_space)
        params = {k: v for k, v in current_best.items() if k in known}
        grid_point = self._grid[self._index]
        self._index += 1

        changes: list[str] = []
        for k, v in grid_point.items():
            if params.get(k) != v:
                changes.append(f"{k}={v}")
            params[k] = v

        desc = ", ".join(changes) if changes else "grid point (same as current)"
        return params, f"grid[{self._index}/{len(self._grid)}]: {desc}"

    def _build_grid(self) -> list[dict[str, Any]]:
        from stateset_agents.training.hpo.base import SearchSpaceType

        dim_values: list[list[tuple[str, Any]]] = []
        n = self.points_per_dim

        for dim in self.search_space.dimensions:
            if dim.type in (SearchSpaceType.CATEGORICAL, SearchSpaceType.CHOICE):
                values = [(dim.name, c) for c in dim.choices]
            elif dim.type == SearchSpaceType.INT:
                lo, hi = int(dim.low), int(dim.high)
                if hi - lo < n:
                    pts = list(range(lo, hi + 1))
                else:
                    pts = [lo + round(i * (hi - lo) / (n - 1)) for i in range(n)]
                values = [(dim.name, p) for p in pts]
            elif dim.type == SearchSpaceType.LOGUNIFORM:
                log_lo, log_hi = math.log(dim.low), math.log(dim.high)
                pts = [math.exp(log_lo + i * (log_hi - log_lo) / (n - 1)) for i in range(n)]
                values = [(dim.name, p) for p in pts]
            else:
                pts = [dim.low + i * (dim.high - dim.low) / (n - 1) for i in range(n)]
                values = [(dim.name, p) for p in pts]

            dim_values.append(values)

        grid = [dict(combo) for combo in product(*dim_values)]
        random.shuffle(grid)
        return grid


class BayesianProposer(ExperimentProposer):
    """Optuna-backed Bayesian optimization proposer.

    Uses Tree-structured Parzen Estimator (TPE) to suggest hyperparameters
    based on previous trial history.
    """

    def __init__(
        self,
        search_space: Any,
        direction: str = "maximize",
        study_name: str = "auto_research",
    ):
        self.search_space = search_space
        self.direction = direction
        self._study_name = study_name
        self._study: Any = None
        self._trial_count = 0
        self._current_trial: Any = None

    def _ensure_study(self) -> None:
        if self._study is not None:
            return
        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            self._study = optuna.create_study(
                study_name=self._study_name,
                direction=self.direction,
            )
        except ImportError as exc:
            raise ImportError(
                "BayesianProposer requires optuna. Install with: pip install optuna"
            ) from exc

    def propose(
        self,
        current_best: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], str]:
        import optuna
        from stateset_agents.training.hpo.base import SearchSpaceType

        self._ensure_study()

        # Sync study with any history entries it hasn't seen
        while self._trial_count < len(history):
            entry = history[self._trial_count]
            trial = self._study.ask()
            if entry["status"] == "crash":
                self._study.tell(trial, state=optuna.trial.TrialState.FAIL)
            else:
                self._study.tell(trial, entry["objective"])
            self._trial_count += 1

        # Ask for new suggestion
        trial = self._study.ask()
        known = _dim_names(self.search_space)
        params = {k: v for k, v in current_best.items() if k in known}
        changes: list[str] = []

        for dim in self.search_space.dimensions:
            if dim.type in (SearchSpaceType.CATEGORICAL, SearchSpaceType.CHOICE):
                val = trial.suggest_categorical(dim.name, dim.choices)
            elif dim.type == SearchSpaceType.INT:
                val = trial.suggest_int(dim.name, int(dim.low), int(dim.high))
            elif dim.type == SearchSpaceType.LOGUNIFORM:
                val = trial.suggest_float(dim.name, dim.low, dim.high, log=True)
            else:
                val = trial.suggest_float(dim.name, dim.low, dim.high)

            if val != params.get(dim.name):
                changes.append(f"{dim.name}={val}")
            params[dim.name] = val

        desc = ", ".join(changes) if changes else "bayesian sample (no changes)"
        self._current_trial = trial
        return params, f"bayesian: {desc}"

    def report_result(self, objective_value: float, crashed: bool = False) -> None:
        """Report the result of the most recent trial back to Optuna."""
        if self._current_trial is None:
            return
        try:
            import optuna

            if crashed:
                self._study.tell(
                    self._current_trial, state=optuna.trial.TrialState.FAIL
                )
            else:
                self._study.tell(self._current_trial, objective_value)
        except Exception as exc:
            logger.warning("Failed to report result to Optuna: %s", exc)
        finally:
            self._trial_count += 1
            self._current_trial = None


class AdaptivePerturbationProposer(PerturbationProposer):
    """PerturbationProposer that adapts its perturbation factor over time.

    Starts with broad exploration (large perturbations) and narrows down
    as improvements become harder to find — inspired by simulated annealing
    and Population Based Training.

    The factor decreases when experiments are being kept (exploit good region)
    and increases when experiments are being discarded (explore more).
    """

    def __init__(
        self,
        search_space: Any,
        initial_factor: float = 0.3,
        min_factor: float = 0.05,
        max_factor: float = 0.5,
        adaptation_window: int = 5,
        **kwargs: Any,
    ):
        super().__init__(search_space, perturbation_factor=initial_factor, **kwargs)
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.adaptation_window = adaptation_window

    def propose(
        self,
        current_best: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], str]:
        self._adapt_factor(history)
        params, desc = super().propose(current_best, history)
        return params, f"{desc} [factor={self.perturbation_factor:.3f}]"

    def _adapt_factor(self, history: list[dict[str, Any]]) -> None:
        if len(history) < self.adaptation_window:
            return

        recent = history[-self.adaptation_window :]
        keep_rate = sum(1 for r in recent if r["status"] == "keep") / len(recent)

        if keep_rate > 0.4:
            # Good improvement rate — narrow down
            self.perturbation_factor *= 0.9
        elif keep_rate < 0.15:
            # Poor rate — broaden search
            self.perturbation_factor *= 1.15

        self.perturbation_factor = max(
            self.min_factor, min(self.max_factor, self.perturbation_factor)
        )


class SmartPerturbationProposer(PerturbationProposer):
    """PerturbationProposer that learns which parameters matter.

    After an initial exploration phase, computes which parameters are
    most correlated with objective improvement, then preferentially
    perturbs those dimensions. Also adapts the perturbation factor.

    Combines the best of:
    - PerturbationProposer (neighborhood search)
    - AdaptivePerturbationProposer (annealing)
    - Hyperparameter importance analysis (focus on what matters)
    """

    def __init__(
        self,
        search_space: Any,
        initial_factor: float = 0.3,
        min_factor: float = 0.05,
        max_factor: float = 0.5,
        exploration_phase: int = 8,
        **kwargs: Any,
    ):
        super().__init__(search_space, perturbation_factor=initial_factor, **kwargs)
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.exploration_phase = exploration_phase
        self._importance: dict[str, float] = {}

    def propose(
        self,
        current_best: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], str]:
        self._adapt_factor(history)

        known = _dim_names(self.search_space)
        params = {k: v for k, v in current_best.items() if k in known}
        dims = list(self.search_space.dimensions)

        # Ensure all dims have values
        for dim in dims:
            if dim.name not in params:
                params[dim.name] = (
                    dim.default if dim.default is not None else _sample_dimension(dim)
                )

        if len(history) >= self.exploration_phase:
            # Smart phase: pick dimensions weighted by importance
            self._update_importance(history)
            to_change = self._weighted_select(dims)
        else:
            # Exploration phase: random selection
            n = min(self.num_params_to_change, len(dims))
            to_change = random.sample(dims, n)

        changes: list[str] = []
        for dim in to_change:
            old_val = params.get(dim.name)
            new_val = self._perturb(dim, old_val)
            if new_val != old_val:
                changes.append(f"{dim.name}: {old_val} → {new_val}")
                params[dim.name] = new_val

        if not changes:
            dim = random.choice(dims)
            new_val = _sample_dimension(dim)
            changes.append(f"{dim.name}: {params.get(dim.name)} → {new_val}")
            params[dim.name] = new_val

        # Validate types
        for dim in dims:
            val = params.get(dim.name)
            if val is not None and not self._is_valid_type(dim, val):
                params[dim.name] = _sample_dimension(dim)

        phase = "smart" if len(history) >= self.exploration_phase else "explore"
        desc = "; ".join(changes)
        factor_info = f"[{phase}, factor={self.perturbation_factor:.3f}]"
        return params, f"smart: {desc} {factor_info}"

    def _adapt_factor(self, history: list[dict[str, Any]]) -> None:
        if len(history) < 5:
            return
        recent = history[-5:]
        keep_rate = sum(1 for r in recent if r["status"] == "keep") / len(recent)
        if keep_rate > 0.4:
            self.perturbation_factor *= 0.9
        elif keep_rate < 0.15:
            self.perturbation_factor *= 1.15
        self.perturbation_factor = max(
            self.min_factor, min(self.max_factor, self.perturbation_factor)
        )

    def _update_importance(self, history: list[dict[str, Any]]) -> None:
        """Compute per-dimension importance from experiment history."""
        from .analysis import compute_parameter_importance

        # Build lightweight records for analysis
        class _Record:
            def __init__(self, params, objective_value, status):
                self.params = params
                self.objective_value = objective_value
                self.status = status

        records = [
            _Record(h["params"], h["objective"], h["status"])
            for h in history
        ]
        self._importance = compute_parameter_importance(records)

    def _weighted_select(self, dims: list[Any]) -> list[Any]:
        """Select dimensions weighted by importance (high-importance = more likely)."""
        n = min(self.num_params_to_change, len(dims))

        # Compute weights: importance + exploration bonus for unseen dims
        weights = []
        for dim in dims:
            imp = self._importance.get(dim.name, 0.5)  # Default to moderate
            # Add exploration bonus for dimensions with low importance score
            # (they might not have been tried enough)
            weight = max(0.1, imp + 0.1)
            weights.append(weight)

        # Weighted sampling without replacement
        selected: list[Any] = []
        available = list(range(len(dims)))
        for _ in range(n):
            if not available:
                break
            w = [weights[i] for i in available]
            total = sum(w)
            if total == 0:
                idx = random.choice(available)
            else:
                r = random.uniform(0, total)
                cumulative = 0.0
                idx = available[-1]
                for i in available:
                    cumulative += weights[i]
                    if cumulative >= r:
                        idx = i
                        break
            selected.append(dims[idx])
            available.remove(idx)

        return selected


def create_proposer(
    strategy: str,
    search_space: Any,
    direction: str = "maximize",
    **kwargs: Any,
) -> ExperimentProposer:
    """Factory function to create a proposer by strategy name."""
    if strategy == "random":
        return RandomProposer(search_space)
    elif strategy == "grid":
        return GridProposer(search_space, **kwargs)
    elif strategy == "bayesian":
        return BayesianProposer(search_space, direction=direction, **kwargs)
    elif strategy == "perturbation":
        return PerturbationProposer(search_space, **kwargs)
    elif strategy == "adaptive":
        return AdaptivePerturbationProposer(search_space, **kwargs)
    elif strategy == "smart":
        return SmartPerturbationProposer(search_space, **kwargs)
    elif strategy == "llm":
        from .llm_proposer import LLMProposer

        return LLMProposer(search_space, **kwargs)
    else:
        raise ValueError(f"Unknown proposer strategy: {strategy!r}")
