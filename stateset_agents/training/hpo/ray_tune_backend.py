"""
Ray Tune backend for hyperparameter optimization.

This backend integrates StateSet Agents HPO with Ray Tune for scalable,
distributed hyperparameter search.

Ray Tune supports:
- Distributed trial execution across CPUs/GPUs/nodes
- Advanced schedulers (ASHA, HyperBand, FIFO)
- Multiple search algorithms (BayesOpt, random, grid)
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .base import HPOBackend, HPOCallback, HPOResult, HPOSummary, SearchDimension, SearchSpaceType

try:
    import ray  # type: ignore
    from ray import tune  # type: ignore
    from ray.tune.schedulers import (  # type: ignore
        ASHAScheduler,
        HyperBandScheduler,
        FIFOScheduler,
    )
    from ray.tune.search.basic_variant import BasicVariantGenerator  # type: ignore
    from ray.tune.search.bayesopt import BayesOptSearch  # type: ignore

    RAY_AVAILABLE = True
except ImportError:  # pragma: no cover
    ray = None  # type: ignore
    tune = None  # type: ignore
    ASHAScheduler = HyperBandScheduler = FIFOScheduler = None  # type: ignore
    BasicVariantGenerator = BayesOptSearch = None  # type: ignore
    RAY_AVAILABLE = False

logger = logging.getLogger(__name__)


def _require_ray() -> None:
    if not RAY_AVAILABLE:
        raise ImportError(
            "Ray Tune is required for this HPO backend. "
            "Install with `pip install stateset-agents[hpo]` or `pip install ray[tune]`."
        )


def _resolve_async_value(value: Any) -> Any:
    """Resolve a possibly-async objective to a concrete value."""
    if not asyncio.iscoroutine(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, value).result()


class RayTuneBackend(HPOBackend):
    """Ray Tune backend for HPO."""

    def __init__(
        self,
        search_space,
        objective_metric: str = "reward",
        direction: str = "maximize",
        callbacks: Optional[List[HPOCallback]] = None,
        output_dir: Optional[Path] = None,
        study_name: Optional[str] = None,
        scheduler: str = "asha",
        search_alg: str = "bayesopt",
        max_concurrent: int = 4,
        cpu_per_trial: int = 4,
        gpu_per_trial: float = 0.0,
        ray_init_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(search_space, objective_metric, direction, callbacks)
        self.output_dir = Path(output_dir or "./hpo_results")
        self.study_name = study_name or f"ray_hpo_{int(time.time())}"
        self.scheduler_name = scheduler
        self.search_alg_name = search_alg
        self.max_concurrent = max_concurrent
        self.cpu_per_trial = cpu_per_trial
        self.gpu_per_trial = gpu_per_trial
        self.ray_init_kwargs = ray_init_kwargs or {}

    def _convert_dimension(self, dim: SearchDimension):
        """Convert SearchDimension to Ray Tune distribution."""
        _require_ray()
        if dim.type in (SearchSpaceType.FLOAT, SearchSpaceType.UNIFORM):
            return tune.uniform(dim.low, dim.high)  # type: ignore[arg-type]
        if dim.type == SearchSpaceType.LOGUNIFORM:
            return tune.loguniform(dim.low, dim.high)  # type: ignore[arg-type]
        if dim.type == SearchSpaceType.INT:
            return tune.randint(int(dim.low), int(dim.high) + 1)  # type: ignore[arg-type]
        if dim.type in (SearchSpaceType.CATEGORICAL, SearchSpaceType.CHOICE):
            return tune.choice(dim.choices)
        raise ValueError(f"Unsupported search space type for Ray Tune: {dim.type}")

    def _convert_search_space(self) -> Dict[str, Any]:
        """Convert SearchSpace to Ray Tune param_space."""
        return {dim.name: self._convert_dimension(dim) for dim in self.search_space.dimensions}

    def _create_scheduler(self):
        _require_ray()
        if self.scheduler_name == "asha":
            return ASHAScheduler(metric=self.objective_metric, mode=self._mode())
        if self.scheduler_name == "hyperband":
            return HyperBandScheduler(metric=self.objective_metric, mode=self._mode())
        if self.scheduler_name == "fifo":
            return FIFOScheduler()
        logger.warning(f"Unknown scheduler '{self.scheduler_name}', defaulting to ASHA.")
        return ASHAScheduler(metric=self.objective_metric, mode=self._mode())

    def _create_search_alg(self):
        _require_ray()
        if self.search_alg_name == "bayesopt":
            return BayesOptSearch(metric=self.objective_metric, mode=self._mode())
        if self.search_alg_name == "random":
            return BasicVariantGenerator()
        if self.search_alg_name == "grid":
            return BasicVariantGenerator()
        logger.warning(f"Unknown search_alg '{self.search_alg_name}', defaulting to random.")
        return BasicVariantGenerator()

    def _mode(self) -> str:
        return "max" if self.direction == "maximize" else "min"

    async def suggest_params(self, trial_id: str) -> Dict[str, Any]:
        raise NotImplementedError("Ray Tune suggests params internally via optimize().")

    async def report_result(self, result: HPOResult) -> None:
        self.results.append(result)

    async def should_prune(self, trial_id: str, intermediate_metrics: Dict[str, float]) -> bool:
        return False

    async def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        n_trials: int = 100,
        timeout: Optional[float] = None,
    ) -> HPOSummary:
        """Run Ray Tune optimization."""
        _require_ray()

        if ray is not None and not ray.is_initialized():
            ray.init(ignore_reinit_error=True, include_dashboard=False, **self.ray_init_kwargs)  # type: ignore[arg-type]

        param_space = self._convert_search_space()
        scheduler = self._create_scheduler()
        search_alg = self._create_search_alg()

        self.results = []
        self._notify_hpo_start(n_trials)
        start_time = time.time()

        def trainable(config: Dict[str, Any]) -> None:
            trial_id = tune.get_trial_id() if tune else "unknown"
            self._notify_trial_start(trial_id, config)
            trial_start = time.time()
            status = "success"
            metrics: Dict[str, float] = {}
            try:
                metric_value = float(_resolve_async_value(objective_fn(config)))
                metrics[self.objective_metric] = metric_value
                if tune:
                    tune.report(**metrics)
            except Exception as e:
                status = "failed"
                logger.error(f"Ray Tune trial {trial_id} failed: {e}")
                metrics = {}
                if tune:
                    tune.report(**{self.objective_metric: float("-inf")})
            training_time = time.time() - trial_start
            result = HPOResult(
                trial_id=str(trial_id),
                params=config,
                metrics=metrics,
                best_metric=metrics.get(self.objective_metric, float("-inf")),
                training_time=training_time,
                status=status,
            )
            self.results.append(result)
            self._notify_trial_end(result)

        resources = {"cpu": self.cpu_per_trial}
        if self.gpu_per_trial:
            resources["gpu"] = self.gpu_per_trial

        tune_config = tune.TuneConfig(  # type: ignore[attr-defined]
            num_samples=n_trials,
            metric=self.objective_metric,
            mode=self._mode(),
            scheduler=scheduler,
            search_alg=search_alg,
            max_concurrent_trials=self.max_concurrent,
            time_budget_s=timeout,
        )
        run_config = tune.RunConfig(  # type: ignore[attr-defined]
            name=self.study_name,
            local_dir=str(self.output_dir),
        )

        tuner = tune.Tuner(  # type: ignore[attr-defined]
            tune.with_resources(trainable, resources),  # type: ignore[arg-type]
            tune_config=tune_config,
            run_config=run_config,
            param_space=param_space,
        )

        result_grid = tuner.fit()
        total_time = time.time() - start_time

        best_result = result_grid.get_best_result(metric=self.objective_metric, mode=self._mode())
        best_params = best_result.config
        best_metric = best_result.metrics.get(self.objective_metric, float("-inf"))

        summary = HPOSummary(
            best_params=best_params,
            best_metric=float(best_metric),
            best_trial_id=str(best_result.path),
            total_trials=len(self.results),
            successful_trials=len([r for r in self.results if r.status == "success"]),
            failed_trials=len([r for r in self.results if r.status == "failed"]),
            pruned_trials=0,
            total_time=total_time,
            all_results=self.results,
        )

        self._notify_hpo_end(summary)
        return summary


__all__ = ["RayTuneBackend", "RAY_AVAILABLE"]

