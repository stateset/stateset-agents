"""
Weights & Biases Sweeps backend for hyperparameter optimization.

This backend integrates StateSet Agents HPO with W&B Sweeps.
It is best suited for team workflows and experiment tracking.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .base import HPOBackend, HPOCallback, HPOResult, HPOSummary, SearchDimension, SearchSpaceType

try:
    import wandb  # type: ignore

    WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


def _require_wandb() -> None:
    if not WANDB_AVAILABLE:
        raise ImportError(
            "wandb is required for this HPO backend. "
            "Install with `pip install stateset-agents[hpo]` or `pip install wandb`."
        )


def _resolve_async_value(value: Any) -> Any:
    if not asyncio.iscoroutine(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, value).result()


class WandBSweepsBackend(HPOBackend):
    """W&B sweeps backend."""

    def __init__(
        self,
        search_space,
        objective_metric: str = "reward",
        direction: str = "maximize",
        callbacks: Optional[List[HPOCallback]] = None,
        output_dir: Optional[Path] = None,
        method: str = "bayes",
        project: str = "stateset-hpo",
        entity: Optional[str] = None,
        sweep_name: Optional[str] = None,
    ):
        super().__init__(search_space, objective_metric, direction, callbacks)
        self.output_dir = Path(output_dir or "./hpo_results")
        self.method = method
        self.project = project
        self.entity = entity
        self.sweep_name = sweep_name or f"wandb_hpo_{int(time.time())}"

    def _convert_dimension(self, dim: SearchDimension) -> Dict[str, Any]:
        if dim.type in (SearchSpaceType.CATEGORICAL, SearchSpaceType.CHOICE):
            return {"values": dim.choices}
        if dim.type == SearchSpaceType.INT:
            return {
                "distribution": "int_uniform",
                "min": int(dim.low),
                "max": int(dim.high),
            }
        if dim.type == SearchSpaceType.LOGUNIFORM:
            return {
                "distribution": "log_uniform_values",
                "min": float(dim.low),
                "max": float(dim.high),
            }
        if dim.type in (SearchSpaceType.FLOAT, SearchSpaceType.UNIFORM):
            return {
                "distribution": "uniform",
                "min": float(dim.low),
                "max": float(dim.high),
            }
        raise ValueError(f"Unsupported search space type for W&B: {dim.type}")

    def _build_sweep_config(self) -> Dict[str, Any]:
        params = {dim.name: self._convert_dimension(dim) for dim in self.search_space.dimensions}
        goal = "maximize" if self.direction == "maximize" else "minimize"
        return {
            "name": self.sweep_name,
            "method": self.method,
            "metric": {"name": self.objective_metric, "goal": goal},
            "parameters": params,
        }

    async def suggest_params(self, trial_id: str) -> Dict[str, Any]:
        raise NotImplementedError("W&B sweeps suggest params internally via optimize().")

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
        _require_wandb()

        sweep_config = self._build_sweep_config()
        self.results = []
        self._notify_hpo_start(n_trials)
        start_time = time.time()

        sweep_id = wandb.sweep(  # type: ignore[attr-defined]
            sweep_config, project=self.project, entity=self.entity
        )

        def run_trial() -> None:
            run = wandb.init(  # type: ignore[attr-defined]
                project=self.project,
                entity=self.entity,
                dir=str(self.output_dir),
                config={},
            )
            assert run is not None
            params = dict(wandb.config)
            trial_id = run.id
            self._notify_trial_start(trial_id, params)
            trial_start = time.time()
            status = "success"
            metrics: Dict[str, float] = {}
            try:
                metric_value = float(_resolve_async_value(objective_fn(params)))
                metrics[self.objective_metric] = metric_value
                wandb.log(metrics)
            except Exception as e:
                status = "failed"
                logger.error(f"W&B trial {trial_id} failed: {e}")
                wandb.log({self.objective_metric: float('-inf')})
            training_time = time.time() - trial_start
            result = HPOResult(
                trial_id=str(trial_id),
                params=params,
                metrics=metrics,
                best_metric=metrics.get(self.objective_metric, float("-inf")),
                training_time=training_time,
                status=status,
            )
            self.results.append(result)
            self._notify_trial_end(result)
            wandb.finish()

        # wandb.agent is blocking; run in a worker thread if already in an event loop.
        def run_agent() -> None:
            wandb.agent(  # type: ignore[attr-defined]
                sweep_id,
                function=run_trial,
                count=n_trials,
                project=self.project,
                entity=self.entity,
            )

        try:
            asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(run_agent).result(timeout=timeout)
        except RuntimeError:
            run_agent()

        total_time = time.time() - start_time

        best_metric = float("-inf") if self.direction == "maximize" else float("inf")
        best_params: Dict[str, Any] = {}
        best_trial_id = ""
        for r in self.results:
            val = r.metrics.get(self.objective_metric)
            if val is None:
                continue
            if (self.direction == "maximize" and val > best_metric) or (
                self.direction == "minimize" and val < best_metric
            ):
                best_metric = val
                best_params = r.params
                best_trial_id = r.trial_id

        summary = HPOSummary(
            best_params=best_params,
            best_metric=best_metric,
            best_trial_id=best_trial_id,
            total_trials=len(self.results),
            successful_trials=len([r for r in self.results if r.status == "success"]),
            failed_trials=len([r for r in self.results if r.status == "failed"]),
            pruned_trials=0,
            total_time=total_time,
            all_results=self.results,
        )

        self._notify_hpo_end(summary)
        return summary


__all__ = ["WandBSweepsBackend", "WANDB_AVAILABLE"]

