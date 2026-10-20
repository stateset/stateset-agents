"""Validated estimator and continual-learning state shared by checkpoint APIs."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from .diagnostics import apply_diagnostics, capture_diagnostics, prepare_diagnostics


def _reward_baseline(trainer: Any) -> dict[str, Any] | None:
    if not hasattr(trainer, "_global_reward_mean"):
        return None
    return _validate_reward_baseline(
        {
            "mean": trainer._global_reward_mean,
            "count": trainer._global_reward_count,
            "sum": getattr(trainer, "_global_reward_sum", None),
        }
    )


def _validate_reward_baseline(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict) or set(state) != {"mean", "count", "sum"}:
        raise ValueError("Invalid checkpoint reward baseline schema")
    count, mean, total = state["count"], state["mean"], state["sum"]
    if type(count) is not int or not 0 <= count <= 2**63 - 1:
        raise ValueError("Invalid checkpoint reward count")
    for name, value in (("mean", mean), ("sum", total)):
        if name == "sum" and value is None:
            continue
        try:
            valid = type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("Invalid checkpoint reward baseline value")
    if count == 0 and (mean != 0 or total not in (None, 0)):
        raise ValueError("Empty checkpoint reward baseline must be zero")
    if total is not None and count and total / count != mean:
        raise ValueError("Checkpoint reward sum and mean disagree")
    return dict(state)


def capture_learning_state(
    trainer: Any, *, include_reference: bool = True
) -> dict[str, Any]:
    """Capture the estimator, private replay streams and frozen task state."""
    manager = getattr(trainer, "continual_manager", None)
    continual = None
    if manager is not None:
        from .continual_learning import ContinualLearningManager

        continual = (
            manager.state_dict(include_reference=include_reference)
            if isinstance(manager, ContinualLearningManager)
            else manager.state_dict()
        )
    return {
        "schema_version": 2,
        "diagnostics": capture_diagnostics(trainer),
        "reward_baseline": _reward_baseline(trainer),
        "continual_state": continual,
        "continual_type": (
            f"{type(manager).__module__}.{type(manager).__qualname__}"
            if manager is not None
            else None
        ),
        "current_task_id": getattr(trainer, "_current_task_id", None),
        "reference_from_continual": (
            manager is not None
            and getattr(manager, "reference_model", None) is not None
            and getattr(trainer, "reference_model", None)
            is getattr(manager, "reference_model", None)
        ),
    }


def prepare_learning_state(trainer: Any, checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Stage a validated restore before any live policy weights change."""
    saved = checkpoint.get("learning_state")
    manager = getattr(trainer, "continual_manager", None)
    if "learning_state" in checkpoint and saved is None:
        raise ValueError("Invalid checkpoint learning state")
    if saved is None:
        if (
            hasattr(trainer, "_global_reward_mean")
            and getattr(trainer.config, "baseline_type", None) == "global_mean"
        ):
            raise ValueError("Legacy checkpoint has no global reward baseline state")
        baseline = (
            {"mean": 0.0, "count": 0, "sum": None}
            if hasattr(trainer, "_global_reward_mean")
            else None
        )
        continual = checkpoint.get("continual_state")
        task = checkpoint.get("current_task_id")
        reference_from_continual = manager is not None
    else:
        fields = {
            "schema_version",
            "reward_baseline",
            "continual_state",
            "continual_type",
            "current_task_id",
            "reference_from_continual",
        }
        if (
            not isinstance(saved, dict)
            or type(saved.get("schema_version")) is not int
            or saved["schema_version"] not in (1, 2)
            or set(saved)
            != (fields | {"diagnostics"} if saved["schema_version"] == 2 else fields)
        ):
            raise ValueError("Invalid checkpoint learning state schema")
        baseline = saved["reward_baseline"]
        if baseline is not None:
            baseline = _validate_reward_baseline(baseline)
        if (baseline is None) != (not hasattr(trainer, "_global_reward_mean")):
            raise ValueError("Checkpoint reward estimator differs from trainer")
        continual = saved["continual_state"]
        task = saved["current_task_id"]
        reference_from_continual = saved["reference_from_continual"]
        if type(reference_from_continual) is not bool or (
            reference_from_continual and manager is None
        ):
            raise ValueError("Invalid checkpoint continual reference relationship")
        expected_type = (
            f"{type(manager).__module__}.{type(manager).__qualname__}"
            if manager is not None
            else None
        )
        if saved["continual_type"] != expected_type:
            raise ValueError("Checkpoint continual manager differs from trainer")
    diagnostics = prepare_diagnostics(
        trainer,
        (
            saved["diagnostics"]
            if saved is not None and saved["schema_version"] == 2
            else None
        ),
    )
    if (
        saved is not None
        and saved["schema_version"] == 2
        and saved["diagnostics"] is None
    ):
        raise ValueError("Invalid checkpoint diagnostics availability")
    if task is not None and not isinstance(task, str):
        raise ValueError("Invalid checkpoint current task identity")
    if (manager is None) != (continual is None):
        raise ValueError("Checkpoint continual state availability differs from trainer")
    staged = None
    if manager is not None:
        if not isinstance(continual, dict):
            raise ValueError("Invalid checkpoint continual state")
        from .continual_learning import ContinualLearningManager

        if isinstance(manager, ContinualLearningManager):
            staged = ContinualLearningManager(
                deepcopy(manager.config),
                training_config=manager.training_config,
                seed=0,
            )
            staged._rng = deepcopy(manager._rng)
            staged.buffer._rng = deepcopy(manager.buffer._rng)
            staged.load_state_dict(continual, model=trainer.agent.model)
        else:
            staged = deepcopy(manager)
            staged.load_state_dict(continual)
    return {
        "reward_baseline": baseline,
        "manager": staged,
        "current_task_id": task,
        "reference_from_continual": reference_from_continual,
        "diagnostics": diagnostics,
    }


def apply_learning_state(trainer: Any, prepared: dict[str, Any]) -> None:
    """Commit prevalidated learning state and synchronize the task reference."""
    baseline = prepared["reward_baseline"]
    apply_diagnostics(trainer, prepared["diagnostics"])
    if baseline is not None:
        trainer._global_reward_mean = baseline["mean"]
        trainer._global_reward_count = baseline["count"]
        if baseline["sum"] is None:
            if hasattr(trainer, "_global_reward_sum"):
                delattr(trainer, "_global_reward_sum")
        else:
            trainer._global_reward_sum = baseline["sum"]
    manager = prepared["manager"]
    if manager is not None:
        # Preserve the public manager object's identity for callback holders.
        if hasattr(trainer.continual_manager, "__dict__") and hasattr(
            manager, "__dict__"
        ):
            trainer.continual_manager.__dict__.update(manager.__dict__)
        else:
            trainer.continual_manager = manager
        if prepared["reference_from_continual"]:
            trainer.reference_model = getattr(manager, "reference_model", None)
    if hasattr(trainer, "_current_task_id"):
        trainer._current_task_id = prepared["current_task_id"]
