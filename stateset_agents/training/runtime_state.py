"""Versioned checkpoint state and lifecycle shared by native RL trainers."""

from __future__ import annotations

import inspect
import json
import random
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Any, cast

import numpy as np

from .checkpoint_publication import (
    checkpoint_read,
    checkpoint_write,
    validate_checkpoint_publication,
)
from .model_checkpoint import load_model_checkpoint


def capture_rng(torch: Any) -> dict[str, Any]:
    """Capture generators using only tensors and weights-only-safe Python data."""
    numpy_state = cast(tuple[Any, ...], np.random.get_state())
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict[str, Any], torch: Any) -> None:
    """Restore RNGs after loading models, which may consume random numbers."""
    random.setstate(state["python"])
    values = state["numpy"]
    np.random.set_state(
        (values[0], np.asarray(values[1], dtype=np.uint32), *values[2:])
    )
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if (
            not torch.cuda.is_available()
            or len(state["cuda"]) != torch.cuda.device_count()
        ):
            raise ValueError("Checkpoint CUDA RNG topology differs from this runtime")
        torch.cuda.set_rng_state_all(state["cuda"])


def validate_training_config(config: Any) -> None:
    """Validate native configuration without importing the optional ML stack."""
    from .config import TrainingConfig

    if isinstance(config, TrainingConfig):
        config.validate_for_training()


def capture_rollout_state(agent: Any) -> dict[str, int] | None:
    """Capture an engine publication version when the agent exposes one."""
    version = getattr(agent, "rollout_backend_version", None)
    if type(version) is not int:
        return None
    if version < 0:
        raise ValueError("Invalid rollout publication version")
    return {"schema_version": 1, "publication_version": version}


def restore_rollout_state(agent: Any, state: Any) -> None:
    """Keep publication versions monotonic before publishing restored weights."""
    if state is None:
        return
    if (
        not isinstance(state, dict)
        or set(state) != {"schema_version", "publication_version"}
        or state["schema_version"] != 1
        or type(state["publication_version"]) is not int
        or state["publication_version"] < 0
    ):
        raise ValueError("Invalid saved rollout state")
    current = getattr(agent, "_rollout_backend_version", None)
    if type(current) is int:
        agent._rollout_backend_version = max(current, state["publication_version"])


_COMPONENTS = (
    "optimizer",
    "scheduler",
    "lr_scheduler",
    "scaler",
    "value_head",
    "ref_model",
    "policy_optimizer",
    "value_optimizer",
    "actor_optimizer",
    "critic_optimizer",
    "policy_scheduler",
    "value_scheduler",
    "actor_scheduler",
    "critic_scheduler",
)
_COUNTERS = (
    "global_step",
    "current_epoch",
    "rollout_samples_total",
    "_grad_accum_step",
    "steps_without_improvement",
    "value_warmup_complete",
    "metrics_history",
    "training_metrics",
    "best_eval_metric",
    "_current_task_id",
)


def _model(trainer: Any) -> Any:
    model = getattr(trainer, "model", None)
    if model is None:
        model = getattr(getattr(trainer, "agent", None), "model", None)
    if model is None:
        raise ValueError("Trainer has no initialized policy model")
    return model


def _gradient_parameters(trainer: Any) -> dict[str, Any]:
    parameters = dict(_model(trainer).named_parameters())
    head = getattr(trainer, "value_head", None)
    if head is not None:
        parameters.update(
            {"value_head." + name: value for name, value in head.named_parameters()}
        )
    return parameters


def _profile_phase(method: Any, name: str) -> Any:
    @wraps(method)
    async def measured(self: Any, *args: Any, **kwargs: Any) -> Any:
        if getattr(self, "_runtime_profile_active", False):
            return await method(self, *args, **kwargs)
        self._runtime_profile_active = True
        started = time.perf_counter()
        try:
            result = await method(self, *args, **kwargs)
        finally:
            self._runtime_profile_active = False
        seconds = time.perf_counter() - started
        if name == "generate_trajectories":
            tokens = successes = 0
            outcomes_known = True
            for group in result:
                for trajectory in group.trajectories:
                    outcome = (getattr(trajectory, "final_state", None) or {}).get(
                        "task_success"
                    )
                    if type(outcome) is not bool:
                        outcome = (
                            (getattr(trajectory, "metadata", None) or {}).get(
                                "reward_breakdown"
                            )
                            or {}
                        ).get("task_success")
                    if type(outcome) is bool or (
                        type(outcome) in (int, float) and outcome in (0, 1)
                    ):
                        successes += int(outcome)
                    else:
                        outcomes_known = False
                    for turn in trajectory.turns:
                        if turn.role == "assistant":
                            tokens += len((turn.metadata or {}).get("token_ids", ()))
            self._runtime_rollout_metrics = {
                "rollout_seconds": seconds,
                "generated_tokens": tokens,
            }
            if outcomes_known:
                self._runtime_rollout_metrics["successful_episodes"] = successes
        elif isinstance(result, dict):
            rollout = getattr(self, "_runtime_rollout_metrics", {})
            result.update(rollout)
            result["step_seconds"] = seconds
            elapsed = seconds + float(rollout.get("rollout_seconds", 0))
            if rollout:
                result["generated_tokens_per_second"] = (
                    float(rollout["generated_tokens"]) / elapsed
                )
            hourly = getattr(self.config, "compute_cost_per_hour", None)
            if isinstance(hourly, (int, float)) and not isinstance(hourly, bool):
                cost = elapsed * hourly / 3600
                result["estimated_compute_cost_usd"] = cost
                if cost > 0 and "successful_episodes" in rollout:
                    result["successful_episodes_per_usd"] = (
                        float(rollout.get("successful_episodes", 0)) / cost
                    )
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_available():
                result["process_peak_vram_bytes"] = torch.cuda.max_memory_allocated()
            self._runtime_rollout_metrics = {}
        return result

    return measured


class TrainingRuntime:
    """Common configuration and resumable lifecycle for native algorithms.

    Sampling and objective computation remain with algorithms. Named stateful
    components make actor/critic and policy-only trainers use the same protocol.
    """

    config: Any

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Measure native phases consistently without changing algorithm signatures."""
        super().__init_subclass__(**kwargs)
        for name in ("train_step", "training_step", "generate_trajectories"):
            method = cls.__dict__.get(name)
            if method is not None and inspect.iscoroutinefunction(method):
                setattr(cls, name, _profile_phase(method, name))

    def save_runtime_checkpoint(self, output_dir: str) -> None:
        """Atomically publish model, components, pending gradients and RNGs."""
        import torch

        model = _model(self)
        parameters = _gradient_parameters(self)
        components = {
            name: getattr(self, name).state_dict()
            for name in _COMPONENTS
            if getattr(self, name, None) is not None
        }
        state = {
            "runtime_schema": 1,
            "trainer": type(self).__name__,
            "config": self.config.to_dict(),
            "components": components,
            "counters": {
                name: getattr(self, name) for name in _COUNTERS if hasattr(self, name)
            },
            "gradients": {
                name: None if p.grad is None else p.grad.detach().cpu().clone()
                for name, p in parameters.items()
            },
            "rng": capture_rng(torch),
            "rollout_state": capture_rollout_state(getattr(self, "agent", None)),
        }
        # Retain scalar keys for tools which inspect training_state.pt.
        state.update(state["counters"])
        with checkpoint_write(Path(output_dir)) as staging:
            model.save_pretrained(staging)
            tokenizer = getattr(self, "tokenizer", None)
            if tokenizer is not None:
                tokenizer.save_pretrained(staging)
            torch.save(state, staging / "training_state.pt")
            config_json = (
                json.dumps(state["config"], indent=2, sort_keys=True, allow_nan=False)
                + "\n"
            )
            (staging / "resolved_config.json").write_text(config_json)
            legacy_config = {
                "DAPOTrainer": "dapo_config.json",
                "GEPOTrainer": "gepo_config.json",
                "VAPOTrainer": "vapo_config.json",
            }.get(type(self).__name__)
            if legacy_config is not None:
                (staging / legacy_config).write_text(config_json)
            if "value_head" in components:
                torch.save(components["value_head"], staging / "value_head.pt")
            if hasattr(self, "training_metrics"):
                (staging / "training_metrics.json").write_text(
                    json.dumps(self.training_metrics, indent=2, allow_nan=False) + "\n"
                )

    def load_runtime_checkpoint(
        self, checkpoint_dir: str, *, trusted: bool = False
    ) -> bool:
        """Restore every registered component; reject partial or mismatched state."""
        import torch

        from .checkpoint_io import load_checkpoint_file

        path = Path(checkpoint_dir)
        with checkpoint_read(path):
            validate_checkpoint_publication(path, required=True)
            state = load_checkpoint_file(
                path / "training_state.pt", trusted=trusted, torch_module=torch
            )
            if (
                state.get("runtime_schema") != 1
                or state.get("trainer") != type(self).__name__
            ):
                raise ValueError("Checkpoint runtime schema or trainer differs")
            for name in (
                "model_name",
                "model_revision",
                "objective",
                "objective_overrides",
                "gradient_accumulation_steps",
                "max_completion_length",
                "num_generations",
                "beta",
            ):
                if state["config"].get(name) != self.config.to_dict().get(name):
                    raise ValueError(f"Checkpoint configuration differs: {name}")
            components = {
                name: getattr(self, name)
                for name in _COMPONENTS
                if getattr(self, name, None) is not None
            }
            if set(components) != set(state["components"]):
                raise ValueError("Checkpoint stateful components differ")
            model = _model(self)
            parameters = _gradient_parameters(self)
            if set(parameters) != set(state["gradients"]):
                raise ValueError("Checkpoint parameter names differ")
            for name, gradient in state["gradients"].items():
                if gradient is not None and (
                    gradient.shape != parameters[name].shape
                    or gradient.dtype != parameters[name].dtype
                    or not torch.isfinite(gradient).all()
                ):
                    raise ValueError(f"Invalid saved gradient: {name}")
            if not load_model_checkpoint(
                model, path, torch_module=torch, trusted=trusted, strict=True
            ):
                raise ValueError("Checkpoint has no loadable model")
            for name, component in components.items():
                component.load_state_dict(state["components"][name])
            for name, value in state["counters"].items():
                if name not in _COUNTERS:
                    raise ValueError("Unknown checkpoint counter")
                setattr(self, name, value)
            for name, parameter in parameters.items():
                value = state["gradients"][name]
                parameter.grad = None if value is None else value.to(parameter.device)
            restore_rng(state["rng"], torch)
            agent = getattr(self, "agent", None)
            restore_rollout_state(agent, state.get("rollout_state"))
            sync = getattr(agent, "sync_rollout_backend", None)
            if callable(sync):
                sync()
        return True
