"""Versioned checkpoint state and lifecycle shared by native RL trainers."""

from __future__ import annotations

import copy
import inspect
import json
import math
import random
import sys
import time
from functools import partial, wraps
from numbers import Integral, Real
from pathlib import Path
from types import BuiltinFunctionType, FunctionType
from typing import Any, cast

import numpy as np

from .checkpoint_publication import (
    checkpoint_read,
    checkpoint_write,
    validate_checkpoint_publication,
)
from .learning_state import (
    apply_learning_state,
    capture_learning_state,
    prepare_learning_state,
)
from .model_checkpoint import load_model_checkpoint
from .profiling import _observed_assistant_tokens


def _reject_unsupported_resume(config: Any, entry_point: str) -> None:
    """Reject an unsupported continuation request before starting fresh work."""
    if getattr(config, "resume_from_checkpoint", None) is not None:
        raise ValueError(
            f"{entry_point} does not support resume_from_checkpoint; "
            "no fresh training started"
        )


def _require_runtime_ready(trainer: Any) -> None:
    """Block an ambiguous learner until a complete runtime reload succeeds."""
    if getattr(trainer, "_runtime_checkpoint_poisoned", False):
        raise RuntimeError(
            "Training runtime is quarantined after a failed state commit; "
            "a complete successful runtime reload is required"
        )


def _guard_runtime_method(method: Any) -> Any:
    if inspect.iscoroutinefunction(method):

        @wraps(method)
        async def guarded_async(self: Any, *args: Any, **kwargs: Any) -> Any:
            _require_runtime_ready(self)
            return await method(self, *args, **kwargs)

        return guarded_async

    @wraps(method)
    def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
        _require_runtime_ready(self)
        return method(self, *args, **kwargs)

    return guarded


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
        # A CUDA map_location also moves serialized generator states onto the
        # GPU. PyTorch requires CPU ByteTensors even for CUDA generators.
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


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
    "_next_episode",
    "rollout_samples_total",
    "_grad_accum_step",
    "steps_without_improvement",
    "value_warmup_complete",
    "metrics_history",
    "training_metrics",
    "best_eval_metric",
    "_current_task_id",
)


def _validate_counters(counters: Any, trainer: Any = None) -> None:
    """Validate restored cursor and metric fields without changing the learner."""
    if not isinstance(counters, dict) or set(counters) - set(_COUNTERS):
        raise ValueError("Invalid checkpoint counter schema")
    if trainer is not None:
        required = {
            name
            for name in _COUNTERS
            if hasattr(trainer, name)
            and name not in {"_current_task_id", "_next_episode"}
        }
        if required - set(counters):
            raise ValueError("Checkpoint is missing initialized learner counters")
    for name, value in counters.items():
        if name in {"metrics_history", "training_metrics"}:
            if not isinstance(value, dict) or any(
                not isinstance(key, str)
                or not isinstance(values, list)
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, Real)
                    or not math.isfinite(float(item))
                    for item in values
                )
                for key, values in value.items()
            ):
                raise ValueError(f"Invalid checkpoint metric history: {name}")
        elif name == "best_eval_metric":
            # Initial positive/negative infinity is a legitimate best-score
            # sentinel; NaN never defines a valid comparison.
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or math.isnan(float(value))
            ):
                raise ValueError("Invalid checkpoint best evaluation metric")
        elif name == "_current_task_id":
            if value is not None and not isinstance(value, str):
                raise ValueError("Invalid checkpoint task identifier")
        elif name == "value_warmup_complete":
            if type(value) is not bool:
                raise ValueError("Invalid checkpoint warmup flag")
        elif type(value) is not int or value < 0:
            raise ValueError(f"Invalid checkpoint counter: {name}")


def _plain_counter_snapshot(counters: dict[str, Any]) -> dict[str, Any]:
    """Own validated counter data using weights-only-safe Python values."""

    def plain(value: Any) -> Any:
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, Integral):
            return int(value)
        if isinstance(value, Real):
            return float(value)
        if isinstance(value, str):
            return str(value)
        if isinstance(value, list):
            return [plain(item) for item in value]
        if isinstance(value, dict):
            return {str(key): plain(item) for key, item in value.items()}
        raise ValueError("Invalid checkpoint counter value")

    return cast(dict[str, Any], plain(counters))


def _validate_rollout_state(state: Any) -> None:
    if state is not None and (
        not isinstance(state, dict)
        or set(state) != {"schema_version", "publication_version"}
        or type(state["schema_version"]) is not int
        or state["schema_version"] != 1
        or type(state["publication_version"]) is not int
        or state["publication_version"] < 0
    ):
        raise ValueError("Invalid saved rollout state")


def _adaptive_kl_state(trainer: Any) -> dict[str, Any] | None:
    controller = getattr(trainer, "kl_controller", None)
    if controller is None:
        return None
    state = {
        "schema_version": 1,
        "kl_coef": controller.kl_coef,
        "target_kl": controller.target_kl,
        "horizon": controller.horizon,
    }
    _prepare_adaptive_kl(trainer, state)
    return state


def _prepare_adaptive_kl(trainer: Any, saved: Any) -> dict[str, Any] | None:
    controller = getattr(trainer, "kl_controller", None)
    if controller is None:
        if saved is not None:
            raise ValueError("Checkpoint adaptive KL controller availability differs")
        return None
    if (
        not isinstance(saved, dict)
        or set(saved) != {"schema_version", "kl_coef", "target_kl", "horizon"}
        or type(saved["schema_version"]) is not int
        or saved["schema_version"] != 1
    ):
        raise ValueError("Checkpoint lacks valid adaptive KL controller state")
    for name in ("kl_coef", "target_kl"):
        value = saved[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid checkpoint adaptive KL controller state")
    if (
        saved["target_kl"] <= 0
        or type(saved["horizon"]) is not int
        or saved["horizon"] < 1
    ):
        raise ValueError("Invalid checkpoint adaptive KL controller state")
    if (
        saved["target_kl"] != controller.target_kl
        or saved["horizon"] != controller.horizon
    ):
        raise ValueError("Checkpoint adaptive KL controller protocol differs")
    return dict(saved)


def _module_state_schema(module: Any, saved: Any, torch: Any) -> None:
    """Check module state without cloning potentially enormous reference weights."""
    from .state_fingerprint import state_fingerprint

    if type(module).load_state_dict is not torch.nn.Module.load_state_dict:
        raise ValueError("Custom checkpoint module loaders are unsupported")
    if any(
        child._load_state_dict_pre_hooks or child._load_state_dict_post_hooks
        for child in module.modules()
    ):
        raise ValueError("Checkpoint module load hooks are unsupported")
    expected = module.state_dict()
    if not isinstance(saved, dict) or set(saved) != set(expected):
        raise ValueError("Checkpoint module state names differ")
    for name, target in expected.items():
        value = saved[name]
        if (
            not torch.is_tensor(value)
            or not torch.is_tensor(target)
            or value.shape != target.shape
            or value.dtype != target.dtype
        ):
            raise ValueError("Checkpoint module tensor schema differs")
    parameters = dict(module.named_parameters(remove_duplicate=False))
    state_fingerprint({name: saved[name] for name in parameters})
    aliases: dict[int, Any] = {}
    for name, target in module.named_parameters(remove_duplicate=False):
        value = saved[name]
        if id(target) in aliases and not torch.equal(value, aliases[id(target)]):
            raise ValueError("Checkpoint module tied parameter values differ")
        aliases[id(target)] = value


def _component_identities(components: dict[str, Any], torch: Any) -> dict[str, Any]:
    from .scheduler_identity import scheduler_identity

    identities = {}
    for name, component in components.items():
        identity: dict[str, Any] = {
            "kind": f"{type(component).__module__}.{type(component).__qualname__}"
        }
        if isinstance(component, torch.optim.lr_scheduler.LRScheduler) or isinstance(
            component, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            identity["scheduler"] = scheduler_identity(component)
        identities[name] = identity
    return identities


def _scheduler_schema(
    value: Any,
    expected: Any,
    torch: Any,
    *,
    allow_sentinels: bool = False,
    field: str = "",
) -> None:
    """Check plain scheduler data without executing a schedule or its callables."""
    if torch.is_tensor(expected):
        if (
            not torch.is_tensor(value)
            or value.shape != expected.shape
            or value.dtype != expected.dtype
        ):
            raise ValueError("Checkpoint scheduler tensor schema differs")
        from .state_fingerprint import state_fingerprint

        state_fingerprint(value)
        if field in {"base_lrs", "_last_lr", "min_lrs"} and (
            value.numel() != 1
            or value.dtype == torch.bool
            or value.is_complex()
            or value.item() < 0
        ):
            raise ValueError("Invalid checkpoint scheduler learning-rate tensor")
    elif isinstance(expected, dict):
        if not isinstance(value, dict) or set(value) != set(expected):
            raise ValueError("Checkpoint scheduler mapping schema differs")
        for name in expected:
            _scheduler_schema(
                value[name],
                expected[name],
                torch,
                allow_sentinels=allow_sentinels,
                field=str(name),
            )
    elif isinstance(expected, (list, tuple)):
        if type(value) is not type(expected) or len(value) != len(expected):
            raise ValueError("Checkpoint scheduler sequence schema differs")
        for item, template in zip(value, expected, strict=True):
            _scheduler_schema(
                item, template, torch, allow_sentinels=allow_sentinels, field=field
            )
    elif type(expected) is bool:
        if type(value) is not bool:
            raise ValueError("Checkpoint scheduler boolean schema differs")
    elif type(expected) is int:
        if type(value) is not int:
            raise ValueError("Checkpoint scheduler integer schema differs")
        if field == "last_epoch" and value < -1 or field == "_step_count" and value < 0:
            raise ValueError("Invalid checkpoint scheduler counter")
    elif isinstance(expected, Real):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("Checkpoint scheduler numeric schema differs")
        if not math.isfinite(float(value)) and not (
            allow_sentinels
            and field in {"best", "mode_worse"}
            and math.isinf(float(value))
        ):
            raise ValueError("Checkpoint scheduler values must be finite")
        if field in {"base_lrs", "_last_lr", "min_lrs"} and value < 0:
            raise ValueError("Checkpoint scheduler learning rates must be nonnegative")
    elif expected is None or isinstance(expected, str):
        if type(value) is not type(expected):
            raise ValueError("Checkpoint scheduler scalar schema differs")
    else:
        raise ValueError("Opaque checkpoint scheduler fields are unsupported")


def _scheduler_copy_graph(scheduler: Any, optimizers: set[int], torch: Any) -> None:
    """Admit the actual deepcopy graph before any copying can invoke user hooks.

    Standard composite scheduler children recurse. Lambda bindings may be plain
    functions or exact partials containing plain data; callable objects/custom
    copy protocols are unsupported. No schedule or callable is executed.
    """
    from .scheduler_identity import callable_identity

    active: set[int] = set()
    seen: set[int] = set()
    copy_protocols = {
        "__copy__",
        "__deepcopy__",
        "__reduce__",
        "__reduce_ex__",
        "__getstate__",
        "__setstate__",
    }

    def plain(value: Any) -> None:
        if value is None or type(value) in (bool, int, float, str, bytes):
            return
        if type(value) in (FunctionType, BuiltinFunctionType):
            callable_identity(value)
            return
        if type(value) in (torch.Tensor, torch.nn.Parameter):
            if (
                value.numel() > 1024
                or value.layout != torch.strided
                or value.device.type == "meta"
            ):
                raise ValueError(
                    "Checkpoint scheduler tensors must be small materialized metadata"
                )
            if copy_protocols & vars(value).keys():
                raise ValueError(
                    "Custom scheduler tensor copy protocols are unsupported"
                )
            plain(vars(value))
            return
        if id(value) in active:
            raise ValueError("Cyclic checkpoint scheduler bindings are unsupported")
        active.add(id(value))
        try:
            if type(value) is partial:
                if copy_protocols & vars(value).keys():
                    raise ValueError(
                        "Custom scheduler partial copy protocols are unsupported"
                    )
                if type(value.func) not in (FunctionType, BuiltinFunctionType, partial):
                    raise ValueError(
                        "Opaque scheduler partial functions are unsupported"
                    )
                plain(value.func)
                plain(value.args)
                plain(value.keywords)
                plain(vars(value))
                callable_identity(value)
            elif type(value) in (tuple, list):
                for child in value:
                    plain(child)
            elif type(value) is dict:
                for key, child in value.items():
                    if type(key) not in (str, int):
                        raise ValueError(
                            "Opaque scheduler mapping keys are unsupported"
                        )
                    plain(child)
            else:
                raise ValueError(
                    "Opaque checkpoint scheduler copy bindings are unsupported"
                )
        finally:
            active.remove(id(value))

    def visit(node: Any) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        cls = type(node)
        if (
            cls.__module__ != "torch.optim.lr_scheduler"
            or getattr(torch.optim.lr_scheduler, cls.__name__, None) is not cls
        ):
            raise ValueError("Custom composite scheduler children are unsupported")
        if id(node.optimizer) not in optimizers:
            raise ValueError(
                "Checkpoint scheduler references an unregistered optimizer"
            )
        for name, value in vars(node).items():
            if name in copy_protocols:
                raise ValueError("Custom scheduler copy protocols are unsupported")
            if name == "optimizer":
                continue
            if name == "_schedulers":
                if type(value) not in (list, tuple):
                    raise ValueError("Invalid composite scheduler children")
                for child in value:
                    visit(child)
            else:
                plain(value)

    visit(scheduler)


def _validate_optimizer_groups(optimizer: Any, saved: Any, torch: Any) -> None:
    """Validate builtin optimizer controls before load_state_dict can install them.

    Slot tensors and parameter-ID coverage are validated separately by
    ``_optimizer_stage``. This helper performs no optimizer/scheduler updates or
    user-defined methods. Tensor hyperparameters must be materialized, dense,
    real, finite and single-element; builtin optimizer loading owns its copies.
    """
    cls = type(optimizer)
    if cls not in {torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW}:
        raise ValueError("Custom checkpoint optimizer types are unsupported")
    if (
        type(saved) is not dict
        or set(saved) != {"state", "param_groups"}
        or type(saved["param_groups"]) is not list
        or len(saved["param_groups"]) != len(optimizer.param_groups)
    ):
        raise ValueError("Invalid checkpoint optimizer group schema")

    def scalar(value: Any, name: str, *, tensor: bool = False) -> float:
        if tensor and type(value) in {torch.Tensor, torch.nn.Parameter}:
            if (
                value.layout != torch.strided
                or value.is_quantized
                or value.device.type == "meta"
                or value.numel() != 1
                or value.dtype == torch.bool
                or value.is_complex()
            ):
                raise ValueError(f"Invalid checkpoint optimizer group {name}")
            number = value.detach().item()
        elif type(value) in {int, float}:
            number = value
        else:
            raise ValueError(f"Invalid checkpoint optimizer group {name}")
        try:
            number = float(number)
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid checkpoint optimizer group {name}") from exc
        if not math.isfinite(number):
            raise ValueError(f"Invalid checkpoint optimizer group {name}")
        return number

    for current, group in zip(
        optimizer.param_groups, saved["param_groups"], strict=True
    ):
        if type(group) is not dict or set(group) != set(current):
            raise ValueError("Checkpoint optimizer group schema differs")
        required = {"lr", "weight_decay", "maximize", "differentiable", "foreach"}
        required |= (
            {"momentum", "dampening", "nesterov"}
            if cls is torch.optim.SGD
            else {"betas", "eps", "amsgrad", "capturable"}
        )
        if required - set(group):
            raise ValueError("Checkpoint optimizer group controls are missing")
        for name in (
            "maximize",
            "differentiable",
            "capturable",
            "amsgrad",
            "nesterov",
            "decoupled_weight_decay",
        ):
            if name in group and type(group[name]) is not bool:
                raise ValueError(f"Invalid checkpoint optimizer group {name}")
        for name in ("foreach", "fused"):
            if (
                name in group
                and group[name] is not None
                and type(group[name]) is not bool
            ):
                raise ValueError(f"Invalid checkpoint optimizer group {name}")
        for name in ("lr", "initial_lr", "weight_decay"):
            if (
                name in group
                and scalar(group[name], name, tensor=name != "weight_decay") < 0
            ):
                raise ValueError(f"Invalid checkpoint optimizer group {name}")
        if group.get("fused") is True and (
            group.get("foreach") is True or group["differentiable"]
        ):
            raise ValueError("Incompatible checkpoint optimizer group fused flags")
        if cls is torch.optim.SGD:
            momentum = scalar(group["momentum"], "momentum")
            dampening = scalar(group["dampening"], "dampening")
            if momentum < 0 or dampening < 0:
                raise ValueError(
                    "Invalid checkpoint optimizer group momentum/dampening"
                )
            if group["nesterov"] and (momentum <= 0 or dampening != 0):
                raise ValueError("Invalid checkpoint optimizer group Nesterov controls")
        else:
            if scalar(group["eps"], "eps") < 0:
                raise ValueError("Invalid checkpoint optimizer group eps")
            betas = group["betas"]
            if type(betas) not in {tuple, list} or len(betas) != 2:
                raise ValueError("Invalid checkpoint optimizer group betas")
            if any(not 0 <= scalar(beta, "betas") < 1 for beta in betas):
                raise ValueError("Invalid checkpoint optimizer group betas")
            if (
                group["foreach"] is True
                and not group["capturable"]
                and any(
                    type(value) in {torch.Tensor, torch.nn.Parameter}
                    for value in (group["lr"], *betas)
                )
            ):
                raise ValueError(
                    "Incompatible checkpoint optimizer group tensor/foreach controls"
                )
            if (
                cls is torch.optim.AdamW
                and group.get("decoupled_weight_decay", True) is not True
            ):
                raise ValueError("Invalid checkpoint optimizer group AdamW decay mode")


def _prepare_components(
    components: dict[str, Any], saved: Any, torch: Any
) -> dict[str, Any]:
    """Stage supported optimizer/scheduler/scaler state before live mutation."""
    from .checkpoint_state import _validate_scaler
    from .ddp_checkpoint_state import _optimizer_stage
    from .state_fingerprint import state_fingerprint

    if not isinstance(saved, dict) or set(saved) != set(components):
        raise ValueError("Checkpoint stateful components differ")
    memo: dict[int, Any] = {}
    prepared: dict[str, Any] = {}
    for name, component in components.items():
        if isinstance(component, torch.optim.Optimizer):
            if type(component) not in {
                torch.optim.SGD,
                torch.optim.Adam,
                torch.optim.AdamW,
            }:
                raise ValueError("Custom checkpoint optimizer types are unsupported")
            _validate_optimizer_groups(component, saved[name], torch)
            prepared[name] = _optimizer_stage(component, saved[name], torch)
            memo[id(component)] = prepared[name]
    for name, component in components.items():
        value = saved[name]
        if isinstance(component, torch.optim.Optimizer):
            continue
        if isinstance(component, torch.nn.Module):
            _module_state_schema(component, value, torch)
            # A scheduler must never deepcopy a reference module through an
            # incidental binding. Supported built-in schedulers only use their
            # optimizer and plain/callable configuration fields.
            memo[id(component)] = component
            continue
        cls = type(component)
        scheduler = (
            cls.__module__ == "torch.optim.lr_scheduler"
            and getattr(torch.optim.lr_scheduler, cls.__name__, None) is cls
        )
        scaler_types = {
            getattr(torch.amp, "GradScaler", None),
            torch.cuda.amp.GradScaler,
        }
        if not scheduler and cls not in scaler_types:
            raise ValueError("Custom checkpoint scheduler/scaler types are unsupported")
        if scheduler:
            _scheduler_copy_graph(
                component,
                {
                    id(item)
                    for item in components.values()
                    if isinstance(item, torch.optim.Optimizer)
                },
                torch,
            )
        if not isinstance(value, dict) or set(value) != set(component.state_dict()):
            raise ValueError("Checkpoint scheduler/scaler state schema differs")
        if scheduler:
            _scheduler_schema(
                value,
                component.state_dict(),
                torch,
                allow_sentinels=cls is torch.optim.lr_scheduler.ReduceLROnPlateau,
            )
            group_count = len(component.optimizer.param_groups)
            for field in ("base_lrs", "_last_lr"):
                if field in value and len(value[field]) != group_count:
                    raise ValueError(
                        "Checkpoint scheduler learning-rate group count differs"
                    )
            fingerprint_state = dict(value)
            # ReduceLROnPlateau legitimately starts at an infinite comparison
            # sentinel, analogous to the trainer's best_eval_metric.
            if cls is torch.optim.lr_scheduler.ReduceLROnPlateau:
                for key in ("best", "mode_worse"):
                    if isinstance(fingerprint_state.get(key), float) and math.isinf(
                        fingerprint_state[key]
                    ):
                        fingerprint_state[key] = str(fingerprint_state[key])
            state_fingerprint(fingerprint_state)
        else:
            _validate_scaler(value)
        staged = copy.deepcopy(component, memo)
        cls.load_state_dict(staged, value)
        memo[id(component)] = staged
        prepared[name] = staged
    return prepared


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
        _require_runtime_ready(self)
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
                    tokens += _observed_assistant_tokens(trajectory)
            self._runtime_rollout_metrics = {
                "rollout_seconds": seconds,
                "generated_tokens": tokens,
            }
            if outcomes_known:
                self._runtime_rollout_metrics["successful_episodes"] = successes
        elif isinstance(result, dict):
            self._record_training_metrics(result, seconds)
        return result

    return measured


class TrainingRuntime:
    """Common configuration and resumable lifecycle for native algorithms.

    Sampling and objective computation remain with algorithms. Named stateful
    components make actor/critic and policy-only trainers use the same protocol.
    """

    config: Any
    kl_controller: Any

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Measure native phases consistently without changing algorithm signatures."""
        super().__init_subclass__(**kwargs)
        profiled = ("train_step", "training_step", "generate_trajectories")
        for name in profiled:
            method = cls.__dict__.get(name)
            if method is not None and inspect.iscoroutinefunction(method):
                setattr(cls, name, _profile_phase(method, name))
        for name in (
            "train_step_token_level",
            "train",
            "generate_group_responses",
            "save_checkpoint",
            "save_model",
            "_apply_optimizer_step",
        ):
            method = cls.__dict__.get(name)
            if method is not None:
                setattr(cls, name, _guard_runtime_method(method))

    def _record_training_metrics(self, result: dict[str, Any], seconds: float) -> None:
        """Record local efficiency and observe a nonempty learner population."""
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
        self._runtime_rollout_metrics: dict[str, Any] = {}
        if result.get("global_empty_batch") is True:
            # A collective no-op is not an observed policy population.
            # Retain its timing/cost, without advancing or clearing a real
            # zero-signal streak, or creating a monitor solely for no-ops.
            return
        from .diagnostics import DiagnosticsMonitor

        monitor = getattr(self, "diagnostics", None)
        if monitor is None:
            monitor = self.diagnostics = DiagnosticsMonitor(self.config)
        observation = getattr(self, "_diagnostic_observations", 0) + 1
        self._diagnostic_observations = observation
        monitor("step_end", {**result, "step": observation})

    def save_runtime_checkpoint(self, output_dir: str) -> None:
        """Atomically publish model, components, pending gradients and RNGs."""
        import torch

        from .checkpoint_state import _validate_gradients
        from .ddp_checkpoint_state import _validate_rng
        from .state_fingerprint import state_fingerprint

        _require_runtime_ready(self)
        model = _model(self)
        state_fingerprint(dict(model.named_parameters()))
        parameters = _gradient_parameters(self)
        live_components = {
            name: getattr(self, name)
            for name in _COMPONENTS
            if getattr(self, name, None) is not None
        }
        optimizer_ids = {
            id(component)
            for component in live_components.values()
            if isinstance(component, torch.optim.Optimizer)
        }
        for component in live_components.values():
            if isinstance(
                component,
                (
                    torch.optim.lr_scheduler.LRScheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ),
            ):
                _scheduler_copy_graph(component, optimizer_ids, torch)
        components = {
            name: component.state_dict() for name, component in live_components.items()
        }
        _prepare_components(live_components, components, torch)
        state = {
            "runtime_schema": 1,
            "trainer": type(self).__name__,
            "config": json.loads(json.dumps(self.config.to_dict(), allow_nan=False)),
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
            "learning_state": capture_learning_state(self),
            "runtime_component_identities": _component_identities(
                live_components, torch
            ),
            "adaptive_kl_controller": _adaptive_kl_state(self),
        }
        _validate_counters(state["counters"], self)
        state["counters"] = _plain_counter_snapshot(state["counters"])
        _validate_gradients(state["gradients"], parameters, torch, allow_empty=True)
        _validate_rng(state["rng"], torch)
        _validate_rollout_state(state["rollout_state"])
        if getattr(self, "_grpo_accumulation", None) is not None:
            from .checkpoint_state import _capture_accumulation

            state["grpo_normalization"] = _capture_accumulation(self, torch)[
                "normalization"
            ]
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
                    json.dumps(
                        state["counters"]["training_metrics"], indent=2, allow_nan=False
                    )
                    + "\n"
                )

    def load_runtime_checkpoint(
        self, checkpoint_dir: str, *, trusted: bool = False
    ) -> bool:
        """Preflight learner state, then restore or quarantine a failed commit.

        Component/cursor/RNG checks precede policy mutation. A later model shard,
        load hook or device failure can still leave a partial commit; training
        and publication stay blocked until a complete successful runtime reload.
        This protocol does not claim to roll back failed commits.
        """
        import torch

        from .checkpoint_io import load_checkpoint_file
        from .checkpoint_state import _validate_gradients
        from .ddp_checkpoint_state import _owned_optimizer_state, _validate_rng
        from .state_fingerprint import state_fingerprint

        path = Path(checkpoint_dir)
        with checkpoint_read(path):
            validate_checkpoint_publication(path, required=True)
            state = load_checkpoint_file(
                path / "training_state.pt", trusted=trusted, torch_module=torch
            )
            if (
                not isinstance(state, dict)
                or type(state.get("runtime_schema")) is not int
                or state.get("runtime_schema") != 1
                or state.get("trainer") != type(self).__name__
            ):
                raise ValueError("Checkpoint runtime schema or trainer differs")
            if not isinstance(state.get("config"), dict):
                raise ValueError("Invalid checkpoint configuration schema")
            _validate_counters(state.get("counters"), self)
            counters = _plain_counter_snapshot(state["counters"])
            _validate_rng(state.get("rng"), torch)
            _validate_rollout_state(state.get("rollout_state"))
            adaptive_kl = _prepare_adaptive_kl(
                self, state.get("adaptive_kl_controller")
            )
            for name in (
                "model_name",
                "model_revision",
                "objective",
                "objective_overrides",
                "gradient_accumulation_steps",
                "max_completion_length",
                "num_generations",
                "beta",
                "entropy_coef",
                "clip_ratio",
                "seq_clip_ratio",
                "reward_clip",
                "baseline_type",
                "advantage_normalization",
                "max_prompt_length",
                "max_grad_norm",
                "gamma",
                "gae_lambda",
                "value_clip_eps",
                "value_coef",
                "clip_eps",
                "num_ppo_epochs",
                "mini_batch_size",
                "ppo_prompt_sha256",
                "target_kl",
                "use_adaptive_kl",
                "num_gradient_updates",
                "clip_range_left",
                "clip_range_right",
                "rescore_old_log_probs",
                "clip_eps_low",
                "clip_eps_high",
                "use_token_level_loss",
                "use_positive_lm_loss",
                "positive_lm_weight",
                "value_loss_coef",
                "lambda_critic",
                "lambda_policy_alpha",
                "value_warmup_steps",
                "group_size",
                "use_dynamic_sampling",
                "min_accuracy_threshold",
                "max_accuracy_threshold",
                "use_overlong_shaping",
                "max_generation_length",
                "overlong_cache_length",
                "overlong_penalty",
                "logprob_dtype",
                "temperature",
                "top_p",
                "top_k",
                "use_reference_model",
            ):
                if state["config"].get(name) != self.config.to_dict().get(name):
                    raise ValueError(f"Checkpoint configuration differs: {name}")
            components = {
                name: getattr(self, name)
                for name in _COMPONENTS
                if getattr(self, name, None) is not None
            }
            prepared_components = _prepare_components(
                components, state.get("components"), torch
            )
            if "runtime_component_identities" in state and (
                state["runtime_component_identities"]
                != _component_identities(components, torch)
            ):
                raise ValueError("Checkpoint component implementations/bindings differ")
            model = _model(self)
            parameters = _gradient_parameters(self)
            _validate_gradients(
                state.get("gradients"), parameters, torch, allow_empty=True
            )
            normalization = None
            if "grpo_normalization" in state:
                from .checkpoint_state import (
                    _prepare_normalization,
                    _validate_gradients,
                )

                steps = self.config.gradient_accumulation_steps
                count = state["counters"].get("_grad_accum_step")
                if (
                    type(count) is not int
                    or count < 0
                    or type(steps) is not int
                    or steps < 1
                ):
                    raise ValueError("Invalid checkpoint gradient accumulation counter")
                _validate_gradients(state["gradients"], parameters, torch)
                normalization = _prepare_normalization(
                    state["grpo_normalization"], parameters, torch, count % steps, steps
                )
            learning_input = state
            if "learning_state" not in state and "_current_task_id" in counters:
                task = counters["_current_task_id"]
                if "current_task_id" in state and state["current_task_id"] != task:
                    raise ValueError("Checkpoint task identifiers disagree")
                learning_input = {**state, "current_task_id": task}
            learning = prepare_learning_state(self, learning_input)
            if (
                "_current_task_id" in counters
                and counters["_current_task_id"] != learning["current_task_id"]
            ):
                raise ValueError("Checkpoint task identifiers disagree")
            from .checkpoint_state import _prepare_episode_cursor

            cursor = (
                _prepare_episode_cursor(state["counters"])
                if hasattr(self, "_next_episode")
                else None
            )
            # From this point forward, failure can have changed model shards,
            # components or RNGs. Only a complete successful commit clears this
            # flag; trusted deserialization does not bypass it.
            self._runtime_checkpoint_poisoned = True
            if not load_model_checkpoint(
                model, path, torch_module=torch, trusted=trusted, strict=True
            ):
                raise ValueError("Checkpoint has no loadable model")
            for name, component in components.items():
                if isinstance(component, torch.optim.Optimizer):
                    value = _owned_optimizer_state(
                        component, prepared_components[name], torch
                    )
                elif isinstance(component, torch.nn.Module):
                    value = state["components"][name]
                else:
                    value = copy.deepcopy(prepared_components[name].state_dict())
                component.load_state_dict(value)
            state_fingerprint(dict(model.named_parameters()))
            if adaptive_kl is not None:
                self.kl_controller.kl_coef = adaptive_kl["kl_coef"]
            for name, value in counters.items():
                setattr(self, name, value)
            if cursor is not None:
                self._next_episode = cursor
            for name, parameter in parameters.items():
                value = state["gradients"][name]
                parameter.grad = (
                    None if value is None else value.to(parameter.device, copy=True)
                )
            if hasattr(self, "_grpo_accumulation") or normalization is not None:
                self._grpo_accumulation = normalization
            apply_learning_state(self, learning)
            scaler = getattr(self, "scaler", None)
            if scaler is not None and state["components"].get("scaler"):
                pending_gradient = next(
                    (
                        parameter.grad
                        for parameter in parameters.values()
                        if parameter.grad is not None
                    ),
                    None,
                )
                if pending_gradient is not None:
                    # Initialize GradScaler's lazy device tensors before a
                    # resumed partial window can flush without another backward.
                    scaler.scale(torch.zeros((), device=pending_gradient.device))
            agent = getattr(self, "agent", None)
            restore_rollout_state(agent, state.get("rollout_state"))
            sync = getattr(agent, "sync_rollout_backend", None)
            if callable(sync):
                if sync() is False:
                    raise RuntimeError("Checkpoint rollout publication failed")
            # Publication may consume global RNG. Restore the learner stream
            # last; any publication/RNG failure still retains quarantine.
            restore_rng(state["rng"], torch)
        self._runtime_checkpoint_poisoned = False
        return True
