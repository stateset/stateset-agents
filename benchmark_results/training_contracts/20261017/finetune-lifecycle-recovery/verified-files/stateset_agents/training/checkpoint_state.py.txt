"""Shared model, optimizer, gradient accumulation, and AMP checkpoint state."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from .checkpoint_io import load_checkpoint_file
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
from .runtime_state import (
    _component_identities,
    _plain_counter_snapshot,
    _prepare_components,
    _require_runtime_ready,
    _validate_counters,
    _validate_rollout_state,
    capture_rng,
    capture_rollout_state,
    restore_rng,
    restore_rollout_state,
)

_NATIVE_COUNTER_KEYS = {
    "global_step": "global_step",
    "current_epoch": "current_epoch",
    "best_eval_metric": "best_eval_metric",
    "steps_without_improvement": "steps_without_improvement",
    "grad_accum_step": "_grad_accum_step",
    "_next_episode": "_next_episode",
}
_NATIVE_REQUIRED_COUNTERS = set(_NATIVE_COUNTER_KEYS) - {"_next_episode"}


def _native_components(trainer: Any) -> dict[str, Any]:
    return {
        name: getattr(trainer, name)
        for name in ("optimizer", "lr_scheduler", "scaler")
        if getattr(trainer, name, None) is not None
    }


def _native_policy_finite(trainer: Any) -> None:
    """Validate materialized policy parameters with bounded temporary copies."""
    from .state_fingerprint import state_fingerprint

    named = getattr(trainer.agent.model, "named_parameters", None)
    if not callable(named):
        raise ValueError(
            "Native learner checkpoints require verifiable policy parameters"
        )
    state_fingerprint(dict(named()))


def _sync_restored_policy(trainer: Any) -> None:
    """Publish restored weights once, respecting explicit asynchronous opt-out."""
    trainer.last_rollout_sync = None
    if not bool(getattr(getattr(trainer, "config", None), "rollout_sync", True)):
        return
    sync = getattr(trainer.agent, "sync_rollout_backend", None)
    if callable(sync):
        outcome = sync()
        trainer.last_rollout_sync = None if outcome is None else bool(outcome)
        if outcome is False:
            raise RuntimeError("Native checkpoint rollout publication failed")


def _native_counters(state: dict[str, Any]) -> dict[str, Any]:
    counters = {
        target: state[source]
        for source, target in _NATIVE_COUNTER_KEYS.items()
        if source in state
    }
    _validate_counters(counters)
    return _plain_counter_snapshot(counters)


def _accumulation_steps(trainer: Any) -> int:
    getter = getattr(trainer, "_get_grad_accum_steps", None)
    if callable(getter):
        return int(getter())
    return max(1, int(getattr(trainer.config, "gradient_accumulation_steps", 1) or 1))


def _prepare_episode_cursor(state: dict[str, Any]) -> int:
    """Validate explicit next-episode position, preserving legacy resume rules."""
    if "_next_episode" in state:
        cursor = state["_next_episode"]
    else:
        epoch = state.get("current_epoch", 0)
        if type(epoch) is not int or epoch < 0:
            raise ValueError("Invalid checkpoint current episode")
        cursor = epoch + 1
    if type(cursor) is not int or not 0 <= cursor <= 2**63 - 1:
        raise ValueError("Invalid checkpoint next episode cursor")
    return cursor


def _parameters(trainer: Any) -> dict[str, Any]:
    model = trainer.agent.model
    named = getattr(model, "named_parameters", None)
    return dict(named()) if callable(named) else {}


def _validate_scaler(state: Any) -> None:
    if state is None or state == {}:
        return
    if not isinstance(state, dict) or set(state) != {
        "scale",
        "growth_factor",
        "backoff_factor",
        "growth_interval",
        "_growth_tracker",
    }:
        raise ValueError("Invalid checkpoint AMP scaler state")
    for name in ("scale", "growth_factor", "backoff_factor"):
        value = state[name]
        try:
            valid = type(value) in (int, float) and math.isfinite(value) and value > 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("Invalid checkpoint AMP scaler state")
    if state["growth_factor"] <= 1 or state["backoff_factor"] >= 1:
        raise ValueError("Invalid checkpoint AMP scaler state")
    if (
        type(state["growth_interval"]) is not int
        or state["growth_interval"] < 1
        or type(state["_growth_tracker"]) is not int
        or not 0 <= state["_growth_tracker"] < state["growth_interval"]
    ):
        raise ValueError("Invalid checkpoint AMP scaler state")


def _validate_gradients(
    gradients: Any, parameters: dict[str, Any], torch: Any, *, allow_empty: bool = False
) -> None:
    if not isinstance(gradients, dict) or set(gradients) != set(parameters):
        raise ValueError("Checkpoint gradients differ from model parameters")
    if not allow_empty and not any(value is not None for value in gradients.values()):
        raise ValueError("Pending accumulation window has no saved gradients")
    for name, gradient in gradients.items():
        if gradient is None:
            continue
        parameter = parameters[name]
        if (
            not torch.is_tensor(gradient)
            or gradient.layout != torch.strided
            or gradient.shape != parameter.shape
            or gradient.dtype != parameter.dtype
            or not bool(torch.isfinite(gradient).all())
        ):
            raise ValueError(f"Invalid checkpoint gradient for {name}")


def _capture_accumulation(trainer: Any, torch: Any) -> dict[str, Any]:
    steps = _accumulation_steps(trainer)
    count = trainer._grad_accum_step
    if type(count) is not int or count < 0:
        raise ValueError("Invalid gradient accumulation counter")
    gradients = {}
    if count % steps:
        if trainer.optimizer is None:
            raise ValueError("Pending accumulation requires an optimizer")
        parameters = _parameters(trainer)
        gradients = {name: parameter.grad for name, parameter in parameters.items()}
        _validate_gradients(gradients, parameters, torch)
        gradients = {
            name: None if value is None else value.detach().cpu().clone()
            for name, value in gradients.items()
        }
    scaler = getattr(trainer, "scaler", None)
    scaler_state = scaler.state_dict() if scaler is not None else None
    _validate_scaler(scaler_state)
    result: dict[str, Any] = {
        "schema_version": 1,
        "steps": steps,
        "gradients": gradients,
        "scaler": scaler_state,
    }
    normalization = getattr(trainer, "_grpo_accumulation", None)
    if normalization is not None:
        result["schema_version"] = 2
        result["normalization"] = _prepare_normalization(
            normalization, _parameters(trainer), torch, count % steps, steps
        )
        result["normalization"]["entropy_gradients"] = {
            name: None if value is None else value.detach().cpu().clone()
            for name, value in result["normalization"]["entropy_gradients"].items()
        }
    return result


def _prepare_normalization(
    saved: Any,
    parameters: dict[str, Any],
    torch: Any,
    pending: int,
    steps: int,
    *,
    copy_tensors: bool = True,
    allow_empty: bool = False,
) -> dict[str, Any]:
    """Validate and allocate both normalization banks before policy mutation."""
    if not isinstance(saved, dict) or set(saved) != {
        "measure",
        "policy_count",
        "policy_normalizer",
        "entropy_count",
        "entropy_normalizer",
        "entropy_gradients",
    }:
        raise ValueError("Invalid checkpoint GRPO normalization schema")
    if (
        not pending
        or not isinstance(saved["measure"], str)
        or saved["measure"] not in {"trajectory", "token"}
    ):
        raise ValueError("Invalid checkpoint GRPO normalization window")
    for name in ("policy_count", "policy_normalizer"):
        minimum = 0 if allow_empty and name == "policy_count" else pending
        if type(saved[name]) is not int or not minimum <= saved[name] <= 2**63 - 1:
            raise ValueError("Invalid checkpoint GRPO policy normalization")
    if saved["policy_normalizer"] % steps:
        raise ValueError("Invalid checkpoint GRPO policy normalizer scale")
    for name in ("entropy_count", "entropy_normalizer"):
        if type(saved[name]) is not int or not 0 <= saved[name] <= 2**63 - 1:
            raise ValueError("Invalid checkpoint GRPO entropy normalization")
    auxiliary = saved["entropy_gradients"]
    if saved["policy_count"] == 0 and saved["entropy_count"] != 0:
        raise ValueError("Empty policy contributors cannot have entropy samples")
    if saved["entropy_normalizer"]:
        if (
            saved["measure"] != "trajectory"
            or saved["entropy_normalizer"] % steps
            or saved["entropy_count"] < saved["policy_count"]
        ):
            raise ValueError("Invalid checkpoint GRPO entropy normalizer scale")
        _validate_gradients(
            auxiliary,
            parameters,
            torch,
            allow_empty=allow_empty and saved["entropy_count"] == 0,
        )
    elif saved["entropy_count"] or not isinstance(auxiliary, dict) or auxiliary:
        raise ValueError("Unexpected checkpoint GRPO entropy gradients")
    return {
        **saved,
        "entropy_gradients": {
            name: (
                None
                if value is None
                else (
                    value.detach().to(device=parameters[name].device).clone()
                    if copy_tensors
                    else value.detach()
                )
            )
            for name, value in auxiliary.items()
        },
    }


def _prepare_accumulation(
    trainer: Any, state: dict[str, Any], torch: Any, *, copy_tensors: bool = True
) -> tuple[dict[str, Any], Any, dict[str, Any] | None]:
    """Validate and allocate gradients before mutating live training state."""
    steps = _accumulation_steps(trainer)
    count = state.get("grad_accum_step", 0)
    if type(count) is not int or count < 0:
        raise ValueError("Invalid checkpoint gradient accumulation counter")
    saved = state.get("gradient_accumulation")
    scaler = getattr(trainer, "scaler", None)
    parameters = _parameters(trainer)
    if saved is None:
        config = state.get("config")
        saved_steps = (
            config.get("gradient_accumulation_steps", steps)
            if isinstance(config, dict)
            else steps
        )
        if type(saved_steps) is not int or saved_steps < 1 or count % saved_steps:
            raise ValueError(
                "Legacy checkpoint has a pending accumulation window without gradients"
            )
        if scaler is not None:
            raise ValueError("Legacy checkpoint has no AMP scaler state")
        if count % steps:
            raise ValueError("Checkpoint accumulation schedule differs from trainer")
        return dict.fromkeys(parameters), None, None
    if (
        not isinstance(saved, dict)
        or type(saved.get("schema_version")) is not int
        or saved["schema_version"] not in (1, 2)
        or type(saved.get("steps")) is not int
        or saved["steps"] != steps
        or set(saved)
        != (
            {"schema_version", "steps", "gradients", "scaler"}
            | ({"normalization"} if saved.get("schema_version") == 2 else set())
        )
    ):
        raise ValueError(
            "Checkpoint accumulation schedule or schema differs from trainer"
        )
    scaler_state = saved["scaler"]
    _validate_scaler(scaler_state)
    if (scaler is None) != (scaler_state is None):
        raise ValueError("Checkpoint AMP scaler differs from trainer")
    if scaler is not None and bool(scaler.state_dict()) != bool(scaler_state):
        raise ValueError("Checkpoint AMP scaler enablement differs from trainer")
    gradients = saved["gradients"]
    from .ddp_population import native_empty_contributor

    allow_empty = native_empty_contributor(trainer)
    normalization = (
        _prepare_normalization(
            saved["normalization"],
            parameters,
            torch,
            count % steps,
            steps,
            copy_tensors=copy_tensors,
            allow_empty=allow_empty,
        )
        if saved["schema_version"] == 2
        else None
    )
    if count % steps:
        _validate_gradients(
            gradients,
            parameters,
            torch,
            allow_empty=allow_empty
            and normalization is not None
            and normalization["policy_count"] == 0,
        )
        return (
            {
                name: (
                    None
                    if value is None
                    else (
                        value.detach().to(device=parameters[name].device).clone()
                        if copy_tensors
                        else value.detach()
                    )
                )
                for name, value in gradients.items()
            },
            scaler_state,
            normalization,
        )
    if not isinstance(gradients, dict) or gradients:
        raise ValueError("Completed accumulation window cannot contain gradients")
    return dict.fromkeys(parameters), scaler_state, None


def save_training_checkpoint(
    trainer: Any, checkpoint_path: Path, *, require_torch_fn: Any
) -> None:
    """Persist a resumable checkpoint; propagate every artifact write failure."""
    from .ddp_checkpoint_state import _validate_rng

    _require_runtime_ready(trainer)
    torch = require_torch_fn()
    if getattr(trainer.agent, "model", None) is None:
        raise ValueError("Cannot save a training checkpoint without a model")
    _native_policy_finite(trainer)
    components = _native_components(trainer)
    _prepare_components(
        components,
        {name: component.state_dict() for name, component in components.items()},
        torch,
    )
    _validate_rng(capture_rng(torch), torch)
    _native_counters(
        {
            source: getattr(trainer, target)
            for source, target in _NATIVE_COUNTER_KEYS.items()
            if hasattr(trainer, target)
        }
    )
    accumulation = _capture_accumulation(trainer, torch)
    with checkpoint_write(checkpoint_path) as staging:
        _write_training_checkpoint(trainer, staging, torch, accumulation)


def _write_training_checkpoint(
    trainer: Any, checkpoint_path: Path, torch: Any, accumulation: dict[str, Any]
) -> None:
    trainer.agent.model.save_pretrained(checkpoint_path)
    if getattr(trainer.agent, "tokenizer", None) is not None:
        trainer.agent.tokenizer.save_pretrained(checkpoint_path)

    training_state = {
        "global_step": trainer.global_step,
        "current_epoch": trainer.current_epoch,
        "best_eval_metric": trainer.best_eval_metric,
        "steps_without_improvement": trainer.steps_without_improvement,
        "grad_accum_step": trainer._grad_accum_step,
        "gradient_accumulation": accumulation,
        "checkpoint_publication": 1,
        "native_checkpoint_schema": 1,
        "rng": capture_rng(torch),
        "rollout_state": capture_rollout_state(trainer.agent),
        "learning_state": capture_learning_state(trainer),
        "native_component_identities": _component_identities(
            _native_components(trainer), torch
        ),
    }
    if hasattr(trainer, "_next_episode"):
        training_state["_next_episode"] = _prepare_episode_cursor(
            {"_next_episode": trainer._next_episode}
        )
    if trainer.optimizer is not None:
        training_state["optimizer_state_dict"] = trainer.optimizer.state_dict()

    if hasattr(trainer.config, "__dict__"):
        training_state["config"] = json.loads(
            json.dumps(trainer.config.__dict__, allow_nan=False)
        )
        (checkpoint_path / "resolved_config.json").write_text(
            json.dumps(
                training_state["config"], sort_keys=True, indent=2, allow_nan=False
            )
            + "\n"
        )
    else:
        training_state["config"] = {}

    if trainer.lr_scheduler is not None:
        training_state["scheduler_state_dict"] = trainer.lr_scheduler.state_dict()
    if trainer.continual_manager is not None:
        training_state["continual_state"] = training_state["learning_state"][
            "continual_state"
        ]
        training_state["current_task_id"] = trainer._current_task_id

    owned = _native_counters(training_state)
    for source, target in _NATIVE_COUNTER_KEYS.items():
        if target in owned:
            training_state[source] = owned[target]

    torch.save(training_state, checkpoint_path / "training_state.pt")


def load_training_checkpoint(
    trainer: Any,
    checkpoint_path: Any,
    *,
    require_torch_fn: Any,
    trainer_exceptions: tuple[type[BaseException], ...],
    logger: Any,
    trusted: bool = False,
) -> bool:
    """Load model and training state from a checkpoint directory.

    Args:
        trusted: When ``False`` (the default) checkpoints are unpickled with
            ``weights_only=True``, so only tensors and plain data are restored
            and a malicious checkpoint cannot execute code.  Pass ``True`` only
            for checkpoints from a source you control.
    """
    path = Path(checkpoint_path).resolve()
    with checkpoint_read(path):
        loaded = _load_training_checkpoint(
            trainer,
            path,
            require_torch_fn=require_torch_fn,
            trainer_exceptions=trainer_exceptions,
            logger=logger,
            trusted=trusted,
        )
    if loaded is True:
        trainer._runtime_checkpoint_poisoned = False
    return loaded


def _load_training_checkpoint(
    trainer: Any,
    path: Path,
    *,
    require_torch_fn: Any,
    trainer_exceptions: tuple[type[BaseException], ...],
    logger: Any,
    trusted: bool,
) -> bool:
    from .ddp_checkpoint_state import _owned_optimizer_state, _validate_rng

    try:
        torch = require_torch_fn()
    except ImportError:
        logger.warning("Cannot load checkpoint without PyTorch.")
        return False
    if not path.exists():
        logger.warning("Checkpoint path not found: %s", path)
        return False

    published = validate_checkpoint_publication(path)
    model_dir = path / "model" if (path / "model").is_dir() else path
    state_path = path / "training_state.pt"
    state = None
    gradients: dict[str, Any] = {}
    scaler_state = None
    normalization = None
    counters: dict[str, Any] = {}
    components: dict[str, Any] = {}
    prepared: dict[str, Any] = {}
    learning = None
    resumable = False
    if state_path.exists():
        try:
            state = load_checkpoint_file(
                state_path, trusted=trusted, torch_module=torch
            )
        except trainer_exceptions as exc:
            logger.warning("Failed to load training state: %s", exc)
            return False
        if not isinstance(state, dict):
            logger.warning("Unexpected training state format in %s", state_path)
            return False
        if "checkpoint_publication" in state and (
            type(state["checkpoint_publication"]) is not int
            or state["checkpoint_publication"] != 1
            or not published
        ):
            raise ValueError("Checkpoint publication is incomplete or unsupported")
        current_schema = "native_checkpoint_schema" in state
        if current_schema and (
            type(state["native_checkpoint_schema"]) is not int
            or state["native_checkpoint_schema"] != 1
            or not published
            or state.get("checkpoint_publication") != 1
        ):
            raise ValueError("Native checkpoint schema/publication is unsupported")
        counters = _native_counters(state)
        resumable = _NATIVE_REQUIRED_COUNTERS <= set(state)
        if current_schema and (
            not resumable
            or not {
                "gradient_accumulation",
                "rng",
                "config",
                "learning_state",
                "rollout_state",
                "native_component_identities",
            }
            <= set(state)
        ):
            raise ValueError("Published native checkpoint lacks complete learner state")
        gradients, scaler_state, normalization = _prepare_accumulation(
            trainer, state, torch
        )
        if hasattr(trainer, "_next_episode"):
            counters["_next_episode"] = _prepare_episode_cursor(state)
        if "rng" in state:
            _validate_rng(state["rng"], torch)
        _validate_rollout_state(state.get("rollout_state"))
        if "rng" in state:
            saved_config = state.get("config")
            if not isinstance(saved_config, dict):
                raise ValueError("Checkpoint has no resolved configuration")
            for name in (
                "model_name",
                "model_revision",
                "objective",
                "objective_overrides",
                "gradient_accumulation_steps",
                "max_completion_length",
                "max_prompt_length",
                "entropy_coef",
                "clip_ratio",
                "seq_clip_ratio",
                "reward_clip",
                "baseline_type",
                "advantage_normalization",
                "beta",
                "max_grad_norm",
                "num_gradient_updates",
                "old_logprobs_source",
                "allow_sequence_fallback",
                "allow_missing_sampler_logprobs",
                "rollout_sync",
                "num_generations",
            ):
                if saved_config.get(name) != getattr(trainer.config, name, None):
                    raise ValueError(f"Checkpoint configuration differs: {name}")
        components = _native_components(trainer)
        saved_components = {}
        for name, key in (
            ("optimizer", "optimizer_state_dict"),
            ("lr_scheduler", "scheduler_state_dict"),
        ):
            if resumable and ((name not in components) != (key not in state)):
                raise ValueError(f"Checkpoint {key} availability differs from trainer")
            if name in components and key in state:
                saved_components[name] = state[key]
        if "scaler" in components:
            saved_components["scaler"] = scaler_state
        if resumable:
            prepared = _prepare_components(components, saved_components, torch)
            if "native_component_identities" in state and (
                state["native_component_identities"]
                != _component_identities(components, torch)
            ):
                raise ValueError("Checkpoint component implementations/bindings differ")
        learning = prepare_learning_state(trainer, state)
        if (
            "current_task_id" in state
            and state["current_task_id"] != learning["current_task_id"]
        ):
            raise ValueError("Checkpoint task identifiers disagree")

    model = getattr(trainer.agent, "model", None)
    previous_poison = getattr(trainer, "_runtime_checkpoint_poisoned", False)
    weights_loaded = False
    if model is not None and hasattr(model, "load_state_dict"):
        # Model loaders can apply earlier keys/shards before a later failure.
        trainer._runtime_checkpoint_poisoned = True
        try:
            weights_loaded = load_model_checkpoint(
                model, model_dir, torch_module=torch, trusted=trusted, strict=resumable
            )
        except trainer_exceptions as exc:
            if resumable:
                raise
            logger.warning("Failed to import model weights: %s", exc)
            return False
    if resumable and not weights_loaded:
        raise ValueError("Checkpoint is missing loadable model weights")
    if state is None and not weights_loaded:
        # A False model-loader result means no candidate layout was found; no
        # policy/counter/component commit occurred in this branch.
        trainer._runtime_checkpoint_poisoned = previous_poison
        return False
    if weights_loaded and callable(getattr(model, "named_parameters", None)):
        _native_policy_finite(trainer)
    elif weights_loaded and resumable:
        raise ValueError(
            "Native learner checkpoints require verifiable policy parameters"
        )
    if weights_loaded:
        tokenizer = getattr(trainer.agent, "tokenizer", None)
        loader = getattr(tokenizer, "from_pretrained", None)
        if callable(loader):
            try:
                trainer.agent.tokenizer = loader(model_dir)
            except trainer_exceptions as exc:
                if resumable:
                    raise
                logger.warning("Failed to import tokenizer: %s", exc)
                return False
    if not resumable:
        # Preserve model/tokenizer imports without installing partial counters
        # or certifying a complete learner. A no-layout result made no model
        # change; otherwise construct a fresh trainer for deliberate warm start.
        if not weights_loaded:
            trainer._runtime_checkpoint_poisoned = previous_poison
        return False

    assert state is not None
    # All registered learner components were validated before the first model
    # mutation. Remaining device/hooks/publication failures retain quarantine.
    trainer._runtime_checkpoint_poisoned = True
    for name, component in components.items():
        if name == "optimizer":
            value = _owned_optimizer_state(component, prepared[name], torch)
        else:
            value = prepared[name].state_dict()
            # Scaler/scheduler state is small; detach cached/mapped metadata.
            value = copy.deepcopy(value)
        component.load_state_dict(value)
    _native_policy_finite(trainer)
    for name, value in counters.items():
        setattr(trainer, name, value)
    if learning is not None:
        apply_learning_state(trainer, learning)
    if scaler_state is not None and scaler_state:
        pending_gradient = next(
            (gradient for gradient in gradients.values() if gradient is not None), None
        )
        if pending_gradient is not None:
            trainer.scaler.scale(torch.zeros((), device=pending_gradient.device))
    parameters = _parameters(trainer)
    for name, gradient in gradients.items():
        parameters[name].grad = gradient
    trainer._grpo_accumulation = normalization

    restore_rollout_state(trainer.agent, state.get("rollout_state"))
    _sync_restored_policy(trainer)
    if "rng" in state:
        # Publication may consume global RNG; resume the learner stream only
        # after it succeeds, while any late failure still leaves quarantine.
        restore_rng(state["rng"], torch)
    else:
        logger.warning(
            "Legacy checkpoint has no RNG state; exact stochastic continuation is unavailable"
        )
    logger.info("Loaded native learner checkpoint from %s", path)
    return True
