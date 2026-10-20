"""Shared and rank-local payloads for complete native DDP checkpoints."""

from __future__ import annotations

import copy
import json
import random
import sys
from collections import defaultdict
from dataclasses import asdict
from enum import Enum
from types import SimpleNamespace
from typing import Any, cast

from .diagnostics import diagnostic_control
from .implementation_identity import learner_implementation, model_implementation
from .learning_state import (
    apply_learning_state,
    capture_learning_state,
    prepare_learning_state,
)
from .scheduler_identity import scheduler_identity
from .state_fingerprint import state_fingerprint

PROTOCOL = "stateset-native-ddp-checkpoint-v3"
_CONFIG = (
    "model_name",
    "model_revision",
    "objective",
    "objective_overrides",
    "gradient_accumulation_steps",
    "num_gradient_updates",
    "max_completion_length",
    "max_prompt_length",
    "entropy_coef",
    "clip_ratio",
    "seq_clip_ratio",
    "reward_clip",
    "baseline_type",
    "advantage_normalization",
    "beta",
    "use_reference_model",
    "bf16",
    "fp16",
    "max_grad_norm",
    "old_logprobs_source",
    "allow_sequence_fallback",
    "allow_missing_sampler_logprobs",
    "learning_rate",
    "weight_decay",
    "num_generations",
    "zero_signal_patience",
)
_COUNTERS = (
    "global_step",
    "current_epoch",
    "_next_episode",
    "best_eval_metric",
    "steps_without_improvement",
    "_grad_accum_step",
)


def _kind(value: Any) -> str | None:
    return (
        None
        if value is None
        else f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    raise TypeError("Native checkpoint identities require plain configuration values")


def _canonical(value: Any) -> Any:
    return json.loads(
        json.dumps(value, sort_keys=True, allow_nan=False, default=_plain)
    )


def _model_config(model: Any) -> Any:
    config = getattr(model, "config", None)
    if config is None:
        return None
    values = (
        config.to_dict()
        if callable(getattr(config, "to_dict", None))
        else vars(config).copy()
    )
    for name in ("_name_or_path", "transformers_version", "use_cache"):
        values.pop(name, None)
    return _canonical(values)


def model_identity(model: Any) -> dict[str, Any]:
    """Identify the supplied architecture/configuration without executing it."""
    return {
        "class": _kind(model),
        "implementation": model_implementation(model),
        "config": _model_config(model),
        "state": [
            (
                (name, list(value.shape), str(value.dtype))
                if hasattr(value, "shape")
                else (name, _kind(value))
            )
            for name, value in model.state_dict().items()
        ],
        "buffers": [
            (name, list(value.shape), str(value.dtype))
            for name, value in model.named_buffers(remove_duplicate=False)
        ],
        "trainable": [(name, p.requires_grad) for name, p in model.named_parameters()],
    }


def _tokenizer_identity(tokenizer: Any) -> Any:
    if tokenizer is None:
        return None
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        state = json.loads(backend.to_str())
        # Fast tokenizers leave these per-call settings in their backend after
        # encoding. Actual training lengths and tokenizer sides are bound
        # separately; vocabulary, added tokens and processing remain pinned.
        for name in ("padding", "truncation"):
            if name in state:
                state[name] = None
        vocabulary = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    else:
        vocabulary = tokenizer.get_vocab()
    return state_fingerprint(
        _canonical(
            {
                "type": _kind(tokenizer),
                "vocabulary": vocabulary,
                "template": getattr(tokenizer, "chat_template", None),
                "special_tokens": getattr(tokenizer, "special_tokens_map", None),
                "padding_side": getattr(tokenizer, "padding_side", None),
                "truncation_side": getattr(tokenizer, "truncation_side", None),
            }
        )
    )


def static_identity(trainer: Any) -> dict[str, Any]:
    """Bind configuration, component classes, parameter groups and tokenizer."""
    import torch

    policy = trainer.agent.model.module
    names = {id(p): name for name, p in trainer.agent.model.named_parameters()}
    manager = trainer.continual_manager
    return cast(
        dict[str, Any],
        _canonical(
            {
                "protocol": PROTOCOL,
                "trainer": _kind(trainer),
                "implementation": learner_implementation(trainer),
                "runtime": {
                    "python": list(sys.version_info[:3]),
                    "torch": str(torch.__version__),
                },
                "device_type": next(policy.parameters()).device.type,
                "config": {
                    name: getattr(trainer.config, name, None) for name in _CONFIG
                },
                "policy": model_identity(policy),
                "optimizer": _kind(trainer.optimizer),
                "optimizer_groups": (
                    None
                    if trainer.optimizer is None
                    else [
                        [names[id(p)] for p in group["params"]]
                        for group in trainer.optimizer.param_groups
                    ]
                ),
                "scheduler": _kind(trainer.lr_scheduler),
                "scaler": _kind(trainer.scaler),
                "continual": (
                    None
                    if manager is None
                    else [_kind(manager), asdict(manager.config)]
                ),
                "tokenizer": _tokenizer_identity(
                    getattr(trainer.agent, "tokenizer", None)
                ),
            }
        ),
    )


def split_model(model: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep model state once, with persistent/nonpersistent buffers rank-local."""
    buffers = dict(model.named_buffers(remove_duplicate=False))
    return {
        name: value for name, value in model.state_dict().items() if name not in buffers
    }, buffers


def _accumulation(trainer: Any, torch: Any) -> dict[str, Any]:
    from .checkpoint_state import (
        _prepare_normalization,
        _validate_gradients,
        _validate_scaler,
    )

    steps = trainer._get_grad_accum_steps()
    count = trainer._grad_accum_step
    if type(count) is not int or not 0 <= count <= 2**63 - 1:
        raise ValueError("Invalid native checkpoint accumulation counter")
    parameters = dict(trainer.agent.model.named_parameters())
    gradients = {}
    if count % steps:
        if trainer.optimizer is None:
            raise ValueError("Pending native gradients require an optimizer")
        gradients = {name: p.grad for name, p in parameters.items()}
        normalization = trainer._grpo_accumulation
        _validate_gradients(
            gradients,
            parameters,
            torch,
            allow_empty=normalization is not None
            and normalization["policy_count"] == 0,
        )
    scaler = None if trainer.scaler is None else trainer.scaler.state_dict()
    _validate_scaler(scaler)
    result = {
        "schema_version": 1,
        "steps": steps,
        "gradients": gradients,
        "scaler": scaler,
    }
    if trainer._grpo_accumulation is not None:
        result["schema_version"] = 2
        result["normalization"] = _prepare_normalization(
            trainer._grpo_accumulation,
            parameters,
            torch,
            count % steps,
            steps,
            copy_tensors=False,
            allow_empty=True,
        )
    return result


def capture_payload(trainer: Any) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Capture a paused learner without cloning full parameter/gradient banks."""
    import torch

    from .learner_preflight import _reference
    from .runtime_state import capture_rng, capture_rollout_state

    policy, buffers = split_model(trainer.agent.model.module)
    reference, _ = _reference(trainer)
    ref_state, ref_buffers = (
        split_model(reference) if reference is not None else (None, None)
    )
    shared = {
        "protocol": PROTOCOL,
        "identity": static_identity(trainer),
        "policy": policy,
        "optimizer": (
            None if trainer.optimizer is None else trainer.optimizer.state_dict()
        ),
        "scheduler": (
            None if trainer.lr_scheduler is None else trainer.lr_scheduler.state_dict()
        ),
        "scheduler_identity": scheduler_identity(trainer.lr_scheduler),
        "reference": ref_state,
        "reference_identity": None if reference is None else model_identity(reference),
    }
    learning = capture_learning_state(trainer, include_reference=False)
    local: dict[str, Any] = {
        "protocol": PROTOCOL,
        "counters": {name: getattr(trainer, name) for name in _COUNTERS},
        "buffers": buffers,
        "reference_buffers": ref_buffers,
        "gradient_accumulation": _accumulation(trainer, torch),
        "rng": capture_rng(torch),
        "rollout_state": capture_rollout_state(trainer.agent),
        "learning_state": learning,
    }
    # Only shared state must match; RNG, buffers, replay/EWC and pending banks
    # deliberately remain local. Scalar AMP state is also part of agreement.
    agreement = state_fingerprint(
        {
            "shared": shared,
            "global_step": trainer.global_step,
            "grad_accum_step": trainer._grad_accum_step,
            "baseline": learning["reward_baseline"],
            "diagnostics": diagnostic_control(trainer),
            "scaler": local["gradient_accumulation"]["scaler"],
            "normalization": (
                None
                if trainer._grpo_accumulation is None
                else {
                    name: trainer._grpo_accumulation[name]
                    for name in ("measure", "policy_normalizer", "entropy_normalizer")
                }
            ),
        }
    )
    return shared, local, agreement


def _model_state(model: Any, shared: Any, buffers: Any, torch: Any) -> dict[str, Any]:
    expected = model.state_dict()
    current_buffers = dict(model.named_buffers(remove_duplicate=False))
    if (
        not isinstance(shared, dict)
        or not isinstance(buffers, dict)
        or set(buffers) != set(current_buffers)
    ):
        raise ValueError("Native checkpoint model/buffer names differ")
    merged = {
        **shared,
        **{name: value for name, value in buffers.items() if name in expected},
    }
    if set(merged) != set(expected):
        raise ValueError("Native checkpoint model state names differ")
    aliases: dict[int, Any] = {}
    for values, targets in ((merged, expected), (buffers, current_buffers)):
        for name, value in values.items():
            target = targets[name]
            if torch.is_tensor(target):
                if (
                    not torch.is_tensor(value)
                    or value.shape != target.shape
                    or value.dtype != target.dtype
                ):
                    raise ValueError(
                        "Native checkpoint model tensor shape/dtype differs"
                    )
                # Buffers may contain legitimate infinity masks. Parameters and
                # optimizer values are validated separately with bounded hashes.
            elif type(value) is not type(target):
                raise ValueError("Native checkpoint extra-state type differs")
    for iterator in (
        model.named_parameters(remove_duplicate=False),
        model.named_buffers(remove_duplicate=False),
    ):
        for name, target in iterator:
            value = merged[name] if name in merged else buffers[name]
            previous = aliases.get(id(target))
            if previous is not None and not torch.equal(previous, value):
                raise ValueError("Native checkpoint tied tensor values disagree")
            aliases[id(target)] = value
    return merged


def _optimizer_stage(optimizer: Any, saved: Any, torch: Any) -> Any:
    if optimizer is None:
        if saved is not None:
            raise ValueError("Native checkpoint optimizer availability differs")
        return None
    if not isinstance(saved, dict) or set(saved) != {"state", "param_groups"}:
        raise ValueError("Invalid native checkpoint optimizer schema")
    if len(saved["param_groups"]) != len(optimizer.param_groups):
        raise ValueError("Native checkpoint optimizer group count differs")
    expected = {}
    for current, group in zip(
        optimizer.param_groups, saved["param_groups"], strict=True
    ):
        if not isinstance(group, dict) or set(group) != set(current):
            raise ValueError("Native checkpoint optimizer group schema differs")
        if len(current["params"]) != len(group["params"]):
            raise ValueError("Native checkpoint optimizer parameter count differs")
        for parameter, key in zip(current["params"], group["params"], strict=True):
            if type(key) is not int or key in expected:
                raise ValueError("Invalid native optimizer parameter identity")
            expected[key] = parameter
    if not isinstance(saved["state"], dict) or set(saved["state"]) - expected.keys():
        raise ValueError("Native checkpoint optimizer slot names differ")
    for key, values in saved["state"].items():
        if not isinstance(values, dict):
            raise ValueError("Invalid native optimizer parameter slots")
        if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)) and values:
            group = next(g for g in saved["param_groups"] if key in g["params"])
            required = {"step", "exp_avg", "exp_avg_sq"}
            if group["amsgrad"]:
                required.add("max_exp_avg_sq")
            if set(values) != required:
                raise ValueError("Native checkpoint Adam slot schema differs")
        elif isinstance(optimizer, torch.optim.SGD) and values:
            if set(values) != {"momentum_buffer"}:
                raise ValueError("Native checkpoint SGD slot schema differs")
        for name, value in values.items():
            if isinstance(
                optimizer, (torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW)
            ):
                if not torch.is_tensor(value):
                    raise ValueError("Native optimizer slot must be a tensor")
                if name == "step":
                    if (
                        value.numel() != 1
                        or not value.is_floating_point()
                        or value.item() < 0
                        or value.item() != int(value.item())
                    ):
                        raise ValueError("Invalid native optimizer step slot")
                elif value.dtype != expected[key].dtype:
                    raise ValueError("Native optimizer slot tensor dtype differs")
            if torch.is_tensor(value) and value.shape != expected[key].shape:
                if name != "step" or value.numel() != 1:
                    raise ValueError("Native optimizer slot tensor shape differs")
    state_fingerprint(saved)
    staged = copy.copy(optimizer)
    staged.defaults = copy.deepcopy(optimizer.defaults)
    # Hooks can close over the live optimizer or mutate external state. They
    # cannot provide validation-before-mutation guarantees during staging.
    if (
        optimizer._optimizer_load_state_dict_pre_hooks
        or optimizer._optimizer_load_state_dict_post_hooks
    ):
        raise ValueError("Native checkpoint optimizer load hooks are unsupported")
    staged.state = defaultdict(dict)
    staged.param_groups = []
    for group in optimizer.param_groups:
        staged.param_groups.append(
            {
                **{
                    key: copy.deepcopy(value)
                    for key, value in group.items()
                    if key != "params"
                },
                "params": [
                    torch.nn.Parameter(
                        torch.empty(0, dtype=p.dtype), requires_grad=p.requires_grad
                    )
                    for p in group["params"]
                ],
            }
        )
    # Use the class implementation, so per-instance commit fault injectors do
    # not accidentally act on the live optimizer during validation.
    type(optimizer).load_state_dict(staged, saved)
    return staged


def _validate_rng(state: Any, torch: Any) -> None:
    import numpy as np

    if not isinstance(state, dict) or set(state) != {
        "python",
        "numpy",
        "torch",
        "cuda",
    }:
        raise ValueError("Invalid native RNG schema")
    random.Random().setstate(state["python"])
    values = state["numpy"]
    np.random.RandomState(0).set_state(
        (values[0], np.asarray(values[1], dtype=np.uint32), *values[2:])
    )
    torch.Generator(device="cpu").set_state(state["torch"].cpu())
    if state["cuda"]:
        if (
            not torch.cuda.is_available()
            or len(state["cuda"]) != torch.cuda.device_count()
        ):
            raise ValueError("Native checkpoint CUDA RNG topology differs")
        for index, value in enumerate(state["cuda"]):
            torch.Generator(device=f"cuda:{index}").set_state(value.cpu())


def prepare_payload(trainer: Any, shared: Any, local: Any) -> dict[str, Any]:
    """Validate all restored state before any live model/optimizer mutation."""
    import torch

    from .checkpoint_state import _prepare_accumulation
    from .learner_preflight import _reference

    if (
        not isinstance(shared, dict)
        or shared.get("protocol") != PROTOCOL
        or not isinstance(local, dict)
        or local.get("protocol") != PROTOCOL
    ):
        raise ValueError("Invalid native DDP checkpoint payload protocol")
    if set(shared) != {
        "protocol",
        "identity",
        "policy",
        "optimizer",
        "scheduler",
        "scheduler_identity",
        "reference",
        "reference_identity",
    } or set(local) != {
        "protocol",
        "counters",
        "buffers",
        "reference_buffers",
        "gradient_accumulation",
        "rng",
        "rollout_state",
        "learning_state",
    }:
        raise ValueError("Invalid native DDP checkpoint payload schema")
    if shared["identity"] != static_identity(trainer):
        raise ValueError("Native checkpoint configuration/component identity differs")
    counters = local["counters"]
    if not isinstance(counters, dict) or set(counters) not in (
        set(_COUNTERS),
        set(_COUNTERS) - {"_next_episode"},
    ):
        raise ValueError("Invalid native checkpoint counters")
    from .checkpoint_state import _prepare_episode_cursor

    # Prepared state is private; loading an older native checkpoint retains its
    # historical current_epoch + 1 behavior without mutating the input payload.
    counters = {**counters, "_next_episode": _prepare_episode_cursor(counters)}
    local = {**local, "counters": counters}
    for name in (
        "global_step",
        "current_epoch",
        "steps_without_improvement",
        "_grad_accum_step",
    ):
        if type(counters[name]) is not int or not 0 <= counters[name] <= 2**63 - 1:
            raise ValueError("Invalid native checkpoint counter value")
    if (
        type(counters["best_eval_metric"]) not in (int, float)
        or counters["best_eval_metric"] != counters["best_eval_metric"]
    ):
        raise ValueError("Invalid native checkpoint best metric")
    policy = trainer.agent.model.module
    policy_state = _model_state(policy, shared["policy"], local["buffers"], torch)
    state_fingerprint(shared["policy"])
    reference, _ = _reference(trainer)
    if shared["reference"] is not None:
        template = reference if reference is not None else policy
        candidate = {
            key: value
            for key, value in model_identity(template).items()
            if key != "trainable"
        }
        expected = {
            key: value
            for key, value in shared["reference_identity"].items()
            if key != "trainable"
        }
        if candidate != expected:
            raise ValueError("Native checkpoint reference architecture differs")
        flags = shared["reference_identity"].get("trainable")
        expected_names = [name for name, _ in template.named_parameters()]
        if (
            not isinstance(flags, list)
            or len(flags) != len(expected_names)
            or any(
                not isinstance(row, (list, tuple))
                or len(row) != 2
                or row[0] != name
                or type(row[1]) is not bool
                for row, name in zip(flags, expected_names, strict=True)
            )
        ):
            raise ValueError("Invalid native checkpoint reference trainable flags")
        reference_state = _model_state(
            template, shared["reference"], local["reference_buffers"], torch
        )
        state_fingerprint(shared["reference"])
    else:
        if (
            shared["reference_identity"] is not None
            or local["reference_buffers"] is not None
        ):
            raise ValueError("Native checkpoint reference availability differs")
        reference_state = None
        reference = None
    staged_optimizer = _optimizer_stage(trainer.optimizer, shared["optimizer"], torch)
    staged_scheduler = None
    if trainer.lr_scheduler is not None:
        staged_scheduler = copy.deepcopy(
            trainer.lr_scheduler, {id(trainer.optimizer): staged_optimizer}
        )
        type(trainer.lr_scheduler).load_state_dict(
            staged_scheduler, shared["scheduler"]
        )
    elif shared["scheduler"] is not None:
        raise ValueError("Native checkpoint scheduler availability differs")
    if scheduler_identity(staged_scheduler) != shared["scheduler_identity"]:
        raise ValueError("Native checkpoint scheduler implementation/bindings differ")
    gradients, scaler, normalization = _prepare_accumulation(
        trainer,
        {
            "grad_accum_step": counters["_grad_accum_step"],
            "gradient_accumulation": local["gradient_accumulation"],
        },
        torch,
        copy_tensors=False,
    )
    learning = copy.deepcopy(local["learning_state"])
    if learning["reference_from_continual"]:
        if learning["continual_state"] is None or reference_state is None:
            raise ValueError("Native checkpoint continual reference is missing")
        if any(flag for _, flag in shared["reference_identity"]["trainable"]):
            raise ValueError("Native checkpoint continual reference must be frozen")
        learning["continual_state"]["reference_state"] = reference_state
    # EWC tracks the wrapper's parameter names, whereas the reference template
    # must be unwrapped to avoid deep-copying a DDP process group. Translate only
    # for staging, then restore the public manager's original parameter names.
    plain_names = {id(p): name for name, p in policy.named_parameters()}
    to_plain = {
        name: plain_names[id(p)] for name, p in trainer.agent.model.named_parameters()
    }
    continual = learning["continual_state"]
    if continual is not None:
        for field in ("ewc_fisher", "ewc_params"):
            if field not in continual:
                continue
            if any(name not in to_plain for name in continual[field]):
                raise ValueError("Native checkpoint EWC parameter names differ")
            continual[field] = {
                to_plain[name]: value for name, value in continual[field].items()
            }
    proxy = SimpleNamespace(**vars(trainer))
    proxy.agent = SimpleNamespace(model=policy)
    prepared_learning = prepare_learning_state(proxy, {"learning_state": learning})
    if prepared_learning["manager"] is not None:
        to_wrapped = {plain: name for name, plain in to_plain.items()}
        manager = prepared_learning["manager"]
        for field in ("_ewc_fisher", "_ewc_params"):
            if hasattr(manager, field):
                setattr(
                    manager,
                    field,
                    {
                        to_wrapped[name]: value
                        for name, value in getattr(manager, field).items()
                    },
                )
    if (
        reference_state is not None
        and reference is None
        and not learning["reference_from_continual"]
    ):
        reference = copy.deepcopy(policy)
    _validate_rng(local["rng"], torch)
    rollout = local["rollout_state"]
    if rollout is not None and (
        not isinstance(rollout, dict)
        or set(rollout) != {"schema_version", "publication_version"}
        or rollout["schema_version"] != 1
        or type(rollout["publication_version"]) is not int
        or rollout["publication_version"] < 0
    ):
        raise ValueError("Invalid native checkpoint rollout state")
    return {
        "shared": shared,
        "local": local,
        "policy": policy_state,
        "reference": reference,
        "reference_state": reference_state,
        "optimizer": staged_optimizer,
        "scheduler": staged_scheduler,
        "gradients": gradients,
        "scaler": scaler,
        "normalization": normalization,
        "learning": prepared_learning,
    }


def _apply_model(model: Any, state: Any, buffers: Any, torch: Any) -> None:
    model.load_state_dict(state, strict=True)
    with torch.no_grad():
        for name, value in model.named_buffers(remove_duplicate=False):
            value.copy_(buffers[name].to(device=value.device))


def _owned_optimizer_state(optimizer: Any, staged: Any, torch: Any) -> Any:
    """Detach CPU slots from mapped files without duplicating full GPU slots.

    Torch loads slots onto the destination parameter's device. CPU destinations
    and ordinary CPU step counters need an explicit copy; GPU-bound slots are
    copied by the optimizer loader itself. No live state may retain a writable
    mapped view, even if the caller configured Torch's mmap mode to MAP_SHARED.
    """
    saved = staged.state_dict()

    def own(value: Any, *, clone: bool) -> Any:
        if torch.is_tensor(value):
            return value.clone() if clone else value
        if isinstance(value, dict):
            return {name: own(item, clone=clone) for name, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(own(item, clone=clone) for item in value)
        return value

    slots = {}
    for current, group in zip(
        optimizer.param_groups, saved["param_groups"], strict=True
    ):
        for parameter, key in zip(current["params"], group["params"], strict=True):
            if key in saved["state"]:
                slots[key] = {
                    name: own(
                        value, clone=parameter.device.type == "cpu" or name == "step"
                    )
                    for name, value in saved["state"][key].items()
                }
    return {"state": slots, "param_groups": saved["param_groups"]}


def apply_payload(trainer: Any, prepared: dict[str, Any]) -> None:
    """Apply prevalidated state; the caller coordinates errors and poisons peers."""
    import torch

    from .runtime_state import restore_rng, restore_rollout_state

    shared, local = prepared["shared"], prepared["local"]
    _apply_model(
        trainer.agent.model.module, prepared["policy"], local["buffers"], torch
    )
    # Release old slot/bank allocations only after all peers validated. An apply
    # failure requires a fresh successful restore, not continued training.
    trainer._grpo_accumulation = None
    for parameter in trainer.agent.model.parameters():
        parameter.grad = None
    if trainer.optimizer is not None:
        trainer.optimizer.state.clear()
        trainer.optimizer.load_state_dict(
            _owned_optimizer_state(trainer.optimizer, prepared["optimizer"], torch)
        )
    if trainer.lr_scheduler is not None:
        trainer.lr_scheduler.load_state_dict(
            copy.deepcopy(prepared["scheduler"].state_dict())
        )
    if trainer.scaler is not None:
        trainer.scaler.load_state_dict(prepared["scaler"])
    apply_learning_state(trainer, prepared["learning"])
    if shared["reference"] is None:
        trainer.reference_model = None
    if (
        not local["learning_state"]["reference_from_continual"]
        and prepared["reference"] is not None
    ):
        _apply_model(
            prepared["reference"],
            prepared["reference_state"],
            local["reference_buffers"],
            torch,
        )
        flags = dict(shared["reference_identity"]["trainable"])
        for name, parameter in prepared["reference"].named_parameters():
            parameter.requires_grad_(flags[name])
        prepared["reference"].eval()
        trainer.reference_model = prepared["reference"]
    elif local["learning_state"]["reference_from_continual"]:
        # Nonpersistent teacher buffers are not part of state_dict().
        with torch.no_grad():
            for name, value in trainer.reference_model.named_buffers(
                remove_duplicate=False
            ):
                value.copy_(local["reference_buffers"][name].to(device=value.device))
    parameters = dict(trainer.agent.model.named_parameters())
    for name, gradient in prepared["gradients"].items():
        parameters[name].grad = (
            None
            if gradient is None
            else gradient.to(device=parameters[name].device, copy=True)
        )
    normalization = prepared["normalization"]
    if normalization is not None:
        normalization = {
            **normalization,
            "entropy_gradients": {
                name: (
                    None
                    if value is None
                    else value.to(device=parameters[name].device, copy=True)
                )
                for name, value in normalization["entropy_gradients"].items()
            },
        }
    trainer._grpo_accumulation = normalization
    for name, value in local["counters"].items():
        setattr(trainer, name, value)
    if trainer.scaler is not None and prepared["scaler"]:
        pending = next(
            (p.grad for p in parameters.values() if p.grad is not None), None
        )
        if (
            pending is not None
            or trainer._grad_accum_step % trainer._get_grad_accum_steps()
        ):
            trainer.scaler.scale(
                torch.zeros((), device=next(trainer.agent.model.parameters()).device)
            )
    restore_rng(local["rng"], torch)
    restore_rollout_state(trainer.agent, local["rollout_state"])
