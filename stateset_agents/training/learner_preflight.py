"""Fixed-size native DDP admission checks before buffers or model scoring.

The handshake exchanges configuration digests and validity bits, never sampled
IDs, prompt text, parameter values or pickled objects. Uneven complete prompt
groups remain supported. Initial and invalidated replica values use a separate
bounded content-digest phase after scalar admission.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from typing import Any

from .ddp_control import ddp_learner_poisoned

LEARNER_PREFLIGHT_PROTOCOL = "stateset-native-learner-preflight-v5"
_COMPONENTS = ("loss", "schedule", "model", "optimization", "estimator", "rollouts")


def _kind(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _integer(value: Any, *, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        raise ValueError(
            "Learner counters and schedules require exact bounded integers"
        )
    return value


def _vocabulary(model: Any) -> tuple[int | None, int | None]:
    sizes = []
    for getter in ("get_input_embeddings", "get_output_embeddings"):
        method = getattr(model, getter, None)
        embedding = method() if callable(method) else None
        weight = getattr(embedding, "weight", None)
        size = (
            weight.shape[0]
            if weight is not None
            else getattr(getattr(model, "config", None), "vocab_size", None)
        )
        sizes.append(None if size is None else _integer(size, minimum=1))
    return sizes[0], sizes[1]


def _layout(model: Any) -> Any:
    from .implementation_identity import model_implementation

    if model is None:
        return None
    return {
        "type": _kind(model),
        "implementation": model_implementation(model),
        "parameters": [
            (name, list(p.shape), str(p.dtype), p.requires_grad)
            for name, p in model.named_parameters()
        ],
        "buffers": [
            (name, list(b.shape), str(b.dtype)) for name, b in model.named_buffers()
        ],
        "vocabulary": _vocabulary(model),
    }


def _reference(trainer: Any) -> tuple[Any, float]:
    reference = trainer.reference_model
    beta = float(getattr(trainer.config, "beta", 0.0))
    manager = trainer.continual_manager
    if manager is not None:
        manager_reference = getattr(manager, "reference_model", None)
        if manager_reference is not None:
            reference = manager_reference
        beta = manager.get_effective_beta(beta)
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("Learner beta must be finite and nonnegative")
    return reference, beta


def _loss(trainer: Any) -> dict[str, Any]:
    from .loss_computation import _token_kl_objective, _token_objective

    config = trainer.config
    reference, beta = _reference(trainer)
    objective = _token_objective(config).with_(
        entropy_coef=float(getattr(config, "entropy_coef", 0.0))
    )
    if beta > 0 and reference is not None:
        objective = _token_kl_objective(objective, config, beta=beta)
    disable_dropout = getattr(config, "disable_dropout", False)
    if type(disable_dropout) is not bool:
        raise ValueError("Learner disable_dropout must be a boolean")
    clip = getattr(config, "reward_clip", None)
    if clip is not None and (not math.isfinite(float(clip)) or float(clip) < 0):
        raise ValueError("Learner reward clipping must be finite and nonnegative")
    if getattr(config, "old_logprobs_source", "recompute") not in {
        "recompute",
        "sampler",
    }:
        raise ValueError("Unknown learner old-policy likelihood source")
    return {
        "objective": asdict(objective),
        "explicit_objective": getattr(config, "objective", None) is not None,
        "baseline_type": getattr(config, "baseline_type", "group_mean"),
        "advantage_normalization": bool(
            getattr(config, "advantage_normalization", True)
        ),
        "reward_clip": None if clip is None else float(clip),
        "max_prompt_length": getattr(config, "max_prompt_length", None),
        "disable_dropout": disable_dropout,
        "beta": beta,
        "enhanced": beta > 0
        or bool(getattr(config, "use_reference_model", False))
        or reference is not None,
        "old_logprobs_source": getattr(config, "old_logprobs_source", "recompute"),
        "allow_sequence_fallback": bool(
            getattr(config, "allow_sequence_fallback", False)
        ),
        "allow_missing_sampler_logprobs": bool(
            getattr(config, "allow_missing_sampler_logprobs", False)
        ),
        "allow_unmasked_labels": bool(getattr(config, "allow_unmasked_labels", False)),
    }


def _schedule(trainer: Any) -> dict[str, Any]:
    from .diagnostics import diagnostic_control
    from .loss_computation import _token_objective

    config = trainer.config
    if ddp_learner_poisoned(trainer):
        raise ValueError(
            "A failed native checkpoint apply requires a successful collective reload"
        )
    guard = diagnostic_control(trainer)
    if guard["failure"] is not None:
        raise ValueError("A failed native diagnostic guard requires a healthy reload")
    inner = _integer(getattr(config, "num_gradient_updates", 1), minimum=1)
    steps = _integer(getattr(config, "gradient_accumulation_steps", 1), minimum=1)
    counter = _integer(trainer._grad_accum_step)
    # Chunk counts may differ across ranks, but must be locally usable before
    # an old-policy forward. They deliberately do not enter the digest.
    _integer(getattr(config, "generation_batch_size", 0) or 0)
    max_norm = getattr(config, "max_grad_norm", 1.0)
    if max_norm is not None and (
        type(max_norm) not in (int, float)
        or not math.isfinite(max_norm)
        or max_norm < 0
    ):
        raise ValueError("Learner gradient norm limit must be finite and nonnegative")
    if getattr(config, "bf16", False) and getattr(config, "fp16", False):
        raise ValueError("Learner precision modes are mutually exclusive")
    pending = getattr(trainer, "_grpo_accumulation", None)
    normalization = None
    if pending is not None:
        normalization = {
            key: pending[key]
            for key in ("measure", "policy_normalizer", "entropy_normalizer")
        }
        _integer(pending["policy_count"])
        _integer(pending["entropy_count"])
        _integer(pending["policy_normalizer"], minimum=1)
        _integer(pending["entropy_normalizer"])
        if pending["measure"] not in {"trajectory", "token"} or not counter % steps:
            raise ValueError("Invalid pending native accumulation window")
        if inner == 1:
            objective = _token_objective(config)
            sequence = objective.aggregate in {"seq_mean", "seq_sum_const"}
            if pending["measure"] != ("trajectory" if sequence else "token") or bool(
                pending["entropy_normalizer"]
            ) != (sequence and float(getattr(config, "entropy_coef", 0)) > 0):
                raise ValueError("Cannot change a pending native accumulation measure")
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    if isinstance(trainer.agent.model, DistributedDataParallel):
        if trainer.distributed_config.world_size != dist.get_world_size(
            trainer.agent.model.process_group
        ):
            raise ValueError(
                "Reported native DDP world size differs from the process group"
            )
    elif trainer.distributed_config.world_size < 2:
        raise ValueError("Native DDP requires global diagnostic synchronization")
    return {
        "inner_updates": inner,
        "accumulation_steps": steps,
        "global_step": _integer(trainer.global_step),
        "grad_accum_step": counter,
        "normalization": normalization,
        "bf16": bool(getattr(config, "bf16", False)),
        "fp16": bool(getattr(config, "fp16", False)),
        "max_grad_norm": getattr(config, "max_grad_norm", 1.0),
        "strategy": trainer.distributed_config.strategy,
        "diagnostics": guard,
    }


def _model(trainer: Any) -> dict[str, Any]:
    from .implementation_identity import learner_implementation

    model = trainer.agent.model
    reference, _ = _reference(trainer)
    return {
        "implementation": learner_implementation(trainer),
        "policy": _layout(model.module),
        "reference": _layout(reference),
        "broadcast_buffers": model.broadcast_buffers,
        "find_unused_parameters": model.find_unused_parameters,
        "static_graph": model.static_graph,
        "gradient_as_bucket_view": model.gradient_as_bucket_view,
    }


def _optimization(trainer: Any) -> dict[str, Any]:
    from .scheduler_identity import scheduler_identity

    optimizer = trainer.optimizer
    names = {id(p): name for name, p in trainer.agent.model.named_parameters()}
    groups = None
    if optimizer is not None:
        groups = []
        for group in optimizer.param_groups:
            values = {name: value for name, value in group.items() if name != "params"}
            values["parameters"] = [names[id(p)] for p in group["params"]]
            groups.append(values)
    scheduler = trainer.lr_scheduler
    scaler = trainer.scaler
    return {
        "optimizer": None if optimizer is None else _kind(optimizer),
        "groups": groups,
        "scheduler": (
            None
            if scheduler is None
            else [
                _kind(scheduler),
                scheduler.state_dict(),
                scheduler_identity(scheduler),
            ]
        ),
        "scaler": None if scaler is None else [_kind(scaler), scaler.state_dict()],
    }


def _estimator(trainer: Any) -> dict[str, Any]:
    from .learning_state import _reward_baseline

    config = trainer.config
    manager = trainer.continual_manager
    return {
        "baseline": (
            _reward_baseline(trainer)
            if getattr(config, "objective", None) is None
            and getattr(config, "baseline_type", None) == "global_mean"
            else None
        ),
        "continual": (
            None if manager is None else [_kind(manager), asdict(manager.config)]
        ),
        "task": getattr(trainer, "_current_task_id", None),
    }


def _rollouts(trainer: Any, groups: list[Any]) -> dict[str, Any]:
    from .loss_computation import _grpo_objective, _token_rows_for_group
    from .rollout_contract import SampledTurn

    config = trainer.config
    if any(
        not hasattr(group, "trajectories") or group.trajectories is None
        for group in groups
    ):
        raise ValueError("Invalid DDP rollout group")
    groups = [group for group in groups if getattr(group, "trajectories", None)]
    if not groups:
        return {"path": "empty", "source": "empty"}
    input_size, output_size = _vocabulary(trainer.agent.model.module)
    token_path, sampler_complete = True, True
    clip = getattr(config, "reward_clip", None)
    for group in groups:
        rows = _token_rows_for_group(group)
        token_path = token_path and rows is not None
        if rows is not None:
            for prompt, response, _ in rows:
                if input_size is not None and any(
                    token >= input_size for token in prompt + response
                ):
                    raise ValueError("Sampled input token exceeds the model vocabulary")
                if output_size is not None and any(
                    token >= output_size for token in response
                ):
                    raise ValueError(
                        "Sampled response token exceeds the output vocabulary"
                    )
        for trajectory in group.trajectories:
            reward = float(trajectory.total_reward)
            if not math.isfinite(reward):
                raise ValueError("Sampled rewards must be finite before clipping")
            clipped = (
                reward if clip is None else max(-float(clip), min(float(clip), reward))
            )
            if abs(clipped) > 3.4028234663852886e38:
                raise ValueError("Sampled clipped rewards must fit FP32")
            for turn in trajectory.turns:
                role = turn.get("role") if isinstance(turn, dict) else turn.role
                if role == "assistant":
                    metadata = (
                        turn.get("metadata")
                        if isinstance(turn, dict)
                        else turn.metadata
                    )
                    metadata = metadata or {}
                    sampler_complete = (
                        sampler_complete
                        and metadata.get("sampler_log_probs") is not None
                    )
                    # An earlier incomplete turn may have selected fallback;
                    # still validate every later turn with complete token IDs.
                    if metadata.get("prompt_token_ids") and metadata.get("token_ids"):
                        SampledTurn.from_metadata(metadata)
    if not token_path:
        if not getattr(config, "allow_sequence_fallback", False):
            raise ValueError("DDP rollout lacks exact sampled-token metadata")
        _grpo_objective(config)
        source = "sequence"
    elif getattr(config, "num_gradient_updates", 1) > 1:
        source = "snapshot"
    elif getattr(config, "old_logprobs_source", "recompute") == "sampler":
        if not sampler_complete and not getattr(
            config, "allow_missing_sampler_logprobs", False
        ):
            raise ValueError("DDP rollout lacks required sampler likelihoods")
        source = "sampler" if sampler_complete else "recompute_fallback"
    else:
        source = "recompute"
    return {"path": "token" if token_path else "sequence", "source": source}


def _scalar_tensor(value: Any) -> Any:
    import torch

    if torch.is_tensor(value) and value.numel() == 1:
        return value.detach().item()
    raise TypeError("Learner settings require JSON values or scalar tensors")


def verify_ddp_learner_batch(trainer: Any, groups: list[Any]) -> dict[str, Any] | None:
    """Agree on native learner admission before any buffers or forwards change.

    A rejected batch leaves optimizer, counters, gradient banks, model buffers
    and running reward state untouched. Peers must still call this collective
    on the same DDP process group; process/network failures are not recoverable
    through a metadata handshake. Policy/reference parameters and optimizer slots
    are checked on first or invalidated admission; buffers and pending gradients
    remain rank-local. Uneven row/chunk counts are deliberately allowed.
    """
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    model = trainer.agent.model
    if (
        not isinstance(model, DistributedDataParallel)
        or dist.get_world_size(model.process_group) < 2
    ):
        return None
    device = next(model.parameters()).device
    rollout: dict[str, Any] = {}

    def describe_rollouts() -> dict[str, Any]:
        rollout.update(_rollouts(trainer, groups))
        return rollout

    factories = {
        "loss": lambda: _loss(trainer),
        "schedule": lambda: _schedule(trainer),
        "model": lambda: _model(trainer),
        "optimization": lambda: _optimization(trainer),
        "estimator": lambda: _estimator(trainer),
        "rollouts": describe_rollouts,
    }
    hashes: list[bytes] = []
    mask = 0
    error: Exception | None = None
    for index, name in enumerate(_COMPONENTS):
        try:
            encoded = json.dumps(
                factories[name](),
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
                default=_scalar_tensor,
            ).encode()
            digest = hashlib.sha256(encoded).digest()
            mask |= 1 << index
        except (
            ValueError,
            TypeError,
            LookupError,
            AttributeError,
            OverflowError,
            RuntimeError,
            OSError,
        ) as exc:
            digest = bytes(32)
            error = exc if error is None else error
        hashes.append(digest)
    # A high validity-word bit requests value agreement on every peer. Ranks
    # need not have the same local cache state after a checkpoint-load attempt.
    needs_values = bool(getattr(trainer, "_ddp_replica_consensus_pending", True))
    header = torch.tensor(
        [
            5,
            mask
            | (64 if needs_values else 0)
            | (128 if rollout.get("path") == "empty" else 0),
            *b"".join(hashes),
        ],
        dtype=torch.int64,
        device=device,
    )
    peers = [
        torch.empty_like(header)
        for _ in range(dist.get_world_size(model.process_group))
    ]
    dist.all_gather(peers, header, group=model.process_group)
    decoded = [peer.cpu().tolist() for peer in peers]
    if any(row[0] != 5 for row in decoded):
        raise ValueError("Native DDP learner preflight protocol differs")
    invalid = {
        rank: [
            name for index, name in enumerate(_COMPONENTS) if not row[1] & (1 << index)
        ]
        for rank, row in enumerate(decoded)
        if row[1] & 63 != (1 << len(_COMPONENTS)) - 1
    }
    if invalid:
        raise ValueError(
            f"Invalid native DDP learner preflight components: {invalid}"
        ) from error
    differences = [
        name
        for index, name in enumerate(_COMPONENTS)
        if len(
            {
                tuple(row[2 + index * 32 : 2 + (index + 1) * 32])
                for row in decoded
                if name != "rollouts" or not row[1] & 128
            }
        )
        > 1
    ]
    if differences:
        raise ValueError(
            f"Native DDP learner preflight differs across ranks: {differences}"
        )
    populated = [row for row in decoded if not row[1] & 128]
    effective = {"path": "empty", "source": "empty"}
    if populated:
        wanted = bytes(populated[0][-32:])
        candidates = [
            {"path": "token", "source": source}
            for source in ("snapshot", "sampler", "recompute", "recompute_fallback")
        ]
        candidates.append({"path": "sequence", "source": "sequence"})
        for candidate in candidates:
            encoded = json.dumps(
                candidate, sort_keys=True, allow_nan=False, separators=(",", ":")
            ).encode()
            if hashlib.sha256(encoded).digest() == wanted:
                effective = candidate
                break
        if effective["path"] == "empty":
            raise ValueError("Unknown DDP rollout path identity")
        if effective["path"] == "sequence" and len(populated) != len(decoded):
            raise ValueError(
                "Empty contributors require exact token rollouts; sequence compatibility is unsupported"
            )
    if any(row[1] & 64 for row in decoded):
        from .replica_consensus import verify_ddp_replica_state

        trainer._ddp_replica_consensus_pending = True
        verify_ddp_replica_state(trainer)
        trainer._ddp_replica_consensus_pending = False
    return {
        "local_empty": bool(int(header[1].item()) & 128),
        "global_empty": not populated,
        "rollout_path": effective["path"],
        "rollout_source": effective["source"],
        "protocol": LEARNER_PREFLIGHT_PROTOCOL,
        "components": {
            name: digest.hex() for name, digest in zip(_COMPONENTS, hashes, strict=True)
        },
    }
