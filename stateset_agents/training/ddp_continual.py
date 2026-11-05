"""Collective task-memory consolidation without DDP forward/reducer side effects."""

from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any

from .ddp_control import DDPControlGroup, ddp_learner_poisoned
from .state_fingerprint import state_fingerprint

_REPORT = (
    "attempted",
    "accepted",
    "skipped",
    "sample_errors",
    "invalid_loss",
    "invalid_gradients",
    "no_supervised_tokens",
    "failed",
)
_BUCKET_BYTES = 4 * 1024**2


def _identity(manager: Any, agent: Any, task_id: Any) -> dict[str, Any]:
    import torch

    from .continual_learning import ContinualLearningManager, _validate_continual_config
    from .ddp_checkpoint_state import _canonical, _tokenizer_identity, model_identity

    if type(manager) is not ContinualLearningManager:
        raise ValueError(
            "Native DDP consolidation requires the standard continual manager"
        )
    if manager._ddp_restore_failed:
        raise ValueError("Failed task-memory commit requires full checkpoint reload")
    if task_id is not None and not isinstance(task_id, str):
        raise ValueError("Task identifier must be a string or None")
    _validate_continual_config(manager.config)
    model = agent.model
    parameters = dict(model.named_parameters())
    if set(manager._ewc_fisher) != set(manager._ewc_params):
        raise ValueError("EWC Fisher and anchor names differ")
    for name, fisher in manager._ewc_fisher.items():
        if name not in parameters:
            raise ValueError("EWC parameter names differ from the DDP wrapper")
        parameter = parameters[name]
        dtype = torch.float64 if parameter.dtype == torch.float64 else torch.float32
        anchor = manager._ewc_params[name]
        if (
            fisher.shape != parameter.shape
            or anchor.shape != parameter.shape
            or fisher.dtype != dtype
            or anchor.dtype != dtype
            or bool((fisher < 0).any())
        ):
            raise ValueError("Invalid prior EWC tensor shape/dtype/value")
    if type(model.broadcast_buffers) is not bool:
        raise ValueError("Invalid DDP buffer synchronization preference")
    device = next(model.parameters()).device
    if any(
        b.device != device or b.layout != torch.strided for b in model.module.buffers()
    ):
        raise ValueError("Task consolidation requires colocated dense buffers")
    return {
        "protocol": "stateset-native-ddp-task-memory-v1",
        "task": task_id,
        "config": _canonical(asdict(manager.config)),
        "preprocessing": {
            k: getattr(manager.training_config, k, None)
            for k in ("max_prompt_length", "max_completion_length")
        },
        "model": model_identity(model.module),
        "tokenizer": _tokenizer_identity(getattr(agent, "tokenizer", None)),
        "broadcast_buffers": model.broadcast_buffers,
        "prior_ewc": state_fingerprint([manager._ewc_fisher, manager._ewc_params]),
        "policy": state_fingerprint(dict(model.named_parameters())),
    }


def _local(
    manager: Any, agent: Any, task: Any, groups: Any, torch: Any
) -> tuple[Any, Any, Any]:
    from .continual_learning import _flatten_trajectories

    staged = copy(manager)
    staged.config = deepcopy(manager.config)
    staged._rng = deepcopy(manager._rng)
    staged.buffer = copy(manager.buffer)
    staged.buffer._rng = deepcopy(manager.buffer._rng)
    staged._task_boundary = None
    policy = agent.model.module
    proxy = SimpleNamespace(model=policy, tokenizer=getattr(agent, "tokenizer", None))
    if staged.config.uses_lwf():
        staged._snapshot_reference_model(proxy)
    fisher, anchors = {}, {}
    staged.last_fisher_estimate = dict.fromkeys(_REPORT, 0)
    if staged.config.uses_ewc():
        samples = list(groups) if groups else []
        if not samples:
            samples = staged.buffer.sample_groups(
                staged.config.ewc_num_samples, task_id=task
            )
        with torch.enable_grad():
            fisher, anchors = staged._estimate_fisher(
                proxy, _flatten_trajectories(samples)
            )
        names = {id(p): name for name, p in agent.model.named_parameters()}
        translated = {name: names[id(p)] for name, p in policy.named_parameters()}
        fisher = {translated[name]: value for name, value in fisher.items()}
        anchors = {translated[name]: value for name, value in anchors.items()}
    return staged, fisher, anchors


def _report(staged: Any) -> list[int]:
    report = staged.last_fisher_estimate
    if (
        set(report) != set(_REPORT)
        or any(type(v) is not int or not 0 <= v < 2**63 for v in report.values())
        or report["failed"] != 0
        or report["attempted"] != report["accepted"] + report["skipped"]
        or report["skipped"] != sum(report[k] for k in _REPORT[3:-1])
    ):
        raise ValueError("Invalid local empirical-Fisher population report")
    return [report[k] for k in _REPORT]


def _population(rows: Any) -> dict[str, int]:
    report = {name: sum(int(row[i]) for row in rows) for i, name in enumerate(_REPORT)}
    if any(value >= 2**63 for value in report.values()):
        raise ValueError("Global empirical-Fisher population overflows int64")
    return report


def _average(group: DDPControlGroup, fisher: Any, accepted: int, total: int) -> None:
    from .grpo_accumulation import _gradient_chunks

    bucket: list[Any] = []
    elements = 0
    maximum = _BUCKET_BYTES // 8

    def flush() -> None:
        nonlocal bucket, elements
        if not bucket:
            return
        packed = group.torch.cat(
            [chunk.reshape(-1).to(group.torch.float64) for chunk in bucket]
        )
        packed.mul_(accepted / total)
        group.dist.all_reduce(packed, group=group.group)
        offset = 0
        for chunk in bucket:
            count = chunk.numel()
            chunk.copy_(packed[offset : offset + count].reshape(chunk.shape))
            offset += count
        bucket, elements = [], 0

    for tensor in fisher.values():
        for chunk in _gradient_chunks(tensor, maximum):
            if elements + chunk.numel() > maximum:
                flush()
            bucket.append(chunk)
            elements += chunk.numel()
    flush()


def _candidate(staged: Any) -> str:
    reference = staged.reference_model
    return state_fingerprint(
        {
            "fisher": staged._ewc_fisher,
            "anchors": staged._ewc_params,
            "reference": (
                None if reference is None else dict(reference.named_parameters())
            ),
            "report": staged.last_fisher_estimate,
        }
    )


def _commit(manager: Any, staged: Any) -> None:
    manager.__dict__.update(staged.__dict__)


def consolidate_ddp_task(
    manager: Any, agent: Any, task_id: Any = None, recent_groups: Any = None
) -> None:
    """Stage global Fisher/teacher memory, preserving live state on rejection.

    Local Fisher means are weighted by accepted examples, including zero on
    empty peers. All participants enter the same operation. Apply/cleanup
    failures poison task memory until a full native checkpoint reload succeeds.
    """
    from .runtime_state import capture_rng, restore_rng

    group = DDPControlGroup(SimpleNamespace(agent=agent), operation="task-memory")
    group.run("admission", lambda: _identity(manager, agent, task_id), agree=True)
    buffers, rng = group.run(
        "local-state",
        lambda: (
            [(b, b.detach().clone()) for b in agent.model.module.buffers()],
            capture_rng(group.torch),
        ),
    )

    def restore() -> None:
        try:
            with group.torch.no_grad():
                for buffer, saved in buffers:
                    buffer.copy_(saved)
        finally:
            restore_rng(rng, group.torch)

    try:
        if agent.model.broadcast_buffers:
            with group.torch.no_grad():
                for buffer, _ in buffers:
                    group.dist.broadcast(buffer, src=group.ranks[0], group=group.group)
        staged, fisher, anchors = group.run(
            "estimate",
            lambda: _local(manager, agent, task_id, recent_groups, group.torch),
        )
        local_report = group.run("local-population", lambda: _report(staged))
        encoded = group.torch.tensor(
            local_report, dtype=group.torch.int64, device=group.device
        )
        gathered = [group.torch.empty_like(encoded) for _ in range(group.world)]
        group.dist.all_gather(gathered, encoded, group=group.group)
        population = group.run(
            "population",
            lambda: _population([row.cpu().tolist() for row in gathered]),
            agree=True,
        )
        total = population["accepted"]
        if total:

            def allocate() -> None:
                nonlocal fisher, anchors
                parameters = [
                    (n, p) for n, p in agent.model.named_parameters() if p.requires_grad
                ]
                if fisher and set(fisher) != {n for n, _ in parameters}:
                    raise ValueError(
                        "Local Fisher layout differs from trainable parameters"
                    )
                if not fisher:
                    fisher = {
                        n: group.torch.zeros_like(
                            p,
                            dtype=(
                                group.torch.float64
                                if p.dtype == group.torch.float64
                                else group.torch.float32
                            ),
                        )
                        for n, p in parameters
                    }
                    anchors = {
                        n: p.detach().to(dtype=fisher[n].dtype).clone()
                        for n, p in parameters
                    }
                for name, parameter in parameters:
                    dtype = (
                        group.torch.float64
                        if parameter.dtype == group.torch.float64
                        else group.torch.float32
                    )
                    if (
                        fisher[name].shape != parameter.shape
                        or anchors[name].shape != parameter.shape
                        or fisher[name].dtype != dtype
                        or anchors[name].dtype != dtype
                        or fisher[name].device != group.device
                        or anchors[name].device != group.device
                        or fisher[name].layout != group.torch.strided
                        or bool((fisher[name] < 0).any())
                    ):
                        raise ValueError("Invalid local Fisher tensor layout/value")
                fisher = {name: fisher[name] for name, _ in parameters}
                anchors = {name: anchors[name] for name, _ in parameters}
                state_fingerprint([fisher, anchors])

            group.run("allocate", allocate)
            _average(group, fisher, local_report[1], total)
            group.run(
                "consolidate", lambda: staged._consolidate_fisher(fisher, anchors)
            )
        staged.last_local_fisher_estimate = dict(
            zip(_REPORT, local_report, strict=True)
        )
        staged.last_fisher_estimate = population
    finally:
        try:
            group.run("restore-local", restore)
        except BaseException:
            manager._ddp_restore_failed = True
            raise
    digest = group.run("candidate", lambda: _candidate(staged), agree=True)
    manager._ddp_restore_failed = True
    try:
        group.run("commit", lambda: _commit(manager, staged))

        def committed() -> str:
            actual = _candidate(manager)
            if actual != digest:
                raise ValueError(
                    "Committed task memory differs from the prepared candidate"
                )
            return actual

        group.run("committed", committed, agree=True)
    except BaseException:
        manager._ddp_restore_failed = True
        raise
    manager._ddp_restore_failed = False


def _window(trainer: Any, torch: Any) -> dict[str, Any]:
    from .ddp_checkpoint_state import _accumulation
    from .learner_preflight import _optimization

    if trainer.scaler is not None and trainer.scaler.is_enabled():
        from .checkpoint_state import _validate_scaler
        from .learner_preflight import _schedule

        # This is transient update admission, not checkpoint publication.
        # Overflowed scaled banks must reach GradScaler's coherent skip path;
        # checkpoints still reject nonfinite pending gradients in _accumulation.
        _schedule(trainer)
        scaler_state = trainer.scaler.state_dict()
        _validate_scaler(scaler_state)
        state = {
            "steps": trainer._get_grad_accum_steps(),
            "scaler": scaler_state,
            "normalization": trainer._grpo_accumulation,
        }
    else:
        state = _accumulation(trainer, torch)
    normalization = state.get("normalization")
    return {
        "steps": state["steps"],
        "count": trainer._grad_accum_step,
        "global_step": trainer.global_step,
        "scaler": state["scaler"],
        "normalization": (
            None
            if normalization is None
            else {
                key: normalization[key]
                for key in ("measure", "policy_normalizer", "entropy_normalizer")
            }
        ),
        "optimization": _optimization(trainer),
        "optimizer_values": (
            None
            if trainer.optimizer is None
            else state_fingerprint(trainer.optimizer.state_dict())
        ),
    }


def handle_ddp_task(trainer: Any, task_id: str | None, *, final: bool = False) -> None:
    """Agree on task boundaries, flush old updates, then commit task memory.

    A successful pending-window flush remains a committed optimizer update if
    later estimation fails. Prior memory and task identity remain unchanged;
    retries do not repeat that update. Partial flush/commit failures require
    full checkpoint reload. Receipt state prevents duplicate final consolidation.
    """
    group = DDPControlGroup(trainer, operation="task-boundary")

    def admission() -> dict[str, Any]:
        from .continual_learning import (
            ContinualLearningManager,
            _validate_continual_config,
        )
        from .ddp_checkpoint_state import _canonical

        if ddp_learner_poisoned(trainer):
            raise ValueError("Failed learner restore requires full checkpoint reload")
        manager = trainer.continual_manager
        if manager is not None:
            if type(manager) is not ContinualLearningManager:
                raise ValueError(
                    "Native task boundaries require the standard continual manager"
                )
            _validate_continual_config(manager.config)
            if any(
                value is not None and not isinstance(value, str)
                for value in (task_id, trainer._current_task_id)
            ):
                raise ValueError("Invalid task identifier")
        return {
            "final": final,
            "current": trainer._current_task_id if manager is not None else None,
            "next": task_id if manager is not None else None,
            "memory": None if manager is None else _canonical(asdict(manager.config)),
        }

    group.run("admission", admission, agree=True)
    manager = trainer.continual_manager
    active = manager is not None and manager.enabled
    if not final and (not active or task_id is None):
        return
    if not final and trainer._current_task_id is None:
        trainer._current_task_id = task_id
        return
    if not final and task_id == trainer._current_task_id:
        return
    if active:
        group.run(
            "memory",
            lambda: _identity(manager, trainer.agent, trainer._current_task_id),
            agree=True,
        )
    group.run("window", lambda: _window(trainer, group.torch), agree=True)
    pending = bool(trainer._grad_accum_step % trainer._get_grad_accum_steps())
    if pending:
        trainer._ddp_restore_failed = True
        group.run("flush", lambda: trainer._flush_accumulation(group.torch), agree=True)
        group.run(
            "flushed-values",
            lambda: state_fingerprint(
                [
                    dict(trainer.agent.model.named_parameters()),
                    trainer.optimizer.state_dict(),
                    (
                        None
                        if trainer.lr_scheduler is None
                        else trainer.lr_scheduler.state_dict()
                    ),
                ]
            ),
            agree=True,
        )
        trainer._ddp_restore_failed = False
    trainer._grad_accum_step = 0
    if not active or trainer._current_task_id is None or task_id is None:
        return
    receipt = trainer._task_boundary_receipt()
    completed = group.torch.tensor(
        [int(manager._task_boundary == receipt)],
        dtype=group.torch.int64,
        device=group.device,
    )
    peers = [group.torch.empty_like(completed) for _ in range(group.world)]
    group.dist.all_gather(peers, completed, group=group.group)
    if not all(row.item() for row in peers):
        consolidate_ddp_task(manager, trainer.agent, trainer._current_task_id)
        manager._task_boundary = receipt
    trainer.reference_model = (
        manager.reference_model
        if manager.reference_model is not None
        else trainer.reference_model
    )
    if not final:
        trainer._current_task_id = task_id
    trainer._ddp_replica_consensus_pending = True
