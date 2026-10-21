"""Empty contributors and local token forwards for native DDP gradient banks."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, TypeVar, cast

T = TypeVar("T")


def run_token_local(model: Any, phase: str, function: Callable[[], T]) -> T:
    """Notify every token-bank worker before leaving local loss/backward work."""
    if getattr(model, "_stateset_token_forward", False) is True:
        from .ddp_control import DDPControlGroup

        return cast(
            T,
            DDPControlGroup(
                SimpleNamespace(agent=SimpleNamespace(model=model)),
                operation="token-loss",
            ).run(phase, function),
        )
    return function()


def token_policy(model: Any) -> Any:
    """Use the unwrapped actor only inside native token-bank training."""
    return (
        model.module
        if getattr(model, "_stateset_token_forward", False) is True
        else model
    )


@contextmanager
def token_forward_context(model: Any, source: str) -> Iterator[None]:
    """Keep DDP metadata/parameters public while forwarding without its reducer."""
    missing = object()
    names = {"_stateset_token_forward": True, "_stateset_rollout_source": source}
    previous = {name: getattr(model, name, missing) for name in names}
    try:
        for name, value in names.items():
            setattr(model, name, value)
        yield
    finally:
        for name, value in previous.items():
            if value is missing:
                delattr(model, name)
            else:
                setattr(model, name, value)


def empty_batch_metrics(trainer: Any) -> dict[str, Any]:
    """Report a global no-op without consuming an accumulation or scheduler step."""
    from .metric_reduction import METRIC_REDUCTION_PROTOCOL

    return {
        **dict.fromkeys(
            (
                "total_loss",
                "policy_loss",
                "policy_objective_loss",
                "entropy",
                "kl_penalty",
                "clip_fraction",
                "mean_advantage",
                "advantage_std",
                "nonzero_advantage_fraction",
            ),
            0.0,
        ),
        **dict.fromkeys(
            (
                "num_trajectories",
                "num_rows",
                "num_tokens",
                "policy_denominator",
                "optimizer_updates",
                "inner_updates",
            ),
            0,
        ),
        "ratio_mean": 1.0,
        "global_empty_batch": True,
        "entropy_measured": False,
        "optimizer_step": False,
        "global_step": trainer.global_step,
        "grad_accum_step": trainer._grad_accum_step,
        "learning_rate": (
            float(trainer.optimizer.param_groups[0]["lr"])
            if trainer.optimizer is not None
            else 0.0
        ),
        "metric_scope": "ddp_global_batch",
        "metric_reduction_protocol": METRIC_REDUCTION_PROTOCOL,
    }


def native_empty_contributor(trainer: Any) -> bool:
    """Identify the native DDP contract that permits zero local pending counts."""
    from torch.nn.parallel import DistributedDataParallel

    return isinstance(trainer.agent.model, DistributedDataParallel)
