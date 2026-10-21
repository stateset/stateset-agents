"""Sample-weighted evaluation and shared checkpoint decisions for native DDP."""

from __future__ import annotations

import math
from typing import Any, cast

from .ddp_control import DDPControlGroup, ddp_learner_poisoned
from .state_fingerprint import state_fingerprint

DDP_EVALUATION_PROTOCOL = "stateset-native-ddp-evaluation-v1"
_FIELDS = {
    "eval_reward",
    "eval_reward_std",
    "eval_episode_length",
    "eval_success_rate",
    "eval_num_episodes",
    "eval_attempted_episodes",
    "eval_failed_episodes",
    "eval_error_rate",
    "eval_complete",
}
_MAX_COUNT = 2**53 - 1


def _pack(metrics: Any) -> tuple[list[int], list[float]]:
    from .evaluation import _EvaluationMetrics

    if not isinstance(metrics, dict) or set(metrics) != _FIELDS:
        raise ValueError("Invalid native evaluation metric schema")
    if any(
        type(v) not in (int, float) or not math.isfinite(v) for v in metrics.values()
    ):
        raise ValueError("Native evaluation metrics must be finite numbers")
    counts = []
    for name in (
        "eval_num_episodes",
        "eval_attempted_episodes",
        "eval_failed_episodes",
    ):
        value = metrics[name]
        if not 0 <= value <= _MAX_COUNT or int(value) != value:
            raise ValueError("Invalid native evaluation count")
        counts.append(int(value))
    valid, attempted, failed = counts
    if isinstance(metrics, _EvaluationMetrics):
        successes = metrics.successes
        if type(successes) is not int:
            raise ValueError("Invalid native evaluation success count")
    else:
        # Division followed by multiplication can lose an integer near 2**53.
        # Ordinary dictionaries are supported only within a range where the
        # nearest integer reliably recovers an actual count. Native evaluation
        # carries the exact count separately without adding public JSON fields.
        if attempted > 2**31 - 1:
            raise ValueError("Large evaluation requires exact native count statistics")
        successes = round(metrics["eval_success_rate"] * attempted)
    if (
        valid + failed != attempted
        or not 0 <= successes <= valid
        or not 0 <= metrics["eval_success_rate"] <= 1
        or metrics["eval_success_rate"] != (successes / attempted if attempted else 0)
        or metrics["eval_error_rate"] != (failed / attempted if attempted else 0)
        or metrics["eval_complete"] != float(valid == attempted and attempted > 0)
        or metrics["eval_reward_std"] < 0
        or valid < 2
        and metrics["eval_reward_std"] != 0
        or metrics["eval_episode_length"] < 0
    ):
        raise ValueError("Inconsistent native evaluation metrics")
    values = [
        float(metrics[name])
        for name in ("eval_reward", "eval_reward_std", "eval_episode_length")
    ]
    if not valid and any(values):
        raise ValueError("Empty native evaluation must have zero moments")
    return [*counts, successes], values


def _aggregate(rows: list[tuple[list[int], list[float]]]) -> dict[str, float]:
    from .evaluation import _EvaluationMetrics

    valid, attempted, failed, successes = [
        sum(row[0][i] for row in rows) for i in range(4)
    ]
    if attempted > _MAX_COUNT:
        raise ValueError("Global evaluation count exceeds exact float reporting range")
    scale = max((max(abs(v[0]), v[1]) for c, v in rows if c[0]), default=0.0)
    normalized_mean = (
        math.fsum((v[0] / scale) * (c[0] / valid) for c, v in rows if c[0])
        if scale
        else 0.0
    )
    normalized_variance = (
        math.fsum(
            ((v[1] / scale) ** 2 + (v[0] / scale - normalized_mean) ** 2)
            * (c[0] / valid)
            for c, v in rows
            if c[0]
        )
        if scale
        else 0.0
    )
    result = {
        "eval_reward": normalized_mean * scale,
        "eval_reward_std": math.sqrt(normalized_variance) * scale,
        "eval_episode_length": math.fsum(
            v[2] * (c[0] / valid) for c, v in rows if c[0]
        ),
        "eval_success_rate": successes / attempted if attempted else 0.0,
        "eval_num_episodes": float(valid),
        "eval_attempted_episodes": float(attempted),
        "eval_failed_episodes": float(failed),
        "eval_error_rate": failed / attempted if attempted else 0.0,
        "eval_complete": float(valid == attempted and attempted > 0),
    }
    receipt = _EvaluationMetrics(result, successes=successes)
    _pack(receipt)
    return receipt


def aggregate_evaluation_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    """Combine validated local moments using actual valid and attempted counts."""
    if not rows:
        raise ValueError("Native evaluation requires at least one participant")
    return _aggregate([_pack(row) for row in rows])


def _admission(trainer: Any, episodes: int) -> dict[str, Any]:
    import torch

    from stateset_agents.evaluation.isolation import _validate_seed

    from .ddp_checkpoint_state import _tokenizer_identity, model_identity

    if ddp_learner_poisoned(trainer):
        raise ValueError("A failed native restore must succeed before evaluation")
    if type(episodes) is not int or episodes < 0:
        raise ValueError("num_eval_episodes must be a nonnegative integer")
    best = trainer.best_eval_metric
    if type(best) not in (int, float) or math.isnan(best):
        raise ValueError("Invalid best evaluation metric")
    patience = trainer.steps_without_improvement
    if type(patience) is not int or not 0 <= patience < 2**63:
        raise ValueError("Invalid evaluation patience counter")
    model = trainer.agent.model
    if type(model.broadcast_buffers) is not bool:
        raise ValueError("Invalid DDP buffer synchronization preference")
    device = next(model.parameters()).device
    for buffer in model.module.buffers():
        if buffer.device != device or buffer.layout != torch.strided:
            raise ValueError("Native evaluation requires colocated dense buffers")
    seed = getattr(trainer.config, "seed", 42)
    if seed is None:
        seed = 42
    _validate_seed(seed)
    counters = {}
    for name in ("global_step", "_grad_accum_step"):
        value = getattr(trainer, name)
        if type(value) is not int or not 0 <= value < 2**63:
            raise ValueError("Invalid evaluation learner counter")
        counters[name] = value
    return {
        "protocol": DDP_EVALUATION_PROTOCOL,
        "episodes": episodes,
        "seed": seed,
        "generations": getattr(trainer.config, "eval_num_generations", 4),
        "max_turns": getattr(trainer.config, "max_steps_per_episode", None),
        "concurrency": getattr(trainer.config, "rollout_concurrency", 1),
        "best": float(best).hex(),
        "patience": patience,
        "learner_counters": counters,
        "broadcast_buffers": model.broadcast_buffers,
        "model": model_identity(model.module),
        "tokenizer": _tokenizer_identity(getattr(trainer.agent, "tokenizer", None)),
    }


async def evaluate_ddp(trainer: Any, scenarios: Any, episodes: int) -> dict[str, float]:
    """Evaluate uneven local partitions and select one shared best checkpoint.

    Inference uses the unwrapped policy so local forward counts need not match.
    Buffer broadcasts follow the wrapper preference once at entry; local buffer
    values, RNGs and module modes are restored before checkpoint selection.
    Every live peer must participate; collective timeouts must accommodate the
    slowest partition. Arbitrary custom callback collectives are outside this
    protocol.
    """
    group = DDPControlGroup(trainer, operation="evaluation")
    group.run("admission", lambda: _admission(trainer, episodes), agree=True)
    group.run(
        "policy",
        lambda: state_fingerprint(dict(trainer.agent.model.module.named_parameters())),
        agree=True,
    )
    wrapper = trainer.agent.model
    buffers = group.run(
        "buffers", lambda: [(b, b.detach().clone()) for b in wrapper.module.buffers()]
    )

    def restore_buffers() -> None:
        with group.torch.no_grad():
            for buffer, saved in buffers:
                buffer.copy_(saved)

    try:
        if wrapper.broadcast_buffers:
            with group.torch.no_grad():
                for buffer, _ in buffers:
                    group.dist.broadcast(buffer, src=group.ranks[0], group=group.group)

        async def local() -> dict[str, float]:
            trainer.agent.model = wrapper.module
            try:
                result: dict[str, float] = await trainer._evaluate_local(
                    scenarios, episodes
                )
                return result
            finally:
                trainer.agent.model = wrapper

        metrics = await group.arun("local", local)
    finally:
        try:
            group.run("buffer-restore", restore_buffers)
        except BaseException:
            # A failed restoration can leave a worker's buffers partly applied.
            # Reuse checkpoint admission's poison until a full reload succeeds.
            trainer._ddp_restore_failed = True
            raise
    counts, values = group.run("statistics", lambda: _pack(metrics))
    integers = group.torch.tensor(counts, dtype=group.torch.int64, device=group.device)
    floats = group.torch.tensor(values, dtype=group.torch.float64, device=group.device)
    integer_rows = [group.torch.empty_like(integers) for _ in range(group.world)]
    float_rows = [group.torch.empty_like(floats) for _ in range(group.world)]
    group.dist.all_gather(integer_rows, integers, group=group.group)
    group.dist.all_gather(float_rows, floats, group=group.group)
    global_metrics = group.run(
        "aggregate",
        lambda: _aggregate(
            [
                (c.cpu().tolist(), v.cpu().tolist())
                for c, v in zip(integer_rows, float_rows, strict=True)
            ]
        ),
        agree=True,
    )
    await trainer._update_best_checkpoint(global_metrics)
    return cast(dict[str, float], global_metrics)
