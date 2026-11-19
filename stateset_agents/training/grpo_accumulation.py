"""Count-weighted native GRPO gradients without retaining microbatch graphs."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .metric_reduction import LossMetricReduction

_DDP_BUCKET_BYTES = 25 * 1024**2


def _gradient_chunks(gradient: Any, max_elements: int) -> Iterator[Any]:
    """Yield bounded views, including noncontiguous and oversized gradients."""
    if gradient.numel() <= max_elements:
        yield gradient
        return
    axis = next(index for index, size in enumerate(gradient.shape) if size > 1)
    width = gradient.shape[axis]
    per_index = gradient.numel() // width
    chunk = max(1, max_elements // per_index)
    for start in range(0, width, chunk):
        yield from _gradient_chunks(
            gradient.narrow(axis, start, min(chunk, width - start)), max_elements
        )


def _ddp_counts(
    trainer: Any,
    torch: Any,
    policy: int,
    entropy: int,
    *,
    measure: str,
    separate_entropy: bool,
) -> tuple[int, int, int]:
    """Agree on the measure and sum counts on DDP's own process group."""
    from torch.nn.parallel import DistributedDataParallel

    model = trainer.agent.model
    if not isinstance(model, DistributedDataParallel):
        return policy, entropy, 1
    import torch.distributed as dist

    valid = (
        type(policy) is int
        and 0 <= policy <= 2**63 - 1
        and type(entropy) is int
        and 0 <= entropy <= 2**63 - 1
        and (bool(entropy) == separate_entropy or (policy == 0 and entropy == 0))
        and (policy > 0 or entropy == 0)
        and measure in {"trajectory", "token"}
    )
    counts = torch.tensor(
        [
            int(valid),
            policy if valid else 0,
            entropy if valid else 0,
            int(measure == "trajectory"),
            int(separate_entropy),
        ],
        dtype=torch.int64,
        device=next(model.parameters()).device,
    )
    world_size = dist.get_world_size(model.process_group)
    gathered = [torch.empty_like(counts) for _ in range(world_size)]
    dist.all_gather(gathered, counts, group=model.process_group)
    decoded = [row.cpu().tolist() for row in gathered]
    if not all(row[0] for row in decoded):
        raise ValueError("Invalid GRPO normalization counts on a DDP rank")
    if len({tuple(row[3:]) for row in decoded}) != 1:
        raise ValueError(
            "GRPO normalization measure or entropy differs across DDP ranks"
        )
    if not sum(row[1] for row in decoded):
        raise ValueError("GRPO population is globally empty")
    return sum(row[1] for row in decoded), sum(row[2] for row in decoded), world_size


def _reduce_bank(
    pieces: list[Any],
    factor: tuple[int, int],
    torch: Any,
    dist: Any,
    group: Any,
    ranks: int,
) -> tuple[Any, Any]:
    """Return the weighted rank sum and its optional binary64 power-of-two unit."""
    source_dtype = pieces[0].dtype
    single = source_dtype == torch.float16 or pieces[0].device.type == "mps"
    dtype = torch.float32 if single else torch.float64
    values = torch.cat([piece.reshape(-1) for piece in pieces]).to(dtype)
    numerator, denominator = factor
    lo = hi = headroom_scale = units = None
    if ranks > 1 and source_dtype == torch.float64:
        # There is no wider collective for binary64. Scale just the coordinates
        # that could overflow in a SUM, by the same exact power of two on all ranks.
        lo, hi = values.clone(), values.clone()
        dist.all_reduce(lo, op=dist.ReduceOp.MIN, group=group)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX, group=group)
        headroom = 1 << (ranks - 1).bit_length()
        large = torch.maximum(lo.abs(), hi.abs()) > torch.finfo(dtype).max / headroom
        headroom_scale = torch.ones_like(values).masked_fill_(large, headroom)
        values.div_(headroom_scale)
    if ranks > 1:
        dist.all_reduce(values, group=group)
    if source_dtype == torch.float64:
        # Preserve a fractional subnormal, or a temporarily out-of-range term,
        # until policy and entropy have been combined. Select each coordinate's
        # unit after summation so an exactly cancelled bank cannot hide the other.
        magnitude = values.abs()
        units = torch.ones_like(values)
        units.masked_fill_(magnitude <= 2.0**-512, 2.0**-512)
        units.masked_fill_(magnitude >= 2.0**512, 2.0**512)
        values.div_(units)
    if numerator == 1:
        values.div_(float(denominator))
    else:
        values.mul_(numerator / denominator)
    if headroom_scale is not None:
        assert lo is not None and hi is not None and units is not None
        values.mul_(headroom_scale)
        # The true weighted sum is inside these bounds. Projection preserves
        # a replicated finite coordinate even for non-power-of-two rank counts.
        weight = (ranks * numerator) / denominator
        values.clamp_(min=(lo / units) * weight, max=(hi / units) * weight)
    return values, units


def _local_sparse_gradient(
    policy: Any,
    entropy: Any,
    factors: tuple[tuple[int, int], tuple[int, int]],
    torch: Any,
) -> Any:
    """Combine sparse embedding gradients without allocating the vocabulary."""
    source = policy if policy is not None else entropy
    wide = torch.complex128 if source.is_complex() else torch.float64
    if source.device.type == "mps":
        wide = torch.complex64 if source.is_complex() else torch.float32
    terms = [
        grad.to(wide).mul(n / d)
        for grad, (n, d) in zip((policy, entropy), factors, strict=True)
        if grad is not None
    ]
    combined = terms[0]
    if len(terms) == 2:
        # PyTorch mixed-layout addition takes the dense operand first.
        other = terms[1]
        combined = (
            other + combined if other.layout == torch.strided else combined + other
        )
    if combined.is_sparse:
        combined = combined.coalesce()
    return combined.to(source.dtype)


def _average_ddp_banks(
    trainer: Any,
    state: dict[str, Any],
    torch: Any,
    *,
    population_factors: tuple[tuple[int, int], tuple[int, int]] | None = None,
) -> None:
    """Average raw banks, or finalize both population-weighted banks in one cast."""
    from itertools import repeat

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    model = trainer.agent.model
    group = None
    ranks = 1
    if isinstance(model, DistributedDataParallel):
        group = model.process_group
        ranks = dist.get_world_size(group)
    if ranks == 1:
        if population_factors is None:
            return
        (numerator, denominator), _ = population_factors
        if numerator == denominator and not state["entropy_gradients"]:
            return

    entropy_bank = state["entropy_gradients"]
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    active = [
        [p.grad is not None, entropy_bank.get(name) is not None]
        for name, p in parameters
    ]
    if ranks > 1:
        presence = torch.tensor(
            active, dtype=torch.int64, device=next(model.parameters()).device
        )
        dist.all_reduce(presence, op=dist.ReduceOp.MAX, group=group)
        active = presence.cpu().tolist()

    policy_factor, entropy_factor = population_factors or ((1, ranks), (1, ranks))
    policy_bucket: list[Any] = []
    entropy_bucket: list[Any] = []
    bucket_key = None
    bucket_bytes = 0

    def flush() -> None:
        policy, policy_unit = (
            _reduce_bank(policy_bucket, policy_factor, torch, dist, group, ranks)
            if policy_bucket
            else (None, None)
        )
        entropy, entropy_unit = (
            _reduce_bank(entropy_bucket, entropy_factor, torch, dist, group, ranks)
            if entropy_bucket
            else (None, None)
        )
        if population_factors is not None and entropy is not None:
            assert policy is not None
            if policy_unit is not None:
                common = torch.maximum(policy_unit, entropy_unit)
                policy.mul_(policy_unit / common)
                entropy.mul_(entropy_unit / common)
                policy_unit = common
            policy.add_(entropy)
            entropy = None
        for pieces, result, unit in (
            (policy_bucket, policy, policy_unit),
            (entropy_bucket, entropy, entropy_unit),
        ):
            if result is not None and unit is not None:
                result.mul_(unit)
            offset = 0
            if result is not None:
                for piece in pieces:
                    size = piece.numel()
                    piece.copy_(result[offset : offset + size].view_as(piece))
                    offset += size
            pieces.clear()

    with torch.no_grad():
        for (name, parameter), (has_policy, has_entropy) in zip(
            parameters, active, strict=True
        ):
            if not (has_policy or has_entropy):
                continue
            policy = parameter.grad
            entropy = entropy_bank.get(name)
            if (
                ranks == 1
                and population_factors is not None
                and any(
                    g is not None and g.layout != torch.strided
                    for g in (policy, entropy)
                )
            ):
                parameter.grad = _local_sparse_gradient(
                    policy, entropy, population_factors, torch
                )
                continue
            # A normalized update needs a policy destination even when the only
            # contribution on any rank comes from entropy. Missing ranks add zero.
            if (has_policy or population_factors is not None) and policy is None:
                policy = parameter.grad = torch.zeros_like(parameter)
            if has_entropy and entropy is None:
                entropy = entropy_bank[name] = torch.zeros_like(parameter)
            key = (
                parameter.device,
                parameter.dtype,
                policy is not None,
                entropy is not None,
            )
            # Treat complex gradients componentwise, also for signed extrema.
            if parameter.is_complex():
                if policy is not None:
                    policy = parameter.grad = policy.resolve_conj().resolve_neg()
                    policy = torch.view_as_real(policy)
                if entropy is not None:
                    entropy = entropy_bank[name] = entropy.resolve_conj().resolve_neg()
                    entropy = torch.view_as_real(entropy)
            component = policy if policy is not None else entropy
            element_bytes = 4 if component.dtype == torch.float16 else 8
            elements = max(1, _DDP_BUCKET_BYTES // element_bytes)
            policy_parts = (
                _gradient_chunks(policy, elements)
                if policy is not None
                else repeat(None)
            )
            entropy_parts = (
                _gradient_chunks(entropy, elements)
                if entropy is not None
                else repeat(None)
            )
            for policy_part, entropy_part in zip(
                policy_parts, entropy_parts, strict=False
            ):
                part = policy_part if policy_part is not None else entropy_part
                size = part.numel() * element_bytes
                if bucket_key != key or bucket_bytes + size > _DDP_BUCKET_BYTES:
                    flush()
                    bucket_bytes = 0
                    bucket_key = key
                if policy_part is not None:
                    policy_bucket.append(policy_part)
                if entropy_part is not None:
                    entropy_bucket.append(entropy_part)
                bucket_bytes += size
        flush()


class NormalizedLoss(dict[str, Any]):
    """Loss metrics with private differentiable terms for trainer accumulation.

    Dictionary contents remain ordinary loss metrics. Separate terms preserve
    the policy's trajectory/token denominator and entropy's token denominator.
    """

    def __init__(
        self,
        metrics: dict[str, Any],
        *,
        policy_mean: Any,
        policy_count: int,
        measure: str,
        entropy_loss: Any = None,
        token_count: int = 0,
        metric_reduction: LossMetricReduction | None = None,
    ) -> None:
        super().__init__(metrics)
        self.policy_mean = policy_mean
        self.policy_count = policy_count
        self.measure = measure
        self.entropy_loss = entropy_loss
        self.token_count = token_count
        self.metric_reduction = metric_reduction


def backward_accumulated_loss(
    trainer: Any,
    loss: Any,
    metrics: dict[str, Any],
    torch: Any,
    *,
    scaler: Any = None,
    regularizer: Any = None,
    accumulation_steps: int | None = None,
) -> None:
    """Accumulate sums at fixed scales, releasing each microbatch's graph.

    Legacy/custom scalar windows retain their original microbatch mean. A
    restored legacy window finishes under that rule before starting a window
    with the new normalization. Policy and entropy need at most two gradient
    banks, independent of the number of accumulated microbatches.
    """
    from .ddp_population import run_token_local
    from .trainer_utils import _discard_gradients, backward_training_loss

    steps = (
        trainer._get_grad_accum_steps()
        if accumulation_steps is None
        else accumulation_steps
    )
    state = getattr(trainer, "_grpo_accumulation", None)
    from torch.nn.parallel import DistributedDataParallel

    distributed = isinstance(trainer.agent.model, DistributedDataParallel)
    if getattr(trainer.agent.model, "_stateset_token_forward", False) is True:
        from .ddp_control import DDPControlGroup

        def contract() -> dict[str, Any]:
            if (
                not torch.is_tensor(loss)
                or loss.ndim != 0
                or not loss.requires_grad
                or not bool(torch.isfinite(loss))
            ):
                raise ValueError(
                    "Native token loss must be a finite differentiable scalar"
                )
            return {
                "normalized": isinstance(metrics, NormalizedLoss),
                "steps": steps,
                "measure": (
                    metrics.measure if isinstance(metrics, NormalizedLoss) else None
                ),
                "entropy": isinstance(metrics, NormalizedLoss)
                and metrics.entropy_loss is not None,
            }

        DDPControlGroup(trainer, operation="token-loss").run(
            "contract", contract, agree=True
        )
    if steps == 1 and state is None and not distributed:
        backward_training_loss(trainer, loss, torch, scaler=scaler)
        return
    if not isinstance(metrics, NormalizedLoss) or (
        state is None and trainer._grad_accum_step % steps
    ):
        if state is not None:
            _discard_gradients(trainer)
            raise ValueError("Cannot mix normalized and custom GRPO loss windows")
        run_token_local(
            trainer.agent.model,
            "legacy-backward",
            lambda: backward_training_loss(trainer, loss / steps, torch, scaler=scaler),
        )
        return

    parameters = dict(trainer.agent.model.named_parameters())
    separate_entropy = metrics.entropy_loss is not None
    if state is None:
        try:
            policy_count, entropy_count, ranks = _ddp_counts(
                trainer,
                torch,
                metrics.policy_count,
                metrics.token_count if separate_entropy else 0,
                measure=metrics.measure,
                separate_entropy=separate_entropy,
            )
        except BaseException:
            _discard_gradients(trainer)
            raise
        state = {
            "measure": metrics.measure,
            "policy_count": 0,
            # One common scale near the local batch size avoids an extra
            # division by world size in every rank's parameter dtype.
            "policy_normalizer": ((policy_count + ranks - 1) // ranks) * steps,
            "entropy_count": 0,
            "entropy_normalizer": ((entropy_count + ranks - 1) // ranks) * steps,
            "entropy_gradients": dict.fromkeys(parameters) if separate_entropy else {},
        }
        trainer._grpo_accumulation = state
    try:
        if (
            state["measure"] != metrics.measure
            or bool(state["entropy_normalizer"]) != separate_entropy
        ):
            raise ValueError("GRPO objective normalization changed within a window")
        policy = metrics.policy_mean
        if regularizer is not None:
            policy = policy + regularizer

        def policy_backward() -> None:
            backward_training_loss(
                trainer,
                policy * (metrics.policy_count / state["policy_normalizer"]),
                torch,
                scaler=scaler,
                retain_graph=separate_entropy,
            )

        run_token_local(trainer.agent.model, "policy-backward", policy_backward)
        if separate_entropy:
            # Use ordinary backward hooks for both terms. Restore the policy
            # bank even on failure; the outer handler then discards all banks.
            policy_gradients = {name: p.grad for name, p in parameters.items()}
            try:
                for parameter in parameters.values():
                    parameter.grad = None

                def entropy_backward() -> None:
                    backward_training_loss(
                        trainer,
                        metrics.entropy_loss
                        * (metrics.token_count / state["entropy_normalizer"]),
                        torch,
                        scaler=scaler,
                    )

                run_token_local(
                    trainer.agent.model, "entropy-backward", entropy_backward
                )
                for name, parameter in parameters.items():
                    gradient = parameter.grad
                    if gradient is not None:
                        saved = state["entropy_gradients"][name]
                        if saved is None:
                            state["entropy_gradients"][name] = gradient.detach()
                        else:
                            saved.add_(gradient)
            finally:
                for name, parameter in parameters.items():
                    parameter.grad = policy_gradients[name]
            state["entropy_count"] += metrics.token_count
        state["policy_count"] += metrics.policy_count
    except BaseException:
        _discard_gradients(trainer)
        raise


def finalize_accumulated_gradients(trainer: Any, torch: Any) -> bool:
    """Merge normalized sums before AMP overflow detection and gradient clipping."""
    state = getattr(trainer, "_grpo_accumulation", None)
    if state is None:
        from torch.nn.parallel import DistributedDataParallel

        if isinstance(trainer.agent.model, DistributedDataParallel):
            _average_ddp_banks(
                trainer, {"entropy_gradients": {}, "entropy_normalizer": 0}, torch
            )
        return False
    policy_count, entropy_count, _ = _ddp_counts(
        trainer,
        torch,
        state["policy_count"],
        state["entropy_count"],
        measure=state["measure"],
        separate_entropy=bool(state["entropy_normalizer"]),
    )
    _average_ddp_banks(
        trainer,
        state,
        torch,
        population_factors=(
            (state["policy_normalizer"], policy_count),
            (state["entropy_normalizer"], entropy_count or 1),
        ),
    )
    return True
