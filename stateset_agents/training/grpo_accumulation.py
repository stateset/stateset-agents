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


def _average_ddp_banks(trainer: Any, state: dict[str, Any], torch: Any) -> None:
    """Reduce both banks, including entropy's second backward and no-sync work.

    DDP's reducer can finish after the first backward on a retained graph.
    Its policy bank may already be averaged; averaging it again is idempotent.
    Entropy and pending no-sync gradients still need their complete reduction.
    A presence vector lets a parameter contribute on only some ranks. Buckets
    bound temporary storage without issuing a collective per adapter tensor.
    """
    from torch.nn.parallel import DistributedDataParallel

    model = trainer.agent.model
    if not isinstance(model, DistributedDataParallel):
        return
    import torch.distributed as dist

    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    entries = [
        (parameter, bank, name)
        for bank in (None, state["entropy_gradients"])
        for name, parameter in parameters
        if bank is None or state["entropy_normalizer"]
    ]
    presence = torch.tensor(
        [
            int((p.grad if bank is None else bank[name]) is not None)
            for p, bank, name in entries
        ],
        dtype=torch.int64,
        device=next(model.parameters()).device,
    )
    dist.all_reduce(presence, group=model.process_group)
    active = presence.cpu().tolist()
    bucket: list[Any] = []
    bucket_bytes = 0
    world_size = dist.get_world_size(model.process_group)

    def flush() -> None:
        if not bucket:
            return
        dtype = bucket[0].dtype
        if dtype in {torch.float16, torch.bfloat16}:
            dtype = torch.float32
        flattened = torch.cat(
            [gradient.reshape(-1).to(dtype=dtype) for gradient in bucket]
        )
        # Prescale to avoid overflowing a sum whose final mean is finite.
        # Half-precision banks use FP32 so prescaling preserves subnormals.
        flattened.div_(world_size)
        dist.all_reduce(flattened, group=model.process_group)
        offset = 0
        for gradient in bucket:
            size = gradient.numel()
            gradient.copy_(flattened[offset : offset + size].view_as(gradient))
            offset += size
        bucket.clear()

    with torch.no_grad():
        for (parameter, bank, name), used in zip(entries, active, strict=True):
            if not used:
                continue
            gradient = parameter.grad if bank is None else bank[name]
            if gradient is None:
                gradient = torch.zeros_like(parameter)
                if bank is None:
                    parameter.grad = gradient
                else:
                    bank[name] = gradient
            for piece in _gradient_chunks(
                gradient, _DDP_BUCKET_BYTES // max(4, gradient.element_size())
            ):
                size = piece.numel() * max(4, piece.element_size())
                if bucket and (
                    bucket_bytes + size > _DDP_BUCKET_BYTES
                    or piece.dtype != bucket[0].dtype
                    or piece.device != bucket[0].device
                ):
                    flush()
                    bucket_bytes = 0
                bucket.append(piece)
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


def _ensure_accumulation_window(
    trainer: Any, metrics: NormalizedLoss, torch: Any, steps: int
) -> None:
    """Agree on complete batch counts before any rank enters its local chunks."""
    from torch.nn.parallel import DistributedDataParallel

    distributed = isinstance(trainer.agent.model, DistributedDataParallel)
    if getattr(trainer, "_grpo_accumulation", None) is not None:
        return
    if (steps == 1 and not distributed) or trainer._grad_accum_step % steps:
        return
    policy_count, entropy_count, _ = _ddp_counts(
        trainer,
        torch,
        metrics.policy_count,
        metrics.token_count if metrics.entropy_loss is not None else 0,
        measure=metrics.measure,
        separate_entropy=metrics.entropy_loss is not None,
    )
    trainer._grpo_accumulation = {
        "measure": metrics.measure,
        "policy_count": 0,
        "policy_normalizer": policy_count * steps,
        "entropy_count": 0,
        "entropy_normalizer": (
            entropy_count * steps if metrics.entropy_loss is not None else 0
        ),
        "entropy_gradients": (
            dict.fromkeys(dict(trainer.agent.model.named_parameters()))
            if metrics.entropy_loss is not None
            else {}
        ),
    }


class NativeChunkedBackward:
    """Backpropagate local chunks with one collective count agreement per batch.

    Ranks may have different numbers of chunks or no local rows. Their local
    backwards contain no collectives; the caller coordinates failures after
    the complete local phase. Outer accumulation records batch counts once.
    """

    def __init__(
        self, trainer: Any, torch: Any, *, steps: int, scaler: Any = None
    ) -> None:
        self.trainer, self.torch = trainer, torch
        self.steps, self.scaler = steps, scaler

    def prepare(
        self,
        *,
        policy_count: int,
        token_count: int,
        measure: str,
        separate_entropy: bool,
    ) -> None:
        if getattr(self.trainer.agent.model, "_stateset_token_forward", False) is True:
            from .ddp_control import DDPControlGroup

            DDPControlGroup(self.trainer, operation="token-loss").run(
                "chunk-contract",
                lambda: {
                    "steps": self.steps,
                    "measure": measure,
                    "entropy": separate_entropy,
                    "pending": self.trainer._grad_accum_step,
                    "window": getattr(self.trainer, "_grpo_accumulation", None)
                    is not None,
                },
                agree=True,
            )
        metrics = NormalizedLoss(
            {},
            policy_mean=None,
            policy_count=policy_count,
            measure=measure,
            entropy_loss=True if separate_entropy else None,
            token_count=token_count,
        )
        _ensure_accumulation_window(self.trainer, metrics, self.torch, self.steps)
        if self.scaler is not None and self.scaler.is_enabled():
            # Empty contributors still receive reduced gradients at the flush.
            # Initialize GradScaler through its public API without inventing a
            # local forward, backward, optimizer update or normalization count.
            device = next(self.trainer.agent.model.parameters()).device
            self.scaler.scale(self.torch.zeros((), device=device))

    def backward_normalized(self, metrics: NormalizedLoss) -> None:
        device = next(self.trainer.agent.model.parameters()).device.type
        with self.torch.autocast(device_type=device, enabled=False):
            backward_accumulated_loss(
                self.trainer,
                metrics["total_loss"],
                metrics,
                self.torch,
                scaler=self.scaler,
                accumulation_steps=self.steps,
                local_chunk=True,
                record_counts=False,
            )

    def __call__(self, share: Any) -> None:
        """Retain the scalar callback contract for explicitly custom losses."""
        from .trainer_utils import backward_training_loss

        backward_training_loss(
            self.trainer, share / self.steps, self.torch, scaler=self.scaler
        )


def backward_accumulated_loss(
    trainer: Any,
    loss: Any,
    metrics: dict[str, Any],
    torch: Any,
    *,
    scaler: Any = None,
    regularizer: Any = None,
    accumulation_steps: int | None = None,
    local_chunk: bool = False,
    record_counts: bool = True,
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
    streamed = (
        isinstance(metrics, NormalizedLoss) and metrics.get("backward_done") is True
    )
    if (
        not local_chunk
        and getattr(trainer.agent.model, "_stateset_token_forward", False) is True
    ):
        from .ddp_control import DDPControlGroup

        def contract() -> dict[str, Any]:
            if (
                not torch.is_tensor(loss)
                or loss.ndim != 0
                or (not loss.requires_grad and not streamed)
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

    def run(phase: str, function: Any) -> Any:
        return (
            function()
            if local_chunk
            else run_token_local(trainer.agent.model, phase, function)
        )

    if streamed:
        assert isinstance(metrics, NormalizedLoss)
        if state is not None:
            if state["measure"] != metrics.measure or bool(
                state["entropy_normalizer"]
            ) != (metrics.entropy_loss is not None):
                raise ValueError("GRPO objective normalization changed within a window")
            if regularizer is not None:
                run(
                    "regularizer-backward",
                    lambda: backward_training_loss(
                        trainer,
                        regularizer
                        * (metrics.policy_count / state["policy_normalizer"]),
                        torch,
                        scaler=scaler,
                    ),
                )
            state["policy_count"] += metrics.policy_count
            if metrics.entropy_loss is not None:
                state["entropy_count"] += metrics.token_count
        elif regularizer is not None:
            run(
                "regularizer-backward",
                lambda: backward_training_loss(
                    trainer,
                    regularizer / steps,
                    torch,
                    scaler=scaler,
                ),
            )
        return
    if steps == 1 and state is None and not distributed:
        backward_training_loss(trainer, loss, torch, scaler=scaler)
        return
    if not isinstance(metrics, NormalizedLoss) or (
        state is None and trainer._grad_accum_step % steps
    ):
        if state is not None:
            _discard_gradients(trainer)
            raise ValueError("Cannot mix normalized and custom GRPO loss windows")
        run(
            "legacy-backward",
            lambda: backward_training_loss(trainer, loss / steps, torch, scaler=scaler),
        )
        return

    parameters = dict(trainer.agent.model.named_parameters())
    separate_entropy = metrics.entropy_loss is not None
    if state is None:
        if local_chunk:
            raise RuntimeError(
                "Chunked normalization must be prepared before local backward"
            )
        try:
            policy_count, entropy_count, _ = _ddp_counts(
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
            "policy_normalizer": policy_count * steps,
            "entropy_count": 0,
            "entropy_normalizer": (entropy_count * steps if separate_entropy else 0),
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

        run("policy-backward", policy_backward)
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

                run("entropy-backward", entropy_backward)
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
            if record_counts:
                state["entropy_count"] += metrics.token_count
        if record_counts:
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
    policy_count, entropy_count, world_size = _ddp_counts(
        trainer,
        torch,
        state["policy_count"],
        state["entropy_count"],
        measure=state["measure"],
        separate_entropy=bool(state["entropy_normalizer"]),
    )
    _average_ddp_banks(trainer, state, torch)
    policy_scale = world_size * state["policy_normalizer"] / policy_count
    entropy_scale = (
        world_size * state["entropy_normalizer"] / entropy_count
        if entropy_count
        else 0.0
    )
    with torch.no_grad():
        for name, parameter in trainer.agent.model.named_parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(policy_scale)
            entropy = state["entropy_gradients"].get(name)
            if entropy is not None:
                if parameter.grad is None:
                    parameter.grad = entropy.mul(entropy_scale)
                else:
                    parameter.grad.add_(entropy, alpha=entropy_scale)
    return True
