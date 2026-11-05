"""Training-loop plumbing shared by the preference and reward-model trainers.

Optimizer steps go through :func:`stateset_agents.training.trainer_utils.safe_optimizer_step`
(finite-gradient validation, clipping, step counting, scheduler advance) via a
small adapter exposing the attributes it reads.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import random
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from stateset_agents.core.checkpoint_io import load_checkpoint_file

from ..trainer_utils import backward_training_loss, require_torch, safe_optimizer_step
from .encoding import EncodedSequence

logger = logging.getLogger(__name__)


def resolve_device(device: str, model: Any) -> Any:
    """The device to train on; moves ``model`` there unless it is sharded."""
    torch = require_torch()
    if getattr(model, "hf_device_map", None):
        return next(model.parameters()).device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    model.to(resolved)
    return resolved


def seed_everything(seed: int) -> None:
    torch = require_torch()
    random.seed(seed)
    torch.manual_seed(seed)


def build_optimizer(model: Any, config: Any) -> Any:
    """AdamW over trainable parameters; no weight decay on 1-D parameters."""
    torch = require_torch()
    decay: list[Any] = []
    no_decay: list[Any] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    if not decay and not no_decay:
        raise ValueError("the model has no trainable parameters")
    groups = [
        {"params": decay, "weight_decay": float(config.weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        [group for group in groups if group["params"]],
        lr=float(config.learning_rate),
        betas=(float(config.adam_beta1), float(config.adam_beta2)),
        eps=float(config.adam_epsilon),
    )


class _Schedule:
    """Picklable LR multiplier (its ``__dict__`` is what LambdaLR saves)."""

    def __init__(self, kind: str, warmup: int, total: int) -> None:
        self.kind = kind
        self.warmup = warmup
        self.total = max(total, 1)

    def __call__(self, step: int) -> float:
        if self.warmup and step < self.warmup:
            return (step + 1) / (self.warmup + 1)
        if self.kind == "constant":
            return 1.0
        span = max(self.total - self.warmup, 1)
        progress = min(max(step - self.warmup, 0) / span, 1.0)
        if self.kind == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(0.0, 1.0 - progress)


def build_scheduler(optimizer: Any, kind: str, warmup: int, total: int) -> Any:
    torch = require_torch()
    return torch.optim.lr_scheduler.LambdaLR(optimizer, _Schedule(kind, warmup, total))


class StepState:
    """The trainer attributes ``safe_optimizer_step`` reads and updates."""

    def __init__(self, model: Any, optimizer: Any, lr_scheduler: Any) -> None:
        self.agent = SimpleNamespace(model=model)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.global_step = 0
        self._grad_accum_step = 0
        self._grpo_accumulation: Any = None

    def _sync_rollout_backend(self) -> None:
        """No rollout engine to synchronize."""


def backward(state: StepState, loss: Any) -> None:
    """Backpropagate a finite differentiable scalar (gradients are discarded
    on failure)."""
    backward_training_loss(state, loss, require_torch())


def _gradients_finite(optimizer: Any, torch: Any) -> bool:
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            grad = parameter.grad
            if grad is not None and not bool(torch.isfinite(grad).all()):
                return False
    return True


def optimizer_step(
    state: StepState, *, max_grad_norm: float, skip_nonfinite: bool
) -> bool:
    """Clip, validate and apply one update. Returns whether it was committed.

    With ``skip_nonfinite`` a window with non-finite gradients is discarded
    instead of raising.
    """
    torch = require_torch()
    if skip_nonfinite and not _gradients_finite(state.optimizer, torch):
        state.optimizer.zero_grad(set_to_none=True)
        logger.warning("Skipping an update with non-finite gradients")
        return False
    return safe_optimizer_step(state, torch, max_grad_norm=max_grad_norm)


def plan_epoch(
    count: int, batch_size: int, accumulation: int, seed: int, epoch: int
) -> list[list[list[int]]]:
    """Shuffled item indices grouped into optimizer windows of micro-batches."""
    order = list(range(count))
    random.Random(f"{seed}:{epoch}").shuffle(order)
    batches = [order[i : i + batch_size] for i in range(0, count, batch_size)]
    return [batches[i : i + accumulation] for i in range(0, len(batches), accumulation)]


def windows_per_epoch(count: int, batch_size: int, accumulation: int) -> int:
    batches = math.ceil(count / batch_size)
    return math.ceil(batches / accumulation)


def collate(
    sequences: Sequence[EncodedSequence], pad_id: int, device: Any
) -> tuple[Any, Any, Any]:
    """Right-padded ``(input_ids, attention_mask, response_mask)``."""
    torch = require_torch()
    width = max(len(sequence) for sequence in sequences)
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(sequences), width), dtype=torch.long)
    response = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        tokens = sequence.input_ids
        ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        attention[row, : len(tokens)] = 1
        response[row, len(sequence.prompt_ids) : len(tokens)] = 1
    return ids.to(device), attention.to(device), response.to(device)


def model_outputs(
    model: Any, input_ids: Any, attention_mask: Any, **kwargs: Any
) -> Any:
    """Call a Hugging Face style model without its KV cache."""
    if getattr(model, "config", None) is not None:
        kwargs.setdefault("use_cache", False)
    return model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)


def logits_of(outputs: Any) -> Any:
    logits = getattr(outputs, "logits", None)
    if logits is None and isinstance(outputs, tuple | list):
        logits = outputs[0]
    if logits is None:
        logits = outputs
    return logits


def autocast(
    device: Any, mixed_precision: str
) -> contextlib.AbstractContextManager[Any]:
    if mixed_precision != "bf16":
        return contextlib.nullcontext()
    torch = require_torch()
    context: contextlib.AbstractContextManager[Any] = torch.autocast(
        device_type=device.type, dtype=torch.bfloat16
    )
    return context


@contextlib.contextmanager
def evaluation_mode(*models: Any) -> Iterator[None]:
    """Eval mode and no gradients, restoring each model's previous mode."""
    torch = require_torch()
    previous = [(model, model.training) for model in models if model is not None]
    for model, _ in previous:
        model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        for model, training in previous:
            model.train(training)


def current_lr(optimizer: Any) -> float:
    return float(optimizer.param_groups[0]["lr"])


def save_model_dir(model: Any, tokenizer: Any, directory: Path) -> None:
    """Save weights (``save_pretrained`` when available) and the tokenizer."""
    directory.mkdir(parents=True, exist_ok=True)
    save_pretrained = getattr(model, "save_pretrained", None)
    if callable(save_pretrained):
        save_pretrained(str(directory))
    else:
        torch = require_torch()
        torch.save(model.state_dict(), directory / "pytorch_model.bin")
    tokenizer_save = getattr(tokenizer, "save_pretrained", None)
    if callable(tokenizer_save):
        tokenizer_save(str(directory))


def save_state(path: Path, payload: dict[str, Any]) -> None:
    """``torch.save`` of plain tensors/containers (loadable weights-only)."""
    torch = require_torch()
    torch.save(payload, path)


def load_state(path: Path, *, trusted: bool) -> dict[str, Any]:
    state = load_checkpoint_file(path, trusted=trusted, torch_module=require_torch())
    if not isinstance(state, dict):
        raise ValueError(f"{path} does not hold a trainer state dictionary")
    return state


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read checkpoint metadata {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint metadata {path} must be a JSON object")
    return payload


def mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


class WindowedTrainer:
    """Epochs of shuffled optimizer windows, shared by both trainers.

    Subclasses set ``config``, ``model``, ``tokenizer`` and ``encoder``, call
    :meth:`_init_optimization`, and implement the hooks below.
    """

    config: Any
    model: Any
    tokenizer: Any
    trainer_kind = "trainer"

    def _init_optimization(self, item_count: int) -> None:
        config = self.config
        self.optimizer = build_optimizer(self.model, config)
        per_epoch = windows_per_epoch(
            item_count, config.batch_size, config.gradient_accumulation_steps
        )
        self.total_steps = int(config.max_steps or per_epoch * config.num_epochs)
        self.lr_scheduler = build_scheduler(
            self.optimizer, config.lr_scheduler, config.warmup_steps, self.total_steps
        )
        self._step = StepState(self.model, self.optimizer, self.lr_scheduler)
        self._item_total = item_count
        self.epoch = 0
        self.windows_done_in_epoch = 0
        self.update_attempts = 0
        self.skipped_steps = 0
        self.history: list[dict[str, float]] = []

    @property
    def global_step(self) -> int:
        """Committed optimizer updates."""
        return int(self._step.global_step)

    # ------------------------------------------------------------- hooks

    def _train_window(self, window: list[list[int]]) -> dict[str, float] | None:
        raise NotImplementedError

    def _has_eval_data(self) -> bool:
        raise NotImplementedError

    def evaluate(self) -> dict[str, float]:
        raise NotImplementedError

    def save_checkpoint(self, directory: str | Path) -> Path:
        raise NotImplementedError

    # ---------------------------------------------------------- the loop

    def _accumulate(
        self,
        window: list[list[int]],
        weight_of: Any,
        compute: Any,
    ) -> list[tuple[float, Any]] | None:
        """Backpropagate each micro-batch as its share of the window's
        weighted mean, then commit one update.

        ``compute(batch)`` returns ``(PreferenceLoss, weights, extra)``.
        Returns ``[(reduced loss, (PreferenceLoss, extra)), ...]``, or ``None``
        when the window was skipped (non-finite under ``skip``, or nothing
        committed).
        """
        from .losses import NonFiniteLossError, reduce_losses

        total_weight = math.fsum(weight_of(i) for batch in window for i in batch)
        results: list[tuple[float, Any]] = []
        try:
            for batch in window:
                loss, weights, extra = compute(batch)
                reduced = reduce_losses(loss.losses, weights, normalizer=total_weight)
                backward(self._step, reduced)
                results.append((float(reduced.detach()), (loss, extra)))
        except NonFiniteLossError:
            self.optimizer.zero_grad(set_to_none=True)
            if self.config.nonfinite_policy == "raise":
                raise
            logger.warning("Skipping a window with a non-finite loss")
            return None
        committed = optimizer_step(
            self._step,
            max_grad_norm=float(self.config.max_grad_norm),
            skip_nonfinite=self.config.nonfinite_policy == "skip",
        )
        return results if committed else None

    def _run(self) -> None:
        """Run the configured epochs; ``max_steps`` (update attempts)
        overrides ``num_epochs``."""
        config = self.config
        self.model.train()
        finished = False
        while not finished and (
            config.max_steps is not None or self.epoch < config.num_epochs
        ):
            windows = plan_epoch(
                self._item_total,
                config.batch_size,
                config.gradient_accumulation_steps,
                config.seed,
                self.epoch,
            )
            for index, window in enumerate(windows):
                if index < self.windows_done_in_epoch:
                    continue
                if (
                    config.max_steps is not None
                    and self.update_attempts >= config.max_steps
                ):
                    finished = True
                    break
                metrics = self._train_window(window)
                self.update_attempts += 1
                self.windows_done_in_epoch = index + 1
                if metrics is None:
                    self.skipped_steps += 1
                    continue
                self.history.append(metrics)
                step = self.global_step
                if step % config.logging_steps == 0:
                    logger.info("%s step %d: %s", self.trainer_kind, step, metrics)
                if config.eval_steps and step % config.eval_steps == 0:
                    if self._has_eval_data():
                        self.history[-1].update(self.evaluate())
                if (
                    config.save_steps
                    and config.output_dir
                    and step % config.save_steps == 0
                ):
                    self.save_checkpoint(Path(config.output_dir) / f"checkpoint-{step}")
            else:
                self.epoch += 1
                self.windows_done_in_epoch = 0

    # ----------------------------------------------------- checkpointing

    def _progress(self) -> dict[str, int]:
        return {
            "global_step": self.global_step,
            "update_attempts": self.update_attempts,
            "skipped_steps": self.skipped_steps,
            "epoch": self.epoch,
            "windows_done_in_epoch": self.windows_done_in_epoch,
        }

    def _restore_progress(self, meta: dict[str, Any]) -> None:
        self._step.global_step = int(meta["global_step"])
        self.update_attempts = int(meta["update_attempts"])
        self.skipped_steps = int(meta["skipped_steps"])
        self.epoch = int(meta["epoch"])
        self.windows_done_in_epoch = int(meta["windows_done_in_epoch"])

    def _save_optimization_state(self, path: Path) -> None:
        torch = require_torch()
        save_state(
            path,
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.lr_scheduler.state_dict(),
                "torch_rng": torch.get_rng_state(),
            },
        )

    def _load_optimization_state(self, path: Path, *, trusted: bool) -> None:
        torch = require_torch()
        state = load_state(path, trusted=trusted)
        self.optimizer.load_state_dict(state["optimizer"])
        self.lr_scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
