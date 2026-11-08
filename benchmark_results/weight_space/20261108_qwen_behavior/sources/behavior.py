"""Whole-model, teacher-forced acceptance checks for a proposed weight edit.

These checks measure declared validation cases. They are neither a proof about
unseen prompts nor a replacement for an untouched final evaluation. Every valid
locality position contributes a full-vocabulary KL value; no top-k approximation
or padding token is used.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...core.errors import ConfigurationError

if TYPE_CHECKING:
    import torch

    from .session import WeightSpaceSession

__all__ = ["BehaviorLimits", "BehaviorMetrics", "EditTrial"]


@dataclass(frozen=True)
class BehaviorLimits:
    """Predeclared limits; KL is base-to-edited and loss/gain are in nats."""

    max_mean_kl: float
    max_token_kl: float
    max_nll_increase: float
    min_top1_agreement: float = 0.0
    min_target_gain: float | None = None
    min_each_target_gain: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_mean_kl", "max_token_kl", "max_nll_increase"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ConfigurationError(f"{name} must be finite and nonnegative")
        if not 0 <= self.min_top1_agreement <= 1:
            raise ConfigurationError("min_top1_agreement must be in [0, 1]")
        for value in (self.min_target_gain, self.min_each_target_gain):
            if value is not None and not math.isfinite(value):
                raise ConfigurationError("target gain thresholds must be finite")


@dataclass(frozen=True)
class BehaviorMetrics:
    """Token-weighted locality metrics and per-case mean continuation gains."""

    locality_tokens: int
    observed_tokens: int
    mean_kl: float
    max_token_kl: float
    top1_agreement: float
    base_nll: float
    edited_nll: float
    nll_increase: float
    target_gains: tuple[float, ...]
    mean_target_gain: float | None


@dataclass(frozen=True)
class EditTrial:
    """Acceptance decision; rejected trials have already been reverted."""

    edit_id: str
    accepted: bool
    violations: tuple[str, ...]
    cases_sha256: str
    metrics: BehaviorMetrics
    limits: BehaviorLimits


@dataclass(frozen=True)
class _Case:
    tokens: tuple[int, ...]
    target_start: int | None = None


@dataclass(frozen=True)
class _Plan:
    cases: tuple[_Case, ...]
    locality_count: int
    cases_sha256: str
    max_logit_elements: int


@dataclass
class _Snapshot:
    locality_logits: list[torch.Tensor]
    target_logprobs: list[float]


def _prepare(
    session: WeightSpaceSession,
    prompts: Sequence[str],
    targets: Sequence[tuple[str, str]],
    limits: BehaviorLimits,
    max_logit_elements: int,
) -> _Plan:
    if not isinstance(limits, BehaviorLimits):
        raise ConfigurationError("limits must be BehaviorLimits")
    if not prompts:
        raise ConfigurationError("behavior trials need locality prompts")
    if type(max_logit_elements) is not int or max_logit_elements < 1:
        raise ConfigurationError("max_logit_elements must be a positive integer")
    if session.batch_size < 1:
        raise ConfigurationError("batch_size must be positive")
    if any(m.training for m in session.model.modules()):
        raise ConfigurationError("behavior trials require every module in eval mode")
    if not targets and (
        limits.min_target_gain is not None or limits.min_each_target_gain is not None
    ):
        raise ConfigurationError("target gain limits require target_pairs")

    def encode(text: str) -> tuple[int, ...]:
        if not isinstance(text, str) or not text:
            raise ConfigurationError("behavior cases need nonempty strings")
        tokens = tuple(session.encode(text))
        if not tokens or any(type(t) is not int or t < 0 for t in tokens):
            raise ConfigurationError(
                "behavior cases must encode to nonnegative token ids"
            )
        return tokens

    cases = [_Case(encode(p)) for p in prompts]
    if sum(len(c.tokens) - 1 for c in cases) == 0:
        raise ConfigurationError("locality cases need at least one observed next token")
    for prompt, continuation in targets:
        p, c = encode(prompt), encode(continuation)
        cases.append(_Case(p + c, len(p)))
    vocab = getattr(getattr(session.model, "config", None), "vocab_size", None)
    if type(vocab) is int and vocab > 0:
        if any(t >= vocab for c in cases for t in c.tokens):
            raise ConfigurationError("case token id exceeds model vocabulary")
        elements = sum(len(c.tokens) for c in cases) * vocab
        padded = max(
            len(cases[i : i + session.batch_size])
            * max(len(c.tokens) for c in cases[i : i + session.batch_size])
            * vocab
            for i in range(0, len(cases), session.batch_size)
        )
        if max(elements, padded) > max_logit_elements:
            raise ConfigurationError(
                "behavior logits exceed max_logit_elements; reduce cases or batch_size"
            )
    canonical = [(c.tokens, c.target_start) for c in cases]
    digest = hashlib.sha256(
        json.dumps(canonical, separators=(",", ":")).encode()
    ).hexdigest()
    return _Plan(tuple(cases), len(prompts), digest, max_logit_elements)


def _snapshot(session: WeightSpaceSession, plan: _Plan, reason: str) -> _Snapshot:
    import torch

    session.budget.spend(reason)
    snapshot = _Snapshot([], [])
    elements, vocab = 0, None
    for offset in range(0, len(plan.cases), session.batch_size):
        chunk = plan.cases[offset : offset + session.batch_size]
        ids, mask = session._pad([list(c.tokens) for c in chunk])
        with torch.no_grad():
            logits = session.model(
                input_ids=ids, attention_mask=mask, use_cache=False
            ).logits
        if logits.ndim != 3 or tuple(logits.shape[:2]) != tuple(ids.shape):
            raise ConfigurationError("model must return logits at every input position")
        if logits.numel() > plan.max_logit_elements:
            raise ConfigurationError("padded model logits exceed max_logit_elements")
        if vocab is not None and vocab != logits.shape[-1]:
            raise ConfigurationError("model vocabulary changed during snapshot")
        vocab = logits.shape[-1]
        for i, case in enumerate(chunk):
            row = logits[i, : len(case.tokens)].detach().float().cpu().clone()
            elements += row.numel()
            if elements > plan.max_logit_elements:
                raise ConfigurationError("model logits exceed max_logit_elements")
            if not bool(torch.isfinite(row).all()):
                raise ConfigurationError("model produced nonfinite logits")
            if case.target_start is None:
                snapshot.locality_logits.append(row)
            else:
                # Score only the requested continuation, conditioning on its
                # exact supplied prefix. Prompt/continuation tokenize separately.
                start = case.target_start
                selected = row[start - 1 : -1].double().log_softmax(-1)
                wanted = torch.tensor(case.tokens[start:], dtype=torch.long)
                snapshot.target_logprobs.append(
                    float(selected.gather(1, wanted[:, None]).mean())
                )
        del logits
    return snapshot


def _compare(plan: _Plan, before: _Snapshot, after: _Snapshot) -> BehaviorMetrics:
    import torch

    tokens, observed, matches = 0, 0, 0
    kl_sum, kl_max, base_loss, edited_loss = 0.0, 0.0, 0.0, 0.0
    if (
        len(before.locality_logits) != plan.locality_count
        or len(after.locality_logits) != plan.locality_count
    ):
        raise ConfigurationError("behavior snapshots omit locality cases")
    for case, base, edited in zip(
        plan.cases[: plan.locality_count],
        before.locality_logits,
        after.locality_logits,
        strict=True,
    ):
        if base.shape != edited.shape:
            raise ConfigurationError("base and edited vocabulary dimensions differ")
        # Full-vocabulary float64 normalization in bounded chunks. Keeping the
        # raw float32 logits avoids committing float32 softmax rounding to disk.
        for start in range(0, len(case.tokens), 16):
            end = min(start + 16, len(case.tokens))
            p, q = base[start:end].double().log_softmax(-1), edited[
                start:end
            ].double().log_softmax(-1)
            kl = (p.exp() * (p - q)).sum(-1).clamp_min(0)
            kl_sum += float(kl.sum())
            kl_max = max(kl_max, float(kl.max()))
            matches += int((p.argmax(-1) == q.argmax(-1)).sum())
            tokens += end - start
            n = min(end, len(case.tokens) - 1) - start
            if n > 0:
                targets = torch.tensor(
                    case.tokens[start + 1 : start + n + 1], dtype=torch.long
                )
                base_loss -= float(p[:n].gather(1, targets[:, None]).sum())
                edited_loss -= float(q[:n].gather(1, targets[:, None]).sum())
                observed += n
    expected_targets = len(plan.cases) - plan.locality_count
    if (
        len(before.target_logprobs) != expected_targets
        or len(after.target_logprobs) != expected_targets
    ):
        raise ConfigurationError("behavior snapshots omit target cases")
    gains = tuple(
        a - b
        for a, b in zip(after.target_logprobs, before.target_logprobs, strict=True)
    )
    return BehaviorMetrics(
        tokens,
        observed,
        kl_sum / tokens,
        kl_max,
        matches / tokens,
        base_loss / observed,
        edited_loss / observed,
        (edited_loss - base_loss) / observed,
        gains,
        sum(gains) / len(gains) if gains else None,
    )


def _violations(metrics: BehaviorMetrics, limits: BehaviorLimits) -> tuple[str, ...]:
    failures = []
    for name, actual, cap in (
        ("mean_kl", metrics.mean_kl, limits.max_mean_kl),
        ("max_token_kl", metrics.max_token_kl, limits.max_token_kl),
        ("nll_increase", metrics.nll_increase, limits.max_nll_increase),
    ):
        if not math.isfinite(actual) or actual > cap:
            failures.append(name)
    if metrics.top1_agreement < limits.min_top1_agreement:
        failures.append("top1_agreement")
    if limits.min_target_gain is not None and (
        metrics.mean_target_gain is None
        or metrics.mean_target_gain < limits.min_target_gain
    ):
        failures.append("mean_target_gain")
    if limits.min_each_target_gain is not None and (
        not metrics.target_gains
        or min(metrics.target_gains) < limits.min_each_target_gain
    ):
        failures.append("each_target_gain")
    return tuple(failures)
