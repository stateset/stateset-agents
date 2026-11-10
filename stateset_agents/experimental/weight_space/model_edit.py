"""Optimize a constrained rank-one edit through a frozen causal language model."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from ...core.errors import ConfigurationError
from .constrained import _matrix, constrained_edit
from .deltas import LowRankDelta
from .generation import _digest

if TYPE_CHECKING:
    import torch

    from .session import WeightSpaceSession

__all__ = ["KeyedEditConfig", "KeyedEditStep", "OptimizedKeyedEdit"]


@dataclass(frozen=True)
class KeyedEditConfig:
    """Fixed design objective, optimizer limits, and execution caps.

    Minimize mean target continuation NLL plus ``kl_weight`` times token-mean
    KL(base || candidate) on calibration prompts, plus ``l2_weight * ||D||²``.
    Only the rank-one output vector is learned; no model parameter is trained.
    Each of ``steps + 1`` evaluations spends one session measurement unit.
    """

    steps: int = 20
    learning_rate: float = 0.5
    kl_weight: float = 1.0
    l2_weight: float = 0.01
    max_output_norm: float = 40.0
    max_update_norm: float = 5.0
    max_context_tokens: int = 512
    max_total_tokens: int = 4096
    max_logit_elements: int = 4_000_000
    max_reference_elements: int = 16_000_000
    max_model_forwards: int = 512
    target_atol: float = 0.002
    preservation_atol: float = 0.0005

    def __post_init__(self) -> None:
        for name in (
            "steps",
            "max_context_tokens",
            "max_total_tokens",
            "max_logit_elements",
            "max_reference_elements",
            "max_model_forwards",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ConfigurationError(f"{name} must be a positive integer")
        for name in ("learning_rate", "max_output_norm", "max_update_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ConfigurationError(f"{name} must be finite and positive")
        for name in ("kl_weight", "l2_weight", "target_atol", "preservation_atol"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ConfigurationError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class KeyedEditStep:
    """Design-set objective at an evaluated output vector, including step zero."""

    step: int
    objective: float
    target_nll: float
    locality_kl: float
    output_norm: float
    target_nlls: tuple[float, ...]


@dataclass(frozen=True)
class OptimizedKeyedEdit:
    """Owned factors and design evidence; this proposal has not been applied.

    Training-objective selection is not behavioral acceptance. Run ``trial_edit``
    with separate validation and generation cases before retaining the proposal.
    """

    delta: LowRankDelta
    history: tuple[KeyedEditStep, ...]
    selected_step: int
    cases_sha256: str
    model_forwards: int
    backward_calls: int
    update_norm: float
    protected_span_error: float
    target_max_error: float


_Target = tuple[tuple[int, ...], int, tuple[int, ...]]


def _design_layer(
    session: WeightSpaceSession,
    module: str,
    config: KeyedEditConfig,
    target_pairs: Sequence[tuple[str, str]],
    locality_prompts: Sequence[str],
) -> tuple[Any, torch.Tensor, tuple[Any, ...]]:
    """Check the model is fit for a design; return the layer, weight, parameters."""
    import torch

    if not isinstance(config, KeyedEditConfig):
        raise ConfigurationError("config must be KeyedEditConfig")
    if torch.is_inference_mode_enabled():
        raise ConfigurationError(
            "keyed optimization needs autograd, not inference_mode"
        )
    if any(m.training for m in session.model.modules()):
        raise ConfigurationError(
            "keyed optimization requires every module in eval mode"
        )
    layer = session._linear(module)
    weight = layer.weight.detach()
    if weight.dtype != torch.float32:
        raise ConfigurationError(
            "keyed optimization currently requires float32 weights"
        )
    parameters = tuple(session.model.parameters())
    if any(p.device != weight.device for p in parameters):
        raise ConfigurationError("keyed optimization requires one model device")
    if not target_pairs or (config.kl_weight > 0 and not locality_prompts):
        raise ConfigurationError("supply targets and locality prompts for a KL penalty")
    return layer, weight, parameters


def _encode_design(
    session: WeightSpaceSession,
    target_pairs: Sequence[tuple[str, str]],
    locality_prompts: Sequence[str],
    config: KeyedEditConfig,
) -> tuple[list[_Target], tuple[tuple[int, ...], ...]]:
    """Token IDs for the design cases, checked against the config's caps."""
    from .session import BudgetExhausted

    def encode(text: Any) -> tuple[int, ...]:
        if not isinstance(text, str) or not text:
            raise ConfigurationError("design strings must be nonempty")
        ids = tuple(session.encode(text))
        if not ids or any(type(i) is not int or i < 0 for i in ids):
            raise ConfigurationError("design strings must encode to nonnegative IDs")
        return ids

    targets: list[_Target] = []
    for pair in target_pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ConfigurationError(
                "target_pairs must contain prompt/continuation pairs"
            )
        prompt, continuation = pair
        p, c = encode(prompt), encode(continuation)
        targets.append((p + c, len(p), c))
    held = tuple(encode(p) for p in locality_prompts)
    all_ids = [t[0] for t in targets] + list(held)
    calls = len(all_ids) * (config.steps + 1)
    if (
        max(map(len, all_ids)) > config.max_context_tokens
        or sum(map(len, all_ids)) > config.max_total_tokens
        or calls > config.max_model_forwards
    ):
        raise ConfigurationError("keyed optimization exceeds token/forward limits")
    vocab = getattr(getattr(session.model, "config", None), "vocab_size", None)
    if type(vocab) is int and vocab > 0:
        if max(max(ids) for ids in all_ids) >= vocab:
            raise ConfigurationError("design token exceeds model vocabulary")
        if max(map(len, all_ids)) * vocab > config.max_logit_elements:
            raise ConfigurationError("design logits exceed max_logit_elements")
        if sum(map(len, held)) * vocab > config.max_reference_elements:
            raise ConfigurationError("design references exceed max_reference_elements")
    if session.budget.remaining < config.steps + 1:
        raise BudgetExhausted(
            f"keyed optimization requires {config.steps + 1} measurement units"
        )
    return targets, held


def _keyed_selector(
    weight: torch.Tensor,
    keys: torch.Tensor,
    protect: torch.Tensor | None,
    locality: torch.Tensor | None,
    locality_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """The keys, the fixed selector, its norm, and the protected row basis."""
    import torch

    k = _matrix(keys, "keys").to(weight)
    if keys.dtype != weight.dtype or k.shape[1] != weight.shape[1]:
        raise ConfigurationError("keys must match weight input width and dtype")
    # Solve a fixed selector with unit response at every supplied key. Its
    # output vector is then learned through all downstream nonlinear layers.
    selector = (
        constrained_edit(
            weight.new_zeros((1, weight.shape[1])),
            k,
            weight.new_ones((len(k), 1)),
            protect=protect,
            locality=locality,
            locality_weight=locality_weight,
            target_atol=1e-5,
            preservation_atol=1e-5,
        )
        .delta.dense()
        .detach()
    )
    selector_norm = float(selector.norm())
    if not math.isfinite(selector_norm) or selector_norm <= 0:
        raise ConfigurationError("invalid keyed selector")
    p = (
        weight.new_empty((0, weight.shape[1]))
        if protect is None
        else _matrix(protect, "protect").to(weight)
    )
    q = p.double()
    if len(p):
        pn = q / q.norm(dim=1, keepdim=True).clamp_min(torch.finfo(torch.float64).tiny)
        _, singular, vectors = torch.linalg.svd(pn, full_matrices=False)
        q = vectors[singular > singular[0] * 1e-10]
    return k, selector, selector_norm, q


class _KeyedDesign:
    """The optimisation state of one design: the vector, counters and history."""

    def __init__(
        self,
        session: WeightSpaceSession,
        config: KeyedEditConfig,
        weight: torch.Tensor,
        selector: torch.Tensor,
        selector_norm: float,
        targets: list[_Target],
        held: tuple[tuple[int, ...], ...],
    ) -> None:
        import torch

        self.session = session
        self.config = config
        self.weight = weight
        self.selector = selector
        self.targets = targets
        self.held = held
        self.b = weight.new_zeros((weight.shape[0], 1), requires_grad=True)
        self.radius = min(
            config.max_output_norm, config.max_update_norm / selector_norm
        )
        self.optimizer = torch.optim.Adam([self.b], lr=config.learning_rate)
        self.history: list[KeyedEditStep] = []
        self.baseline: list[torch.Tensor] = []
        self.reference_elements = 0
        self.forwards = self.backwards = self.invocations = 0
        self.best = self.b.detach().clone()
        self.best_step, self.best_objective = 0, math.inf

    def hook(self, _layer: Any, args: Any, kwargs: Any, output: Any) -> torch.Tensor:
        import torch
        from torch.nn import functional as F

        self.invocations += 1
        if self.invocations != 1:
            raise ConfigurationError("edited module must be invoked once per forward")
        inputs = args[0] if args else kwargs.get("input")
        if (
            not isinstance(inputs, torch.Tensor)
            or not isinstance(output, torch.Tensor)
            or inputs.ndim != 3
            or output.shape != (*inputs.shape[:-1], len(self.b))
            or inputs.dtype != torch.float32
            or output.dtype != torch.float32
        ):
            raise ConfigurationError(
                "keyed edit requires full-sequence float32 Linear tensors"
            )
        return output + F.linear(F.linear(inputs, self.selector), self.b)

    def forward(self, ids: tuple[int, ...]) -> torch.Tensor:
        import torch

        self.invocations = 0
        x = torch.tensor([ids], device=self.weight.device)
        self.forwards += 1
        output = self.session.model(
            input_ids=x,
            attention_mask=torch.ones_like(x),
            use_cache=False,
            logits_to_keep=0,
        )
        if self.invocations != 1:
            raise ConfigurationError("edited module was not invoked")
        logits: torch.Tensor = output.logits
        if (
            logits.ndim != 3
            or logits.shape[:2] != (1, len(ids))
            or logits.shape[-1] < 1
            or logits.numel() > self.config.max_logit_elements
            or not bool(torch.isfinite(logits).all())
        ):
            raise ConfigurationError("invalid or oversized design logits")
        return logits[0]

    def _target_pass(self, gradient: torch.Tensor, compute_grad: bool) -> list[float]:
        """Continuation NLL of each target; adds their gradient in place."""
        import torch
        from torch.nn import functional as F

        nlls: list[float] = []
        for ids, start, continuation in self.targets:
            logits = self.forward(ids)[start - 1 : len(ids) - 1]
            labels = torch.tensor(continuation, device=self.weight.device)
            nll = F.cross_entropy(logits, labels)
            nlls.append(float(nll.detach()))
            if compute_grad:
                if not nll.requires_grad:
                    raise ConfigurationError(
                        "model does not expose a differentiable edit path"
                    )
                self.backwards += 1
                gradient += torch.autograd.grad(nll / len(self.targets), self.b)[0]
            del logits, nll
        return nlls

    def _locality_pass(
        self, step: int, gradient: torch.Tensor, compute_grad: bool
    ) -> float:
        """Token-mean KL from the step-zero outputs; adds its gradient in place."""
        import torch

        config, held = self.config, self.held
        kl_total = 0.0
        for index, ids in enumerate(held):
            logq = self.forward(ids).double().log_softmax(-1)
            if step == 0:
                self.reference_elements += logq.numel()
                if self.reference_elements > config.max_reference_elements:
                    raise ConfigurationError(
                        "design references exceed max_reference_elements"
                    )
                self.baseline.append(logq.detach().cpu())
            logp = self.baseline[index].to(logq.device)
            kl = (logp.exp() * (logp - logq)).sum() / sum(map(len, held))
            kl_total += float(kl.detach())
            if compute_grad and config.kl_weight > 0:
                self.backwards += 1
                gradient += torch.autograd.grad(config.kl_weight * kl, self.b)[0]
            del logq, logp, kl
        return kl_total

    def step(self, step: int) -> None:
        """Evaluate the objective at the current vector; update it unless last."""
        import torch

        config, b = self.config, self.b
        self.session.budget.spend("design:keyed")
        gradient = torch.zeros_like(b)
        compute_grad = step < config.steps
        with (
            torch.set_grad_enabled(compute_grad),
            torch.autocast(device_type=self.weight.device.type, enabled=False),
        ):
            nlls = self._target_pass(gradient, compute_grad)
            kl_total = self._locality_pass(step, gradient, compute_grad)
            penalty = config.l2_weight * b.square().sum() * self.selector.square().sum()
            objective = (
                sum(nlls) / len(nlls)
                + config.kl_weight * kl_total
                + float(penalty.detach())
            )
            if not math.isfinite(objective) or not bool(torch.isfinite(gradient).all()):
                raise ConfigurationError("nonfinite keyed objective or gradient")
            self.history.append(
                KeyedEditStep(
                    step,
                    objective,
                    sum(nlls) / len(nlls),
                    kl_total,
                    float(b.detach().norm()),
                    tuple(nlls),
                )
            )
            if objective < self.best_objective:
                self.best, self.best_step, self.best_objective = (
                    b.detach().clone(),
                    step,
                    objective,
                )
            if compute_grad:
                gradient += torch.autograd.grad(penalty, b)[0]
                b.grad = gradient
                self.optimizer.step()
                with torch.no_grad():
                    norm = float(b.norm())
                    if not math.isfinite(norm):
                        raise ConfigurationError("nonfinite keyed optimizer update")
                    if norm > self.radius:
                        b.mul_(self.radius / norm)

    def rounded(
        self, original: torch.Tensor, k: torch.Tensor, q: torch.Tensor
    ) -> tuple[LowRankDelta, float, float, float]:
        """The selected edit, checked as it would be applied in float32."""
        import torch
        from torch.nn import functional as F

        config = self.config
        delta = LowRankDelta(self.best.clone(), self.selector.clone())
        with (
            torch.no_grad(),
            torch.autocast(device_type=self.weight.device.type, enabled=False),
        ):
            edited = original + delta.dense()
            realized = edited.double() - original.double()
            update_norm = float(realized.norm())
            span_error = float((realized @ q.T).norm())
            drift = F.linear(k, edited) - F.linear(k, original)
            target_error = float((drift - self.best.T).abs().max())
            if (
                not bool(torch.isfinite(edited).all())
                or update_norm > config.max_update_norm
                or target_error > config.target_atol
                or span_error > config.preservation_atol
            ):
                raise ConfigurationError(
                    "rounded keyed edit fails norm, target or protection tolerances"
                )
        return delta, update_norm, span_error, target_error


def _optimize_keyed_edit(
    session: WeightSpaceSession,
    module: str,
    keys: torch.Tensor,
    target_pairs: Sequence[tuple[str, str]],
    locality_prompts: Sequence[str],
    config: KeyedEditConfig,
    protect: torch.Tensor | None,
    locality: torch.Tensor | None,
    locality_weight: float,
) -> OptimizedKeyedEdit:
    import torch

    layer, weight, parameters = _design_layer(
        session, module, config, target_pairs, locality_prompts
    )
    targets, held = _encode_design(session, target_pairs, locality_prompts, config)
    k, selector, selector_norm, q = _keyed_selector(
        weight, keys, protect, locality, locality_weight
    )
    for name in session._weight_hashes:
        session._check_weight(name)
    original = weight.clone()
    design = _KeyedDesign(
        session, config, weight, selector, selector_norm, targets, held
    )
    flags = tuple(param.requires_grad for param in parameters)
    cases_hash = _digest((targets, held))
    audit = {
        "module": module,
        "cases_sha256": cases_hash,
        "config": asdict(config),
        "active_edit_ids": session.active_edit_ids,
    }
    handle = None
    try:
        for param in parameters:
            param.requires_grad_(False)
        handle = layer.register_forward_hook(design.hook, with_kwargs=True)
        for step in range(config.steps + 1):
            design.step(step)
        delta, update_norm, span_error, target_error = design.rounded(original, k, q)
        if not torch.equal(weight, original):
            raise ConfigurationError("model weight changed during keyed optimization")
        for name in session._weight_hashes:
            session._check_weight(name)
        result = OptimizedKeyedEdit(
            delta,
            tuple(design.history),
            design.best_step,
            cases_hash,
            design.forwards,
            design.backwards,
            update_norm,
            span_error,
            target_error,
        )
        session.edit_designs.append(
            {
                **audit,
                "status": "completed",
                "selected_step": design.best_step,
                "history": [asdict(h) for h in design.history],
                "model_forwards": design.forwards,
                "backward_calls": design.backwards,
                "update_norm": update_norm,
                "protected_span_error": span_error,
                "target_max_error": target_error,
            }
        )
        return result
    except BaseException as exc:
        session.edit_designs.append(
            {
                **audit,
                "status": "failed",
                "error_type": type(exc).__name__,
                "model_forwards": design.forwards,
                "backward_calls": design.backwards,
                "history": [asdict(h) for h in design.history],
            }
        )
        raise
    finally:
        if handle is not None:
            handle.remove()
        for param, flag in zip(parameters, flags, strict=True):
            param.requires_grad_(flag)
