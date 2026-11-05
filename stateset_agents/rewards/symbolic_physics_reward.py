"""
Reward function for symbolic physics relation discovery.

A candidate expression is evaluated at seeded samples of the task's declared
variables and scored on validity, equivalence to the task's
``target_expression``, the task's constraints (symmetry, zeros, scaling,
parity, sum and product forms) and simplicity.

A constant is not a relation: an expression that names none of the declared
variables, or takes the same value at every sample ("0", "1", "x - x"),
scores 0, because a constant satisfies symmetry, parity and zero constraints
trivially. The one exception is a task whose ``target_expression`` is itself
constant. A task without declared ``variables`` is misconfigured and raises
``ConfigurationError`` (``fail_on_error``) rather than scoring 0.
"""

from __future__ import annotations

import json
import logging
import math
import re
import zlib
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from stateset_agents.core.errors import ConfigurationError
from stateset_agents.core.reward_base import RewardFunction, RewardResult, RewardType
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.utils.symbolic_math import (
    DEFAULT_ALLOWED_FUNCTIONS,
    apply_derived_variables,
    extract_expression,
    generate_samples,
    safe_eval_expression,
)

logger = logging.getLogger(__name__)


@dataclass
class SymbolicRewardConfig:
    """Configuration for symbolic physics reward evaluation."""

    num_samples: int = 8
    sample_range: tuple[float, float] = (-2.0, 2.0)
    rel_tol: float = 1e-2
    abs_tol: float = 1e-3
    max_expression_length: int = 140
    component_weights: dict[str, float] = field(
        default_factory=lambda: {
            "validity": 0.1,
            "equivalence": 0.6,
            "constraints": 0.2,
            "simplicity": 0.1,
        }
    )
    allowed_functions: Mapping[str, Callable[..., float]] | None = None


def _close_enough(a: float, b: float, rel_tol: float, abs_tol: float) -> bool:
    return abs(a - b) <= abs_tol + rel_tol * max(abs(a), abs(b))


_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
#: Samples within this tolerance of each other count as one constant value
#: (floating-point noise, as in ``sin(x)**2 + cos(x)**2``).
_CONSTANT_REL_TOL = 1e-9
_CONSTANT_ABS_TOL = 1e-12


def is_constant_expression(
    expression: str, values: Sequence[float], variables: Collection[str]
) -> bool:
    """Whether ``expression`` is a constant over the task: it names none of
    the declared ``variables`` or evaluates to the same value at every sample
    (``values``)."""
    if not set(_IDENTIFIER_RE.findall(expression)) & set(variables):
        return True
    if len(values) < 2:
        return False
    first = values[0]
    return all(
        math.isclose(value, first, rel_tol=_CONSTANT_REL_TOL, abs_tol=_CONSTANT_ABS_TOL)
        for value in values[1:]
    )


def _zero_result(error: str, **metadata: Any) -> RewardResult:
    return RewardResult(score=0.0, metadata={"error": error, **metadata})


def _weighted_average(scores: dict[str, float], weights: dict[str, float]) -> float:
    total = 0.0
    weighted = 0.0
    for name, value in scores.items():
        weight = float(weights.get(name, 1.0))
        total += weight
        weighted += weight * value
    if total <= 0:
        return 0.0
    return weighted / total


def _extract_last_assistant(turns: list[Any]) -> ConversationTurn | None:
    for item in reversed(turns):
        if isinstance(item, ConversationTurn) and item.role == "assistant":
            return item
        role = getattr(item, "role", None)
        content = getattr(item, "content", None)
        if role == "assistant" and content is not None:
            return ConversationTurn(role="assistant", content=str(content))
        if isinstance(item, dict) and item.get("role") == "assistant":
            return ConversationTurn(
                role="assistant", content=str(item.get("content", ""))
            )
    return None


def _sample_seed(context: Mapping[str, Any]) -> int:
    """Stable per-task seed, so identical answers always get identical scores."""
    key = json.dumps(
        {
            "task_id": context.get("task_id"),
            "target": context.get("target_expression"),
            "variables": list(context.get("variables") or []),
            "derived": dict(context.get("derived_variables") or {}),
        },
        sort_keys=True,
        default=str,
    )
    return zlib.crc32(key.encode("utf-8"))


def _rederive(
    values: dict[str, float],
    derived_variables: Mapping[str, str],
    keep: set[str],
    allowed_functions: Mapping[str, Callable[..., float]],
) -> dict[str, float]:
    """Recompute derived variables after base values change (``keep`` stays)."""
    for name, expr in derived_variables.items():
        if name not in keep:
            values[name] = safe_eval_expression(expr, values, allowed_functions)
    return values


def _symmetry_score(
    expression: str,
    samples: list[dict[str, float]],
    pairs: list[list[str]],
    sign: float,
    rel_tol: float,
    abs_tol: float,
    allowed_functions: Mapping[str, Callable[..., float]],
    derived_variables: Mapping[str, str] | None = None,
) -> float:
    derived_variables = derived_variables or {}
    matches = 0
    total = 0
    for values in samples:
        try:
            base_val = safe_eval_expression(expression, values, allowed_functions)
        except Exception:
            total += len(pairs)
            continue
        for pair in pairs:
            if len(pair) != 2:
                continue
            a, b = pair
            swapped = dict(values)
            swapped_a = swapped[a]
            swapped_b = swapped[b]
            swapped[a], swapped[b] = swapped_b, swapped_a
            try:
                _rederive(swapped, derived_variables, {a, b}, allowed_functions)
                swapped_val = safe_eval_expression(
                    expression, swapped, allowed_functions
                )
            except Exception:
                total += 1
                continue
            if _close_enough(base_val, sign * swapped_val, rel_tol, abs_tol):
                matches += 1
            total += 1
    return matches / total if total else 0.0


def _zeros_score(
    expression: str,
    variables: list[str],
    points: list[dict[str, float]],
    derived_variables: Mapping[str, str],
    rel_tol: float,
    abs_tol: float,
    allowed_functions: Mapping[str, Callable[..., float]],
) -> float:
    if not points:
        return 0.0
    matches = 0
    total = 0
    base_vars = [var for var in variables if var not in derived_variables]
    for point in points:
        base_values = {var: float(point.get(var, 0.0)) for var in base_vars}
        values = apply_derived_variables(
            base_values, derived_variables, allowed_functions
        )
        for var in variables:
            values.setdefault(var, float(point.get(var, 0.0)))
        try:
            value = safe_eval_expression(expression, values, allowed_functions)
        except Exception:
            total += 1
            continue
        if _close_enough(value, 0.0, rel_tol, abs_tol):
            matches += 1
        total += 1
    return matches / total if total else 0.0


def _scale_score(
    expression: str,
    samples: list[dict[str, float]],
    variables: list[str],
    derived_variables: Mapping[str, str],
    factors: list[float],
    power: float,
    rel_tol: float,
    abs_tol: float,
    allowed_functions: Mapping[str, Callable[..., float]],
) -> float:
    if not factors:
        return 0.0
    matches = 0
    total = 0
    base_vars = [var for var in variables if var not in derived_variables]
    for values in samples:
        base_values = {var: values.get(var, 0.0) for var in base_vars}
        try:
            base_val = safe_eval_expression(expression, values, allowed_functions)
        except Exception:
            total += len(factors)
            continue
        for factor in factors:
            scaled_base = {var: base_values[var] * factor for var in base_vars}
            scaled = apply_derived_variables(
                scaled_base, derived_variables, allowed_functions
            )
            for var in variables:
                scaled.setdefault(var, values.get(var, 0.0) * factor)
            try:
                scaled_val = safe_eval_expression(expression, scaled, allowed_functions)
            except Exception:
                total += 1
                continue
            target = (factor**power) * base_val
            if _close_enough(scaled_val, target, rel_tol, abs_tol):
                matches += 1
            total += 1
    return matches / total if total else 0.0


def _sign_flip_score(
    expression: str,
    samples: list[dict[str, float]],
    variables: list[str],
    sign: float,
    rel_tol: float,
    abs_tol: float,
    allowed_functions: Mapping[str, Callable[..., float]],
    derived_variables: Mapping[str, str] | None = None,
) -> float:
    derived_variables = derived_variables or {}
    if not variables:
        return 0.0
    matches = 0
    total = 0
    for values in samples:
        try:
            base_val = safe_eval_expression(expression, values, allowed_functions)
        except Exception:
            total += len(variables)
            continue
        for var in variables:
            if var not in values:
                continue
            flipped = dict(values)
            flipped[var] = -flipped[var]
            try:
                _rederive(flipped, derived_variables, {var}, allowed_functions)
                flipped_val = safe_eval_expression(
                    expression, flipped, allowed_functions
                )
            except Exception:
                total += 1
                continue
            if _close_enough(base_val, sign * flipped_val, rel_tol, abs_tol):
                matches += 1
            total += 1
    return matches / total if total else 0.0


def _sum_score(
    expression: str,
    samples: list[dict[str, float]],
    terms: list[str],
    weights: list[float] | None,
    rel_tol: float,
    abs_tol: float,
    allowed_functions: Mapping[str, Callable[..., float]],
    candidate_values: list[float] | None = None,
) -> float:
    if not terms:
        return 0.0
    weights = weights or []
    padded_weights = [
        float(weights[idx]) if idx < len(weights) else 1.0 for idx in range(len(terms))
    ]
    matches = 0
    total = 0
    for idx, values in enumerate(samples):
        try:
            target = 0.0
            for term, weight in zip(terms, padded_weights, strict=False):
                term_val = safe_eval_expression(term, values, allowed_functions)
                target += weight * term_val
            if candidate_values is None:
                candidate = safe_eval_expression(expression, values, allowed_functions)
            else:
                candidate = candidate_values[idx]
            if _close_enough(candidate, target, rel_tol, abs_tol):
                matches += 1
            total += 1
        except Exception:
            total += 1
    return matches / total if total else 0.0


def _product_score(
    expression: str,
    samples: list[dict[str, float]],
    factors: list[str],
    exponents: list[float] | None,
    rel_tol: float,
    abs_tol: float,
    allowed_functions: Mapping[str, Callable[..., float]],
    candidate_values: list[float] | None = None,
) -> float:
    if not factors:
        return 0.0
    exponents = exponents or []
    padded_exponents = [
        float(exponents[idx]) if idx < len(exponents) else 1.0
        for idx in range(len(factors))
    ]
    matches = 0
    total = 0
    for idx, values in enumerate(samples):
        try:
            target = 1.0
            for factor, exponent in zip(factors, padded_exponents, strict=False):
                factor_val = safe_eval_expression(factor, values, allowed_functions)
                target *= factor_val**exponent
            if candidate_values is None:
                candidate = safe_eval_expression(expression, values, allowed_functions)
            else:
                candidate = candidate_values[idx]
            if _close_enough(candidate, target, rel_tol, abs_tol):
                matches += 1
            total += 1
        except Exception:
            total += 1
    return matches / total if total else 0.0


class SymbolicPhysicsRewardFunction(RewardFunction):
    """Reward function that scores symbolic expressions against constraints.

    Raises ``ConfigurationError`` for a task without declared ``variables``;
    a constant candidate scores 0 unless the target is constant (see the
    module docs).
    """

    #: A task without declared variables is a configuration error that must
    #: stop training, not an episode to drop.
    fail_on_error = True

    def __init__(
        self,
        config: SymbolicRewardConfig | None = None,
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            weight=weight,
            reward_type=RewardType.IMMEDIATE,
            name="SymbolicPhysicsReward",
        )
        self.config = config or SymbolicRewardConfig()
        self.allowed_functions = (
            dict(self.config.allowed_functions)
            if self.config.allowed_functions
            else dict(DEFAULT_ALLOWED_FUNCTIONS)
        )

    async def compute_reward(
        self,
        turns: list[Any] | None = None,
        context: dict[str, Any] | None = None,
        trajectory: Any | None = None,
        turn: Any | None = None,
        **_: Any,
    ) -> RewardResult:
        context = context or {}
        variables = list(context.get("variables") or [])
        if not variables:
            raise ConfigurationError(
                "SymbolicPhysicsRewardFunction needs the task's declared "
                "variables (context['variables']); without them no expression "
                "can be checked",
                task_id=context.get("task_id"),
            )
        if turns is None and turn is not None:
            turns = [turn]
        turns = turns or []

        assistant_turn = _extract_last_assistant(turns)
        if assistant_turn is None:
            return _zero_result("no_assistant_turn")

        expression = extract_expression(assistant_turn.content or "")
        if not expression:
            return _zero_result("no_expression")

        target_expression = context.get("target_expression")
        derived_variables = dict(context.get("derived_variables") or {})
        constraints = list(context.get("constraints") or [])

        samples = generate_samples(
            variables=variables,
            derived_variables=derived_variables,
            num_samples=self.config.num_samples,
            sample_range=self.config.sample_range,
            seed=_sample_seed(context),
            allowed_functions=self.allowed_functions,
        )

        try:
            candidate_values = [
                safe_eval_expression(expression, sample, self.allowed_functions)
                for sample in samples
            ]
            target_values = None
            if target_expression:
                target_values = [
                    safe_eval_expression(
                        target_expression, sample, self.allowed_functions
                    )
                    for sample in samples
                ]
        except Exception as exc:
            logger.debug("Expression evaluation failed: %s", exc)
            return _zero_result("evaluation_failed", expression=expression)

        names = [*variables, *derived_variables]
        constant_target = target_values is not None and is_constant_expression(
            str(target_expression), target_values, names
        )
        if not constant_target and is_constant_expression(
            expression, candidate_values, names
        ):
            # A constant satisfies symmetry, parity and zero constraints
            # trivially; it earns no validity and nothing can lift it.
            return _zero_result("trivial_expression", expression=expression)

        components: dict[str, float] = {}
        component_weights = dict(self.config.component_weights)

        components["validity"] = 1.0

        if target_expression and target_values is not None:
            matches = sum(
                1
                for candidate, target in zip(
                    candidate_values, target_values, strict=False
                )
                if _close_enough(
                    candidate, target, self.config.rel_tol, self.config.abs_tol
                )
            )
            components["equivalence"] = matches / max(1, len(target_values))
        else:
            component_weights.pop("equivalence", None)

        constraint_scores = {
            f"constraint_{idx}": self._constraint_score(
                constraint,
                expression,
                samples,
                variables,
                derived_variables,
                candidate_values,
            )
            for idx, constraint in enumerate(constraints)
        }
        if constraint_scores:
            components["constraints"] = sum(constraint_scores.values()) / len(
                constraint_scores
            )
        else:
            component_weights.pop("constraints", None)

        length = len(expression)
        if length <= self.config.max_expression_length:
            simplicity = 1.0
        else:
            over = length - self.config.max_expression_length
            simplicity = max(0.0, 1.0 - over / self.config.max_expression_length)
        components["simplicity"] = float(simplicity)

        score = _weighted_average(components, component_weights)
        score = max(0.0, min(1.0, score))

        return RewardResult(
            score=score,
            components=components,
            metadata={
                "expression": expression,
                "constraint_scores": constraint_scores,
                "num_samples": len(samples),
            },
        )

    def _constraint_score(
        self,
        constraint: Mapping[str, Any],
        expression: str,
        samples: list[dict[str, float]],
        variables: list[str],
        derived_variables: Mapping[str, str],
        candidate_values: list[float],
    ) -> float:
        """The share of samples on which ``expression`` keeps ``constraint``
        (0 for an unsupported constraint type)."""
        rel_tol, abs_tol = self.config.rel_tol, self.config.abs_tol
        ctype = str(constraint.get("type", "")).lower()
        if ctype in {"symmetry", "antisymmetry"}:
            sign = float(
                constraint.get("sign", -1.0 if ctype == "antisymmetry" else 1.0)
            )
            return _symmetry_score(
                expression,
                samples,
                constraint.get("pairs", []),
                sign,
                rel_tol,
                abs_tol,
                self.allowed_functions,
                derived_variables,
            )
        if ctype == "zeros":
            return _zeros_score(
                expression,
                variables,
                constraint.get("points", []),
                derived_variables,
                rel_tol,
                abs_tol,
                self.allowed_functions,
            )
        if ctype == "scale":
            factors = (
                constraint.get("factors")
                or constraint.get("scale_factors")
                or [0.5, 2.0]
            )
            return _scale_score(
                expression,
                samples,
                variables,
                derived_variables,
                list(factors),
                float(constraint.get("power", 0.0)),
                rel_tol,
                abs_tol,
                self.allowed_functions,
            )
        if ctype == "sign_flip":
            return self._sign_flip_constraint(
                constraint, expression, samples, derived_variables
            )
        if ctype == "sum":
            weights = constraint.get("weights") or constraint.get("coefficients")
            return _sum_score(
                expression,
                samples,
                [str(term) for term in constraint.get("terms") or []],
                [float(w) for w in weights] if weights else None,
                rel_tol,
                abs_tol,
                self.allowed_functions,
                candidate_values=candidate_values,
            )
        if ctype == "product":
            exponents = constraint.get("exponents")
            return _product_score(
                expression,
                samples,
                [str(factor) for factor in constraint.get("factors") or []],
                [float(e) for e in exponents] if exponents else None,
                rel_tol,
                abs_tol,
                self.allowed_functions,
                candidate_values=candidate_values,
            )
        logger.debug("Unsupported constraint type: %s", ctype)
        return 0.0

    def _sign_flip_constraint(
        self,
        constraint: Mapping[str, Any],
        expression: str,
        samples: list[dict[str, float]],
        derived_variables: Mapping[str, str],
    ) -> float:
        variables_to_flip = constraint.get("variables")
        if variables_to_flip is None:
            single = constraint.get("variable")
            variables_to_flip = [single] if single else []
        flipped = [str(item) for item in variables_to_flip if item is not None]
        parity = str(constraint.get("parity", "")).lower().strip()
        if parity in {"odd", "negative"}:
            sign = -1.0
        elif parity in {"even", "positive"}:
            sign = 1.0
        else:
            sign = float(constraint.get("sign", 1.0))
        return _sign_flip_score(
            expression,
            samples,
            flipped,
            sign,
            self.config.rel_tol,
            self.config.abs_tol,
            self.allowed_functions,
            derived_variables,
        )
