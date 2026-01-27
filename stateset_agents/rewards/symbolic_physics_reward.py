"""
Reward function for symbolic physics relation discovery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

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
    sample_range: Tuple[float, float] = (-2.0, 2.0)
    rel_tol: float = 1e-2
    abs_tol: float = 1e-3
    max_expression_length: int = 140
    component_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "validity": 0.1,
            "equivalence": 0.6,
            "constraints": 0.2,
            "simplicity": 0.1,
        }
    )
    allowed_functions: Optional[Mapping[str, Callable[..., float]]] = None


def _close_enough(a: float, b: float, rel_tol: float, abs_tol: float) -> bool:
    return abs(a - b) <= abs_tol + rel_tol * max(abs(a), abs(b))


def _weighted_average(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    total = 0.0
    weighted = 0.0
    for name, value in scores.items():
        weight = float(weights.get(name, 1.0))
        total += weight
        weighted += weight * value
    if total <= 0:
        return 0.0
    return weighted / total


def _extract_last_assistant(turns: List[Any]) -> Optional[ConversationTurn]:
    for item in reversed(turns):
        if isinstance(item, ConversationTurn) and item.role == "assistant":
            return item
        role = getattr(item, "role", None)
        content = getattr(item, "content", None)
        if role == "assistant" and content is not None:
            return ConversationTurn(role="assistant", content=str(content))
        if isinstance(item, dict) and item.get("role") == "assistant":
            return ConversationTurn(role="assistant", content=str(item.get("content", "")))
    return None


def _symmetry_score(
    expression: str,
    samples: List[Dict[str, float]],
    pairs: List[List[str]],
    sign: float,
    rel_tol: float,
    abs_tol: float,
    allowed_functions: Mapping[str, Callable[..., float]],
) -> float:
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
            swapped[a], swapped[b] = swapped.get(b), swapped.get(a)
            try:
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
    variables: List[str],
    points: List[Dict[str, float]],
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
        values = apply_derived_variables(base_values, derived_variables, allowed_functions)
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
    samples: List[Dict[str, float]],
    variables: List[str],
    derived_variables: Mapping[str, str],
    factors: List[float],
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
                scaled_val = safe_eval_expression(
                    expression, scaled, allowed_functions
                )
            except Exception:
                total += 1
                continue
            target = (factor ** power) * base_val
            if _close_enough(scaled_val, target, rel_tol, abs_tol):
                matches += 1
            total += 1
    return matches / total if total else 0.0


def _sign_flip_score(
    expression: str,
    samples: List[Dict[str, float]],
    variables: List[str],
    sign: float,
    rel_tol: float,
    abs_tol: float,
    allowed_functions: Mapping[str, Callable[..., float]],
) -> float:
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
    samples: List[Dict[str, float]],
    terms: List[str],
    weights: Optional[List[float]],
    rel_tol: float,
    abs_tol: float,
    allowed_functions: Mapping[str, Callable[..., float]],
    candidate_values: Optional[List[float]] = None,
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
            for term, weight in zip(terms, padded_weights):
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
    samples: List[Dict[str, float]],
    factors: List[str],
    exponents: Optional[List[float]],
    rel_tol: float,
    abs_tol: float,
    allowed_functions: Mapping[str, Callable[..., float]],
    candidate_values: Optional[List[float]] = None,
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
            for factor, exponent in zip(factors, padded_exponents):
                factor_val = safe_eval_expression(factor, values, allowed_functions)
                target *= factor_val ** exponent
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
    """Reward function that scores symbolic expressions against constraints."""

    def __init__(
        self,
        config: Optional[SymbolicRewardConfig] = None,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight=weight, reward_type=RewardType.IMMEDIATE, name="SymbolicPhysicsReward")
        self.config = config or SymbolicRewardConfig()
        self.allowed_functions = (
            dict(self.config.allowed_functions)
            if self.config.allowed_functions
            else dict(DEFAULT_ALLOWED_FUNCTIONS)
        )

    async def compute_reward(  # type: ignore[override]
        self,
        turns: Optional[List[Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        trajectory: Optional[Any] = None,
        turn: Optional[Any] = None,
        **_: Any,
    ) -> RewardResult:
        context = context or {}
        if turns is None and turn is not None:
            turns = [turn]
        turns = turns or []

        assistant_turn = _extract_last_assistant(turns)
        if assistant_turn is None:
            result = RewardResult(score=0.0, metadata={"error": "no_assistant_turn"})
            setattr(result, "total_reward", 0.0)
            return result

        expression = extract_expression(assistant_turn.content)
        if not expression:
            result = RewardResult(score=0.0, metadata={"error": "no_expression"})
            setattr(result, "total_reward", 0.0)
            return result

        variables = list(context.get("variables") or [])
        target_expression = context.get("target_expression")
        derived_variables = dict(context.get("derived_variables") or {})
        constraints = list(context.get("constraints") or [])

        if not variables:
            result = RewardResult(score=0.0, metadata={"error": "no_variables"})
            setattr(result, "total_reward", 0.0)
            return result

        samples = generate_samples(
            variables=variables,
            derived_variables=derived_variables,
            num_samples=self.config.num_samples,
            sample_range=self.config.sample_range,
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
                    safe_eval_expression(target_expression, sample, self.allowed_functions)
                    for sample in samples
                ]
        except Exception as exc:
            logger.debug("Expression evaluation failed: %s", exc)
            result = RewardResult(
                score=0.0,
                metadata={"error": "evaluation_failed", "expression": expression},
            )
            setattr(result, "total_reward", 0.0)
            return result

        components: Dict[str, float] = {}
        component_weights = dict(self.config.component_weights)

        components["validity"] = 1.0

        if target_expression and target_values is not None:
            matches = sum(
                1
                for candidate, target in zip(candidate_values, target_values)
                if _close_enough(candidate, target, self.config.rel_tol, self.config.abs_tol)
            )
            components["equivalence"] = matches / max(1, len(target_values))
        else:
            component_weights.pop("equivalence", None)

        constraint_scores: Dict[str, float] = {}
        if constraints:
            for idx, constraint in enumerate(constraints):
                ctype = str(constraint.get("type", "")).lower()
                score = 0.0
                if ctype in {"symmetry", "antisymmetry"}:
                    pairs = constraint.get("pairs", [])
                    sign = float(
                        constraint.get(
                            "sign", -1.0 if ctype == "antisymmetry" else 1.0
                        )
                    )
                    score = _symmetry_score(
                        expression,
                        samples,
                        pairs,
                        sign,
                        self.config.rel_tol,
                        self.config.abs_tol,
                        self.allowed_functions,
                    )
                elif ctype == "zeros":
                    points = constraint.get("points", [])
                    score = _zeros_score(
                        expression,
                        variables,
                        points,
                        derived_variables,
                        self.config.rel_tol,
                        self.config.abs_tol,
                        self.allowed_functions,
                    )
                elif ctype == "scale":
                    factors = constraint.get("factors") or constraint.get("scale_factors") or [0.5, 2.0]
                    power = float(constraint.get("power", 0.0))
                    score = _scale_score(
                        expression,
                        samples,
                        variables,
                        derived_variables,
                        list(factors),
                        power,
                        self.config.rel_tol,
                        self.config.abs_tol,
                        self.allowed_functions,
                    )
                elif ctype == "sign_flip":
                    variables_to_flip = constraint.get("variables")
                    if variables_to_flip is None:
                        single = constraint.get("variable")
                        variables_to_flip = [single] if single else []
                    variables_to_flip = [
                        str(item) for item in variables_to_flip if item is not None
                    ]
                    parity = str(constraint.get("parity", "")).lower().strip()
                    if parity in {"odd", "negative"}:
                        sign = -1.0
                    elif parity in {"even", "positive"}:
                        sign = 1.0
                    else:
                        sign = float(constraint.get("sign", 1.0))
                    score = _sign_flip_score(
                        expression,
                        samples,
                        variables_to_flip,
                        sign,
                        self.config.rel_tol,
                        self.config.abs_tol,
                        self.allowed_functions,
                    )
                elif ctype == "sum":
                    terms = list(constraint.get("terms") or [])
                    weights = constraint.get("weights") or constraint.get("coefficients")
                    score = _sum_score(
                        expression,
                        samples,
                        [str(term) for term in terms],
                        [float(w) for w in weights] if weights else None,
                        self.config.rel_tol,
                        self.config.abs_tol,
                        self.allowed_functions,
                        candidate_values=candidate_values,
                    )
                elif ctype == "product":
                    factors = list(constraint.get("factors") or [])
                    exponents = constraint.get("exponents")
                    score = _product_score(
                        expression,
                        samples,
                        [str(factor) for factor in factors],
                        [float(e) for e in exponents] if exponents else None,
                        self.config.rel_tol,
                        self.config.abs_tol,
                        self.allowed_functions,
                        candidate_values=candidate_values,
                    )
                else:
                    logger.debug("Unsupported constraint type: %s", ctype)
                constraint_scores[f"constraint_{idx}"] = float(score)

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

        result = RewardResult(
            score=score,
            components=components,
            metadata={
                "expression": expression,
                "constraint_scores": constraint_scores,
                "num_samples": len(samples),
            },
        )
        setattr(result, "total_reward", float(score))
        return result
