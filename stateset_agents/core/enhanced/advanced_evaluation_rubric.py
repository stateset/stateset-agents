"""Deterministic rubric checks with explicit coverage and inference methods."""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

CriterionCheck = Callable[[str], bool | None]


@dataclass(frozen=True)
class _Criterion:
    check: CriterionCheck
    source: str


def _greeting(response: str) -> bool:
    return re.search(r"\b(?:hello|hi|hey)\b", response.casefold()) is not None


def _polite(response: str) -> bool:
    return (
        re.search(r"\b(?:please|thank you|appreciate)\b", response.casefold())
        is not None
    )


class EvaluationRubric:
    """Versioned, synchronous checks for response characteristics.

    Checks must be fast, deterministic functions without I/O. Return a bool for
    an assessed criterion or None to abstain. A declared method is not validation
    of its quality; all resulting response-quality scores remain inferred.
    """

    def __init__(self) -> None:
        self._criteria = {
            "greeting": _Criterion(_greeting, "greeting_word_presence/v2"),
            "polite": _Criterion(_polite, "polite_phrase_presence/v2"),
        }

    def register_criterion(
        self, name: str, check: CriterionCheck, *, source: str
    ) -> None:
        """Register a synchronous check and its versioned method before a suite runs."""
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise ValueError("Criterion names must be nonempty and trimmed")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Criterion checks require a nonempty source method")
        if (
            not callable(check)
            or inspect.iscoroutinefunction(check)
            or inspect.isasyncgenfunction(check)
        ):
            raise ValueError("Criterion checks must be synchronous callables")
        self._criteria[name] = _Criterion(check, source)

    def snapshot(self) -> EvaluationRubric:
        """Capture registrations; callers must not mutate checker behavior mid-run."""
        rubric = EvaluationRubric()
        rubric._criteria = dict(self._criteria)
        return rubric

    def capabilities(self) -> dict[str, Any]:
        """Describe supported checks without treating heuristics as factual judges."""
        return {
            "characteristics": {
                name: {"source": criterion.source}
                for name, criterion in self._criteria.items()
            },
            "expected_answer": {"source": "expected_answer_substring/v1"},
            "score_policy": "characteristic_fraction_with_answer_floor/v1",
        }

    @staticmethod
    def validate_case(test_case: dict[str, Any]) -> None:
        """Reject ambiguous rubric configuration before invoking an agent."""
        names = test_case.get("expected_characteristics", [])
        if (
            not isinstance(names, (list, tuple))
            or any(
                not isinstance(name, str) or not name.strip() or name != name.strip()
                for name in names
            )
            or len(names) != len(set(names))
        ):
            raise ValueError("Expected characteristics must be unique, nonempty names")
        if "expected_answer" in test_case and (
            not isinstance(test_case["expected_answer"], str)
            or not test_case["expected_answer"].strip()
        ):
            raise ValueError("Expected answers must be nonempty strings")

    def evaluate(self, response: str, test_case: dict[str, Any]) -> dict[str, Any]:
        """Assess supported criteria, retaining unavailable/error outcomes separately."""
        self.validate_case(test_case)
        if not isinstance(response, str):
            raise TypeError("Rubric evaluation requires a string response")
        criteria: list[dict[str, Any]] = []
        names = tuple(test_case.get("expected_characteristics", []))
        expected_answer = test_case.get("expected_answer")
        # Keep expectations/registrations stable through checker callbacks.
        registered = dict(self._criteria)
        for name in names:
            criterion = registered.get(name)
            item: dict[str, Any] = {
                "kind": "characteristic",
                "name": name,
                "status": "unavailable",
                "passed": None,
                "source": criterion.source if criterion else None,
                "reason": "No checker is registered for this criterion",
            }
            if criterion is not None:
                try:
                    verdict: Any = criterion.check(response)
                    if verdict is not None and type(verdict) is not bool:
                        if inspect.iscoroutine(verdict):
                            verdict.close()
                        raise TypeError("Criterion checks must return bool or None")
                except Exception as error:
                    item.update(
                        status="error", reason=f"{type(error).__name__}: {error}"
                    )
                else:
                    item.update(
                        status=(
                            "unavailable"
                            if verdict is None
                            else "passed" if verdict else "failed"
                        ),
                        passed=verdict,
                        reason="The checker abstained" if verdict is None else None,
                    )
            criteria.append(item)
        if expected_answer is not None:
            matched = expected_answer.lower() in response.lower()
            criteria.append(
                {
                    "kind": "expected_answer",
                    "name": "expected_answer",
                    "status": "passed" if matched else "failed",
                    "passed": matched,
                    "source": "expected_answer_substring/v1",
                    "reason": None,
                }
            )
        matched_names = [
            item["name"]
            for item in criteria
            if item["kind"] == "characteristic" and item["passed"] is True
        ]
        score = len(matched_names) / len(names) if names else 0.0
        if any(
            item["kind"] == "expected_answer" and item["passed"] is True
            for item in criteria
        ):
            score = max(score, 0.8)
        assessed = sum(type(item["passed"]) is bool for item in criteria)
        errors = sum(item["status"] == "error" for item in criteria)
        complete = bool(criteria) and assessed == len(criteria)
        issues = []
        if len(response) < 10:
            issues.append("Response too short")
        elif len(response) > 1000:
            issues.append("Response too long")
        if not criteria:
            issues.append("No rubric criteria were configured")
        return {
            "schema_version": 1,
            "status": "assessed" if complete else "error" if errors else "unavailable",
            "overall_score": score,
            "score_available": complete,
            "score_policy": "characteristic_fraction_with_answer_floor/v1",
            "characteristics_matched": matched_names,
            "criteria": criteria,
            "coverage": {
                "total": len(criteria),
                "assessed": assessed,
                "unavailable": len(criteria) - assessed - errors,
                "errors": errors,
                "complete": complete,
            },
            "issues": issues,
        }
