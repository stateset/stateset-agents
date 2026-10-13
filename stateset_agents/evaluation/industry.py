"""Dependency-free, paired reference evaluation for prepared industry projects.

Each assistant turn is evaluated with the preceding reference history. This is
teacher-forced reference agreement, not an autonomous task-success benchmark or
a semantic judge. Model identities in imported predictions are declarations,
not attestations that a particular checkpoint actually generated the responses.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from stateset_agents.data.finetuning import (
    group_finetuning_data,
    validate_finetuning_row,
)
from stateset_agents.training.industry import load_industry_project

SCORER_VERSION = "industry-reference-v1"
Predictor = Callable[[list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


@dataclass(frozen=True)
class IndustryEvaluationPolicy:
    """Explicit reference-agreement and resource limits for a comparison gate."""

    min_groups: int = 30
    min_success_rate: float = 0.9
    min_improvement: float = 0.0
    max_regression_rate: float = 0.0
    max_mean_latency_seconds: float | None = None
    max_mean_cost_usd: float | None = None

    def __post_init__(self) -> None:
        if type(self.min_groups) is not int or self.min_groups < 1:
            raise ValueError("min_groups must be a positive integer")
        for key in ("min_success_rate", "min_improvement", "max_regression_rate"):
            if _number(getattr(self, key), key) > 1:
                raise ValueError(f"{key} must be between 0 and 1")
        for key in ("max_mean_latency_seconds", "max_mean_cost_usd"):
            if getattr(self, key) is not None:
                _number(getattr(self, key), key)


def _suite(project: str | Path) -> dict[str, Any]:
    manifest, train, validation = load_industry_project(project)
    cases = []
    for group in group_finetuning_data(validation):
        group_id = _hash(group)
        for row in group:
            for index, message in enumerate(row["messages"]):
                if message["role"] != "assistant":
                    continue
                cases.append(
                    {
                        "case_id": _hash({"row": row, "turn": index}),
                        "group_id": group_id,
                        "messages": copy.deepcopy(row["messages"][:index]),
                        "tools": copy.deepcopy(row.get("tools", [])),
                        "expected": copy.deepcopy(message),
                    }
                )
    return {
        "schema_version": 1,
        "kind": "stateset-industry-evaluation-requests",
        "scorer_version": SCORER_VERSION,
        "suite_sha256": _hash({"manifest": manifest, "scorer": SCORER_VERSION}),
        "base_model": manifest["base_model"],
        "model_revision": manifest.get("model_revision"),
        "industry": manifest["industry"],
        "synthetic_rows": sum(
            row.get("synthetic") is True for row in train + validation
        ),
        "cases": sorted(cases, key=lambda case: case["case_id"]),
    }


def export_industry_evaluation(project: str | Path) -> dict[str, Any]:
    """Export prediction requests without the current or future target answers.

    Earlier reference assistant turns and tool results remain in each prefix.
    Cases from the same source group must not be treated as independent samples.
    """
    suite = _suite(project)
    for case in suite["cases"]:
        del case["expected"]
    return suite


def _validate_identity(
    model: Any, settings: Any, variant: str, suite: dict[str, Any]
) -> None:
    if variant not in {"baseline", "candidate"}:
        raise ValueError("variant must be baseline or candidate")
    if not isinstance(model, dict) or model.get("base_model") != suite["base_model"]:
        raise ValueError("Model identity must match the project's base_model")
    if not isinstance(model.get("revision"), str) or not model["revision"].strip():
        raise ValueError(
            "Model revision is required; use an immutable checkpoint revision"
        )
    if (
        suite.get("model_revision") is not None
        and model["revision"] != suite["model_revision"]
    ):
        raise ValueError(
            "Model revision differs from the prepared project's pinned revision"
        )
    adapter = model.get("adapter")
    if variant == "baseline" and adapter is not None:
        raise ValueError("The baseline must not declare an adapter")
    if variant == "candidate":
        if (
            not isinstance(adapter, dict)
            or not isinstance(adapter.get("id"), str)
            or not adapter["id"].strip()
        ):
            raise ValueError("Candidate adapter identity is required")
        digest = adapter.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("Candidate adapter sha256 is required")
    if not isinstance(settings, dict) or not settings:
        raise ValueError("Nonempty generation settings are required")
    _canonical({"model": model, "settings": settings})


def collect_industry_predictions(
    project: str | Path,
    predictor: Predictor,
    *,
    variant: str,
    model: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Run a model callback on every held-out prefix, retaining failed requests.

    The callback receives detached messages and tools, never targets. It returns
    ``{"response": <assistant message>, "generated_tokens": int | None,
    "cost_usd": float | None}``. Timing is measured here; missing usage/cost is
    unknown, never zero. Model loading and decoding settings belong to the caller.
    """
    suite = export_industry_evaluation(project)
    _validate_identity(model, settings, variant, suite)
    bundle = {
        "schema_version": 1,
        "kind": "stateset-industry-predictions",
        "suite_sha256": suite["suite_sha256"],
        "variant": variant,
        "model": copy.deepcopy(model),
        "settings": copy.deepcopy(settings),
        "predictions": [],
    }
    predictions: list[dict[str, Any]] = []
    for case in suite["cases"]:
        started = time.perf_counter()
        try:
            result = predictor(
                copy.deepcopy(case["messages"]), copy.deepcopy(case["tools"])
            )
            if not isinstance(result, dict) or "response" not in result:
                raise ValueError("Predictor must return an object containing response")
            result = copy.deepcopy(result)
        except Exception as exc:
            result = {"response": None, "error": f"{type(exc).__name__}: {exc}"}
        predictions.append(
            {
                "case_id": case["case_id"],
                "response": result["response"],
                "error": result.get("error"),
                "elapsed_seconds": time.perf_counter() - started,
                "generated_tokens": result.get("generated_tokens"),
                "cost_usd": result.get("cost_usd"),
            }
        )
    bundle["predictions"] = predictions
    return bundle


def _validate_bundle(
    bundle: Any, suite: dict[str, Any], variant: str
) -> dict[str, dict[str, Any]]:
    if (
        not isinstance(bundle, dict)
        or type(bundle.get("schema_version")) is not int
        or bundle["schema_version"] != 1
    ):
        raise ValueError("Expected prediction bundle schema_version 1")
    if (
        bundle.get("kind") != "stateset-industry-predictions"
        or bundle.get("variant") != variant
    ):
        raise ValueError(f"Expected {variant} prediction bundle")
    if bundle.get("suite_sha256") != suite["suite_sha256"]:
        raise ValueError("Predictions do not match the prepared evaluation suite")
    _validate_identity(bundle.get("model"), bundle.get("settings"), variant, suite)
    predictions = bundle.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("predictions must be a list")
    indexed = {}
    for row in predictions:
        if not isinstance(row, dict):
            raise ValueError("Each prediction must be an object")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id in indexed:
            raise ValueError("Prediction case_id must be a unique string")
        error = row.get("error")
        if error is not None and (not isinstance(error, str) or not error.strip()):
            raise ValueError("Prediction error must be nonempty text or null")
        if "response" not in row:
            raise ValueError("Each prediction requires response (null for failures)")
        _number(row.get("elapsed_seconds"), "elapsed_seconds")
        tokens = row.get("generated_tokens")
        if tokens is not None and (type(tokens) is not int or tokens < 0):
            raise ValueError("generated_tokens must be a nonnegative integer or null")
        if row.get("cost_usd") is not None:
            _number(row["cost_usd"], "cost_usd")
        indexed[case_id] = row
    if set(indexed) != {case["case_id"] for case in suite["cases"]}:
        raise ValueError(
            "Predictions must cover every case exactly once; missing/extra cases found"
        )
    _canonical(bundle)
    return indexed


def _calls(message: dict[str, Any]) -> str:
    calls = []
    for call in message.get("tool_calls", []):
        function = call["function"]
        arguments = function["arguments"]
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        calls.append({"name": function["name"], "arguments": arguments})
    # Preserve call order and JSON value types; generated transport IDs may differ.
    return _canonical(calls)


def _score(case: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    expected = case["expected"]
    response = prediction["response"]
    reason = prediction.get("error")
    valid = reason is None
    if valid:
        try:
            if not isinstance(response, dict) or response.get("role") != "assistant":
                raise ValueError("Response must be an assistant message")
            validate_finetuning_row(
                {"messages": case["messages"] + [response], "tools": case["tools"]}
            )
        except (ValueError, TypeError) as exc:
            valid, reason = False, str(exc)
    tool_match = valid and _calls(response) == _calls(expected)
    text_match = valid and " ".join(
        (response.get("content") or "").split()
    ) == " ".join((expected.get("content") or "").split())
    return {
        "case_id": case["case_id"],
        "group_id": case["group_id"],
        "reference_match": bool(tool_match and text_match),
        "tool_case": bool(expected.get("tool_calls")),
        "tool_match": bool(tool_match),
        "text_match": bool(text_match),
        "valid_response": valid,
        "error": reason,
    }


def _summarize(
    scores: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    groups: dict[str, list[bool]] = {}
    for score in scores:
        groups.setdefault(score["group_id"], []).append(score["reference_match"])
    group_results = {key: all(values) for key, values in groups.items()}
    tools = [score for score in scores if score["tool_case"]]
    texts = [score for score in scores if not score["tool_case"]]
    costs = [row.get("cost_usd") for row in predictions.values()]
    tokens = [row.get("generated_tokens") for row in predictions.values()]
    return {
        "cases": len(scores),
        "groups": len(groups),
        "group_reference_match_rate": sum(group_results.values()) / len(groups),
        "case_reference_match_rate": sum(s["reference_match"] for s in scores)
        / len(scores),
        "tool_cases": len(tools),
        "tool_exact_match_rate": (
            sum(s["tool_match"] for s in tools) / len(tools) if tools else None
        ),
        "text_cases": len(texts),
        "text_case_match_rate": (
            sum(s["reference_match"] for s in texts) / len(texts) if texts else None
        ),
        "invalid_responses": sum(not s["valid_response"] for s in scores),
        "mean_latency_seconds": statistics.mean(
            row["elapsed_seconds"] for row in predictions.values()
        ),
        "mean_cost_usd": (
            statistics.mean(c for c in costs if c is not None)
            if all(c is not None for c in costs)
            else None
        ),
        "generated_tokens": sum(tokens) if all(t is not None for t in tokens) else None,
        "group_results": group_results,
    }


def evaluate_industry_project(
    project: str | Path,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    policy: IndustryEvaluationPolicy | None = None,
) -> dict[str, Any]:
    """Score paired predictions and apply a fail-closed reference regression gate.

    Missing, duplicate, stale, or incomparable predictions raise ValueError.
    Valid bundles with poor quality, synthetic data, insufficient groups, or
    unknown required costs produce a failed gate with explicit reasons.
    """
    policy = policy or IndustryEvaluationPolicy()
    suite = _suite(project)
    base_rows = _validate_bundle(baseline, suite, "baseline")
    candidate_rows = _validate_bundle(candidate, suite, "candidate")
    if baseline["model"]["revision"] != candidate["model"]["revision"]:
        raise ValueError("Base model revisions differ")
    if _canonical(baseline["settings"]) != _canonical(candidate["settings"]):
        raise ValueError("Generation settings differ")
    base_scores = [_score(case, base_rows[case["case_id"]]) for case in suite["cases"]]
    candidate_scores = [
        _score(case, candidate_rows[case["case_id"]]) for case in suite["cases"]
    ]
    base = _summarize(base_scores, base_rows)
    adapted = _summarize(candidate_scores, candidate_rows)
    regressions = [
        old["case_id"]
        for old, new in zip(base_scores, candidate_scores, strict=True)
        if old["reference_match"] and not new["reference_match"]
    ]
    improvements = [
        old["case_id"]
        for old, new in zip(base_scores, candidate_scores, strict=True)
        if not old["reference_match"] and new["reference_match"]
    ]
    delta = adapted["group_reference_match_rate"] - base["group_reference_match_rate"]
    reasons = []
    if suite["synthetic_rows"]:
        reasons.append("synthetic_data: replace demonstrations with reviewed data")
    if adapted["groups"] < policy.min_groups:
        reasons.append("insufficient_groups")
    if adapted["group_reference_match_rate"] < policy.min_success_rate:
        reasons.append("candidate_below_reference_match_floor")
    if delta + 1e-12 < policy.min_improvement:
        reasons.append("insufficient_improvement")
    if len(regressions) / adapted["cases"] > policy.max_regression_rate:
        reasons.append("case_regression_limit_exceeded")
    for metric, limit in (
        ("mean_latency_seconds", policy.max_mean_latency_seconds),
        ("mean_cost_usd", policy.max_mean_cost_usd),
    ):
        if limit is not None:
            if adapted[metric] is None:
                reasons.append(f"unknown_{metric}")
            elif adapted[metric] > limit:
                reasons.append(f"{metric}_limit_exceeded")
    report = {
        "schema_version": 1,
        "kind": "stateset-industry-evaluation",
        "scorer_version": SCORER_VERSION,
        "suite_sha256": suite["suite_sha256"],
        "industry": suite["industry"],
        "evaluation_mode": "teacher_forced_reference_agreement",
        "identity_verification": "caller_declared",
        "prediction_sha256": {
            "baseline": _hash(baseline),
            "candidate": _hash(candidate),
        },
        "models": {"baseline": baseline["model"], "candidate": candidate["model"]},
        "settings": baseline["settings"],
        "policy": asdict(policy),
        "baseline": base,
        "candidate": adapted,
        "group_reference_match_delta": delta,
        "regressed_case_ids": regressions,
        "improved_case_ids": improvements,
        "gate": {"passed": not reasons, "reasons": reasons},
        "outcomes": {"baseline": base_scores, "candidate": candidate_scores},
        "limitations": [
            "Exact reference agreement does not measure semantic quality, clinical safety, or business outcomes.",
            "Reference history is supplied at each turn; errors do not propagate as in a live conversation.",
            "Group boundaries and synthetic flags depend on supplied metadata; semantic leakage is not detected.",
            "Passing configured thresholds is not a statistical significance claim or production certification.",
        ],
    }
    _canonical(report)
    return copy.deepcopy(report)
