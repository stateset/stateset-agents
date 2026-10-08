"""Exercise reward functions on reviewed candidate groups before RL training.

This audit measures reward behavior on supplied examples, not policy gradients,
model quality, or performance on an independently held-out evaluation set.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


class AuditableReward(Protocol):
    """The asynchronous reward interface used by native StateSet trainers."""

    async def compute_reward(
        self, turns: list[Any], context: dict[str, Any] | None = None
    ) -> Any:
        """Return an object exposing a finite numeric ``score``."""
        ...


@dataclass(frozen=True)
class RewardAuditPolicy:
    """Checks for a deterministic reward on prespecified candidate groups."""

    repeats: int = 3
    score_tolerance: float = 1e-8
    min_informative_fraction: float = 1.0
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if type(self.repeats) is not int or self.repeats < 2:
            raise ValueError("repeats must be an integer of at least two")
        for name in (
            "score_tolerance",
            "min_informative_fraction",
            "timeout_seconds",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        if self.score_tolerance < 0:
            raise ValueError("score_tolerance must be nonnegative")
        if not 0 < self.min_informative_fraction <= 1:
            raise ValueError("min_informative_fraction must be in (0, 1]")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _object(value: Any, required: set[str], optional: set[str], label: str) -> None:
    if not isinstance(value, dict) or not required <= value.keys():
        raise ValueError(f"{label} must contain {sorted(required)}")
    if value.keys() - required - optional:
        raise ValueError(f"{label} contains unknown fields")


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _message(value: Any, *, response: bool = False) -> None:
    _object(
        value,
        {"role", "content"},
        {"tool_calls", "tool_results", "metadata"},
        "message",
    )
    if value["role"] not in ("system", "user", "assistant", "tool"):
        raise ValueError("message role must be system, user, assistant, or tool")
    if response and value["role"] != "assistant":
        raise ValueError("candidate response must have role assistant")
    if value["content"] is not None and not isinstance(value["content"], str):
        raise ValueError("message content must be a string or null")
    for name in ("tool_calls", "tool_results"):
        if name in value and (
            not isinstance(value[name], list)
            or any(not isinstance(item, dict) for item in value[name])
        ):
            raise ValueError(f"{name} must be a list of objects")
    if "metadata" in value and not isinstance(value["metadata"], dict):
        raise ValueError("message metadata must be an object")


def validate_reward_suite(suite: Any) -> dict[str, Any]:
    """Validate and copy JSON cases before invoking any reward code."""
    _object(suite, {"schema_version", "cases"}, {"description"}, "suite")
    if type(suite["schema_version"]) is not int or suite["schema_version"] != 1:
        raise ValueError("reward audit schema_version must be 1")
    if "description" in suite and not isinstance(suite["description"], str):
        raise ValueError("description must be a string")
    if not isinstance(suite["cases"], list) or not suite["cases"]:
        raise ValueError("cases must be a nonempty list")
    case_ids: set[str] = set()
    for case in suite["cases"]:
        _object(
            case,
            {"id", "messages", "context", "candidates", "preferences"},
            set(),
            "case",
        )
        case_id = _identifier(case["id"], "case id")
        if case_id in case_ids:
            raise ValueError(f"duplicate case id: {case_id}")
        case_ids.add(case_id)
        if not isinstance(case["context"], dict):
            raise ValueError("case context must be an object")
        if not isinstance(case["messages"], list) or not case["messages"]:
            raise ValueError("messages must contain a shared prompt")
        for message in case["messages"]:
            _message(message)
        if case["messages"][-1]["role"] not in ("user", "tool"):
            raise ValueError("shared prompt must end with a user or tool message")
        if not isinstance(case["candidates"], list) or len(case["candidates"]) < 2:
            raise ValueError("each case requires at least two candidates")
        candidate_ids: set[str] = set()
        for candidate in case["candidates"]:
            _object(candidate, {"id", "response"}, set(), "candidate")
            candidate_id = _identifier(candidate["id"], "candidate id")
            if candidate_id in candidate_ids:
                raise ValueError(f"duplicate candidate id: {candidate_id}")
            candidate_ids.add(candidate_id)
            _message(candidate["response"], response=True)
        if not isinstance(case["preferences"], list):
            raise ValueError(
                "preferences must be a list of [preferred, rejected] pairs"
            )
        pairs: set[tuple[str, str]] = set()
        for pair in case["preferences"]:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or any(
                    not isinstance(item, str) or item not in candidate_ids
                    for item in pair
                )
                or pair[0] == pair[1]
            ):
                raise ValueError(
                    "preferences must name two distinct candidates in this case"
                )
            if tuple(pair) in pairs:
                raise ValueError("duplicate preference pair")
            pairs.add((pair[0], pair[1]))
    if not any(case["preferences"] for case in suite["cases"]):
        raise ValueError("suite requires at least one reviewed preference pair")
    # Reject non-JSON or non-finite context/metadata and detach caller-owned data.
    detached: dict[str, Any] = json.loads(_canonical(suite))
    return detached


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def load_reward_suite(path: str | Path) -> dict[str, Any]:
    """Read a strict JSON suite, rejecting duplicate fields and invalid cases."""
    return validate_reward_suite(
        json.loads(
            Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object
        )
    )


def _separated(high: float, low: float, tolerance: float) -> bool:
    return high > low and not math.isclose(high, low, rel_tol=0, abs_tol=tolerance)


async def audit_reward(
    reward: AuditableReward,
    suite: dict[str, Any],
    *,
    policy: RewardAuditPolicy | None = None,
    reward_name: str | None = None,
) -> dict[str, Any]:
    """Score every candidate repeatedly, retaining failures and ranking mistakes.

    Inputs are copied before each call. Async timeouts are cooperative; a reward
    that blocks the event loop cannot be interrupted by this in-process audit.
    Cancellation propagates, while ordinary reward exceptions become failed rows.
    """
    policy = policy or RewardAuditPolicy()
    suite = validate_reward_suite(suite)
    if not callable(getattr(reward, "compute_reward", None)):
        raise ValueError("reward must expose async compute_reward(turns, context)")
    from stateset_agents.core.trajectory import ConversationTurn

    rows = []
    for case in suite["cases"]:
        candidates: dict[str, Any] = {}
        for candidate in case["candidates"]:
            scores: list[float | None] = []
            errors: list[dict[str, Any]] = []
            for repeat in range(policy.repeats):
                messages = copy.deepcopy(case["messages"] + [candidate["response"]])
                turns = [ConversationTurn(**message) for message in messages]
                try:
                    result = await asyncio.wait_for(
                        reward.compute_reward(turns, copy.deepcopy(case["context"])),
                        timeout=policy.timeout_seconds,
                    )
                    value = result.score
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise ValueError("reward score must be numeric, not boolean")
                    score = float(value)
                    if not math.isfinite(score):
                        raise ValueError("reward score must be finite")
                    scores.append(score)
                except Exception as exc:
                    scores.append(None)
                    errors.append(
                        {
                            "repeat": repeat,
                            "type": type(exc).__name__,
                            "message": str(exc)[:500],
                        }
                    )
            finite_scores = [score for score in scores if score is not None]
            stable = not errors and not _separated(
                max(finite_scores), min(finite_scores), policy.score_tolerance
            )
            candidates[candidate["id"]] = {
                "scores": scores,
                "errors": errors,
                "stable": stable,
            }
        complete = all(not value["errors"] for value in candidates.values())
        stable = all(value["stable"] for value in candidates.values())
        observed = [
            value["scores"][0] for value in candidates.values() if not value["errors"]
        ]
        informative = (
            complete
            and stable
            and _separated(max(observed), min(observed), policy.score_tolerance)
        )
        preferences = []
        for preferred, rejected in case["preferences"]:
            good, bad = candidates[preferred], candidates[rejected]
            passed = (
                good["stable"]
                and bad["stable"]
                and _separated(
                    min(good["scores"]), max(bad["scores"]), policy.score_tolerance
                )
            )
            preferences.append(
                {"preferred": preferred, "rejected": rejected, "passed": passed}
            )
        rows.append(
            {
                "id": case["id"],
                "candidates": candidates,
                "complete": complete,
                "stable": stable,
                "informative": informative,
                "preferences": preferences,
            }
        )
    informative_count = sum(row["informative"] for row in rows)
    failed_preferences = sum(
        not pair["passed"] for row in rows for pair in row["preferences"]
    )
    reasons = []
    if any(not row["complete"] for row in rows):
        reasons.append("reward_errors")
    if any(row["complete"] and not row["stable"] for row in rows):
        reasons.append("unstable_scores")
    if informative_count / len(rows) < policy.min_informative_fraction:
        reasons.append("insufficient_group_signal")
    if failed_preferences:
        reasons.append("preference_violations")
    return {
        "schema_version": 1,
        "kind": "stateset-reward-audit",
        "reward": reward_name
        or f"{type(reward).__module__}:{type(reward).__qualname__}",
        "suite_sha256": hashlib.sha256(_canonical(suite)).hexdigest(),
        "policy": asdict(policy),
        "passed": not reasons,
        "failure_reasons": reasons,
        "summary": {
            "cases": len(rows),
            "candidates": sum(len(row["candidates"]) for row in rows),
            "informative_groups": informative_count,
            "informative_fraction": informative_count / len(rows),
            "failed_preferences": failed_preferences,
        },
        "cases": rows,
    }
