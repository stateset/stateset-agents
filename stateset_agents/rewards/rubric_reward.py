"""Hierarchical Issue-Group Binary Rubric Reward for Closed-Universe Agent Tasks.

Scores agent deliverables against an expert rubric of binary pass/fail criteria
organized into domain issue groups (e.g., legal issue groups in a client matter):

    R = w_criteria * (criteria_passed / total_criteria)
      + w_groups   * (issue_groups_passed_in_full / total_issue_groups)
      + w_bonus    * I[all criteria passed]

By design this reward includes **no explicit length penalty** (``length_penalty == 0.0``).
Supports both pluggable LLM-as-a-Judge evaluation (:class:`LLMRubricJudge`) and
deterministic offline verification (:class:`DeterministicRubricJudge`).

Zero eager ``torch`` imports.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from stateset_agents.core.reward_base import RewardFunction, RewardResult, RewardType
from stateset_agents.core.trajectory import ConversationTurn

JudgeCompletionFn = Callable[[str], str | Awaitable[str]]


@dataclass(frozen=True)
class RubricCriterion:
    """One binary pass/fail expert rubric criterion belonging to an issue group."""

    criterion_id: str
    description: str
    issue_group: str
    required_keywords: tuple[str, ...] = ()
    forbidden_keywords: tuple[str, ...] = ()
    regex_patterns: tuple[str, ...] = ()
    required_citations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.criterion_id.strip():
            raise ValueError("criterion_id must be non-empty")
        if not self.description.strip():
            raise ValueError("description must be non-empty")
        if not self.issue_group.strip():
            raise ValueError("issue_group must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CriterionVerdict:
    """Binary pass/fail outcome for a single rubric criterion."""

    criterion_id: str
    issue_group: str
    passed: bool
    rationale: str = ""
    judge_source: str = "deterministic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RubricRewardConfig:
    """Weights and invariants for :class:`HierarchicalRubricReward`.

    Default weights sum to ``1.0`` (``0.50`` criteria fraction + ``0.35`` full
    issue-group fraction + ``0.15`` all-criteria bonus) with explicitly zero
    length penalty.
    """

    criteria_weight: float = 0.50
    issue_group_weight: float = 0.35
    all_pass_bonus: float = 0.15
    length_penalty: float = 0.0
    forbid_length_penalty: bool = True
    strict_fail_closed: bool = True

    def __post_init__(self) -> None:
        for fname in (
            "criteria_weight",
            "issue_group_weight",
            "all_pass_bonus",
            "length_penalty",
        ):
            val = float(getattr(self, fname))
            if not math.isfinite(val):
                raise ValueError(f"{fname} must be finite")
            if fname != "length_penalty" and val < 0.0:
                raise ValueError(f"{fname} must be >= 0.0")
        if self.forbid_length_penalty and self.length_penalty != 0.0:
            raise ValueError(
                "length_penalty must be 0.0 when forbid_length_penalty=True"
            )
        total = (
            float(self.criteria_weight)
            + float(self.issue_group_weight)
            + float(self.all_pass_bonus)
        )
        if total <= 0.0:
            raise ValueError("Sum of rubric reward weights must be > 0.0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RubricJudge(Protocol):
    """Protocol for binary pass/fail rubric evaluators (LLM judge or deterministic)."""

    async def evaluate_criteria(
        self,
        criteria: Sequence[RubricCriterion],
        deliverable: str,
        *,
        instruction: str = "",
        context: dict[str, Any] | None = None,
    ) -> list[CriterionVerdict]:
        """Grade every criterion as binary pass/fail against ``deliverable``."""
        ...


class DeterministicRubricJudge:
    """Zero-cost deterministic rubric grader for unit tests, CI, and offline rollouts.

    A criterion passes iff:
    - ``deliverable`` is non-empty,
    - every entry in ``required_keywords`` appears case-insensitively,
    - no entry in ``forbidden_keywords`` appears case-insensitively,
    - every entry in ``regex_patterns`` matches (``re.IGNORECASE | re.DOTALL``), and
    - every entry in ``required_citations`` appears case-insensitively.
    """

    def evaluate_criterion_sync(
        self, criterion: RubricCriterion, deliverable: str
    ) -> CriterionVerdict:
        text = deliverable.strip()
        if not text:
            return CriterionVerdict(
                criterion_id=criterion.criterion_id,
                issue_group=criterion.issue_group,
                passed=False,
                rationale="Deliverable is empty.",
                judge_source="deterministic",
            )
        lowered = text.lower()
        missing_kw = [
            kw for kw in criterion.required_keywords if kw.lower() not in lowered
        ]
        present_forbid = [
            kw for kw in criterion.forbidden_keywords if kw.lower() in lowered
        ]
        missing_cites = [
            cite for cite in criterion.required_citations if cite.lower() not in lowered
        ]
        failed_regex = [
            pat
            for pat in criterion.regex_patterns
            if re.search(pat, text, flags=re.IGNORECASE | re.DOTALL) is None
        ]

        reasons: list[str] = []
        if missing_kw:
            reasons.append(f"missing keywords: {missing_kw}")
        if present_forbid:
            reasons.append(f"forbidden terms present: {present_forbid}")
        if missing_cites:
            reasons.append(f"missing citations: {missing_cites}")
        if failed_regex:
            reasons.append(f"unmatched patterns: {failed_regex}")

        passed = not reasons
        return CriterionVerdict(
            criterion_id=criterion.criterion_id,
            issue_group=criterion.issue_group,
            passed=passed,
            rationale=(
                "All deterministic checks passed." if passed else "; ".join(reasons)
            ),
            judge_source="deterministic",
        )

    def evaluate_criteria_sync(
        self,
        criteria: Sequence[RubricCriterion],
        deliverable: str,
    ) -> list[CriterionVerdict]:
        return [self.evaluate_criterion_sync(c, deliverable) for c in criteria]

    async def evaluate_criteria(
        self,
        criteria: Sequence[RubricCriterion],
        deliverable: str,
        *,
        instruction: str = "",
        context: dict[str, Any] | None = None,
    ) -> list[CriterionVerdict]:
        del instruction, context
        return self.evaluate_criteria_sync(criteria, deliverable)


def _parse_judge_verdict_json(
    raw: str, *, strict_fail_closed: bool
) -> tuple[bool, str]:
    """Parse a binary judge JSON object (or PASS/FAIL fallback) fail-closed."""
    text = raw.strip()
    if not text:
        return False, "Empty judge response (fail-closed)."

    # Strip optional markdown code fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidate = fence_match.group(1) if fence_match else text
    brace_match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if brace_match:
        try:
            parsed = json.loads(brace_match.group(0))
            if isinstance(parsed, dict) and "passed" in parsed:
                val = parsed["passed"]
                if isinstance(val, bool):
                    return val, str(parsed.get("rationale") or "")
                if isinstance(val, str) and val.strip().lower() in {
                    "true",
                    "pass",
                    "passed",
                    "yes",
                }:
                    return True, str(parsed.get("rationale") or "")
                if isinstance(val, str) and val.strip().lower() in {
                    "false",
                    "fail",
                    "failed",
                    "no",
                }:
                    return False, str(parsed.get("rationale") or "")
        except json.JSONDecodeError:
            pass

    if not strict_fail_closed:
        upper = text.upper()
        if re.search(r"\bPASS\b", upper) and not re.search(r"\bFAIL\b", upper):
            return True, text[:240]
    return False, f"Unparseable judge response (fail-closed): {text[:160]}"


class LLMRubricJudge:
    """Binary pass/fail LLM judge evaluating each rubric criterion independently.

    Wraps any synchronous or asynchronous text completion function
    ``judge_fn(prompt: str) -> str`` (e.g., OpenAI, Anthropic, River, or vLLM)
    and parses structured JSON verdicts fail-closed.
    """

    def __init__(
        self,
        judge_fn: JudgeCompletionFn,
        *,
        judge_model: str = "configured-llm-judge",
        max_concurrency: int = 8,
        strict_fail_closed: bool = True,
    ) -> None:
        self._judge_fn = judge_fn
        self.judge_model = judge_model
        self.max_concurrency = max(1, int(max_concurrency))
        self.strict_fail_closed = bool(strict_fail_closed)

    @staticmethod
    def build_criterion_prompt(
        criterion: RubricCriterion,
        deliverable: str,
        *,
        instruction: str = "",
    ) -> str:
        return (
            "You are an expert rubric grader. Grade the deliverable strictly as "
            "binary PASS or FAIL against the single criterion below. Do NOT "
            "penalize or reward response length.\n\n"
            f"Client Instruction:\n{instruction or '(not provided)'}\n\n"
            f"Issue Group: {criterion.issue_group}\n"
            f"Criterion ({criterion.criterion_id}): {criterion.description}\n\n"
            f"Deliverable:\n{deliverable}\n\n"
            'Respond ONLY with a JSON object: {"passed": true|false, "rationale": "<brief reason>"}'
        )

    async def _grade_one(
        self,
        sem: asyncio.Semaphore,
        criterion: RubricCriterion,
        deliverable: str,
        instruction: str,
    ) -> CriterionVerdict:
        if not deliverable.strip():
            return CriterionVerdict(
                criterion_id=criterion.criterion_id,
                issue_group=criterion.issue_group,
                passed=False,
                rationale="Deliverable is empty.",
                judge_source=self.judge_model,
            )
        prompt = self.build_criterion_prompt(
            criterion, deliverable, instruction=instruction
        )
        async with sem:
            try:
                raw_or_coro = self._judge_fn(prompt)
                raw = (
                    await raw_or_coro
                    if inspect.isawaitable(raw_or_coro)
                    else raw_or_coro
                )
            except Exception as exc:
                return CriterionVerdict(
                    criterion_id=criterion.criterion_id,
                    issue_group=criterion.issue_group,
                    passed=False,
                    rationale=f"Judge call error ({type(exc).__name__}); fail-closed.",
                    judge_source=self.judge_model,
                )
        passed, rationale = _parse_judge_verdict_json(
            str(raw), strict_fail_closed=self.strict_fail_closed
        )
        return CriterionVerdict(
            criterion_id=criterion.criterion_id,
            issue_group=criterion.issue_group,
            passed=passed,
            rationale=rationale,
            judge_source=self.judge_model,
        )

    async def evaluate_criteria(
        self,
        criteria: Sequence[RubricCriterion],
        deliverable: str,
        *,
        instruction: str = "",
        context: dict[str, Any] | None = None,
    ) -> list[CriterionVerdict]:
        del context
        sem = asyncio.Semaphore(self.max_concurrency)
        return list(
            await asyncio.gather(
                *(self._grade_one(sem, c, deliverable, instruction) for c in criteria)
            )
        )


def score_rubric_verdicts(
    verdicts: Sequence[CriterionVerdict],
    config: RubricRewardConfig | None = None,
) -> dict[str, Any]:
    """Compute the exact hierarchical issue-group rubric reward from binary verdicts.

    Formula:
        ``R = w_criteria * (passed / total) + w_groups * (groups_all_passed / total_groups) + w_bonus * I[all_passed]``

    Explicitly applies zero length penalty.
    """
    cfg = config or RubricRewardConfig()
    if not verdicts:
        return {
            "reward": 0.0,
            "criteria_passed": 0,
            "criteria_total": 0,
            "criteria_fraction": 0.0,
            "issue_groups_passed": 0,
            "issue_groups_total": 0,
            "issue_group_fraction": 0.0,
            "all_criteria_passed": False,
            "all_pass_bonus_awarded": 0.0,
            "length_penalty": 0.0,
            "issue_groups": {},
            "verdicts": [],
        }

    total_criteria = len(verdicts)
    passed_criteria = sum(1 for v in verdicts if v.passed)
    criteria_fraction = passed_criteria / total_criteria

    grouped: dict[str, list[CriterionVerdict]] = {}
    for verdict in verdicts:
        grouped.setdefault(verdict.issue_group, []).append(verdict)

    issue_groups_summary: dict[str, dict[str, Any]] = {}
    groups_passed_in_full = 0
    for group_name, items in grouped.items():
        grp_passed = sum(1 for v in items if v.passed)
        full_pass = grp_passed == len(items)
        if full_pass:
            groups_passed_in_full += 1
        issue_groups_summary[group_name] = {
            "passed": grp_passed,
            "total": len(items),
            "passed_in_full": full_pass,
        }

    total_groups = len(grouped)
    issue_group_fraction = groups_passed_in_full / total_groups
    all_passed = passed_criteria == total_criteria
    bonus_awarded = cfg.all_pass_bonus if all_passed else 0.0

    reward = (
        cfg.criteria_weight * criteria_fraction
        + cfg.issue_group_weight * issue_group_fraction
        + bonus_awarded
    )
    return {
        "reward": round(float(reward), 6),
        "criteria_passed": passed_criteria,
        "criteria_total": total_criteria,
        "criteria_fraction": round(float(criteria_fraction), 6),
        "issue_groups_passed": groups_passed_in_full,
        "issue_groups_total": total_groups,
        "issue_group_fraction": round(float(issue_group_fraction), 6),
        "all_criteria_passed": all_passed,
        "all_pass_bonus_awarded": round(float(bonus_awarded), 6),
        "length_penalty": 0.0,
        "issue_groups": issue_groups_summary,
        "verdicts": [v.to_dict() for v in verdicts],
    }


def _coerce_criteria(
    raw: Sequence[RubricCriterion | dict[str, Any]],
) -> list[RubricCriterion]:
    out: list[RubricCriterion] = []
    for item in raw:
        if isinstance(item, RubricCriterion):
            out.append(item)
        else:
            out.append(
                RubricCriterion(
                    criterion_id=str(item["criterion_id"]),
                    description=str(item["description"]),
                    issue_group=str(item["issue_group"]),
                    required_keywords=tuple(item.get("required_keywords") or ()),
                    forbidden_keywords=tuple(item.get("forbidden_keywords") or ()),
                    regex_patterns=tuple(item.get("regex_patterns") or ()),
                    required_citations=tuple(item.get("required_citations") or ()),
                )
            )
    return out


class HierarchicalRubricReward(RewardFunction):
    """StateSet :class:`RewardFunction` implementing hierarchical binary rubric grading.

    Combines:
    1. Fraction of binary rubric criteria passed
    2. Fraction of legal/domain issue groups passed in full (all criteria in group pass)
    3. Bonus for passing every criterion in the rubric
    4. Explicitly zero length penalty
    """

    def __init__(
        self,
        *,
        criteria: Sequence[RubricCriterion] = (),
        judge: RubricJudge | None = None,
        config: RubricRewardConfig | None = None,
        weight: float = 1.0,
        name: str = "hierarchical_rubric_reward",
    ) -> None:
        super().__init__(weight=weight, reward_type=RewardType.SPARSE, name=name)
        self.default_criteria = tuple(criteria)
        self.judge: RubricJudge = judge or DeterministicRubricJudge()
        self.config = config or RubricRewardConfig()

    def score_deliverable_sync(
        self,
        deliverable: str,
        *,
        criteria: Sequence[RubricCriterion] | None = None,
    ) -> dict[str, Any]:
        """Synchronously score a deliverable when using :class:`DeterministicRubricJudge`."""
        active = list(criteria) if criteria is not None else list(self.default_criteria)
        if not isinstance(self.judge, DeterministicRubricJudge):
            raise TypeError(
                "score_deliverable_sync requires DeterministicRubricJudge; "
                "await compute_reward() for async LLM judges."
            )
        verdicts = self.judge.evaluate_criteria_sync(active, deliverable)
        return score_rubric_verdicts(verdicts, self.config)

    async def compute_reward(
        self,
        turns: list[ConversationTurn],
        context: dict[str, Any] | None = None,
    ) -> RewardResult:
        ctx = dict(context or {})
        raw_criteria = ctx.get("rubric_criteria") or ctx.get("criteria")
        active_criteria = (
            _coerce_criteria(raw_criteria)
            if raw_criteria
            else list(self.default_criteria)
        )
        deliverable = str(ctx.get("deliverable") or "").strip()
        if not deliverable:
            assistant_turns = [
                str(t.content)
                for t in turns
                if t.role == "assistant" and t.content is not None
            ]
            deliverable = assistant_turns[-1].strip() if assistant_turns else ""
        instruction = str(ctx.get("instruction") or ctx.get("prompt") or "")

        verdicts = await self.judge.evaluate_criteria(
            active_criteria,
            deliverable,
            instruction=instruction,
            context=ctx,
        )
        scored = score_rubric_verdicts(verdicts, self.config)
        breakdown = {
            "criteria_fraction": float(scored["criteria_fraction"]),
            "issue_group_fraction": float(scored["issue_group_fraction"]),
            "all_pass_bonus": float(scored["all_pass_bonus_awarded"]),
            "length_penalty": 0.0,
        }
        return RewardResult(
            score=float(scored["reward"]) * float(self.weight),
            breakdown=breakdown,
            metadata={
                "criteria_passed": scored["criteria_passed"],
                "criteria_total": scored["criteria_total"],
                "issue_groups_passed": scored["issue_groups_passed"],
                "issue_groups_total": scored["issue_groups_total"],
                "all_criteria_passed": scored["all_criteria_passed"],
                "issue_groups": scored["issue_groups"],
                "verdicts": scored["verdicts"],
                "config": self.config.to_dict(),
            },
            explanation=(
                f"Passed {scored['criteria_passed']}/{scored['criteria_total']} criteria "
                f"and {scored['issue_groups_passed']}/{scored['issue_groups_total']} "
                f"issue groups in full (all_pass={scored['all_criteria_passed']})."
            ),
        )


__all__ = [
    "CriterionVerdict",
    "DeterministicRubricJudge",
    "HierarchicalRubricReward",
    "JudgeCompletionFn",
    "LLMRubricJudge",
    "RubricCriterion",
    "RubricJudge",
    "RubricRewardConfig",
    "score_rubric_verdicts",
]
