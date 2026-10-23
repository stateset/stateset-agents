"""Calibrate an LLM judge against labelled examples and measure score drift.

A model judge is only as good as its agreement with people. This harness
scores a labelled JSONL set with a judge and reports how closely it agrees
and how stable its scores are:

* **pointwise rows** ``{"query", "response", "label"}`` (``label`` in
  ``[0, 1]``; ``"score"`` is accepted for ``"label"``; optional ``"reference"``
  and ``"id"``): mean absolute error, the share within ``tolerance`` of the
  label, the share on the same side of ``threshold`` (pass/fail agreement)
  and Spearman rank correlation;
* **pairwise rows** ``{"query", "response_a", "response_b", "preferred"}``
  (``"A"``, ``"B"`` or ``"tie"``): the share of verdicts matching the
  preference and, for judges that judge both orders
  (``LLMJudge.compare_pairwise_verdicts``), the share of pairs whose two
  orders agree;
* **drift**: every pointwise row is scored ``repeats`` times with the judge's
  cache bypassed, and the report gives the mean and largest spread; across
  judge versions or models, :func:`score_drift` compares two reports.

Judge outages (``RewardUnavailableError``) are counted as failures, never
scored. The harness needs only an object with ``async evaluate(query,
response, reference=None)`` (and ``compare_pairwise`` for pairwise rows), so it
is tested with a mock judge; see ``tests/unit/test_judge_calibration.py``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from stateset_agents.core.reward_base import RewardUnavailableError

__all__ = [
    "CalibrationExample",
    "CalibrationReport",
    "PairwiseExample",
    "calibrate_judge",
    "load_calibration_set",
    "score_drift",
]


@dataclass(frozen=True)
class CalibrationExample:
    """A response with the score a person gave it (in ``[0, 1]``)."""

    query: str
    response: str
    label: float
    reference: str | None = None
    id: str = ""


@dataclass(frozen=True)
class PairwiseExample:
    """Two responses and which one a person preferred."""

    query: str
    response_a: str
    response_b: str
    preferred: str
    id: str = ""


@dataclass
class CalibrationReport:
    """Agreement and stability of a judge on a labelled set."""

    pointwise: int = 0
    pairwise: int = 0
    failures: int = 0
    mean_absolute_error: float | None = None
    agreement: float | None = None
    decision_agreement: float | None = None
    spearman: float | None = None
    drift_mean: float | None = None
    drift_max: float | None = None
    pairwise_agreement: float | None = None
    pairwise_consistency: float | None = None
    tolerance: float = 0.2
    threshold: float = 0.5
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def passes(self, *, min_agreement: float = 0.8, max_drift: float = 0.1) -> bool:
        """Whether the judge agrees with the labels at least ``min_agreement``
        of the time (pointwise and pairwise), drifts by at most ``max_drift``
        and never failed."""
        checks = [self.failures == 0]
        if self.agreement is not None:
            checks.append(self.agreement >= min_agreement)
        if self.pairwise_agreement is not None:
            checks.append(self.pairwise_agreement >= min_agreement)
        if self.drift_max is not None:
            checks.append(self.drift_max <= max_drift)
        return all(checks)


def _row_id(row: Mapping[str, Any], index: int) -> str:
    return str(row.get("id", index))


def _parse_row(
    row: Any, index: int
) -> CalibrationExample | PairwiseExample:
    if not isinstance(row, Mapping):
        raise ValueError(f"row {index}: expected a JSON object")
    query = row.get("query")
    if not isinstance(query, str):
        raise ValueError(f"row {index}: 'query' must be a string")
    if "response_a" in row or "response_b" in row:
        preferred = str(row.get("preferred", "")).upper()
        if preferred not in {"A", "B", "TIE"}:
            raise ValueError(f"row {index}: 'preferred' must be A, B or tie")
        a, b = row.get("response_a"), row.get("response_b")
        if not isinstance(a, str) or not isinstance(b, str):
            raise ValueError(f"row {index}: responses must be strings")
        return PairwiseExample(query, a, b, preferred, _row_id(row, index))
    response = row.get("response")
    label = row.get("label", row.get("score"))
    if not isinstance(response, str):
        raise ValueError(f"row {index}: 'response' must be a string")
    if isinstance(label, bool) or not isinstance(label, (int, float)):
        raise ValueError(f"row {index}: 'label' must be a number")
    if not (math.isfinite(label) and 0.0 <= label <= 1.0):
        raise ValueError(f"row {index}: 'label' must be in [0, 1]")
    reference = row.get("reference")
    if reference is not None and not isinstance(reference, str):
        raise ValueError(f"row {index}: 'reference' must be a string")
    return CalibrationExample(
        query, response, float(label), reference, _row_id(row, index)
    )


def load_calibration_set(
    source: str | Path | Iterable[Mapping[str, Any]],
) -> tuple[list[CalibrationExample], list[PairwiseExample]]:
    """Pointwise and pairwise examples from a JSONL file or rows.

    Raises ``ValueError`` naming the row of any malformed example.
    """
    if isinstance(source, (str, Path)):
        rows: list[Any] = []
        lines = Path(source).read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"line {number}: not JSON ({error})") from error
    else:
        rows = list(source)
    pointwise: list[CalibrationExample] = []
    pairwise: list[PairwiseExample] = []
    for index, row in enumerate(rows):
        parsed = _parse_row(row, index)
        if isinstance(parsed, PairwiseExample):
            pairwise.append(parsed)
        else:
            pointwise.append(parsed)
    return pointwise, pairwise


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2.0
        for k in range(position, end + 1):
            ranks[order[k]] = average
        position = end + 1
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = math.fsum(rx) / len(rx), math.fsum(ry) / len(ry)
    cov = math.fsum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx = math.fsum((a - mx) ** 2 for a in rx)
    vy = math.fsum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return cov / math.sqrt(vx * vy)


async def _evaluate(judge: Any, example: CalibrationExample) -> float:
    if example.reference is None:
        return float(await judge.evaluate(example.query, example.response))
    return float(
        await judge.evaluate(
            example.query, example.response, reference=example.reference
        )
    )


async def _score_pointwise(
    judge: Any, examples: Sequence[CalibrationExample], repeats: int
) -> tuple[dict[str, list[float]], int]:
    # Bypass a result cache so repeats measure the judge, not the cache.
    cache = getattr(judge, "cache", None)
    if cache is not None:
        judge.cache = None
    runs: dict[str, list[float]] = {}
    failures = 0
    try:
        for example in examples:
            for _ in range(repeats):
                try:
                    score = await _evaluate(judge, example)
                except RewardUnavailableError:
                    failures += 1
                    continue
                runs.setdefault(example.id, []).append(score)
    finally:
        if cache is not None:
            judge.cache = cache
    return runs, failures


async def _score_pairwise(
    judge: Any, pairs: Sequence[PairwiseExample]
) -> tuple[list[bool], list[bool], int]:
    matches: list[bool] = []
    consistent: list[bool] = []
    failures = 0
    detailed = hasattr(judge, "compare_pairwise_verdicts")
    for pair in pairs:
        try:
            if detailed:
                verdict = await judge.compare_pairwise_verdicts(
                    pair.query, pair.response_a, pair.response_b
                )
                winner, agrees = verdict.winner, verdict.consistent
            else:
                winner, _, _ = await judge.compare_pairwise(
                    pair.query, pair.response_a, pair.response_b
                )
                winner, agrees = str(winner).upper(), True
        except RewardUnavailableError:
            failures += 1
            continue
        matches.append(winner == pair.preferred)
        consistent.append(bool(agrees))
    return matches, consistent, failures


async def calibrate_judge(
    judge: Any,
    examples: str | Path | Iterable[Mapping[str, Any]] | Sequence[CalibrationExample],
    pairs: Sequence[PairwiseExample] = (),
    *,
    tolerance: float = 0.2,
    threshold: float = 0.5,
    repeats: int = 2,
) -> CalibrationReport:
    """Score a labelled set with ``judge`` and report agreement and drift.

    ``examples`` is a JSONL path, raw rows, or parsed
    :class:`CalibrationExample` objects (pairwise rows in a file or row list
    are picked up too; parsed pairs go in ``pairs``).
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    items = list(examples) if not isinstance(examples, (str, Path)) else None
    if items is not None and all(isinstance(i, CalibrationExample) for i in items):
        pointwise = list(items)
        pairwise = list(pairs)
    else:
        pointwise, loaded_pairs = load_calibration_set(
            examples if items is None else items  # type: ignore[arg-type]
        )
        pairwise = [*loaded_pairs, *pairs]

    runs, failures = await _score_pointwise(judge, pointwise, repeats)
    matches, consistent, pair_failures = await _score_pairwise(judge, pairwise)
    report = CalibrationReport(
        pointwise=len(pointwise),
        pairwise=len(pairwise),
        failures=failures + pair_failures,
        tolerance=tolerance,
        threshold=threshold,
    )
    scored = [e for e in pointwise if runs.get(e.id)]
    if scored:
        means = [math.fsum(runs[e.id]) / len(runs[e.id]) for e in scored]
        labels = [e.label for e in scored]
        errors = [abs(m - y) for m, y in zip(means, labels, strict=True)]
        report.mean_absolute_error = math.fsum(errors) / len(errors)
        report.agreement = sum(e <= tolerance for e in errors) / len(errors)
        report.decision_agreement = sum(
            (m >= threshold) == (y >= threshold)
            for m, y in zip(means, labels, strict=True)
        ) / len(scored)
        report.spearman = _spearman(means, labels)
        spreads = [max(runs[e.id]) - min(runs[e.id]) for e in scored]
        report.drift_mean = math.fsum(spreads) / len(spreads)
        report.drift_max = max(spreads)
        report.scores = {
            e.id: m for e, m in zip(scored, means, strict=True)
        }
    if matches:
        report.pairwise_agreement = sum(matches) / len(matches)
        report.pairwise_consistency = sum(consistent) / len(consistent)
    return report


def score_drift(
    baseline: CalibrationReport | Mapping[str, float],
    current: CalibrationReport | Mapping[str, float],
) -> dict[str, float]:
    """How far ``current`` scores moved from ``baseline`` on shared examples
    (a new judge model, prompt or provider version): mean and largest
    absolute change, and how many examples were compared."""
    old = baseline.scores if isinstance(baseline, CalibrationReport) else baseline
    new = current.scores if isinstance(current, CalibrationReport) else current
    shared = sorted(set(old) & set(new))
    if not shared:
        return {"compared": 0.0, "mean": 0.0, "max": 0.0}
    changes = [abs(float(new[key]) - float(old[key])) for key in shared]
    return {
        "compared": float(len(shared)),
        "mean": math.fsum(changes) / len(changes),
        "max": max(changes),
    }
