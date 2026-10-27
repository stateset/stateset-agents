"""Compare the support-conversation rewards with recorded judge scores.

``audit_support_rewards`` re-scores paired base/trained transcripts with the
v2 and v3 ``SupportConversationReward`` and reports how often each reward
agrees with the recorded judge. It is retrospective reward development, not
held-out validation.

Moved here from ``stateset_agents.evaluation.reward_audit`` in v0.70.0: it
depends on the support rewards, and ``evaluation`` sits below ``rewards``.
The generic ``audit_reward`` stays in ``evaluation.reward_audit``.

Usage::

    python -m stateset_agents.rewards.support_reward_audit pairs.json \
        --expected-pairs 200 --output report.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.rewards.support_conversation_reward import (
    SupportConversationReward,
    SupportConversationRewardV3,
)

__all__ = ["audit_support_rewards", "main"]


def _score(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError("Recorded judge scores must be finite in [0, 1]")
    return float(value)


def _ranks(values: Sequence[float]) -> list[float]:
    """Average tied ranks; no scipy dependency is needed for this diagnostic."""
    positions: dict[float, list[int]] = {}
    for index, value in enumerate(sorted(values)):
        positions.setdefault(value, []).append(index + 1)
    return [statistics.mean(positions[value]) for value in values]


def _spearman(first: Sequence[float], second: Sequence[float]) -> float | None:
    a, b = _ranks(first), _ranks(second)
    if len(set(a)) < 2 or len(set(b)) < 2:
        return None
    return statistics.correlation(a, b)


async def audit_support_rewards(
    conversations: Sequence[Mapping[str, Any]], *, expected_pairs: int
) -> dict[str, Any]:
    """Compare v2/v3 with recorded judge scores, retaining per-item diagnostics.

    This is retrospective reward development, not held-out validation: examples
    inspected while designing v3 must not become evidence of new quality gains.
    Scores are checked for shape and range; authenticating the judge execution
    and provenance remains the artifact collector's responsibility.
    """
    if (
        type(expected_pairs) is not int
        or expected_pairs < 1
        or len(conversations) != expected_pairs
    ):
        raise ValueError(
            "Input must contain exactly the declared number of paired cases"
        )
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    versions = {"v2": SupportConversationReward(), "v3": SupportConversationRewardV3()}
    for item in conversations:
        identity = item.get("id")
        if not isinstance(identity, str) or not identity.strip() or identity in seen:
            raise ValueError("Paired conversation IDs must be nonempty and unique")
        seen.add(identity)
        context = item.get("context")
        if not isinstance(context, Mapping):
            raise ValueError("Each paired conversation needs scenario context")
        judge = item.get("judge")
        if not isinstance(judge, Mapping):
            raise ValueError(
                "Each paired conversation needs base and trained judge scores"
            )
        row: dict[str, Any] = {
            "id": identity,
            "judge": {key: _score(judge[key]) for key in ("base", "trained")},
        }
        for version, reward in versions.items():
            row[version] = {}
            for policy in ("base", "trained"):
                raw = item[policy]
                if not isinstance(raw, (list, tuple)) or not raw:
                    raise ValueError("Both policies need nonempty transcripts")
                turns = []
                for pair in raw:
                    if (
                        not isinstance(pair, (list, tuple))
                        or len(pair) != 2
                        or pair[0] not in {"user", "assistant"}
                        or not isinstance(pair[1], str)
                    ):
                        raise ValueError(
                            "Transcripts must contain user/assistant text pairs"
                        )
                    turns.append(ConversationTurn(role=pair[0], content=pair[1]))
                if not any(
                    t.role == "assistant" and (t.content or "").strip() for t in turns
                ):
                    raise ValueError("Both policies need assistant completions")
                result = await reward.compute_reward(turns, dict(context))
                row[version][policy] = {
                    "score": result.score,
                    "breakdown": result.breakdown,
                }
        rows.append(row)
    judge_values = [row["judge"][p] for row in rows for p in ("base", "trained")]
    summary: dict[str, Any] = {}
    for version in versions:
        agree = disagree = judge_ties = reward_ties = 0
        for row in rows:
            jd = row["judge"]["trained"] - row["judge"]["base"]
            rd = row[version]["trained"]["score"] - row[version]["base"]["score"]
            if math.isclose(jd, 0, abs_tol=1e-12):
                judge_ties += 1
            elif math.isclose(rd, 0, abs_tol=1e-12):
                reward_ties += 1
            elif (jd > 0) == (rd > 0):
                agree += 1
            else:
                disagree += 1
        base = statistics.mean(row[version]["base"]["score"] for row in rows)
        trained = statistics.mean(row[version]["trained"]["score"] for row in rows)
        jd_mean = statistics.mean(
            row["judge"]["trained"] - row["judge"]["base"] for row in rows
        )
        summary[version] = {
            "base_mean": base,
            "trained_mean": trained,
            "direction_agreements": agree,
            "direction_disagreements": disagree,
            "judge_ties": judge_ties,
            "reward_ties_on_judge_decisive_pairs": reward_ties,
            "agreement_fraction_all_pairs": agree / len(rows),
            "spearman_all_transcripts": _spearman(
                [row[version][p]["score"] for row in rows for p in ("base", "trained")],
                judge_values,
            ),
            "reward_up_judge_down": trained > base and jd_mean < 0,
        }
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "quality_gain_established": False,
        "pairs": len(rows),
        "judge": {
            "base_mean": statistics.mean(row["judge"]["base"] for row in rows),
            "trained_mean": statistics.mean(row["judge"]["trained"] for row in rows),
        },
        "rewards": summary,
        "items": rows,
    }


def main() -> None:
    """Audit an explicitly sized paired artifact; never fill missing cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--expected-pairs", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    document = json.loads(raw)
    result = asyncio.run(
        audit_support_rewards(
            document["conversations"], expected_pairs=args.expected_pairs
        )
    )
    result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
