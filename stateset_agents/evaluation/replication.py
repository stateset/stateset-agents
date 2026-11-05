"""Paired inference over training seeds and shared case clusters.

Aggregate repeated rollouts within each case before inference. Seed sign flips
condition on the fixed case set and assume exchangeable signs under the null.
The bootstrap resamples training seeds and, within each family, case clusters;
the same resampled cases are shared across all resampled seeds.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Mapping, Sequence
from typing import Any


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _quantile(values: list[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def validate_river_replication_seeds(
    seeds: Sequence[int],
    *,
    max_groups: int,
    group_size: int,
    max_sample_calls: int,
) -> dict[str, Any]:
    """Reject cross-run sampler-seed overlap under River's pinned seed schedule.

    River's inspected native engine uses
    ``(run_seed + group_index * 1000003 + member + sample_index * 104729)
    % 2**31``. Bounds cover every possible group/member/sample index, not just
    observed trajectories. Parent/candidate pairing within a run is intentional.
    This checks distinct seed identities, not the statistical independence of a
    provider's pseudorandom generator. Revalidate the schedule when the SDK changes.
    """
    planned = list(seeds)
    if not 2 <= len(planned) <= 16:
        raise ValueError("Seed audit requires 2–16 planned runs")
    modulus = 2**31
    for seed in planned:
        _integer(seed, "run seed", 0, modulus - 1)
    _integer(max_groups, "max_groups", 1, 1024)
    _integer(group_size, "group_size", 1, 128)
    _integer(max_sample_calls, "max_sample_calls", 1, 32768)
    turn_differences = {
        (-index * 104729) % modulus: index
        for index in range(1 - max_sample_calls, max_sample_calls)
    }
    for left, right in itertools.combinations(planned, 2):
        for group_delta in range(1 - max_groups, max_groups):
            for member_delta in range(1 - group_size, group_size):
                value = (left - right + group_delta * 1000003 + member_delta) % modulus
                if value in turn_differences:
                    raise ValueError(
                        f"Cross-run sampler seed collision for {left} and {right}: "
                        f"group delta {group_delta}, member delta {member_delta}, "
                        f"sample delta {turn_differences[value]}"
                    )
    return {
        "passed": True,
        "seeds": planned,
        "max_groups": max_groups,
        "group_size": group_size,
        "max_sample_calls": max_sample_calls,
        "group_stride": 1000003,
        "sample_stride": 104729,
        "modulus": modulus,
        "cross_run_sampler_seed_overlap": 0,
    }


def paired_seed_case_summary(
    runs: Mapping[int, Mapping[str, Sequence[int]]],
    *,
    expected_seeds: Sequence[int],
    case_families: Mapping[str, str],
    group_size: int,
    analysis_seed: int,
    bootstrap_replicates: int = 10_000,
) -> dict[str, Any]:
    """Summarize complete paired success counts without pooling replicates as IID.

    ``runs[seed][case_id]`` contains ``(parent_passed, candidate_passed)``
    counts out of ``group_size`` rollouts. Every planned seed must contain
    every shared case. Missing/skipped seeds and ties cannot be silently
    discarded. Attribution, trace replay, and policy safety require separate
    validation before calling this count-based analysis.

    The exact two-sided sign-flip p-value uses training-seed paired deltas.
    The percentile bootstrap preserves family sizes and resamples a shared
    case set across seeds. These conditional estimates do not establish
    production generalization or independent merchant evidence.
    """
    seeds = list(expected_seeds)
    if not 2 <= len(seeds) <= 16:
        raise ValueError("Inference requires 2–16 planned training seeds")
    for seed in seeds:
        _integer(seed, "training seed", 0, 2**63 - 1)
    if len(set(seeds)) != len(seeds):
        raise ValueError("Planned training seeds must be unique")
    if any(type(seed) is not int for seed in runs) or set(runs) != set(seeds):
        raise ValueError("Every planned training seed must be present exactly once")
    _integer(group_size, "group_size", 1, 1_000_000)
    _integer(analysis_seed, "analysis_seed", 0, 2**63 - 1)
    _integer(bootstrap_replicates, "bootstrap_replicates", 100, 100_000)
    if not case_families or any(
        not isinstance(key, str)
        or not key.strip()
        or not isinstance(family, str)
        or not family.strip()
        for key, family in case_families.items()
    ):
        raise ValueError("Nonempty case IDs and family labels are required")
    case_ids = sorted(case_families)
    families: dict[str, list[int]] = {}
    for index, key in enumerate(case_ids):
        families.setdefault(case_families[key], []).append(index)
    deltas: list[list[int]] = []
    totals: list[dict[str, Any]] = []
    for seed in seeds:
        if set(runs[seed]) != set(case_ids):
            raise ValueError("Every seed must contain the complete shared case set")
        parent = candidate = 0
        local: list[int] = []
        for key in case_ids:
            pair = runs[seed][key]
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(
                    "Each case needs paired parent/candidate success counts"
                )
            before = _integer(pair[0], "parent successes", 0, group_size)
            after = _integer(pair[1], "candidate successes", 0, group_size)
            parent += before
            candidate += after
            local.append(after - before)
        deltas.append(local)
        totals.append(
            {
                "seed": seed,
                "parent_passed": parent,
                "candidate_passed": candidate,
                "rollouts_per_arm": len(case_ids) * group_size,
                "delta_count": candidate - parent,
                "delta_rate": (candidate - parent) / (len(case_ids) * group_size),
            }
        )
    seed_deltas = [sum(local) for local in deltas]
    observed = abs(sum(seed_deltas))
    extreme = sum(
        abs(sum(sign * delta for sign, delta in zip(signs, seed_deltas, strict=True)))
        >= observed
        for signs in itertools.product((-1, 1), repeat=len(seeds))
    )
    rng = random.Random(analysis_seed)
    bootstrap: list[float] = []
    denominator = len(seeds) * len(case_ids) * group_size
    for _ in range(bootstrap_replicates):
        chosen = [rng.randrange(len(seeds)) for _ in seeds]
        by_case = [
            sum(deltas[seed][index] for seed in chosen)
            for index in range(len(case_ids))
        ]
        sampled = sum(
            by_case[rng.choice(indices)]
            for family in sorted(families)
            for indices in [families[family]]
            for _ in indices
        )
        bootstrap.append(sampled / denominator)
    bootstrap.sort()
    return {
        "schema_version": 1,
        "expected_seeds": seeds,
        "training_seeds": len(seeds),
        "shared_case_clusters": len(case_ids),
        "rollouts_per_case": group_size,
        "rollouts_per_arm": denominator,
        "parent_passed": sum(row["parent_passed"] for row in totals),
        "candidate_passed": sum(row["candidate_passed"] for row in totals),
        "delta_rate": sum(seed_deltas) / denominator,
        "seed_results": totals,
        "positive_seeds": sum(delta > 0 for delta in seed_deltas),
        "tied_seeds": sum(delta == 0 for delta in seed_deltas),
        "negative_seeds": sum(delta < 0 for delta in seed_deltas),
        "exact_two_sided_seed_sign_flip_p": extreme / (2 ** len(seeds)),
        "seed_and_family_case_bootstrap_95": [
            _quantile(bootstrap, 0.025),
            _quantile(bootstrap, 0.975),
        ],
        "bootstrap_replicates": bootstrap_replicates,
        "analysis_seed": analysis_seed,
        "scope": "Conditional on the shared cases; seed sign exchangeability and seed/case resampling assumptions apply. Repeated rollouts are aggregated within case. Not production or independent merchant evidence.",
    }
