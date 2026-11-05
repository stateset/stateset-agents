"""Protect seed/case clustering and complete prespecified cohort analysis."""

import copy

import pytest

from stateset_agents.evaluation.replication import (
    paired_seed_case_summary,
    validate_river_replication_seeds,
)

SEEDS = list(range(10, 16))
FAMILIES = {"A": "clean", "B": "clean", "C": "uncertain", "D": "uncertain"}


def analyze(runs, **overrides):
    options = {
        "expected_seeds": SEEDS,
        "case_families": FAMILIES,
        "group_size": 4,
        "analysis_seed": 910,
        "bootstrap_replicates": 1000,
    }
    options.update(overrides)
    return paired_seed_case_summary(runs, **options)


def cohort(before=2, after=3):
    return {seed: dict.fromkeys(FAMILIES, (before, after)) for seed in SEEDS}


def test_six_consistently_positive_seed_deltas_have_exact_two_sided_probability():
    result = analyze(cohort())
    assert result["exact_two_sided_seed_sign_flip_p"] == 2 / 64
    assert result["delta_rate"] == 0.25
    assert result["seed_and_family_case_bootstrap_95"] == [0.25, 0.25]
    assert result["shared_case_clusters"] == 4
    assert result["positive_seeds"] == 6


def test_tied_seeds_are_not_dropped_to_create_significance():
    runs = cohort()
    runs[SEEDS[-1]] = dict.fromkeys(FAMILIES, (3, 3))
    result = analyze(runs)
    assert result["exact_two_sided_seed_sign_flip_p"] == 4 / 64
    assert result["tied_seeds"] == 1


def test_all_ties_have_probability_one_and_zero_interval():
    result = analyze(cohort(4, 4))
    assert result["exact_two_sided_seed_sign_flip_p"] == 1
    assert result["seed_and_family_case_bootstrap_95"] == [0, 0]


def test_duplicate_rollout_counts_do_not_fabricate_more_inference_units():
    runs = cohort()
    runs[10]["A"] = (0, 4)
    runs[11]["D"] = (4, 1)
    result = analyze(runs)
    doubled = {
        seed: {key: (before * 2, after * 2) for key, (before, after) in rows.items()}
        for seed, rows in runs.items()
    }
    repeated = analyze(doubled, group_size=8)
    for key in (
        "delta_rate",
        "exact_two_sided_seed_sign_flip_p",
        "seed_and_family_case_bootstrap_95",
        "shared_case_clusters",
        "training_seeds",
    ):
        assert repeated[key] == result[key]


def test_bootstrap_resamples_cases_and_preserves_shared_case_dependence():
    runs = cohort(4, 4)
    for seed in SEEDS:
        runs[seed]["A"] = (0, 4)
    result = analyze(runs)
    assert result["delta_rate"] == 0.25
    assert result["seed_and_family_case_bootstrap_95"] == [0.0, 0.5]
    assert result == analyze(copy.deepcopy(runs))


@pytest.mark.parametrize(
    "change", ["missing_seed", "extra_seed", "missing_case", "extra_case"]
)
def test_incomplete_or_changed_cohort_is_rejected(change):
    runs = cohort()
    if change == "missing_seed":
        runs.pop(10)
    elif change == "extra_seed":
        runs[99] = dict(runs[10])
    elif change == "missing_case":
        runs[10].pop("A")
    else:
        runs[10]["unplanned"] = (1, 2)
    with pytest.raises(ValueError):
        analyze(runs)


@pytest.mark.parametrize(
    "pair", [(True, 2), (1, False), (-1, 2), (1, 5), (1.0, 2), (1,), "12"]
)
def test_invalid_success_counts_are_rejected(pair):
    runs = cohort()
    runs[10]["A"] = pair
    with pytest.raises(ValueError):
        analyze(runs)


@pytest.mark.parametrize(
    "options",
    [
        {"expected_seeds": [10] * 6},
        {"expected_seeds": [True, 11, 12, 13, 14, 15]},
        {"group_size": True},
        {"analysis_seed": -1},
        {"bootstrap_replicates": 0},
        {"case_families": {}},
        {"case_families": {"A": " "}},
    ],
)
def test_invalid_analysis_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        analyze(cohort(), **options)


@pytest.mark.parametrize("offset", [1, 3, 1000003, 104729, 2**31 - 1])
def test_sampler_stream_audit_detects_member_group_turn_and_wrap_collisions(offset):
    with pytest.raises(ValueError, match="sampler seed collision"):
        validate_river_replication_seeds(
            [0, offset], max_groups=80, group_size=4, max_sample_calls=1024
        )


def test_spaced_runs_have_disjoint_sampler_seeds_for_every_permitted_call():
    seeds = [20261101 + index * 350_000_000 for index in range(6)]
    result = validate_river_replication_seeds(
        seeds, max_groups=80, group_size=4, max_sample_calls=1024
    )
    assert result["passed"] and result["cross_run_sampler_seed_overlap"] == 0


@pytest.mark.parametrize("seeds", [[0, 1], [0, 4], [0, 104729], [0, 1000003], [0, 50]])
def test_stream_audit_matches_explicit_small_schedule_intersections(seeds):
    streams = [
        {
            (seed + group * 1000003 + member + call * 104729) % (2**31)
            for group in range(2)
            for member in range(2)
            for call in range(2)
        }
        for seed in seeds
    ]
    if streams[0] & streams[1]:
        with pytest.raises(ValueError):
            validate_river_replication_seeds(
                seeds, max_groups=2, group_size=2, max_sample_calls=2
            )
    else:
        assert validate_river_replication_seeds(
            seeds, max_groups=2, group_size=2, max_sample_calls=2
        )["passed"]
