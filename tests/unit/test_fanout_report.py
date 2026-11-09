"""The fan-out report: per-arm statistics and an honest verdict."""

from __future__ import annotations

import json
from typing import Any

import pytest

from stateset_agents.fanout.report import build_report, render_report


def _run(
    arm: str,
    seed: int,
    value: float | None,
    *,
    status: str = "succeeded",
    params: dict[str, Any] | None = None,
    cost_cents: float | None = 1.0,
) -> dict[str, Any]:
    return {
        "run_id": f"{arm}-seed{seed}",
        "arm_id": arm,
        "seed": seed,
        "params": params if params is not None else {"arm": arm},
        "status": status,
        "metrics": {} if value is None else {"reward": value},
        "cost_cents": cost_cents,
    }


def _report(runs: list[dict[str, Any]], mode: str = "max") -> dict[str, Any]:
    return build_report(
        name="demo", digest="abc", metric="reward", mode=mode, runs=runs
    )


def _arm(report: dict[str, Any], arm_id: str) -> dict[str, Any]:
    return next(a for a in report["arms"] if a["arm_id"] == arm_id)


class TestArmStatistics:
    def test_summarizes_each_arm_across_its_seeds(self) -> None:
        arm = _arm(
            _report([_run("a", 1, 1.0), _run("a", 2, 2.0), _run("a", 3, 3.0)]), "a"
        )

        assert arm["n"] == 3
        assert arm["mean"] == pytest.approx(2.0)
        assert arm["std"] == pytest.approx(1.0)
        assert arm["min"] == 1.0
        assert arm["max"] == 3.0
        assert arm["seeds"] == [1, 2, 3]
        # Student's t: 4.303 * 1.0 / sqrt(3), not a normal approximation.
        assert arm["ci95"] == pytest.approx([2.0 - 2.484, 2.0 + 2.484], abs=0.01)

    def test_one_seed_has_no_interval(self) -> None:
        arm = _arm(_report([_run("a", 1, 1.0)]), "a")

        assert arm["std"] is None
        assert arm["ci95"] is None

    def test_failed_seed_leaves_the_arm_incomplete(self) -> None:
        arm = _arm(
            _report([_run("a", 1, 1.0), _run("a", 2, None, status="failed")]), "a"
        )

        assert arm["n"] == 1
        assert arm["complete"] is False
        assert arm["failed_seeds"] == [2]

    def test_succeeded_run_without_the_metric_is_not_a_sample(self) -> None:
        arm = _arm(_report([_run("a", 1, 1.0), _run("a", 2, None)]), "a")

        assert arm["n"] == 1
        assert arm["complete"] is False
        assert arm["missing_metric_seeds"] == [2]

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), "high", True])
    def test_unusable_metric_values_are_not_samples(self, value: Any) -> None:
        run = _run("a", 1, 1.0)
        run["metrics"] = {"reward": value}

        arm = _arm(_report([run]), "a")

        assert arm["n"] == 0
        assert arm["mean"] is None


class TestVerdict:
    def test_ranks_by_mean_in_the_requested_direction(self) -> None:
        runs = [_run("low", 1, 1.0), _run("high", 1, 5.0)]

        assert _report(runs, "max")["ranking"] == ["high", "low"]
        assert _report(runs, "min")["ranking"] == ["low", "high"]

    def test_separated_intervals_resolve(self) -> None:
        report = _report(
            [
                _run("good", 1, 10.0),
                _run("good", 2, 10.5),
                _run("good", 3, 11.0),
                _run("bad", 1, 1.0),
                _run("bad", 2, 1.5),
                _run("bad", 3, 2.0),
            ]
        )

        assert report["verdict"]["status"] == "resolved"
        assert report["verdict"]["best_arm"] == "good"

    def test_separated_intervals_resolve_when_minimizing(self) -> None:
        report = _report(
            [
                _run("good", 1, 1.0),
                _run("good", 2, 1.5),
                _run("good", 3, 2.0),
                _run("bad", 1, 10.0),
                _run("bad", 2, 10.5),
                _run("bad", 3, 11.0),
            ],
            "min",
        )

        assert report["verdict"]["status"] == "resolved"
        assert report["verdict"]["best_arm"] == "good"

    def test_overlapping_intervals_name_no_winner(self) -> None:
        report = _report(
            [
                _run("a", 1, 1.0),
                _run("a", 2, 5.0),
                _run("a", 3, 9.0),
                _run("b", 1, 0.5),
                _run("b", 2, 4.5),
                _run("b", 3, 8.5),
            ]
        )

        assert report["verdict"]["status"] == "unresolved"
        assert report["verdict"]["best_arm"] is None
        assert report["ranking"][0] == "a"

    def test_top_arm_with_failed_seeds_cannot_win_by_survivorship(self) -> None:
        report = _report(
            [
                _run("lucky", 1, 100.0),
                _run("lucky", 2, 100.0),
                _run("lucky", 3, None, status="failed"),
                _run("steady", 1, 1.0),
                _run("steady", 2, 1.0),
                _run("steady", 3, 1.0),
            ]
        )

        assert report["verdict"]["status"] == "unresolved"
        assert report["verdict"]["best_arm"] is None
        assert "lucky" in report["verdict"]["reason"]

    def test_one_seed_per_arm_cannot_separate_arms(self) -> None:
        report = _report([_run("a", 1, 2.0), _run("b", 1, 1.0)])

        assert report["verdict"]["status"] == "unresolved"
        assert "seed" in report["verdict"]["reason"]

    def test_single_arm_has_nothing_to_compare(self) -> None:
        report = _report([_run("a", 1, 1.0), _run("a", 2, 2.0)])

        assert report["verdict"]["status"] == "single_arm"
        assert report["verdict"]["best_arm"] is None

    def test_no_usable_samples(self) -> None:
        report = _report([_run("a", 1, None, status="failed")])

        assert report["verdict"]["status"] == "no_data"
        assert report["ranking"] == []


class TestVerdictConsidersEveryArm:
    GOOD = [_run("good", seed, 10.0 + seed / 10) for seed in (1, 2, 3)]
    BAD = [_run("bad", seed, 1.0 + seed / 10) for seed in (1, 2, 3)]

    def test_a_noisy_third_arm_could_still_be_the_best(self) -> None:
        # Ranked third by mean, yet its interval reaches past the best arm's.
        noisy = [_run("noisy", 1, -30.0), _run("noisy", 2, 1.1), _run("noisy", 3, 30.0)]

        report = _report([*self.GOOD, *self.BAD, *noisy])

        assert report["verdict"]["status"] == "unresolved"
        assert report["verdict"]["best_arm"] is None
        assert "noisy" in report["verdict"]["reason"]

    def test_a_third_arm_that_never_ran_leaves_the_question_open(self) -> None:
        pending = [_run("late", seed, None, status="pending") for seed in (1, 2, 3)]

        report = _report([*self.GOOD, *self.BAD, *pending])

        assert report["verdict"]["status"] == "unresolved"
        assert "late" in report["verdict"]["reason"]

    def test_a_third_arm_with_a_lost_seed_leaves_the_question_open(self) -> None:
        partial = [
            _run("partial", 1, -5.0),
            _run("partial", 2, None, status="failed"),
            _run("partial", 3, -4.0),
        ]

        report = _report([*self.GOOD, *self.BAD, *partial])

        assert report["verdict"]["status"] == "unresolved"
        assert "partial" in report["verdict"]["reason"]

    def test_three_cleanly_separated_arms_resolve(self) -> None:
        worst = [_run("worst", seed, -20.0 + seed / 10) for seed in (1, 2, 3)]

        report = _report([*self.GOOD, *self.BAD, *worst])

        assert report["verdict"]["status"] == "resolved"
        assert report["verdict"]["best_arm"] == "good"


class TestAccounting:
    def test_counts_runs_by_status(self) -> None:
        report = _report(
            [
                _run("a", 1, 1.0),
                _run("a", 2, None, status="failed"),
                _run("a", 3, None, status="cancelled"),
                _run("a", 4, None, status="pending"),
            ]
        )

        assert report["runs"] == {
            "total": 4,
            "succeeded": 1,
            "failed": 1,
            "cancelled": 1,
            "incomplete": 1,
        }

    def test_unknown_cost_is_reported_rather_than_read_as_free(self) -> None:
        report = _report(
            [
                _run("a", 1, 1.0, cost_cents=2.5),
                _run("a", 2, 1.0, cost_cents=None),
            ]
        )

        assert report["cost"] == {
            "known_cents": 2.5,
            "runs_with_unknown_cost": 1,
        }

    def test_report_is_strict_json(self) -> None:
        run = _run("a", 1, 1.0)
        run["metrics"]["other"] = float("nan")

        json.dumps(_report([run]), allow_nan=False)


class TestRender:
    def test_shows_the_verdict_and_every_arm(self) -> None:
        text = render_report(
            _report(
                [
                    _run("a", 1, 2.0, params={"lr": 0.1}),
                    _run("b", 1, 1.0, params={"lr": 0.2}),
                ]
            )
        )

        assert "unresolved" in text
        assert "lr=0.1" in text
        assert "lr=0.2" in text

    def test_says_when_cost_is_partly_unknown(self) -> None:
        text = render_report(_report([_run("a", 1, 1.0, cost_cents=None)]))

        assert "unknown" in text
