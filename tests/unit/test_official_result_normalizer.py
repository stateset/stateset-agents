"""Tests for normalization of official agent benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.adapters.official_result_normalizer import (
    OfficialResultError,
    main,
    normalize_bfcl_v4,
    normalize_swe_bench_verified,
    normalize_tau3,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_costs(path: Path, values: dict[str, float]) -> None:
    path.write_text(
        "".join(
            json.dumps({"task_id": task_id, "cost_usd": cost}) + "\n"
            for task_id, cost in values.items()
        ),
        encoding="utf-8",
    )


def test_tau3_normalizes_monolithic_official_results(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    _write_json(
        results,
        {
            "simulations": [
                {
                    "task_id": "airline-1",
                    "reward_info": {"reward": 1.0},
                    "agent_cost": 0.12,
                },
                {
                    "task_id": "airline-2",
                    "reward_info": {"reward": 0.5},
                    "agent_cost": 0.08,
                },
            ]
        },
    )
    assert normalize_tau3(results) == [
        {"task_id": "airline-1", "success": True, "cost_usd": 0.12},
        {"task_id": "airline-2", "success": False, "cost_usd": 0.08},
    ]


def test_tau3_supports_directory_index_and_rejects_trials(tmp_path: Path) -> None:
    result_dir = tmp_path / "run"
    _write_json(
        result_dir / "results.json",
        {
            "simulation_index": [
                {"task_id": 7, "reward": 1, "agent_cost": 0.1},
                {"task_id": 7, "reward": 0, "agent_cost": 0.1},
            ]
        },
    )
    with pytest.raises(OfficialResultError, match="exactly one trial"):
        normalize_tau3(result_dir)


def test_bfcl_v4_joins_generated_and_incorrect_case_ids(tmp_path: Path) -> None:
    results = tmp_path / "result"
    scores = tmp_path / "score"
    costs = tmp_path / "costs.jsonl"
    _write_json(
        results / "agentic" / "web_search.json",
        [{"id": "web-1"}, {"id": "web-2"}, {"id": "web-3"}],
    )
    _write_json(
        scores / "agentic" / "web_search.json",
        [
            {"accuracy": 2 / 3, "correct_count": 2, "total_count": 3},
            {"id": "web-2", "error": "wrong answer"},
        ],
    )
    _write_costs(costs, {"web-1": 0.1, "web-2": 0.2, "web-3": 0.3})

    assert normalize_bfcl_v4(results, scores, costs_path=costs) == [
        {"task_id": "web-1", "success": True, "cost_usd": 0.1},
        {"task_id": "web-2", "success": False, "cost_usd": 0.2},
        {"task_id": "web-3", "success": True, "cost_usd": 0.3},
    ]


def test_bfcl_v4_rejects_partial_score_totals(tmp_path: Path) -> None:
    results = tmp_path / "result.json"
    scores = tmp_path / "score.json"
    costs = tmp_path / "costs.jsonl"
    _write_json(results, [{"id": "one"}, {"id": "two"}])
    _write_json(
        scores,
        [{"accuracy": 1.0, "correct_count": 1, "total_count": 1}],
    )
    _write_costs(costs, {"one": 0.1, "two": 0.1})
    with pytest.raises(OfficialResultError, match="totals do not match"):
        normalize_bfcl_v4(results, scores, costs_path=costs)


def test_swe_bench_normalizes_full_schema_v2_report(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    costs = tmp_path / "costs.jsonl"
    _write_json(
        results,
        {
            "schema_version": 2,
            "total_instances": 3,
            "submitted_ids": ["django-2", "django-1", "flask-1"],
            "resolved_ids": ["django-1"],
        },
    )
    _write_costs(costs, {"django-1": 1.0, "django-2": 2.0, "flask-1": 3.0})
    assert normalize_swe_bench_verified(results, costs_path=costs) == [
        {"task_id": "django-1", "success": True, "cost_usd": 1.0},
        {"task_id": "django-2", "success": False, "cost_usd": 2.0},
        {"task_id": "flask-1", "success": False, "cost_usd": 3.0},
    ]


def test_swe_bench_rejects_partial_runs_and_cost_drift(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    costs = tmp_path / "costs.jsonl"
    _write_json(
        results,
        {
            "schema_version": 2,
            "total_instances": 2,
            "submitted_ids": ["one"],
            "resolved_ids": ["one"],
        },
    )
    _write_costs(costs, {"one": 0.1})
    with pytest.raises(OfficialResultError, match="partial SWE-bench"):
        normalize_swe_bench_verified(results, costs_path=costs)

    _write_json(
        results,
        {
            "schema_version": 2,
            "total_instances": 1,
            "submitted_ids": ["one"],
            "resolved_ids": ["one"],
        },
    )
    _write_costs(costs, {"two": 0.1})
    with pytest.raises(OfficialResultError, match="task/cost ID mismatch"):
        normalize_swe_bench_verified(results, costs_path=costs)


def test_cli_writes_paired_harness_jsonl(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    output = tmp_path / "normalized" / "tasks.jsonl"
    _write_json(
        results,
        {"simulation_index": [{"task_id": "one", "reward": 1, "agent_cost": 0.2}]},
    )
    assert (
        main(
            [
                "--suite",
                "tau3-bench",
                "--results",
                str(results),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "cost_usd": 0.2,
        "success": True,
        "task_id": "one",
    }
