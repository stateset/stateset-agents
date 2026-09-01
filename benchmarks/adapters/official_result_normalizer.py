#!/usr/bin/env python3
"""Normalize official agent benchmark artifacts into paired task JSONL.

The normalizer deliberately derives task outcomes from official suite artifacts
and requires exact per-task cost records where the suite does not report cost.
It never accepts aggregate-only scores or estimated costs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


class OfficialResultError(ValueError):
    """Raised when an official result artifact is incomplete or inconsistent."""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OfficialResultError(f"invalid JSON artifact: {path}") from exc


def _task_id(value: Any, *, source: Path) -> str:
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise OfficialResultError(f"invalid task ID in {source}")
    result = str(value).strip()
    if not result:
        raise OfficialResultError(f"empty task ID in {source}")
    return result


def _finite_cost(value: Any, *, task_id: str, source: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OfficialResultError(f"missing numeric cost for {task_id} in {source}")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise OfficialResultError(
            f"cost for {task_id} in {source} must be finite and non-negative"
        )
    return result


def load_cost_records(path: Path) -> dict[str, float]:
    """Load exact task-keyed metering records from JSONL."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise OfficialResultError(f"cost records do not exist: {path}") from exc
    if not lines:
        raise OfficialResultError(f"cost records are empty: {path}")
    costs: dict[str, float] = {}
    for line_number, line in enumerate(lines, start=1):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise OfficialResultError(
                f"invalid cost JSONL at {path}:{line_number}"
            ) from exc
        if not isinstance(raw, Mapping):
            raise OfficialResultError(f"cost record must be an object: {path}")
        task_id = _task_id(raw.get("task_id"), source=path)
        if task_id in costs:
            raise OfficialResultError(f"duplicate cost record for {task_id}")
        costs[task_id] = _finite_cost(raw.get("cost_usd"), task_id=task_id, source=path)
    return costs


def _attach_costs(
    outcomes: Mapping[str, bool], costs: Mapping[str, float], *, source: Path
) -> list[dict[str, Any]]:
    outcome_ids = set(outcomes)
    cost_ids = set(costs)
    if outcome_ids != cost_ids:
        raise OfficialResultError(
            f"task/cost ID mismatch in {source}: "
            f"missing_costs={sorted(outcome_ids - cost_ids)}, "
            f"extra_costs={sorted(cost_ids - outcome_ids)}"
        )
    return [
        {"task_id": task_id, "success": success, "cost_usd": costs[task_id]}
        for task_id, success in outcomes.items()
    ]


def normalize_tau3(
    results_path: Path, *, costs_path: Path | None = None
) -> list[dict[str, Any]]:
    """Normalize τ³-bench monolithic or directory-index results."""
    source = results_path / "results.json" if results_path.is_dir() else results_path
    raw = _load_json(source)
    if not isinstance(raw, Mapping):
        raise OfficialResultError(f"τ³ results must be an object: {source}")
    simulations = raw.get("simulations")
    indexed = False
    if not isinstance(simulations, list) or not simulations:
        simulations = raw.get("simulation_index")
        indexed = True
    if not isinstance(simulations, list) or not simulations:
        raise OfficialResultError(f"τ³ results contain no simulations: {source}")

    outcomes: dict[str, bool] = {}
    embedded_costs: dict[str, float] = {}
    for simulation in simulations:
        if not isinstance(simulation, Mapping):
            raise OfficialResultError(f"invalid τ³ simulation in {source}")
        task_id = _task_id(simulation.get("task_id"), source=source)
        if task_id in outcomes:
            raise OfficialResultError(
                f"τ³ must contain exactly one trial per task; duplicate {task_id}"
            )
        reward = simulation.get("reward") if indexed else None
        if not indexed:
            reward_info = simulation.get("reward_info")
            if isinstance(reward_info, Mapping):
                reward = reward_info.get("reward")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise OfficialResultError(f"missing reward for {task_id} in {source}")
        numeric_reward = float(reward)
        if not math.isfinite(numeric_reward) or not 0 <= numeric_reward <= 1:
            raise OfficialResultError(f"invalid reward for {task_id} in {source}")
        outcomes[task_id] = math.isclose(numeric_reward, 1.0, abs_tol=1e-12)
        if costs_path is None:
            embedded_costs[task_id] = _finite_cost(
                simulation.get("agent_cost"), task_id=task_id, source=source
            )
    costs = load_cost_records(costs_path) if costs_path is not None else embedded_costs
    return _attach_costs(outcomes, costs, source=source)


def _json_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise OfficialResultError(f"artifact path does not exist: {path}")
    files = sorted(path.rglob("*.json"))
    if not files:
        raise OfficialResultError(f"no JSON artifacts found under {path}")
    return files


def _load_json_records(path: Path) -> list[Mapping[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise OfficialResultError(f"invalid JSON artifact: {path}") from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        try:
            raw = [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError as exc:
            raise OfficialResultError(f"invalid JSON/JSONL artifact: {path}") from exc
    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, Mapping):
        records = [raw]
    else:
        raise OfficialResultError(f"expected JSON object or array: {path}")
    if any(not isinstance(record, Mapping) for record in records):
        raise OfficialResultError(f"invalid record in {path}")
    return list(records)


def normalize_bfcl_v4(
    result_path: Path, score_path: Path, *, costs_path: Path
) -> list[dict[str, Any]]:
    """Normalize BFCL V4 result IDs and official incorrect-case score files."""
    generated_ids: dict[str, None] = {}
    for path in _json_files(result_path):
        for record in _load_json_records(path):
            task_id = _task_id(record.get("id"), source=path)
            if task_id in generated_ids:
                raise OfficialResultError(f"duplicate BFCL result ID {task_id}")
            generated_ids[task_id] = None

    incorrect_ids: set[str] = set()
    total_count = 0
    correct_count = 0
    for path in _json_files(score_path):
        records = _load_json_records(path)
        if not records:
            raise OfficialResultError(f"empty BFCL score file: {path}")
        header = records[0]
        total = header.get("total_count")
        correct = header.get("correct_count")
        if (
            isinstance(total, bool)
            or not isinstance(total, int)
            or total < 1
            or isinstance(correct, bool)
            or not isinstance(correct, int)
            or not 0 <= correct <= total
        ):
            raise OfficialResultError(f"invalid BFCL score header: {path}")
        total_count += total
        correct_count += correct
        file_incorrect: set[str] = set()
        for record in records[1:]:
            task_id = _task_id(record.get("id"), source=path)
            if task_id in incorrect_ids:
                raise OfficialResultError(f"duplicate BFCL score ID {task_id}")
            incorrect_ids.add(task_id)
            file_incorrect.add(task_id)
        if len(file_incorrect) != total - correct:
            raise OfficialResultError(
                f"BFCL incorrect-case count does not match header: {path}"
            )
    if total_count != len(generated_ids) or correct_count != total_count - len(
        incorrect_ids
    ):
        raise OfficialResultError("BFCL score totals do not match generated task IDs")
    if not incorrect_ids.issubset(generated_ids):
        raise OfficialResultError("BFCL score files contain unknown task IDs")
    outcomes = {task_id: task_id not in incorrect_ids for task_id in generated_ids}
    return _attach_costs(outcomes, load_cost_records(costs_path), source=score_path)


def _string_id_set(raw: Mapping[str, Any], key: str, source: Path) -> set[str]:
    values = raw.get(key)
    if not isinstance(values, list):
        raise OfficialResultError(f"SWE-bench {key} must be a list: {source}")
    result = {_task_id(value, source=source) for value in values}
    if len(result) != len(values):
        raise OfficialResultError(f"SWE-bench {key} contains duplicate IDs: {source}")
    return result


def normalize_swe_bench_verified(
    results_path: Path, *, costs_path: Path
) -> list[dict[str, Any]]:
    """Normalize the official SWE-bench schema-v2 run report."""
    source = results_path / "results.json" if results_path.is_dir() else results_path
    raw = _load_json(source)
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 2:
        raise OfficialResultError(f"SWE-bench schema_version=2 is required: {source}")
    total = raw.get("total_instances")
    if isinstance(total, bool) or not isinstance(total, int) or total < 1:
        raise OfficialResultError(f"invalid SWE-bench total_instances: {source}")
    submitted = _string_id_set(raw, "submitted_ids", source)
    resolved = _string_id_set(raw, "resolved_ids", source)
    if len(submitted) != total:
        raise OfficialResultError(
            "partial SWE-bench runs cannot satisfy standard benchmark evidence"
        )
    if not resolved.issubset(submitted):
        raise OfficialResultError("SWE-bench resolved IDs are not all submitted")
    outcomes = {task_id: task_id in resolved for task_id in sorted(submitted)}
    return _attach_costs(outcomes, load_cost_records(costs_path), source=source)


def write_jsonl(records: Iterable[Mapping[str, Any]], output: Path) -> None:
    """Write normalized records atomically enough for a completed CLI process."""
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    output.write_text(payload, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Build the official-artifact normalization CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("tau3-bench", "bfcl-v4", "swe-bench-verified"),
        required=True,
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--cost-records", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Normalize one official suite artifact into the paired-harness contract."""
    args = build_parser().parse_args(argv)
    try:
        if args.suite == "tau3-bench":
            records = normalize_tau3(args.results, costs_path=args.cost_records)
        elif args.suite == "bfcl-v4":
            if args.scores is None or args.cost_records is None:
                raise OfficialResultError(
                    "BFCL V4 requires --scores and --cost-records"
                )
            records = normalize_bfcl_v4(
                args.results, args.scores, costs_path=args.cost_records
            )
        else:
            if args.cost_records is None:
                raise OfficialResultError("SWE-bench Verified requires --cost-records")
            records = normalize_swe_bench_verified(
                args.results, costs_path=args.cost_records
            )
        write_jsonl(records, args.output)
    except OfficialResultError as exc:
        print(f"official result rejected: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
