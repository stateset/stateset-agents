"""
Evaluate symbolic physics predictions against task constraints.

Usage:
    python examples/physics_symbolic_evaluate.py \
        --tasks examples/data/symbolic_physics_tasks.jsonl \
        --predictions /path/to/predictions.jsonl
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.environments.symbolic_physics import (
    SymbolicPhysicsTask,
    load_symbolic_tasks,
)
from stateset_agents.rewards.symbolic_physics_reward import (
    SymbolicPhysicsRewardFunction,
    SymbolicRewardConfig,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_json_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError("Unsupported predictions format")


def _load_predictions(path: Path) -> Dict[str, str]:
    records = _load_json_records(path)
    predictions: Dict[str, str] = {}
    for record in records:
        task_id = record.get("id") or record.get("task_id") or record.get("name")
        expression = record.get("expression") or record.get("prediction")
        if task_id and expression:
            predictions[str(task_id)] = str(expression)
    return predictions


async def _score_task(
    task: SymbolicPhysicsTask,
    expression: str,
    reward_fn: SymbolicPhysicsRewardFunction,
) -> Dict[str, Any]:
    turn = ConversationTurn(role="assistant", content=expression)
    result = await reward_fn.compute_reward(turns=[turn], context=task.to_dict())
    return {
        "id": task.task_id,
        "expression": expression,
        "score": result.score,
        "components": result.components,
        "metadata": result.metadata,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate symbolic physics predictions.")
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("examples/data/symbolic_physics_tasks.jsonl"),
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--rel-tol", type=float, default=1e-2)
    parser.add_argument("--abs-tol", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    tasks = load_symbolic_tasks(args.tasks)
    predictions = _load_predictions(args.predictions)

    reward_fn = SymbolicPhysicsRewardFunction(
        config=SymbolicRewardConfig(
            num_samples=args.num_samples,
            rel_tol=args.rel_tol,
            abs_tol=args.abs_tol,
        )
    )

    results: List[Dict[str, Any]] = []
    missing: List[str] = []
    for task in tasks:
        expr = predictions.get(task.task_id)
        if not expr:
            missing.append(task.task_id)
            continue
        results.append(await _score_task(task, expr, reward_fn))

    if results:
        avg_score = sum(r["score"] for r in results) / len(results)
    else:
        avg_score = 0.0

    summary = {
        "num_tasks": len(tasks),
        "num_scored": len(results),
        "num_missing": len(missing),
        "avg_score": avg_score,
        "missing": missing,
    }

    if args.output:
        payload = {"summary": summary, "results": results}
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        logger.info("Summary: %s", summary)


if __name__ == "__main__":
    asyncio.run(main())
