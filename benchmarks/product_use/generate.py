"""Generate synthetic product-use holdout tasks and an optional local oracle.

The task file is evaluator-only. Share the prompts file with participants, and
keep the seed and oracle private when using this for an independent evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "0.4"
FAMILIES = (
    "ingest",
    "grade",
    "preview",
    "discover_grade",
    "discover_preview",
)
PRESETS = ("qwen3.5-27b", "qwen3.5-0.8b")


def _token(seed: str, family: str, index: int) -> str:
    """Derive a stable opaque suffix without copying the seed into a task."""
    digest = hmac.new(
        seed.encode("utf-8"), f"{family}:{index}".encode(), hashlib.sha256
    ).hexdigest()
    return digest[:12]


def generate(
    seed: str, variants: int = 1
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, Any]]]:
    """Build evaluator tasks, participant prompts, and passing local traces."""
    if not seed:
        raise ValueError("seed must be nonempty")
    if not 1 <= variants <= 50:
        raise ValueError("variants must be between 1 and 50")
    tasks: list[dict[str, Any]] = []
    prompts: list[dict[str, str]] = []
    oracle: list[dict[str, Any]] = []
    for index in range(variants):
        for family in FAMILIES:
            token = _token(seed, family, index)
            task_id = f"{family}-{token}"
            task: dict[str, Any] = {
                "id": task_id,
                "schema_version": SCHEMA_VERSION,
                "interface": "mcp",
            }
            calls: list[dict[str, Any]] = []
            if family == "ingest":
                input_path = f"inputs/{token}.jsonl"
                output_dir = f"converted/{token}"
                task.update(
                    {
                        "fixture": "openai_support",
                        "goal": "ingest_transcripts",
                        "params": {
                            "input_path": input_path,
                            "output_dir": output_dir,
                        },
                        "prompt": (
                            f"Ingest {input_path} in OpenAI format and write "
                            f"transcripts under {output_dir}."
                        ),
                    }
                )
                calls.append(
                    {
                        "name": "ingest_transcripts",
                        "arguments": {
                            "input_path": input_path,
                            "format": "openai",
                            "output_dir": output_dir,
                        },
                    }
                )
            elif family in {"grade", "discover_grade"}:
                history_path = f"cases/{token}/good.jsonl"
                task.update(
                    {
                        "fixture": "transcripts_support",
                        "goal": "grade_transcript",
                        "params": {"history_path": history_path},
                        "prompt": (
                            f"{'List supported rewards, confirm customer_support exists, then ' if family == 'discover_grade' else ''}"
                            f"grade {history_path} with the customer_support reward."
                        ),
                    }
                )
                if family == "discover_grade":
                    task["discovery"] = "list_rewards"
                    calls.append({"name": "list_rewards", "arguments": {}})
                calls.append(
                    {
                        "name": "grade_transcript",
                        "arguments": {
                            "history_path": history_path,
                            "reward": "customer_support",
                        },
                    }
                )
            else:
                preset = PRESETS[index % len(PRESETS)]
                task.update(
                    {
                        "fixture": "none",
                        "goal": "dry_run_finetune",
                        "params": {"preset": preset},
                        "prompt": (
                            f"{'List model presets, confirm this preset exists, then ' if family == 'discover_preview' else ''}"
                            f"preview a fine-tuning config for {preset} without starting training."
                        ),
                    }
                )
                if family == "discover_preview":
                    task["discovery"] = "list_model_presets"
                    calls.append({"name": "list_model_presets", "arguments": {}})
                calls.append(
                    {
                        "name": "dry_run_finetune",
                        "arguments": {"model_preset": preset},
                    }
                )
            tasks.append(task)
            prompts.append(
                {
                    "id": task_id,
                    "schema_version": SCHEMA_VERSION,
                    "interface": "mcp",
                    "prompt": task["prompt"],
                }
            )
            oracle.append({"task_id": task_id, "calls": calls})
    return tasks, prompts, oracle


def _write_json(path: Path, value: Any) -> None:
    """Write one UTF-8 JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    """Generate a local holdout from a seed stored outside the repository."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-file", type=Path, required=True)
    parser.add_argument("--variants", type=int, default=1)
    parser.add_argument("--tasks-output", type=Path, required=True)
    parser.add_argument("--prompts-output", type=Path, required=True)
    parser.add_argument("--oracle-output", type=Path)
    args = parser.parse_args()
    outputs = [args.tasks_output, args.prompts_output]
    if args.oracle_output:
        outputs.append(args.oracle_output)
    resolved = [path.resolve() for path in outputs]
    if len(set(resolved)) != len(resolved) or args.seed_file.resolve() in resolved:
        parser.error("seed and output files must have distinct paths")
    seed = args.seed_file.read_text(encoding="utf-8").strip()
    tasks, prompts, oracle = generate(seed, args.variants)
    _write_json(args.tasks_output, tasks)
    _write_json(args.prompts_output, prompts)
    if args.oracle_output:
        _write_json(args.oracle_output, oracle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
