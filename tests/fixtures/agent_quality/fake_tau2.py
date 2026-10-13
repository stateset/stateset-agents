"""Offline stand-in for ``tau2 run`` that queries the bound endpoint.

Each task's outcome is derived from which policy the endpoint reports having
served, so the dry run proves the shim routed every policy to its own weights.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

TASKS = 10


def _served_policy(model: str) -> str:
    base = os.environ["HOSTED_VLLM_API_BASE"]
    request = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(
            {"model": model, "messages": [{"role": "user", "content": "hi"}]}
        ).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['HOSTED_VLLM_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())["choices"][0]["message"]["content"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--agent-llm", required=True)
    parser.add_argument("--user-llm", required=True)
    parser.add_argument("--num-trials", type=int, required=True)
    parser.add_argument("--task-split-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--max-concurrency", type=int, required=True)
    parser.add_argument("--save-to", required=True)
    args = parser.parse_args()
    assert args.action == "run"
    assert os.environ.get("OPENAI_API_KEY"), "user simulator key missing"
    provider, model = args.agent_llm.split("/", 1)
    assert provider == "hosted_vllm", args.agent_llm
    policy = _served_policy(model)
    threshold = 7 - args.seed % 2 if policy == "trained" else 4
    simulations = [
        {
            "task_id": str(task),
            "reward_info": {
                "reward": 1.0 if (task + args.seed) % TASKS < threshold else 0.0
            },
            "agent_cost": 0.0,
            "user_cost": 0.001,
        }
        for task in range(TASKS)
    ]
    save_dir = Path(args.save_to)
    save_dir.mkdir(parents=True, exist_ok=False)
    (save_dir / "results.json").write_text(
        json.dumps({"simulations": simulations}), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
