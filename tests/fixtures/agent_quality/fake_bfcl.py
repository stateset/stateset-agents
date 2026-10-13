"""Offline stand-in for the BFCL ``generate``/``evaluate`` CLI.

``generate`` sends the request model name BFCL's local handlers send (the
``--local-model-path``) to the bound endpoint and records what answered;
``evaluate`` writes official-format score files (header + incorrect cases).
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

CATEGORIES = {
    "non_live/BFCL_v4_simple_python": 8,
    "multi_turn/BFCL_v4_multi_turn_base": 4,
}


def _completion(model: str) -> str:
    request = urllib.request.Request(
        f"{os.environ['REMOTE_OPENAI_BASE_URL']}/completions",
        data=json.dumps({"model": model, "prompt": "call a tool"}).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['REMOTE_OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())["choices"][0]["text"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("generate", "evaluate"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-category", required=True)
    parser.add_argument("--backend")
    parser.add_argument("--skip-server-setup", action="store_true")
    parser.add_argument("--local-model-path")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--score-dir")
    parser.add_argument("--num-threads")
    args = parser.parse_args()
    project_root = Path(os.environ["BFCL_PROJECT_ROOT"])
    project_root.mkdir(parents=True, exist_ok=True)
    model_dir = args.model.replace("/", "_")
    result_root = Path(args.result_dir) / model_dir
    if args.action == "generate":
        assert args.skip_server_setup and args.local_model_path
        assert (Path(args.local_model_path) / "tokenizer_config.json").is_file()
        text = _completion(args.local_model_path)
        for stem, count in CATEGORIES.items():
            path = result_root / f"{stem}_result.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            category = Path(stem).name.removeprefix("BFCL_v4_")
            path.write_text(
                "".join(
                    json.dumps({"id": f"{category}_{index}", "result": text}) + "\n"
                    for index in range(count)
                ),
                encoding="utf-8",
            )
        return
    score_root = Path(args.score_dir) / model_dir
    for stem in CATEGORIES:
        rows = [
            json.loads(line)
            for line in (result_root / f"{stem}_result.json").read_text().splitlines()
        ]
        incorrect = [
            {"id": row["id"], "valid": False}
            for index, row in enumerate(rows)
            if (index % 4 == 0 if row["result"] == "trained" else index % 2 == 1)
        ]
        header = {
            "accuracy": (len(rows) - len(incorrect)) / len(rows),
            "correct_count": len(rows) - len(incorrect),
            "total_count": len(rows),
        }
        path = score_root / f"{stem}_score.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in [header, *incorrect]),
            encoding="utf-8",
        )
    (Path(args.score_dir) / "data_overall.csv").write_text("rank,model\n")


if __name__ == "__main__":
    main()
