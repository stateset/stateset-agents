"""Check the installed distribution from outside its source checkout."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.metadata
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def verify_improvement_loop() -> None:
    """Exercise grading and MCP previews without repository-only helpers."""
    from stateset_agents.cli_improve import run_improve
    from stateset_agents.mcp_server import dry_run_finetune, grade_transcript

    with tempfile.TemporaryDirectory(prefix="stateset-wheel-improve-") as directory:
        root = Path(directory)
        transcripts = root / "transcripts"
        transcripts.mkdir()
        history = transcripts / "sample.jsonl"
        history.write_text(
            "\n".join(
                json.dumps(turn)
                for turn in [
                    {"role": "user", "content": "I need a refund for order 1234."},
                    {
                        "role": "assistant",
                        "content": "I would be happy to help you with a refund "
                        "for order 1234 right away.",
                    },
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        summary = run_improve(
            transcripts=str(transcripts),
            reward="customer_support",
            output=str(root / "improved"),
        )
        assert summary["assistant_turn_count"] == 1
        assert summary["curated_count"] == 1
        graded = asyncio.run(grade_transcript(str(history), "customer_support"))
        assert graded["assistant_turn_count"] == 1
        assert graded["mean_score"] == summary["mean_score"]
        preview = dry_run_finetune("qwen3.5-0.8b")
        assert preview["config"]["model_id"] == "Qwen/Qwen3.5-0.8B-Base"


def verify_private_workflow() -> None:
    """Exercise the packaged private workflow and bundled public evidence offline."""
    from stateset_agents.improvement import (
        Experiment,
        ExperimentConfig,
        evidence_catalog,
    )
    from stateset_agents.improvement.fixtures import create_synthetic_fixture

    with tempfile.TemporaryDirectory(prefix="stateset-wheel-private-") as directory:
        root = Path(directory)
        dataset = create_synthetic_fixture(root / "fixtures", count=8)
        experiment = Experiment.create(
            root / "experiment",
            dataset,
            ExperimentConfig(
                "Qwen/Qwen3.5-9B", "river://offline-placeholder", min_cases=8
            ),
        )
        assert experiment.status()["completed_stages"] == 0
        assert experiment.status()["source_kind"] == "synthetic"
        assert evidence_catalog()["entries"]


def verify_quickstart(path: Path) -> None:
    """Execute the marked offline documentation steps outside the checkout."""
    source = path.read_text(encoding="utf-8")
    start, end = "<!-- quickstart-smoke:start -->", "<!-- quickstart-smoke:end -->"
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError("Quickstart must have exactly one marked smoke section")
    if source.index(end) < source.index(start):
        raise ValueError("Quickstart smoke markers are out of order")
    section = source.split(start, 1)[1].split(end, 1)[0]
    blocks = re.findall(r"^```bash\n(.*?)^```\s*$", section, re.MULTILINE | re.DOTALL)
    if not any(block.strip() for block in blocks):
        raise ValueError("Quickstart smoke section contains no Bash commands")
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME"}
    }
    env.update(
        PATH=str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", ""),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    with tempfile.TemporaryDirectory(prefix="stateset-wheel-quickstart-") as directory:
        env["TMPDIR"] = directory
        subprocess.run(
            ["bash", "-e", "-o", "pipefail", "-c", "\n".join(blocks)],
            cwd=directory,
            env=env,
            check=True,
            timeout=180,
        )


def main() -> None:
    """Exercise public imports and an optional real training/adapter round trip."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extra", choices=["minimal", "api", "training", "workers"], default="minimal"
    )
    parser.add_argument(
        "--quickstart", type=Path, help="Execute offline steps from this guide"
    )
    args = parser.parse_args()
    package = importlib.import_module("stateset_agents")
    location = Path(package.__file__).resolve()
    if "site-packages" not in location.parts or not location.is_relative_to(
        Path(sys.prefix).resolve()
    ):
        raise RuntimeError(f"Expected an installed wheel, got {location}")
    importlib.import_module("traceforge_rl")
    from stateset_agents.training.config import TrainingConfig

    TrainingConfig(report_to="none").validate_for_training()
    if args.extra == "minimal" and importlib.util.find_spec("torch") is not None:
        raise RuntimeError("Minimal wheel smoke must run without torch")
    verify_improvement_loop()
    verify_private_workflow()
    if args.quickstart is not None:
        verify_quickstart(args.quickstart)
    if args.extra in {"api", "workers"}:
        from stateset_agents.api.main import create_app

        assert create_app().openapi()["paths"]
    if args.extra == "workers":
        importlib.import_module("psycopg")
    if args.extra == "training":
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import GPT2Config, GPT2LMHeadModel

        from stateset_agents.training.sampled_workflow import workflow_tokenizer

        torch.set_num_threads(1)
        model = get_peft_model(
            GPT2LMHeadModel(
                GPT2Config(vocab_size=16, n_embd=8, n_layer=1, n_head=1, n_positions=8)
            ),
            LoraConfig(r=2, target_modules=["c_attn"], task_type="CAUSAL_LM"),
        )
        model(
            torch.tensor([[1, 2, 3]]), labels=torch.tensor([[1, 2, 3]])
        ).loss.backward()
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            assert list(Path(directory).glob("*.safetensors"))
        assert workflow_tokenizer().encode("request")
        with tempfile.TemporaryDirectory(prefix="stateset-wheel-smoke-") as output:
            importlib.import_module("trl").GRPOConfig(
                output_dir=output, bf16=False, report_to="none"
            )
    print(
        json.dumps(
            {
                "package": importlib.metadata.version("stateset-agents"),
                "extra": args.extra,
                "path": str(location),
                "passed": True,
            }
        )
    )


if __name__ == "__main__":
    main()
