"""Check the installed distribution from outside its source checkout."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path


def main() -> None:
    """Exercise public imports and an optional real training/adapter round trip."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extra", choices=["minimal", "api", "training", "workers"], default="minimal"
    )
    args = parser.parse_args()
    package = importlib.import_module("stateset_agents")
    location = Path(package.__file__).resolve()
    if "site-packages" not in location.parts:
        raise RuntimeError(f"Expected an installed wheel, got {location}")
    importlib.import_module("traceforge_rl")
    from stateset_agents.training.config import TrainingConfig

    TrainingConfig(report_to="none").validate_for_training()
    if args.extra == "minimal" and importlib.util.find_spec("torch") is not None:
        raise RuntimeError("Minimal wheel smoke must run without torch")
    if args.extra in {"api", "workers"}:
        from stateset_agents.api.main import create_app

        assert create_app().openapi()["paths"]
    if args.extra == "workers":
        importlib.import_module("psycopg")
    if args.extra == "training":
        import tempfile

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
