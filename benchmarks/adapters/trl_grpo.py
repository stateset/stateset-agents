#!/usr/bin/env python3
"""Independent upstream-TRL adapter for the measured shootout protocol."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

CONFIG_FIELDS = {
    "num_train_examples",
    "num_eval_examples",
    "max_steps",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "num_generations",
    "max_prompt_length",
    "max_completion_length",
    "temperature",
    "beta",
    "lora_r",
    "lora_alpha",
}


def canonical_digest(config: dict[str, Any]) -> str:
    """Return the shootout protocol's canonical config digest."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def supported_kwargs(callable_obj: Any, values: dict[str, Any]) -> dict[str, Any]:
    """Filter values across supported TRL 0.14–1.x API signatures."""
    parameters = inspect.signature(callable_obj).parameters
    if any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values()):
        return values
    return {name: value for name, value in values.items() if name in parameters}


def completion_text(completion: Any) -> str:
    """Normalize plain-text and conversational TRL completion formats."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict) and isinstance(last.get("content"), str):
            return last["content"]
    raise TypeError(f"unsupported TRL completion shape: {type(completion).__name__}")


def evaluate(
    model: Any, tokenizer: Any, adapter: Any, examples: list[Any], max_tokens: int
) -> float:
    """Measure deterministic greedy task score for one model state."""
    import torch

    model.eval()
    scores: list[float] = []
    for example in examples:
        encoded = tokenizer(adapter.format_prompt(example), return_tensors="pt")
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_length = encoded["input_ids"].shape[-1]
        response = tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True)
        score, _ = adapter.score_response(example, response)
        scores.append(float(score))
    return sum(scores) / max(len(scores), 1)


def main() -> int:
    """Train upstream TRL directly and emit a neutral measured result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--task", choices=["gsm8k"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--adapter-output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()

    if len(args.model_revision) != 40 or len(args.dataset_revision) != 40:
        parser.error("model and dataset revisions must be full 40-character commits")
    try:
        config = json.loads(args.config_json)
    except json.JSONDecodeError as exc:
        parser.error(f"invalid --config-json: {exc}")
    if not isinstance(config, dict) or set(config) != CONFIG_FIELDS:
        parser.error("--config-json has an unsupported schema")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import trl
    from trl import GRPOConfig, GRPOTrainer

    from scripts.run_phase0_benchmark import GSM8KAdapter
    from stateset_agents.utils.reproducibility import set_all_seeds

    if not torch.cuda.is_available():
        parser.error("the measured TRL adapter requires CUDA")
    set_all_seeds(args.seed)
    adapter = GSM8KAdapter()
    train_examples, eval_examples = adapter.load(
        int(config["num_train_examples"]),
        int(config["num_eval_examples"]),
        dataset_revision=args.dataset_revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.model_revision, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        torch_dtype=torch.bfloat16,
    )
    model.to("cuda")
    baseline = evaluate(
        model,
        tokenizer,
        adapter,
        eval_examples,
        int(config["max_completion_length"]),
    )
    dataset = Dataset.from_dict(
        {
            "prompt": [adapter.format_prompt(example) for example in train_examples],
            "example_index": list(range(len(train_examples))),
        }
    )

    def reward_func(
        completions: list[Any], example_index: list[int], **_: Any
    ) -> list[float]:
        return [
            float(
                adapter.score_response(
                    train_examples[int(index)], completion_text(completion)
                )[0]
            )
            for completion, index in zip(completions, example_index, strict=True)
        ]

    training_args = GRPOConfig(
        **supported_kwargs(
            GRPOConfig,
            {
                "output_dir": str(args.artifact_dir),
                "max_steps": int(config["max_steps"]),
                "per_device_train_batch_size": int(
                    config["per_device_train_batch_size"]
                ),
                "gradient_accumulation_steps": int(
                    config["gradient_accumulation_steps"]
                ),
                "learning_rate": float(config["learning_rate"]),
                "num_generations": int(config["num_generations"]),
                "max_prompt_length": int(config["max_prompt_length"]),
                "max_completion_length": int(config["max_completion_length"]),
                "temperature": float(config["temperature"]),
                "beta": float(config["beta"]),
                "bf16": True,
                "report_to": [],
                "save_strategy": "no",
            },
        )
    )
    peft_config = LoraConfig(
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        task_type="CAUSAL_LM",
    )
    trainer = GRPOTrainer(
        **supported_kwargs(
            GRPOTrainer,
            {
                "model": model,
                "args": training_args,
                "reward_funcs": reward_func,
                "train_dataset": dataset,
                "processing_class": tokenizer,
                "peft_config": peft_config,
            },
        )
    )
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    final_score = evaluate(
        trainer.model,
        tokenizer,
        adapter,
        eval_examples,
        int(config["max_completion_length"]),
    )
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.artifact_dir))
    tokenizer.save_pretrained(args.artifact_dir)
    result = {
        "status": "completed",
        "measured": True,
        "config_sha256": canonical_digest(config),
        "framework_version": trl.__version__,
        "artifact_path": str(args.artifact_dir.resolve()),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "cuda": str(torch.version.cuda),
        },
        "metrics": {
            "samples_processed": int(config["max_steps"])
            * int(config["per_device_train_batch_size"])
            * int(config["gradient_accumulation_steps"])
            * int(config["num_generations"]),
            "peak_vram_mb": torch.cuda.max_memory_allocated(0) / (1024 * 1024),
            "eval_score_baseline": baseline,
            "eval_score_final": final_score,
        },
    }
    args.adapter_output.parent.mkdir(parents=True, exist_ok=True)
    args.adapter_output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
