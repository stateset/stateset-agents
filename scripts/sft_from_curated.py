"""
Supervised fine-tune (SFT) from a curated chat-format JSONL.

Takes the output of ``prepare_sft_dataset.py --format chat`` — JSONL where
each line is ``{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}``
— and fine-tunes a Hugging Face causal LM with LoRA.

This script closes the curation loop: chat → capture → grade → curate →
prepare_sft → **sft_from_curated** → trained adapter → chat with it →
repeat.

The implementation is deliberately minimal and uses ``transformers.Trainer``
directly rather than a framework-level abstraction:

* The framework's existing trainers (GSPO/GRPO/DAPO) are RL trainers — they
  expect an environment + reward function, not a static dataset.
* SFT is well-served by HF Trainer; reinventing it here would add maintenance
  burden without benefit.
* The minimal HF Trainer call composes cleanly with the framework's
  ``AgentConfig.peft_path`` field — drop the adapter directory at the path
  and ``stateset-agents serve --checkpoint`` picks it up.

Stub-aware: when ``torch.cuda.is_available()`` is False or transformers isn't
importable, the script prints the training plan it *would* run and exits 0.
This lets the integration test run on CPU-only CI while the real training
happens on GPU hosts.

Usage::

    # Prepare the dataset first
    python scripts/prepare_sft_dataset.py \\
        --input curated.jsonl --format chat \\
        --output sft_train.jsonl --min-score 0.7 --dedup

    # Train
    python scripts/sft_from_curated.py \\
        --dataset sft_train.jsonl \\
        --base-model Qwen/Qwen3.5-0.8B \\
        --output-dir outputs/sft_v1 \\
        --num-epochs 3 \\
        --lora-r 16
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("sft_from_curated")


def load_chat_dataset(path: Path) -> list[dict[str, Any]]:
    """Load a chat-format JSONL.

    Each row must have a ``messages`` key with a list of ``{role, content}``
    dicts. Returns the list of rows after validation.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    rows: list[dict[str, Any]] = []
    for line_num, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            logger.warning("Skipping line %d: %s", line_num, e)
            continue
        if "messages" not in row or not isinstance(row["messages"], list):
            logger.warning("Skipping line %d: missing 'messages' list", line_num)
            continue
        rows.append(row)
    return rows


def gpu_available() -> bool:
    """Detect whether we have a CUDA GPU + transformers."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def print_training_plan(
    rows: list[dict[str, Any]],
    base_model: str,
    output_dir: Path,
    num_epochs: int,
    lora_r: int,
    learning_rate: float,
    max_length: int,
) -> None:
    """Dump a human-readable summary of what training *would* do."""
    print()
    print("=" * 60)
    print("SFT Training Plan (dry run — no GPU detected)")
    print("=" * 60)
    print(f"  Dataset size:     {len(rows)} examples")
    print(f"  Base model:       {base_model}")
    print(f"  Output dir:       {output_dir}")
    print(f"  Epochs:           {num_epochs}")
    print(f"  LoRA r:           {lora_r}")
    print(f"  Learning rate:    {learning_rate}")
    print(f"  Max sequence:     {max_length}")
    print()
    sample = rows[0] if rows else None
    if sample:
        print("  First example:")
        for msg in sample.get("messages", [])[:2]:
            content = msg.get("content", "")[:60]
            print(f"    [{msg.get('role')}] {content}…")
    print()
    print("  To execute, run this script on a host with a CUDA-capable GPU.")
    print("=" * 60)


def run_sft(
    rows: list[dict[str, Any]],
    base_model: str,
    output_dir: Path,
    num_epochs: int,
    lora_r: int,
    lora_alpha: int,
    learning_rate: float,
    max_length: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
) -> Path:
    """Run the actual SFT training on GPU.

    Uses ``transformers.Trainer`` directly with a PEFT LoRA adapter on top of
    the base model. The result is saved as a LoRA adapter directory
    consumable by ``AgentConfig.peft_path``.

    Raises ``ImportError`` if transformers / peft / datasets aren't available.
    """
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    logger.info("Loading tokenizer and model: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype="bfloat16",
    )

    logger.info("Applying LoRA adapter (r=%d, alpha=%d)", lora_r, lora_alpha)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def render(row: dict[str, Any]) -> dict[str, Any]:
        """Apply the model's chat template to each row."""
        text = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = Dataset.from_list(rows).map(render, remove_columns=["messages"])
    dataset = dataset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        ),
        remove_columns=["text"],
    )

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],  # local-only by default
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    logger.info("Starting SFT…")
    trainer.train()
    logger.info("Training complete. Saving adapter to %s", output_dir)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Chat-format JSONL from prepare_sft_dataset.py.",
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Hugging Face base model name (e.g. Qwen/Qwen3.5-0.8B).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sft_v1"))
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the training plan without running it (forced "
        "automatically when no GPU is detected).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    rows = load_chat_dataset(args.dataset)
    if not rows:
        logger.error("No usable rows in %s", args.dataset)
        return 1
    logger.info("Loaded %d examples from %s", len(rows), args.dataset)

    if args.dry_run or not gpu_available():
        print_training_plan(
            rows=rows,
            base_model=args.base_model,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            lora_r=args.lora_r,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
        )
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = run_sft(
            rows=rows,
            base_model=args.base_model,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            per_device_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
    except ImportError as e:
        logger.error(
            "SFT requires the 'training' extras. Install with: pip install -e '.[training]'\n"
            "Details: %s",
            e,
        )
        return 2
    except Exception as e:  # noqa: BLE001 — surface the error
        logger.error("Training failed: %s: %s", type(e).__name__, e)
        return 2

    logger.info("Adapter saved to %s", path)
    print()
    print("Next steps:")
    print(
        f"  Chat with it:   stateset-agents chat --model {args.base_model} --checkpoint {path}"
    )
    print(
        f"  Serve it:       stateset-agents serve --checkpoint {path} --base-model {args.base_model}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
