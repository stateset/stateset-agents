"""Supervised fine-tune from a curated chat-format JSONL.

This is the job at the end of the improvement loop: chat → capture → grade →
curate → **sft** → trained adapter → chat with it → repeat.

The logic lives in the installed package rather than in ``scripts/`` because
``scripts*`` is excluded from the wheel (see ``[tool.setuptools.packages.find]``).
A remote worker provisioned by ``stateset_agents.remote`` installs the
published package and nothing else, so anything it must execute has to be
importable from here. ``scripts/sft_from_curated.py`` re-exports these names
and keeps the standalone CLI.

The implementation deliberately uses ``transformers.Trainer`` directly rather
than a framework-level abstraction:

* The framework's existing trainers (GSPO/GRPO/DAPO) are RL trainers — they
  expect an environment + reward function, not a static dataset.
* SFT is well-served by HF Trainer; reinventing it here would add maintenance
  burden without benefit.
* The minimal HF Trainer call composes cleanly with the framework's
  ``AgentConfig.peft_path`` field — drop the adapter directory at the path
  and ``stateset-agents serve --checkpoint`` picks it up.

Stub-aware: when no CUDA GPU is present or transformers isn't importable, the
job prints the training plan it *would* run and succeeds. That keeps the whole
submit → run → fetch path exercisable on CPU-only CI.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("sft_from_curated")

__all__ = [
    "build_training_arguments",
    "gpu_available",
    "load_base_model_for_sft",
    "load_chat_dataset",
    "print_training_plan",
    "run_sft",
    "run_sft_job",
]

#: Keys of a ``RemoteJobSpec`` dict that configure the provider rather than the
#: job. ``run_sft_job`` ignores them so a full spec can be passed straight in.
_PROVIDER_ONLY_KEYS = frozenset({"gpu", "timeout_s", "package_version"})


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

        return bool(torch.cuda.is_available())
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


#: Linear-layer names LoRA is normally applied to, across common decoder
#: architectures: separate q/k/v/o projections (Llama, Qwen, Mistral), fused
#: qkv (GPT-2 style ``c_attn``), and the MLP projections.
_LORA_TARGET_CANDIDATES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "c_attn",
    "c_proj",
    "c_fc",
    "query_key_value",
    "dense",
    "fc1",
    "fc2",
)


def build_training_arguments(training_arguments_cls: Any, **kwargs: Any) -> Any:
    """Construct ``TrainingArguments`` tolerating removed keyword arguments.

    transformers 5.x dropped arguments the 4.x line accepted (hit for real on
    a RunPod pod resolving transformers 5.15.0: ``warmup_ratio`` no longer
    exists and the job died after a 63GB model download). Filter kwargs
    against the constructor's actual signature and log what was dropped, so
    an optional tuning knob degrades gracefully instead of killing the run.
    """
    import inspect

    try:
        accepted = set(inspect.signature(training_arguments_cls.__init__).parameters)
    except (TypeError, ValueError):  # exotic __init__; pass everything through
        return training_arguments_cls(**kwargs)
    if any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in inspect.signature(
            training_arguments_cls.__init__
        ).parameters.values()
    ):
        return training_arguments_cls(**kwargs)
    dropped = sorted(k for k in kwargs if k not in accepted)
    if dropped:
        logger.warning(
            "TrainingArguments does not accept %s in this transformers "
            "version; continuing without them.",
            ", ".join(dropped),
        )
    return training_arguments_cls(**{k: v for k, v in kwargs.items() if k in accepted})


def load_base_model_for_sft(base_model: str):
    """Load ``base_model`` for text-only SFT, tolerating multimodal repos.

    Composite multimodal checkpoints (e.g. ``meta-models/Muse-Glimmer-30B``,
    model_type ``muse_glimmer``) register only under transformers'
    image-text-to-text auto-mapping, so ``AutoModelForCausalLM`` raises
    ValueError on them. Text-only SFT of the language stack still works:
    retry via ``AutoModelForImageTextToText`` and let LoRA target the
    text-stack projections.
    """
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(  # nosec: B615
            base_model,
            trust_remote_code=True,
            torch_dtype="bfloat16",
        )
    except ValueError as causal_exc:
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            raise causal_exc from None
        logger.info(
            "AutoModelForCausalLM rejected %s (%s); retrying as an "
            "image-text-to-text checkpoint and training the text stack.",
            base_model,
            causal_exc,
        )
        return AutoModelForImageTextToText.from_pretrained(  # nosec: B615
            base_model,
            trust_remote_code=True,
            torch_dtype="bfloat16",
        )


def infer_lora_target_modules(model: Any) -> list[str]:
    """Pick the module names LoRA should adapt on ``model``.

    peft can infer these itself, but only for architectures in its built-in
    mapping; for anything else it raises "Please specify ``target_modules``".
    That failure only appears once the real model is loaded on a GPU, so it
    escapes every CPU dry run.

    Returns the recognised names present on this model, or an empty list —
    which means "we did not recognise anything, let peft try" rather than a
    guess that would fail confusingly later.
    """
    found = set()
    for name, _module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in _LORA_TARGET_CANDIDATES:
            found.add(leaf)
    # The output head is deliberately excluded: adapting it inflates the
    # adapter with a vocab-sized matrix for no benefit on SFT.
    found.discard("lm_head")
    return sorted(found)


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
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    logger.info("Loading tokenizer and model: %s", base_model)
    # base_model is a caller-supplied CLI argument (public HF model repo id),
    # not attacker-controlled input; pinning a fixed revision would break
    # support for arbitrary user-chosen base models.
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True
    )  # nosec: B615
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_base_model_for_sft(base_model)

    target_modules = infer_lora_target_modules(model)
    logger.info(
        "Applying LoRA adapter (r=%d, alpha=%d, targets=%s)",
        lora_r,
        lora_alpha,
        ",".join(target_modules) or "<peft default>",
    )
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        # Passed explicitly: peft only infers these for architectures in its
        # built-in mapping and raises on anything else (hit for real on
        # Qwen3.5). None means "nothing recognised — let peft try".
        target_modules=target_modules or None,
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

    args = build_training_arguments(
        TrainingArguments,
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


def run_sft_job(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one SFT job from a ``RemoteJobSpec``-shaped dict.

    This is the single entrypoint every executor calls — locally in a
    subprocess, or inside a provider's container. It never raises: failures
    are reported as a non-zero ``returncode`` with the captured output, so
    the calling executor has one uniform thing to interpret.

    Returns ``{"returncode": int, "logs": list[str], "output_dir": str}``.
    """
    job = {k: v for k, v in payload.items() if k not in _PROVIDER_ONLY_KEYS}
    output_dir = Path(job["output_dir"])
    buffer = io.StringIO()

    handler = logging.StreamHandler(buffer)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)
    previous_level = logger.level
    logger.setLevel(logging.INFO)

    returncode = 0
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            rows = load_chat_dataset(Path(job["dataset"]))
            if not rows:
                logger.error("No usable rows in %s", job["dataset"])
                returncode = 1
            else:
                logger.info("Loaded %d examples from %s", len(rows), job["dataset"])
                if job.get("dry_run") or not gpu_available():
                    print_training_plan(
                        rows=rows,
                        base_model=job["base_model"],
                        output_dir=output_dir,
                        num_epochs=job["num_epochs"],
                        lora_r=job["lora_r"],
                        learning_rate=job["learning_rate"],
                        max_length=job["max_length"],
                    )
                else:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    run_sft(
                        rows=rows,
                        base_model=job["base_model"],
                        output_dir=output_dir,
                        num_epochs=job["num_epochs"],
                        lora_r=job["lora_r"],
                        lora_alpha=job["lora_alpha"],
                        learning_rate=job["learning_rate"],
                        max_length=job["max_length"],
                        per_device_batch_size=job["per_device_batch_size"],
                        gradient_accumulation_steps=job["gradient_accumulation_steps"],
                    )
    except Exception as exc:  # reported, never raised — see docstring
        logger.error("SFT job failed: %s", exc)
        returncode = 1
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)

    return {
        "returncode": returncode,
        "logs": buffer.getvalue().splitlines(),
        "output_dir": str(output_dir),
    }


def build_parser() -> Any:
    """Argument parser for the job. Shared by the CLI and the ``scripts/`` wrapper."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Supervised fine-tune from curated data."
    )
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
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the job from command-line arguments.

    Available as ``python -m stateset_agents.training.sft`` — the only
    invocation a remote worker has, since it holds the wheel and no checkout.
    """
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    outcome = run_sft_job(
        {
            "dataset": str(args.dataset),
            "base_model": args.base_model,
            "output_dir": str(args.output_dir),
            "num_epochs": args.num_epochs,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "per_device_batch_size": args.per_device_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "dry_run": args.dry_run,
        }
    )

    for line in outcome["logs"]:
        print(line)

    if outcome["returncode"] != 0:
        return int(outcome["returncode"])

    path = outcome["output_dir"]
    print()
    print("Next steps:")
    print(
        f"  Chat with it:   stateset-agents chat "
        f"--model {args.base_model} --checkpoint {path}"
    )
    print(
        f"  Serve it:       stateset-agents serve "
        f"--checkpoint {path} --base-model {args.base_model}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover — exercised via subprocess tests
    import sys

    sys.exit(main())
