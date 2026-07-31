"""
Supervised fine-tune (SFT) from a curated chat-format JSONL.

Takes the output of ``prepare_sft_dataset.py --format chat`` — JSONL where
each line is ``{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}``
— and fine-tunes a Hugging Face causal LM with LoRA.

This script closes the curation loop: chat → capture → grade → curate →
prepare_sft → **sft_from_curated** → trained adapter → chat with it →
repeat.

**The implementation lives in** ``stateset_agents.training.sft``. This file is
the standalone CLI over it. The logic moved into the package because
``scripts*`` is excluded from the wheel, and remote workers provisioned by
``stateset_agents.remote`` install the published package and nothing else —
so the job has to be importable, not merely present in a checkout. Every
public name this module used to define is re-exported below, so existing
callers and tests are unaffected.

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

import sys

# The framework is the source of truth for the job itself; this script only
# parses arguments. Re-exported so existing importers keep working.
from stateset_agents.training.sft import (  # noqa: F401
    build_parser,
    gpu_available,
    load_chat_dataset,
    logger,
    print_training_plan,
    run_sft,
    run_sft_job,
)
from stateset_agents.training.sft import main as _main

__all__ = [
    "gpu_available",
    "load_chat_dataset",
    "main",
    "print_training_plan",
    "run_sft",
    "run_sft_job",
]


def main() -> int:
    """Delegate to the packaged CLI — see stateset_agents.training.sft."""
    return _main()


if __name__ == "__main__":
    sys.exit(main())
