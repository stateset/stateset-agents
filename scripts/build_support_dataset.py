#!/usr/bin/env python3
"""Build and export ``stateset/customer-support-multiturn-v1``.

Writes ``train.jsonl``, ``eval.jsonl`` and a ``README.md`` dataset card to
``--output`` and prints the SHA-256 over the two JSONL files, which is the
value to pin when publishing or validating the dataset.

Usage::

    python scripts/build_support_dataset.py --output data/support-multiturn-v1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stateset_agents.data.support_multiturn import (  # noqa: E402
    DATASET_NAME,
    build_support_dataset,
    export_dataset,
)

logger = logging.getLogger("build_support_dataset")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, build the dataset, export it, and print its sha256."""
    parser = argparse.ArgumentParser(description=f"Build {DATASET_NAME}")
    parser.add_argument("--output", required=True, type=Path, help="Output dir")
    parser.add_argument("--seed", type=int, default=0, help="Generator seed")
    parser.add_argument("--n-train", type=int, default=560)
    parser.add_argument("--n-eval", type=int, default=240)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dataset = build_support_dataset(
        seed=args.seed, n_train=args.n_train, n_eval=args.n_eval
    )
    sha256 = export_dataset(dataset, args.output)
    logger.info(
        "wrote %d train / %d eval rows to %s",
        len(dataset["train"]),
        len(dataset["eval"]),
        args.output,
    )
    print(sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
