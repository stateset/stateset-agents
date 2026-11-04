"""Recover a byte from a corrupted codeword using a fully handwritten neural net.

    python examples/handwritten_decoder.py --message 173 --flips 95

No training or model download. Optionally load an exported safetensors file
with ``--weights``. The code uses 384 transmitted bits per 8-bit message;
its purpose here is an exact, inspectable neural construction.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from stateset_agents.experimental.weight_space.handwritten_decoder import (
    CODE_BITS,
    HIDDEN,
    make_codebook,
    make_decoder,
    read_messages,
)


def main(argv: list[str] | None = None) -> None:
    """Encode, corrupt, and decode a user-specified byte with standard NN inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--message", type=int, default=173)
    parser.add_argument("--flips", type=int, default=95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args(argv)
    if not 0 <= args.message < HIDDEN:
        parser.error("--message must be between 0 and 255")
    if not 0 <= args.flips <= CODE_BITS:
        parser.error("--flips must be between 0 and 384 (the guarantee ends at 95)")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    model = make_decoder()
    if args.weights:
        from safetensors.torch import load_file

        model.load_state_dict(load_file(args.weights))
    received = make_codebook()[args.message].clone()
    positions = torch.randperm(
        CODE_BITS, generator=torch.Generator().manual_seed(args.seed)
    )[: args.flips]
    received[positions] *= -1
    with torch.inference_mode():
        decoded = int(read_messages(model(received)))
    logging.info(
        "parameters=%d message=%d bit_flips=%d decoded=%s correct=%s",
        sum(p.numel() for p in model.parameters()),
        args.message,
        args.flips,
        "erasure" if decoded == -1 else decoded,
        decoded == args.message,
    )


if __name__ == "__main__":
    main()
