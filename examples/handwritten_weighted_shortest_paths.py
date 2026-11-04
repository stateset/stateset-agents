"""Use a two-million-parameter handwritten NN to solve a weighted routing problem.

    PYTHONPATH=. python examples/handwritten_weighted_shortest_paths.py

A direct edge 0 -> 1 costs 7. The two-edge route 0 -> 2 -> 1 costs 2 + 3 = 5.
The NN returns the minimum cost to all seventeen nodes; 113 means unreachable.
No training or model download is required. Use --weights to load an export.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from stateset_agents.experimental.weight_space.handwritten_weighted_graph import (
    encode_weighted_graphs,
    make_weighted_shortest_paths,
)


def main(argv: list[str] | None = None) -> None:
    """Solve a cheaper detour using only the standard neural forward pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    model = make_weighted_shortest_paths()
    if args.weights:
        from safetensors.torch import load_model

        load_model(model, str(args.weights))
    costs = torch.zeros(1, 17, 17)
    costs[0, 0, 1] = 7
    costs[0, 0, 2] = 2
    costs[0, 2, 1] = 3
    with torch.inference_mode():
        output = model(encode_weighted_graphs(costs, torch.tensor([0])))
    logging.info(
        "parameters=%d nonzero=%d",
        sum(p.numel() for p in model.parameters()),
        sum(int(p.count_nonzero()) for p in model.parameters()),
    )
    logging.info("Minimum costs from node 0: %s (113=unreachable)", output[0].tolist())
    logging.info(
        "Cost to node 1: %g; direct edge costs 7, cheaper detour costs 5", output[0, 1]
    )


if __name__ == "__main__":
    main()
