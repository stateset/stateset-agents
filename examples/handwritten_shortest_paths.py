"""Find exact hop distances with a fully handwritten recurrent neural network.

    PYTHONPATH=. python examples/handwritten_shortest_paths.py
    PYTHONPATH=. python examples/handwritten_shortest_paths.py --cut-edge 5

The graph is a directed twelve-node chain. The network uses only standard
Linear and ReLU modules, with no training or model download. A distance of
12 means unreachable. Optionally load an exported file with ``--weights``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from stateset_agents.experimental.weight_space.handwritten_graph import (
    DEFAULT_NODES,
    encode_graphs,
    make_shortest_paths,
)


def main(argv: list[str] | None = None) -> None:
    """Solve a chain, optionally with an edge removed, through ordinary NN inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--cut-edge", type=int)
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args(argv)
    n = DEFAULT_NODES
    if not 0 <= args.source < n:
        parser.error("--source must be between 0 and 11")
    if args.cut_edge is not None and not 0 <= args.cut_edge < n - 1:
        parser.error("--cut-edge must be between 0 and 10 (removes i -> i+1)")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    model = make_shortest_paths()
    if args.weights:
        from safetensors.torch import load_model

        # load_model restores the shared tensors used in each recurrent round.
        load_model(model, str(args.weights))
    adjacency = torch.zeros(1, n, n)
    adjacency[0, torch.arange(n - 1), torch.arange(1, n)] = 1
    if args.cut_edge is not None:
        adjacency[0, args.cut_edge, args.cut_edge + 1] = 0
    inputs = encode_graphs(adjacency, torch.tensor([args.source]))
    with torch.inference_mode():
        distances = model(inputs)[0].tolist()
    logging.info(
        "parameters=%d nonzero=%d rounds=%d source=%d distances=%s (12=unreachable)",
        sum(p.numel() for p in model.parameters()),
        sum(int(p.count_nonzero()) for p in model.parameters()),
        n - 1,
        args.source,
        distances,
    )


if __name__ == "__main__":
    main()
