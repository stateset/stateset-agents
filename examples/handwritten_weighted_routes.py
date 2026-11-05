"""Write every weight of a network that returns an exact minimum-cost route."""

from __future__ import annotations

import logging

import torch

from stateset_agents.experimental.weight_space.handwritten_routes import (
    encode_weighted_routes,
    make_weighted_routes,
)


def main() -> None:
    """Show a cheaper detour and display the network's one-hot route output."""
    model = make_weighted_routes()
    graph = torch.zeros(1, 17, 17)
    graph[0, 0, 1], graph[0, 0, 2], graph[0, 2, 1] = 7, 2, 3
    inputs = encode_weighted_routes(graph, torch.tensor([0]), torch.tensor([1]))
    with torch.inference_mode():
        output = model(inputs)[0]
    route = output[1:].reshape(17, 17)
    # Decode one-hot rows; the NN has already selected and traversed the route.
    vertices = route[route.sum(1) > 0].argmax(1).tolist()
    logging.info("Minimum cost: %g; route: %s", output[0], vertices)
    logging.info(
        "Stored parameters: %d; nonzero: %d",
        sum(p.numel() for p in model.parameters()),
        sum(int(p.count_nonzero()) for p in model.parameters()),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
