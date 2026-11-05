"""Compile exact weighted routes into ordinary Linear/ReLU layers.

For a positive-integer directed graph, a source, and a target, the network
returns [minimum_cost, flattened_route]. The route is an [n,n] one-hot matrix:
one vertex per row, source first, followed by zero rows after the target.
Unreachable pairs return cost (n-1)*max_cost+1 and an all-zero route.

The network first computes distances to the target on the reversed graph,
selects the lowest-index tight outgoing neighbor, and follows those neural
one-hot transitions for n-1 rounds. Both recurrence blocks share weights.
Positive costs imply strictly decreasing distances along each selected edge,
so the route is simple and has minimum cost. Ties yield the lexicographically
smallest shortest route under the supplied vertex labels.

There is no search, argmin, multiplication gate, or custom forward method at
inference time. Every weight is analytic; no data or optimizer is used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .handwritten_weighted_graph import (
    DEFAULT_MAX_COST,
    DEFAULT_NODES,
    encode_weighted_graphs,
    make_weighted_shortest_paths,
)

if TYPE_CHECKING:
    import torch


def encode_weighted_routes(
    costs: torch.Tensor,
    sources: torch.Tensor,
    targets: torch.Tensor,
    max_cost: int = DEFAULT_MAX_COST,
) -> torch.Tensor:
    """Validate and pack graphs/endpoints without computing distances or routes.

    The cost domain matches ``encode_weighted_graphs``: zero is an absent edge,
    diagonals are zero, and present edges have integer costs in 1..max_cost.
    Both endpoint tensors have shape [batch] and use int32 or int64.
    """
    import torch

    forward = encode_weighted_graphs(costs, sources, max_cost)
    reversed_graph = encode_weighted_graphs(costs.transpose(1, 2), targets, max_cost)
    nodes = costs.shape[1]
    source_flags = (forward[:, -nodes:] == 0).float()
    return torch.cat((reversed_graph, source_flags), dim=1)


def make_weighted_routes(
    nodes: int = DEFAULT_NODES, max_cost: int = DEFAULT_MAX_COST
) -> torch.nn.Sequential:
    """Construct a complete exact cost-and-route NN without consuming RNG state.

    Outputs have shape [batch, 1+n*n]. Reshape output[:,1:] to [batch,n,n]
    to read the route. Float32 exactness has the same configuration bound as
    ``make_weighted_shortest_paths``. Parameter count is reported as constructed;
    this extension does not impose an arbitrary dense-parameter target.
    """
    import torch
    from torch import nn

    distance_model = make_weighted_shortest_paths(nodes, max_cost, exact_budget=False)
    edges = nodes * nodes
    sentinel = (nodes - 1) * max_cost + 1
    distance_state = edges + 2 * nodes

    def linear(inputs: int, outputs: int) -> nn.Linear:
        layer = nn.Linear(inputs, outputs, device="meta", dtype=torch.float32)
        layer.to_empty(device="cpu")
        with torch.no_grad():
            layer.weight.zero_()
            layer.bias.zero_()
        return layer

    lifted: list[nn.Module] = []
    original_block = distance_model[0]
    assert isinstance(original_block, nn.Sequential)
    with torch.no_grad():
        # Reuse the proven distance circuit while carrying the source flags.
        for module in original_block:
            if isinstance(module, nn.Linear):
                layer = linear(module.in_features + nodes, module.out_features + nodes)
                layer.weight[: module.out_features, : module.in_features].copy_(
                    module.weight
                )
                layer.bias[: module.out_features].copy_(module.bias)
                for i in range(nodes):
                    layer.weight[module.out_features + i, module.in_features + i] = 1
                lifted.append(layer)
            else:
                lifted.append(nn.ReLU())
        distance_block = nn.Sequential(*lifted)

        missing = distance_state
        reach_a = missing + edges
        reach_b = reach_a + nodes
        cost_terms = reach_b + nodes
        features = linear(distance_state, cost_terms + nodes)
        for i in range(distance_state):
            features.weight[i, i] = 1
        for e in range(edges):
            features.weight[missing + e, e] = 1
            features.bias[missing + e] = -max_cost
        for i in range(nodes):
            # ReLU(M-d)-ReLU(M-1-d) is one iff d<M on this integer domain.
            features.weight[reach_a + i, edges + i] = -1
            features.bias[reach_a + i] = sentinel
            features.weight[reach_b + i, edges + i] = -1
            features.bias[reach_b + i] = sentinel - 1
            # ReLU(d + M*source_flag - M) selects d at the source.
            features.weight[cost_terms + i, edges + i] = 1
            features.weight[cost_terms + i, edges + nodes + i] = sentinel
            features.bias[cost_terms + i] = -sentinel

        policy_state = edges + nodes + 1
        tight = linear(cost_terms + nodes, policy_state)
        for i in range(nodes):
            for j in range(nodes):
                if i == j:
                    continue
                row, reversed_edge = i * nodes + j, j * nodes + i
                # A tight valid edge has c+d[j]-d[i]=0. The extra penalty
                # excludes absent edges into the target from unreachable nodes.
                tight.bias[row] = 1
                tight.weight[row, edges + i] = 1
                tight.weight[row, edges + j] = -1
                tight.weight[row, reversed_edge] = -1
                tight.weight[row, missing + reversed_edge] = -1
            tight.weight[edges + i, edges + nodes + i] = 1
            tight.weight[edges + i, reach_a + i] = 1
            tight.weight[edges + i, reach_b + i] = -1
            tight.bias[edges + i] = -1
            tight.weight[-1, cost_terms + i] = 1

        choose = linear(policy_state, policy_state)
        for i in range(nodes):
            for j in range(nodes):
                row = i * nodes + j
                choose.weight[row, row] = 1
                choose.weight[row, i * nodes : row] = -1
        for i in range(edges, policy_state):
            choose.weight[i, i] = 1

        history = edges + nodes
        route_state = history + edges + 1
        initialize = linear(policy_state, route_state)
        for i in range(edges + nodes):
            initialize.weight[i, i] = 1
        for i in range(nodes):
            initialize.weight[history + (nodes - 1) * nodes + i, edges + i] = 1
        initialize.weight[-1, -1] = 1

        transitions, kept_history = edges, 2 * edges
        gates = linear(route_state, 3 * edges + 1)
        for e in range(edges):
            gates.weight[e, e] = 1
            gates.weight[transitions + e, e] = 1
            gates.weight[transitions + e, edges + e // nodes] = 1
            gates.bias[transitions + e] = -1
            gates.weight[kept_history + e, history + e] = 1
        gates.weight[-1, -1] = 1
        advance = linear(3 * edges + 1, route_state)
        for e in range(edges):
            advance.weight[e, e] = 1
        for j in range(nodes):
            for i in range(nodes):
                transition = transitions + i * nodes + j
                advance.weight[edges + j, transition] = 1
                advance.weight[history + (nodes - 1) * nodes + j, transition] = 1
        for e in range(edges - nodes):
            advance.weight[history + e, kept_history + nodes + e] = 1
        advance.weight[-1, -1] = 1
        hop_block = nn.Sequential(gates, nn.ReLU(), advance, nn.ReLU())

        readout = linear(route_state, 1 + edges)
        readout.weight[0, -1] = 1
        for e in range(edges):
            readout.weight[1 + e, history + e] = 1

    return nn.Sequential(
        *([distance_block] * (nodes - 1)),
        features,
        nn.ReLU(),
        tight,
        nn.ReLU(),
        choose,
        nn.ReLU(),
        initialize,
        nn.ReLU(),
        *([hop_block] * (nodes - 1)),
        readout,
    ).eval()
