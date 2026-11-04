"""Analytically set a two-million-parameter network for weighted shortest paths.

The default solves seventeen-node directed graphs with integer edge costs 1..7.
Zero in the input adjacency means no edge; output 113 means unreachable.
All computation is ordinary Linear/ReLU inference: sixteen shared Bellman-Ford
rounds, each using a tree of exact pairwise minima, followed by a linear readout.

The natural circuit has 1,865,053 dense parameters. To meet the exact requested
2,000,000 budget, 137 active hidden neurons are split into identical pairs, with
their outgoing weights halved. This preserves the function and adds 134,947
stored coefficients, including zeros. It is representation redundancy, not
additional algorithmic capability; there are no unused padding tensors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from stateset_agents.core.errors import ConfigurationError

if TYPE_CHECKING:
    import torch

DEFAULT_NODES = 17
DEFAULT_MAX_COST = 7


def _validate_config(nodes: int, max_cost: int, exact_budget: bool) -> None:
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (nodes, max_cost)):
        raise ConfigurationError("nodes and max_cost must be integers")
    if nodes < 2 or max_cost < 1:
        raise ConfigurationError("need at least two nodes and positive max_cost")
    # Reserve exact half-integers for the split neurons as well as integer sums.
    if 4 * ((nodes - 1) * max_cost + 1) > 2**24:
        raise ConfigurationError("cost bound exceeds exact float32 integer arithmetic")
    if exact_budget and nodes != DEFAULT_NODES:
        raise ConfigurationError("the exact 2,000,000 budget requires seventeen nodes")


def make_weighted_shortest_paths(
    nodes: int = DEFAULT_NODES,
    max_cost: int = DEFAULT_MAX_COST,
    *,
    exact_budget: bool = True,
) -> torch.nn.Sequential:
    """Construct all weights directly, without data, randomness, or optimization.

    Use ``encode_weighted_graphs`` with the same configuration. Set exact_budget
    to False for the natural circuit, including when constructing other sizes.
    All coefficients are trainable; this function performs no fitting.
    """
    import torch
    from torch import nn

    _validate_config(nodes, max_cost, exact_budget)
    edges, state = nodes * nodes, nodes * nodes + nodes

    def linear(inputs: int, outputs: int) -> nn.Linear:
        layer = nn.Linear(inputs, outputs, device="meta", dtype=torch.float32)
        layer.to_empty(device="cpu")
        with torch.no_grad():
            layer.weight.zero_()
            layer.bias.zero_()
        return layer

    count = nodes + 1
    width = edges + nodes * count
    expand = linear(state, width)
    modules: list[nn.Module] = [expand, nn.ReLU()]
    with torch.no_grad():
        for e in range(edges):
            expand.weight[e, e] = 1
        for j in range(nodes):
            base = edges + j * count
            expand.weight[base, edges + j] = 1
            for i in range(nodes):
                expand.weight[base + 1 + i, edges + i] = 1
                expand.weight[base + 1 + i, i * nodes + j] = 1

        stage = 0
        while count > 1:
            next_count = (count + 1) // 2
            out_width = edges + nodes * next_count
            extra = (106, 27, 4, 0, 0)[stage] if exact_budget else 0
            compare = linear(width, width + extra)
            reduce = linear(width + extra, out_width)
            for e in range(edges):
                compare.weight[e, e] = 1
                reduce.weight[e, e] = 1
            for j in range(nodes):
                base = edges + j * count
                out_base = edges + j * next_count
                for k in range(count // 2):
                    a, b = base + 2 * k, base + 2 * k + 1
                    # First ReLU retains a and max(a-b, 0).
                    compare.weight[a, a] = 1
                    compare.weight[b, a] = 1
                    compare.weight[b, b] = -1
                    # Second affine computes a - ReLU(a-b) = min(a,b).
                    reduce.weight[out_base + k, a] = 1
                    reduce.weight[out_base + k, b] = -1
                if count % 2:
                    compare.weight[base + count - 1, base + count - 1] = 1
                    reduce.weight[out_base + next_count - 1, base + count - 1] = 1
            # Split active comparison neurons, not unused channels. Both copies
            # participate in the minimum computation through half outgoing weights.
            for k in range(extra):
                original, duplicate = edges + k, width + k
                compare.weight[duplicate].copy_(compare.weight[original])
                compare.bias[duplicate].copy_(compare.bias[original])
                reduce.weight[:, original].mul_(0.5)
                reduce.weight[:, duplicate].copy_(reduce.weight[:, original])
            modules.extend((compare, nn.ReLU(), reduce, nn.ReLU()))
            width, count = out_width, next_count
            stage += 1
        readout = linear(state, nodes)
        for j in range(nodes):
            readout.weight[j, edges + j] = 1
    block = nn.Sequential(*modules)
    return nn.Sequential(*([block] * (nodes - 1)), readout).eval()


def encode_weighted_graphs(
    costs: torch.Tensor, sources: torch.Tensor, max_cost: int = DEFAULT_MAX_COST
) -> torch.Tensor:
    """Pack [batch,n,n] integer costs and [batch] source indices for NN inference.

    Zero means no edge; positive entries must be in 1..max_cost. Diagonal costs
    must be zero. The encoder only replaces absent edges by a sentinel and sets
    the initial distances; it performs no path search or shortest-path updates.
    """
    import torch

    if costs.ndim != 3 or costs.shape[1] != costs.shape[2]:
        raise ConfigurationError("costs must have shape [batch,n,n]")
    batch, nodes, _ = costs.shape
    _validate_config(nodes, max_cost, False)
    if sources.shape != (batch,) or sources.dtype not in (torch.int32, torch.int64):
        raise ConfigurationError("need one integer source index per graph")
    if sources.device != costs.device:
        raise ConfigurationError("costs and sources must use the same device")
    if bool(((sources < 0) | (sources >= nodes)).any()):
        raise ConfigurationError("source index is outside the graph")
    if costs.is_complex() or not bool(torch.isfinite(costs).all()):
        raise ConfigurationError("costs must be finite real integers")
    costs = costs.float()
    if bool(((costs < 0) | (costs > max_cost) | (costs != costs.round())).any()):
        raise ConfigurationError("costs must be integers between zero and max_cost")
    if bool(costs.diagonal(dim1=1, dim2=2).any()):
        raise ConfigurationError("diagonal costs must be zero")
    sentinel = (nodes - 1) * max_cost + 1
    edges = torch.where(costs > 0, costs, sentinel).float()
    edges[:, torch.arange(nodes), torch.arange(nodes)] = 0
    initial = torch.full(
        (batch, nodes), float(sentinel), device=costs.device, dtype=torch.float32
    )
    initial.scatter_(1, sources.long()[:, None], 0)
    return torch.cat((edges.reshape(batch, nodes * nodes), initial), dim=1)
