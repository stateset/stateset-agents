"""A handwritten recurrent ReLU network for unweighted directed shortest paths.

The standard Sequential forward pass uses only Linear and ReLU modules.
A shared two-layer block propagates reachability by one edge and counts
previously reachable rounds. After n-1 rounds, a linear readout recovers
the shortest distance to each vertex; n means unreachable.

For 12 vertices this stores 114,984 dense parameters, of which 1,308 are
nonzero. The same block is used eleven times: these are unique parameter
counts, not the number of scalar uses in the expanded computation.
The sparse construction encodes an algorithm, not learned knowledge.

State is [A, p, q, c], where A is row-major adjacency, r=p-q is binary
reachability and c counts previous rounds. One block computes
z_ij=ReLU(A_ij+r_i-1), t_j=r_j+sum_i(z_ij), then writes
[A, t, ReLU(t-1), c+r]. Thus r'=min(t,1). Starting at r=one_hot(source),
c=0, the readout n-c-r gives distance d for reachable nodes and n otherwise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from stateset_agents.core.errors import ConfigurationError

if TYPE_CHECKING:
    import torch

DEFAULT_NODES = 12


def make_shortest_paths(
    nodes: int = DEFAULT_NODES, *, handwritten: bool = True, seed: int = 0
) -> torch.nn.Sequential:
    """Build a standard shared-weight Sequential graph-distance network on CPU.

    Use ``encode_graphs`` to form valid initial states. All returned parameters
    are trainable, including analytically zero coefficients. Random baselines
    have the identical recurrence and use Kaiming initialization for the ReLU
    block, Xavier initialization for the readout, and zero biases.
    The handwritten path consumes no random numbers; the random path preserves
    the caller's CPU RNG and does not seed accelerator RNGs.
    """
    import torch
    from torch import nn

    if isinstance(nodes, bool) or not isinstance(nodes, int) or nodes < 2:
        raise ConfigurationError("nodes must be an integer of at least two")
    edges = nodes * nodes
    state, hidden = edges + 3 * nodes, 2 * edges + 2 * nodes

    def layers(device: str) -> tuple[nn.Sequential, nn.Linear]:
        block = nn.Sequential(
            nn.Linear(state, hidden, device=device),
            nn.ReLU(),
            nn.Linear(hidden, state, device=device),
            nn.ReLU(),
        )
        return block, nn.Linear(state, nodes, device=device)

    if not handwritten:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            block, readout = layers("cpu")
            for index in (0, 2):
                linear = cast(nn.Linear, block[index])
                nn.init.kaiming_uniform_(linear.weight, nonlinearity="relu")
                nn.init.zeros_(linear.bias)
            nn.init.xavier_uniform_(readout.weight)
            nn.init.zeros_(readout.bias)
    else:
        block, readout = layers("meta")
        block.to_empty(device="cpu")
        readout.to_empty(device="cpu")
        first, second = cast(nn.Linear, block[0]), cast(nn.Linear, block[2])
        with torch.no_grad():
            for param in (*block.parameters(), *readout.parameters()):
                param.zero_()
            for i in range(nodes):
                for j in range(nodes):
                    edge = i * nodes + j
                    first.weight[edge, edge] = 1
                    second.weight[edge, edge] = 1
                    first.weight[edges + edge, edge] = 1
                    first.weight[edges + edge, edges + i] = 1
                    first.weight[edges + edge, edges + nodes + i] = -1
                    first.bias[edges + edge] = -1
                    second.weight[edges + j, edges + edge] = 1
                    second.weight[edges + nodes + j, edges + edge] = 1
                first.weight[2 * edges + i, edges + i] = 1
                first.weight[2 * edges + i, edges + nodes + i] = -1
                first.weight[2 * edges + nodes + i, edges + 2 * nodes + i] = 1
                first.weight[2 * edges + nodes + i, edges + i] = 1
                first.weight[2 * edges + nodes + i, edges + nodes + i] = -1
                second.weight[edges + i, 2 * edges + i] = 1
                second.weight[edges + nodes + i, 2 * edges + i] = 1
                second.bias[edges + nodes + i] = -1
                second.weight[edges + 2 * nodes + i, 2 * edges + nodes + i] = 1
                readout.weight[i, edges + i] = -1
                readout.weight[i, edges + nodes + i] = 1
                readout.weight[i, edges + 2 * nodes + i] = -1
                readout.bias[i] = nodes
    # Reusing the same object ties parameters while Sequential invokes it n-1 times.
    return nn.Sequential(*([block] * (nodes - 1)), readout).eval()


def encode_graphs(adjacency: torch.Tensor, sources: torch.Tensor) -> torch.Tensor:
    """Pack binary [batch,n,n] adjacency and integer [batch] sources as initial states.

    A[i,j]=1 means a directed edge i -> j. Source distance is zero, even in
    an otherwise empty graph. Input validation is outside the neural forward.
    """
    import torch

    if adjacency.ndim != 3 or adjacency.shape[1] != adjacency.shape[2]:
        raise ConfigurationError("adjacency must have shape [batch,n,n]")
    batch, nodes, _ = adjacency.shape
    if nodes < 2 or sources.shape != (batch,):
        raise ConfigurationError("need n >= 2 and one source per graph")
    if (
        sources.dtype not in (torch.int32, torch.int64)
        or sources.device != adjacency.device
    ):
        raise ConfigurationError("sources must be integers on the adjacency device")
    if bool(((sources < 0) | (sources >= nodes)).any()):
        raise ConfigurationError("source index is outside the graph")
    if bool(((adjacency != 0) & (adjacency != 1)).any()):
        raise ConfigurationError("adjacency must contain only binary entries")
    flat = adjacency.reshape(batch, nodes * nodes).float()
    reached = torch.nn.functional.one_hot(sources.long(), nodes).float()
    zeros = torch.zeros(batch, 2 * nodes, device=adjacency.device)
    return torch.cat((flat, reached, zeros), dim=1)
