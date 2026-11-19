"""Exact pooled-loss and multi-step residual certificates for a finite logged table."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stateset_agents.training.offline_rl_batches import CriticBatchSnapshot

Scalar = int | float | Fraction


@dataclass(frozen=True)
class CriticRow:
    """One observation with its published target. A terminal row has no successor."""

    state: int
    successor: int | None
    reward: Scalar
    target: Scalar


@dataclass(frozen=True)
class BlockCertificate:
    """Discounted survival and exact Bellman residual after a block of decisions."""

    horizon: int
    contraction_factor: Fraction
    residual: Fraction

    def certifies(self, tolerance: Scalar) -> bool:
        """Test the block residual bound using exact rational arithmetic."""
        return self.contraction_factor < 1 and self.residual <= (
            1 - self.contraction_factor
        ) * Fraction(tolerance)


@dataclass(frozen=True)
class CriticCertificate:
    """Exact inputs to the pooled critic theorem for the supplied empirical table."""

    loss: Fraction
    within_state_variance: Fraction
    minimum_frequency: Fraction
    mean_target_bias_bound: Fraction
    target_lag: Fraction
    contraction_factor: Fraction
    block: BlockCertificate | None = None

    def certifies(self, tolerance: Scalar) -> bool:
        """Check the pooled bound and any measured multi-step residual bound."""
        epsilon = Fraction(tolerance)
        if epsilon < 0:
            return False
        if self.block is not None and self.block.certifies(epsilon):
            return True
        if self.contraction_factor >= 1:
            return False
        margin = (
            (1 - self.contraction_factor) * epsilon
            - self.mean_target_bias_bound
            - self.contraction_factor * self.target_lag
        )
        return margin >= 0 and (
            self.loss - self.within_state_variance <= self.minimum_frequency * margin**2
        )


def audit_logged_critic(
    fitted: Sequence[Scalar],
    frozen: Sequence[Scalar],
    rows: Iterable[CriticRow],
    discount: Scalar,
    *,
    horizon: int | None = None,
) -> CriticCertificate:
    """Audit uniformly weighted rows against two tables indexed by the same state IDs.

    Supply the actual published target for each row and the tables used at that
    update. Python floats are read as their exact binary values. All reductions
    use rationals. Every state in the declared table must occur as a source.
    At unit one-step survival, find the shortest horizon at which every state
    can reach termination. A closed class retains its one-step witness.
    An explicit horizon measures that particular block.
    """
    n = len(fitted)
    if n == 0 or len(frozen) != n:
        raise ValueError("Expected nonempty fitted and frozen tables of equal size")
    if horizon is not None and (type(horizon) is not int or horizon < 1):
        raise ValueError("The audit horizon must be a positive integer")
    gamma = Fraction(discount)
    if not 0 <= gamma <= 1:
        raise ValueError("Discount must be between zero and one")
    u = tuple(Fraction(value) for value in fitted)
    v = tuple(Fraction(value) for value in frozen)
    visits = [0] * n
    live = [0] * n
    edges: list[Counter[int]] = [Counter() for _ in range(n)]
    reward_sums = [Fraction()] * n
    residuals = [Fraction()] * n
    target_errors = [Fraction()] * n
    squared = Fraction()
    for row in rows:
        s = row.state
        if type(s) is not int or not 0 <= s < n:
            raise ValueError("Source state is outside the declared table")
        backup = Fraction(row.reward)
        reward_sums[s] += backup
        if row.successor is not None:
            j = row.successor
            if type(j) is not int or not 0 <= j < n:
                raise ValueError("Successor state is outside the declared table")
            backup += gamma * v[j]
            live[s] += 1
            edges[s][j] += 1
        target = Fraction(row.target)
        residual = target - u[s]
        visits[s] += 1
        residuals[s] += residual
        squared += residual**2
        target_errors[s] += target - backup
    if min(visits) == 0:
        raise ValueError("Every declared state needs a logged source row")
    total = sum(visits)
    # Subtract the squared state means from the pooled second moment.
    between = sum(r**2 / c for r, c in zip(residuals, visits, strict=True))
    rate = gamma * max(Fraction(a, c) for a, c in zip(live, visits, strict=True))
    block = None
    if horizon is not None or rate == 1:
        if horizon is None:
            # A positive chance of stopping is enough. Reverse breadth-first
            # distances give the first common horizon, or reveal a closed class.
            parents: list[list[int]] = [[] for _ in range(n)]
            for s, successors in enumerate(edges):
                for j in successors:
                    parents[j].append(s)
            distances = [
                int(continuing < total_visits)
                for continuing, total_visits in zip(live, visits, strict=True)
            ]
            queue = deque(s for s in range(n) if distances[s])
            while queue:
                j = queue.popleft()
                for s in parents[j]:
                    if not distances[s]:
                        distances[s] = distances[j] + 1
                        queue.append(s)
            horizon = max(distances) if all(distances) else 1

        def carry(table: Sequence[Fraction], s: int) -> Fraction:
            return (
                gamma
                * sum((count * table[j] for j, count in edges[s].items()), Fraction())
                / visits[s]
            )

        means = [r / c for r, c in zip(reward_sums, visits, strict=True)]
        values = list(u)
        survival = [Fraction(1)] * n
        for _ in range(horizon):
            values = [means[s] + carry(values, s) for s in range(n)]
            survival = [carry(survival, s) for s in range(n)]
        block = BlockCertificate(
            horizon,
            max(survival),
            max(abs(x - y) for x, y in zip(values, u, strict=True)),
        )
    return CriticCertificate(
        loss=squared / total,
        within_state_variance=(squared - between) / total,
        minimum_frequency=Fraction(min(visits), total),
        mean_target_bias_bound=max(
            abs(e) / c for e, c in zip(target_errors, visits, strict=True)
        ),
        target_lag=max(abs(a - b) for a, b in zip(u, v, strict=True)),
        contraction_factor=rate,
        block=block,
    )


def audit_critic_snapshot(
    snapshot: CriticBatchSnapshot,
    states: Sequence[tuple[float, ...]],
    *,
    horizon: int | None = None,
) -> CriticCertificate:
    """Certify one captured update over an explicitly declared set of state features."""
    ids = {s: i for i, s in enumerate(states)}
    if not ids or len(ids) != len(states):
        raise ValueError("Declare a nonempty table of distinct state features")
    values: dict[tuple[float, ...], tuple[float, float]] = {}
    rows = []
    for row in snapshot.rows:
        if row.state not in ids or (
            row.successor is not None and row.successor not in ids
        ):
            raise ValueError("Captured transition is outside the declared state table")
        pair = (row.prediction, row.frozen_prediction)
        if values.setdefault(row.state, pair) != pair:
            raise ValueError(
                "Repeated state has inconsistent live or frozen predictions"
            )
        successor = None if row.successor is None else ids[row.successor]
        rows.append(CriticRow(ids[row.state], successor, row.reward, row.target))
    if len(values) != len(ids):
        raise ValueError("Every declared state needs a logged source row")
    return audit_logged_critic(
        [values[s][0] for s in states],
        [values[s][1] for s in states],
        rows,
        snapshot.discount,
        horizon=horizon,
    )
