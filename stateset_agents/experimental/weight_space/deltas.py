"""Low-rank weight deltas: the unit of a weight-space edit.

Every edit the harness applies is ``ΔW = B @ A`` for a ``Linear`` weight of
shape ``(out, in)``. That is exactly a LoRA adapter's parameterisation, so a
kept edit ships through the same adapter/merge/serve path as a fine-tune.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = ["LowRankDelta", "direction_ablation"]


@dataclass(frozen=True)
class LowRankDelta:
    """``ΔW = B @ A`` with ``B`` of shape ``(out, r)`` and ``A`` of ``(r, in)``."""

    B: torch.Tensor
    A: torch.Tensor

    def __post_init__(self) -> None:
        if self.B.dim() != 2 or self.A.dim() != 2 or self.B.shape[1] != self.A.shape[0]:
            raise ValueError(
                f"B {tuple(self.B.shape)} and A {tuple(self.A.shape)} do not form "
                "a low-rank product (need (out, r) @ (r, in))"
            )

    @property
    def rank(self) -> int:
        return int(self.A.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.B.shape[0]), int(self.A.shape[1])

    def dense(self) -> torch.Tensor:
        """Materialise ``ΔW`` (float32)."""
        import torch

        with torch.autocast(device_type=self.B.device.type, enabled=False):
            return self.B.float() @ self.A.float()


def direction_ablation(
    weight: torch.Tensor, direction: torch.Tensor, k: int, alpha: float = 1.0
) -> LowRankDelta:
    """Remove ``direction`` from the ``k`` input columns that write most along it.

    For a ``Linear`` weight ``W`` (``out × in``), column ``j`` is the vector that
    input unit ``j`` writes into the output space. With ``u`` the unit-normalised
    direction and ``S`` the ``k`` columns with the largest ``|uᵀ W[:, j]|``::

        ΔW[:, S] = -alpha · u (uᵀ W[:, S])        (zero elsewhere)

    ``alpha=1`` projects ``u`` out of those columns entirely. This is the edit
    shape behind "subtracted that direction from 64 output columns" in the
    weight-space programming write-up: rank 1, ``k · out`` weights changed.
    """
    import torch

    if direction.dim() != 1 or direction.shape[0] != weight.shape[0]:
        raise ValueError(
            f"direction length {tuple(direction.shape)} does not match the "
            f"weight's output dimension {weight.shape[0]}"
        )
    if not 0 < k <= weight.shape[1]:
        raise ValueError(f"k={k} must be in [1, {weight.shape[1]}] (input columns)")
    norm = direction.float().norm()
    if norm == 0:
        raise ValueError("direction is the zero vector")

    w = weight.detach().float()
    # Build the edit on the weight's device: a CPU direction against an
    # accelerator-resident weight must not fail or split across devices.
    u = (direction.float() / norm).to(w.device)
    proj = u @ w  # (in,) how much each unit writes along u
    cols = proj.abs().topk(k).indices
    A = torch.zeros(1, w.shape[1], device=w.device)
    A[0, cols] = alpha * proj[cols]
    return LowRankDelta(B=-u[:, None], A=A)
