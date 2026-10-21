"""Bounded status phases for operations entered by every native DDP worker."""

from __future__ import annotations

import hashlib
from typing import Any

from .state_fingerprint import state_fingerprint


def ddp_learner_poisoned(trainer: Any) -> bool:
    """Require full learner reload after a partial learner or memory commit."""
    return bool(
        getattr(trainer, "_ddp_restore_failed", False)
        or getattr(
            getattr(trainer, "continual_manager", None), "_ddp_restore_failed", False
        )
    )


class DDPControlGroup:
    """Coordinate local failures and identities without pickled wire objects.

    All participants must enter the same phase. This coordinates application
    errors among live peers; it cannot recover a lost process or process group.
    """

    def __init__(self, trainer: Any, *, operation: str) -> None:
        import torch
        import torch.distributed as dist

        self.torch, self.dist = torch, dist
        self.group = trainer.agent.model.process_group
        self.rank = dist.get_rank(self.group)
        self.world = dist.get_world_size(self.group)
        self.ranks = [
            dist.get_global_rank(self.group, index) for index in range(self.world)
        ]
        self.device = next(trainer.agent.model.parameters()).device
        self.operation = operation

    def _finish(
        self, phase: str, value: Any, error: BaseException | None, *, agree: bool
    ) -> Any:
        digest = bytes(32)
        if error is None and agree:
            try:
                digest = bytes.fromhex(state_fingerprint(value))
            except BaseException as exc:
                error = exc
        phase_digest = hashlib.sha256(f"{self.operation}:{phase}".encode()).digest()
        header = self.torch.tensor(
            [1, int(error is None), *phase_digest, *digest],
            dtype=self.torch.int64,
            device=self.device,
        )
        gathered = [self.torch.empty_like(header) for _ in range(self.world)]
        self.dist.all_gather(gathered, header, group=self.group)
        rows = [tensor.cpu().tolist() for tensor in gathered]
        if any(row[0] != 1 or row[2:34] != list(phase_digest) for row in rows):
            raise ValueError(
                "Native DDP operation/phase differs across ranks"
            ) from error
        failed = [rank for rank, row in enumerate(rows) if row[1] != 1]
        if failed:
            if error is not None and not isinstance(error, Exception):
                # Preserve local cancellation/interrupt semantics after notifying
                # every peer. Peers without that exception reject coherently.
                raise error
            raise ValueError(
                f"Native DDP {self.operation} {phase} failed on process-group ranks {failed}"
            ) from error
        if agree and len({tuple(row[34:]) for row in rows}) != 1:
            raise ValueError(
                f"Native DDP {self.operation} {phase} identity differs across ranks"
            )
        return value

    def run(self, phase: str, function: Any, *, agree: bool = False) -> Any:
        """Run local synchronous work, then share status and optional identity."""
        value, error = None, None
        try:
            value = function()
        except BaseException as exc:
            error = exc
        return self._finish(phase, value, error, agree=agree)

    async def arun(self, phase: str, function: Any, *, agree: bool = False) -> Any:
        """Await local work and notify peers even on local task cancellation."""
        value, error = None, None
        try:
            value = await function()
        except BaseException as exc:
            error = exc
        return self._finish(phase, value, error, agree=agree)
