"""On-demand native DDP parameter/optimizer agreement using bounded hashes."""

from __future__ import annotations

from typing import Any

from .state_fingerprint import state_fingerprint

REPLICA_CONSENSUS_PROTOCOL = "stateset-native-replica-consensus-v1"
_COMPONENTS = ("policy", "optimizer", "reference")


def verify_ddp_replica_state(trainer: Any) -> dict[str, Any]:
    """Reject divergent replica values without broadcasting or changing them.

    Called at first learner admission and after checkpoint-load attempts. The
    digest covers parameters and optimizer slots; buffers and pending gradients
    are rank-local and deliberately excluded. Only validity bits and digests
    cross the wire. All participating peers must call on the same process group.
    """
    import torch
    import torch.distributed as dist

    from .learner_preflight import _reference

    model = trainer.agent.model
    reference, _ = _reference(trainer)
    factories = {
        "policy": lambda: dict(model.module.named_parameters()),
        "optimizer": lambda: (
            None if trainer.optimizer is None else trainer.optimizer.state_dict()
        ),
        "reference": lambda: (
            None if reference is None else dict(reference.named_parameters())
        ),
    }
    hashes: list[bytes] = []
    mask = 0
    error: Exception | None = None
    for index, name in enumerate(_COMPONENTS):
        try:
            hashes.append(bytes.fromhex(state_fingerprint(factories[name]())))
            mask |= 1 << index
        except (
            ValueError,
            TypeError,
            LookupError,
            AttributeError,
            OverflowError,
            RuntimeError,
            OSError,
        ) as exc:
            hashes.append(bytes(32))
            error = exc if error is None else error
    header = torch.tensor(
        [1, mask, *b"".join(hashes)],
        dtype=torch.int64,
        device=next(model.parameters()).device,
    )
    peers = [
        torch.empty_like(header)
        for _ in range(dist.get_world_size(model.process_group))
    ]
    dist.all_gather(peers, header, group=model.process_group)
    decoded = [peer.cpu().tolist() for peer in peers]
    if any(row[0] != 1 for row in decoded):
        raise ValueError("Native DDP replica consensus protocol differs")
    invalid = {
        rank: [
            name for index, name in enumerate(_COMPONENTS) if not row[1] & (1 << index)
        ]
        for rank, row in enumerate(decoded)
        if row[1] != 7
    }
    if invalid:
        raise ValueError(f"Invalid native DDP replica state: {invalid}") from error
    differences = [
        name
        for index, name in enumerate(_COMPONENTS)
        if len({tuple(row[2 + index * 32 : 2 + (index + 1) * 32]) for row in decoded})
        != 1
    ]
    if differences:
        raise ValueError(
            f"Native DDP replica values differ across ranks: {differences}"
        )
    return {
        "protocol": REPLICA_CONSENSUS_PROTOCOL,
        "components": {
            name: digest.hex() for name, digest in zip(_COMPONENTS, hashes, strict=True)
        },
    }
