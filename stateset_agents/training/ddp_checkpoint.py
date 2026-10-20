"""Atomic, collective native DDP checkpoints on a shared filesystem.

One shared model/optimizer payload and one local shard per process-group rank
retain complete learner state. All DDP participants must enter the same save or
load call. Python payloads load through the weights-only checkpoint loader.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, cast

from .checkpoint_io import load_checkpoint_file
from .checkpoint_publication import (
    checkpoint_read,
    checkpoint_write,
    validate_checkpoint_publication,
)
from .checkpoint_state import _sync_restored_policy
from .ddp_checkpoint_state import (
    PROTOCOL,
    apply_payload,
    capture_payload,
    prepare_payload,
    static_identity,
)
from .ddp_control import DDPControlGroup, ddp_learner_poisoned
from .diagnostics import diagnostic_control, prepare_diagnostics
from .runtime_state import restore_rng
from .state_fingerprint import state_fingerprint

_METADATA = "native-ddp.json"


class _Group(DDPControlGroup):
    def __init__(self, trainer: Any) -> None:
        super().__init__(trainer, operation="checkpoint")

    def handoff(self, path: Path, staging: Path | None) -> tuple[Path, str]:
        """Broadcast a bounded random stage suffix and generation, not data."""

        def encode() -> bytes:
            if self.rank != 0:
                return b""
            assert staging is not None
            prefix = f".{path.name}.stateset-stage-"
            if not staging.name.startswith(prefix):
                raise ValueError("Invalid native checkpoint staging directory")
            result = json.dumps(
                {"suffix": staging.name[len(prefix) :], "generation": uuid.uuid4().hex}
            ).encode()
            if len(result) > 512:
                raise ValueError("Native checkpoint staging handoff exceeds bounds")
            return result

        encoded = self.run("staging-encode", encode)
        tensor = self.torch.zeros(513, dtype=self.torch.int64, device=self.device)
        if self.rank == 0:
            tensor[0] = len(encoded)
            tensor[1 : len(encoded) + 1] = self.torch.tensor(
                list(encoded), dtype=self.torch.int64, device=self.device
            )
        self.dist.broadcast(tensor, src=self.ranks[0], group=self.group)

        def decode() -> tuple[Path, str]:
            data = tensor.cpu().tolist()
            if not 0 < data[0] <= 512 or any(
                not 0 <= value <= 255 for value in data[1 : data[0] + 1]
            ):
                raise ValueError("Invalid native checkpoint stage control payload")
            decoded = json.loads(bytes(data[1 : data[0] + 1]))
            suffix, generation = decoded["suffix"], decoded["generation"]
            if (
                not isinstance(suffix, str)
                or not 0 < len(suffix) <= 128
                or not all(c.isascii() and (c.isalnum() or c in "_-") for c in suffix)
            ):
                raise ValueError("Invalid native checkpoint stage suffix")
            if (
                not isinstance(generation, str)
                or len(generation) != 32
                or any(c not in "0123456789abcdef" for c in generation)
            ):
                raise ValueError("Invalid native checkpoint generation")
            return path.parent / f".{path.name}.stateset-stage-{suffix}", generation

        return cast(tuple[Path, str], self.run("staging-handoff", decode))


def _path_identity(
    trainer: Any, path: Any, operation: str, group: _Group
) -> tuple[Path, Any]:
    target = Path(path).absolute()
    if not target.name or target.name in {".", ".."}:
        raise ValueError("Native checkpoint requires a named directory")
    return target, {
        "operation": operation,
        "path": str(target),
        "topology": group.ranks,
        "identity": static_identity(trainer),
    }


def _agreement(shared: dict[str, Any], local: dict[str, Any]) -> str:
    from types import SimpleNamespace

    diagnostic_proxy = SimpleNamespace(
        config=SimpleNamespace(**shared["identity"]["config"])
    )
    diagnostics = prepare_diagnostics(
        diagnostic_proxy, local["learning_state"]["diagnostics"]
    )
    diagnostic_proxy.diagnostics = diagnostics["monitor"]
    diagnostic_proxy._diagnostic_observations = diagnostics["observations"]
    accumulation = local["gradient_accumulation"]
    normalization = accumulation.get("normalization")
    return state_fingerprint(
        {
            "shared": shared,
            "global_step": local["counters"]["global_step"],
            "grad_accum_step": local["counters"]["_grad_accum_step"],
            "baseline": local["learning_state"]["reward_baseline"],
            "diagnostics": diagnostic_control(diagnostic_proxy),
            "scaler": accumulation["scaler"],
            "normalization": (
                None
                if normalization is None
                else {
                    name: normalization[name]
                    for name in ("measure", "policy_normalizer", "entropy_normalizer")
                }
            ),
        }
    )


def save_ddp_checkpoint(trainer: Any, checkpoint_path: Any) -> None:
    """Collectively publish shared and local state; preserve prior generations."""
    group = _Group(trainer)
    resolved = group.run(
        "start", lambda: _path_identity(trainer, checkpoint_path, "save", group)
    )
    path, identity = resolved
    group.run("save-admission", lambda: identity, agree=True)

    def capture() -> Any:
        if ddp_learner_poisoned(trainer):
            raise ValueError(
                "A failed native restore must succeed before checkpoint save"
            )
        return capture_payload(trainer)

    shared, local, agreement = group.run("capture", capture)
    group.run("shared-state", lambda: agreement, agree=True)
    context, staging, active = None, None, False
    try:

        def enter() -> None:
            nonlocal context, staging, active
            if group.rank == 0:
                context = checkpoint_write(path)
                staging = context.__enter__()
                active = True

        group.run("save-open", enter)
        staging, generation = group.handoff(path, staging)
        group.run(
            "rank-write",
            lambda: group.torch.save(
                {
                    "protocol": PROTOCOL,
                    "generation": generation,
                    "rank": group.rank,
                    "state": local,
                },
                staging / f"rank-{group.rank:05d}.pt",
            ),
        )

        def write_shared() -> None:
            if group.rank == 0:
                group.torch.save(
                    {
                        "protocol": PROTOCOL,
                        "generation": generation,
                        "topology": group.ranks,
                        "state": shared,
                    },
                    staging / "training_state.pt",
                )
                metadata = {
                    "protocol": PROTOCOL,
                    "generation": generation,
                    "topology": group.ranks,
                    "shared_sha256": state_fingerprint(shared),
                    "agreement": agreement,
                }
                (staging / _METADATA).write_text(
                    json.dumps(metadata, sort_keys=True, allow_nan=False) + "\n"
                )

        group.run("shared-write", write_shared)

        def publish() -> None:
            nonlocal active
            if group.rank == 0:
                active = False
                assert context is not None
                context.__exit__(None, None, None)

        group.run("publish", publish)
    finally:
        if group.rank == 0 and active:
            assert context is not None
            error = ValueError("Native checkpoint transaction aborted")
            context.__exit__(ValueError, error, None)


def load_ddp_checkpoint(
    trainer: Any, checkpoint_path: Any, *, trusted: bool = False
) -> bool:
    """Collectively validate then restore a same-topology native DDP checkpoint.

    Validation errors preserve live state. Unexpected apply errors poison every
    peer against further learning/saving until a successful coordinated reload.
    Process/network failure recovery requires a new process group.
    """
    group = _Group(trainer)
    trainer._ddp_replica_consensus_pending = True
    path, identity = group.run(
        "start", lambda: _path_identity(trainer, checkpoint_path, "load", group)
    )
    group.run(
        "load-admission", lambda: {"identity": identity, "trusted": trusted}, agree=True
    )
    context, active = None, False
    try:

        def enter() -> None:
            nonlocal context, active
            if group.rank == 0:
                context = checkpoint_read(path)
                context.__enter__()
                active = True
                validate_checkpoint_publication(path, required=True)

        group.run("load-open", enter)

        def read() -> tuple[Any, Any, Any]:
            metadata = json.loads((path / _METADATA).read_text())
            if (
                not isinstance(metadata, dict)
                or set(metadata)
                != {"protocol", "generation", "topology", "shared_sha256", "agreement"}
                or metadata["protocol"] != PROTOCOL
                or metadata["topology"] != group.ranks
            ):
                raise ValueError("Native checkpoint protocol/topology differs")
            if type(metadata["topology"]) is not list or any(
                type(rank) is not int for rank in metadata["topology"]
            ):
                raise ValueError("Invalid native checkpoint topology")
            for name, size in (
                ("generation", 32),
                ("shared_sha256", 64),
                ("agreement", 64),
            ):
                value = metadata[name]
                if (
                    not isinstance(value, str)
                    or len(value) != size
                    or any(character not in "0123456789abcdef" for character in value)
                ):
                    raise ValueError("Invalid native checkpoint generation/digest")
            common = load_checkpoint_file(
                path / "training_state.pt",
                trusted=trusted,
                torch_module=group.torch,
                mmap=True,
            )
            packet = load_checkpoint_file(
                path / f"rank-{group.rank:05d}.pt",
                trusted=trusted,
                torch_module=group.torch,
                mmap=True,
            )
            if (
                not isinstance(common, dict)
                or set(common) != {"protocol", "generation", "topology", "state"}
                or common["protocol"] != PROTOCOL
                or common["topology"] != group.ranks
                or common["generation"] != metadata["generation"]
            ):
                raise ValueError("Invalid native shared checkpoint generation")
            if (
                not isinstance(packet, dict)
                or set(packet) != {"protocol", "generation", "rank", "state"}
                or packet["protocol"] != PROTOCOL
                or packet["generation"] != metadata["generation"]
                or type(packet["rank"]) is not int
                or packet["rank"] != group.rank
            ):
                raise ValueError("Invalid native rank checkpoint generation")
            shared, local = common["state"], packet["state"]
            if (
                state_fingerprint(shared) != metadata["shared_sha256"]
                or _agreement(shared, local) != metadata["agreement"]
            ):
                raise ValueError("Native checkpoint content/control identity differs")
            return shared, local, metadata

        shared, local, metadata = group.run("read", read)
        group.run("generation", lambda: metadata, agree=True)
        prepared = group.run(
            "validate", lambda: prepare_payload(trainer, shared, local)
        )
        trainer._ddp_restore_failed = True
        group.run("apply", lambda: apply_payload(trainer, prepared))

        # Re-capture the restored shared state/control to bind actual live values
        # to this exact saved generation before admission can clear its poison.
        def restored() -> str:
            restored_shared, restored_local, digest = capture_payload(trainer)
            if digest != metadata["agreement"]:
                raise ValueError(
                    "Applied native checkpoint values differ from saved generation"
                )
            return digest

        group.run("restored-values", restored, agree=True)
        group.run("rollout-sync", lambda: _sync_restored_policy(trainer))
        # Backend setup/publication can consume global RNG after apply_payload
        # restores it. Restore each rank's validated stream last, coordinating
        # failures without requiring different rank-local streams to agree.
        group.run(
            "rng-restore", lambda: restore_rng(prepared["local"]["rng"], group.torch)
        )
        trainer._ddp_restore_failed = False
        trainer._ddp_replica_consensus_pending = False
        return True
    finally:
        if group.rank == 0 and active:
            assert context is not None
            context.__exit__(None, None, None)
