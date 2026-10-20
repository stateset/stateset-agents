"""Source-bound SFT identities and rank agreement before weight loading.

Identity construction is Torch-free. The collective uses only fixed-size
integer tensors and hashes, and requires an already initialized process group.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

SFT_CONSENSUS_PROTOCOL = "stateset-sft-consensus-v1"
_COMPONENTS = (
    "model",
    "revision",
    "training_config",
    "dataset",
    "tokenizer",
    "encoded_data",
    "runtime",
    "implementation",
)


def _digest(value: Any) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def fingerprint_encoded_rows(rows: Iterable[Mapping[str, Any]]) -> str:
    """Hash ordered input/label rows with explicit JSONL record boundaries."""
    digest = hashlib.sha256()
    for row in rows:
        encoded = json.dumps(
            dict(row),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


def sft_preflight_identity(
    *,
    model_id: str,
    revision: str,
    training_config: Mapping[str, Any],
    dataset: Mapping[str, Any],
    tokenizer: Any,
    encoded_data: Mapping[str, Any],
    runtime_versions: Mapping[str, str],
    implementation_modules: Iterable[str],
) -> dict[str, Any]:
    """Bind native tokenizer serialization, inputs, settings and implementation.

    Filesystem locations are deliberately absent: nodes may stage identical
    files at different paths. Callers exclude source/output paths from config.
    The backend serialization includes vocabulary, merges and postprocessing;
    an identical chat template alone does not establish identical token IDs.
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    serialized = backend.to_str() if backend is not None else None
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(serialized, str) or not isinstance(template, str):
        raise ValueError(
            "SFT rank identity requires a serialized native fast tokenizer and chat template"
        )
    implementation = {}
    for name in sorted(set(implementation_modules) | {__name__}):
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise ValueError(f"Cannot bind SFT implementation source for {name}")
        implementation[name] = hashlib.sha256(
            Path(spec.origin).read_bytes()
        ).hexdigest()
    return {
        "model": model_id,
        "revision": revision,
        "training_config": dict(training_config),
        "dataset": dict(dataset),
        "tokenizer": {
            "backend_sha256": _digest(json.loads(serialized)),
            "chat_template_sha256": _digest(template),
            "special_tokens_sha256": _digest(tokenizer.special_tokens_map),
            "padding_side": tokenizer.padding_side,
            "truncation_side": tokenizer.truncation_side,
        },
        "encoded_data": dict(encoded_data),
        "runtime": dict(runtime_versions),
        "implementation": implementation,
    }


def verify_sft_rank_consensus(
    identity: Mapping[str, Any] | Callable[[], Mapping[str, Any]], *, world_size: int
) -> dict[str, Any]:
    """Reject mismatched inputs on every rank before constructing a trainer.

    The v1 wire header always has three int64s (version, expected world size,
    local validity), followed by eight SHA-256 values only when every header
    is valid. Invalid identities and world-size mismatches therefore fail on
    all participating ranks without leaving peers in the second collective.
    A callable builds the identity inside the validity handshake, so malformed
    local tokenizer/source metadata also rejects every rank. No dataset text,
    configuration values or pickled objects are transmitted.
    """
    import torch
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("SFT rank consensus requires an initialized process group")
    actual_world_size = dist.get_world_size()
    if actual_world_size < 2:
        raise RuntimeError(
            "SFT rank consensus requires at least two participating ranks"
        )
    backend = str(dist.get_backend()).lower()
    if backend not in {"nccl", "gloo"}:
        raise RuntimeError(f"SFT rank consensus does not support backend {backend}")
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if backend == "nccl"
        else torch.device("cpu")
    )
    requested_world_size = (
        world_size if type(world_size) is int and 2 <= world_size <= 2**31 - 1 else 0
    )
    hashes: dict[str, str] = {}
    valid = requested_world_size != 0
    try:
        resolved = identity() if callable(identity) else identity
        if set(resolved) != set(_COMPONENTS):
            raise ValueError("SFT identity must provide exactly the v1 components")
        hashes = {name: _digest(resolved[name]) for name in _COMPONENTS}
    except (
        TypeError,
        ValueError,
        OverflowError,
        OSError,
        ImportError,
        AttributeError,
        RecursionError,
    ):
        valid = False
    header = torch.tensor(
        [1, requested_world_size, int(valid)], dtype=torch.int64, device=device
    )
    headers = [torch.empty_like(header) for _ in range(actual_world_size)]
    dist.all_gather(headers, header)
    decoded = [item.cpu().tolist() for item in headers]
    if any(item[0] != 1 for item in decoded):
        raise RuntimeError("SFT rank consensus protocol mismatch")
    invalid = [rank for rank, item in enumerate(decoded) if item[2] != 1]
    if invalid:
        raise RuntimeError(f"Invalid SFT rank identity on ranks {invalid}")
    if any(item[1] != actual_world_size for item in decoded):
        raise RuntimeError(
            "SFT rank consensus WORLD_SIZE differs from the process group"
        )
    payload = torch.tensor(
        list(b"".join(bytes.fromhex(hashes[name]) for name in _COMPONENTS)),
        dtype=torch.uint8,
        device=device,
    )
    gathered = [torch.empty_like(payload) for _ in range(actual_world_size)]
    dist.all_gather(gathered, payload)
    fingerprints = [bytes(item.cpu().tolist()) for item in gathered]
    differences = [
        name
        for index, name in enumerate(_COMPONENTS)
        if len({row[index * 32 : (index + 1) * 32] for row in fingerprints}) != 1
    ]
    if differences:
        raise RuntimeError(
            "SFT preflight disagreement across ranks: " + ", ".join(differences)
        )
    return {
        "protocol": SFT_CONSENSUS_PROTOCOL,
        "world_size": actual_world_size,
        "backend": backend,
        "verified": True,
        "identity_sha256": _digest(
            {"protocol": SFT_CONSENSUS_PROTOCOL, "components": hashes}
        ),
        "components_sha256": hashes,
    }


__all__ = [
    "SFT_CONSENSUS_PROTOCOL",
    "fingerprint_encoded_rows",
    "sft_preflight_identity",
    "verify_sft_rank_consensus",
]
