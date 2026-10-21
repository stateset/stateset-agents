"""Bounded, device-independent fingerprints of dense native training state."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

STATE_FINGERPRINT_PROTOCOL = "stateset-training-state-sha256-v1"
DEFAULT_CHUNK_BYTES = 4 * 1024**2


def state_fingerprint(value: Any, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> str:
    """Hash typed plain state and finite dense tensors without a full CPU copy.

    Mapping insertion order, tensor devices and strides do not affect identity.
    Dtypes, shapes, sequence types and exact value bytes do. Temporary tensor
    copies are bounded by ``chunk_bytes`` (at least one element). This is a
    content identity, not a checkpoint parser or a mathematical equivalence test.
    """
    import torch

    from .grpo_accumulation import _gradient_chunks

    if type(chunk_bytes) is not int or chunk_bytes < 1:
        raise ValueError("State fingerprint chunk_bytes must be a positive integer")
    digest = hashlib.sha256()
    digest.update(STATE_FINGERPRINT_PROTOCOL.encode())
    active: set[int] = set()

    def frame(tag: str, payload: bytes = b"") -> None:
        digest.update(
            tag.encode("ascii") + b":" + str(len(payload)).encode() + b":" + payload
        )

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            if (
                item.layout != torch.strided
                or item.is_quantized
                or item.device.type == "meta"
            ):
                raise ValueError(
                    "State fingerprints require materialized dense tensors"
                )
            frame("tensor", json.dumps([str(item.dtype), list(item.shape)]).encode())
            for view in _gradient_chunks(
                item.detach(), max(1, chunk_bytes // item.element_size())
            ):
                # A singleton view may be considered contiguous while retaining
                # an arbitrary last stride. Allocate explicitly so byte views
                # have stride one even when the chunk holds a single element.
                piece = torch.empty(view.shape, dtype=view.dtype, device="cpu")
                piece.copy_(view)
                if not bool(torch.isfinite(piece).all()):
                    raise ValueError("Training-state tensors must be finite")
                # uint8 also handles BF16/FP8, which NumPy cannot represent directly.
                digest.update(piece.reshape(-1).view(torch.uint8).numpy().tobytes())
            frame("tensor_end")
        elif item is None:
            frame("none")
        elif type(item) is bool:
            frame("bool", b"1" if item else b"0")
        elif type(item) is int:
            frame("int", str(item).encode())
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError("Training-state scalar values must be finite")
            frame("float", item.hex().encode())
        elif type(item) is str:
            frame("str", item.encode("utf-8"))
        elif type(item) is bytes:
            frame("bytes", item)
        elif isinstance(item, (Mapping, list, tuple)):
            if id(item) in active:
                raise ValueError("Cyclic training-state containers are unsupported")
            active.add(id(item))
            try:
                if isinstance(item, Mapping):
                    keys = []
                    for key in item:
                        if type(key) not in (str, int, bool, type(None)):
                            raise ValueError(
                                "Training-state mapping keys must be plain scalars"
                            )
                        keys.append(
                            (json.dumps([type(key).__name__, key]).encode(), key)
                        )
                    frame("mapping", str(len(keys)).encode())
                    for _, key in sorted(keys):
                        visit(key)
                        visit(item[key])
                else:
                    frame(
                        "list" if isinstance(item, list) else "tuple",
                        str(len(item)).encode(),
                    )
                    for child in item:
                        visit(child)
                frame("container_end")
            finally:
                active.remove(id(item))
        else:
            raise ValueError("Unsupported training-state value type")

    visit(value)
    return digest.hexdigest()
