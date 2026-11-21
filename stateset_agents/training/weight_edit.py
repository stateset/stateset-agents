"""Checkpoint-level weight edits on safetensors directories.

These transforms operate on tensor files directly and never instantiate a
model class, so they also work for checkpoints whose architecture the
installed Transformers runtime cannot load yet (for example
``inkling2_loop``). A patched checkpoint is a fresh directory: untouched
shards are copied byte-for-byte, edited shards are rewritten one at a
time, and ``weight_edit_manifest.json`` records exactly which tensors
changed. See ``docs/NATIVE_LOCAL_ENGINE.md`` for how edits compose with
GGUF export and local verification.

Fact-teaching specs remain owned by the experimental
``stateset_agents.experimental.weight_space`` work; this module covers
publication-style edits: folding a LoRA adapter into a base checkpoint
and applying a safetensors delta file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "MANIFEST_NAME",
    "TensorInfo",
    "apply_safetensors_delta",
    "inspect_checkpoint",
    "merge_lora_adapter",
]

MANIFEST_NAME = "weight_edit_manifest.json"

_LORA_A_RE = re.compile(r"^(?P<prefix>.+)\.lora_A(?:\.(?P<adapter>[^.]+))?\.weight$")
_LORA_B_RE = re.compile(r"^(?P<prefix>.+)\.lora_B(?:\.(?P<adapter>[^.]+))?\.weight$")


@dataclass(frozen=True)
class TensorInfo:
    """Inventory entry for one checkpoint tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    shard: str


def _safetensors() -> Any:
    try:
        import safetensors
        import safetensors.torch

        return safetensors
    except ImportError as exc:  # pragma: no cover - requires minimal env
        raise ImportError(
            "weight edits need 'safetensors'; "
            "install with `pip install stateset-agents[training]`"
        ) from exc


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - requires minimal env
        raise ImportError(
            "weight edits need 'torch'; "
            "install with `pip install stateset-agents[training]`"
        ) from exc
    return torch


def _shard_names(checkpoint: Path) -> list[str]:
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        names = sorted(set(index.get("weight_map", {}).values()))
        if names:
            return names
    names = sorted(p.name for p in checkpoint.glob("*.safetensors"))
    if not names:
        raise FileNotFoundError(
            f"no safetensors shards found in checkpoint: {checkpoint}"
        )
    return names


def _resolve_checkpoint(checkpoint: str | Path) -> Path:
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {path}")
    return path


def inspect_checkpoint(checkpoint: str | Path) -> dict[str, TensorInfo]:
    """Inventory a safetensors checkpoint without loading tensor data."""
    safetensors = _safetensors()
    root = _resolve_checkpoint(checkpoint)
    inventory: dict[str, TensorInfo] = {}
    for shard in _shard_names(root):
        with safetensors.safe_open(
            str(root / shard), framework="pt", device="cpu"
        ) as handle:
            for name in handle.keys():
                tensor_slice = handle.get_slice(name)
                inventory[name] = TensorInfo(
                    name=name,
                    shape=tuple(tensor_slice.get_shape()),
                    dtype=str(tensor_slice.get_dtype()),
                    shard=shard,
                )
    return inventory


def _load_safetensors_file(path: Path) -> dict[str, Any]:
    _safetensors()
    _torch()
    from safetensors.torch import load_file

    return dict(load_file(str(path)))


def _sha256_tensor(tensor: Any) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    return hashlib.sha256(bytes(contiguous.untyped_storage())).hexdigest()


def _prepare_output(base: Path, output: str | Path) -> Path:
    out = Path(output).expanduser().resolve()
    if out == base:
        raise ValueError("refusing an in-place weight edit; pass a separate output_dir")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _copy_untouched(base: Path, out: Path, *, skip: set[str]) -> None:
    """Copy everything except rewritten shards and stale manifests."""
    for source in sorted(base.rglob("*")):
        if source.is_dir():
            continue
        relative = source.relative_to(base)
        if relative.name in skip or relative.name == MANIFEST_NAME:
            continue
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _apply_edits(
    base_checkpoint: str | Path,
    output_dir: str | Path,
    edits: Mapping[str, Callable[[Any], Any]],
    *,
    operation: str,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply per-tensor transforms shard by shard and write a manifest."""
    torch = _torch()
    _safetensors()
    from safetensors.torch import save_file

    base = _resolve_checkpoint(base_checkpoint)
    out = _prepare_output(base, output_dir)
    inventory = inspect_checkpoint(base)
    unknown = sorted(set(edits) - set(inventory))
    if unknown:
        raise ValueError(f"edit target(s) not in checkpoint {base}: {unknown}")

    changed_shards = {inventory[name].shard for name in edits}
    changed_entries: list[dict[str, Any]] = []
    for shard in sorted(changed_shards):
        tensors = _load_safetensors_file(base / shard)
        for name in sorted(edits):
            if inventory[name].shard != shard:
                continue
            before = tensors[name]
            after = edits[name](before)
            if tuple(after.shape) != tuple(before.shape):
                raise ValueError(
                    f"edit for {name} changed shape "
                    f"{tuple(before.shape)} -> {tuple(after.shape)}"
                )
            delta = after.float() - before.float()
            tensors[name] = after
            changed_entries.append(
                {
                    "name": name,
                    "shard": shard,
                    "delta_l2": float(torch.linalg.vector_norm(delta)),
                    "delta_max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
                    "sha256": _sha256_tensor(after),
                }
            )
        save_file(tensors, str(out / shard), metadata={"format": "pt"})

    _copy_untouched(base, out, skip=changed_shards)
    manifest: dict[str, Any] = {
        "operation": operation,
        "base_checkpoint": str(base),
        "parameters": dict(parameters),
        "changed_tensors": sorted(
            changed_entries, key=lambda entry: str(entry["name"])
        ),
    }
    (out / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(
        "%s: edited %d tensor(s) into %s",
        operation,
        len(changed_entries),
        out,
    )
    return manifest


def apply_safetensors_delta(
    base_checkpoint: str | Path,
    delta_path: str | Path,
    output_dir: str | Path,
    *,
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Write ``base + alpha * delta`` for every tensor in a delta file."""
    base = _resolve_checkpoint(base_checkpoint)
    inventory = inspect_checkpoint(base)
    delta_file = Path(delta_path).expanduser().resolve()
    if not delta_file.is_file():
        raise FileNotFoundError(f"delta file not found: {delta_file}")
    delta_tensors = _load_safetensors_file(delta_file)
    if not delta_tensors:
        raise ValueError(f"delta file contains no tensors: {delta_file}")

    edits: dict[str, Callable[[Any], Any]] = {}
    for name, delta in delta_tensors.items():
        info = inventory.get(name)
        if info is None:
            edits[name] = lambda base_tensor: base_tensor  # rejected below
            continue
        if tuple(delta.shape) != info.shape:
            raise ValueError(
                f"delta shape for {name} is {tuple(delta.shape)}, "
                f"checkpoint shape is {info.shape}"
            )

        def _add(base_tensor: Any, _delta: Any = delta, _alpha: float = alpha) -> Any:
            return (base_tensor.float() + float(_alpha) * _delta.float()).to(
                base_tensor.dtype
            )

        edits[name] = _add
    # Unknown names surface through _apply_edits with their names attached.
    return _apply_edits(
        base,
        output_dir,
        edits,
        operation="safetensors_delta",
        parameters={"delta_path": str(delta_file), "alpha": float(alpha)},
    )


def _base_candidates(adapter_prefix: str) -> list[str]:
    candidates = [f"{adapter_prefix}.weight"]
    for strip in ("base_model.model.", "base_model.", "model."):
        if adapter_prefix.startswith(strip):
            candidates.append(f"{adapter_prefix[len(strip):]}.weight")
    seen: list[str] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.append(candidate)
    return seen


def merge_lora_adapter(
    base_checkpoint: str | Path,
    adapter_dir: str | Path,
    output_dir: str | Path,
    *,
    scale: float | None = None,
) -> dict[str, Any]:
    """Fold a PEFT LoRA adapter into a base checkpoint, tensor by tensor.

    The merge is ``W + (lora_alpha / r) * (B @ A)`` per adapted linear,
    computed in float32 and cast back to each base tensor's dtype. No
    model class is instantiated. Embedding adapters
    (``lora_embedding_A/B``) are rejected explicitly.
    """
    base = _resolve_checkpoint(base_checkpoint)
    inventory = inspect_checkpoint(base)
    adapter = Path(adapter_dir).expanduser().resolve()
    config_path = adapter / "adapter_config.json"
    weights_path = adapter / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            f"PEFT adapter needs adapter_config.json and "
            f"adapter_model.safetensors in {adapter}"
        )
    adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    if scale is None:
        rank = adapter_config.get("r")
        alpha = adapter_config.get("lora_alpha")
        if not rank or alpha is None:
            raise ValueError(
                "adapter_config.json must define r and lora_alpha, "
                "or pass scale explicitly"
            )
        resolved_scale = float(alpha) / float(rank)
    else:
        resolved_scale = float(scale)

    adapter_tensors = _load_safetensors_file(weights_path)
    if any("lora_embedding" in name for name in adapter_tensors):
        raise ValueError("embedding LoRA adapters (lora_embedding_A/B) are unsupported")
    pairs: dict[str, tuple[Any, Any]] = {}
    b_tensors: dict[tuple[str, str | None], Any] = {}
    for name, tensor in adapter_tensors.items():
        match = _LORA_B_RE.match(name)
        if match:
            b_tensors[(match.group("prefix"), match.group("adapter"))] = tensor
    for name, a_tensor in adapter_tensors.items():
        match = _LORA_A_RE.match(name)
        if not match:
            continue
        key = (match.group("prefix"), match.group("adapter"))
        if key not in b_tensors:
            raise ValueError(f"LoRA A tensor {name} has no matching B tensor")
        pairs[match.group("prefix")] = (a_tensor, b_tensors[key])
    if not pairs:
        raise ValueError(f"no LoRA A/B tensor pairs found in {weights_path}")

    edits: dict[str, Callable[[Any], Any]] = {}
    for prefix, (a_tensor, b_tensor) in sorted(pairs.items()):
        base_name = next((c for c in _base_candidates(prefix) if c in inventory), None)
        if base_name is None:
            raise ValueError(
                f"no base tensor matches LoRA prefix {prefix!r} "
                f"(tried {_base_candidates(prefix)})"
            )
        expected = inventory[base_name].shape
        if a_tensor.ndim != 2 or b_tensor.ndim != 2:
            raise ValueError(f"LoRA tensors for {base_name} must be 2-D")
        product_shape = (b_tensor.shape[0], a_tensor.shape[1])
        if b_tensor.shape[1] != a_tensor.shape[0] or product_shape != expected:
            raise ValueError(
                f"LoRA shape mismatch for {base_name}: B{tuple(b_tensor.shape)}"
                f" @ A{tuple(a_tensor.shape)} != base shape {expected}"
            )

        def _merge(
            base_tensor: Any,
            _a: Any = a_tensor,
            _b: Any = b_tensor,
            _scale: float = resolved_scale,
        ) -> Any:
            update = _b.float() @ _a.float()
            return (base_tensor.float() + _scale * update).to(base_tensor.dtype)

        edits[base_name] = _merge

    return _apply_edits(
        base,
        output_dir,
        edits,
        operation="merge_lora_adapter",
        parameters={
            "adapter_dir": str(adapter),
            "scale": resolved_scale,
        },
    )
