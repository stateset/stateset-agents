"""Load local model weights with complete shard coverage and tied-weight support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stateset_agents.core.checkpoint_io import validate_local_shard_indexes

from .checkpoint_io import load_checkpoint_file


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate key in model checkpoint index")
        result[key] = value
    return result


def _read_weights(path: Path, torch: Any, *, trusted: bool) -> dict[str, Any]:
    state: object
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path))
    elif path.suffix == ".bin":
        state = load_checkpoint_file(path, trusted=trusted, torch_module=torch)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
    else:
        raise ValueError(f"Unsupported checkpoint shard format: {path.name}")
    if not isinstance(state, dict) or any(not isinstance(key, str) for key in state):
        raise ValueError("Model checkpoint must contain a named state dictionary")
    return state


def _tied_names(model: Any, expected: dict[str, Any]) -> list[list[str]]:
    identities: dict[int, list[str]] = {}
    for named in (model.named_parameters, model.named_buffers):
        for name, value in named(remove_duplicate=False):
            if name in expected:
                identities.setdefault(id(value), []).append(name)
    return [names for names in identities.values() if len(names) > 1]


def _missing_aliases(
    keys: set[str], expected: dict[str, Any], ties: list[list[str]]
) -> dict[str, str]:
    if keys - expected.keys():
        raise ValueError("Model checkpoint contains unexpected weight names")
    missing = expected.keys() - keys
    aliases = {}
    for names in ties:
        sources = sorted(set(names) & keys)
        if sources:
            for name in set(names) & missing:
                aliases[name] = sources[0]
    if missing - aliases.keys():
        raise ValueError("Model checkpoint is missing untied weights")
    return aliases


def _check_ties(
    piece: dict[str, Any],
    seen: set[str],
    expected: dict[str, Any],
    ties: list[list[str]],
    torch: Any,
) -> None:
    for names in ties:
        present = [name for name in names if name in piece]
        if not present:
            continue
        values = [piece[name] for name in present]
        if any(not torch.is_tensor(value) for value in values):
            raise ValueError("Tied checkpoint weights must be tensors")
        if any(not torch.equal(values[0], value) for value in values[1:]):
            raise ValueError("Conflicting values for tied checkpoint weights")
        previous = next((name for name in names if name in seen), None)
        if previous is not None:
            target = expected[previous]
            if not torch.equal(
                values[0].to(device=target.device, dtype=target.dtype), target
            ):
                raise ValueError(
                    "Conflicting values for tied checkpoint weights across shards"
                )


def _require_overlap(names: Any, model: Any) -> None:
    """Non-strict loads still need at least one name the model recognises;
    ``load_state_dict(strict=False)`` otherwise reports success while
    changing nothing (e.g. ``module.``-prefixed keys from a DDP wrapper)."""
    state_dict = getattr(model, "state_dict", None)
    if names and callable(state_dict) and not set(names) & set(state_dict()):
        raise ValueError("Model checkpoint weight names match no model parameter")


def load_model_checkpoint(
    model: Any,
    directory: Path,
    *,
    torch_module: Any,
    trusted: bool = False,
    strict: bool = True,
) -> bool:
    """Load one local PyTorch or safetensors checkpoint, optionally sharded.

    Strict loading permits omitted names only for parameter/buffer objects that
    are actually shared by the target model. Shards are read one at a time.
    Index coverage and paths are checked before loading; a later corrupt shard
    can still leave earlier model weights changed. No provider requests occur.
    """
    candidates = [
        directory / name
        for name in (
            "adapter_model.safetensors",
            "adapter_model.bin",
            "pytorch_model.bin",
            "model.safetensors",
            "pytorch_model.bin.index.json",
            "model.safetensors.index.json",
        )
        if (directory / name).exists() or (directory / name).is_symlink()
    ]
    if not candidates:
        return False
    if len(candidates) != 1:
        raise ValueError("Ambiguous model checkpoint: multiple weight layouts")
    path = candidates[0]
    if path.is_symlink() or not path.is_file():
        raise ValueError("Model checkpoint must be a regular local file")
    if path.name.startswith("adapter_model."):
        from peft import get_peft_model_state_dict, set_peft_model_state_dict

        adapter = getattr(model, "active_adapter", "default")
        if not isinstance(adapter, str) or not hasattr(model, "peft_config"):
            raise ValueError("Adapter checkpoints require a matching PEFT model")
        state = _read_weights(path, torch_module, trusted=trusted)
        expected_adapter = get_peft_model_state_dict(model, adapter_name=adapter)
        if set(state) != set(expected_adapter):
            raise ValueError("Adapter checkpoint parameter names differ from model")
        for name, tensor in state.items():
            target = expected_adapter[name]
            if tensor.shape != target.shape or not bool(
                torch_module.isfinite(tensor).all()
            ):
                raise ValueError(f"Invalid adapter checkpoint tensor: {name}")
        result = set_peft_model_state_dict(model, state, adapter_name=adapter)
        # peft annotates the return as ``namedtuple``; it is torch's
        # ``_IncompatibleKeys`` from ``Module.load_state_dict``.
        if result.unexpected_keys:  # type: ignore[attr-defined, unused-ignore]
            raise ValueError("Adapter checkpoint has unexpected parameters")
        return True
    expected = model.state_dict() if strict else {}
    ties = _tied_names(model, expected) if strict else []
    if not path.name.endswith(".index.json"):
        state = _read_weights(path, torch_module, trusted=trusted)
        if not strict:
            _require_overlap(state, model)
        if strict:
            aliases = _missing_aliases(set(state), expected, ties)
            _check_ties(state, set(), expected, ties, torch_module)
            state.update({alias: state[source] for alias, source in aliases.items()})
        model.load_state_dict(state, strict=strict)
        return True

    validate_local_shard_indexes(directory)
    payload = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
    )
    weight_map = payload["weight_map"]
    if any(not isinstance(name, str) or not name for name in weight_map):
        raise ValueError("Checkpoint index requires nonempty weight names")
    if not strict:
        _require_overlap(weight_map, model)
    aliases = _missing_aliases(set(weight_map), expected, ties) if strict else {}
    files: dict[str, set[str]] = {}
    for name, filename in weight_map.items():
        files.setdefault(filename, set()).add(name)
    seen: set[str] = set()
    for filename, names in sorted(files.items()):
        piece = _read_weights(directory / filename, torch_module, trusted=trusted)
        if set(piece) != names:
            raise ValueError(f"Checkpoint shard contents differ from index: {filename}")
        if strict:
            _check_ties(piece, seen, expected, ties, torch_module)
            piece.update(
                {
                    alias: piece[source]
                    for alias, source in aliases.items()
                    if source in piece
                }
            )
        model.load_state_dict(piece, strict=False)
        seen.update(piece)
        del piece
    return True
