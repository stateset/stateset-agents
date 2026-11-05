"""Verify the structural export and source lineage of freshly trained LoRA adapters."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

from stateset_agents.data.finetuning import parse_finetuning_json
from stateset_agents.data.source import SFTDatasetIdentity

_JSON_LIMIT = 2 * 1024 * 1024


def _json_file(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    if path.is_symlink():
        raise ValueError("SFT export metadata must be a regular file")
    try:
        with os.fdopen(os.open(path, flags), "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= _JSON_LIMIT:
                raise ValueError("SFT export metadata must be a bounded regular file")
            raw = stream.read(_JSON_LIMIT + 1)
        if len(raw) > _JSON_LIMIT:
            raise ValueError("SFT export metadata exceeds its size limit")
        value = parse_finetuning_json(raw.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(
            f"Invalid or missing SFT export metadata: {path.name}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError("SFT export metadata must be an object")
    return value


def _weight_shapes(path: Path) -> dict[str, list[int]]:
    if path.is_symlink():
        raise ValueError("SFT adapter weights must be regular files")
    try:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size == 0:
            raise ValueError("Empty or nonregular adapter weights")
        # Header/slice inspection does not materialize tensors or unpickle data.
        from safetensors import safe_open

        with safe_open(str(path), framework="np") as weights:
            shapes = {
                name: weights.get_slice(name).get_shape() for name in weights.keys()
            }
    except (OSError, ValueError) as exc:
        raise ValueError("Invalid or missing safetensors adapter weights") from exc
    except Exception as exc:
        raise ValueError("Could not inspect safetensors adapter weights") from exc
    if not shapes or any(
        not shape or any(size <= 0 for size in shape) for shape in shapes.values()
    ):
        raise ValueError("SFT adapter contains no nonempty tensor payload")
    return shapes


def _shapes(root: Path) -> tuple[list[str], dict[str, list[int]]]:
    single = root / "adapter_model.safetensors"
    index = root / "adapter_model.safetensors.index.json"
    if single.exists() or single.is_symlink():
        if index.exists() or index.is_symlink():
            raise ValueError("SFT export has ambiguous single/sharded adapter weights")
        return [single.name], _weight_shapes(single)
    payload = _json_file(index)
    mapping = payload.get("weight_map")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("SFT adapter shard index requires a nonempty weight map")
    for name, shard in mapping.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(shard, str)
            or Path(shard).name != shard
            or not shard.endswith(".safetensors")
            or any(char in shard for char in ("/", "\\", ":"))
            or any(ord(char) < 32 for char in shard)
        ):
            raise ValueError("Invalid SFT adapter shard reference")
    shapes: dict[str, list[int]] = {}
    names = sorted(set(mapping.values()))
    for name in names:
        shard_shapes = _weight_shapes(root / name)
        if set(shard_shapes) != {
            key for key, value in mapping.items() if value == name
        }:
            raise ValueError("SFT adapter shard contents differ from the index")
        shapes.update(shard_shapes)
    return names, shapes


def verify_sft_adapter(
    output: Path,
    *,
    base_model: str,
    lora_r: int,
    dataset_identity: SFTDatasetIdentity,
    model_revision: str | None = None,
    expected_lineage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check a fresh safe LoRA export before declaring training complete.

    Reads tensor headers, configuration and bounded lineage metadata. This does
    not load a base model, check numerical finiteness, prove optimizer updates,
    or establish inference compatibility/quality. Keep export files stable while
    checking them; these local consistency checks are not authenticated receipts.
    """
    if output.is_symlink() or not output.is_dir():
        raise ValueError("SFT export must be a real adapter directory")
    config = _json_file(output / "adapter_config.json")
    if (
        config.get("peft_type") != "LORA"
        or type(config.get("r")) is not int
        or config["r"] != lora_r
    ):
        raise ValueError("SFT adapter configuration differs from the requested LoRA")
    _json_file(output / "tokenizer_config.json")
    names, shapes = _shapes(output)
    pairs = 0
    for name, shape in shapes.items():
        suffix = next(
            (s for s in (".lora_A.weight", ".lora_embedding_A") if name.endswith(s)),
            None,
        )
        if suffix is None:
            continue
        other = name[: -len(suffix)] + suffix.replace("_A", "_B")
        paired = shapes.get(other)
        if (
            paired is None
            or len(shape) != 2
            or len(paired) != 2
            or shape[0] != lora_r
            or shape[0] != paired[1]
        ):
            raise ValueError(
                "SFT adapter has missing or incompatible LoRA tensor pairs"
            )
        pairs += 1
    if pairs == 0 or any(
        name.endswith((".lora_B.weight", ".lora_embedding_B"))
        and name.replace(".lora_B.weight", ".lora_A.weight").replace(
            ".lora_embedding_B", ".lora_embedding_A"
        )
        not in shapes
        for name in shapes
    ):
        raise ValueError("SFT adapter requires complete LoRA tensor pairs")
    lineage = _json_file(output / "stateset_manifest.json")
    source = dataset_identity.source
    expected = {
        "base_model": base_model,
        "dataset_path": source.path if source else None,
        "dataset_sha256": source.sha256 if source else None,
        "dataset_rows": source.rows if source else dataset_identity.examples,
    }
    if (
        any(lineage.get(key) != value for key, value in expected.items())
        or type(lineage.get("dataset_rows")) is not int
    ):
        raise ValueError("SFT export lineage differs from the captured training source")
    hyperparameters = lineage.get("hyperparameters")
    if not isinstance(hyperparameters, dict) or (
        hyperparameters.get("training_content_sha256")
        != dataset_identity.content_sha256
        or type(hyperparameters.get("training_input_rows")) is not int
        or hyperparameters["training_input_rows"] != dataset_identity.examples
        or type(hyperparameters.get("effective_training_rows")) is not int
        or not 0
        < hyperparameters["effective_training_rows"]
        <= dataset_identity.examples
        or hyperparameters.get("model_revision") != model_revision
    ):
        raise ValueError("SFT export lineage differs from the parsed training input")
    if expected_lineage is not None and lineage != expected_lineage:
        raise ValueError("SFT lineage was not persisted for this training run")
    return {
        "format": "peft_lora_safetensors",
        "weights": names,
        "tensor_count": len(shapes),
        "lora_pairs": pairs,
        "training_input_rows": dataset_identity.examples,
        "effective_training_rows": hyperparameters["effective_training_rows"],
        "quality_gain_verified": False,
    }
