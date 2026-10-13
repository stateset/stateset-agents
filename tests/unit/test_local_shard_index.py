"""Local sharded checkpoint indexes cannot escape their artifact directory."""

import json
from pathlib import Path

import pytest

from stateset_agents.core.checkpoint_io import validate_local_shard_indexes


def _write_index(directory: Path, shard: object) -> None:
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.weight": shard}}), encoding="utf-8"
    )


def test_local_shard_index_accepts_regular_file(tmp_path: Path) -> None:
    (tmp_path / "model-00001.safetensors").write_bytes(b"weights")
    _write_index(tmp_path, "model-00001.safetensors")
    validate_local_shard_indexes(tmp_path)


@pytest.mark.parametrize(
    "shard", ["../outside", "/tmp/outside", "..\\outside", "C:shard", "missing", []]
)
def test_local_shard_index_rejects_unsafe_shard(tmp_path: Path, shard: object) -> None:
    _write_index(tmp_path, shard)
    with pytest.raises(ValueError, match="checkpoint shard"):
        validate_local_shard_indexes(tmp_path)


def test_local_shard_index_rejects_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "real-shard.safetensors"
    outside.write_bytes(b"weights")
    try:
        (tmp_path / "shard.safetensors").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this runner")
    _write_index(tmp_path, "shard.safetensors")
    with pytest.raises(ValueError, match="regular file"):
        validate_local_shard_indexes(tmp_path)


def test_local_shard_index_rejects_index_symlink(tmp_path: Path) -> None:
    source = tmp_path / "outside-index.json"
    source.write_text('{"weight_map": {"x": "shard.safetensors"}}', encoding="utf-8")
    try:
        (tmp_path / "model.safetensors.index.json").symlink_to(source)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this runner")
    with pytest.raises(ValueError, match="unsafe checkpoint index"):
        validate_local_shard_indexes(tmp_path)
