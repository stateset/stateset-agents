"""Tests for adapter provenance manifests and lineage reconstruction."""

from __future__ import annotations

import json
from pathlib import Path

from stateset_agents.training.lineage import (
    MANIFEST_NAME,
    AdapterManifest,
    build_lineage,
    discover_adapters,
    hash_dataset,
    read_manifest,
    write_manifest,
)


def _adapter(root, name, manifest=None):
    """Create a plausible adapter directory, optionally with a manifest."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    if manifest is not None:
        write_manifest(d, manifest)
    return d


class TestDatasetHashing:
    def test_hashes_content_and_counts_rows(self, tmp_path):
        data = tmp_path / "d.jsonl"
        data.write_text('{"a": 1}\n{"a": 2}\n\n', encoding="utf-8")
        digest, rows = hash_dataset(data)
        assert rows == 2
        assert digest and len(digest) == 64

    def test_same_name_different_bytes_hash_differently(self, tmp_path):
        """The point of hashing: a reused filename is not evidence."""
        a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
        a.write_text('{"x": 1}\n', encoding="utf-8")
        b.write_text('{"x": 2}\n', encoding="utf-8")
        assert hash_dataset(a)[0] != hash_dataset(b)[0]

    def test_missing_dataset_is_unknown_not_an_error(self, tmp_path):
        assert hash_dataset(tmp_path / "nope.jsonl") == (None, None)


class TestManifestIO:
    def test_round_trips(self, tmp_path):
        d = _adapter(
            tmp_path,
            "gen1",
            AdapterManifest(
                base_model="meta-models/Muse-Glimmer-30B",
                dataset_rows=140,
                hyperparameters={"lora_r": 16},
            ),
        )
        loaded = read_manifest(d)
        assert loaded["base_model"] == "meta-models/Muse-Glimmer-30B"
        assert loaded["hyperparameters"]["lora_r"] == 16
        assert loaded["created_at"]

    def test_absent_manifest_reads_as_none(self, tmp_path):
        assert read_manifest(_adapter(tmp_path, "bare")) is None

    def test_corrupt_manifest_reads_as_none(self, tmp_path):
        d = _adapter(tmp_path, "broken")
        (d / MANIFEST_NAME).write_text("{not json", encoding="utf-8")
        assert read_manifest(d) is None

    def test_write_failure_never_raises(self, tmp_path):
        """A trained adapter must not be reported as failed because its
        bookkeeping could not be written."""
        blocker = tmp_path / "file"
        blocker.write_text("x", encoding="utf-8")
        write_manifest(blocker / "nested", AdapterManifest(base_model="m"))

    def test_manifest_is_valid_json_on_disk(self, tmp_path):
        d = _adapter(tmp_path, "gen1", AdapterManifest(base_model="m"))
        assert json.loads((d / MANIFEST_NAME).read_text(encoding="utf-8"))


class TestDiscovery:
    def test_finds_adapters_with_and_without_manifests(self, tmp_path):
        _adapter(tmp_path, "with", AdapterManifest(base_model="m"))
        _adapter(tmp_path, "without")
        found = discover_adapters(tmp_path)
        assert {a["name"] for a in found} == {"with", "without"}
        assert sum(1 for a in found if a["manifest"]) == 1

    def test_notes_whether_an_eval_ran(self, tmp_path):
        d = _adapter(tmp_path, "evaluated", AdapterManifest(base_model="m"))
        (d / "eval_results.json").write_text("[]", encoding="utf-8")
        _adapter(tmp_path, "unevaluated")
        found = {a["name"]: a["has_eval"] for a in discover_adapters(tmp_path)}
        assert found == {"evaluated": True, "unevaluated": False}

    def test_missing_root_is_empty_not_an_error(self, tmp_path):
        assert discover_adapters(tmp_path / "nope") == []


class TestLineage:
    def test_links_child_to_parent_by_path(self, tmp_path):
        parent = _adapter(tmp_path, "gen1", AdapterManifest(base_model="m"))
        _adapter(
            tmp_path,
            "gen2",
            AdapterManifest(base_model="m", parent_adapter=str(parent)),
        )
        lineage = build_lineage(discover_adapters(tmp_path))
        # Path separators differ by OS; compare on the directory name.
        assert [Path(p).name for p in lineage[str(parent)]] == ["gen2"]

    def test_links_by_name_when_the_path_moved(self, tmp_path):
        """A manifest written on a pod records that pod's absolute path;
        the adapter is then fetched somewhere else entirely."""
        parent = _adapter(tmp_path, "gen1", AdapterManifest(base_model="m"))
        _adapter(
            tmp_path,
            "gen2",
            AdapterManifest(base_model="m", parent_adapter="/workspace/out/gen1"),
        )
        lineage = build_lineage(discover_adapters(tmp_path))
        assert str(parent) in lineage

    def test_first_generation_has_no_links(self, tmp_path):
        _adapter(tmp_path, "gen1", AdapterManifest(base_model="m"))
        assert build_lineage(discover_adapters(tmp_path)) == {}
