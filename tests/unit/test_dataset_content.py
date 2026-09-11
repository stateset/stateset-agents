"""Tests for encoding-independent benchmark dataset identity."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
SPEC = importlib.util.spec_from_file_location(
    "dataset_content", BENCHMARKS / "dataset_content.py"
)
assert SPEC is not None and SPEC.loader is not None
dataset_content = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dataset_content
SPEC.loader.exec_module(dataset_content)


def test_json_and_jsonl_share_a_canonical_ordered_record_digest(tmp_path: Path) -> None:
    records = [{"answer": 2, "prompt": "1+1"}, {"prompt": "2+2", "answer": 4}]
    json_path = tmp_path / "data.json"
    jsonl_path = tmp_path / "data.jsonl"
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    jsonl_path.write_text(
        "\n".join(json.dumps(record, sort_keys=False) for record in records) + "\n",
        encoding="utf-8",
    )
    assert dataset_content.canonical_dataset_content_sha256(
        json_path
    ) == dataset_content.canonical_dataset_content_sha256(jsonl_path)
    assert dataset_content.canonical_dataset_content_sha256(
        json_path.as_uri()
    ) == dataset_content.canonical_dataset_content_sha256(json_path)
    assert dataset_content.main([str(json_path), str(jsonl_path)]) == 0


def test_record_order_and_values_are_semantic(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text('[{"id":1},{"id":2}]', encoding="utf-8")
    second.write_text('[{"id":2},{"id":1}]', encoding="utf-8")
    assert dataset_content.canonical_dataset_content_sha256(
        first
    ) != dataset_content.canonical_dataset_content_sha256(second)
    assert dataset_content.main([str(first), str(second)]) == 2


@pytest.mark.parametrize("name", ["empty.jsonl", "data.csv"])
def test_empty_or_unsupported_datasets_fail_closed(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.write_text("", encoding="utf-8")
    with pytest.raises(dataset_content.DatasetContentError):
        dataset_content.canonical_dataset_content_sha256(path)
