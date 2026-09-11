"""Canonical logical-record hashing across benchmark dataset encodings."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


class DatasetContentError(ValueError):
    """Raised when dataset content cannot be canonicalized safely."""


def _json_records(path: Path) -> list[Any]:
    try:
        if path.suffix.lower() == ".jsonl":
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
            records = value if isinstance(value, list) else value.get("data")
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        raise DatasetContentError(
            f"could not parse JSON dataset {path}: {exc}"
        ) from exc
    if not isinstance(records, list) or not records:
        raise DatasetContentError(
            f"{path}: dataset must contain a non-empty record list"
        )
    return records


def _parquet_records(path: Path) -> list[Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise DatasetContentError(
            "canonical Parquet hashing requires pyarrow in the runner environment"
        ) from exc
    try:
        records = parquet.read_table(path).to_pylist()
    except Exception as exc:
        raise DatasetContentError(
            f"could not parse Parquet dataset {path}: {exc}"
        ) from exc
    if not records:
        raise DatasetContentError(f"{path}: dataset must contain at least one record")
    return records


def canonical_dataset_content_sha256(path: Path | str) -> str:
    """Hash ordered logical records independently of JSON/Parquet encoding."""
    raw_path = str(path)
    parsed = urlparse(raw_path)
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise DatasetContentError("file dataset URI must not name a remote host")
        path = Path(unquote(parsed.path)).resolve()
    elif parsed.scheme:
        raise DatasetContentError("dataset content hashing requires a local file")
    else:
        path = Path(raw_path).resolve()
    if not path.is_file():
        raise DatasetContentError(f"dataset is not a file: {path}")
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        records = _json_records(path)
    elif suffix == ".parquet":
        records = _parquet_records(path)
    else:
        raise DatasetContentError(
            f"unsupported dataset encoding {suffix!r}; use JSON, JSONL, or Parquet"
        )
    try:
        payload = json.dumps(
            records,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DatasetContentError(
            f"{path}: records are not canonical JSON values"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=Path)
    args = parser.parse_args(argv)
    try:
        results = {
            path.as_posix(): canonical_dataset_content_sha256(path)
            for path in args.datasets
        }
        if len(set(results.values())) != 1:
            raise DatasetContentError(
                "datasets do not contain identical ordered records"
            )
    except DatasetContentError as exc:
        print(f"dataset content rejected: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
