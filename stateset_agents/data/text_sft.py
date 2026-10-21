"""Strict text conversation data and source-bound training snapshots."""

from __future__ import annotations

import hashlib
import io
import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class DatasetFormatError(ValueError):
    """A content-free dataset diagnostic with one-based source locations."""

    def __init__(self, code: str, message: str, **location: int | str) -> None:
        self.diagnostic: dict[str, Any] = {
            "code": code,
            "message": message,
            "location": location,
        }
        super().__init__(message)


@dataclass(frozen=True)
class TextSFTSnapshot:
    """Parsed data and hashes computed from the same captured file bytes."""

    train: list[dict[str, Any]]
    validation: list[dict[str, Any]] | None
    dataset_sha256: dict[str, str]
    content_sha256: dict[str, str]
    provenance: dict[str, dict[str, Any]] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        """Return counts and independent copies of source/content hash metadata."""
        return {
            "train_examples": len(self.train),
            "validation_examples": len(self.validation) if self.validation else 0,
            "dataset_sha256": dict(self.dataset_sha256),
            "training_content_sha256": dict(self.content_sha256),
            "dataset_provenance": deepcopy(self.provenance),
        }


class _InvalidJson(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _InvalidJson()
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise _InvalidJson()


def parse_text_sft_records(
    raw: bytes, path: str, *, expected_split: str = "train"
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Normalize text chats and retain declared industry provenance separately.

    Provenance is a source declaration, not proof of independent review. Factory
    validation/test rows cannot enter a training read, even after file renaming.
    Only the normalized messages enter loss computation and semantic hashes.
    """
    if expected_split not in {"train", "validation"}:
        raise ValueError("Only train and validation inputs are supported")
    rows = []
    synthetic = 0
    domains: set[str] = set()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetFormatError(
            "invalid_encoding", "Dataset must use UTF-8 encoding.", byte=exc.start + 1
        ) from None
    for number, line in enumerate(io.StringIO(text, newline=None), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(
                line, object_pairs_hook=_unique_object, parse_constant=_reject_constant
            )
        except json.JSONDecodeError as exc:
            raise DatasetFormatError(
                "invalid_json",
                f"{path}:{number}: Expected one valid JSON object per nonblank line.",
                line=number,
                column=exc.colno,
            ) from None
        except _InvalidJson:
            raise DatasetFormatError(
                "invalid_json",
                f"{path}:{number}: Duplicate keys or non-finite JSON values are not supported.",
                line=number,
            ) from None
        except (ValueError, RecursionError):
            raise DatasetFormatError(
                "parser_limit",
                "JSON nesting or numeric values exceed parser limits.",
                line=number,
            ) from None
        try:
            if not isinstance(row, dict):
                raise ValueError("rows require messages, with optional id")
            if set(row) - {"id", "messages"}:
                if set(row) != {"messages", "synthetic", "group_id", "metadata"}:
                    raise ValueError(
                        "rows require messages, with optional id or generated industry provenance"
                    )
                metadata = row["metadata"]
                if (
                    row["synthetic"] is not True
                    or not isinstance(row["group_id"], str)
                    or not row["group_id"].strip()
                    or not isinstance(metadata, dict)
                    or set(metadata)
                    != {
                        "industry_domain",
                        "family",
                        "split",
                        "case_sha256",
                        "source",
                        "review_status",
                    }
                    or any(not isinstance(value, str) for value in metadata.values())
                    or metadata["source"] != "industry-policy-simulations-v1"
                    or metadata["review_status"] != "not_independently_reviewed"
                    or metadata["industry_domain"]
                    not in {
                        "ecommerce",
                        "logistics",
                        "saas",
                        "travel",
                        "services",
                        "telecommunications",
                        "media",
                        "healthcare",
                        "public-sector",
                    }
                    or len(metadata["case_sha256"]) != 64
                    or any(
                        char not in "0123456789abcdef"
                        for char in metadata["case_sha256"]
                    )
                ):
                    raise ValueError("invalid generated industry provenance")
                if metadata["split"] != expected_split:
                    raise DatasetFormatError(
                        "held_out_split",
                        f"{path}:{number}: Declared dataset split is incompatible with this {expected_split} input.",
                        line=number,
                    )
                synthetic += 1
                domains.add(metadata["industry_domain"])
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("messages must be a nonempty list")
            expected = "user"
            for index, message in enumerate(messages):
                if not isinstance(message, dict) or set(message) != {"role", "content"}:
                    raise ValueError("messages require exactly role and content")
                if message["role"] == "system" and index == 0:
                    pass
                elif message["role"] != expected:
                    raise ValueError("messages must alternate user/assistant")
                else:
                    expected = "assistant" if expected == "user" else "user"
                if (
                    not isinstance(message["content"], str)
                    or not message["content"].strip()
                ):
                    raise ValueError("this text SFT path requires nonempty text")
            if messages[-1]["role"] != "assistant" or expected != "user":
                raise ValueError("a conversation must end with an assistant reply")
            rows.append({"messages": messages})
        except DatasetFormatError:
            raise
        except ValueError as exc:
            # These messages are static schema rules, never source values.
            raise DatasetFormatError(
                "invalid_conversation", f"{path}:{number}: {exc}", line=number
            ) from None
    if not rows:
        raise DatasetFormatError("empty_dataset", f"{path}: dataset is empty")
    return rows, {
        "synthetic_examples": synthetic,
        "unannotated_examples": len(rows) - synthetic,
        "industry_domains": sorted(domains),
        "status": "declared_not_independently_verified",
    }


def _parse_conversations(raw: bytes, path: str) -> list[dict[str, Any]]:
    return parse_text_sft_records(raw, path)[0]


def read_text_conversations(path: str) -> list[dict[str, Any]]:
    """Load strict alternating text chats, rejecting malformed rows with locations."""
    return _parse_conversations(Path(path).read_bytes(), path)


def load_text_sft_snapshot(
    train_data: str, validation_data: str | None = None
) -> TextSFTSnapshot:
    """Read each split once, binding validation and provenance to those bytes.

    A later edit or replacement of either path cannot change this run's parsed
    examples or give its manifest the hash of a different file revision.
    """
    contents = {"train": Path(train_data).read_bytes()}
    if validation_data is not None:
        contents["validation"] = Path(validation_data).read_bytes()
    train, train_provenance = parse_text_sft_records(contents["train"], train_data)
    provenance = {"train": train_provenance}
    validation = None
    if validation_data is not None:
        validation, validation_provenance = parse_text_sft_records(
            contents["validation"], validation_data, expected_split="validation"
        )
        provenance["validation"] = validation_provenance
    if validation is not None:
        train_keys = {json.dumps(row, sort_keys=True) for row in train}
        if any(json.dumps(row, sort_keys=True) in train_keys for row in validation):
            raise ValueError("training and validation conversations overlap")
    return TextSFTSnapshot(
        train=train,
        validation=validation,
        dataset_sha256={
            name: hashlib.sha256(raw).hexdigest() for name, raw in contents.items()
        },
        content_sha256={
            name: hashlib.sha256(
                json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            for name, rows in (("train", train), ("validation", validation))
            if rows is not None
        },
        provenance=provenance,
    )


def load_text_sft_data(
    train_data: str, validation_data: str | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Load text-only chats and reject cross-split duplicate conversations."""
    snapshot = load_text_sft_snapshot(train_data, validation_data)
    return snapshot.train, snapshot.validation


__all__ = [
    "TextSFTSnapshot",
    "read_text_conversations",
    "load_text_sft_data",
    "load_text_sft_snapshot",
    "parse_text_sft_records",
]
