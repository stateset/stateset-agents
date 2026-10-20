"""Content identities captured from the same bytes consumed by a data reader."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetSourceIdentity:
    """Original path, byte hash and nonblank source-line count, without content.

    This is local source lineage, not provider attestation or a count of effective
    training targets after filtering, rendering and truncation.
    """

    path: str
    sha256: str
    rows: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.path, str)
            or not self.path
            or "\x00" in self.path
            or not isinstance(self.sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", self.sha256)
            or type(self.rows) is not int
            or self.rows < 0
        ):
            raise ValueError("Invalid captured dataset source identity")

    @classmethod
    def from_bytes(cls, path: str | Path, data: bytes) -> DatasetSourceIdentity:
        """Identify captured UTF-8 JSONL bytes without rereading a mutable path."""
        return cls(
            path=str(path),
            sha256=hashlib.sha256(data).hexdigest(),
            rows=sum(bool(line.strip()) for line in data.decode("utf-8").splitlines()),
        )

    @classmethod
    def from_dict(cls, value: Any) -> DatasetSourceIdentity:
        """Validate a retained identity; unknown legacy sources stay separate."""
        if not isinstance(value, dict) or set(value) != {"path", "sha256", "rows"}:
            raise ValueError("Invalid captured dataset source identity")
        return cls(path=value["path"], sha256=value["sha256"], rows=value["rows"])

    def to_dict(self) -> dict[str, Any]:
        """Return independent JSON-serializable source metadata."""
        return asdict(self)


@dataclass(frozen=True)
class SFTDatasetIdentity:
    """Bind parsed training rows to their captured source, without retaining text.

    Text normalization can remove metadata from model inputs while the source
    identity continues to refer to complete original file bytes.
    """

    source: DatasetSourceIdentity | None
    content_sha256: str
    examples: int

    def __post_init__(self) -> None:
        if (
            (
                self.source is not None
                and not isinstance(self.source, DatasetSourceIdentity)
            )
            or not isinstance(self.content_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", self.content_sha256)
            or type(self.examples) is not int
            or self.examples < 0
        ):
            raise ValueError("Invalid captured SFT dataset identity")

    @classmethod
    def from_rows(
        cls,
        rows: list[dict[str, Any]],
        *,
        source: DatasetSourceIdentity | None = None,
    ) -> SFTDatasetIdentity:
        """Fingerprint finite JSON rows after any explicit normalization."""
        try:
            content = json.dumps(
                rows, sort_keys=True, ensure_ascii=False, allow_nan=False
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("SFT rows must be finite JSON records") from exc
        return cls(source, hashlib.sha256(content).hexdigest(), len(rows))

    def verify_rows(self, rows: list[dict[str, Any]]) -> None:
        """Reject changes to parsed input before model loading or training."""
        current = self.from_rows(rows)
        if (
            current.content_sha256 != self.content_sha256
            or current.examples != self.examples
        ):
            raise ValueError("Training rows differ from the captured SFT input")
