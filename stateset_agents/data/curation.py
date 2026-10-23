"""Redact and split production conversations without crossing source boundaries."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from stateset_agents.data.jsonl import read_jsonl_lines

if TYPE_CHECKING:
    from stateset_agents.data.decontamination import (
        DecontaminationConfig,
        TextSource,
    )

_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"(?<!\w)(?:\+\d[\d ()-]{8,}\d|\(\d{3}\)\s*\d{3}[- ]\d{4})(?!\w)")
_CREDENTIAL = re.compile(
    r"(?i)\b(?:bearer\s+[A-Za-z0-9._~+/-]{12,}|(?:sk-|rpa_)[A-Za-z0-9_-]{12,})"
)
_PRIVATE_KEYS = {
    "api_key",
    "apikey",
    "access_token",
    "authorization",
    "password",
    "secret",
    "phone",
    "email",
    "address",
    "full_name",
}


def redact_record(value: Any) -> Any:
    """Redact known contact and credential patterns recursively without mutation.

    This deterministic baseline is not exhaustive PII recognition. Deployments
    must add domain-specific review before exporting sensitive production data.
    """
    if isinstance(value, str):
        return _CREDENTIAL.sub(
            "[CREDENTIAL]", _PHONE.sub("[PHONE]", _EMAIL.sub("[EMAIL]", value))
        )
    if isinstance(value, list):
        return [redact_record(item) for item in value]
    if isinstance(value, dict):
        return {
            key: (
                "[REDACTED]"
                if key.lower().replace("-", "_") in _PRIVATE_KEYS
                else redact_record(item)
            )
            for key, item in value.items()
        }
    return value


def _time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Conversation timestamps must be timezone-aware ISO strings")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Conversation timestamps require a timezone")
    return result.astimezone(timezone.utc)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False, ensure_ascii=False).encode()
    ).hexdigest()


def curate_conversations(
    records: list[dict[str, Any]],
    *,
    validation_after: str,
    test_after: str,
    minimum_reward: float | None = None,
    decontaminate_against: Sequence[TextSource] = (),
    decontamination: DecontaminationConfig | None = None,
) -> dict[str, Any]:
    """Return redacted, deduplicated temporal splits with source-group isolation.

    Duplicate conversations union their source groups before splitting; a group
    spanning a cutoff moves entirely into its newest split. Reward filtering
    applies only to training, leaving all held-out outcomes available for eval.

    ``decontaminate_against`` names eval sets (see
    :mod:`stateset_agents.data.decontamination`): training rows that share an
    n-gram (or, when configured, a near duplicate) with any of them are removed
    and counted in the manifest's ``decontamination`` block. Held-out splits are
    never filtered.
    """
    validation_time, test_time = _time(validation_after), _time(test_after)
    if validation_time >= test_time:
        raise ValueError("Validation cutoff must precede test cutoff")
    if minimum_reward is not None and (
        type(minimum_reward) not in (int, float) or not math.isfinite(minimum_reward)
    ):
        raise ValueError("Minimum reward must be finite")
    parents: dict[str, str] = {}

    def root(source: str) -> str:
        parents.setdefault(source, source)
        while parents[source] != source:
            parents[source] = parents[parents[source]]
            source = parents[source]
        return source

    prepared = []
    contents: dict[str, str] = {}
    for record in records:
        source = record.get("source_id")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Every conversation requires a nonempty source_id")
        timestamp = _time(record.get("timestamp"))
        messages = record.get("messages")
        if (
            not isinstance(messages, list)
            or not messages
            or not any(
                isinstance(message, dict) and message.get("role") == "assistant"
                for message in messages
            )
        ):
            raise ValueError(
                "Conversation requires messages including an assistant action"
            )
        for message in messages:
            if (
                not isinstance(message, dict)
                or message.get("role") not in {"system", "user", "assistant", "tool"}
                or not isinstance(message.get("content"), str)
            ):
                raise ValueError(
                    "Conversation messages require a supported role and text content"
                )
        reward = record.get("reward")
        if reward is not None and (
            type(reward) not in (int, float) or not math.isfinite(reward)
        ):
            raise ValueError("Conversation reward must be finite")
        clean = redact_record(copy.deepcopy(record))
        fingerprint = _digest(clean["messages"])
        root(source)
        if fingerprint in contents:
            first, second = sorted((root(source), root(contents[fingerprint])))
            parents[second] = first
        else:
            contents[fingerprint] = source
        prepared.append((source, timestamp, fingerprint, clean))
    newest: dict[str, datetime] = {}
    for source, timestamp, _, _ in prepared:
        group = root(source)
        newest[group] = max(newest.get(group, timestamp), timestamp)
    splits: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    seen = set()
    filtered = 0
    for source, _timestamp, fingerprint, clean in sorted(
        prepared, key=lambda item: (item[1], item[2], item[0])
    ):
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        group = root(source)
        split = (
            "test"
            if newest[group] >= test_time
            else "validation" if newest[group] >= validation_time else "train"
        )
        if (
            split == "train"
            and minimum_reward is not None
            and (clean.get("reward") is None or clean["reward"] < minimum_reward)
        ):
            filtered += 1
            continue
        clean["source_id"] = _digest(source)
        clean["source_group"] = _digest(group)
        clean["content_sha256"] = fingerprint
        clean["split"] = split
        splits[split].append(clean)
    decontaminated = (
        _decontaminate_training(splits, decontaminate_against, decontamination)
        if decontaminate_against
        else None
    )
    manifest = {
        "schema_version": 1,
        "input_rows": len(records),
        "duplicate_rows": len(prepared) - len(seen),
        "filtered_training_rows": filtered,
        "validation_after": validation_after,
        "test_after": test_after,
        "minimum_reward": minimum_reward,
        "redaction": "known-contact-and-credential-patterns-v1",
        "split_sha256": {name: _digest(rows) for name, rows in splits.items()},
        "split_rows": {name: len(rows) for name, rows in splits.items()},
    }
    if decontaminated is not None:
        manifest["decontamination"] = decontaminated
    return {"manifest": manifest, "splits": splits}


def _decontaminate_training(
    splits: dict[str, list[dict[str, Any]]],
    evals: Sequence[TextSource],
    config: DecontaminationConfig | None,
) -> dict[str, Any]:
    """Drop training rows that overlap the eval sets; return the manifest block."""
    from stateset_agents.data.decontamination import (
        find_contamination,
        text_source_from_rows,
    )

    train = text_source_from_rows(
        "train",
        [(str(index), row) for index, row in enumerate(splits["train"])],
        config=config,
    )
    result = find_contamination([train], evals, config)
    removed = {int(row_id) for row_id in result.contaminating}
    splits["train"] = [
        row for index, row in enumerate(splits["train"]) if index not in removed
    ]
    return {
        "method": result.report["method"],
        "config": result.report["config"],
        "eval_sets": {
            name: {
                "sha256": report["sha256"],
                "rows": report["rows"],
                "contaminated_rows": report["contaminated_rows"],
            }
            for name, report in result.report["eval_sets"].items()
        },
        "removed_training_rows": len(removed),
    }


def main() -> None:
    """Write a new curated dataset bundle with fingerprints and isolated splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-after", required=True)
    parser.add_argument("--test-after", required=True)
    parser.add_argument("--minimum-reward", type=float)
    parser.add_argument(
        "--decontaminate-against",
        action="append",
        default=[],
        metavar="EVAL",
        help="Eval set (path, NAME=PATH or builtin:NAME) whose n-grams training "
        "rows must not contain; repeatable.",
    )
    parser.add_argument("--ngram-size", type=int, default=13)
    args = parser.parse_args()
    records = [json.loads(line) for _, line in read_jsonl_lines(args.input)]
    evals: list[TextSource] = []
    config = None
    if args.decontaminate_against:
        from stateset_agents.data.decontamination import (
            DecontaminationConfig,
            resolve_eval_source,
        )

        config = DecontaminationConfig(
            ngram_size=args.ngram_size, min_tokens=min(8, args.ngram_size)
        )
        evals = [
            resolve_eval_source(spec, config=config)
            for spec in args.decontaminate_against
        ]
    bundle = curate_conversations(
        records,
        validation_after=args.validation_after,
        test_after=args.test_after,
        minimum_reward=args.minimum_reward,
        decontaminate_against=evals,
        decontamination=config,
    )
    bundle["manifest"]["source_file_sha256"] = hashlib.sha256(
        args.input.read_bytes()
    ).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    for split, rows in bundle["splits"].items():
        (args.output / f"{split}.jsonl").write_text(
            "".join(
                json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
            )
        )
    (args.output / "manifest.json").write_text(
        json.dumps(bundle["manifest"], indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
