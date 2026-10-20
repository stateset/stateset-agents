"""Deterministic split checks for independently judged support studies.

These checks detect exact and normalized input reuse, not semantic paraphrases.
Human review and representative held-out data remain separate requirements.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from .agent_runs import content_hash


def _normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def validate_case_separation(
    partitions: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, str]]:
    """Reject repeated cases and source groups crossing study partitions.

    Support inputs are identified by ``user_query`` and unordered policy facts,
    independently of labels, IDs and extra metadata. Optional ``source_id``
    groups may occur repeatedly within one split, but never across splits.
    Other scenario schemas retain the full-content duplicate check. Returned
    hashes bind the original, unnormalized cases for evidence verification.
    """
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    seen_inputs: set[str] = set()
    sources: dict[str, str] = {}
    result: dict[str, dict[str, str]] = {}
    for split, cases in partitions.items():
        if not cases:
            raise ValueError(f"{split} cases cannot be empty")
        hashes: dict[str, str] = {}
        for case in cases:
            if not isinstance(case, Mapping):
                raise ValueError("Study cases must be scenario objects")
            case_id = case.get("id")
            if (
                not isinstance(case_id, str)
                or not _normalized(case_id)
                or _normalized(case_id) in seen_ids
            ):
                raise ValueError("Study case IDs must be nonempty and globally unique")
            digest = content_hash({k: v for k, v in case.items() if k != "id"})
            if digest in seen_content:
                raise ValueError("Study partitions contain repeated case content")
            seen_ids.add(_normalized(case_id))
            seen_content.add(digest)
            if "user_query" in case:
                query, facts = case["user_query"], case.get("policy_facts")
                if (
                    not isinstance(query, str)
                    or not _normalized(query)
                    or not isinstance(facts, list)
                    or not facts
                    or any(not isinstance(f, str) or not _normalized(f) for f in facts)
                ):
                    raise ValueError("Support cases require a query and policy facts")
                identity = content_hash(
                    {
                        "query": _normalized(query),
                        "facts": sorted({_normalized(f) for f in facts}),
                    }
                )
                if identity in seen_inputs:
                    raise ValueError("Study partitions contain repeated support inputs")
                seen_inputs.add(identity)
            if "source_id" in case:
                source = case["source_id"]
                if not isinstance(source, str) or not _normalized(source):
                    raise ValueError("Study source_id must be a nonempty string")
                source = _normalized(source)
                if source in sources and sources[source] != split:
                    raise ValueError("Study source groups overlap across partitions")
                sources[source] = split
            hashes[case_id] = content_hash(dict(case))
        result[split] = hashes
    return result
