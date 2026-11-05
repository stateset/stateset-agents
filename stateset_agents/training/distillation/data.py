"""Typed distillation data: prompts with an optional reference completion
(torch-free).

A :class:`DistillationExample` is one context (a raw ``prompt`` string or a
list of chat messages) and, for offline (teacher-forced) distillation, the
``completion`` the teacher's distribution is matched on: ground truth, a
teacher sample (sequence-level KD) or any other text. On-policy distillation
only needs the prompt: the student samples its own completion.

Records load from strict JSON Lines with the same reader, strict JSON parser
and message validation as the preference data, and split prompt-disjointly on
the shared request identity plus any ``group_id`` / ``source_id``.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stateset_agents.data.finetuning import (
    parse_finetuning_json,
    validate_finetuning_split,
)
from stateset_agents.data.jsonl import read_jsonl_lines

from ..preference.data import (
    Message,
    _identity_text,
    _metadata,
    _optional_id,
    _prompt,
    _response,
    _weight,
    check_preference_overlap,
    group_preference_records,
    split_records,
)

__all__ = [
    "DistillationDataset",
    "DistillationExample",
    "as_distillation_dataset",
    "load_distillation_jsonl",
    "parse_distillation_record",
    "resolve_distillation_splits",
    "split_distillation_dataset",
    "write_distillation_jsonl",
]


@dataclass(frozen=True)
class DistillationExample:
    """One prompt and, optionally, the completion to distil on.

    ``prompt`` is a raw prompt string or chat messages ending before the
    reply; ``completion`` a string or an assistant message (``None`` for a
    prompt-only, on-policy example). ``weight`` scales the example's tokens in
    the loss; ``group_id`` and ``source_id`` keep related examples in one
    split.
    """

    prompt: str | Sequence[Message]
    completion: str | Message | None = None
    weight: float = 1.0
    group_id: str | None = None
    source_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt", _prompt(self.prompt))
        if self.completion is not None:
            object.__setattr__(
                self, "completion", _response(self.completion, "completion")
            )
        object.__setattr__(self, "weight", _weight(self.weight))
        object.__setattr__(self, "group_id", _optional_id(self.group_id, "group_id"))
        object.__setattr__(self, "source_id", _optional_id(self.source_id, "source_id"))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @property
    def prompt_value(self) -> str | tuple[Message, ...]:
        prompt = self.prompt
        return prompt if isinstance(prompt, str) else tuple(prompt)

    @property
    def identity_text(self) -> str:
        """The request text used for split identity (first user message)."""
        return _identity_text(self.prompt_value)

    def to_dict(self) -> dict[str, Any]:
        """The JSONL record this example round-trips through."""
        record: dict[str, Any] = {
            "prompt": (
                self.prompt
                if isinstance(self.prompt, str)
                else [dict(message) for message in self.prompt]
            )
        }
        if self.completion is not None:
            record["completion"] = self.completion
        if self.weight != 1.0:
            record["weight"] = self.weight
        if self.group_id is not None:
            record["group_id"] = self.group_id
        if self.source_id is not None:
            record["source_id"] = self.source_id
        if self.metadata:
            record["metadata"] = dict(self.metadata)
        return copy.deepcopy(record)


def _split_messages(messages: Any) -> tuple[Any, Any]:
    """``messages`` ending in an assistant turn -> (context, final reply)."""
    if isinstance(messages, str | Mapping) or not isinstance(messages, Sequence):
        raise ValueError("messages must be a list of chat messages")
    if not messages:
        raise ValueError("messages must not be empty")
    last = messages[-1]
    if isinstance(last, Mapping) and last.get("role") == "assistant":
        return list(messages[:-1]), last
    return list(messages), None


def parse_distillation_record(row: Any) -> DistillationExample:
    """Validate one JSON record into a :class:`DistillationExample`.

    Accepted shapes: ``{"prompt": str | messages, "completion": ...}``,
    ``{"prompt": ...}`` alone (on-policy only), and ``{"messages": [...]}``
    whose final assistant message (if any) is the completion. Optional:
    ``weight``, ``group_id``, ``source_id``, ``metadata`` and a ``split``
    declaration. Preference fields (``chosen``/``rejected``/``label``) are
    rejected so a preference file is never silently read as prompts.
    """
    if not isinstance(row, Mapping):
        raise ValueError("Expected a JSON object")
    validate_finetuning_split(dict(row))
    for key in ("chosen", "rejected", "label"):
        if key in row:
            raise ValueError(
                f"{key!r} is a preference field; distillation records hold a "
                "prompt and an optional completion"
            )
    if "prompt" in row and "messages" in row:
        raise ValueError("use either prompt or messages, not both")
    common: dict[str, Any] = {
        "weight": row.get("weight", 1.0),
        "group_id": row.get("group_id"),
        "source_id": row.get("source_id"),
        "metadata": row.get("metadata", {}),
    }
    if "messages" in row:
        if "completion" in row:
            raise ValueError("messages carry their own final assistant reply")
        prompt, completion = _split_messages(row["messages"])
        return DistillationExample(prompt=prompt, completion=completion, **common)
    if "prompt" not in row:
        raise ValueError("a distillation record needs prompt or messages")
    return DistillationExample(
        prompt=row["prompt"], completion=row.get("completion"), **common
    )


@dataclass(frozen=True)
class DistillationDataset:
    """An immutable collection of distillation examples."""

    examples: tuple[DistillationExample, ...] = ()

    def __post_init__(self) -> None:
        examples = tuple(self.examples)
        if any(not isinstance(item, DistillationExample) for item in examples):
            raise TypeError("examples must contain DistillationExample records")
        object.__setattr__(self, "examples", examples)

    def __len__(self) -> int:
        return len(self.examples)

    @property
    def records(self) -> tuple[DistillationExample, ...]:
        return self.examples

    @property
    def has_completions(self) -> bool:
        """Whether every example carries a completion (needed offline)."""
        return all(example.completion is not None for example in self.examples)

    @classmethod
    def from_records(
        cls, records: Iterable[DistillationExample | Mapping[str, Any]]
    ) -> DistillationDataset:
        examples: list[DistillationExample] = []
        for index, record in enumerate(records):
            if isinstance(record, Mapping):
                try:
                    record = parse_distillation_record(record)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"record {index}: {exc}") from exc
            if not isinstance(record, DistillationExample):
                raise TypeError(
                    f"record {index} is not a distillation example: {type(record)!r}"
                )
            examples.append(record)
        return cls(tuple(examples))

    def summary(self) -> dict[str, Any]:
        """Counts for logs and dry runs."""
        return {
            "examples": len(self.examples),
            "with_completion": sum(
                1 for example in self.examples if example.completion is not None
            ),
            "prompt_groups": len(group_preference_records(self.examples)),
        }


def load_distillation_jsonl(
    path: str | Path, *, expected_split: str | None = None
) -> DistillationDataset:
    """Read strict UTF-8 distillation JSONL, optionally enforcing its split."""
    validate_finetuning_split({}, expected_split=expected_split)
    examples: list[DistillationExample] = []
    for number, line in read_jsonl_lines(path):
        try:
            row = parse_finetuning_json(line)
            validate_finetuning_split(row, expected_split=expected_split)
            examples.append(parse_distillation_record(row))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}: line {number}: {exc}") from exc
    if not examples:
        raise ValueError(f"{path}: dataset is empty")
    return DistillationDataset(tuple(examples))


def write_distillation_jsonl(dataset: DistillationDataset, path: str | Path) -> Path:
    """Write ``dataset`` as JSONL that :func:`load_distillation_jsonl` reads."""
    target = Path(path)
    lines = [
        json.dumps(example.to_dict(), ensure_ascii=False, allow_nan=False)
        for example in dataset.examples
    ]
    target.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return target


def split_distillation_dataset(
    dataset: DistillationDataset, validation_fraction: float = 0.1, seed: int = 42
) -> tuple[DistillationDataset, DistillationDataset]:
    """Prompt-disjoint, deterministic, order-independent split (exact
    duplicates dropped), as for preference data."""
    train, evaluation = split_records(dataset.examples, validation_fraction, seed)
    result = DistillationDataset(tuple(train)), DistillationDataset(tuple(evaluation))
    check_preference_overlap(*result)
    return result


def as_distillation_dataset(data: Any) -> DistillationDataset:
    """Accept a dataset, a JSONL path, or records/mappings."""
    if isinstance(data, DistillationDataset):
        return data
    if isinstance(data, str | Path):
        return load_distillation_jsonl(data)
    return DistillationDataset.from_records(data)


def resolve_distillation_splits(
    train_data: Any, eval_data: Any, validation_fraction: float, seed: int
) -> tuple[DistillationDataset, DistillationDataset | None]:
    """Training and held-out data that never share a prompt identity."""
    train = as_distillation_dataset(train_data)
    if eval_data is not None:
        evaluation = as_distillation_dataset(eval_data)
        check_preference_overlap(train, evaluation)
        return train, evaluation
    if validation_fraction > 0:
        return split_distillation_dataset(train, validation_fraction, seed)
    return train, None
