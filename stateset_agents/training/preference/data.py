"""Typed pairwise and unpaired preference data (torch-free).

A :class:`PreferencePair` is one context (a raw ``prompt`` string or a list of
chat messages) with a *chosen* and a *rejected* assistant response. KTO also
accepts :class:`UnpairedExample` records (one response with a desirable or
undesirable ``label``). Records load from strict JSON Lines through the shared
reader (:mod:`stateset_agents.data.jsonl`) and the strict JSON parser of the
fine-tuning data path (NaN and duplicate keys are rejected).

Train/evaluation splits group records by the shared request identity
(:func:`stateset_agents.data.identity.prompt_identity` of the first user
message, or of the raw prompt) plus any ``group_id`` / ``source_id``, so the
two splits never share a prompt, conversation or source.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stateset_agents.data.finetuning import (
    parse_finetuning_json,
    validate_finetuning_split,
)
from stateset_agents.data.identity import first_user_text, prompt_identity
from stateset_agents.data.jsonl import read_jsonl_lines

logger = logging.getLogger(__name__)

__all__ = [
    "Message",
    "PreferenceDataset",
    "PreferencePair",
    "PreferenceRecord",
    "UnpairedExample",
    "as_preference_dataset",
    "check_preference_overlap",
    "group_preference_records",
    "load_preference_jsonl",
    "parse_preference_record",
    "resolve_splits",
    "split_preference_dataset",
    "write_preference_jsonl",
]

#: One chat message (``{"role": ..., "content": ...}`` plus optional fields
#: such as ``tool_calls``), stored as a private copy.
Message = dict[str, Any]

_ROLES = frozenset({"system", "developer", "user", "assistant", "tool"})


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _message(value: Any, where: str) -> Message:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a message object")
    role = value.get("role")
    if not isinstance(role, str) or role not in _ROLES:
        raise ValueError(f"{where} has an unsupported role {role!r}")
    content = value.get("content")
    if content is None and role == "assistant" and value.get("tool_calls"):
        pass
    elif not isinstance(content, str):
        raise ValueError(f"{where}.content must be a string")
    message = copy.deepcopy(dict(value))
    _canonical(message)  # JSON-serializable and finite
    return message


def _prompt(value: Any) -> str | tuple[Message, ...]:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("prompt must be a nonempty string")
        return value
    if isinstance(value, Mapping) or not isinstance(value, Sequence) or not value:
        raise ValueError("prompt must be a nonempty string or list of messages")
    messages = tuple(
        _message(item, f"prompt[{index}]") for index, item in enumerate(value)
    )
    if not any(message["role"] == "user" for message in messages):
        raise ValueError("prompt messages must contain a user message")
    if messages[-1]["role"] == "assistant":
        raise ValueError(
            "prompt messages must not end with an assistant turn; the response "
            "belongs in chosen/rejected/completion"
        )
    return messages


def _response(value: Any, where: str) -> str | Message:
    if isinstance(value, str):
        return value
    if (
        isinstance(value, Sequence)
        and not isinstance(value, Mapping)
        and len(value) == 1
    ):
        value = value[0]
    message = _message(value, where)
    if message["role"] != "assistant":
        raise ValueError(f"{where} must be an assistant message")
    return message


def _optional_id(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string when supplied")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _weight(value: Any) -> float:
    weight = _number(value, "weight")
    if weight <= 0:
        raise ValueError("weight must be positive")
    return weight


def _metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be an object")
    copied = copy.deepcopy(dict(value))
    _canonical(copied)
    return copied


def _response_text(response: str | Message) -> str:
    if isinstance(response, str):
        return response
    content = response.get("content")
    return content if isinstance(content, str) else ""


def _identity_text(prompt: str | tuple[Message, ...]) -> str:
    if isinstance(prompt, str):
        return prompt
    text = first_user_text(prompt)
    return text if text is not None else ""


@dataclass(frozen=True)
class PreferencePair:
    """One context with a preferred (``chosen``) and dispreferred response.

    ``prompt`` is a raw prompt string or a sequence of chat messages ending
    before the response; ``chosen`` and ``rejected`` are response strings or
    assistant messages. ``margin`` (how much better ``chosen`` is) is used by
    the Bradley-Terry reward model loss; ``weight`` scales the pair's loss in
    every method. ``group_id`` and ``source_id`` keep related records in one
    split.
    """

    prompt: str | Sequence[Message]
    chosen: str | Message
    rejected: str | Message
    margin: float | None = None
    weight: float = 1.0
    group_id: str | None = None
    source_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt", _prompt(self.prompt))
        object.__setattr__(self, "chosen", _response(self.chosen, "chosen"))
        object.__setattr__(self, "rejected", _response(self.rejected, "rejected"))
        if _canonical(self.chosen) == _canonical(self.rejected):
            raise ValueError("chosen and rejected responses are identical")
        if self.margin is not None:
            object.__setattr__(self, "margin", _number(self.margin, "margin"))
        object.__setattr__(self, "weight", _weight(self.weight))
        object.__setattr__(self, "group_id", _optional_id(self.group_id, "group_id"))
        object.__setattr__(self, "source_id", _optional_id(self.source_id, "source_id"))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @property
    def identity_text(self) -> str:
        """The request text used for split identity (first user message)."""
        return _identity_text(self.prompt_value)

    @property
    def prompt_value(self) -> str | tuple[Message, ...]:
        """The normalized prompt (a string or a tuple of messages)."""
        prompt = self.prompt
        return prompt if isinstance(prompt, str) else tuple(prompt)

    @property
    def chosen_text(self) -> str:
        return _response_text(self.chosen)

    @property
    def rejected_text(self) -> str:
        return _response_text(self.rejected)

    def to_dict(self) -> dict[str, Any]:
        """The JSONL record this pair round-trips through."""
        record: dict[str, Any] = {
            "prompt": (
                self.prompt
                if isinstance(self.prompt, str)
                else [dict(message) for message in self.prompt]
            ),
            "chosen": self.chosen,
            "rejected": self.rejected,
        }
        _add_common(record, self)
        if self.margin is not None:
            record["margin"] = self.margin
        return copy.deepcopy(record)


@dataclass(frozen=True)
class UnpairedExample:
    """One response labelled desirable (``label=True``) or not, for KTO."""

    prompt: str | Sequence[Message]
    completion: str | Message
    label: bool
    weight: float = 1.0
    group_id: str | None = None
    source_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt", _prompt(self.prompt))
        object.__setattr__(self, "completion", _response(self.completion, "completion"))
        if type(self.label) is not bool:
            raise ValueError("label must be a boolean")
        object.__setattr__(self, "weight", _weight(self.weight))
        object.__setattr__(self, "group_id", _optional_id(self.group_id, "group_id"))
        object.__setattr__(self, "source_id", _optional_id(self.source_id, "source_id"))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @property
    def identity_text(self) -> str:
        return _identity_text(self.prompt_value)

    @property
    def prompt_value(self) -> str | tuple[Message, ...]:
        prompt = self.prompt
        return prompt if isinstance(prompt, str) else tuple(prompt)

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "prompt": (
                self.prompt
                if isinstance(self.prompt, str)
                else [dict(message) for message in self.prompt]
            ),
            "completion": self.completion,
            "label": self.label,
        }
        _add_common(record, self)
        return copy.deepcopy(record)


PreferenceRecord = PreferencePair | UnpairedExample


def _add_common(record: dict[str, Any], item: PreferenceRecord) -> None:
    if item.weight != 1.0:
        record["weight"] = item.weight
    if item.group_id is not None:
        record["group_id"] = item.group_id
    if item.source_id is not None:
        record["source_id"] = item.source_id
    if item.metadata:
        record["metadata"] = dict(item.metadata)


def _split_implicit_prompt(chosen: Any, rejected: Any) -> tuple[list[Any], Any, Any]:
    """Split full conversations sharing a context into (prompt, chosen, rejected)."""
    if not all(
        isinstance(side, Sequence) and not isinstance(side, str | Mapping)
        for side in (chosen, rejected)
    ):
        raise ValueError(
            "a record without prompt/messages needs chosen and rejected "
            "conversations that share their context"
        )
    if len(chosen) < 2 or len(rejected) < 2:
        raise ValueError("implicit-prompt conversations need a context and a reply")
    if _canonical(list(chosen[:-1])) != _canonical(list(rejected[:-1])):
        raise ValueError(
            "chosen and rejected conversations must share every message but the last"
        )
    return list(chosen[:-1]), chosen[-1], rejected[-1]


def parse_preference_record(row: Any) -> PreferenceRecord:
    """Validate one JSON record into a :class:`PreferencePair` or
    :class:`UnpairedExample`.

    Accepted shapes: ``{"prompt": str | messages, "chosen", "rejected"}``,
    ``{"messages": [...], "chosen", "rejected"}``, TRL's implicit-prompt
    conversations (``chosen``/``rejected`` full conversations sharing all but
    the last message), and unpaired ``{"prompt", "completion", "label"}``.
    Optional: ``margin``, ``weight``, ``group_id``, ``source_id``,
    ``metadata`` and a ``split`` declaration. Other keys are ignored.
    """
    if not isinstance(row, Mapping):
        raise ValueError("Expected a JSON object")
    validate_finetuning_split(dict(row))
    if "prompt" in row and "messages" in row:
        raise ValueError("use either prompt or messages, not both")
    prompt = row.get("prompt", row.get("messages"))
    common: dict[str, Any] = {
        "weight": row.get("weight", 1.0),
        "group_id": row.get("group_id"),
        "source_id": row.get("source_id"),
        "metadata": row.get("metadata", {}),
    }
    if "completion" in row or "label" in row:
        if "chosen" in row or "rejected" in row:
            raise ValueError(
                "a record is either paired (chosen/rejected) or unpaired "
                "(completion/label), not both"
            )
        if "completion" not in row or "label" not in row:
            raise ValueError("an unpaired record needs completion and label")
        if prompt is None:
            raise ValueError("an unpaired record needs prompt or messages")
        if "margin" in row:
            raise ValueError("margin applies to paired records only")
        return UnpairedExample(
            prompt=prompt, completion=row["completion"], label=row["label"], **common
        )
    if "chosen" not in row or "rejected" not in row:
        raise ValueError("a preference record needs chosen and rejected")
    chosen, rejected = row["chosen"], row["rejected"]
    if prompt is None:
        prompt, chosen, rejected = _split_implicit_prompt(chosen, rejected)
    margin = row.get("margin")
    return PreferencePair(
        prompt=prompt, chosen=chosen, rejected=rejected, margin=margin, **common
    )


@dataclass(frozen=True)
class PreferenceDataset:
    """An immutable collection of preference pairs and unpaired examples."""

    pairs: tuple[PreferencePair, ...] = ()
    unpaired: tuple[UnpairedExample, ...] = ()

    def __post_init__(self) -> None:
        pairs = tuple(self.pairs)
        unpaired = tuple(self.unpaired)
        if any(not isinstance(item, PreferencePair) for item in pairs):
            raise TypeError("pairs must contain PreferencePair records")
        if any(not isinstance(item, UnpairedExample) for item in unpaired):
            raise TypeError("unpaired must contain UnpairedExample records")
        object.__setattr__(self, "pairs", pairs)
        object.__setattr__(self, "unpaired", unpaired)

    def __len__(self) -> int:
        return len(self.pairs) + len(self.unpaired)

    @property
    def records(self) -> tuple[PreferenceRecord, ...]:
        return (*self.pairs, *self.unpaired)

    @classmethod
    def from_records(
        cls, records: Iterable[PreferenceRecord | Mapping[str, Any]]
    ) -> PreferenceDataset:
        """Build a dataset from records or JSON-like mappings."""
        pairs: list[PreferencePair] = []
        unpaired: list[UnpairedExample] = []
        for index, record in enumerate(records):
            if isinstance(record, Mapping):
                try:
                    record = parse_preference_record(record)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"record {index}: {exc}") from exc
            if isinstance(record, PreferencePair):
                pairs.append(record)
            elif isinstance(record, UnpairedExample):
                unpaired.append(record)
            else:
                raise TypeError(
                    f"record {index} is not a preference record: {type(record)!r}"
                )
        return cls(tuple(pairs), tuple(unpaired))

    @classmethod
    def from_jsonl(
        cls, path: str | Path, *, expected_split: str | None = None
    ) -> PreferenceDataset:
        return load_preference_jsonl(path, expected_split=expected_split)

    def kto_examples(self) -> tuple[UnpairedExample, ...]:
        """Unpaired examples, with every pair split into a desirable and an
        undesirable example (TRL's ``unpair_preference_dataset``)."""
        examples = list(self.unpaired)
        for pair in self.pairs:
            for completion, label in ((pair.chosen, True), (pair.rejected, False)):
                examples.append(
                    UnpairedExample(
                        prompt=pair.prompt,
                        completion=completion,
                        label=label,
                        weight=pair.weight,
                        group_id=pair.group_id,
                        source_id=pair.source_id,
                        metadata=pair.metadata,
                    )
                )
        return tuple(examples)

    def split(
        self, validation_fraction: float = 0.1, seed: int = 42
    ) -> tuple[PreferenceDataset, PreferenceDataset]:
        return split_preference_dataset(
            self, validation_fraction=validation_fraction, seed=seed
        )

    def summary(self) -> dict[str, Any]:
        """Counts for logs and dry runs."""
        return {
            "pairs": len(self.pairs),
            "unpaired": len(self.unpaired),
            "unpaired_desirable": sum(1 for item in self.unpaired if item.label),
            "prompt_groups": len(group_preference_records(self.records)),
        }


def load_preference_jsonl(
    path: str | Path, *, expected_split: str | None = None
) -> PreferenceDataset:
    """Read strict UTF-8 preference JSONL, optionally enforcing its split role."""
    validate_finetuning_split({}, expected_split=expected_split)
    records: list[PreferenceRecord] = []
    for number, line in read_jsonl_lines(path):
        try:
            row = parse_finetuning_json(line)
            validate_finetuning_split(row, expected_split=expected_split)
            records.append(parse_preference_record(row))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}: line {number}: {exc}") from exc
    if not records:
        raise ValueError(f"{path}: dataset is empty")
    return PreferenceDataset.from_records(records)


def write_preference_jsonl(dataset: PreferenceDataset, path: str | Path) -> Path:
    """Write ``dataset`` as JSONL that :func:`load_preference_jsonl` reads back."""
    target = Path(path)
    lines = [
        json.dumps(record.to_dict(), ensure_ascii=False, allow_nan=False)
        for record in dataset.records
    ]
    target.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return target


def _identity_keys(record: PreferenceRecord) -> set[str]:
    keys = {"prompt:" + prompt_identity(record.identity_text)}
    if record.group_id is not None:
        keys.add("group:" + record.group_id)
    if record.source_id is not None:
        keys.add("source:" + record.source_id)
    return keys


def _record_key(record: PreferenceRecord) -> str:
    kind = "pair" if isinstance(record, PreferencePair) else "unpaired"
    return _canonical({"kind": kind, **record.to_dict()})


def group_preference_records(
    records: Iterable[PreferenceRecord],
) -> list[list[PreferenceRecord]]:
    """Deduplicate records and group those transitively sharing a request
    identity, ``group_id`` or ``source_id``. Input order does not matter."""
    unique: dict[str, PreferenceRecord] = {}
    for record in records:
        unique.setdefault(_record_key(record), record)
    ordered = [unique[key] for key in sorted(unique)]
    parents = list(range(len(ordered)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owners: dict[str, int] = {}
    for index, record in enumerate(ordered):
        for key in sorted(_identity_keys(record)):
            if key in owners:
                parents[root(index)] = root(owners[key])
            owners[key] = index
    groups: dict[int, list[PreferenceRecord]] = {}
    for index, record in enumerate(ordered):
        groups.setdefault(root(index), []).append(record)
    return list(groups.values())


def check_preference_overlap(
    train: PreferenceDataset, evaluation: PreferenceDataset
) -> None:
    """Reject a shared request identity, group or source across two splits."""
    train_keys: set[str] = set()
    for record in train.records:
        train_keys |= _identity_keys(record)
    shared = {
        key
        for record in evaluation.records
        for key in _identity_keys(record)
        if key in train_keys
    }
    if shared:
        raise ValueError(
            f"Train/evaluation overlap: {len(shared)} shared prompt, group_id "
            "or source_id identities"
        )


def split_preference_dataset(
    dataset: PreferenceDataset, validation_fraction: float = 0.1, seed: int = 42
) -> tuple[PreferenceDataset, PreferenceDataset]:
    """Deterministically split connected prompt/source groups.

    The fraction applies to groups, so the record fraction may differ. Exact
    duplicate records are removed. The result does not depend on input order,
    and the two splits never share a request identity, group or source.
    """
    if (
        isinstance(validation_fraction, bool)
        or not isinstance(validation_fraction, int | float)
        or not math.isfinite(validation_fraction)
        or not 0 < validation_fraction < 1
    ):
        raise ValueError("validation_fraction must be between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    groups = group_preference_records(dataset.records)
    if len(groups) < 2:
        raise ValueError("At least two independent prompt/source groups are required")
    duplicates = len(dataset) - sum(len(group) for group in groups)
    if duplicates:
        logger.info("Dropped %d duplicate preference records", duplicates)
    ranked = sorted(
        groups,
        key=lambda group: hashlib.sha256(
            f"{seed}:{_canonical([_record_key(r) for r in group])}".encode()
        ).hexdigest(),
    )
    count = max(1, min(len(ranked) - 1, round(len(ranked) * validation_fraction)))
    evaluation = PreferenceDataset.from_records(
        record for group in ranked[:count] for record in group
    )
    train = PreferenceDataset.from_records(
        record for group in ranked[count:] for record in group
    )
    check_preference_overlap(train, evaluation)
    return train, evaluation


def as_preference_dataset(
    data: (
        PreferenceDataset | str | Path | Iterable[PreferenceRecord | Mapping[str, Any]]
    ),
) -> PreferenceDataset:
    """Accept a dataset, a JSONL path, or records/mappings."""
    if isinstance(data, PreferenceDataset):
        return data
    if isinstance(data, str | Path):
        return load_preference_jsonl(data)
    return PreferenceDataset.from_records(data)


def resolve_splits(
    train_data: Any, eval_data: Any, validation_fraction: float, seed: int
) -> tuple[PreferenceDataset, PreferenceDataset | None]:
    """Training and held-out data that never share a prompt identity.

    Explicit ``eval_data`` is checked for overlap; otherwise a positive
    ``validation_fraction`` holds out prompt-disjoint groups.
    """
    train = as_preference_dataset(train_data)
    if eval_data is not None:
        evaluation = as_preference_dataset(eval_data)
        check_preference_overlap(train, evaluation)
        return train, evaluation
    if validation_fraction > 0:
        return split_preference_dataset(train, validation_fraction, seed)
    return train, None
