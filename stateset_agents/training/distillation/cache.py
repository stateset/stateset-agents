"""A precomputed top-k teacher cache for offline distillation.

Offline (teacher-forced) distillation scores fixed completions, so the
teacher's distribution at every response position can be computed once and
the teacher released. :func:`build_teacher_cache` stores the teacher's top
``k`` token ids and log-probs (at the distillation temperature, over the
shared support of the vocabulary mapping) for every encoded sequence, keyed
by a hash of its token ids; the trainer then needs no teacher at all
(``teacher_top_k`` in :class:`DistillationConfig`). The loss against a cache
is :func:`~.losses.topk_divergence` (top ``k`` plus a remaining-mass bucket).

Caches save with ``torch.save`` of plain tensors and containers and load
through ``core.checkpoint_io`` (weights-only unless ``trusted=True``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stateset_agents.core.checkpoint_io import load_checkpoint_file

from ..preference import _loop
from ..preference.encoding import EncodedSequence, PreferenceEncoder
from ..trainer_utils import require_torch
from .losses import topk_logprobs
from .vocab import VocabMapping

__all__ = ["CACHE_SCHEMA", "TeacherCache", "build_teacher_cache", "sequence_key"]

CACHE_SCHEMA = "stateset-teacher-topk-cache-v1"


def sequence_key(sequence: EncodedSequence) -> str:
    """A content key for an encoded sequence (prompt and response ids)."""
    payload = json.dumps([list(sequence.prompt_ids), list(sequence.response_ids)])
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class TeacherCache:
    """Top-``k`` teacher ``(ids, log-probs)`` ``[response tokens, k]`` per
    sequence, with the settings they were computed under."""

    top_k: int
    temperature: float
    support: tuple[int, ...]
    encoder: dict[str, Any]
    entries: dict[str, tuple[Any, Any]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.entries)

    def lookup(self, sequence: EncodedSequence) -> tuple[Any, Any]:
        entry = self.entries.get(sequence_key(sequence))
        if entry is None:
            raise ValueError(
                "the teacher cache has no entry for a training sequence; rebuild "
                "it from the same data, tokenizer and encoder settings"
            )
        if entry[0].shape[0] != len(sequence.response_ids):
            raise ValueError("a teacher cache entry does not match its sequence")
        return entry

    def check_compatible(
        self,
        *,
        top_k: int,
        temperature: float,
        encoder: dict[str, Any],
        support: Sequence[int],
    ) -> None:
        """Reject a cache built under different settings."""
        problems = []
        if self.top_k != top_k:
            problems.append(f"top_k {self.top_k} != {top_k}")
        if self.temperature != float(temperature):
            problems.append(f"temperature {self.temperature} != {temperature}")
        if self.encoder != encoder:
            problems.append("encoder settings differ")
        if tuple(self.support) != tuple(support):
            problems.append("vocabulary alignment differs")
        if problems:
            raise ValueError("incompatible teacher cache: " + "; ".join(problems))

    def save(self, path: str | Path) -> Path:
        torch = require_torch()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self.entries)
        torch.save(
            {
                "schema": CACHE_SCHEMA,
                "top_k": int(self.top_k),
                "temperature": float(self.temperature),
                "support": torch.tensor(self.support, dtype=torch.long),
                "encoder": json.dumps(self.encoder, sort_keys=True),
                "keys": keys,
                "ids": [self.entries[key][0] for key in keys],
                "logps": [self.entries[key][1] for key in keys],
            },
            target,
        )
        return target

    @classmethod
    def load(cls, path: str | Path, *, trusted: bool = False) -> TeacherCache:
        torch = require_torch()
        state = load_checkpoint_file(Path(path), trusted=trusted, torch_module=torch)
        if not isinstance(state, dict) or state.get("schema") != CACHE_SCHEMA:
            raise ValueError(f"{path} is not a teacher top-k cache")
        keys, ids, logps = state["keys"], state["ids"], state["logps"]
        if not len(keys) == len(ids) == len(logps):
            raise ValueError(f"{path} is a corrupt teacher cache")
        entries = {}
        for key, index, value in zip(keys, ids, logps, strict=True):
            if index.shape != value.shape or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{path} holds an invalid cache entry")
            entries[str(key)] = (index.long(), value.float())
        return cls(
            top_k=int(state["top_k"]),
            temperature=float(state["temperature"]),
            support=tuple(int(i) for i in state["support"].tolist()),
            encoder=json.loads(state["encoder"]),
            entries=entries,
        )


def teacher_response_logits(
    teacher: Any,
    mapping: VocabMapping,
    sequences: Sequence[EncodedSequence],
    *,
    pad_id: int,
    teacher_pad_id: int,
    device: Any,
    mixed_precision: str = "no",
) -> Any:
    """Teacher logits ``[N, support]`` at every response position of
    ``sequences`` (row-major: sequence by sequence, position by position),
    restricted to the shared support. Gradients are not tracked."""
    torch = require_torch()
    ids, attention, response = _loop.collate(sequences, pad_id, device)
    mask = response[:, 1:].bool()
    teacher_ids = mapping.translate(ids, attention, teacher_pad_id)
    with torch.no_grad(), _loop.autocast(device, mixed_precision):
        outputs = _loop.model_outputs(teacher, teacher_ids, attention)
        logits = _loop.logits_of(outputs)[:, :-1][mask]
    return mapping.teacher_support(logits)


def build_teacher_cache(
    teacher: Any,
    encoder: PreferenceEncoder,
    mapping: VocabMapping,
    sequences: Sequence[EncodedSequence],
    *,
    top_k: int,
    temperature: float,
    batch_size: int,
    device: Any,
    teacher_pad_id: int | None = None,
    mixed_precision: str = "no",
) -> TeacherCache:
    """Score ``sequences`` with the (frozen, eval-mode) ``teacher`` once."""
    cache = TeacherCache(
        top_k=int(top_k),
        temperature=float(temperature),
        support=mapping.student_ids if not mapping.identity else (),
        encoder=encoder.settings(),
    )
    unique = {sequence_key(s): s for s in sequences}
    ordered = list(unique.items())
    pad = encoder.pad_token_id
    with _loop.evaluation_mode(teacher):
        for start in range(0, len(ordered), batch_size):
            chunk = ordered[start : start + batch_size]
            logits = teacher_response_logits(
                teacher,
                mapping,
                [sequence for _, sequence in chunk],
                pad_id=pad,
                teacher_pad_id=pad if teacher_pad_id is None else teacher_pad_id,
                device=device,
                mixed_precision=mixed_precision,
            )
            ids, logps = topk_logprobs(logits, k=top_k, temperature=temperature)
            offset = 0
            for key, sequence in chunk:
                rows = len(sequence.response_ids)
                cache.entries[key] = (
                    ids[offset : offset + rows].detach().cpu(),
                    logps[offset : offset + rows].detach().float().cpu(),
                )
                offset += rows
    return cache
