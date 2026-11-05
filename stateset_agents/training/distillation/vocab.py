"""How a teacher's vocabulary lines up with the student's.

Distillation compares two distributions over tokens, so the two models must
agree on what each logit column means. :func:`resolve_vocab_mapping` either
proves they do (``strict``: identical tokenizer vocabularies and logit
widths) or builds an explicit token-string alignment (``shared``): the loss
then runs over the tokens both vocabularies contain, each side renormalized
over that shared support, and the teacher reads the same text through its own
ids. Nothing is silently truncated: any other mismatch raises
:class:`VocabularyMismatchError`, and a sequence holding a token outside the
shared support raises instead of being dropped.

torch is fetched lazily, so importing this module never requires torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..trainer_utils import require_torch
from .losses import VocabularyMismatchError

__all__ = [
    "VocabMapping",
    "VocabularyMismatchError",
    "logit_width",
    "resolve_vocab_mapping",
]


def logit_width(model: Any) -> int | None:
    """The number of logits ``model`` produces per position, if knowable."""
    get_output = getattr(model, "get_output_embeddings", None)
    head = get_output() if callable(get_output) else None
    weight = getattr(head, "weight", None)
    shape = getattr(weight, "shape", None)
    if shape is not None and len(shape) == 2:
        return int(shape[0])
    value = getattr(getattr(model, "config", None), "vocab_size", None)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _vocab(tokenizer: Any, name: str) -> dict[str, int]:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise VocabularyMismatchError(
            f"the {name} tokenizer has no get_vocab(); its vocabulary cannot be "
            "compared"
        )
    vocab = get_vocab()
    return {str(token): int(index) for token, index in dict(vocab).items()}


@dataclass(frozen=True)
class VocabMapping:
    """Student logit columns and the teacher columns that mean the same token.

    ``identity`` mappings (the ``strict`` case) restrict nothing.
    """

    student_ids: tuple[int, ...]
    teacher_ids: tuple[int, ...]
    student_width: int | None
    teacher_width: int | None
    identity: bool
    _tables: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def strict(cls, width: int | None) -> VocabMapping:
        ids = tuple(range(width)) if width is not None else ()
        return cls(ids, ids, width, width, True)

    @property
    def support_size(self) -> int | None:
        return len(self.student_ids) if self.student_ids else self.student_width

    def stats(self) -> dict[str, int | None]:
        shared = len(self.student_ids) if self.student_ids else None
        return {
            "shared_tokens": shared,
            "student_width": self.student_width,
            "teacher_width": self.teacher_width,
            "student_only": (
                None
                if shared is None or self.student_width is None
                else self.student_width - shared
            ),
            "teacher_only": (
                None
                if shared is None or self.teacher_width is None
                else self.teacher_width - shared
            ),
        }

    def _table(self, name: str, device: Any) -> Any:
        key = f"{name}:{device}"
        if key not in self._tables:
            torch = require_torch()
            if name == "student_cols":
                table = torch.tensor(self.student_ids, dtype=torch.long)
            elif name == "teacher_cols":
                table = torch.tensor(self.teacher_ids, dtype=torch.long)
            else:  # student id -> support index / teacher id, -1 when absent
                size = max(self.student_ids) + 1
                table = torch.full((size,), -1, dtype=torch.long)
                values = (
                    range(len(self.student_ids))
                    if name == "support"
                    else (self.teacher_ids)
                )
                table[torch.tensor(self.student_ids)] = torch.tensor(
                    list(values), dtype=torch.long
                )
            self._tables[key] = table.to(device)
        return self._tables[key]

    def student_support(self, logits: Any) -> Any:
        """Student logits ``[N, V_s]`` restricted to the shared support."""
        if self.identity:
            return logits
        return logits.index_select(-1, self._table("student_cols", logits.device))

    def teacher_support(self, logits: Any) -> Any:
        """Teacher logits ``[N, V_t]`` restricted to the shared support, in the
        student's column order."""
        if self.identity:
            return logits
        return logits.index_select(-1, self._table("teacher_cols", logits.device))

    def _lookup(self, name: str, ids: Any, what: str) -> Any:
        table = self._table(name, ids.device)
        if int(ids.max()) >= table.shape[0] or int(ids.min()) < 0:
            raise VocabularyMismatchError(f"{what} holds a token outside the support")
        mapped = table[ids]
        if bool((mapped < 0).any()):
            missing = sorted({int(t) for t in ids[mapped < 0].tolist()})[:5]
            raise VocabularyMismatchError(
                f"{what} holds student token ids {missing} that the teacher's "
                "vocabulary lacks"
            )
        return mapped

    def support_index(self, ids: Any) -> Any:
        """Student token ids -> their column in the shared support."""
        if self.identity:
            return ids
        return self._lookup("support", ids, "a target sequence")

    def translate(self, ids: Any, attention_mask: Any, pad_id: int) -> Any:
        """Student-tokenized ``ids`` as teacher ids (padding -> ``pad_id``)."""
        if self.identity:
            return ids
        torch = require_torch()
        attended = attention_mask.bool()
        result = torch.full_like(ids, int(pad_id))
        result[attended] = self._lookup("teacher", ids[attended], "a sequence")
        return result

    def unsupported_student_ids(self) -> list[int]:
        """Student ids outside the shared support (suppressed when sampling)."""
        if self.identity or self.student_width is None:
            return []
        supported = set(self.student_ids)
        return [i for i in range(self.student_width) if i not in supported]


def resolve_vocab_mapping(
    student: Any,
    teacher: Any,
    tokenizer: Any,
    teacher_tokenizer: Any = None,
    *,
    alignment: str = "strict",
) -> VocabMapping:
    """Check (``strict``) or build (``shared``) the teacher/student alignment.

    ``strict`` rejects tokenizers whose token-to-id maps differ and models
    whose logit widths differ (common for padded embedding matrices: use
    ``shared``). ``shared`` pairs every token string both tokenizers know
    whose id is a real logit column in both models.
    """
    student_width = logit_width(student)
    teacher_width = logit_width(teacher) if teacher is not None else None
    if alignment == "strict":
        if teacher_tokenizer is not None and teacher_tokenizer is not tokenizer:
            if _vocab(tokenizer, "student") != _vocab(teacher_tokenizer, "teacher"):
                raise VocabularyMismatchError(
                    "the teacher and student tokenizers have different "
                    "vocabularies; distil across them with "
                    "vocab_alignment='shared' (token-string alignment)"
                )
        if (
            student_width is not None
            and teacher_width is not None
            and student_width != teacher_width
        ):
            raise VocabularyMismatchError(
                f"the student has {student_width} logits per position and the "
                f"teacher {teacher_width}; with the same tokenizer this is "
                "usually embedding padding, so use vocab_alignment='shared'"
            )
        return VocabMapping.strict(student_width or teacher_width)
    if alignment != "shared":
        raise ValueError(f"unknown vocab_alignment {alignment!r}")
    student_vocab = _vocab(tokenizer, "student")
    teacher_vocab = _vocab(teacher_tokenizer or tokenizer, "teacher")
    pairs = sorted(
        (index, teacher_vocab[token])
        for token, index in student_vocab.items()
        if token in teacher_vocab
        and (student_width is None or index < student_width)
        and (teacher_width is None or teacher_vocab[token] < teacher_width)
    )
    if not pairs:
        raise VocabularyMismatchError("the tokenizers share no tokens")
    student_ids = tuple(pair[0] for pair in pairs)
    teacher_ids = tuple(pair[1] for pair in pairs)
    if len(set(student_ids)) != len(pairs) or len(set(teacher_ids)) != len(pairs):
        raise VocabularyMismatchError(
            "the shared tokens do not map one-to-one between the vocabularies"
        )
    identity = (
        student_ids == teacher_ids
        and student_width == teacher_width == len(pairs)
        and student_ids == tuple(range(len(pairs)))
    )
    return VocabMapping(
        student_ids, teacher_ids, student_width, teacher_width, identity
    )
