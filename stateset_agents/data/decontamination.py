"""Find eval-set text in training data: exact n-gram overlap and near duplicates.

A model trained on text from its own eval set measures recall, not skill.
This module compares training data (SFT chat rows, preference pairs, RL
prompt files) with one or more eval sets and reports, per eval set, which
rows are contaminated and the evidence: the overlapping n-grams and the
training rows they came from.

Matching follows the convention of the GPT-3 and lm-evaluation-harness
decontamination studies:

* **Exact n-grams.** Text is normalized (NFKC, casefold, punctuation and
  underscores become spaces, whitespace collapses; optionally every digit run
  becomes ``0``) and split into words. An eval row is contaminated when one of
  its 13-word n-grams (``ngram_size``) appears in any training row. An eval
  segment shorter than ``ngram_size`` but at least ``min_tokens`` words long
  is matched whole; a segment shorter than that (but at least
  ``duplicate_min_tokens`` words) only matches an identical training segment,
  which still catches duplicated short benchmark prompts. Shorter segments are
  too generic to be evidence and are counted as unchecked rather than
  silently passed.
* **Near duplicates (optional).** MinHash signatures over word shingles,
  bucketed with locality-sensitive hashing, find candidate pairs of rows;
  each candidate is then verified with the exact Jaccard similarity of the
  two rows' shingle sets, so the reported pairs and scores are exact. This
  catches paraphrases that change a word every few tokens, which no 13-gram
  survives. Rows are compared whole, so one shared boilerplate turn does not
  make two different conversations near duplicates.

Text is compared per *segment* (one message, prompt or answer), never across
segment boundaries. System and developer messages are skipped by default:
a fixed instruction shared by training and eval rows is not leaked data, and
including it would mark every row.

Everything is deterministic: hashing uses CRC-32 and fixed MinHash
coefficients, and every list in the report follows input order. Results
depend only on the inputs and the configuration.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import unicodedata
import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from itertools import accumulate
from pathlib import Path
from typing import Any, Literal

from stateset_agents.data.finetuning import parse_finetuning_json
from stateset_agents.data.jsonl import jsonl_lines

__all__ = [
    "BUILTIN_EVAL_SETS",
    "DEFAULT_ROLES",
    "LSH_RECALL",
    "DecontaminationConfig",
    "DecontaminationResult",
    "TextRecord",
    "TextSource",
    "apply_decontamination",
    "builtin_eval_set",
    "extract_segments",
    "find_contamination",
    "load_text_source",
    "lsh_shape",
    "normalize_text",
    "read_rows",
    "resolve_eval_source",
    "text_source_from_rows",
]

SCHEMA_VERSION = 1

#: Message roles compared by default. System and developer prompts are shared
#: instructions; a fixed one in training and eval would match every row.
DEFAULT_ROLES: tuple[str, ...] = ("user", "assistant", "tool")

#: Row keys read as text, in this order, when no ``fields`` are given. Covers
#: chat/SFT rows (``messages``), preference pairs (``prompt``/``chosen``/
#: ``rejected``, ``completion``), RL prompt files (``prompt``, ``question``,
#: ``user_query``, ``turns``), scenario files (``user_turns``,
#: ``expected_resolution``) and question/answer benchmarks.
TEXT_KEYS: tuple[str, ...] = (
    "messages",
    "conversation",
    "prompt",
    "question",
    "query",
    "user_query",
    "instruction",
    "input",
    "problem",
    "context",
    "turns",
    "user_turns",
    "chosen",
    "rejected",
    "completion",
    "response",
    "answer",
    "output",
    "solution",
    "expected_resolution",
    "text",
)

#: Keys of a JSON (not JSONL) file that may hold its list of rows.
_ROW_LIST_KEYS = ("rows", "examples", "data", "tasks", "scenarios", "cases", "items")

_NON_WORD = re.compile(r"[\W_]+")
_DIGITS = re.compile(r"\d+")
#: ASCII characters that ``_NON_WORD`` would replace (everything but [a-z0-9]).
_ASCII_SEPARATORS = {c: " " for c in range(128) if not chr(c).isalnum()}

Action = Literal["filter", "flag"]


@dataclass(frozen=True)
class DecontaminationConfig:
    """How text is normalized and matched."""

    ngram_size: int = 13
    #: Eval segments with fewer than ``ngram_size`` words but at least this
    #: many are matched whole, anywhere inside a training segment. Equal to
    #: ``ngram_size`` disables that.
    min_tokens: int = 8
    #: Eval segments shorter than ``min_tokens`` but at least this long only
    #: match a training segment that is the same text in full (an exact
    #: duplicate prompt). At ``min_tokens`` or above this check is off.
    duplicate_min_tokens: int = 3
    #: Replace every digit run with ``0`` before matching (templated sets whose
    #: rows differ only in reference numbers).
    mask_digits: bool = False
    roles: tuple[str, ...] = DEFAULT_ROLES
    #: Row keys to read instead of :data:`TEXT_KEYS`.
    fields: tuple[str, ...] | None = None
    #: An eval row is contaminated when at least this fraction of its n-grams
    #: occurs in training data; ``0.0`` means any single n-gram.
    min_overlap: float = 0.0
    near_duplicates: bool = False
    #: Minimum exact Jaccard similarity of two segments' shingle sets.
    near_duplicate_threshold: float = 0.8
    shingle_size: int = 3
    #: MinHash functions; LSH bands and rows per band follow from the threshold
    #: (see :func:`lsh_shape`) so candidates are found with >= 99% probability.
    minhash_permutations: int = 120
    minhash_seed: int = 13
    #: Overlapping n-grams and training rows listed per contaminated eval row.
    max_evidence: int = 5

    def __post_init__(self) -> None:
        for name, low, high in (
            ("ngram_size", 1, 64),
            ("min_tokens", 1, 64),
            ("duplicate_min_tokens", 1, 64),
            ("shingle_size", 1, 16),
            ("minhash_permutations", 16, 1024),
            ("max_evidence", 1, 1000),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        if self.min_tokens > self.ngram_size:
            raise ValueError("min_tokens cannot exceed ngram_size")
        if type(self.minhash_seed) is not int:
            raise ValueError("minhash_seed must be an integer")
        for name, low_open in (
            ("min_overlap", False),
            ("near_duplicate_threshold", True),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")
            if not (0.0 < value <= 1.0 if low_open else 0.0 <= value <= 1.0):
                bound = "(0, 1]" if low_open else "[0, 1]"
                raise ValueError(f"{name} must be in {bound}")
        roles = tuple(self.roles)
        if not roles or any(not isinstance(r, str) or not r for r in roles):
            raise ValueError("roles must be nonempty strings")
        object.__setattr__(self, "roles", roles)
        if self.fields is not None:
            fields = tuple(self.fields)
            if not fields or any(not isinstance(f, str) or not f for f in fields):
                raise ValueError("fields must be nonempty strings")
            object.__setattr__(self, "fields", fields)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready settings, recorded in every report."""
        data = asdict(self)
        data["roles"] = list(self.roles)
        data["fields"] = None if self.fields is None else list(self.fields)
        if self.near_duplicates:
            bands, rows = lsh_shape(
                self.near_duplicate_threshold, self.minhash_permutations
            )
            data["lsh_bands"], data["lsh_rows"] = bands, rows
        return data


#: Probability that a pair exactly at the near-duplicate threshold becomes an
#: LSH candidate. Pairs above the threshold are found more often still.
LSH_RECALL = 0.99


def lsh_shape(threshold: float, permutations: int) -> tuple[int, int]:
    """``(bands, rows)`` with the most rows that still meet :data:`LSH_RECALL`.

    A pair with Jaccard ``j`` shares at least one band with probability
    ``1 - (1 - j**rows) ** bands``. More rows per band mean fewer false
    candidates (less verification work); this keeps as many as the recall
    target allows at ``threshold``.
    """
    for rows in range(permutations, 0, -1):
        bands = permutations // rows
        if 1 - (1 - threshold**rows) ** bands >= LSH_RECALL:
            return bands, rows
    return permutations, 1


@dataclass(frozen=True)
class TextRecord:
    """One row's comparable text: its id and one segment per message or field."""

    record_id: str
    segments: tuple[str, ...]


@dataclass(frozen=True)
class TextSource:
    """A named set of records and the identity of the bytes they came from."""

    name: str
    records: tuple[TextRecord, ...]
    path: str | None = None
    sha256: str | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "sha256": self.sha256,
            "rows": len(self.records),
            "rows_without_text": sum(not r.segments for r in self.records),
        }


def normalize_text(text: str, *, mask_digits: bool = False) -> str:
    """Words of ``text`` under NFKC, casefold, punctuation-as-space, one space."""
    if text.isascii():  # Same result, without the Unicode regex machinery.
        folded = text.lower().translate(_ASCII_SEPARATORS)
    else:
        folded = _NON_WORD.sub(" ", unicodedata.normalize("NFKC", text).casefold())
    if mask_digits:
        folded = _DIGITS.sub("0", folded)
    return " ".join(folded.split())


# ---------------------------------------------------------------------------
# Reading rows and extracting text
# ---------------------------------------------------------------------------


def read_rows(path: str | Path) -> tuple[list[tuple[str, Any]], str]:
    """``([(row id, row), ...], sha256)`` for a JSONL file or a JSON file of rows.

    JSONL rows are identified as ``<file>:<line>``; a JSON file may hold a list
    of rows, or an object whose ``rows``/``examples``/``data``/``tasks``/
    ``scenarios``/``cases``/``items`` key is that list (``<file>[index]``).
    Invalid JSON, NaN and duplicate keys raise ``ValueError`` with the line.
    """
    source = Path(path)
    data = source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    text = data.decode("utf-8")
    if source.suffix.lower() == ".json":
        try:
            value = parse_finetuning_json(text)
        except ValueError as exc:
            raise ValueError(f"{source}: invalid JSON: {exc}") from None
        if isinstance(value, dict):
            lists = [key for key in _ROW_LIST_KEYS if isinstance(value.get(key), list)]
            if len(lists) != 1:
                raise ValueError(
                    f"{source}: a JSON file must be a list of rows or hold exactly "
                    f"one of {', '.join(_ROW_LIST_KEYS)} as a list"
                )
            value = value[lists[0]]
        if not isinstance(value, list):
            raise ValueError(f"{source}: a JSON file must be a list of rows")
        return [(f"{source.name}[{i}]", row) for i, row in enumerate(value)], digest
    rows: list[tuple[str, Any]] = []
    for number, line in jsonl_lines(text):
        try:
            rows.append((f"{source.name}:{number}", parse_finetuning_json(line)))
        except ValueError as exc:
            raise ValueError(f"{source}:{number}: invalid JSON record: {exc}") from None
    return rows, digest


def _content_text(content: Any) -> list[str]:
    """Text of a message ``content``: a string or a list of text parts."""
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return parts
    return []


def _value_text(value: Any, roles: frozenset[str]) -> list[str]:
    """Segments of one field: a string, a message, or a list of either."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        if "role" in value:
            return _content_text(value.get("content")) if value["role"] in roles else []
        return _content_text(value.get("content"))
    if isinstance(value, list):
        segments: list[str] = []
        for item in value:
            if isinstance(item, (str, dict)):
                segments.extend(_value_text(item, roles))
        return segments
    return []


def extract_segments(
    row: Any,
    *,
    roles: Sequence[str] = DEFAULT_ROLES,
    fields: Sequence[str] | None = None,
) -> list[str]:
    """Text segments of one row, in a fixed order, without empty strings.

    A bare string is one segment; a bare list is read as messages. For an
    object, ``fields`` (or :data:`TEXT_KEYS`) are read in order; message
    lists keep only the given ``roles``.
    """
    allowed = frozenset(roles)
    if isinstance(row, (str, list)):
        found = _value_text(row, allowed)
    elif isinstance(row, dict):
        found = []
        for key in fields if fields is not None else TEXT_KEYS:
            if key in row:
                found.extend(_value_text(row[key], allowed))
    else:
        found = []
    return [segment for segment in found if segment.strip()]


def text_source_from_rows(
    name: str,
    rows: Iterable[tuple[str, Any]],
    *,
    config: DecontaminationConfig | None = None,
    path: str | None = None,
    sha256: str | None = None,
) -> TextSource:
    """Build a :class:`TextSource` from ``(row id, row)`` pairs."""
    config = config or DecontaminationConfig()
    records = tuple(
        TextRecord(
            record_id,
            tuple(extract_segments(row, roles=config.roles, fields=config.fields)),
        )
        for record_id, row in rows
    )
    return TextSource(name=name, records=records, path=path, sha256=sha256)


def load_text_source(
    path: str | Path,
    *,
    name: str | None = None,
    config: DecontaminationConfig | None = None,
) -> TextSource:
    """Read a JSONL/JSON file into a :class:`TextSource` named after the file."""
    rows, digest = read_rows(path)
    return text_source_from_rows(
        name or Path(path).name, rows, config=config, path=str(path), sha256=digest
    )


def _support_bench_rows() -> list[tuple[str, Any]]:
    from stateset_agents.data.customer_support_bench import load_support_scenarios

    return [
        (f"support-bench[{i}]", {"user_query": s.user_query})
        for i, s in enumerate(load_support_scenarios())
    ]


def _tool_calling_bench_rows() -> list[tuple[str, Any]]:
    from stateset_agents.data.tool_calling_bench import load_tool_call_scenarios

    return [
        (f"tool-calling-bench[{i}]", {"user_query": s.user_query})
        for i, s in enumerate(load_tool_call_scenarios())
    ]


def _gsm8k_rows(split: str) -> list[tuple[str, Any]]:
    from stateset_agents.data.gsm8k import load_gsm8k

    examples = load_gsm8k(split=split)
    if not isinstance(examples, list):  # pragma: no cover - named split
        raise TypeError("load_gsm8k(split=...) must return a list")
    return [
        (f"gsm8k-{split}[{i}]", {"question": e.question, "answer": e.answer_text})
        for i, e in enumerate(examples)
    ]


#: Eval sets that ship with (or are pinned by) the package, by name. GSM8K
#: needs the ``datasets`` library and the pinned revision in the local cache
#: or network access.
BUILTIN_EVAL_SETS: dict[str, str] = {
    "support-bench": "24 bundled customer-support queries (data.customer_support_bench)",
    "tool-calling-bench": "8 bundled tool-calling queries (data.tool_calling_bench)",
    "gsm8k-test": "GSM8K test split at the pinned revision (question and answer)",
    "gsm8k-train": "GSM8K train split at the pinned revision (question and answer)",
}


def builtin_eval_set(
    name: str, *, config: DecontaminationConfig | None = None
) -> TextSource:
    """One of :data:`BUILTIN_EVAL_SETS` as a :class:`TextSource`."""
    loaders = {
        "support-bench": _support_bench_rows,
        "tool-calling-bench": _tool_calling_bench_rows,
        "gsm8k-test": lambda: _gsm8k_rows("test"),
        "gsm8k-train": lambda: _gsm8k_rows("train"),
    }
    if name not in loaders:
        raise ValueError(
            f"Unknown built-in eval set {name!r}; choose from {', '.join(loaders)}"
        )
    rows = loaders[name]()
    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return text_source_from_rows(
        name, rows, config=config, path=f"builtin:{name}", sha256=digest
    )


def resolve_eval_source(
    spec: str, *, config: DecontaminationConfig | None = None
) -> TextSource:
    """A source (eval or training) from ``builtin:NAME``, ``NAME=PATH`` or a path.

    A plain path is named after its file; ``NAME=PATH`` names it explicitly
    (two files with the same name in different directories need this).
    """
    if spec.startswith("builtin:"):
        return builtin_eval_set(spec[len("builtin:") :], config=config)
    name, separator, path = spec.partition("=")
    if separator and name and path and not Path(spec).exists():
        return load_text_source(path, name=name, config=config)
    return load_text_source(spec, config=config)


# ---------------------------------------------------------------------------
# Exact n-gram matching
# ---------------------------------------------------------------------------


@dataclass
class _Segment:
    """A normalized segment: its words and where each word starts."""

    text: str
    words: list[str]
    #: ``bounds[i]`` is where word ``i`` starts; one past the last word ends.
    bounds: list[int]

    @classmethod
    def build(cls, raw: str, mask_digits: bool) -> _Segment:
        text = normalize_text(raw, mask_digits=mask_digits)
        words = text.split(" ") if text else []
        return cls(
            text, words, list(accumulate((len(w) + 1 for w in words), initial=0))
        )

    def __len__(self) -> int:
        return len(self.words)

    def windows(self, size: int) -> list[str]:
        """Every ``size``-word window, in order."""
        text = self.text
        bounds = self.bounds
        # The shorter list ends the zip: one window per starting word that fits.
        return [text[a : b - 1] for a, b in zip(bounds, bounds[size:], strict=False)]

    def window(self, start: int, size: int) -> str:
        return self.text[self.bounds[start] : self.bounds[start + size] - 1]


@dataclass
class _EvalRow:
    source: int
    index: int
    segments: list[_Segment]
    #: The row's matchable keys (n-grams, or whole short segments), in order.
    keys: list[str] = field(default_factory=list)
    short_unchecked: int = 0


@dataclass
class _Matches:
    """Which index keys training rows hit, and which rows hit them."""

    #: key -> training rows (global index) that contain it, capped.
    key_rows: dict[str, list[int]] = field(default_factory=dict)
    #: training row -> keys it contains.
    row_keys: dict[int, set[str]] = field(default_factory=dict)
    #: Near duplicates: best (jaccard, training row) per eval row.
    near: dict[int, tuple[float, int]] = field(default_factory=dict)
    #: Near duplicates: training row -> eval rows it nearly duplicates.
    near_by_train: dict[int, set[int]] = field(default_factory=dict)


class _NgramIndex:
    """Eval n-grams (and whole short segments) mapped to the rows they occur in."""

    def __init__(self, rows: list[_EvalRow], config: DecontaminationConfig) -> None:
        self.n = config.ngram_size
        self.m = config.min_tokens
        self.keys: dict[str, list[int]] = {}
        #: m-word prefix of a short segment -> [(length, whole segment)].
        self.short: dict[str, list[tuple[int, str]]] = {}
        #: Whole texts of very short segments, matched by full equality only.
        self.duplicates: set[str] = set()
        for gid, row in enumerate(rows):
            for segment in row.segments:
                if len(segment) >= self.n:
                    found = segment.windows(self.n)
                elif len(segment) >= self.m:
                    found = [segment.text]
                    entry = (len(segment), segment.text)
                    bucket = self.short.setdefault(segment.windows(self.m)[0], [])
                    if entry not in bucket:
                        bucket.append(entry)
                elif len(segment) >= config.duplicate_min_tokens:
                    found = [segment.text]
                    self.duplicates.add(segment.text)
                else:
                    row.short_unchecked += 1
                    continue
                for key in found:
                    owners = self.keys.setdefault(key, [])
                    if not owners or owners[-1] != gid:
                        owners.append(gid)
                        row.keys.append(key)

    def hits(self, segment: _Segment) -> set[str]:
        """Index keys that occur in a training ``segment``."""
        keys = self.keys
        found: set[str] = set()
        if segment.text in self.duplicates:
            found.add(segment.text)
        if len(segment) >= self.n:
            found.update(k for k in segment.windows(self.n) if k in keys)
        if self.short and len(segment) >= self.m:
            for i, prefix in enumerate(segment.windows(self.m)):
                for length, whole in self.short.get(prefix, ()):
                    if (
                        i + length <= len(segment)
                        and segment.window(i, length) == whole
                    ):
                        found.add(whole)
        return found


# ---------------------------------------------------------------------------
# Near-duplicate matching (MinHash candidates, exact Jaccard verification)
# ---------------------------------------------------------------------------

#: Shingles per vectorized MinHash step: small enough that the
#: shingles x hash-functions matrix stays in cache (measured 6x faster than
#: 50k-shingle steps).
_MINHASH_BATCH = 4096
#: Training shingles queued before candidates are verified.
_VERIFY_BATCH = 65_536


class _WordHashes(dict[str, int]):
    """CRC-32 of each distinct word, computed on first use."""

    def __missing__(self, word: str) -> int:
        value = zlib.crc32(word.encode("utf-8"))
        self[word] = value
        return value


class _NearDuplicates:
    """Row-level MinHash/LSH over word shingles, verified by exact Jaccard.

    A row's shingles are the union of its segments' shingles (shingles never
    span two messages). Words hash with CRC-32, shingles with a fixed random
    linear map, and signatures with multiply-shift hash functions seeded by
    ``minhash_seed``, so candidates depend only on the text. Every candidate
    pair is verified with the exact Jaccard similarity of the two shingle-hash
    sets. Rows are compared whole: a shared boilerplate turn does not make two
    otherwise different conversations near duplicates.
    """

    def __init__(self, eval_rows: list[_EvalRow], config: DecontaminationConfig):
        import numpy as np

        self.np = np
        self.k = config.shingle_size
        self.m = config.min_tokens
        self.bands, self.rows = lsh_shape(
            config.near_duplicate_threshold, config.minhash_permutations
        )
        self.threshold = config.near_duplicate_threshold
        rng = random.Random(config.minhash_seed)

        def odd(count: int) -> Any:
            return np.array([rng.getrandbits(64) | 1 for _ in range(count)], np.uint64)

        self.shingle_mult = odd(self.k)
        self.a = odd(self.bands * self.rows)
        self.b = np.array(
            [rng.getrandbits(64) for _ in range(self.bands * self.rows)], np.uint64
        )
        self.band_mix = odd(self.rows)
        self.word_hashes = _WordHashes()
        #: (eval row, shingle set) per indexed eval row.
        self.indexed: list[tuple[int, set[int]]] = []
        self.buckets: list[dict[int, list[int]]] = [{} for _ in range(self.bands)]
        pending: list[tuple[int, Any]] = []
        for gid, row in enumerate(eval_rows):
            shingles = self.row_shingles(row.segments)
            if shingles is not None:
                pending.append((gid, shingles))
        for (gid, shingles), keys in zip(
            pending, self.band_keys([p[1] for p in pending]), strict=True
        ):
            for band, key in enumerate(keys):
                self.buckets[band].setdefault(key, []).append(len(self.indexed))
            self.indexed.append((gid, set(shingles.tolist())))
        self.pending: list[tuple[int, Any]] = []
        self.pending_size = 0

    def row_shingles(self, segments: Sequence[_Segment]) -> Any | None:
        """All shingle hashes of a row, or None when it has too few words."""
        if sum(len(segment) for segment in segments) < self.m:
            return None
        parts = [self.shingles(segment) for segment in segments if len(segment)]
        return parts[0] if len(parts) == 1 else self.np.concatenate(parts)

    def shingles(self, segment: _Segment) -> Any:
        """Hashes of the segment's ``k``-word shingles (the whole segment if shorter)."""
        np = self.np
        words = np.fromiter(
            map(self.word_hashes.__getitem__, segment.words),
            np.uint64,
            count=len(segment.words),
        )
        k = min(self.k, len(words))
        count = len(words) - k + 1
        with np.errstate(over="ignore"):
            hashed = words[:count] * self.shingle_mult[0]
            for j in range(1, k):
                hashed = hashed + words[j : j + count] * self.shingle_mult[j]
        return hashed

    def band_keys(self, shingle_arrays: Sequence[Any]) -> list[list[int]]:
        """One LSH key per band for each shingle array, in bounded batches."""
        np = self.np
        keys: list[list[int]] = []
        start = 0
        while start < len(shingle_arrays):
            stop, total = start, 0
            while stop < len(shingle_arrays) and (
                stop == start or total + len(shingle_arrays[stop]) <= _MINHASH_BATCH
            ):
                total += len(shingle_arrays[stop])
                stop += 1
            batch = shingle_arrays[start:stop]
            offsets = np.cumsum([0] + [len(a) for a in batch[:-1]])
            hashed = np.concatenate(batch)[:, None] * self.a
            with np.errstate(over="ignore"):
                hashed += self.b
            hashed >>= np.uint64(32)
            signatures = np.minimum.reduceat(hashed, offsets, axis=0)
            banded = signatures.reshape(len(batch), self.bands, self.rows)
            with np.errstate(over="ignore"):
                mixed = (banded * self.band_mix).sum(axis=2, dtype=np.uint64)
            keys.extend(mixed.tolist())
            start = stop
        return keys

    def add_training(
        self, gid: int, segments: Sequence[_Segment], matches: _Matches
    ) -> None:
        """Queue a training row; verify candidates batch by batch."""
        shingles = self.row_shingles(segments)
        if shingles is None:
            return
        self.pending.append((gid, shingles))
        self.pending_size += len(shingles)
        if self.pending_size >= _VERIFY_BATCH:
            self.flush(matches)

    def flush(self, matches: _Matches) -> None:
        pending, self.pending, self.pending_size = self.pending, [], 0
        keys = self.band_keys([p[1] for p in pending])
        for (gid, shingles), band_keys in zip(pending, keys, strict=True):
            candidates: set[int] = set()
            for band, key in enumerate(band_keys):
                candidates.update(self.buckets[band].get(key, ()))
            if not candidates:
                continue
            train_set = set(shingles.tolist())
            for candidate in sorted(candidates):
                eval_gid, eval_set = self.indexed[candidate]
                score = len(train_set & eval_set) / len(train_set | eval_set)
                if score < self.threshold:
                    continue
                matches.near_by_train.setdefault(gid, set()).add(eval_gid)
                best = matches.near.get(eval_gid)
                if best is None or score > best[0]:
                    matches.near[eval_gid] = (score, gid)


def _scan_training(
    index: _NgramIndex,
    near: _NearDuplicates | None,
    train: Sequence[TextRecord],
    config: DecontaminationConfig,
) -> _Matches:
    """One pass over the training rows: n-gram hits, then near duplicates."""
    matches = _Matches()
    cap = config.max_evidence
    for gid, record in enumerate(train):
        segments = [_Segment.build(raw, config.mask_digits) for raw in record.segments]
        hit: set[str] = set()
        for segment in segments:
            hit |= index.hits(segment)
        if near is not None:
            near.add_training(gid, segments, matches)
        if not hit:
            continue
        matches.row_keys[gid] = hit
        for key in hit:
            rows = matches.key_rows.setdefault(key, [])
            if len(rows) < cap:
                rows.append(gid)
    if near is not None and near.pending:
        near.flush(matches)
    return matches


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class DecontaminationResult:
    """The report plus the training-row decisions the actions need."""

    report: dict[str, Any]
    #: training record id -> names of the eval sets it contaminates.
    contaminating: dict[str, list[str]]
    #: eval set name -> contaminated eval record ids.
    contaminated: dict[str, list[str]]

    @property
    def is_clean(self) -> bool:
        return not self.contaminating and not any(self.contaminated.values())


def _prepare_eval(
    evals: Sequence[TextSource], config: DecontaminationConfig
) -> list[_EvalRow]:
    return [
        _EvalRow(
            source_index,
            index,
            [_Segment.build(raw, config.mask_digits) for raw in record.segments],
        )
        for source_index, source in enumerate(evals)
        for index, record in enumerate(source.records)
    ]


def _eval_row_entry(
    row: _EvalRow,
    record_id: str,
    hit_keys: list[str],
    matches: _Matches,
    train_ids: list[str],
    config: DecontaminationConfig,
    near: tuple[float, int] | None,
) -> dict[str, Any]:
    examples = sorted({gid for key in hit_keys for gid in matches.key_rows[key]})
    entry: dict[str, Any] = {
        "row": record_id,
        "index": row.index,
        "match": (["ngram"] if hit_keys else [])
        + (["near_duplicate"] if near is not None else []),
        "ngram_overlap": round(len(hit_keys) / len(row.keys), 6) if row.keys else 0.0,
        "overlap_ngram_count": len(hit_keys),
        "overlap_ngrams": hit_keys[: config.max_evidence],
        "train_examples": [train_ids[gid] for gid in examples[: config.max_evidence]],
    }
    if near is not None:
        score, train_gid = near
        entry["near_duplicate"] = {
            "jaccard": round(score, 6),
            "train_row": train_ids[train_gid],
        }
    return entry


def _source_report(
    source: TextSource, entries: list[dict[str, Any]], rows: list[_EvalRow]
) -> dict[str, Any]:
    return {
        **source.describe(),
        "checked_rows": sum(bool(row.keys) for row in rows),
        "unchecked_short_segments": sum(row.short_unchecked for row in rows),
        "contaminated_rows": len(entries),
        "contamination_rate": round(len(entries) / len(rows), 6) if rows else 0.0,
        "contaminated": entries,
    }


def _contaminating_rows(
    train_ids: list[str],
    eval_rows: list[_EvalRow],
    names: list[str],
    index: _NgramIndex,
    matches: _Matches,
    contaminated_rows: set[int],
    min_overlap: float,
) -> dict[str, list[str]]:
    """Training row id -> eval sets of the contaminated rows it overlaps.

    With ``min_overlap`` 0 any shared n-gram counts. Above 0, a training row
    must itself contain that fraction of a contaminated eval row's n-grams:
    a row that shares only boilerplate with a leaked eval row is kept.
    """
    contaminating: dict[str, list[str]] = {}
    key_sets: dict[str, frozenset[str]] = {}
    for gid, record_id in enumerate(train_ids):
        sets: set[str] = set()
        keys = matches.row_keys.get(gid, set())
        if keys and min_overlap > 0:
            shared: dict[int, int] = {}
            for key in keys:
                for e in index.keys[key]:
                    if e in contaminated_rows:
                        shared[e] = shared.get(e, 0) + 1
            sets.update(
                names[eval_rows[e].source]
                for e, count in shared.items()
                if count / len(eval_rows[e].keys) >= min_overlap
            )
        else:
            for key in keys:
                # Computed once per key so a template n-gram shared by every
                # row does not cost rows x rows.
                if key not in key_sets:
                    key_sets[key] = frozenset(
                        names[eval_rows[e].source]
                        for e in index.keys[key]
                        if e in contaminated_rows
                    )
                sets |= key_sets[key]
        sets.update(
            names[eval_rows[e].source]
            for e in matches.near_by_train.get(gid, ())
            if e in contaminated_rows
        )
        if sets:
            contaminating[record_id] = sorted(sets)
    return contaminating


def _method(config: DecontaminationConfig) -> str:
    method = f"exact {config.ngram_size}-gram overlap on normalized words"
    if config.near_duplicates:
        method += (
            f" + MinHash/LSH near duplicates (exact Jaccard >= "
            f"{config.near_duplicate_threshold} over {config.shingle_size}-word shingles)"
        )
    return method


def find_contamination(
    train: Sequence[TextSource],
    evals: Sequence[TextSource],
    config: DecontaminationConfig | None = None,
) -> DecontaminationResult:
    """Compare every training source with every eval set.

    Eval set names must be unique. The report lists, per eval set, each
    contaminated row with its overlapping n-grams (the first ``max_evidence``
    in eval-text order), the fraction of its n-grams found in training data,
    example training rows, and its best near duplicate when that check is on.
    """
    config = config or DecontaminationConfig()
    names = [source.name for source in evals]
    if not evals or len(set(names)) != len(names):
        raise ValueError("Eval sets must be nonempty and uniquely named")
    train_records = [record for source in train for record in source.records]
    train_ids = [record.record_id for record in train_records]
    eval_rows = _prepare_eval(evals, config)
    index = _NgramIndex(eval_rows, config)
    near = _NearDuplicates(eval_rows, config) if config.near_duplicates else None
    matches = _scan_training(index, near, train_records, config)

    contaminated_rows: set[int] = set()
    per_source: list[list[dict[str, Any]]] = [[] for _ in evals]
    for gid, row in enumerate(eval_rows):
        hit = [key for key in row.keys if key in matches.key_rows]
        if hit and len(hit) / len(row.keys) < config.min_overlap:
            hit = []
        if not hit and gid not in matches.near:
            continue
        contaminated_rows.add(gid)
        record_id = evals[row.source].records[row.index].record_id
        per_source[row.source].append(
            _eval_row_entry(
                row, record_id, hit, matches, train_ids, config, matches.near.get(gid)
            )
        )
    contaminating = _contaminating_rows(
        train_ids,
        eval_rows,
        names,
        index,
        matches,
        contaminated_rows,
        config.min_overlap,
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "decontamination_report",
        "method": _method(config),
        "config": config.to_dict(),
        "train": {
            "sources": [source.describe() for source in train],
            "rows": len(train_records),
            "contaminating_rows": len(contaminating),
            "contaminating": [
                {"row": row_id, "eval_sets": sets}
                for row_id, sets in contaminating.items()
            ],
        },
        "eval_sets": {
            source.name: _source_report(
                source, per_source[i], [row for row in eval_rows if row.source == i]
            )
            for i, source in enumerate(evals)
        },
        "summary": {
            "eval_rows": len(eval_rows),
            "contaminated_eval_rows": len(contaminated_rows),
            "contaminating_train_rows": len(contaminating),
            "clean": not contaminated_rows,
        },
    }
    contaminated = {
        source.name: [entry["row"] for entry in entries]
        for source, entries in zip(evals, per_source, strict=True)
    }
    return DecontaminationResult(report, contaminating, contaminated)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def apply_decontamination(
    rows: Sequence[tuple[str, Any]],
    result: DecontaminationResult,
    action: Action,
    *,
    target: Literal["train", "eval"] = "train",
    eval_set: str | None = None,
) -> tuple[list[Any], int]:
    """Filter out or flag contaminated rows; return ``(rows, affected)``.

    ``target="train"`` acts on the training rows that overlap any eval set;
    ``target="eval"`` acts on the contaminated rows of ``eval_set`` (to score
    a clean subset). ``flag`` keeps every row and records the decision under
    ``metadata.contamination`` (rows must be objects whose ``metadata``, if
    present, is an object); inputs are never mutated.
    """
    if action not in ("filter", "flag"):
        raise ValueError("action must be 'filter' or 'flag'")
    if target == "train":
        marked: Mapping[str, list[str]] = result.contaminating
    elif target == "eval":
        if eval_set not in result.contaminated:
            raise ValueError(f"Unknown eval set {eval_set!r}")
        marked = {row_id: [eval_set] for row_id in result.contaminated[eval_set]}
    else:
        raise ValueError("target must be 'train' or 'eval'")
    kept: list[Any] = []
    affected = 0
    for row_id, row in rows:
        sets = marked.get(row_id)
        if sets is None:
            kept.append(row)
            continue
        affected += 1
        if action == "filter":
            continue
        if not isinstance(row, dict) or not isinstance(row.get("metadata", {}), dict):
            raise ValueError(
                f"{row_id}: only object rows with object metadata can be flagged"
            )
        flagged = copy.deepcopy(row)
        flagged.setdefault("metadata", {})["contamination"] = {
            "contaminated": True,
            "eval_sets": list(sets),
            "method": result.report["method"],
        }
        kept.append(flagged)
    return kept, affected
