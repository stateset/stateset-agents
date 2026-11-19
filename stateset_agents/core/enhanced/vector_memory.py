"""Memory retrieval with explicit backend readiness and atomic reindexing."""

from __future__ import annotations

import copy
import logging
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from threading import RLock
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..agent_initialization import BackendInitializationState, _checked_component

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Memory content with retrieval metadata and an optional dense embedding."""

    content: str
    embedding: np.ndarray | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 1.0
    context: dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: datetime | None = None

    def update_access(self) -> None:
        """Record a successful retrieval of this entry."""
        self.access_count += 1
        self.last_accessed = datetime.now()


@dataclass
class _MemoryBackend:
    state: BackendInitializationState = BackendInitializationState.NEW
    request: tuple[str, str] | None = None
    model: Any = None
    backend: str = "uninitialized"
    generation: int = 0
    error: BaseException | None = None
    mutation_revision: int = 0
    snapshot_operation: dict[str, Any] | None = None
    last_snapshot_operation: dict[str, Any] | None = None
    lock: RLock = field(default_factory=RLock, repr=False, compare=False)

    def __getstate__(self) -> dict[str, Any]:
        with self.lock:
            if self.state is BackendInitializationState.INITIALIZING:
                raise RuntimeError("Cannot copy memory while initializing its backend")
            if self.snapshot_operation is not None:
                raise RuntimeError("Cannot copy memory during a snapshot operation")
            return {key: value for key, value in vars(self).items() if key != "lock"}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.mutation_revision = state.get("mutation_revision", 0)
        self.snapshot_operation = None
        self.last_snapshot_operation = state.get("last_snapshot_operation")
        self.lock = RLock()


class VectorMemory:
    """Dense retrieval with a deterministic lexical fallback when the SDK is absent."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_entries: int = 1000,
    ) -> None:
        if type(max_entries) is not int or max_entries < 0:
            raise ValueError("max_entries must be a nonnegative integer")
        self.embedding_model = embedding_model
        self.max_entries = max_entries
        self.entries: list[MemoryEntry] = []
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self._backend = _MemoryBackend()
        if torch is not None:
            try:
                cuda_available = torch.cuda.is_available()
            except (AttributeError, RuntimeError):  # pragma: no cover - device probing
                cuda_available = False
            self.device = torch.device("cuda" if cuda_available else "cpu")
        else:
            self.device = "cpu"

    def _owner(self) -> _MemoryBackend:
        # Previously serialized instances have no lifecycle receipt.
        if not hasattr(self, "_backend"):
            self._backend = _MemoryBackend()
        return self._backend

    def _request(self) -> tuple[str, str]:
        return self.embedding_model, str(self.device)

    async def save_snapshot(
        self,
        path: str | os.PathLike[str],
        *,
        expected_revision: str | None = None,
        max_bytes: int = 64 * 1024 * 1024,
    ) -> dict[str, Any]:
        """Save portable entry data; existing files require their current revision.

        The parent directory must exist. Cancellation drains file work and may
        still publish a snapshot; inspect ``get_snapshot_status()`` afterward.
        Embeddings, runtime objects, and subclass-specific state are excluded.
        """
        from .memory_snapshot import save_snapshot

        return await save_snapshot(self, path, expected_revision, max_bytes)

    async def restore_snapshot(
        self,
        path: str | os.PathLike[str],
        *,
        expected_revision: str | None = None,
        max_bytes: int = 64 * 1024 * 1024,
    ) -> dict[str, Any]:
        """Rebuild a snapshot with this backend and publish only if memory is unchanged.

        Source configuration is informational; this object's model and capacity
        remain in force. Cancellation or failed preparation leaves entries intact.
        """
        from .memory_snapshot import restore_snapshot

        return await restore_snapshot(self, path, expected_revision, max_bytes)

    @staticmethod
    async def inspect_snapshot(
        path: str | os.PathLike[str], *, max_bytes: int = 64 * 1024 * 1024
    ) -> dict[str, Any]:
        """Validate a saved file and return metadata without loading an encoder."""
        from .memory_snapshot import inspect_snapshot

        return await inspect_snapshot(path, max_bytes)

    def get_snapshot_status(self) -> dict[str, Any]:
        """Return detached active/latest snapshot receipts without entry content."""
        owner = self._owner()
        with owner.lock:
            return copy.deepcopy(
                {
                    "active": owner.snapshot_operation,
                    "last_finished": owner.last_snapshot_operation,
                }
            )

    def get_initialization_status(self) -> dict[str, Any]:
        """Inspect readiness without loading or changing the embedding backend."""
        owner = self._owner()
        return {
            "state": owner.state.value,
            "backend": owner.backend,
            "generation": owner.generation,
            "request": (
                {"model": owner.request[0], "device": owner.request[1]}
                if owner.request is not None
                else None
            ),
            "backend_matches": owner.state is BackendInitializationState.READY
            and self.model is owner.model
            and self._request() == owner.request,
            "error_type": (
                type(owner.error).__name__ if owner.error is not None else None
            ),
        }

    async def initialize(self, *, force: bool = False) -> None:
        """Prepare a backend and all embeddings before publishing either.

        Repeated calls reuse a ready backend. ``force=True`` explicitly retries
        dependency discovery, including promotion from lexical to dense retrieval.
        SDK loading/encoding remains synchronous. Stop direct entry mutation first.
        """
        owner = self._owner()
        with owner.lock:
            if owner.state is BackendInitializationState.INITIALIZING:
                raise RuntimeError("Memory initialization cannot reenter itself")
            request = self._request()
            if (
                not force
                and owner.state is BackendInitializationState.READY
                and request == owner.request
                and self.model is owner.model
            ):
                return
            owner.state = BackendInitializationState.INITIALIZING
            owner.error = None
            try:
                if not isinstance(request[0], str) or not request[0].strip():
                    raise ValueError("embedding_model must be a nonempty name")
                try:
                    from sentence_transformers import SentenceTransformer
                except ModuleNotFoundError as error:
                    if error.name != "sentence_transformers":
                        raise
                    candidate = None
                    backend = "lexical"
                    logger.warning(
                        "sentence-transformers is absent; using lexical memory retrieval"
                    )
                else:
                    candidate = _checked_component(
                        "Memory model",
                        SentenceTransformer(self.embedding_model),
                        owned=True,
                    )
                    moved = candidate.to(self.device)
                    if moved is not None:
                        _checked_component("Memory device setup", moved, owned=True)
                    backend = "sentence_transformers"
                entries = tuple(self.entries)
                embeddings = [
                    (
                        self._encode(candidate, entry.content)
                        if candidate is not None
                        else None
                    )
                    for entry in entries
                ]
                dimensions = {len(value) for value in embeddings if value is not None}
                if len(dimensions) > 1:
                    raise ValueError(
                        "Embedding model returned inconsistent vector dimensions"
                    )
                if self._request() != request or tuple(map(id, self.entries)) != tuple(
                    map(id, entries)
                ):
                    raise ValueError(
                        "Memory configuration or entries changed during initialization"
                    )
            except BaseException as error:
                owner.error = error
                owner.state = BackendInitializationState.FAILED
                raise
            else:
                # No awaits or user callbacks occur during publication.
                for entry, embedding in zip(entries, embeddings, strict=True):
                    entry.embedding = embedding
                self.model = candidate
                owner.model = candidate
                owner.request = request
                owner.backend = backend
                owner.generation += 1
                owner.state = BackendInitializationState.READY

    def _ensure_usable(self) -> None:
        owner = self._owner()
        if owner.state in (
            BackendInitializationState.INITIALIZING,
            BackendInitializationState.FAILED,
        ):
            raise RuntimeError(
                f"Memory initialization is {owner.state.value}; initialize memory before use"
            ) from owner.error
        if owner.state is BackendInitializationState.READY and (
            self.model is not owner.model or self._request() != owner.request
        ):
            raise RuntimeError(
                "Memory embedding backend changed; initialize memory before use"
            )

    def _token(self) -> tuple[Any, int, tuple[str, str]]:
        self._ensure_usable()
        return self.model, self._owner().generation, self._request()

    def _check_token(self, token: tuple[Any, int, tuple[str, str]]) -> None:
        self._ensure_usable()
        if (
            self.model is not token[0]
            or self._owner().generation != token[1]
            or self._request() != token[2]
        ):
            raise RuntimeError(
                "Memory embedding backend changed or index restored during retrieval or insertion"
            )

    async def add_entry(
        self,
        content: str,
        context: dict[str, Any] | None = None,
        importance: float = 1.0,
    ) -> None:
        """Insert one entry only after its embedding has completed successfully."""
        with self._owner().lock:
            token = self._token()
            entry = MemoryEntry(
                content=content,
                context=copy.deepcopy(context or {}),
                importance=importance,
            )
        if token[0] is not None:
            entry.embedding = await self._generate_embedding(content)
        with self._owner().lock:
            self._check_token(token)
            if entry.embedding is not None and any(
                previous.embedding is not None
                and len(previous.embedding) != len(entry.embedding)
                for previous in self.entries
            ):
                raise ValueError(
                    "Embedding model returned inconsistent vector dimensions"
                )
            self.entries.append(entry)
            if len(self.entries) > self.max_entries:
                self.entries.sort(
                    key=lambda item: item.importance * (1 + item.access_count * 0.1),
                    reverse=True,
                )
                self.entries = self.entries[: self.max_entries]
            self._owner().mutation_revision += 1

    async def retrieve_relevant(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Retrieve positive similarities; lexical fallback uses word-count cosine."""
        if type(top_k) is not int or top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")
        with self._owner().lock:
            token = self._token()
            if not self.entries or top_k == 0:
                return []
        query_embedding = (
            await self._generate_embedding(query) if token[0] is not None else None
        )
        with self._owner().lock:
            self._check_token(token)
            query_words = Counter(re.findall(r"\w+", query.casefold()))
            similarities = []
            for entry in self.entries:
                if query_embedding is None:
                    words = Counter(re.findall(r"\w+", entry.content.casefold()))
                    denominator = math.sqrt(
                        sum(value * value for value in query_words.values())
                        * sum(value * value for value in words.values())
                    )
                    score = (
                        sum(count * words[word] for word, count in query_words.items())
                        / denominator
                        if denominator
                        else 0.0
                    )
                elif entry.embedding is not None:
                    if entry.embedding.shape != query_embedding.shape:
                        raise ValueError(
                            "Query and memory embeddings have different dimensions"
                        )
                    denominator = float(
                        np.linalg.norm(query_embedding)
                        * np.linalg.norm(entry.embedding)
                    )
                    score = (
                        float(np.dot(query_embedding, entry.embedding)) / denominator
                        if denominator
                        else 0.0
                    )
                else:
                    continue
                if math.isfinite(score) and score > 0:
                    similarities.append((score * (1 + entry.access_count * 0.1), entry))
            similarities.sort(key=lambda item: item[0], reverse=True)
            result = [entry for _, entry in similarities[:top_k]]
            for entry in result:
                entry.update_access()
            if result:
                self._owner().mutation_revision += 1
            return result

    @staticmethod
    def _encode(model: Any, text: str) -> NDArray[np.float64]:
        if torch is not None:
            with torch.no_grad():
                raw = model.encode(text, convert_to_numpy=True)
        else:
            raw = model.encode(text, convert_to_numpy=True)
        raw = _checked_component("Memory embedding", raw, owned=True)
        embedding = np.asarray(raw, dtype=np.float64).copy()
        if (
            embedding.ndim != 1
            or not embedding.size
            or not np.isfinite(embedding).all()
        ):
            raise ValueError("Memory embeddings must be nonempty finite vectors")
        return embedding

    async def _generate_embedding(self, text: str) -> NDArray[np.float64]:
        with self._owner().lock:
            self._ensure_usable()
            if self.model is None:
                raise RuntimeError("Lexical memory does not produce dense embeddings")
            return self._encode(self.model, text)
