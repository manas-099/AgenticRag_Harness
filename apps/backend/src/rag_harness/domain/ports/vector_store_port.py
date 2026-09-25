"""
Abstract contract for chunk storage/retrieval. Qdrant is the current
implementation (infrastructure/vector_stores/qdrant_chunk_store.py), but
nothing in application/ should import Qdrant directly — only this port.
"""

from abc import ABC, abstractmethod
from typing import Optional

from rag_harness.domain.models import Chunk, RetrievedChunk


class VectorStorePort(ABC):
    @abstractmethod
    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    def dense_search(
        self, query_vector: list[float], top_k: int, filters: Optional[dict] = None
    ) -> list[RetrievedChunk]:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def scroll_all_chunks(self, batch_size: int = 256) -> list[Chunk]:
        """Return every chunk currently stored, for rebuilding derived
        in-memory indexes (e.g. BM25) after a process restart. Qdrant is
        the durable source of truth; anything kept only in process memory
        must be reconstructible from here."""
        raise NotImplementedError