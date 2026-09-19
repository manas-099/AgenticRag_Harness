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