"""
In-memory 3-tier cache: query (exact match, fastest), embedding, and
retrieval (skips vector search on repeated queries). Swap for Redis in
production (redis_cache.py implementing the same interface) without
touching any calling code.

Cache entries are keyed with a document-version string so that re-ingesting
a document can selectively invalidate stale entries rather than wiping
everything.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

from rag_harness.domain.models import RetrievedChunk

logger = logging.getLogger("rag_harness.cache")


class InMemoryCache:
    def __init__(self):
        self.query_cache: dict[str, str] = {}
        self.embedding_cache: dict[str, list[float]] = {}
        self.retrieval_cache: dict[str, list[RetrievedChunk]] = {}

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    # ---- Query cache ----
    def get_query(self, query: str, doc_version_key: str) -> Optional[str]:
        hit = self.query_cache.get(self._hash(f"{query}::{doc_version_key}"))
        if hit:
            logger.info("Query cache HIT")
        return hit

    def set_query(self, query: str, doc_version_key: str, answer: str) -> None:
        self.query_cache[self._hash(f"{query}::{doc_version_key}")] = answer

    # ---- Embedding cache ----
    def get_embedding(self, text: str) -> Optional[list[float]]:
        return self.embedding_cache.get(self._hash(text))

    def set_embedding(self, text: str, embedding: list[float]) -> None:
        self.embedding_cache[self._hash(text)] = embedding

    # ---- Retrieval cache ----
    def get_retrieval(self, query: str, doc_version_key: str) -> Optional[list[RetrievedChunk]]:
        hit = self.retrieval_cache.get(self._hash(f"retrieval::{query}::{doc_version_key}"))
        if hit:
            logger.info("Retrieval cache HIT")
        return hit

    def set_retrieval(self, query: str, doc_version_key: str, chunks: list[RetrievedChunk]) -> None:
        self.retrieval_cache[self._hash(f"retrieval::{query}::{doc_version_key}")] = chunks

    # ---- Invalidation ----
    def invalidate_by_doc_version(self, doc_id: str, old_version: str) -> None:
        stale_query = [k for k in self.query_cache if old_version in k]
        stale_retrieval = [k for k in self.retrieval_cache if old_version in k]
        for k in stale_query:
            del self.query_cache[k]
        for k in stale_retrieval:
            del self.retrieval_cache[k]
        logger.info(f"Invalidated {len(stale_query)} query + {len(stale_retrieval)} retrieval entries for {doc_id} v{old_version}")

    @property
    def query_cache_size(self) -> int:
        return len(self.query_cache)

    @property
    def retrieval_cache_size(self) -> int:
        return len(self.retrieval_cache)