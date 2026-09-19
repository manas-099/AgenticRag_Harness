"""
In-memory BM25 sparse retriever. For production scale, swap for
Elasticsearch/OpenSearch — callers depend on this class's interface
(index/search), not on rank_bm25 directly, so that swap stays contained here.
"""

from __future__ import annotations

import logging

from rank_bm25 import BM25Okapi

from rag_harness.domain.models import Chunk, RetrievedChunk

logger = logging.getLogger("rag_harness.sparse")


class BM25Retriever:
    def __init__(self):
        self.corpus_chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None

    def index(self, chunks: list[Chunk]) -> None:
        self.corpus_chunks = chunks
        tokenized = [c.contextual_text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        logger.info(f"Indexed {len(chunks)} chunks into BM25")

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self.bm25 is None:
            logger.warning("BM25 index not built yet — returning empty results")
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(zip(self.corpus_chunks, scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [RetrievedChunk(chunk=c, score=float(s), source="sparse") for c, s in ranked if s > 0]

    @property
    def is_indexed(self) -> bool:
        return self.bm25 is not None

    @property
    def corpus_size(self) -> int:
        return len(self.corpus_chunks)