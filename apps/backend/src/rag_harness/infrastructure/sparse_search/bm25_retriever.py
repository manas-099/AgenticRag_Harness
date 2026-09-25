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

    def rebuild_from_vector_store(self, vector_store) -> None:
        """Repopulate this in-memory index from Qdrant (the durable source
        of truth). Call this once at startup — BM25Retriever otherwise
        starts empty on every process restart / `--reload`, silently
        losing sparse search for everything ingested before the restart
        even though the dense vectors are still safely in Qdrant."""
        try:
            chunks = vector_store.scroll_all_chunks()
        except Exception as e:
            logger.error(f"BM25 rebuild-from-store failed (starting with empty index): {e}")
            return

        if chunks:
            self.index(chunks)
            logger.info(f"BM25 index rebuilt from vector store: {len(chunks)} chunks")
        else:
            logger.info("BM25 rebuild-from-store: vector store is empty, nothing to index yet")

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self.bm25 is None:
            logger.warning("BM25 index not built yet — returning empty results")
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        # NOTE: previously filtered with `if s > 0`. BM25's IDF term goes
        # negative when a query term appears in most/all documents in a
        # small corpus (common in early-stage or per-tenant indexes), which
        # made genuinely relevant matches score <= 0 and get silently
        # dropped — a real contributor to "no chunks retrieved" on small
        # corpora. We instead keep any chunk that actually matched at least
        # one query token, regardless of score sign, and let RRF fusion +
        # reranking (which use rank position / cross-encoder scores, not
        # this raw BM25 score) do the real relevance filtering downstream.
        query_terms = set(tokenized_query)
        ranked = sorted(zip(self.corpus_chunks, scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            RetrievedChunk(chunk=c, score=float(s), source="sparse")
            for c, s in ranked
            if query_terms & set(c.contextual_text.lower().split())
        ]

    @property
    def is_indexed(self) -> bool:
        return self.bm25 is not None

    @property
    def corpus_size(self) -> int:
        return len(self.corpus_chunks)