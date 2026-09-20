"""
Reciprocal Rank Fusion (rank-position based, so dense and sparse scores on
incomparable scales can be combined fairly), embedding-similarity dedup
(runs before rerank so the expensive step isn't wasted on near-duplicates),
and cross-encoder reranking (final precision pass on the small candidate set).
"""

from __future__ import annotations

import logging

import numpy as np

from rag_harness.application.retrieval.model_registry import CrossEncoderRegistry
from rag_harness.config.settings import RetrievalSettings
from rag_harness.domain.models import RetrievedChunk
from rag_harness.domain.ports import EmbedderPort

logger = logging.getLogger("rag_harness.fuser")


class RetrievalFusionService:
    def __init__(self, settings: RetrievalSettings, embedder: EmbedderPort, model_registry: CrossEncoderRegistry):
        self.settings = settings
        self.embedder = embedder
        self.model_registry = model_registry

    def rrf_fuse(self, ranked_lists: list[list[RetrievedChunk]], k: int | None = None) -> list[RetrievedChunk]:
        k = k or self.settings.RRF_K
        scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievedChunk] = {}

        for ranked_list in ranked_lists:
            for rank, rc in enumerate(ranked_list, start=1):
                cid = rc.chunk.chunk_id
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
                chunk_map[cid] = rc

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [RetrievedChunk(chunk=chunk_map[cid].chunk, score=score, source="fused") for cid, score in fused]

    def fuse(self, dense: list[RetrievedChunk], sparse: list[RetrievedChunk]) -> list[RetrievedChunk]:
        fused = self.rrf_fuse([dense, sparse])
        logger.info(f"RRF fusion: {len(dense)} dense + {len(sparse)} sparse -> {len(fused)} fused")
        return fused

    def deduplicate(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return chunks

        texts = [c.chunk.contextual_text for c in chunks]
        embeddings = np.array(self.embedder.encode(texts))

        keep_indices: list[int] = []
        for i in range(len(chunks)):
            is_duplicate = False
            for j in keep_indices:
                sim = float(np.dot(embeddings[i], embeddings[j]))
                if sim >= self.settings.DEDUP_SIMILARITY_THRESHOLD:
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep_indices.append(i)

        deduped = [chunks[i] for i in keep_indices]
        logger.info(f"Dedup: {len(chunks)} -> {len(deduped)} chunks")
        return deduped

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int | None = None) -> list[RetrievedChunk]:
        if not chunks:
            return chunks
        top_k = top_k or self.settings.TOP_K_FINAL_RERANK
        reranker = self.model_registry.reranker()
        pairs = [(query, c.chunk.contextual_text) for c in chunks]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [RetrievedChunk(chunk=c.chunk, score=float(s), source=c.source) for c, s in ranked]