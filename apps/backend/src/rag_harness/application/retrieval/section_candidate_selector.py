"""
Narrows candidate section titles for a query via native Qdrant dense+sparse
search (no hand-rolled dot product), RRF-fuses the two ranked lists, then
cross-encoder reranks the shortlist before handing it to the LLM for a
final pick.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from rag_harness.application.retrieval.model_registry import CrossEncoderRegistry
from rag_harness.config.settings import RetrievalSettings
from rag_harness.infrastructure.vector_stores import QdrantSectionStore

logger = logging.getLogger("rag_harness.section_selector")


class SectionCandidateSelector:
    def __init__(self, section_store: QdrantSectionStore, model_registry: CrossEncoderRegistry, settings: RetrievalSettings):
        self.section_store = section_store
        self.model_registry = model_registry
        self.settings = settings

    def get_candidates(self, query: str, top_n: int = 20, final_k: int = 10) -> list[str]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(self.section_store.dense_candidates, query, top_n)
            sparse_future = executor.submit(self.section_store.sparse_candidates, query, top_n)
            dense_hits = dense_future.result()
            sparse_hits = sparse_future.result()

        k = self.settings.RRF_K
        scores: dict[str, float] = {}
        for rank, name in enumerate(dense_hits, start=1):
            scores[name] = scores.get(name, 0.0) + 1.0 / (k + rank)
        for rank, name in enumerate(sparse_hits, start=1):
            scores[name] = scores.get(name, 0.0) + 1.0 / (k + rank)

        fused_names = [name for name, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
        if not fused_names:
            logger.info("No section candidates from dense/sparse — treating query as general")
            return []

        reranker = self.model_registry.reranker()
        pairs = [(query, name) for name in fused_names]
        rerank_scores = reranker.predict(pairs)
        reranked = [name for name, _ in sorted(zip(fused_names, rerank_scores), key=lambda x: x[1], reverse=True)]

        final_candidates = reranked[:final_k]
        logger.info(f"Section candidates: {len(dense_hits)} dense + {len(sparse_hits)} sparse -> {len(final_candidates)} final")
        return final_candidates