"""
Top-level retrieval use case: dense and sparse search run in PARALLEL
(independent, no shared data dependency — real speed win via ThreadPoolExecutor),
then RRF fusion -> dedup -> rerank. Optionally accepts a pre-built metadata
filter (from the section-filtering service) applied to the dense leg only.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from rag_harness.application.retrieval.fusion_service import RetrievalFusionService
from rag_harness.config.settings import RetrievalSettings
from rag_harness.domain.exceptions import HarnessError, RetrievalError
from rag_harness.domain.models import RetrievedChunk
from rag_harness.domain.ports import EmbedderPort, VectorStorePort
from rag_harness.infrastructure.sparse_search import BM25Retriever

logger = logging.getLogger("rag_harness.retrieval_use_case")


class RetrieveUseCase:
    def __init__(
        self,
        vector_store: VectorStorePort,
        sparse_retriever: BM25Retriever,
        fusion_service: RetrievalFusionService,
        embedder: EmbedderPort,
        settings: RetrievalSettings,
    ):
        self.vector_store = vector_store
        self.sparse_retriever = sparse_retriever
        self.fusion_service = fusion_service
        self.embedder = embedder
        self.settings = settings

    def _run_dense(self, query: str, filters: dict | None) -> list[RetrievedChunk]:
        try:
            query_vector = self.embedder.encode([query])[0]
            return self.vector_store.dense_search(query_vector, self.settings.TOP_K_DENSE, filters)
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            return []

    def _run_sparse(self, query: str) -> list[RetrievedChunk]:
        try:
            return self.sparse_retriever.search(query, self.settings.TOP_K_SPARSE)
        except Exception as e:
            logger.error(f"Sparse retrieval failed: {e}")
            return []

    def execute(self, query: str, filters: dict | None = None) -> list[RetrievedChunk]:
        try:
            logger.info(f"Running parallel retrieval (dense + sparse) for query='{query[:50]}...'")

            with ThreadPoolExecutor(max_workers=2) as executor:
                dense_future = executor.submit(self._run_dense, query, filters)
                sparse_future = executor.submit(self._run_sparse, query)
                dense_hits = dense_future.result()
                sparse_hits = sparse_future.result()

            logger.info(f"Parallel retrieval done: dense={len(dense_hits)}, sparse={len(sparse_hits)}")

            fused = self.fusion_service.fuse(dense_hits, sparse_hits)
            deduped = self.fusion_service.deduplicate(fused)
            reranked = self.fusion_service.rerank(query, deduped)

            logger.info(f"Retrieval complete: {len(reranked)} final chunks")
            return reranked

        except HarnessError:
            raise
        except Exception as e:
            logger.error(f"Retrieval pipeline failed unexpectedly: {e}")
            raise RetrievalError(f"Retrieval pipeline failed: {e}")