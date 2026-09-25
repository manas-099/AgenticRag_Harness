"""
Top-level retrieval use case: dense and sparse search run in PARALLEL
(independent, no shared data dependency — real speed win via ThreadPoolExecutor),
then RRF fusion -> dedup -> rerank. Optionally accepts a pre-built metadata
filter (from the section-filtering service) applied to the dense leg only.

Failure handling: a leg that raises is NOT silently treated the same as a
leg that legitimately found nothing. If only one leg fails we log a
warning and continue with the other leg's results (a degraded-but-useful
retrieval). If BOTH legs fail we raise RetrievalError instead of quietly
returning [] — a broken embedder or vector store must surface as an error,
not disguise itself as "no relevant chunks" and end up looking like an
ordinary 200 OK with a canned "not enough information" answer downstream.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from rag_harness.application.retrieval.fusion_service import RetrievalFusionService
from rag_harness.config.settings import RetrievalSettings
from rag_harness.domain.exceptions import HarnessError, RetrievalError
from rag_harness.domain.models import RetrievedChunk
from rag_harness.domain.ports import EmbedderPort, VectorStorePort
from rag_harness.infrastructure.sparse_search import BM25Retriever

logger = logging.getLogger("rag_harness.retrieval_use_case")


@dataclass
class _LegResult:
    hits: list[RetrievedChunk]
    error: Exception | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None


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

    def _run_dense(self, query: str, filters: dict | None) -> _LegResult:
        try:
            query_vector = self.embedder.encode([query])[0]
            hits = self.vector_store.dense_search(query_vector, self.settings.TOP_K_DENSE, filters)
            return _LegResult(hits=hits)
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            return _LegResult(hits=[], error=e)

    def _run_sparse(self, query: str) -> _LegResult:
        try:
            hits = self.sparse_retriever.search(query, self.settings.TOP_K_SPARSE)
            return _LegResult(hits=hits)
        except Exception as e:
            logger.error(f"Sparse retrieval failed: {e}")
            return _LegResult(hits=[], error=e)

    def execute(self, query: str, filters: dict | None = None) -> list[RetrievedChunk]:
        try:
            logger.info(f"Running parallel retrieval (dense + sparse) for query='{query[:50]}...'")

            with ThreadPoolExecutor(max_workers=2) as executor:
                dense_future = executor.submit(self._run_dense, query, filters)
                sparse_future = executor.submit(self._run_sparse, query)
                dense_result = dense_future.result()
                sparse_result = sparse_future.result()

            if dense_result.failed and sparse_result.failed:
                # Both legs actually broke — this is NOT "no relevant
                # chunks", it's a broken pipeline, and must not be allowed
                # to look like an empty-but-successful retrieval.
                raise RetrievalError(
                    "Both dense and sparse retrieval failed: "
                    f"dense_error={dense_result.error!r}, sparse_error={sparse_result.error!r}"
                )

            if dense_result.failed:
                logger.warning(f"Continuing with sparse-only results — dense leg failed: {dense_result.error}")
            if sparse_result.failed:
                logger.warning(f"Continuing with dense-only results — sparse leg failed: {sparse_result.error}")

            dense_hits, sparse_hits = dense_result.hits, sparse_result.hits
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