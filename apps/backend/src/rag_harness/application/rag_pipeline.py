"""
Top-level orchestrator tying every layer together. This is the single class
the API layer and CLI scripts depend on.
"""

from __future__ import annotations

import logging

from rag_harness.application.agent.graph_builder import RunAgenticQueryUseCase
from rag_harness.application.generation.generate_answer_use_case import GenerateAnswerUseCase
from rag_harness.application.ingestion.ingest_document_use_case import IngestDocumentUseCase
from rag_harness.application.retrieval.metadata_filter_service import MetadataFilterService
from rag_harness.application.retrieval.retrieve_use_case import RetrieveUseCase
from rag_harness.domain.models import AnswerResult, Chunk, RetrievedChunk
from rag_harness.infrastructure.cache import InMemoryCache
from rag_harness.infrastructure.sparse_search import BM25Retriever
from rag_harness.infrastructure.vector_stores import QdrantChunkStore

logger = logging.getLogger("rag_harness.pipeline")


class RAGPipeline:
    def __init__(
        self,
        ingest_use_case: IngestDocumentUseCase,
        retrieve_use_case: RetrieveUseCase,
        generate_use_case: GenerateAnswerUseCase,
        cache: InMemoryCache,
        vector_store: QdrantChunkStore,
        sparse_retriever: BM25Retriever,
        agentic_use_case: RunAgenticQueryUseCase,
        metadata_filter_service: MetadataFilterService | None = None,
    ):
        self.ingest_use_case = ingest_use_case
        self.retrieve_use_case = retrieve_use_case
        self.generate_use_case = generate_use_case
        self.cache = cache
        self.vector_store = vector_store
        self.sparse_retriever = sparse_retriever
        self.agentic_use_case = agentic_use_case
        self.metadata_filter_service = metadata_filter_service

    def ingest_document(self, file_path: str, doc_id: str, doc_version: str) -> list[Chunk]:
        return self.ingest_use_case.execute(file_path, doc_id, doc_version)

    def query(self, question: str, filters: dict | None = None, doc_version_key: str = "v1") -> list[RetrievedChunk]:
        if self.is_empty():
            logger.warning("Query called but no documents ingested yet")
            return []

        cached = self.cache.get_retrieval(question, doc_version_key)
        if cached is not None:
            return cached

        resolved_filters = filters
        if resolved_filters is None and self.metadata_filter_service is not None:
            try:
                resolved_filters, matched = self.metadata_filter_service.build_filter(question)
                if matched:
                    logger.info(f"Metadata filter applied — sections: {matched}")
            except Exception as e:
                logger.error(f"Metadata filtering failed, proceeding unfiltered: {e}")
                resolved_filters = None

        results = self.retrieve_use_case.execute(question, filters=resolved_filters)
        self.cache.set_retrieval(question, doc_version_key, results)
        return results

    def answer(self, question: str, filters: dict | None = None, doc_version_key: str = "v1") -> AnswerResult:
        chunks = self.query(question, filters=filters, doc_version_key=doc_version_key)
        return self.generate_use_case.execute(question, chunks, doc_version_key=doc_version_key)

    def answer_agentic(self, question: str) -> dict:
        if self.is_empty():
            return {"answer": "No documents have been ingested yet.", "sources_used": [], "degraded": True}
        return self.agentic_use_case.execute(question)

    def is_empty(self) -> bool:
        return self.vector_store.count() <= 0

    def status(self) -> dict:
        return {
            "qdrant_chunk_count": self.vector_store.count(),
            "sparse_index_built": self.sparse_retriever.is_indexed,
            "sparse_corpus_size": self.sparse_retriever.corpus_size,
            "query_cache_size": self.cache.query_cache_size,
            "retrieval_cache_size": self.cache.retrieval_cache_size,
        }