"""
Callable tools the ReAct agent can choose to invoke. Each wraps an
already-tested pipeline/infrastructure component rather than reimplementing logic.
"""

from __future__ import annotations

import logging

from qdrant_client.models import FieldCondition, Filter, MatchValue

from rag_harness.application.retrieval.retrieve_use_case import RetrieveUseCase
from rag_harness.domain.exceptions import RetrievalError
from rag_harness.domain.models import Chunk, RetrievedChunk
from rag_harness.infrastructure.vector_stores import QdrantChunkStore

logger = logging.getLogger("rag_harness.tools")


class AgentTools:
    def __init__(self, retrieve_use_case: RetrieveUseCase, vector_store: QdrantChunkStore):
        self.retrieve_use_case = retrieve_use_case
        self.vector_store = vector_store

    def search_documents(self, query: str, filters: dict | None = None) -> list[RetrievedChunk]:
        logger.info(f"TOOL search_documents(query='{query[:50]}...')")
        return self.retrieve_use_case.execute(query, filters=filters)

    def search_documents_reformulated(self, reformulated_query: str, failed_attempt_reason: str) -> list[RetrievedChunk]:
        logger.info(f"TOOL search_documents_reformulated(reason='{failed_attempt_reason}')")
        return self.retrieve_use_case.execute(reformulated_query)

    def get_context(self, chunk_registry: dict[str, Chunk], chunk_id: str) -> str:
        logger.info(f"TOOL get_context(chunk_id={chunk_id})")
        if chunk_id not in chunk_registry:
            raise RetrievalError(f"chunk_id {chunk_id} not found in registry")
        return chunk_registry[chunk_id].raw_text

    def get_page(self, doc_id: str, page_num: int) -> str:
        logger.info(f"TOOL get_page(doc_id={doc_id}, page_num={page_num})")
        filters = Filter(must=[
            FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
            FieldCondition(key="page_num", match=MatchValue(value=page_num)),
        ])
        payloads = self.vector_store.scroll_by_filter(filters, limit=10)
        return "\n".join(p.get("raw_text", "") for p in payloads)

    def compare_chunks(self, chunk_registry: dict[str, Chunk], chunk_id_a: str, chunk_id_b: str) -> str:
        logger.info(f"TOOL compare_chunks({chunk_id_a}, {chunk_id_b})")
        if chunk_id_a not in chunk_registry or chunk_id_b not in chunk_registry:
            raise RetrievalError("One or both chunk_ids not found in registry")
        return (
            f"CHUNK A ({chunk_id_a}):\n{chunk_registry[chunk_id_a].raw_text}\n\n"
            f"CHUNK B ({chunk_id_b}):\n{chunk_registry[chunk_id_b].raw_text}"
        )

    def check_document_freshness(self, doc_id: str) -> str:
        logger.info(f"TOOL check_document_freshness(doc_id={doc_id})")
        filters = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
        payloads = self.vector_store.scroll_by_filter(filters, limit=1)
        return payloads[0].get("doc_version", "unknown") if payloads else "unknown"