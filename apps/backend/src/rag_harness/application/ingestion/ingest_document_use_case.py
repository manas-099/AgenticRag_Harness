"""
Top-level ingestion use case: route -> chunk -> embed -> store (Qdrant + BM25
+ section store sync). This is the single entry point the API layer and CLI
scripts call — everything else in ingestion/ is a building block this
orchestrates.
"""

from __future__ import annotations

import logging

from rag_harness.application.ingestion.contextual_chunker import ContextualChunker
from rag_harness.application.ingestion.file_router import FileRouter
from rag_harness.domain.exceptions import HarnessError, IngestionError
from rag_harness.domain.models import Chunk
from rag_harness.domain.ports import EmbedderPort, VectorStorePort
from rag_harness.infrastructure.sparse_search import BM25Retriever
from rag_harness.infrastructure.vector_stores import QdrantSectionStore

logger = logging.getLogger("rag_harness.ingestion.use_case")


class IngestDocumentUseCase:
    def __init__(
        self,
        file_router: FileRouter,
        chunker: ContextualChunker,
        embedder: EmbedderPort,
        vector_store: VectorStorePort,
        sparse_retriever: BM25Retriever,
        section_store: QdrantSectionStore | None = None,
    ):
        self.file_router = file_router
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.sparse_retriever = sparse_retriever
        self.section_store = section_store

    def execute(self, file_path: str, doc_id: str, doc_version: str) -> list[Chunk]:
        logger.info(f"=== Ingesting {file_path} (doc_id={doc_id}, version={doc_version}) ===")
        source_type = "pdf" if file_path.lower().endswith(".pdf") else "md"

        try:
            page_results = self.file_router.route_file(file_path, doc_id)
            page_texts = [(r.page_num, r.text, r.route) for r in page_results]

            chunks = self.chunker.chunk_document(doc_id, doc_version, source_type, page_texts)
            if not chunks:
                logger.warning(f"doc={doc_id}: produced zero chunks — nothing to embed/store")
                return []

            texts = [c.contextual_text for c in chunks]
            embeddings = self.embedder.encode(texts)
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb

            self.vector_store.upsert_chunks(chunks)

            self.sparse_retriever.corpus_chunks.extend(chunks)
            self.sparse_retriever.index(self.sparse_retriever.corpus_chunks)

            if self.section_store is not None:
                self._resync_sections()

            logger.info(f"=== Ingestion complete: {len(chunks)} chunks stored for {doc_id} ===")
            return chunks

        except HarnessError:
            raise
        except Exception as e:
            logger.error(f"Ingestion failed unexpectedly: {e}")
            raise IngestionError(f"Ingestion failed for {file_path}: {e}")

    def _resync_sections(self) -> None:
        try:
            all_titles = [c.metadata.section_title for c in self.sparse_retriever.corpus_chunks if c.metadata.section_title]
            if all_titles:
                self.section_store.sync_sections(all_titles)
                logger.info(f"Section store resynced: {len(set(all_titles))} unique sections")
            else:
                logger.warning("No section_title values found in corpus — section store not synced")
        except Exception as e:
            logger.error(f"Section store resync failed (non-fatal): {e}")