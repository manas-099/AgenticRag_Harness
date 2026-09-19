"""
Concrete VectorStorePort implementation using Qdrant. Path is chosen
dynamically by environment (local embedded storage for dev/Colab, a real
server URL for production) — the interface presented to callers is identical
either way.
"""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from rag_harness.config.settings import EmbeddingSettings, EnvironmentSettings
from rag_harness.domain.exceptions import IngestionError, RetrievalError
from rag_harness.domain.models import Chunk, ChunkMetadata, RetrievedChunk
from rag_harness.domain.ports import VectorStorePort

logger = logging.getLogger("rag_harness.vectorstore")


class QdrantChunkStore(VectorStorePort):
    def __init__(self, embedding_settings: EmbeddingSettings, environment_settings: EnvironmentSettings):
        self.embedding_settings = embedding_settings
        self.environment_settings = environment_settings
        self.collection_name = embedding_settings.QDRANT_COLLECTION

        if environment_settings.uses_remote_qdrant:
            logger.info(f"Connecting to remote Qdrant at {environment_settings.QDRANT_URL}")
            self.client = QdrantClient(
                url=environment_settings.QDRANT_URL,
                api_key=environment_settings.QDRANT_API_KEY or None,
                prefer_grpc=True,
            )
        else:
            qdrant_path = environment_settings.qdrant_path
            logger.info(f"Using embedded local Qdrant at {qdrant_path}")
            self.client = QdrantClient(path=qdrant_path)

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in collections:
                logger.info(f"Creating collection: {self.collection_name} (dim={self.embedding_settings.EMBEDDING_DIM})")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.embedding_settings.EMBEDDING_DIM, distance=Distance.COSINE),
                )
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise RetrievalError(f"Qdrant collection setup failed: {e}")

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        missing = [c.chunk_id for c in chunks if c.embedding is None]
        if missing:
            raise IngestionError(f"{len(missing)} chunks missing embeddings, cannot upsert: {missing[:3]}...")

        try:
            points = [
                PointStruct(
                    id=c.chunk_id,
                    vector=c.embedding,
                    payload={
                        "raw_text": c.raw_text,
                        "contextual_text": c.contextual_text,
                        **c.metadata.model_dump(),
                    },
                )
                for c in chunks
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"Batch upserted {len(points)} chunks with metadata")
        except Exception as e:
            logger.error(f"Batch upsert failed: {e}")
            raise IngestionError(f"Qdrant batch upsert failed: {e}")

    def dense_search(
        self, query_vector: list[float], top_k: int, filters: dict | Filter | None = None
    ) -> list[RetrievedChunk]:
        try:
            qdrant_filter = self._build_filter(filters) if filters else None
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=qdrant_filter,
            )
            return [self._hit_to_retrieved_chunk(p) for p in response.points]
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            raise RetrievalError(f"Dense search failed: {e}")

    def count(self) -> int:
        try:
            return self.client.count(collection_name=self.collection_name).count
        except Exception as e:
            logger.error(f"Count failed: {e}")
            return -1

    def scroll_by_filter(self, filters: dict | Filter, limit: int = 10) -> list[dict]:
        """Used by agent tools (get_page, check_document_freshness) that need
        raw payload access rather than a similarity search."""
        try:
            qdrant_filter = self._build_filter(filters)
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_filter,
                limit=limit,
            )
            return [p.payload for p in points]
        except Exception as e:
            logger.error(f"Scroll failed: {e}")
            raise RetrievalError(f"Scroll failed: {e}")

    def close(self) -> None:
        try:
            self.client.close()
            logger.info("Qdrant client closed")
        except Exception as e:
            logger.warning(f"Error closing Qdrant client: {e}")

    def _build_filter(self, filters: dict | Filter) -> Filter:
        if isinstance(filters, Filter):
            return filters
        conditions = [FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()]
        return Filter(must=conditions)

    def _hit_to_retrieved_chunk(self, hit) -> RetrievedChunk:
        payload = hit.payload
        chunk = Chunk(
            chunk_id=str(hit.id),
            raw_text=payload["raw_text"],
            contextual_text=payload["contextual_text"],
            metadata=ChunkMetadata(**{k: v for k, v in payload.items() if k in ChunkMetadata.model_fields}),
        )
        return RetrievedChunk(chunk=chunk, score=hit.score, source="dense")