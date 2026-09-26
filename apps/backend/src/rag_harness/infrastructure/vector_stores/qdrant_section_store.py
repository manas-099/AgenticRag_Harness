"""
Dedicated, small Qdrant collection holding unique section_title values,
with both a dense vector and a native Qdrant sparse vector — used by the
metadata-filtering layer to narrow retrieval by document section without
a hand-rolled sparse-vector implementation.
"""

from __future__ import annotations

import logging

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, SparseVector, SparseVectorParams, VectorParams

from rag_harness.config.settings import EmbeddingSettings, EnvironmentSettings
from rag_harness.domain.exceptions import RetrievalError
from rag_harness.domain.ports import EmbedderPort

logger = logging.getLogger("rag_harness.sections")


class QdrantSectionStore:
    COLLECTION_NAME = "rag_harness_sections"

    def __init__(
        self,
        environment_settings: EnvironmentSettings,
        embedding_settings: EmbeddingSettings,
        embedder: EmbedderPort,
    ):
        self.embedding_settings = embedding_settings
        self.embedder = embedder

        if environment_settings.uses_remote_qdrant:
            self.client = QdrantClient(
                url=environment_settings.QDRANT_URL,
                api_key=environment_settings.QDRANT_API_KEY or None,
            )
        else:
            section_path = environment_settings.qdrant_path.rstrip("/") + "_sections"
            self.client = QdrantClient(path=section_path, force_disable_check_same_thread=True)

        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if self.COLLECTION_NAME not in collections:
                logger.info(f"Creating section collection: {self.COLLECTION_NAME}")
                self.client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config={"dense": VectorParams(size=self.embedding_settings.EMBEDDING_DIM, distance=Distance.COSINE)},
                    sparse_vectors_config={"sparse": SparseVectorParams()},
                )
        except Exception as e:
            logger.error(f"Failed to ensure section collection: {e}")
            raise RetrievalError(f"Section collection setup failed: {e}")

    def sync_sections(self, section_titles: list[str]) -> None:
        section_titles = [s for s in set(section_titles) if s]
        if not section_titles:
            logger.warning("No section titles to sync")
            return

        dense_vecs = self.embedder.encode(section_titles)
        sparse_vecs = list(self.sparse_model.embed(section_titles))

        points = [
            PointStruct(
                id=i,
                vector={
                    "dense": dense_vec,
                    "sparse": SparseVector(indices=sparse_vec.indices.tolist(), values=sparse_vec.values.tolist()),
                },
                payload={"section_title": title},
            )
            for i, (title, dense_vec, sparse_vec) in enumerate(zip(section_titles, dense_vecs, sparse_vecs))
        ]

        self.client.delete_collection(self.COLLECTION_NAME)
        self._ensure_collection()
        self.client.upsert(collection_name=self.COLLECTION_NAME, points=points)
        logger.info(f"Synced {len(points)} unique sections")

    def dense_candidates(self, query: str, top_n: int) -> list[str]:
        try:
            query_vec = self.embedder.encode([query])[0]
            response = self.client.query_points(
                collection_name=self.COLLECTION_NAME, query=query_vec, using="dense", limit=top_n
            )
            return [p.payload["section_title"] for p in response.points]
        except Exception as e:
            logger.error(f"Section dense search failed: {e}")
            return []

    def sparse_candidates(self, query: str, top_n: int) -> list[str]:
        try:
            sparse_vec = list(self.sparse_model.embed([query]))[0]
            response = self.client.query_points(
                collection_name=self.COLLECTION_NAME,
                query=SparseVector(indices=sparse_vec.indices.tolist(), values=sparse_vec.values.tolist()),
                using="sparse",
                limit=top_n,
            )
            return [p.payload["section_title"] for p in response.points]
        except Exception as e:
            logger.error(f"Section sparse search failed: {e}")
            return []

    def get_all_sections(self) -> list[str]:
        try:
            points, _ = self.client.scroll(collection_name=self.COLLECTION_NAME, limit=1000, with_payload=True, with_vectors=False)
            return sorted({p.payload["section_title"] for p in points})
        except Exception as e:
            logger.error(f"Failed to fetch all sections: {e}")
            return []