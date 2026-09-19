"""
Concrete EmbedderPort implementation using sentence-transformers, with a
primary/fallback model pair. The fallback activates on any load-time or
encode-time failure and stays active for the rest of the process — we don't
retry a known-broken primary on every batch.

Primary and fallback MUST share the same vector dimension: mixing dimensions
mid-session would either break storage outright or silently corrupt
similarity search on an already-populated collection.
"""

from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_harness.config.settings import EmbeddingSettings
from rag_harness.domain.exceptions import IngestionError
from rag_harness.domain.ports import EmbedderPort

logger = logging.getLogger("rag_harness.embedding")


class SentenceTransformerEmbedder(EmbedderPort):
    _primary: SentenceTransformer | None = None
    _fallback: SentenceTransformer | None = None
    _active_is_fallback: bool = False

    def __init__(self, settings: EmbeddingSettings):
        self.settings = settings

    @property
    def dimension(self) -> int:
        return self.settings.EMBEDDING_DIM

    def _load_primary(self) -> SentenceTransformer:
        if SentenceTransformerEmbedder._primary is None:
            logger.info(f"Loading primary embedding model: {self.settings.PRIMARY_EMBEDDING_MODEL}")
            SentenceTransformerEmbedder._primary = SentenceTransformer(self.settings.PRIMARY_EMBEDDING_MODEL)
        return SentenceTransformerEmbedder._primary

    def _load_fallback(self) -> SentenceTransformer:
        if SentenceTransformerEmbedder._fallback is None:
            logger.warning(f"Loading fallback embedding model: {self.settings.FALLBACK_EMBEDDING_MODEL}")
            SentenceTransformerEmbedder._fallback = SentenceTransformer(self.settings.FALLBACK_EMBEDDING_MODEL)
        return SentenceTransformerEmbedder._fallback

    def encode(self, texts: list[str], **kwargs) -> list[list[float]]:
        kwargs.setdefault("normalize_embeddings", True)
        kwargs.setdefault("show_progress_bar", False)

        if not SentenceTransformerEmbedder._active_is_fallback:
            try:
                model = self._load_primary()
                result: np.ndarray = model.encode(texts, **kwargs)
                return result.tolist()
            except Exception as e:
                logger.error(f"Primary embedding model failed ({e}) — switching to fallback for remainder of session")
                SentenceTransformerEmbedder._active_is_fallback = True

        try:
            model = self._load_fallback()
            result: np.ndarray = model.encode(texts, **kwargs)
            return result.tolist()
        except Exception as e:
            logger.critical(f"Fallback embedding model ALSO failed: {e}")
            raise IngestionError(f"Both primary and fallback embedding models failed: {e}")