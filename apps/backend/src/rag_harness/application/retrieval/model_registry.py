"""
Lazy-loaded cross-encoder models (reranker, NLI) shared across retrieval and
verification. Separate from the embedder since these are different model
types used for different purposes — loaded once, cached as class state.
"""

from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

from rag_harness.config.settings import RerankSettings

logger = logging.getLogger("rag_harness.model_registry")


class CrossEncoderRegistry:
    _reranker: CrossEncoder | None = None
    _nli_model: CrossEncoder | None = None

    def __init__(self, settings: RerankSettings):
        self.settings = settings

    def reranker(self) -> CrossEncoder:
        if CrossEncoderRegistry._reranker is None:
            logger.info(f"Loading rerank model: {self.settings.RERANK_MODEL}")
            CrossEncoderRegistry._reranker = CrossEncoder(self.settings.RERANK_MODEL)
        return CrossEncoderRegistry._reranker

    def nli_model(self) -> CrossEncoder:
        if CrossEncoderRegistry._nli_model is None:
            logger.info(f"Loading NLI model: {self.settings.NLI_MODEL}")
            CrossEncoderRegistry._nli_model = CrossEncoder(self.settings.NLI_MODEL)
        return CrossEncoderRegistry._nli_model